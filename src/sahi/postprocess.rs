//! SAHI 切片预测合并后处理 —— 对 sahi==0.12.6 numpy 后端
//! 对齐官方 sahi 的 `_numpy_backend` / `combine` / `utils` 实现
//! （对应 SahiPostprocess`）。
//!
//! **与官方实现的对齐要点**（这些细节决定逐框一致性）：
//! - **float32 度量矩阵**：官方 `tonumpy()` 把预测转成 float32 数组，IoU/IoS
//!   矩阵在 float32 上计算；矩阵阶段全程 float 运算，逐位一致
//! - **平局排序**：官方 `np.lexsort((y2, x2, y1, x1, -scores))` —— 分数降序为主键，
//!   同分按 x1/y1/x2/y2 升序，稳定排序保留原始相对顺序
//! - **双层阈值语义**：候选分组/抑制在 float32 矩阵上用 `metric >= threshold`
//!   （阈值按 numpy 弱标量规则转 float32 比较）；合并复核 `has_match` 用 float64
//!   且**严格大于** `metric > threshold` —— 恰好落在阈值上的框会被丢弃
//!   （既不保留为独立框、也不并入任何 keeper）
//! - **合并语义**（`merge_object_prediction_pair`）：bbox 取并集、分数取最大、
//!   类别取分高者（同分取被合并者）；合并结果累积作用于同一 keeper 的后续候选
//! - **按类别分派**（`batched_*`）：类别 id 升序逐类处理；NMS 最终按分数降序
//!   稳定排序（同分保持"类别序 + 类内名次"）；NMM/GREEDYNMM 按 keeper 产生顺序输出
//!
//! 支持官方三种策略：[`PostprocessType::Nms`]（抑制）、[`PostprocessType::Greedynmm`]
//! （贪心合并，SAHI 默认）、[`PostprocessType::Nmm`]（传递合并）。
//! [`PostprocessType::Lsnms`] 依赖外部 lsnms 库（官方标记实验性），按 NMS 等价处理
//! （对应 `config.rs` 中"LSNMS 未实现（等价按 Nms 处理）"的约定）。
//!
//! 本结构不持有可变状态，可复用；输入预测列表不会被修改。

use std::cmp::Ordering;
use std::collections::BTreeSet;

use super::config::{MatchMetric, PostprocessType, SahiBox, SahiPayload};

/// 分割掩码合并器（对应官方 `get_merged_mask` 的并集语义，上游实现
/// `BiFunction<Object, Object, Object> maskMerger`）：
/// 输入 keeper 与被合并框的 payload，输出合并后 payload；检测场景传 `None`。
pub type MaskMerger = Box<dyn Fn(Option<&SahiPayload>, Option<&SahiPayload>) -> Option<SahiPayload> + Send + Sync>;

/// float32 预测行 `[x1, y1, x2, y2, score, category_id]`（对应官方 `tonumpy()` 输出）。
pub type PredRow = [f32; 6];

/// SAHI 合并后处理器（对应 `SahiPostprocess`）。
pub struct SahiPostprocess {
    /// 合并/抑制策略（对应官方 postprocess_type）
    postprocess_type: PostprocessType,
    /// 重叠度量（对应官方 match_metric）
    metric: MatchMetric,
    /// 矩阵阶段阈值（float32，对应 numpy 弱标量提升）
    threshold_f: f32,
    /// 合并复核阈值（float64，对应 Python 原生 float 比较）
    threshold_d: f64,
    /// true 时合并/抑制忽略类别
    class_agnostic: bool,
    /// 掩码合并器（分割场景；NMM/GREEDYNMM 合并时把两个 payload 的掩码做并集），可为 None
    mask_merger: Option<MaskMerger>,
}

impl SahiPostprocess {
    /// 创建后处理器（对应上游 构造器）。
    ///
    /// `mask_merger`：分割掩码合并器（对应官方 `get_merged_mask` 的并集语义）；
    /// 检测场景传 `None`。原版对 LSNMS 抛异常，此处按约定退化为 NMS。
    pub fn new(
        postprocess_type: PostprocessType,
        metric: MatchMetric,
        match_threshold: f64,
        class_agnostic: bool,
        mask_merger: Option<MaskMerger>,
    ) -> Self {
        SahiPostprocess {
            postprocess_type,
            metric,
            threshold_f: match_threshold as f32,
            threshold_d: match_threshold,
            class_agnostic,
            mask_merger,
        }
    }

    /// 对预测列表执行合并后处理（对应官方 `postprocess(object_prediction_list)`）。
    /// 输入列表与其中的框对象不会被修改；结果按官方输出顺序排列。
    pub fn process(&self, predictions: &[SahiBox]) -> Vec<SahiBox> {
        let n = predictions.len();
        if n <= 1 {
            return predictions.to_vec();
        }
        let preds = to_float32_array(predictions);

        // LSNMS 官方依赖外部 lsnms 库（实验性），按 NMS 等价处理
        if matches!(self.postprocess_type, PostprocessType::Nms | PostprocessType::Lsnms) {
            let keep = if self.class_agnostic {
                self.nms_keep(&preds)
            } else {
                self.batched_nms(&preds)
            };
            return keep.into_iter().map(|idx| predictions[idx].clone()).collect();
        }

        let keep_to_merge = if self.class_agnostic {
            if self.postprocess_type == PostprocessType::Greedynmm {
                self.greedy_nmm_map(&preds)
            } else {
                self.nmm_map(&preds)
            }
        } else if self.postprocess_type == PostprocessType::Greedynmm {
            self.batched_merge_map(&preds, true)
        } else {
            self.batched_merge_map(&preds, false)
        };
        self.apply_merge(predictions, &keep_to_merge)
    }

    // ==================== 数值度量（两个精度层，与官方对应） ====================

    /// float32 交并比（度量矩阵阶段，与官方 float32 运算逐位一致）。
    pub fn iou_f(box_a: &PredRow, box_b: &PredRow) -> f32 {
        let inter = Self::inter_area_f(box_a, box_b);
        let area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]);
        let area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]);
        let union = area_a + area_b - inter;
        if union > 0.0 { inter / union } else { 0.0 } // _safe_ratio：分母 <= 0 取 0
    }

    /// float32 交小比（度量矩阵阶段）。
    pub fn ios_f(box_a: &PredRow, box_b: &PredRow) -> f32 {
        let inter = Self::inter_area_f(box_a, box_b);
        let area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]);
        let area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]);
        let smaller = area_a.min(area_b);
        if smaller > 0.0 { inter / smaller } else { 0.0 }
    }

    /// float64 交并比（合并复核阶段，对应官方 calculate_bbox_iou 的 float64 运算）。
    /// 分母 <= 0 时返回 NaN（官方除零产生 nan，nan > 阈值恒为 false）。
    pub fn iou_d(box_a: &[f64; 4], box_b: &[f64; 4]) -> f64 {
        let inter = Self::inter_area_d(box_a, box_b);
        let denom = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
            + (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
            - inter;
        if denom > 0.0 { inter / denom } else { f64::NAN }
    }

    /// float64 交小比（合并复核阶段；分母 <= 0 返回 NaN）。
    pub fn ios_d(box_a: &[f64; 4], box_b: &[f64; 4]) -> f64 {
        let inter = Self::inter_area_d(box_a, box_b);
        let smaller = ((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
            .min((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]));
        if smaller > 0.0 { inter / smaller } else { f64::NAN }
    }

    fn inter_area_f(a: &PredRow, b: &PredRow) -> f32 {
        let w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
        let h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
        w * h
    }

    fn inter_area_d(a: &[f64; 4], b: &[f64; 4]) -> f64 {
        let w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
        let h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
        w * h
    }

    /// 合并复核（官方 `has_match`：float64、严格大于）。
    /// NaN 与任何比较均为 false，与 Python `nan > t` 行为一致。
    fn has_match(&self, a: &SahiBox, b: &SahiBox) -> bool {
        let box_a = [a.min_x as f64, a.min_y as f64, a.max_x as f64, a.max_y as f64];
        let box_b = [b.min_x as f64, b.min_y as f64, b.max_x as f64, b.max_y as f64];
        let v = if self.metric == MatchMetric::Iou {
            Self::iou_d(&box_a, &box_b)
        } else {
            Self::ios_d(&box_a, &box_b)
        };
        v > self.threshold_d
    }

    // ==================== 排序（官方 _score_tiebreak_order） ====================

    /// `np.lexsort((y2, x2, y1, x1, -scores))`：主键分数降序，同分按 x1/y1/x2/y2 升序；
    /// `sort_by` 为稳定排序，完全同分的框保持原始顺序。
    fn score_tiebreak_order(preds: &[PredRow]) -> Vec<usize> {
        let mut order: Vec<usize> = (0..preds.len()).collect();
        order.sort_by(|&a, &b| {
            // 对应 comparingDouble(i -> -preds[i][4])：分数降序
            preds[b][4]
                .partial_cmp(&preds[a][4])
                .unwrap_or(Ordering::Equal)
                .then(preds[a][0].partial_cmp(&preds[b][0]).unwrap_or(Ordering::Equal))
                .then(preds[a][1].partial_cmp(&preds[b][1]).unwrap_or(Ordering::Equal))
                .then(preds[a][2].partial_cmp(&preds[b][2]).unwrap_or(Ordering::Equal))
                .then(preds[a][3].partial_cmp(&preds[b][3]).unwrap_or(Ordering::Equal))
        });
        order
    }

    fn metric_f(&self, box_a: &PredRow, box_b: &PredRow) -> f32 {
        if self.metric == MatchMetric::Iou {
            Self::iou_f(box_a, box_b)
        } else {
            Self::ios_f(box_a, box_b)
        }
    }

    // ==================== NMS（官方 nms_from_matrix） ====================

    /// 非极大值抑制，返回保留索引（分数降序、平局按坐标序）。
    fn nms_keep(&self, preds: &[PredRow]) -> Vec<usize> {
        let n = preds.len();
        let mut keep = Vec::new();
        if n == 0 {
            return keep;
        }
        let sorted = Self::score_tiebreak_order(preds);
        let mut suppressed = vec![false; n];
        for &idx in &sorted {
            if suppressed[idx] {
                continue;
            }
            keep.push(idx);
            for j in 0..n {
                // 官方：mask = matrix[idx] >= threshold; suppressed |= mask（含自身，无副作用）
                if !suppressed[j] && self.metric_f(&preds[idx], &preds[j]) >= self.threshold_f {
                    suppressed[j] = true;
                }
            }
        }
        keep
    }

    // ==================== GREEDYNMM（官方 greedy_nmm_from_matrix） ====================

    /// 贪心合并映射 keeper → 被并入索引列表（`Vec<(keeper, merges)>` 保持官方
    /// LinkedHashMap 的插入顺序）。
    fn greedy_nmm_map(&self, preds: &[PredRow]) -> Vec<(usize, Vec<usize>)> {
        let n = preds.len();
        let mut keep_to_merge: Vec<(usize, Vec<usize>)> = Vec::new();
        if n == 0 {
            return keep_to_merge;
        }
        let sorted = Self::score_tiebreak_order(preds);
        let mut suppressed = vec![false; n];
        for i in 0..n {
            let idx = sorted[i];
            if suppressed[idx] {
                continue;
            }
            let mut merges = Vec::new();
            for k in (i + 1)..n {
                let cand = sorted[k];
                if suppressed[cand] {
                    continue;
                }
                if self.metric_f(&preds[idx], &preds[cand]) >= self.threshold_f {
                    suppressed[cand] = true;
                    merges.push(cand);
                }
            }
            keep_to_merge.push((idx, merges));
        }
        keep_to_merge
    }

    // ==================== NMM（官方 nmm_from_matrix，传递合并） ====================

    /// 传递合并映射。`dominates[i][j]` = i≠j 且（score[i] > score[j]，或分数相等且
    /// j 的坐标字典序 <= i 的坐标）；候选还需度量 >= 阈值。已并入的框可继续吸收
    /// 其它框（传递性），keeper 永不被别人吸收。
    fn nmm_map(&self, preds: &[PredRow]) -> Vec<(usize, Vec<usize>)> {
        let n = preds.len();
        let mut keep_to_merge: Vec<(usize, Vec<usize>)> = Vec::new();
        if n == 0 {
            return keep_to_merge;
        }
        let sorted = Self::score_tiebreak_order(preds);

        // dominates 矩阵：score 严格更低，或同分且坐标字典序不小于自身
        let mut dominates = vec![vec![false; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let lower_score = preds[i][4] > preds[j][4];
                let score_equal = preds[i][4] == preds[j][4];
                let lex_less = lex_less(&preds[i], &preds[j]);
                dominates[i][j] = lower_score || (score_equal && !lex_less);
            }
        }

        let mut merge_to_keep = vec![-1i64; n];
        for pos in 0..n {
            let current = sorted[pos];
            let keep_idx: usize;
            if merge_to_keep[current] < 0 {
                // 成为 keeper（永不并入其它框）
                merge_to_keep[current] = current as i64;
                keep_idx = current;
                keep_to_merge.push((current, Vec::new()));
            } else {
                keep_idx = merge_to_keep[current] as usize;
            }
            for m in 0..n {
                if !dominates[current][m] || merge_to_keep[m] >= 0 {
                    continue;
                }
                if self.metric_f(&preds[current], &preds[m]) >= self.threshold_f {
                    if let Some((_, merge_list)) = keep_to_merge.iter_mut().find(|(k, _)| *k == keep_idx) {
                        merge_list.push(m);
                    }
                    merge_to_keep[m] = keep_idx as i64;
                }
            }
        }
        keep_to_merge
    }

    // ==================== 按类别批量分派（官方 _batched_apply） ====================

    fn batched_nms(&self, preds: &[PredRow]) -> Vec<usize> {
        let mut keep: Vec<usize> = Vec::new();
        for cls in unique_sorted_classes(preds) {
            let local = indices_of_class(preds, cls);
            let sub: Vec<PredRow> = local.iter().map(|&i| preds[i]).collect();
            for li in self.nms_keep(&sub) {
                keep.push(local[li]);
            }
        }
        // 官方：keep.sort(key=lambda i: scores[i], reverse=True)，稳定排序
        keep.sort_by(|&a, &b| preds[b][4].partial_cmp(&preds[a][4]).unwrap_or(Ordering::Equal));
        keep
    }

    fn batched_merge_map(&self, preds: &[PredRow], greedy: bool) -> Vec<(usize, Vec<usize>)> {
        let mut out: Vec<(usize, Vec<usize>)> = Vec::new();
        for cls in unique_sorted_classes(preds) {
            let local = indices_of_class(preds, cls);
            let sub: Vec<PredRow> = local.iter().map(|&i| preds[i]).collect();
            let local_map = if greedy { self.greedy_nmm_map(&sub) } else { self.nmm_map(&sub) };
            for (keeper, merges) in local_map {
                let merged: Vec<usize> = merges.into_iter().map(|li| local[li]).collect();
                out.push((local[keeper], merged));
            }
        }
        out
    }

    // ==================== 合并执行（官方 _apply_merge） ====================

    /// 按 keeper → merges 映射执行合并：keeper 累积吸收所有通过 [`Self::has_match`]
    /// 复核的候选（bbox 并集、分数最大、类别取分高者）；结果按 keeper 产生顺序输出。
    fn apply_merge(&self, predictions: &[SahiBox], keep_to_merge: &[(usize, Vec<usize>)]) -> Vec<SahiBox> {
        let mut selected = Vec::with_capacity(keep_to_merge.len());
        for &(keep_ind, ref merge_inds) in keep_to_merge {
            // 官方每次合并生成新对象（不改动原始预测）；此处以副本累积
            let mut keep = predictions[keep_ind].clone();
            for &merge_ind in merge_inds {
                let merge_pred = &predictions[merge_ind];
                if self.has_match(&keep, merge_pred) {
                    keep.merge_into(merge_pred);
                    // 官方 get_merged_mask：掩码并集（两者都有掩码时才合并，由 merger 决定）
                    if let Some(merger) = &self.mask_merger {
                        keep.payload = merger(keep.payload.as_ref(), merge_pred.payload.as_ref());
                    }
                }
            }
            selected.push(keep);
        }
        selected
    }
}

/// 把预测列表转为 float32 矩阵（对应官方 `tonumpy()`，`[x1,y1,x2,y2,score,categoryId]`）。
fn to_float32_array(predictions: &[SahiBox]) -> Vec<PredRow> {
    predictions
        .iter()
        .map(|b| [b.min_x, b.min_y, b.max_x, b.max_y, b.score, b.category_id as f32])
        .collect()
}

/// (x1, y1, x2, y2) 字典序严格小于（官方 cur_lt_cand 的逐列构建结果）。
fn lex_less(a: &PredRow, b: &PredRow) -> bool {
    for col in 0..4 {
        if a[col] < b[col] {
            return true;
        }
        if a[col] > b[col] {
            return false;
        }
    }
    false
}

/// 类别 id 升序去重（np.unique）。
fn unique_sorted_classes(preds: &[PredRow]) -> BTreeSet<i32> {
    preds.iter().map(|p| p[5] as i32).collect()
}

/// 该类别的全局索引（升序，对应 np.where 的输出顺序）。
fn indices_of_class(preds: &[PredRow], cls: i32) -> Vec<usize> {
    preds
        .iter()
        .enumerate()
        .filter(|(_, p)| p[5] as i32 == cls)
        .map(|(i, _)| i)
        .collect()
}
