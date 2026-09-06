//! 内置的结果类型适配器：
//! [`DetectionAdapter`]（目标检测）与 [`SegmentationAdapter`]
//! （实例分割，含掩码平移与并集合并）。对应用户可见的类型约定：无论 SAHI 是否开启，
//! 引擎返回的都是 `Detection` / `Segmentation`。

use std::collections::HashSet;

use crate::imaging::FloatMask;
use crate::model::{Detection, Segmentation};

use super::config::{SahiBox, SahiPayload};
use super::postprocess::MaskMerger;

/// 引擎结果类型适配器：把引擎原始结果映射到 SAHI 内部框（含官方 clip + 平移语义），
/// 合并完成后再从内部框重建出与引擎类型完全一致的结果对象——对用户无感知的关键。
pub trait SahiResultAdapter<T> {
    /// 引擎原始结果 → 内部框。执行官方 ultralytics 适配器的坐标语义：
    /// `max(0, coord)` → min 到全图尺寸 → 无效框（x1>=x2 或 y1>=y2）返回 `None` 丢弃 →
    /// 加上切片偏移 `(shift_x, shift_y)` 平移到全图坐标。
    ///
    /// 返回的内部框 payload 承载重建结果所需信息；无效框返回 `None`。
    fn to_box(
        &self,
        item: &T,
        shift_x: i32,
        shift_y: i32,
        full_width: i32,
        full_height: i32,
    ) -> Option<SahiBox>;

    /// 内部框（可能已参与合并）→ 与引擎类型一致的结果对象。
    /// 合并语义已由 [`SahiPostprocess`](super::postprocess::SahiPostprocess) 完成
    /// （bbox 并集、分数最大、类别取分高者、掩码并集），
    /// 此处只负责用 box 字段 + payload 重建对象。
    fn to_result(&self, boxed: &SahiBox) -> T;

    /// 分割掩码合并器（对应官方 `get_merged_mask` 的并集语义）：
    /// 输入 keeper 与被合并框的 payload，输出合并后 payload。
    /// 检测场景返回 `None`（无掩码）。
    fn mask_merger(&self) -> Option<MaskMerger> {
        None
    }

    /// 类别排除判断（官方 `filter_predictions`，按类别名 / 类别 id）。
    /// 默认不排除；基于 `Detection` 的结果类型应覆盖此方法。
    fn is_excluded(&self, item: &T, exclude_names: &HashSet<String>, exclude_ids: &HashSet<i32>) -> bool {
        let _ = (item, exclude_names, exclude_ids);
        false
    }
}

/// 目标检测适配器：payload 不需要承载额外信息（类别/分数都在内部框上）。
#[derive(Debug, Clone, Copy, Default)]
/// OBB 旋转框适配器。
///
/// 合并以轴对齐外接矩形（aabb）近似——官方 sahi 不支持旋转框，切片合并阶段
/// 角度信息按"合并簇 aabb 重建、angle=0"处理；需要精确角度的场景请关闭 SAHI。
pub struct ObbAdapter;

impl SahiResultAdapter<crate::engines::obb_detection::ObbResult> for ObbAdapter {
    fn to_box(
        &self,
        item: &crate::engines::obb_detection::ObbResult,
        shift_x: i32,
        shift_y: i32,
        full_width: i32,
        full_height: i32,
    ) -> Option<SahiBox> {
        let (x1, y1, x2, y2) = item.aabb();
        let c = clip(
            format!("ObbResult[{}]", item.class_name),
            x1 as f64,
            y1 as f64,
            x2 as f64,
            y2 as f64,
            full_width,
            full_height,
        )?;
        Some(SahiBox::new(
            c[0] + shift_x as f32,
            c[1] + shift_y as f32,
            c[2] + shift_x as f32,
            c[3] + shift_y as f32,
            item.confidence as f32,
            item.class_id,
            item.class_name.clone(),
        ))
    }

    fn to_result(&self, boxed: &SahiBox) -> crate::engines::obb_detection::ObbResult {
        use crate::engines::obb_detection::ObbResult;
        // 合并簇 aabb 重建旋转框（angle=0 近似；见类型文档）
        let (x1, y1, x2, y2) = (boxed.min_x, boxed.min_y, boxed.max_x, boxed.max_y);
        let corners = [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ];
        ObbResult {
            cx: (x1 + x2) / 2.0,
            cy: (y1 + y2) / 2.0,
            w: x2 - x1,
            h: y2 - y1,
            angle_rad: 0.0,
            corners,
            class_name: boxed.category_name.clone(),
            class_id: boxed.category_id,
            confidence: boxed.score as f64,
        }
    }

    fn is_excluded(
        &self,
        item: &crate::engines::obb_detection::ObbResult,
        exclude_names: &HashSet<String>,
        exclude_ids: &HashSet<i32>,
    ) -> bool {
        exclude_ids.contains(&item.class_id) || exclude_names.contains(&item.class_name)
    }
}

pub struct DetectionAdapter;

impl SahiResultAdapter<Detection> for DetectionAdapter {
    fn to_box(
        &self,
        item: &Detection,
        shift_x: i32,
        shift_y: i32,
        full_width: i32,
        full_height: i32,
    ) -> Option<SahiBox> {
        let c = clip(
            item,
            item.x1(),
            item.y1(),
            item.x2(),
            item.y2(),
            full_width,
            full_height,
        )?;
        Some(SahiBox::new(
            c[0] + shift_x as f32,
            c[1] + shift_y as f32,
            c[2] + shift_x as f32,
            c[3] + shift_y as f32,
            item.confidence() as f32,
            item.class_id,
            item.class_name.clone(),
        ))
    }

    fn to_result(&self, boxed: &SahiBox) -> Detection {
        Detection::new(
            boxed.category_name.clone(),
            boxed.category_id,
            boxed.min_x as f64,
            boxed.min_y as f64,
            boxed.max_x as f64,
            boxed.max_y as f64,
            boxed.score as f64,
        )
    }

    fn is_excluded(&self, item: &Detection, exclude_names: &HashSet<String>, exclude_ids: &HashSet<i32>) -> bool {
        exclude_names.contains(&item.class_name) || exclude_ids.contains(&item.class_id)
    }
}

/// 实例分割适配器：掩码跨切片平移到全图画布，合并时按官方语义做并集。
#[derive(Debug, Clone, Copy, Default)]
pub struct SegmentationAdapter;

impl SahiResultAdapter<Segmentation> for SegmentationAdapter {
    fn to_box(
        &self,
        item: &Segmentation,
        shift_x: i32,
        shift_y: i32,
        full_width: i32,
        full_height: i32,
    ) -> Option<SahiBox> {
        // 官方：segmentation 为空的预测直接丢弃
        let mask = item.mask.as_ref()?;
        // 官方 sahi #235 语义：有掩码时丢弃检测框，改用掩码像素范围的紧致包围盒
        //（ultralytics 掩码为 proto 概率 > 0 且裁剪在检测框内的二值区域）
        let Some((ex, ey, ew, eh)) = mask_extent(mask) else {
            return None;
        };
        let c = clip(
            item,
            ex as f64,
            ey as f64,
            (ex + ew) as f64,
            (ey + eh) as f64,
            full_width,
            full_height,
        )?;
        let mut boxed = SahiBox::new(
            c[0] + shift_x as f32,
            c[1] + shift_y as f32,
            c[2] + shift_x as f32,
            c[3] + shift_y as f32,
            item.confidence() as f32,
            item.class_id(),
            item.class_name(),
        );
        // 掩码惰性物化：仅记录切片帧掩码 + 偏移，合并/出结果时才建全图画布（省内存）
        boxed.payload = Some(SahiPayload::SliceMask {
            data: mask.data().to_vec(),
            width: mask.width(),
            height: mask.height(),
            offset_x: shift_x,
            offset_y: shift_y,
            full_width,
            full_height,
        });
        Some(boxed)
    }

    fn to_result(&self, boxed: &SahiBox) -> Segmentation {
        let mask = materialize_mask(boxed.payload.as_ref());
        Segmentation::new(
            boxed.category_name.clone(),
            boxed.category_id,
            boxed.min_x as f64,
            boxed.min_y as f64,
            boxed.max_x as f64,
            boxed.max_y as f64,
            boxed.score as f64,
            mask,
        )
    }

    fn mask_merger(&self) -> Option<MaskMerger> {
        Some(Box::new(union_mask))
    }

    fn is_excluded(
        &self,
        item: &Segmentation,
        exclude_names: &HashSet<String>,
        exclude_ids: &HashSet<i32>,
    ) -> bool {
        exclude_names.contains(item.class_name()) || exclude_ids.contains(&item.class_id())
    }
}

// ==================== 内部实现 ====================

/// 掩码有效区域包围盒（官方 sahi #235：segmentation 轮廓的 min/max，
/// 对应 ultralytics 掩码 proto 概率 > 0 且裁剪在检测框内的像素范围；
/// 上游实现为 threshold(0.5) → findNonZero → boundingRect）。
///
/// 返回整数像素包围盒 `(x, y, width, height)`；掩码为空/全零时返回 `None`。
fn mask_extent(mask: &FloatMask) -> Option<(i32, i32, i32, i32)> {
    if mask.is_empty() {
        return None;
    }
    // probability > 0.5 → 前景：ultralytics process_mask 对 proto logits 取 gt_(0.0)，
    // 等价于 sigmoid 概率 > 0.5（引擎掩码约定为 [0,1] 概率图）
    let w = mask.width();
    let mut min_x = usize::MAX;
    let mut min_y = usize::MAX;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    let mut found = false;
    for (idx, &v) in mask.data().iter().enumerate() {
        if v > 0.5 {
            let x = idx % w;
            let y = idx / w;
            found = true;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    if !found {
        return None;
    }
    Some((
        min_x as i32,
        min_y as i32,
        (max_x - min_x + 1) as i32,
        (max_y - min_y + 1) as i32,
    ))
}

/// 官方 ultralytics 适配器坐标语义：`max(0,·)` → min 到全图 → 无效框丢弃。
fn clip(
    source: impl std::fmt::Display,
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    full_width: i32,
    full_height: i32,
) -> Option<[f32; 4]> {
    let mut fx1 = x1.max(0.0) as f32;
    let mut fy1 = y1.max(0.0) as f32;
    let mut fx2 = x2.max(0.0) as f32;
    let mut fy2 = y2.max(0.0) as f32;
    fx1 = fx1.min(full_width as f32);
    fy1 = fy1.min(full_height as f32);
    fx2 = fx2.min(full_width as f32);
    fy2 = fy2.min(full_height as f32);
    if !(fx1 < fx2) || !(fy1 < fy2) {
        tracing::warn!("SAHI 忽略无效预测框：{}", source);
        return None;
    }
    Some([fx1, fy1, fx2, fy2])
}

/// 把 payload 物化为全图尺寸的 float32 掩码画布（对应 CV_32F Mat）；掩码为空时返回 `None`。
pub fn materialize_mask(payload: Option<&SahiPayload>) -> Option<FloatMask> {
    let SahiPayload::SliceMask {
        data,
        width,
        height,
        offset_x,
        offset_y,
        full_width,
        full_height,
    } = payload?
    else {
        // FullMask：已是全图画布（None 已被 `payload?` 过滤）
        let Some(SahiPayload::FullMask(m)) = payload else {
            return None;
        };
        return Some(m.clone());
    };
    // sliceMask 为空时返回 None
    if *width == 0 || *height == 0 || data.is_empty() || *full_width <= 0 || *full_height <= 0 {
        return None;
    }
    let mut canvas = FloatMask::new(*full_width as usize, *full_height as usize);
    // 源掩码与目标 ROI 取交集尺寸（切片必然在图内，通常整块贴合）
    let w = (*width as i32).min(full_width - offset_x).max(0) as usize;
    let h = (*height as i32).min(full_height - offset_y).max(0) as usize;
    if w == 0 || h == 0 {
        return Some(canvas);
    }
    let src_w = *width;
    let canvas_w = canvas.width();
    let off_x = (*offset_x).max(0) as usize;
    let off_y = (*offset_y).max(0) as usize;
    for dy in 0..h {
        let src_start = dy * src_w;
        let dst_start = (off_y + dy) * canvas_w + off_x;
        canvas.data_mut()[dst_start..dst_start + w].copy_from_slice(&data[src_start..src_start + w]);
    }
    Some(canvas)
}

/// 掩码并集（官方 `get_merged_mask` 语义）：两个全图概率画布逐像素取 max——
/// 对二值掩码即几何并集，对概率图取各位置更置信的一方。
/// 官方约定"任一方无掩码则合并结果无掩码"，同样遵循（返回 `None`）。
pub fn union_mask(
    keep_payload: Option<&SahiPayload>,
    merge_payload: Option<&SahiPayload>,
) -> Option<SahiPayload> {
    match (materialize_mask(keep_payload), materialize_mask(merge_payload)) {
        (Some(mut a), Some(b)) => {
            for (x, &y) in a.data_mut().iter_mut().zip(b.data()) {
                *x = x.max(y);
            }
            Some(SahiPayload::FullMask(a))
        }
        // 无掩码合并结果：payload 置空（to_result 产出 mask=None 的 Segmentation）
        _ => None,
    }
}
