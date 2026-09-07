//! LoMa-R（github.com/davnords/LoMa, "R" 旋转不变变体）稀疏匹配推理引擎
//! 。
//!
//! 两阶段 pipeline：
//! 1. DeDoDe 检测器抽关键点（[`crate::engines::dedode_g::DedodeGEngine`]）
//! 2. DeDoDe 描述器采样描述子
//! 3. 本引擎以两图的 (kpts, desc) 为输入跑 LoMa-R 主干，输出 scores [1, M, N]
//! 4. 本地做双向最近邻 + 阈值过滤，得到稀疏匹配对（与官方 `filter_matches` 语义一致）
//!
//! # 模型格式
//! - 输入 4 个 float32 张量：`kpts0[1, N, 2]`、`desc0[1, N, 256]`、
//!   `kpts1[1, M, 2]`、`desc1[1, M, 256]`
//! - 输出：`scores[1, M, N]`，match assignment softmax 后的相似度
//!
//! # 两种调用风格
//! - [`LoMaREngine::match_images`]：高层 API，内部用一个共享的
//!   [`DedodeGEngine`](crate::engines::dedode_g::DedodeGEngine) 抽两图的 kpts/desc，再调 LoMa-R 主干
//! - [`LoMaREngine::match_with_descriptors`]：低层 API，由调用方负责检测器，引擎只跑主干与做后处理
//!
//! # 推理实现
//! 上游使用 IoBinding \+ 显式生命周期管理；
//! Rust 侧用 [`BaseOnnxEngine::run_named`]（按名称绑定 4 输入），所有权语义天然安全。
//!
//! 注：原版 `predict/predictBatch` 抛 `UnsupportedOperationException`，
//! 故本引擎不实现 [`OnnxInferenceEngine`](crate::core::engine::OnnxInferenceEngine) trait，
//! 只提供固有方法。

use std::sync::Arc;

use crate::core::base::BaseOnnxEngine;
use crate::core::device_type::DeviceType;
use crate::engines::dedode_g::{Detection, DedodeGEngine};
use crate::error::{Result, VisionError};
use crate::imaging::Image;
use crate::model::{LoMaRFeatureMatch as FeatureMatch, LoMaRMatchResult, LoMaRKeyPoint as MatchKeyPoint};

/// LoMa-R 稀疏匹配引擎。
pub struct LoMaREngine {
    /// 组合基类
    pub base: BaseOnnxEngine,

    /// 双向最近邻过滤使用的分数阈值（默认 0.1，对应 Python 官方 cfg.filter_threshold）
    filter_threshold: f32,

    /// LoMa-R 主干的描述子维度（LoMa-R embed_dim=256）
    descriptor_dim: usize,

    /// 高层 API 用：单图关键点 + 描述子检测器；None 表示只能调低层 API
    ///
    /// 生命周期约定与原实现一致：detector 为调用方共享的 `Arc`，本引擎不负责其关闭。
    detector: Option<Arc<DedodeGEngine>>,

    // 4 个输入名（按官方约定：kpts0 / kpts1 / desc0 / desc1）
    kpts0_name: String,
    kpts1_name: String,
    desc0_name: String,
    desc1_name: String,
}

/// 匹配索引对（与坐标空间解耦，仅含 A/B 关键点索引 + score）。
#[derive(Debug, Clone, Copy)]
struct MatchIndex {
    idx_a: usize,
    idx_b: usize,
    score: f32,
}

impl LoMaREngine {
    // ===================== 构造 =====================

    /// 完整构造：高层 API（自带 detector），filterThreshold 默认 0.1。
    pub fn new(
        matcher_model_path: impl AsRef<std::path::Path>,
        detector: Arc<DedodeGEngine>,
        device_type: DeviceType,
    ) -> Result<Self> {
        Self::new_with_threshold(matcher_model_path, detector, device_type, 0.1)
    }

    /// 完整构造：高层 API（自带 detector），指定 filterThreshold。
    pub fn new_with_threshold(
        matcher_model_path: impl AsRef<std::path::Path>,
        detector: Arc<DedodeGEngine>,
        device_type: DeviceType,
        filter_threshold: f32,
    ) -> Result<Self> {
        let (base, input_names) = Self::create_base(matcher_model_path, device_type)?;
        let [kpts0_name, kpts1_name, desc0_name, desc1_name] = input_names;
        tracing::info!(
            "LoMaR Engine initialized: inputs=[{}, {}, {}, {}], descriptorDim={}, filterThreshold={}",
            kpts0_name,
            kpts1_name,
            desc0_name,
            desc1_name,
            Self::DESCRIPTOR_DIM,
            filter_threshold
        );
        Ok(LoMaREngine {
            base,
            filter_threshold,
            descriptor_dim: Self::DESCRIPTOR_DIM,
            detector: Some(detector),
            kpts0_name,
            kpts1_name,
            desc0_name,
            desc1_name,
        })
    }

    /// 低层 API 构造：不依赖 detector，仅用于接收外部传入的 kpts/desc，filterThreshold 默认 0.1。
    ///
    /// 注意：低层模式下调用 [`match_images`](Self::match_images) 会返回错误，因为没有 detector 可用。
    pub fn new_low_level(
        matcher_model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
    ) -> Result<Self> {
        Self::new_low_level_with_threshold(matcher_model_path, device_type, 0.1)
    }

    /// 低层 API 构造：指定 filterThreshold。
    pub fn new_low_level_with_threshold(
        matcher_model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        filter_threshold: f32,
    ) -> Result<Self> {
        let (base, input_names) = Self::create_base(matcher_model_path, device_type)?;
        let [kpts0_name, kpts1_name, desc0_name, desc1_name] = input_names;
        tracing::info!(
            "LoMaR Engine (low-level) initialized: inputs=[{}, {}, {}, {}], filterThreshold={}",
            kpts0_name,
            kpts1_name,
            desc0_name,
            desc1_name,
            filter_threshold
        );
        Ok(LoMaREngine {
            base,
            filter_threshold,
            descriptor_dim: Self::DESCRIPTOR_DIM,
            detector: None,
            kpts0_name,
            kpts1_name,
            desc0_name,
            desc1_name,
        })
    }

    /// LoMa-R 主干的描述子维度（LoMa-R embed_dim=256）。
    const DESCRIPTOR_DIM: usize = 256;

    /// 创建 matcher 主干基类并读取 4 个输入名（对应 `readInputName` × 4）。
    fn create_base(
        matcher_model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
    ) -> Result<(BaseOnnxEngine, [String; 4])> {
        let base = BaseOnnxEngine::with_input_size(matcher_model_path, device_type, 640, 640)?;

        let input_names: Vec<String> = {
            let session = base.session.lock().unwrap();
            if session.inputs().len() != 4 {
                return Err(VisionError::invalid_argument(format!(
                    "LoMa-R 模型应恰好 4 个输入 (kpts0/kpts1/desc0/desc1)，实际: {}",
                    session.inputs().len()
                )));
            }
            session.inputs()[..4].iter().map(|i| i.name().to_string()).collect()
        };
        let names = [
            input_names[0].clone(),
            input_names[1].clone(),
            input_names[2].clone(),
            input_names[3].clone(),
        ];
        Ok((base, names))
    }

    // ===================== 访问器 =====================

    /// 双向最近邻过滤使用的分数阈值。
    pub fn filter_threshold(&self) -> f32 {
        self.filter_threshold
    }

    /// 设置双向最近邻过滤使用的分数阈值。
    pub fn set_filter_threshold(&mut self, filter_threshold: f32) {
        self.filter_threshold = filter_threshold;
    }

    /// LoMa-R 主干的描述子维度（LoMa-R embed_dim=256）。
    pub fn descriptor_dim(&self) -> usize {
        self.descriptor_dim
    }

    // ===================== 高层 API =====================

    /// 高层 API：两张图 → LoMaRMatchResult。
    ///
    /// 内部用一个共享的 [`DedodeGEngine`](crate::engines::dedode_g::DedodeGEngine)
    /// 抽两图的 kpts/desc，再调 LoMa-R 主干。
    ///
    /// # 坐标空间约定（关键）
    /// matcher 的 Fourier 位置编码要求关键点在 [-1, 1] 归一化空间，
    /// 与 PyTorch 官方 `get_normalized_grid` 一致。本方法把 detector 输出的归一化坐标
    /// 喂给 matcher；输出 [`MatchKeyPoint`] 仍用原图像素坐标，方便下游可视化 / 几何验证。
    pub fn match_images(&self, img_a: &Image, img_b: &Image) -> Result<LoMaRMatchResult> {
        if img_a.is_empty() || img_b.is_empty() {
            return Err(VisionError::invalid_argument(
                "Input images cannot be null or empty",
            ));
        }
        let detector = self.detector.as_ref().ok_or_else(|| {
            VisionError::invalid_argument(
                "LoMaREngine 配置为 low-level 模式（无 detector），请改用 matchWithDescriptors(...)",
            )
        })?;

        let (orig_wa, orig_ha) = (img_a.width() as i32, img_a.height() as i32);
        let (orig_wb, orig_hb) = (img_b.width() as i32, img_b.height() as i32);

        let det_a = detector.detect(img_a)?;
        let det_b = detector.detect(img_b)?;
        self.assemble_result(&det_a, &det_b, orig_wa, orig_ha, orig_wb, orig_hb)
    }

    /// 高层 API 的缓存变体：imgA 的特征由调用方预计算并复用（如批量匹配中不变的模板图），
    /// 本方法只对 imgB 做特征提取。输出与 [`match_images`](Self::match_images) 完全一致
    /// （keypoints/matches 均为原图像素坐标）。
    pub fn match_images_cached(
        &self,
        det_a: &Detection,
        img_b: &Image,
    ) -> Result<LoMaRMatchResult> {
        if img_b.is_empty() {
            return Err(VisionError::invalid_argument(
                "detA cannot be null, imgB cannot be null or empty",
            ));
        }
        let detector = self.detector.as_ref().ok_or_else(|| {
            VisionError::invalid_argument(
                "LoMaREngine 配置为 low-level 模式（无 detector），请改用 matchWithDescriptors(...)",
            )
        })?;

        let det_b = detector.detect(img_b)?;
        self.assemble_result(
            det_a,
            &det_b,
            det_a.image_width,
            det_a.image_height,
            img_b.width() as i32,
            img_b.height() as i32,
        )
    }

    /// 组装结果：跑主干 → 双向最近邻过滤 → 输出像素坐标的 keypoints/matches。
    fn assemble_result(
        &self,
        det_a: &Detection,
        det_b: &Detection,
        orig_wa: i32,
        orig_ha: i32,
        orig_wb: i32,
        orig_hb: i32,
    ) -> Result<LoMaRMatchResult> {
        let n = det_a.size();
        let m = det_b.size();
        if n == 0 || m == 0 {
            tracing::warn!(
                "DeDoDe 检测器在至少一张图上未抽到关键点（A={}, B={}），返回空结果",
                n,
                m
            );
            return Ok(empty_result(orig_wa, orig_ha, orig_wb, orig_hb, n as i32, m as i32, self.filter_threshold));
        }

        // ⚠️ matcher 必须接收归一化坐标 [-1,1]：Fourier 位置编码在该空间才有意义，
        //    喂像素坐标会让 sin/cos 饱和成噪声，几何先验失效。
        let scores = self.run_match(
            &det_a.normalized_keypoints,
            &det_a.descriptors,
            &det_b.normalized_keypoints,
            &det_b.descriptors,
        )?;
        let idx_matches = mutual_filter(&scores, self.filter_threshold);

        // 输出 keypoints/matches 用原图像素坐标（供下游可视化 / findHomography）
        let pix_a = &det_a.keypoints;
        let pix_b = &det_b.keypoints;
        let mut keypoints_a: Vec<MatchKeyPoint> = Vec::with_capacity(n);
        for (i, p) in pix_a.iter().enumerate().take(n) {
            keypoints_a.push(MatchKeyPoint::new(p[0], p[1], i as i32));
        }
        let mut keypoints_b: Vec<MatchKeyPoint> = Vec::with_capacity(m);
        for (j, p) in pix_b.iter().enumerate().take(m) {
            keypoints_b.push(MatchKeyPoint::new(p[0], p[1], j as i32));
        }
        let mut matches: Vec<FeatureMatch> = Vec::with_capacity(idx_matches.len());
        for mi in &idx_matches {
            matches.push(FeatureMatch {
                kp0: MatchKeyPoint::new(pix_a[mi.idx_a][0], pix_a[mi.idx_a][1], mi.idx_a as i32),
                kp1: MatchKeyPoint::new(pix_b[mi.idx_b][0], pix_b[mi.idx_b][1], mi.idx_b as i32),
                score: mi.score,
            });
        }

        tracing::info!(
            "LoMaR result: N={}, M={}, mutual matches={} (threshold={})",
            n,
            m,
            matches.len(),
            self.filter_threshold
        );

        Ok(LoMaRMatchResult {
            keypoints0: keypoints_a,
            keypoints1: keypoints_b,
            matches,
            image0_width: orig_wa,
            image0_height: orig_ha,
            image1_width: orig_wb,
            image1_height: orig_hb,
            detector_num_keypoints0: n as i32,
            detector_num_keypoints1: m as i32,
            filter_threshold: self.filter_threshold,
            score_matrix: Some(scores),
        })
    }

    // ===================== 低层 API =====================

    /// 低层 API（4 张量版）：调用方需自行保证 kpts/desc 与图像坐标空间一致。
    /// 原图宽高使用 `-1` 占位（表示未知）。
    ///
    /// # 坐标空间契约
    /// 传入的 `kpts_a`/`kpts_b` 必须是 `[-1, 1]` 归一化坐标
    /// （对应 PyTorch `get_normalized_grid`，与 `grid_sample` align_corners=False 对齐）。
    /// 这是 matcher 内部 Fourier 位置编码要求的输入空间。传入像素坐标会导致位置编码饱和、
    /// 几何先验失效，匹配效果严重退化。
    ///
    /// 输出 [`MatchKeyPoint`] 直接使用传入的坐标（即本 API 的调用方决定输出坐标空间）；
    /// 若需要像素坐标输出，请自行反归一化，或改用高层 [`match_images`](Self::match_images)。
    pub fn match_with_descriptors(
        &self,
        kpts_a: &[[f32; 2]],
        desc_a: &[Vec<f32>],
        kpts_b: &[[f32; 2]],
        desc_b: &[Vec<f32>],
    ) -> Result<LoMaRMatchResult> {
        self.match_with_descriptors_full(kpts_a, desc_a, kpts_b, desc_b, -1, -1, -1, -1)
    }

    /// 低层 API（带原图尺寸）：语义同 [`match_with_descriptors`](Self::match_with_descriptors)，
    /// 另外填充结果中的 `image0_width`/`image0_height`/`image1_width`/`image1_height`。
    pub fn match_with_descriptors_full(
        &self,
        kpts_a: &[[f32; 2]],
        desc_a: &[Vec<f32>],
        kpts_b: &[[f32; 2]],
        desc_b: &[Vec<f32>],
        orig_wa: i32,
        orig_ha: i32,
        orig_wb: i32,
        orig_hb: i32,
    ) -> Result<LoMaRMatchResult> {
        if kpts_a.len() != desc_a.len() || kpts_b.len() != desc_b.len() {
            return Err(VisionError::invalid_argument(format!(
                "kpts 与 desc 数量不匹配: kptsA={}, descA={}, kptsB={}, descB={}",
                kpts_a.len(),
                desc_a.len(),
                kpts_b.len(),
                desc_b.len()
            )));
        }
        if kpts_a.is_empty() || kpts_b.is_empty() {
            return Ok(empty_result(
                orig_wa,
                orig_ha,
                orig_wb,
                orig_hb,
                kpts_a.len() as i32,
                kpts_b.len() as i32,
                self.filter_threshold,
            ));
        }
        if desc_a[0].len() != self.descriptor_dim || desc_b[0].len() != self.descriptor_dim {
            return Err(VisionError::invalid_argument(format!(
                "描述子维度不符：期望 {}，实际 A={}, B={}",
                self.descriptor_dim,
                desc_a[0].len(),
                desc_b[0].len()
            )));
        }

        let n = kpts_a.len();
        let m = kpts_b.len();

        let scores = self.run_match(kpts_a, desc_a, kpts_b, desc_b)?;
        let idx_matches = mutual_filter(&scores, self.filter_threshold);

        // 输出坐标空间 = 输入坐标空间（本低层 API 不做反归一化，由调用方决定）
        let mut keypoints_a: Vec<MatchKeyPoint> = Vec::with_capacity(n);
        for (i, p) in kpts_a.iter().enumerate() {
            keypoints_a.push(MatchKeyPoint::new(p[0], p[1], i as i32));
        }
        let mut keypoints_b: Vec<MatchKeyPoint> = Vec::with_capacity(m);
        for (j, p) in kpts_b.iter().enumerate() {
            keypoints_b.push(MatchKeyPoint::new(p[0], p[1], j as i32));
        }
        let mut matches: Vec<FeatureMatch> = Vec::with_capacity(idx_matches.len());
        for mi in &idx_matches {
            matches.push(FeatureMatch {
                kp0: MatchKeyPoint::new(kpts_a[mi.idx_a][0], kpts_a[mi.idx_a][1], mi.idx_a as i32),
                kp1: MatchKeyPoint::new(kpts_b[mi.idx_b][0], kpts_b[mi.idx_b][1], mi.idx_b as i32),
                score: mi.score,
            });
        }

        tracing::info!(
            "LoMaR result: N={}, M={}, mutual matches={} (threshold={})",
            n,
            m,
            matches.len(),
            self.filter_threshold
        );

        Ok(LoMaRMatchResult {
            keypoints0: keypoints_a,
            keypoints1: keypoints_b,
            matches,
            image0_width: orig_wa,
            image0_height: orig_ha,
            image1_width: orig_wb,
            image1_height: orig_hb,
            detector_num_keypoints0: n as i32,
            detector_num_keypoints1: m as i32,
            filter_threshold: self.filter_threshold,
            score_matrix: Some(scores),
        })
    }

    // ===================== 推理 =====================

    /// 4 输入推理。返回 scores[N][M]，索引语义：`scores[a][b]` = 图 A 关键点 a 与图 B 关键点 b 的相似度。
    fn run_match(
        &self,
        kpts_a: &[[f32; 2]],
        desc_a: &[Vec<f32>],
        kpts_b: &[[f32; 2]],
        desc_b: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>> {
        let n = kpts_a.len();
        let m = kpts_b.len();
        let dim = self.descriptor_dim;

        // 扁平化输入
        let mut kpts_a_flat = Vec::with_capacity(n * 2);
        for p in kpts_a {
            kpts_a_flat.push(p[0]);
            kpts_a_flat.push(p[1]);
        }
        let mut kpts_b_flat = Vec::with_capacity(m * 2);
        for p in kpts_b {
            kpts_b_flat.push(p[0]);
            kpts_b_flat.push(p[1]);
        }
        let mut desc_a_flat = Vec::with_capacity(n * dim);
        for (i, row) in desc_a.iter().enumerate() {
            if row.len() < dim {
                return Err(VisionError::invalid_argument(format!(
                    "descA[{i}] 维度 {} < {dim}",
                    row.len()
                )));
            }
            desc_a_flat.extend_from_slice(&row[..dim]);
        }
        let mut desc_b_flat = Vec::with_capacity(m * dim);
        for (i, row) in desc_b.iter().enumerate() {
            if row.len() < dim {
                return Err(VisionError::invalid_argument(format!(
                    "descB[{i}] 维度 {} < {dim}",
                    row.len()
                )));
            }
            desc_b_flat.extend_from_slice(&row[..dim]);
        }

        let k_shape_a = vec![1, n as i64, 2];
        let k_shape_b = vec![1, m as i64, 2];
        let d_shape_a = vec![1, n as i64, dim as i64];
        let d_shape_b = vec![1, m as i64, dim as i64];

        let t_ka = ort::value::Tensor::from_array((k_shape_a, kpts_a_flat))?;
        let t_kb = ort::value::Tensor::from_array((k_shape_b, kpts_b_flat))?;
        let t_da = ort::value::Tensor::from_array((d_shape_a, desc_a_flat))?;
        let t_db = ort::value::Tensor::from_array((d_shape_b, desc_b_flat))?;

        let outputs = self.base.run_named(vec![
            (self.kpts0_name.as_str(), t_ka),
            (self.kpts1_name.as_str(), t_kb),
            (self.desc0_name.as_str(), t_da),
            (self.desc1_name.as_str(), t_db),
        ])?;
        parse_scores(&outputs)
    }
}

/// 解析主干输出为 scores[N][M] 矩阵。
///
/// 实际 ONNX 输出是 [1, N, M]（N = kpts0 = 图 A 关键点数；M = kpts1 = 图 B 关键点数）。
/// PyTorch LoMa.forward 用 einsum "bmd,bnd->bmn"，m 对应 0 轴(desc0)=A，n 对应 1 轴(desc1)=B。
/// 本函数的数组语义：`scores[a][b]` = A 关键点 a 与 B 关键点 b 的相似度，即 scores[N][M]。
fn parse_scores(outputs: &[crate::core::base::TensorOutput]) -> Result<Vec<Vec<f32>>> {
    let scores_tensor = outputs
        .first()
        .ok_or_else(|| VisionError::inference("matcher returned no outputs"))?;
    let shape = &scores_tensor.shape;
    if shape.len() < 3 {
        return Err(VisionError::inference(format!(
            "Unexpected scores shape: {shape:?}"
        )));
    }
    let flat = scores_tensor.as_f32()?;

    let n = shape[1] as usize; // N = A 关键点数
    let m = shape[2] as usize; // M = B 关键点数
    if flat.len() < n * m {
        return Err(VisionError::inference(format!(
            "scores element count {} < {}x{}",
            flat.len(),
            n,
            m
        )));
    }
    let mut scores = vec![vec![0f32; m]; n];
    for (a, row) in scores.iter_mut().enumerate() {
        row.copy_from_slice(&flat[a * m..(a + 1) * m]);
    }
    Ok(scores)
}

// ===================== 后处理 =====================

/// 双向最近邻过滤（mutual check），仅返回索引对。
///
/// 对应 Python 官方 `filter_matches` 逻辑：
/// - A→B：对每个 i，找 argmax_j scores[i, j]，记为 b0[i]
/// - B→A：对每个 j，找 argmax_i scores[i, j]，记为 a1[j]
/// - mutual: j == a1[b0[i]] && i == b0[a1[j]]
/// - 保留 score >= threshold 的对
///
/// 这里实现：scores[N][M]，`scores[a][b]` 表示图 A 第 a 个关键点与图 B 第 b 个关键点的相似度。
/// - 对每个 a ∈ [0, N)，找 argmax over b → b0[a]
/// - 对每个 b ∈ [0, M)，找 argmax over a → a0[b]
/// - 互最近邻：b0[a] == b 且 a0[b0[a]] == a
///
/// 本方法与坐标空间解耦：只输出索引对 (idxA, idxB, score)，由调用方按需映射到像素
/// 或归一化坐标。这样 matcher 输入（必须归一化）与输出（应像素）可使用不同坐标空间。
fn mutual_filter(scores: &[Vec<f32>], filter_threshold: f32) -> Vec<MatchIndex> {
    let n = scores.len(); // A 关键点数
    let m = if n > 0 { scores[0].len() } else { 0 }; // B 关键点数

    // A→B argmax: 对每个 a，找最像的 b
    let mut b0 = vec![-1i32; n];
    let mut b0_score = vec![0f32; n];
    for (a, row) in scores.iter().enumerate() {
        let mut best = f32::NEG_INFINITY;
        let mut best_b = -1i32;
        for (b, &v) in row.iter().enumerate().take(m) {
            if v > best {
                best = v;
                best_b = b as i32;
            }
        }
        b0[a] = best_b;
        b0_score[a] = best;
    }

    // B→A argmax: 对每个 b，找最像的 a
    let mut a0 = vec![-1i32; m];
    for b in 0..m {
        let mut best = f32::NEG_INFINITY;
        let mut best_a = -1i32;
        for a in 0..n {
            if scores[a][b] > best {
                best = scores[a][b];
                best_a = a as i32;
            }
        }
        a0[b] = best_a;
    }

    let mut matches = Vec::new();
    for a in 0..n {
        let b = b0[a];
        if b < 0 {
            continue;
        }
        let b = b as usize;
        if a0[b] != a as i32 {
            continue; // 不是双向最近邻
        }
        let score = b0_score[a];
        if score < filter_threshold {
            continue;
        }
        matches.push(MatchIndex {
            idx_a: a,
            idx_b: b,
            score,
        });
    }
    matches
}

/// 空匹配结果（对应 `emptyResult`；scoreMatrix 不填，即 None）。
fn empty_result(
    wa: i32,
    ha: i32,
    wb: i32,
    hb: i32,
    n: i32,
    m: i32,
    filter_threshold: f32,
) -> LoMaRMatchResult {
    LoMaRMatchResult {
        keypoints0: Vec::new(),
        keypoints1: Vec::new(),
        matches: Vec::new(),
        image0_width: wa,
        image0_height: ha,
        image1_width: wb,
        image1_height: hb,
        detector_num_keypoints0: n,
        detector_num_keypoints1: m,
        filter_threshold,
        score_matrix: None,
    }
}
