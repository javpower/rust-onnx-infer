//! Grounded-SAM 端到端流水线。
//!
//! 内部组合 [`GroundingDinoEngine`](crate::engines::grounding_dino::GroundingDinoEngine)
//! （开放词表检测）和 [`Sam2Engine`](crate::engines::sam2::Sam2Engine)（分割）：
//!
//! ```text
//! 文本 + 图像 → Grounding DINO → 检测框 → SAM2.predict_box → 分割 mask
//! ```
//!
//! # 使用示例
//!
//! ```ignore
//! let gsam = GroundedSamEngine::new(
//!     "/models/grounding_dino_tiny/grounding_dino_tiny.onnx",  // Grounding DINO ONNX
//!     "/models/grounding_dino_tiny/tokenizer.json",            // BERT tokenizer
//!     "/models/sam2_1_hiera_base_plus/sam2_fused.onnx",        // SAM2 ONNX
//!     DeviceType::Cpu,
//! )?;
//!
//! // 一行完成"用文字描述分割任意物体"
//! let results = gsam.predict_text(&image, "chair . table .")?;
//! for s in &results {
//!     println!("{} @ {}", s.class_name(), s);
//! }
//! ```
//!
//! # 性能（CPU）
//!
//! - Grounding DINO Tiny：~300-900ms（单次推理，不随检测框数增加）
//! - SAM2 base-plus：~2s/box（fused 模式每次重跑 encoder）
//! - 总延迟：1 个目标 ≈ 3-4s，3 个目标 ≈ 7-10s
//!
//! # 实现说明
//!
//! - 异步推理由 trait 的 `predict_async` + tokio `run_blocking` 提供；
//! - 引擎被 drop 时两个内部引擎（含各自 ort Session）经 RAII 自动释放。

use async_trait::async_trait;

use crate::core::device_type::DeviceType;
use crate::core::engine::{run_blocking, OnnxInferenceEngine};
use crate::engines::grounding_dino::{
    GroundingDinoEngine, DEFAULT_BOX_THRESHOLD, DEFAULT_TEXT_THRESHOLD,
};
use crate::engines::sam2::Sam2Engine;
use crate::error::{Result, VisionError};
use crate::imaging::Image;
use crate::model::Segmentation;

/// Grounded-SAM 端到端流水线引擎（对应 `GroundedSamEngine`）。
///
/// 所有权持有 [`GroundingDinoEngine`] 与 [`Sam2Engine`]；生命周期随本引擎，
/// drop 时自动释放。
pub struct GroundedSamEngine {
    /// Grounding DINO 开放词表检测引擎
    dino: GroundingDinoEngine,
    /// SAM2 分割引擎
    sam2: Sam2Engine,

    /// 默认检测框置信度阈值
    box_threshold: f32,
    /// 默认文本匹配阈值
    text_threshold: f32,
}

impl GroundedSamEngine {
    /// 创建 Grounded-SAM 引擎（对应上游 4 参构造器）。
    pub fn new(
        dino_model_path: impl AsRef<std::path::Path>,
        tokenizer_path: impl AsRef<std::path::Path>,
        sam2_model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
    ) -> Result<Self> {
        let dino = GroundingDinoEngine::new(dino_model_path, tokenizer_path, device_type)?;
        let sam2 = Sam2Engine::new(sam2_model_path, device_type)?;
        tracing::info!("Grounded-SAM Engine initialized (DINO + SAM2)");
        Ok(GroundedSamEngine {
            dino,
            sam2,
            box_threshold: DEFAULT_BOX_THRESHOLD,
            text_threshold: DEFAULT_TEXT_THRESHOLD,
        })
    }

    // ==================== 访问器 ====================

    /// Grounding DINO 子引擎（只读访问）。
    pub fn dino(&self) -> &GroundingDinoEngine {
        &self.dino
    }

    /// SAM2 子引擎（只读访问）。
    pub fn sam2(&self) -> &Sam2Engine {
        &self.sam2
    }

    /// 默认检测框置信度阈值。
    pub fn box_threshold(&self) -> f32 {
        self.box_threshold
    }

    /// 设置默认检测框置信度阈值（对应 `setBoxThreshold`）。
    pub fn set_box_threshold(&mut self, threshold: f32) {
        self.box_threshold = threshold;
    }

    /// 默认文本匹配阈值。
    pub fn text_threshold(&self) -> f32 {
        self.text_threshold
    }

    /// 设置默认文本匹配阈值（对应 `setTextThreshold`）。
    pub fn set_text_threshold(&mut self, threshold: f32) {
        self.text_threshold = threshold;
    }

    // ==================== 核心 Pipeline ====================

    /// 核心入口：文本 + 图像 → 分割结果（使用引擎默认阈值；
    /// 对应上游 双参 `predict(image, textPrompt)`）。
    ///
    /// - `image`: 输入图像 (BGR)
    /// - `text_prompt`: 文本提示，类别用 "." 分隔（如 "tape . defect ."）
    pub fn predict_text(&self, image: &Image, text_prompt: &str) -> Result<Vec<Segmentation>> {
        self.predict_text_with_thresholds(image, text_prompt, self.box_threshold, self.text_threshold)
    }

    /// 带阈值的完整入口（对应上游 4 参 `predict`）。
    pub fn predict_text_with_thresholds(
        &self,
        image: &Image,
        text_prompt: &str,
        box_threshold: f32,
        text_threshold: f32,
    ) -> Result<Vec<Segmentation>> {
        let t0 = std::time::Instant::now();

        // 1. Grounding DINO 检测
        let boxes = self.dino.predict_text_with_thresholds(
            image,
            text_prompt,
            box_threshold,
            text_threshold,
        )?;
        tracing::info!(
            "[Grounded-SAM] DINO detected {} boxes in {}ms (prompt='{}')",
            boxes.len(),
            t0.elapsed().as_millis(),
            text_prompt
        );

        // 2. 对每个 box 跑 SAM2 分割
        let mut results: Vec<Segmentation> = Vec::new();
        for (i, d) in boxes.iter().enumerate() {
            let t2 = std::time::Instant::now();
            let masks = self.sam2.predict_box(
                image,
                d.x1() as f32,
                d.y1() as f32,
                d.x2() as f32,
                d.y2() as f32,
            )?;
            let elapsed = t2.elapsed().as_millis();

            if let Some(mut best) = masks.into_iter().next() {
                // 3 个候选中取 IoU 最高的（sam2 已按 IoU 降序返回首个）；
                // 其余候选随迭代 Drop 自动释放 mask（对应上游 releaseMask()）
                best.detection.class_name = d.class_name.clone();
                tracing::info!(
                    "[Grounded-SAM]   box[{}] '{}' -> mask IoU={:.3} in {}ms",
                    i,
                    d.class_name,
                    best.confidence(),
                    elapsed
                );
                results.push(best);
            }
        }

        tracing::info!(
            "[Grounded-SAM] total: {} masks in {}ms",
            results.len(),
            t0.elapsed().as_millis()
        );
        Ok(results)
    }
}

// ==================== OnnxInferenceEngine 接口实现 ====================

#[async_trait]
impl OnnxInferenceEngine for GroundedSamEngine {
    type Output = Vec<Segmentation>;

    /// 单图推理：Grounded-SAM 需要文本提示（对应上游 抛出
    /// `UnsupportedOperationException`）。
    fn predict(&self, _image: &Image) -> Result<Vec<Segmentation>> {
        Err(VisionError::Unsupported(
            "Use predict_text(image, text_prompt) instead".to_string(),
        ))
    }

    /// 批量推理：逐张以空 prompt 跑流水线。
    fn predict_batch(&self, images: &[Image]) -> Result<Vec<Vec<Segmentation>>> {
        images.iter().map(|img| self.predict_text(img, "")).collect()
    }

    /// 异步推理：对齐 上游 `predictAsync` —— 以空 prompt 跑流水线
    /// （上游 的 4 线程守护线程池由 tokio 运行时 + `run_blocking` 替代）。
    async fn predict_async(&self, image: &Image) -> Result<Vec<Segmentation>> {
        run_blocking(|| self.predict_text(image, "")).await
    }

    /// 输入尺寸来自 SAM2（对应 `getInputSize`）。
    fn input_size(&self) -> (i32, i32) {
        self.sam2.input_size()
    }

    /// 类别标签来自 SAM2（对应 `getLabels`）。
    fn labels(&self) -> Option<&[String]> {
        self.sam2.labels()
    }

    fn set_labels(&mut self, labels: Vec<String>) {
        self.sam2.set_labels(labels)
    }

    /// `set_confidence_threshold` 直接作用于 box_threshold。
    fn set_confidence_threshold(&mut self, threshold: f32) {
        self.box_threshold = threshold;
    }

    fn confidence_threshold(&self) -> f32 {
        self.box_threshold
    }
}
