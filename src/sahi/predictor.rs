//! 单切片预测器契约。
//!
//! 引擎侧 SAHI 集成的内部契约（对用户无感知：引擎的 `predict()` 返回类型不变）。
//! 上游实现为 `SahiSinglePredictor<T>` 接口（引擎以 `this::predictWithoutSahi`
//! 方法引用实现）；Rust 侧为 trait，并由 [`sliced_predictor`](super::sliced_predictor)
//! 的两个具体入口使用，逐切片调用以绕开 SAHI 递归。

use crate::engines::detection::DetectionEngine;
use crate::engines::segmentation::SegmentationEngine;
use crate::engines::yolo_e_runtime::YoloERuntimeEngine;
use crate::error::Result;
use crate::imaging::Image;
use crate::model::{Detection, Segmentation};

/// 引擎侧 SAHI 集成的内部契约（对用户无感知：引擎的 `predict()` 返回类型不变）。
pub trait SahiSinglePredictor<T> {
    /// 不经过 SAHI 的单图原始推理（SAHI 逐切片调用，避免递归）。
    fn predict_single(&self, image: &Image) -> Result<Vec<T>>;
}

// ============ 内置引擎适配（对应上游 引擎传入的 `this::predictWithoutSahi`） ============

impl SahiSinglePredictor<Detection> for DetectionEngine {
    fn predict_single(&self, image: &Image) -> Result<Vec<Detection>> {
        self.predict_without_sahi(image)
    }
}

impl SahiSinglePredictor<Segmentation> for SegmentationEngine {
    fn predict_single(&self, image: &Image) -> Result<Vec<Segmentation>> {
        self.predict_without_sahi(image)
    }
}

/// 双模型架构的 YOLOE 运行时引擎（对应 `YoloERuntimeEngine` 传入的
/// `this::predictWithoutSahi`；上游实现该引擎未实现 `OnnxInferenceEngine`，
/// 经 `predictSliced` 的置信度阈值重载接入 SAHI）。
impl SahiSinglePredictor<Segmentation> for YoloERuntimeEngine {
    fn predict_single(&self, image: &Image) -> Result<Vec<Segmentation>> {
        self.predict_without_sahi(image)
    }
}

impl SahiSinglePredictor<crate::engines::obb_detection::ObbResult>
    for crate::engines::obb_detection::ObbDetectionEngine
{
    fn predict_single(
        &self,
        image: &Image,
    ) -> Result<Vec<crate::engines::obb_detection::ObbResult>> {
        self.predict_without_sahi(image)
    }
}
