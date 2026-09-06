//! SAHI 切片推理（Slicing Aided Hyper Inference）。
//!
//! 对 sahi==0.12.6 `sahi.predict.get_sliced_prediction` 语义的无损还原。
//!
//! 模块组成：
//! - [`config`]：配置与内部框类型（SahiConfig / SahiBox / SahiPayload）
//! - [`slicer`]：切片坐标计算（官方 `get_slice_bboxes` 语义）
//! - [`postprocess`]：合并后处理（GREEDYNMM / NMM / NMS，float32 度量矩阵语义）
//! - [`adapters`]：引擎结果 ↔ 内部框适配器（检测 / 分割，含掩码并集）
//! - [`predictor`]：单切片预测器契约（绕开 SAHI 递归的原始单图路径）
//! - [`sliced_predictor`]：切片推理编排入口（检测 / 分割）
//! - [`result`]：结果容器

pub mod adapters;
pub mod config;
pub mod postprocess;
pub mod predictor;
pub mod result;
pub mod slicer;
pub mod sliced_predictor;

pub use adapters::{DetectionAdapter, SahiResultAdapter, SegmentationAdapter};
pub use config::{MatchMetric, PostprocessType, SahiBox, SahiConfig, SahiPayload};
pub use postprocess::{MaskMerger, SahiPostprocess};
pub use predictor::SahiSinglePredictor;
pub use result::SahiPredictionResult;
pub use slicer::{SahiSlice, SliceBbox, get_slice_bboxes, slice_image};
pub use sliced_predictor::{predict_sliced_detection, predict_sliced_segmentation};
