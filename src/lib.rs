//! # rust-onnx-infer
//!
//! 基于 ONNX Runtime 的视觉模型推理库。
//!
//! 支持多种视觉模型类型的统一推理接口：图像分类、目标检测（YOLO / RT-DETR /
//! RF-DETR）、实例分割（YOLO-Seg / RF-DETR-Seg / YOLOE）、SAM / SAM2 交互式分割、
//! Grounding DINO / Grounded-SAM 开放词表检测分割、BiRefNet 显著性抠图、
//! Real-ESRGAN 超分、图像增强（去噪 / 低光 / 去雾）、特征匹配（LightGlue /
//! DeDoDe / LoMa-R / RoMaV2）、姿态估计（YOLO-Pose）、人脸检测（YuNet）与
//! 人脸识别（SFace）、单目深度估计（Depth Anything V2）、风格迁移（cycleGAN），
//! 以及 SAHI 切片推理。
//!
//! ## 特性
//! - 纯 Rust 图像处理栈（无 OpenCV 原生依赖），静态链接 ONNX Runtime，
//!   可打包为自包含可执行文件
//! - `predict` / `predict_batch` 同步接口 + `predict_async` / `predict_batch_async`
//!   tokio 异步接口
//! - 模型标签自动从 ONNX 元数据加载（YOLO names / labels / categories）
//!
//! ## 风格说明
//! 为保留各引擎实现中的字面量精度、索引循环风格与参数签名，以下风格类
//! clippy lint 在 crate 级豁免（不影响行为）。
#![allow(clippy::excessive_precision)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::neg_cmp_op_on_partial_ord)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_late_init)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::manual_flatten)]
#![allow(clippy::unnecessary_sort_by)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::question_mark)]
#![allow(clippy::needless_question_mark)]
#![allow(clippy::chunks_exact_to_as_chunks)]

pub mod core;
pub mod engines;
pub mod error;
pub mod imaging;
pub mod model;
pub mod sahi;
pub mod util;

pub use core::{
    builder, AsyncBatchOptimizer, BaseOnnxEngine, DeviceType, DynEngine, ModelType,
    OnnxInferenceEngine, OnnxRuntimeConfig,
};
// 新引擎（姿态 / 人脸 / 深度 / 风格迁移）的工厂入口在 crate 根直接可用；
// 全部工厂函数见 `core::factory`。
pub use core::factory::{
    create_age_gender_engine, create_deblur_engine, create_depth_engine, create_expression_engine,
    create_face_detection_engine, create_face_landmark_engine, create_face_recognition_engine,
    create_hand_detection_engine, create_hand_landmark_engine, create_human_parsing_engine,
    create_ocr_pipeline, create_wholebody_engine,
    create_obb_engine, create_portrait_matting_engine, create_pose_engine,
    create_realtime_pose_engine, create_semantic_segmentation_engine, create_style_transfer_engine,
};
pub use error::{Result, VisionError};
pub use imaging::Image;
