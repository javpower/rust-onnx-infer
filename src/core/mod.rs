//! 引擎核心层。

pub mod async_batch;
pub mod base;
pub mod device_type;
pub mod engine;
pub mod factory;
pub mod model_type;
pub mod runtime_config;
pub mod session_factory;

pub use async_batch::{AsyncBatchOptimizer, AsyncBatchOptimizerBuilder};
pub use base::{BaseOnnxEngine, TensorData, TensorOutput};
pub use device_type::DeviceType;
pub use engine::OnnxInferenceEngine;
pub use factory::{
    builder, create_birefnet_engine, create_birefnet_engine_with_input_size,
    create_classification_engine, create_classification_engine_typed,
    create_classification_engine_with_input_size, create_depth_engine, create_detection_engine,
    create_detection_engine_with_input_size, create_engine, create_face_detection_engine,
    create_face_recognition_engine, create_grounded_sam_engine, create_image_enhance_engine,
    create_light_glue_engine, create_light_glue_engine_with_input_size, create_pose_engine,
    create_real_esrgan_engine, create_rfdetr_engine, create_rfdetr_seg_engine,
    create_rtdetr_engine, create_rtdetr_engine_with_input_size, create_sam_engine,
    create_sam2_engine, create_sam2_engine_with_input_size, create_segmentation_engine,
    create_segmentation_engine_with_input_size, create_style_transfer_engine,
    create_yoloe_engine, create_yoloe_runtime_engine, DynEngine, EngineBuilder,
};
pub use model_type::ModelType;
pub use runtime_config::{OnnxRuntimeConfig, DEFAULT_GPU_DEVICE_ID, DEFAULT_INTRA_OP_THREADS, DEFAULT_INTER_OP_THREADS};
