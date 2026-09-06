//! 统一创建 ONNX Runtime `Session`。
//!
//! 封装 DeviceType 分支与线程配置，默认行为与原版一致；
//! GPU EP 不可用时自动回退 CPU（与既有的 warn + fallback 行为对齐）。

use std::path::Path;

use ort::execution_providers::{CUDAExecutionProvider, CoreMLExecutionProvider};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

use crate::core::device_type::DeviceType;
use crate::core::runtime_config::OnnxRuntimeConfig;
use crate::error::{Result, VisionError};

/// CUDA provider 实现选择（上游实现 LEGACY/V2；ort 统一走 V2 API，此处保留枚举对齐语义）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CudaProviderMode {
    #[default]
    Legacy,
    V2,
}

/// 使用默认运行参数创建会话选项（对应 `createSessionOptions`）。
///
/// ort 的会话选项在 builder 链上直接消费，此函数返回配置好的 builder。
pub fn create_session_builder(device_type: DeviceType) -> Result<SessionBuilderWithConfig> {
    create_session_builder_with(device_type, OnnxRuntimeConfig::defaults(), CudaProviderMode::Legacy)
}

/// 创建会话选项（线程数 / GPU 设备号可配置）。
pub fn create_session_builder_with(
    device_type: DeviceType,
    config: OnnxRuntimeConfig,
    cuda_provider_mode: CudaProviderMode,
) -> Result<SessionBuilderWithConfig> {
    config.validate()?;
    let ty = device_type;
    let mut eps: Vec<ort::execution_providers::ExecutionProviderDispatch> = Vec::new();

    match ty {
        DeviceType::Cuda => {
            tracing::info!(
                "Using CUDA V2 provider (deviceId={}, mode={:?})",
                config.gpu_device_id,
                cuda_provider_mode
            );
            eps.push(
                CUDAExecutionProvider::default()
                    .with_device_id(config.gpu_device_id)
                    .build()
                    .fail_silently(),
            );
        }
        DeviceType::Tensorrt => {
            tracing::warn!("TensorRT not supported in this build, falling back to CPU");
        }
        DeviceType::Directml => {
            tracing::warn!("DirectML not supported in this build, falling back to CPU");
        }
        DeviceType::Coreml => {
            tracing::info!("Using CoreML provider");
            eps.push(CoreMLExecutionProvider::default().build().fail_silently());
        }
        DeviceType::Openvino => {
            tracing::warn!("OpenVINO not supported in this build, falling back to CPU");
        }
        DeviceType::Rocm => {
            tracing::warn!("ROCm not supported in this build, falling back to CPU");
        }
        DeviceType::Auto => {
            tracing::info!("Auto device: trying CUDA (deviceId={}), fallback to CPU/CoreML", config.gpu_device_id);
            eps.push(
                CUDAExecutionProvider::default()
                    .with_device_id(config.gpu_device_id)
                    .build()
                    .fail_silently(),
            );
        }
        DeviceType::Cpu => {
            tracing::info!("Using CPU provider");
        }
    }

    Ok(SessionBuilderWithConfig { eps, config })
}

/// 携带设备/线程配置的会话 builder 中间结构。
pub struct SessionBuilderWithConfig {
    eps: Vec<ort::execution_providers::ExecutionProviderDispatch>,
    config: OnnxRuntimeConfig,
}

impl SessionBuilderWithConfig {
    /// 完成 Session 构建（对应 `OnnxSessionFactory.createSession`）。
    pub fn commit(self, model_path: impl AsRef<Path>) -> Result<Session> {
        let path = model_path.as_ref();
        if path.as_os_str().is_empty() {
            return Err(VisionError::invalid_argument("modelPath is blank"));
        }
        if !path.exists() {
            return Err(VisionError::ModelNotFound(path.display().to_string()));
        }

        let mut builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(self.config.intra_op_threads)?
            .with_inter_threads(self.config.inter_op_threads)?;
        if !self.eps.is_empty() {
            builder = builder.with_execution_providers(&self.eps)?;
        }
        let session = builder.commit_from_file(path)?;
        Ok(session)
    }
}

/// 安静关闭会话（ort 的 Session 由 Drop 自动释放；保留接口语义）。
pub fn close_session_quietly(_session: Session) {
    // ort: RAII，Drop 时释放
}
