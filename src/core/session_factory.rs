//! 统一创建 ONNX Runtime `Session`。
//!
//! 封装 DeviceType 分支与线程配置，默认行为与原版一致；
//! GPU EP 不可用时自动回退 CPU（与既有的 warn + fallback 行为对齐）。
//!
//! 执行提供器由 crate feature 决定（`cuda` / `tensorrt` / `directml` / `coreml` /
//! `openvino` / `rocm`，透传至 `ort` 同名 feature）：未启用对应 feature 时不编入
//! EP 代码，选择该设备类型时回退 CPU 并给出 warn。

use std::path::Path;

use ort::ep::ExecutionProviderDispatch;
#[cfg(feature = "coreml")]
use ort::ep::CoreML;
#[cfg(feature = "cuda")]
use ort::ep::CUDA;
#[cfg(feature = "directml")]
use ort::ep::DirectML;
#[cfg(feature = "openvino")]
use ort::ep::OpenVINO;
#[cfg(feature = "rocm")]
use ort::ep::ROCm;
#[cfg(feature = "tensorrt")]
use ort::ep::TensorRT;
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
    let mut eps: Vec<ExecutionProviderDispatch> = Vec::new();

    match ty {
        DeviceType::Cuda => add_cuda_ep(&mut eps, config.gpu_device_id, cuda_provider_mode),
        DeviceType::Tensorrt => add_tensorrt_ep(&mut eps),
        DeviceType::Directml => add_directml_ep(&mut eps),
        DeviceType::Coreml => add_coreml_ep(&mut eps),
        DeviceType::Openvino => add_openvino_ep(&mut eps),
        DeviceType::Rocm => add_rocm_ep(&mut eps),
        DeviceType::Auto => {
            tracing::info!(
                "Auto device: trying enabled GPU EPs (deviceId={}), fallback to CPU",
                config.gpu_device_id
            );
            // 平台不适配的 EP 由 fail_silently 在运行时自动忽略
            add_cuda_ep(&mut eps, config.gpu_device_id, cuda_provider_mode);
            add_coreml_ep(&mut eps);
        }
        DeviceType::Cpu => {
            tracing::info!("Using CPU provider");
        }
    }

    Ok(SessionBuilderWithConfig { eps, config })
}

/// 推入 CUDA EP（需启用 `cuda` feature；未启用时回退 CPU 并警告）。
fn add_cuda_ep(eps: &mut Vec<ExecutionProviderDispatch>, device_id: i32, mode: CudaProviderMode) {
    #[cfg(feature = "cuda")]
    {
        tracing::info!("Using CUDA provider (deviceId={}, mode={:?})", device_id, mode);
        eps.push(
            CUDA::default()
                .with_device_id(device_id)
                .build()
                .fail_silently(),
        );
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (eps, device_id, mode);
        tracing::warn!("CUDA not compiled in (missing `cuda` feature), falling back to CPU");
    }
}

/// 推入 CoreML EP（需启用 `coreml` feature；未启用时回退 CPU 并警告）。
fn add_coreml_ep(eps: &mut Vec<ExecutionProviderDispatch>) {
    #[cfg(feature = "coreml")]
    {
        tracing::info!("Using CoreML provider");
        eps.push(CoreML::default().build().fail_silently());
    }
    #[cfg(not(feature = "coreml"))]
    {
        let _ = eps;
        tracing::warn!("CoreML not compiled in (missing `coreml` feature), falling back to CPU");
    }
}

/// 推入 TensorRT EP（需启用 `tensorrt` feature；未启用时回退 CPU 并警告）。
fn add_tensorrt_ep(eps: &mut Vec<ExecutionProviderDispatch>) {
    #[cfg(feature = "tensorrt")]
    {
        tracing::info!("Using TensorRT provider");
        eps.push(TensorRT::default().build().fail_silently());
    }
    #[cfg(not(feature = "tensorrt"))]
    {
        let _ = eps;
        tracing::warn!("TensorRT not compiled in (missing `tensorrt` feature), falling back to CPU");
    }
}

/// 推入 DirectML EP（需启用 `directml` feature；未启用时回退 CPU 并警告）。
fn add_directml_ep(eps: &mut Vec<ExecutionProviderDispatch>) {
    #[cfg(feature = "directml")]
    {
        tracing::info!("Using DirectML provider");
        eps.push(DirectML::default().build().fail_silently());
    }
    #[cfg(not(feature = "directml"))]
    {
        let _ = eps;
        tracing::warn!("DirectML not compiled in (missing `directml` feature), falling back to CPU");
    }
}

/// 推入 OpenVINO EP（需启用 `openvino` feature；未启用时回退 CPU 并警告）。
fn add_openvino_ep(eps: &mut Vec<ExecutionProviderDispatch>) {
    #[cfg(feature = "openvino")]
    {
        tracing::info!("Using OpenVINO provider");
        eps.push(OpenVINO::default().build().fail_silently());
    }
    #[cfg(not(feature = "openvino"))]
    {
        let _ = eps;
        tracing::warn!("OpenVINO not compiled in (missing `openvino` feature), falling back to CPU");
    }
}

/// 推入 ROCm EP（需启用 `rocm` feature；未启用时回退 CPU 并警告）。
fn add_rocm_ep(eps: &mut Vec<ExecutionProviderDispatch>) {
    #[cfg(feature = "rocm")]
    {
        tracing::info!("Using ROCm provider");
        eps.push(ROCm::default().build().fail_silently());
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = eps;
        tracing::warn!("ROCm not compiled in (missing `rocm` feature), falling back to CPU");
    }
}

/// 携带设备/线程配置的会话 builder 中间结构。
pub struct SessionBuilderWithConfig {
    eps: Vec<ExecutionProviderDispatch>,
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
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort::Error::from)?
            .with_intra_threads(self.config.intra_op_threads)
            .map_err(ort::Error::from)?
            .with_inter_threads(self.config.inter_op_threads)
            .map_err(ort::Error::from)?;
        if !self.eps.is_empty() {
            builder = builder.with_execution_providers(&self.eps).map_err(ort::Error::from)?;
        }
        let session = builder.commit_from_file(path)?;
        Ok(session)
    }
}

/// 安静关闭会话（ort 的 Session 由 Drop 自动释放；保留接口语义）。
pub fn close_session_quietly(_session: Session) {
    // ort: RAII，Drop 时释放
}
