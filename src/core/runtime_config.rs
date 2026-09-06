//! ONNX Runtime 会话运行参数。
//!
//! 默认值与 原版历史行为一致（intra/inter 线程 = 4，CUDA deviceId = 0）。

/// 会话运行参数。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OnnxRuntimeConfig {
    /// Intra-op 线程数
    pub intra_op_threads: usize,
    /// Inter-op 线程数
    pub inter_op_threads: usize,
    /// GPU 设备 id
    pub gpu_device_id: i32,
}

pub const DEFAULT_INTRA_OP_THREADS: usize = 4;
pub const DEFAULT_INTER_OP_THREADS: usize = 4;
pub const DEFAULT_GPU_DEVICE_ID: i32 = 0;

impl Default for OnnxRuntimeConfig {
    fn default() -> Self {
        OnnxRuntimeConfig {
            intra_op_threads: DEFAULT_INTRA_OP_THREADS,
            inter_op_threads: DEFAULT_INTER_OP_THREADS,
            gpu_device_id: DEFAULT_GPU_DEVICE_ID,
        }
    }
}

impl OnnxRuntimeConfig {
    /// 与库历史默认行为一致的配置。
    pub fn defaults() -> Self {
        OnnxRuntimeConfig::default()
    }

    /// 校验参数合法性（与默认值约定一致）。
    pub fn validate(&self) -> crate::error::Result<()> {
        use crate::error::VisionError;
        if self.intra_op_threads < 1 {
            return Err(VisionError::InvalidArgument(format!(
                "intraOpThreads must be >= 1, got {}",
                self.intra_op_threads
            )));
        }
        if self.inter_op_threads < 1 {
            return Err(VisionError::InvalidArgument(format!(
                "interOpThreads must be >= 1, got {}",
                self.inter_op_threads
            )));
        }
        if self.gpu_device_id < 0 {
            return Err(VisionError::InvalidArgument(format!(
                "gpuDeviceId must be >= 0, got {}",
                self.gpu_device_id
            )));
        }
        Ok(())
    }
}

/// Builder（对应 `OnnxRuntimeConfig.Builder`）。
#[derive(Debug, Clone)]
pub struct OnnxRuntimeConfigBuilder {
    config: OnnxRuntimeConfig,
}

impl Default for OnnxRuntimeConfigBuilder {
    fn default() -> Self {
        OnnxRuntimeConfigBuilder {
            config: OnnxRuntimeConfig::defaults(),
        }
    }
}

impl OnnxRuntimeConfigBuilder {
    pub fn intra_op_threads(mut self, threads: usize) -> Self {
        self.config.intra_op_threads = threads;
        self
    }

    pub fn inter_op_threads(mut self, threads: usize) -> Self {
        self.config.inter_op_threads = threads;
        self
    }

    pub fn gpu_device_id(mut self, id: i32) -> Self {
        self.config.gpu_device_id = id;
        self
    }

    pub fn build(self) -> OnnxRuntimeConfig {
        self.config
    }
}

impl OnnxRuntimeConfig {
    pub fn builder() -> OnnxRuntimeConfigBuilder {
        OnnxRuntimeConfigBuilder::default()
    }
}

impl std::fmt::Display for OnnxRuntimeConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "OnnxRuntimeConfig{{intraOpThreads={}, interOpThreads={}, gpuDeviceId={}}}",
            self.intra_op_threads, self.inter_op_threads, self.gpu_device_id
        )
    }
}
