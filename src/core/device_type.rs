//! 推理设备类型。

/// 推理设备类型枚举。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DeviceType {
    /// CPU 推理
    #[default]
    Cpu,
    /// CUDA GPU 推理
    Cuda,
    /// TensorRT 推理
    Tensorrt,
    /// DirectML (Windows)
    Directml,
    /// CoreML (Apple)
    Coreml,
    /// OpenVINO (Intel)
    Openvino,
    /// ROCm (AMD)
    Rocm,
    /// 自动选择（优先 GPU，失败回退 CPU）
    Auto,
}

impl DeviceType {
    /// 根据名称获取设备类型（大小写不敏感；未知名称回退 CPU，与原实现一致）。
    pub fn from_name(name: &str) -> Self {
        match name.to_ascii_lowercase().as_str() {
            "cpu" => DeviceType::Cpu,
            "cuda" => DeviceType::Cuda,
            "tensorrt" => DeviceType::Tensorrt,
            "dml" | "directml" => DeviceType::Directml,
            "coreml" => DeviceType::Coreml,
            "openvino" => DeviceType::Openvino,
            "rocm" => DeviceType::Rocm,
            "auto" => DeviceType::Auto,
            _ => DeviceType::Cpu,
        }
    }

    /// 显示名称（对应 `getName()`）。
    pub fn name(&self) -> &'static str {
        match self {
            DeviceType::Cpu => "CPU",
            DeviceType::Cuda => "CUDA",
            DeviceType::Tensorrt => "TensorRT",
            DeviceType::Directml => "DirectML",
            DeviceType::Coreml => "CoreML",
            DeviceType::Openvino => "OpenVINO",
            DeviceType::Rocm => "ROCm",
            DeviceType::Auto => "Auto",
        }
    }

    /// provider 标识（对应 `getProvider()`）。
    pub fn provider(&self) -> &'static str {
        match self {
            DeviceType::Cpu => "cpu",
            DeviceType::Cuda => "cuda",
            DeviceType::Tensorrt => "tensorrt",
            DeviceType::Directml => "dml",
            DeviceType::Coreml => "coreml",
            DeviceType::Openvino => "openvino",
            DeviceType::Rocm => "rocm",
            DeviceType::Auto => "auto",
        }
    }

    /// 是否使用 GPU。
    pub fn is_gpu(&self) -> bool {
        matches!(self, DeviceType::Cuda | DeviceType::Tensorrt | DeviceType::Rocm)
    }
}
