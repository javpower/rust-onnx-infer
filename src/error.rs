//! 统一错误类型（对应原版各方法抛出的 `Exception`）。

use thiserror::Error;

/// 库统一错误类型。
#[derive(Debug, Error, Clone)]
pub enum VisionError {
    #[error("模型文件不存在: {0}")]
    ModelNotFound(String),

    #[error("参数错误: {0}")]
    InvalidArgument(String),

    #[error("ONNX Runtime 错误: {0}")]
    Ort(String),

    #[error("图像处理错误: {0}")]
    Image(String),

    #[error("图像解码失败: {0}")]
    ImageDecode(String),

    #[error("IO 错误: {0}")]
    Io(String),

    #[error("Tokenizer 错误: {0}")]
    Tokenizer(String),

    #[error("推理失败: {0}")]
    Inference(String),

    #[error("不支持的操作: {0}")]
    Unsupported(String),

    #[error("其他错误: {0}")]
    Other(String),
}

impl VisionError {
    /// 快捷构造 `VisionError::InvalidArgument`。
    pub fn invalid_argument(msg: impl Into<String>) -> Self {
        VisionError::InvalidArgument(msg.into())
    }

    /// 快捷构造 `VisionError::Inference`。
    pub fn inference(msg: impl Into<String>) -> Self {
        VisionError::Inference(msg.into())
    }

    /// 快捷构造 `VisionError::Image`。
    pub fn image(msg: impl Into<String>) -> Self {
        VisionError::Image(msg.into())
    }
}

impl From<image::ImageError> for VisionError {
    fn from(e: image::ImageError) -> Self {
        VisionError::ImageDecode(e.to_string())
    }
}

impl From<ort::Error> for VisionError {
    fn from(e: ort::Error) -> Self {
        VisionError::Ort(e.to_string())
    }
}

impl From<std::io::Error> for VisionError {
    fn from(e: std::io::Error) -> Self {
        VisionError::Io(e.to_string())
    }
}

/// 库统一 Result 类型。
pub type Result<T> = std::result::Result<T, VisionError>;
