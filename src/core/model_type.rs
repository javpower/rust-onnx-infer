//! ONNX 模型类型枚举。

/// ONNX 模型类型枚举。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ModelType {
    /// 图像分类模型（ResNet, MobileNet, EfficientNet, ViT, YOLO-CLS 等）
    #[default]
    Classification,
    /// 目标检测模型（YOLOv5/v8/v9/v10/v11, RT-DETR, DETR 等）
    Detection,
    /// 实例分割模型（YOLO-Seg, Mask R-CNN 等）
    Segmentation,
    /// 关键点检测模型（YOLO-Pose, MediaPipe Pose 等）
    Keypoint,
    /// 人脸检测与识别
    Face,
    /// 深度估计
    Depth,
    /// 光流估计
    OpticalFlow,
    /// 超分辨率
    SuperResolution,
    /// 风格迁移
    StyleTransfer,
    /// 特征匹配模型（LightGlue, SuperGlue 等）
    FeatureMatching,
    /// SAM 交互式分割模型（MobileSAM, SAM 等）
    Sam,
    /// SAM2 交互式分割模型（SAM2.0 / SAM2.1，单文件合并模式）
    Sam2,
    /// Grounded-SAM 开放词表分割（Grounding DINO + SAM2 串联）
    GroundedSam,
    /// 显著性抠图 / 软边缘 matting（BiRefNet、RMBG 等）
    Matting,
    /// 通用模型（自定义后处理）
    Custom,
}

impl ModelType {
    /// 类型代码（对应 `getCode()`）。
    pub fn code(&self) -> &'static str {
        match self {
            ModelType::Classification => "classification",
            ModelType::Detection => "detection",
            ModelType::Segmentation => "segmentation",
            ModelType::Keypoint => "keypoint",
            ModelType::Face => "face",
            ModelType::Depth => "depth",
            ModelType::OpticalFlow => "optical_flow",
            ModelType::SuperResolution => "super_resolution",
            ModelType::StyleTransfer => "style_transfer",
            ModelType::FeatureMatching => "feature_matching",
            ModelType::Sam => "sam",
            ModelType::Sam2 => "sam2",
            ModelType::GroundedSam => "grounded_sam",
            ModelType::Matting => "matting",
            ModelType::Custom => "custom",
        }
    }

    /// 描述（对应 `getDescription()`）。
    pub fn description(&self) -> &'static str {
        match self {
            ModelType::Classification => "图像分类",
            ModelType::Detection => "目标检测",
            ModelType::Segmentation => "实例分割",
            ModelType::Keypoint => "关键点检测",
            ModelType::Face => "人脸检测/识别",
            ModelType::Depth => "深度估计",
            ModelType::OpticalFlow => "光流估计",
            ModelType::SuperResolution => "超分辨率",
            ModelType::StyleTransfer => "风格迁移",
            ModelType::FeatureMatching => "特征匹配",
            ModelType::Sam => "SAM 交互式分割",
            ModelType::Sam2 => "SAM2 交互式分割 (SAM2.0/SAM2.1)",
            ModelType::GroundedSam => "Grounded-SAM 开放词表分割",
            ModelType::Matting => "显著性抠图/Matting",
            ModelType::Custom => "自定义模型",
        }
    }

    /// 根据代码获取类型（未知代码返回 Custom，与原实现一致）。
    pub fn from_code(code: &str) -> Self {
        match code.to_ascii_lowercase().as_str() {
            "classification" => ModelType::Classification,
            "detection" => ModelType::Detection,
            "segmentation" => ModelType::Segmentation,
            "keypoint" => ModelType::Keypoint,
            "face" => ModelType::Face,
            "depth" => ModelType::Depth,
            "optical_flow" => ModelType::OpticalFlow,
            "super_resolution" => ModelType::SuperResolution,
            "style_transfer" => ModelType::StyleTransfer,
            "feature_matching" => ModelType::FeatureMatching,
            "sam" => ModelType::Sam,
            "sam2" => ModelType::Sam2,
            "grounded_sam" => ModelType::GroundedSam,
            "matting" => ModelType::Matting,
            _ => ModelType::Custom,
        }
    }
}
