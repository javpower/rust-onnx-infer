//! ONNX 推理引擎工厂。
//!
//! 所有引擎均支持自动检测模型格式（End2End / Traditional），无需手动指定。
//!
//! 返回接口的方法（如 `create_detection_engine`）返回
//! `Box<dyn OnnxInferenceEngine<Output = ...>>`；返回具体类型的方法保持具体类型。
//! 泛型擦除的 `createEngine(...)` / `Builder` 对应 [`DynEngine`] 枚举。
//!
//! `create_dart_engine`（DART 开放词汇检测分割引擎 v2）见该函数。

use std::path::Path;

use crate::core::device_type::DeviceType;
use crate::core::engine::OnnxInferenceEngine;
use crate::core::model_type::ModelType;
use crate::engines::birefnet::BiRefNetEngine;
use crate::engines::classification::ClassificationEngine;
use crate::engines::depth::DepthEstimationEngine;
use crate::engines::detection::DetectionEngine;
use crate::engines::face_detection::{FaceDetectionEngine, FaceDetectResult};
use crate::engines::face_recognition::FaceRecognitionEngine;
use crate::engines::grounded_sam::GroundedSamEngine;
use crate::engines::image_enhance::{ImageEnhanceEngine, ImageEnhanceType};
use crate::engines::lightglue::LightGlueEngine;
use crate::engines::pose::PoseEngine;
use crate::engines::real_esrgan::RealEsrganEngine;
use crate::engines::roma_v2::RomaV2Engine;
use crate::engines::sam::SamEngine;
use crate::engines::sam2::Sam2Engine;
use crate::engines::segmentation::SegmentationEngine;
use crate::engines::style_transfer::StyleTransferEngine;
use crate::engines::yolo_e_runtime::YoloERuntimeEngine;
use crate::error::Result;
use crate::imaging::{FloatMask, Image};
use crate::model::{ClassificationResult, Detection, PoseResult, Segmentation};

/// 类型擦除的引擎句柄（对应上游 泛型 `OnnxInferenceEngine<T>` 的工厂出口）。
pub enum DynEngine {
    /// 图像分类（Output = Option<ClassificationResult>）
    Classification(Box<dyn OnnxInferenceEngine<Output = Option<ClassificationResult>>>),
    /// 目标检测（Output = Vec<Detection>）
    Detection(Box<dyn OnnxInferenceEngine<Output = Vec<Detection>>>),
    /// 实例分割（Output = Vec<Segmentation>）
    Segmentation(Box<dyn OnnxInferenceEngine<Output = Vec<Segmentation>>>),
    /// 显著性抠图（Output = MattingResult）
    Matting(Box<dyn OnnxInferenceEngine<Output = crate::model::MattingResult>>),
    /// SAM 交互式分割
    Sam(Box<SamEngine>),
    /// SAM2 交互式分割
    Sam2(Box<Sam2Engine>),
}

// ==================== 分类引擎 ====================

/// 创建分类引擎（自动检测归一化方式；对应 `createClassificationEngine`）。
pub fn create_classification_engine(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
) -> Result<Box<dyn OnnxInferenceEngine<Output = Option<ClassificationResult>>>> {
    Ok(Box::new(ClassificationEngine::for_auto(model_path, device_type)?))
}

/// 创建分类引擎（指定模型类型；对应重载 `createClassificationEngine(modelType)`）。
pub fn create_classification_engine_typed(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
    model_type: crate::engines::classification::ClassificationModelType,
) -> Result<Box<dyn OnnxInferenceEngine<Output = Option<ClassificationResult>>>> {
    use crate::engines::classification::ClassificationModelType;
    match model_type {
        ClassificationModelType::Yolo => Ok(Box::new(ClassificationEngine::for_yolo(model_path, device_type)?)),
        ClassificationModelType::ImageNet => {
            Ok(Box::new(ClassificationEngine::for_image_net(model_path, device_type)?))
        }
        ClassificationModelType::Auto => Ok(Box::new(ClassificationEngine::for_auto(model_path, device_type)?)),
        _ => Ok(Box::new(ClassificationEngine::new(model_path, device_type, model_type)?)),
    }
}

/// 创建分类引擎（指定输入尺寸）。
pub fn create_classification_engine_with_input_size(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
    model_type: crate::engines::classification::ClassificationModelType,
    input_height: i32,
    input_width: i32,
) -> Result<Box<dyn OnnxInferenceEngine<Output = Option<ClassificationResult>>>> {
    use crate::engines::classification::ClassificationModelType;
    match model_type {
        ClassificationModelType::Yolo => Ok(Box::new(ClassificationEngine::for_yolo_with_input_size(
            model_path,
            device_type,
            input_height,
            input_width,
        )?)),
        ClassificationModelType::ImageNet => Ok(Box::new(
            ClassificationEngine::for_image_net_with_input_size(
                model_path,
                device_type,
                input_height,
                input_width,
            )?,
        )),
        ClassificationModelType::Auto => Ok(Box::new(ClassificationEngine::for_auto_with_input_size(
            model_path,
            device_type,
            input_height,
            input_width,
        )?)),
        _ => Ok(Box::new(ClassificationEngine::new_with_input_size(
            model_path,
            device_type,
            model_type,
            input_height,
            input_width,
        )?)),
    }
}

// ==================== 检测引擎 ====================

/// 创建检测引擎（自动检测 End2End / Traditional；对应 `createDetectionEngine`）。
pub fn create_detection_engine(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
    conf_threshold: f32,
    nms_threshold: f32,
) -> Result<Box<dyn OnnxInferenceEngine<Output = Vec<Detection>>>> {
    Ok(Box::new(DetectionEngine::for_yolo(
        model_path,
        device_type,
        conf_threshold,
        nms_threshold,
    )?))
}

/// 创建检测引擎（指定输入尺寸）。
pub fn create_detection_engine_with_input_size(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
    conf_threshold: f32,
    nms_threshold: f32,
    input_height: i32,
    input_width: i32,
) -> Result<Box<dyn OnnxInferenceEngine<Output = Vec<Detection>>>> {
    Ok(Box::new(DetectionEngine::for_yolo_with_input_size(
        model_path,
        device_type,
        conf_threshold,
        nms_threshold,
        input_height,
        input_width,
    )?))
}

/// 创建 RT-DETR 检测引擎（Transformer 模型；对应 `createRTDETREngine`）。
pub fn create_rtdetr_engine(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
    conf_threshold: f32,
    nms_threshold: f32,
) -> Result<Box<dyn OnnxInferenceEngine<Output = Vec<Detection>>>> {
    Ok(Box::new(DetectionEngine::for_rtdetr(
        model_path,
        device_type,
        conf_threshold,
        nms_threshold,
    )?))
}

/// 创建 RT-DETR 检测引擎（指定输入尺寸）。
pub fn create_rtdetr_engine_with_input_size(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
    conf_threshold: f32,
    nms_threshold: f32,
    input_height: i32,
    input_width: i32,
) -> Result<Box<dyn OnnxInferenceEngine<Output = Vec<Detection>>>> {
    Ok(Box::new(DetectionEngine::for_rtdetr_with_input_size(
        model_path,
        device_type,
        conf_threshold,
        nms_threshold,
        input_height,
        input_width,
    )?))
}

/// 创建 RF-DETR 检测引擎（Roboflow，对应 `createRFDETREngine`）。
pub fn create_rfdetr_engine(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
    conf_threshold: f32,
) -> Result<Box<dyn OnnxInferenceEngine<Output = Vec<Detection>>>> {
    Ok(Box::new(DetectionEngine::for_rfdetr(
        model_path,
        device_type,
        conf_threshold,
    )?))
}

/// 创建 RF-DETR-Seg 分割引擎（对应 `createRFDETRSegEngine`）。
pub fn create_rfdetr_seg_engine(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
    conf_threshold: f32,
) -> Result<Box<dyn OnnxInferenceEngine<Output = Vec<Segmentation>>>> {
    Ok(Box::new(SegmentationEngine::for_rfdetr_seg(
        model_path,
        device_type,
        conf_threshold,
    )?))
}

// ==================== YOLOE ====================

/// 创建 YOLOE 提示推理引擎（文本/视觉提示已烘焙进模型；对应 `createYOLOEEngine`）。
///
/// YOLOE 是分割架构（输出 [1, 4+nc+32, anchors] + proto），必须走 SegmentationEngine。
/// NMS 阈值建议 0.7（与 Ultralytics 默认一致）；类别名从 ONNX 元数据自动加载，
/// 缺失时可用 `set_labels` 手动指定。
pub fn create_yoloe_engine(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
    conf_threshold: f32,
) -> Result<SegmentationEngine> {
    SegmentationEngine::for_yolo(model_path, device_type, conf_threshold, 0.7)
}

/// 创建 YOLOE 运行时视觉提示引擎（实验性，双模型架构；对应 `createYOLOERuntimeEngine`）。
pub fn create_yoloe_runtime_engine(
    encoder_path: impl AsRef<Path>,
    detector_path: impl AsRef<Path>,
    device_type: DeviceType,
) -> Result<YoloERuntimeEngine> {
    YoloERuntimeEngine::new(encoder_path, detector_path, device_type)
}

// ==================== DART 开放词汇检测分割 ====================

/// 创建 DART 开放词汇检测分割引擎 v2（Meta SAM3 骨干 + DART，training-free）。
///
/// `models_dir`：含 tokenizer.json / text_encoder.onnx / enc_dec_hs.onnx /
/// mask_head.onnx / pos_1008.bin / tier_1008/hf_backbone.onnx 的目录。
/// 内存约 3.5GB，CPU 单类推理约 45s，推荐 CUDA/CoreML。
pub fn create_dart_engine(
    models_dir: impl AsRef<Path>,
    device_type: DeviceType,
) -> Result<crate::engines::dart::DartEngine> {
    crate::engines::dart::DartEngine::new(models_dir, device_type)
}

// ==================== 超分 / 图像增强 ====================

/// 创建 Real-ESRGAN 图像超分引擎（2x / 4x；对应 `createRealEsrganEngine`）。
pub fn create_real_esrgan_engine(
    model_path: impl AsRef<Path>,
    scale: i32,
    device_type: DeviceType,
) -> Result<RealEsrganEngine> {
    RealEsrganEngine::new(model_path, scale, device_type)
}

/// 创建图像增强引擎（去噪 / 低光增强 / 去雾；对应 `createImageEnhanceEngine`）。
pub fn create_image_enhance_engine(
    model_path: impl AsRef<Path>,
    enhance_type: ImageEnhanceType,
    device_type: DeviceType,
) -> Result<ImageEnhanceEngine> {
    ImageEnhanceEngine::new(model_path, enhance_type, device_type)
}

// ==================== 分割引擎 ====================

/// 创建分割引擎（自动检测 End2End / Traditional；对应 `createSegmentationEngine`）。
pub fn create_segmentation_engine(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
    conf_threshold: f32,
    nms_threshold: f32,
) -> Result<Box<dyn OnnxInferenceEngine<Output = Vec<Segmentation>>>> {
    Ok(Box::new(SegmentationEngine::for_yolo(
        model_path,
        device_type,
        conf_threshold,
        nms_threshold,
    )?))
}

/// 创建分割引擎（指定输入尺寸）。
pub fn create_segmentation_engine_with_input_size(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
    conf_threshold: f32,
    nms_threshold: f32,
    input_height: i32,
    input_width: i32,
) -> Result<Box<dyn OnnxInferenceEngine<Output = Vec<Segmentation>>>> {
    Ok(Box::new(SegmentationEngine::for_yolo_with_input_size(
        model_path,
        device_type,
        conf_threshold,
        nms_threshold,
        input_height,
        input_width,
    )?))
}

// ==================== 特征匹配 ====================

/// 创建 LightGlue 特征匹配引擎（对应 `createLightGlueEngine`）。
pub fn create_light_glue_engine(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
    match_threshold: f32,
) -> Result<LightGlueEngine> {
    LightGlueEngine::with_match_threshold(model_path, device_type, match_threshold)
}

/// 创建 LightGlue 特征匹配引擎（指定输入尺寸）。
pub fn create_light_glue_engine_with_input_size(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
    match_threshold: f32,
    input_height: i32,
    input_width: i32,
) -> Result<LightGlueEngine> {
    LightGlueEngine::with_input_size(
        model_path,
        device_type,
        match_threshold,
        input_height,
        input_width,
    )
}

/// 创建 RoMaV2 密集特征匹配引擎（对应 `createRomaV2Engine`）。
pub fn create_roma_v2_engine(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
) -> Result<RomaV2Engine> {
    RomaV2Engine::new(model_path, device_type)
}

/// 创建 RoMaV2 密集特征匹配引擎（指定置信度阈值与输入尺寸）。
pub fn create_roma_v2_engine_with_options(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
    overlap_threshold: f32,
    input_height: i32,
    input_width: i32,
) -> Result<RomaV2Engine> {
    RomaV2Engine::with_input_size(
        model_path,
        device_type,
        overlap_threshold,
        input_height,
        input_width,
    )
}

// ==================== SAM / Grounded-SAM / 抠图 ====================

/// 创建 SAM 交互式分割引擎（对应 `createSamEngine`）。
pub fn create_sam_engine(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
) -> Result<SamEngine> {
    SamEngine::new(model_path, device_type)
}

/// 创建 SAM2 交互式分割引擎（单文件合并 ONNX，默认 1024x1024；对应 `createSam2Engine`）。
pub fn create_sam2_engine(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
) -> Result<Sam2Engine> {
    Sam2Engine::new(model_path, device_type)
}

/// 创建 SAM2 交互式分割引擎（指定输入尺寸）。
pub fn create_sam2_engine_with_input_size(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
    input_size: i32,
) -> Result<Sam2Engine> {
    Sam2Engine::with_input_size(model_path, device_type, input_size)
}

/// 创建 Grounded-SAM 端到端流水线引擎（Grounding DINO + SAM2；对应 `createGroundedSamEngine`）。
pub fn create_grounded_sam_engine(
    dino_model_path: impl AsRef<Path>,
    tokenizer_path: impl AsRef<Path>,
    sam2_model_path: impl AsRef<Path>,
    device_type: DeviceType,
) -> Result<GroundedSamEngine> {
    GroundedSamEngine::new(
        dino_model_path,
        tokenizer_path,
        sam2_model_path,
        device_type,
    )
}

/// 创建 BiRefNet 显著性抠图引擎（默认 1024×1024；对应 `createBiRefNetEngine`）。
pub fn create_birefnet_engine(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
) -> Result<BiRefNetEngine> {
    BiRefNetEngine::new(model_path, device_type)
}

/// 创建 BiRefNet 引擎（指定输入边长，常见 1024）。
pub fn create_birefnet_engine_with_input_size(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
    input_size: i32,
) -> Result<BiRefNetEngine> {
    BiRefNetEngine::with_input_size(model_path, device_type, input_size)
}

// ==================== 姿态 / 人脸 / 深度 / 风格迁移 ====================

/// 创建 YOLO-Pose 姿态估计引擎（COCO 17 关键点；NMS 阈值固定 0.45，与 Ultralytics 默认一致）。
pub fn create_pose_engine(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
    conf_threshold: f32,
) -> Result<Box<dyn OnnxInferenceEngine<Output = Vec<PoseResult>>>> {
    Ok(Box::new(PoseEngine::for_yolo(
        model_path,
        device_type,
        conf_threshold,
        0.45,
    )?))
}

/// 创建 YuNet 人脸检测引擎（检测框 + 5 点 landmark）。
pub fn create_face_detection_engine(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
) -> Result<Box<dyn OnnxInferenceEngine<Output = Vec<FaceDetectResult>>>> {
    Ok(Box::new(FaceDetectionEngine::new(model_path, device_type)?))
}

/// 创建人脸识别引擎（SFace 预处理约定，对应 OpenCV Zoo `face_recognition_sface_2021dec.onnx`）。
///
/// 返回具体类型：embedding 提取依赖 5 点对齐（[`FaceRecognitionEngine::extract`]），
/// 不走统一 `predict` 接口。ArcFace / MobileFaceNet 模型请用
/// `FaceRecognitionEngine::with_norm_mode(path, device, FaceNormMode::InsightFace)`。
pub fn create_face_recognition_engine(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
) -> Result<FaceRecognitionEngine> {
    FaceRecognitionEngine::new_sface(model_path, device_type)
}

/// 创建 Depth Anything V2 深度估计引擎。
///
/// 输出原图分辨率的相对深度 [`FloatMask`]（min-max 归一化到 [0,1]，近 = 1 远 = 0）。
pub fn create_depth_engine(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
) -> Result<Box<dyn OnnxInferenceEngine<Output = FloatMask>>> {
    Ok(Box::new(DepthEstimationEngine::new(model_path, device_type)?))
}

/// 创建风格迁移引擎（cycleGAN Model Zoo 导出；输出与输入同尺寸 BGR 图）。
pub fn create_style_transfer_engine(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
) -> Result<Box<dyn OnnxInferenceEngine<Output = Image>>> {
    Ok(Box::new(StyleTransferEngine::new(model_path, device_type)?))
}

// ==================== 实时姿态 / 人脸属性 / OCR ====================

/// 创建 RTMO 实时姿态引擎（一阶段，实时级）。
pub fn create_realtime_pose_engine(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
) -> Result<Box<dyn OnnxInferenceEngine<Output = Vec<crate::model::PoseResult>>>> {
    Ok(Box::new(crate::engines::pose_rt::RealtimePoseEngine::new(
        model_path,
        device_type,
    )?))
}

/// 创建年龄性别估计引擎（输入人脸框；配合 YuNet 人脸检测使用）。
pub fn create_age_gender_engine(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
) -> Result<crate::engines::face_attribute::AgeGenderEngine> {
    crate::engines::face_attribute::AgeGenderEngine::new(model_path, device_type)
}

/// 创建表情识别引擎（输入人脸框；配合 YuNet 人脸检测使用）。
pub fn create_expression_engine(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
) -> Result<crate::engines::face_attribute::ExpressionEngine> {
    crate::engines::face_attribute::ExpressionEngine::new(model_path, device_type)
}

/// 创建 OCR 检测+识别流水线（PaddleOCR det + rec + 字典）。
pub fn create_ocr_pipeline(
    det_model_path: impl AsRef<Path>,
    rec_model_path: impl AsRef<Path>,
    dict_path: impl AsRef<Path>,
    device_type: DeviceType,
) -> Result<crate::engines::ocr::OcrPipeline> {
    crate::engines::ocr::OcrPipeline::new(det_model_path, rec_model_path, dict_path, device_type)
}

// ==================== 扩展能力：OBB / 质量 / 二维码 / 分割系 / 关键点系 ====================

/// 创建 YOLO-OBB 旋转框检测引擎（v8/11 传统 + YOLO26 End2End 自动识别）。
pub fn create_obb_engine(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
) -> Result<Box<dyn OnnxInferenceEngine<Output = Vec<crate::engines::obb_detection::ObbResult>>>> {
    Ok(Box::new(crate::engines::obb_detection::ObbDetectionEngine::new(
        model_path,
        device_type,
    )?))
}

/// 创建手部检测引擎（MediaPipe palm detector）。
pub fn create_hand_detection_engine(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
) -> Result<Box<dyn OnnxInferenceEngine<Output = Vec<crate::engines::hand_keypoint::HandDetection>>>> {
    Ok(Box::new(crate::engines::hand_keypoint::HandDetectionEngine::new(
        model_path,
        device_type,
    )?))
}

/// 创建语义分割引擎（SegFormer ADE20K）。
pub fn create_semantic_segmentation_engine(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
) -> Result<Box<dyn OnnxInferenceEngine<Output = crate::engines::semantic_segmentation::SemanticMap>>> {
    Ok(Box::new(
        crate::engines::semantic_segmentation::SemanticSegmentationEngine::new(model_path, device_type)?,
    ))
}

/// 创建人体解析引擎（SegFormer-B2 人衣 18 类；建议 512×512 输入）。
pub fn create_human_parsing_engine(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
) -> Result<Box<dyn OnnxInferenceEngine<Output = crate::engines::human_parsing::ParsingMap>>> {
    Ok(Box::new(crate::engines::human_parsing::HumanParsingEngine::new(
        model_path,
        device_type,
    )?))
}

/// 创建人像分割引擎（PP-HumanSeg / MODNet）。
pub fn create_portrait_matting_engine(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
) -> Result<Box<dyn OnnxInferenceEngine<Output = crate::imaging::FloatMask>>> {
    Ok(Box::new(crate::engines::portrait_matting::PortraitMattingEngine::new(
        model_path,
        device_type,
    )?))
}

/// 创建去模糊引擎（NafNet）。
pub fn create_deblur_engine(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
) -> Result<Box<dyn OnnxInferenceEngine<Output = Image>>> {
    Ok(Box::new(crate::engines::deblur::DeblurEngine::new(model_path, device_type)?))
}

/// 创建人脸 106 关键点引擎（insightface 2d106det）。
pub fn create_face_landmark_engine(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
) -> Result<crate::engines::face_landmark106::FaceLandmark106Engine> {
    crate::engines::face_landmark106::FaceLandmark106Engine::new(model_path, device_type)
}

/// 创建手部 21 关键点引擎（RTMPose-hand；配合 create_hand_detection_engine 串联）。
pub fn create_hand_landmark_engine(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
) -> Result<crate::engines::hand_keypoint::HandLandmarkEngine> {
    crate::engines::hand_keypoint::HandLandmarkEngine::new(model_path, device_type)
}

/// 创建 WholeBody 133 关键点引擎（RTMPose-m-wholebody；配合行人检测串联）。
pub fn create_wholebody_engine(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
) -> Result<Box<dyn OnnxInferenceEngine<Output = crate::model::PoseResult>>> {
    Ok(Box::new(crate::engines::wholebody::WholeBodyPoseEngine::new(
        model_path,
        device_type,
    )?))
}

// ==================== 行为 / 手势 / 车牌 ====================

/// 创建骨架动作识别引擎（ST-GCN 系，COCO17 帧序列 → 动作类别）。
pub fn create_action_recognition_engine(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
) -> Result<Box<dyn OnnxInferenceEngine<Output = crate::engines::action_recognition::ActionPrediction>>> {
    Ok(Box::new(
        crate::engines::action_recognition::ActionRecognitionEngine::new(model_path, device_type)?,
    ))
}

/// 创建手势分类引擎（纯几何规则，手部 21 关键点输入）。
pub fn create_gesture_engine() -> crate::engines::gesture::GestureClassifier {
    crate::engines::gesture::GestureClassifier::new()
}

/// 创建车牌检测+识别引擎（mnet 检测 + LPRNet 识别）。
pub fn create_license_plate_engine(
    det_model_path: impl AsRef<Path>,
    rec_model_path: impl AsRef<Path>,
    device_type: DeviceType,
) -> Result<Box<dyn OnnxInferenceEngine<Output = Vec<crate::engines::license_plate::PlateResult>>>> {
    Ok(Box::new(crate::engines::license_plate::LicensePlateEngine::new(
        det_model_path,
        rec_model_path,
        device_type,
    )?))
}

// ==================== 类型擦除工厂（对应上游 createEngine / Builder） ====================

/// 自动检测模型类型并创建引擎（对应上游 泛型 `createEngine`）。
pub fn create_engine(
    model_path: impl AsRef<Path>,
    device_type: DeviceType,
    model_type: ModelType,
) -> Result<DynEngine> {
    match model_type {
        ModelType::Classification => Ok(DynEngine::Classification(
            create_classification_engine(model_path, device_type)?,
        )),
        ModelType::Detection => Ok(DynEngine::Detection(create_detection_engine(
            model_path,
            device_type,
            0.5,
            0.45,
        )?)),
        ModelType::Segmentation => Ok(DynEngine::Segmentation(create_segmentation_engine(
            model_path,
            device_type,
            0.5,
            0.45,
        )?)),
        ModelType::FeatureMatching => {
            // 上游此分支实际返回 LightGlue（特征匹配），但类型擦除容器不承载匹配引擎；
            // 请使用返回具体类型的 create_light_glue_engine
            Err(crate::error::VisionError::Unsupported(
                "FEATURE_MATCHING 请使用 create_light_glue_engine（返回具体类型）".to_string(),
            ))
        }
        ModelType::Sam => Ok(DynEngine::Sam(Box::new(create_sam_engine(
            model_path,
            device_type,
        )?))),
        ModelType::Sam2 => Ok(DynEngine::Sam2(Box::new(create_sam2_engine(
            model_path,
            device_type,
        )?))),
        ModelType::Matting => Ok(DynEngine::Matting(Box::new(create_birefnet_engine(
            model_path,
            device_type,
        )?))),
        other => Err(crate::error::VisionError::InvalidArgument(format!(
            "Unsupported model type: {:?} ({})",
            other,
            other.code()
        ))),
    }
}

impl DynEngine {
    /// 输入尺寸 (width, height)。
    pub fn input_size(&self) -> (i32, i32) {
        match self {
            DynEngine::Classification(e) => e.input_size(),
            DynEngine::Detection(e) => e.input_size(),
            DynEngine::Segmentation(e) => e.input_size(),
            DynEngine::Matting(e) => e.input_size(),
            DynEngine::Sam(e) => e.input_size(),
            DynEngine::Sam2(e) => e.input_size(),
        }
    }

    /// 类别标签。
    pub fn labels(&self) -> Option<&[String]> {
        match self {
            DynEngine::Classification(e) => e.labels(),
            DynEngine::Detection(e) => e.labels(),
            DynEngine::Segmentation(e) => e.labels(),
            DynEngine::Matting(e) => e.labels(),
            DynEngine::Sam(e) => e.labels(),
            DynEngine::Sam2(e) => e.labels(),
        }
    }
}

/// 引擎构建器（对应 `OnnxEngineFactory.Builder`）。
#[derive(Debug, Clone)]
pub struct EngineBuilder {
    model_path: Option<String>,
    device_type: DeviceType,
    model_type: ModelType,
    conf_threshold: f32,
    nms_threshold: f32,
    input_height: i32,
    input_width: i32,
}

impl Default for EngineBuilder {
    fn default() -> Self {
        EngineBuilder {
            model_path: None,
            device_type: DeviceType::Auto,
            model_type: ModelType::Detection,
            conf_threshold: 0.5,
            nms_threshold: 0.45,
            input_height: -1,
            input_width: -1,
        }
    }
}

impl EngineBuilder {
    pub fn model_path(mut self, path: impl Into<String>) -> Self {
        self.model_path = Some(path.into());
        self
    }

    pub fn device_type(mut self, device_type: DeviceType) -> Self {
        self.device_type = device_type;
        self
    }

    pub fn model_type(mut self, model_type: ModelType) -> Self {
        self.model_type = model_type;
        self
    }

    pub fn conf_threshold(mut self, threshold: f32) -> Self {
        self.conf_threshold = threshold;
        self
    }

    pub fn nms_threshold(mut self, threshold: f32) -> Self {
        self.nms_threshold = threshold;
        self
    }

    pub fn input_size(mut self, height: i32, width: i32) -> Self {
        self.input_height = height;
        self.input_width = width;
        self
    }

    /// 构建（对应 `Builder.build()`；泛型擦除由 [`DynEngine`] 承载）。
    pub fn build(self) -> Result<DynEngine> {
        let Some(model_path) = self.model_path else {
            return Err(crate::error::VisionError::InvalidArgument(
                "Model path is required".to_string(),
            ));
        };
        let has_input = self.input_height > 0 && self.input_width > 0;

        match self.model_type {
            ModelType::Classification => {
                if has_input {
                    Ok(DynEngine::Classification(Box::new(
                        ClassificationEngine::for_auto_with_input_size(
                            model_path,
                            self.device_type,
                            self.input_height,
                            self.input_width,
                        )?,
                    )))
                } else {
                    Ok(DynEngine::Classification(create_classification_engine(
                        model_path,
                        self.device_type,
                    )?))
                }
            }
            ModelType::Detection => {
                if has_input {
                    Ok(DynEngine::Detection(create_detection_engine_with_input_size(
                        model_path,
                        self.device_type,
                        self.conf_threshold,
                        self.nms_threshold,
                        self.input_height,
                        self.input_width,
                    )?))
                } else {
                    Ok(DynEngine::Detection(create_detection_engine(
                        model_path,
                        self.device_type,
                        self.conf_threshold,
                        self.nms_threshold,
                    )?))
                }
            }
            ModelType::Segmentation => {
                if has_input {
                    Ok(DynEngine::Segmentation(
                        create_segmentation_engine_with_input_size(
                            model_path,
                            self.device_type,
                            self.conf_threshold,
                            self.nms_threshold,
                            self.input_height,
                            self.input_width,
                        )?,
                    ))
                } else {
                    Ok(DynEngine::Segmentation(create_segmentation_engine(
                        model_path,
                        self.device_type,
                        self.conf_threshold,
                        self.nms_threshold,
                    )?))
                }
            }
            ModelType::Matting => {
                if self.input_height > 0 {
                    Ok(DynEngine::Matting(Box::new(
                        create_birefnet_engine_with_input_size(
                            model_path,
                            self.device_type,
                            self.input_height,
                        )?,
                    )))
                } else {
                    Ok(DynEngine::Matting(Box::new(create_birefnet_engine(
                        model_path,
                        self.device_type,
                    )?)))
                }
            }
            ModelType::Sam => Ok(DynEngine::Sam(Box::new(create_sam_engine(
                model_path,
                self.device_type,
            )?))),
            ModelType::Sam2 => Ok(DynEngine::Sam2(Box::new(create_sam2_engine(
                model_path,
                self.device_type,
            )?))),
            other => Err(crate::error::VisionError::InvalidArgument(format!(
                "Unsupported model type: {:?} ({})",
                other,
                other.code()
            ))),
        }
    }
}

/// 创建引擎构建器（对应 `OnnxEngineFactory.builder()`）。
pub fn builder() -> EngineBuilder {
    EngineBuilder::default()
}
