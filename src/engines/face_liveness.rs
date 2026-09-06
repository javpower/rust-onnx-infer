//! 人脸活体检测引擎（CDCN 分割式 anti-spoofing，huyquangthai/AntiSpoofing 转换版）。
//!
//! 输入 224×224 RGB 人脸区域，输出 28×28 逐像素 spoof 概率图（均值即活体分数，
//! 低=真脸、高=假脸/翻拍）+ 512 维防伪嵌入。配合 YuNet 人脸检测串联使用。
//!
//! 备注：MiniFASNet 系（Silent-Face 官方 pth 转换版）在 torch ≥2.x 加载后
//! 输出退化（对任意输入恒定），故选用 CDCN 分割式模型。

use crate::core::base::BaseOnnxEngine;
use crate::core::device_type::DeviceType;
use crate::core::engine::OnnxInferenceEngine;
use crate::error::{Result, VisionError};
use crate::imaging::{Image, Rect};

/// 模型输入边长。
pub const INPUT_SIZE: usize = 224;

/// 活体检测结果。
#[derive(Debug, Clone, PartialEq)]
pub struct LivenessResult {
    /// 是否判定为真实人脸（spoof 均分 < 阈值）
    pub is_real: bool,
    /// spoof 概率图均值（低=真脸，高=假脸/翻拍）
    pub spoof_score: f32,
    /// spoof 概率图（28×28，原图相对位置）
    pub spoof_map: Vec<f32>,
    /// 防伪嵌入（512 维，可用于聚合同人多次判定）
    pub embedding: Vec<f32>,
}

impl std::fmt::Display for LivenessResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Liveness[{} spoof={:.3}]",
            if self.is_real { "REAL" } else { "FAKE" },
            self.spoof_score
        )
    }
}

/// 人脸活体检测引擎。
pub struct FaceLivenessEngine {
    /// 组合基类
    pub base: BaseOnnxEngine,
    /// spoof 分数阈值（均值 < 阈值判真脸）
    spoof_threshold: f32,
}

impl FaceLivenessEngine {
    /// 创建活体检测引擎（输入 224×224）。
    pub fn new(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        let mut base = BaseOnnxEngine::with_input_size(
            model_path,
            device_type,
            INPUT_SIZE as i32,
            INPUT_SIZE as i32,
        )?;
        // 该模型实测预处理：RGB + ImageNet mean/std（与 HF 模型卡一致）
        base.set_normalization([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
        Ok(FaceLivenessEngine {
            base,
            spoof_threshold: 0.5,
        })
    }

    /// spoof 分数阈值（均值 < 阈值判真脸）。
    pub fn spoof_threshold(&self) -> f32 {
        self.spoof_threshold
    }

    /// 设置 spoof 分数阈值。
    pub fn set_spoof_threshold(&mut self, threshold: f32) {
        self.spoof_threshold = threshold.clamp(0.0, 1.0);
    }

    /// 对单个人脸区域推理活体（`face_box`：原图人脸框，内部拉伸到 224×224）。
    pub fn predict_face(&self, image: &Image, face_box: Rect) -> Result<LivenessResult> {
        let x0 = face_box.x.max(0) as usize;
        let y0 = face_box.y.max(0) as usize;
        let x1 = (face_box.x + face_box.width as i32).clamp(0, image.width() as i32) as usize;
        let y1 = (face_box.y + face_box.height as i32).clamp(0, image.height() as i32) as usize;
        if x1 <= x0 || y1 <= y0 {
            return Err(VisionError::invalid_argument("人脸框无效"));
        }
        let crop = image.crop(x0, y0, x1 - x0, y1 - y0)?;
        let input = self.base.preprocess(&crop)?;
        let tensor = self.base.create_input_tensor(input)?;
        let outputs = self.base.run_multi_output(tensor)?;
        if outputs.len() < 2 {
            return Err(VisionError::inference(
                "活体模型应有 spoofing_map/embedding_map 两个输出",
            ));
        }
        let smap = outputs[0].as_f32()?.to_vec();
        let emb = outputs[1].as_f32()?.to_vec();
        if smap.is_empty() {
            return Err(VisionError::inference("spoofing_map 为空"));
        }
        let spoof_score = smap.iter().sum::<f32>() / smap.len() as f32;
        Ok(LivenessResult {
            is_real: spoof_score < self.spoof_threshold,
            spoof_score,
            spoof_map: smap,
            embedding: emb,
        })
    }

    /// 批量推理（逐脸）。
    pub fn predict_batch(&self, image: &Image, face_boxes: &[Rect]) -> Result<Vec<LivenessResult>> {
        face_boxes.iter().map(|b| self.predict_face(image, *b)).collect()
    }
}

#[async_trait::async_trait]
impl OnnxInferenceEngine for FaceLivenessEngine {
    type Output = LivenessResult;

    fn predict(&self, image: &Image) -> Result<Self::Output> {
        let box_rect = Rect::new(0, 0, image.width() as i32, image.height() as i32);
        self.predict_face(image, box_rect)
    }

    fn input_size(&self) -> (i32, i32) {
        self.base.input_size()
    }

    fn labels(&self) -> Option<&[String]> {
        self.base.labels()
    }

    fn set_labels(&mut self, labels: Vec<String>) {
        self.base.set_labels(labels)
    }

    fn set_confidence_threshold(&mut self, threshold: f32) {
        self.spoof_threshold = threshold;
    }

    fn confidence_threshold(&self) -> f32 {
        self.spoof_threshold
    }
}
