//! 实例分割结果。

use crate::error::{Result, VisionError};
use crate::imaging::{FloatMask, Rect};

use super::detection::Detection;

/// 实例分割结果：检测框 + 类别 + float 掩码（范围 [0,1]，对应 CV_32F Mat）。
#[derive(Debug, Clone, PartialEq)]
pub struct Segmentation {
    /// 基础检测信息
    pub detection: Detection,
    /// 分割掩码（float，[0,1]，尺寸 = 引擎输出掩码尺寸）
    pub mask: Option<FloatMask>,
    /// 掩码阈值（用于二值化）
    pub mask_threshold: f32,
}

impl Segmentation {
    pub fn new(
        class_name: impl Into<String>,
        class_id: i32,
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        confidence: f64,
        mask: Option<FloatMask>,
    ) -> Self {
        Segmentation {
            detection: Detection::new(class_name, class_id, x1, y1, x2, y2, confidence),
            mask,
            mask_threshold: 0.35,
        }
    }

    pub fn class_name(&self) -> &str {
        &self.detection.class_name
    }

    pub fn class_id(&self) -> i32 {
        self.detection.class_id
    }

    pub fn confidence(&self) -> f64 {
        self.detection.bbox.confidence
    }

    /// 获取二值化掩码（> mask_threshold → 255）。
    pub fn binary_mask(&self) -> Result<crate::imaging::Image> {
        match &self.mask {
            None => Err(VisionError::image("segmentation mask is null")),
            Some(m) if m.is_empty() => Err(VisionError::image("segmentation mask is empty")),
            Some(m) => Ok(m.binary_mask(self.mask_threshold)),
        }
    }

    /// 获取掩码面积（二值化后前景像素数；对应 `getMaskArea`）。
    pub fn mask_area(&self) -> f64 {
        match &self.mask {
            None => 0.0,
            Some(m) if m.is_empty() => 0.0,
            Some(m) => m
                .data()
                .iter()
                .filter(|&&v| v > self.mask_threshold)
                .count() as f64,
        }
    }

    /// 掩码在原图坐标系中的前景外接矩形。
    pub fn mask_bounding_rect(&self) -> Rect {
        match &self.mask {
            Some(m) => m.foreground_rect(self.mask_threshold),
            None => Rect::default(),
        }
    }
}

impl std::fmt::Display for Segmentation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mask_desc = match &self.mask {
            Some(m) => format!("{}x{}", m.width(), m.height()),
            None => "null".to_string(),
        };
        write!(
            f,
            "Segmentation[{}(id={}), BBox[{:.1}, {:.1}, {:.1}, {:.1}], conf={:.3}, mask={}]",
            self.detection.class_name,
            self.detection.class_id,
            self.detection.bbox.x1,
            self.detection.bbox.y1,
            self.detection.bbox.x2,
            self.detection.bbox.y2,
            self.detection.bbox.confidence,
            mask_desc
        )
    }
}
