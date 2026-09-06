//! 目标检测结果。

use super::bounding_box::BoundingBox;

/// 目标检测结果。
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    /// 基础边界框（x1/y1/x2/y2/confidence）
    pub bbox: BoundingBox,
    /// 类别名称
    pub class_name: String,
    /// 类别 ID
    pub class_id: i32,
}

impl Detection {
    pub fn new(
        class_name: impl Into<String>,
        class_id: i32,
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        confidence: f64,
    ) -> Self {
        Detection {
            bbox: BoundingBox::new(x1, y1, x2, y2, confidence),
            class_name: class_name.into(),
            class_id,
        }
    }

    pub fn x1(&self) -> f64 {
        self.bbox.x1
    }

    pub fn y1(&self) -> f64 {
        self.bbox.y1
    }

    pub fn x2(&self) -> f64 {
        self.bbox.x2
    }

    pub fn y2(&self) -> f64 {
        self.bbox.y2
    }

    pub fn confidence(&self) -> f64 {
        self.bbox.confidence
    }
}

impl std::fmt::Display for Detection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Detection[{}(id={}), BBox[{:.1}, {:.1}, {:.1}, {:.1}], conf={:.3}]",
            self.class_name,
            self.class_id,
            self.bbox.x1,
            self.bbox.y1,
            self.bbox.x2,
            self.bbox.y2,
            self.bbox.confidence
        )
    }
}
