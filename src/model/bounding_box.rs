//! 边界框。

use super::point2d::Point2D;

/// 边界框（x1/y1 左上、x2/y2 右下，含置信度）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    /// 左上角 X 坐标
    pub x1: f64,
    /// 左上角 Y 坐标
    pub y1: f64,
    /// 右下角 X 坐标
    pub x2: f64,
    /// 右下角 Y 坐标
    pub y2: f64,
    /// 置信度
    pub confidence: f64,
}

impl BoundingBox {
    pub fn new(x1: f64, y1: f64, x2: f64, y2: f64, confidence: f64) -> Self {
        BoundingBox {
            x1,
            y1,
            x2,
            y2,
            confidence,
        }
    }

    /// 无置信度（默认 1.0）。
    pub fn with_default_conf(x1: f64, y1: f64, x2: f64, y2: f64) -> Self {
        BoundingBox::new(x1, y1, x2, y2, 1.0)
    }

    pub fn width(&self) -> f64 {
        self.x2 - self.x1
    }

    pub fn height(&self) -> f64 {
        self.y2 - self.y1
    }

    pub fn center_x(&self) -> f64 {
        (self.x1 + self.x2) / 2.0
    }

    pub fn center_y(&self) -> f64 {
        (self.y1 + self.y2) / 2.0
    }

    pub fn center(&self) -> Point2D {
        Point2D::new(self.center_x(), self.center_y())
    }

    pub fn area(&self) -> f64 {
        self.width() * self.height()
    }

    /// 计算与另一个框的 IoU。
    pub fn iou(&self, other: &BoundingBox) -> f64 {
        let x1 = self.x1.max(other.x1);
        let y1 = self.y1.max(other.y1);
        let x2 = self.x2.min(other.x2);
        let y2 = self.y2.min(other.y2);

        let inter_w = (x2 - x1).max(0.0);
        let inter_h = (y2 - y1).max(0.0);
        let inter_area = inter_w * inter_h;
        let union = self.area() + other.area() - inter_area;

        if union > 0.0 {
            inter_area / union
        } else {
            0.0
        }
    }

    /// 转为数组 [x1, y1, x2, y2]。
    pub fn to_array(&self) -> [f32; 4] {
        [
            self.x1 as f32,
            self.y1 as f32,
            self.x2 as f32,
            self.y2 as f32,
        ]
    }
}

impl std::fmt::Display for BoundingBox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BBox[{:.1}, {:.1}, {:.1}, {:.1}] conf={:.3}",
            self.x1, self.y1, self.x2, self.y2, self.confidence
        )
    }
}
