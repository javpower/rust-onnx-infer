//! 2D 坐标点。

/// 2D 坐标点，可带标签。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
    /// 带标签的坐标点（可选）
    pub label: Option<String>,
}

impl Point2D {
    pub fn new(x: f64, y: f64) -> Self {
        Point2D {
            x,
            y,
            label: None,
        }
    }

    pub fn with_label(x: f64, y: f64, label: impl Into<String>) -> Self {
        Point2D {
            x,
            y,
            label: Some(label.into()),
        }
    }
}

impl std::fmt::Display for Point2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.label {
            Some(l) => write!(f, "{l}({:.2}, {:.2})", self.x, self.y),
            None => write!(f, "({:.2}, {:.2})", self.x, self.y),
        }
    }
}
