//! 纯 Rust 图像处理模块。
//!
//! 提供引擎所需的全部图像原语：`Image` 容器、resize、颜色空间转换、
//! 阈值化、形态学、外接矩形、边界填充与统计。

mod color;
mod image;
mod ops;
mod resize;

pub use color::{cvt_color, ColorConversion};
pub use image::Image;
pub use ops::{
    bounding_rect, convert_to, copy_make_border, get_structuring_element,
    min_max_loc, morphology_ex, threshold, BorderType, FloatMask, KernelShape, MorphType, Rect,
    ThresholdType,
};
pub use resize::{resize, resize_within, Interpolation};

use crate::error::Result;

/// 通用 cvtColor 入口（`cvtColor(src, dst, code)`）。
pub fn convert_color(image: &Image, code: ColorConversion) -> Result<Image> {
    cvt_color(image, code)
}
