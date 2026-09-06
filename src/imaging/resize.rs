//! resize（对应 `opencv_imgproc::resize`，插值语义对齐 INTER_LINEAR/NEAREST/AREA/CUBIC）。

use crate::error::{Result, VisionError};
use crate::imaging::image::Image;

/// 插值方式（对应 OpenCV `INTER_*` 常量）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    /// 最近邻（`INTER_NEAREST`）
    Nearest,
    /// 双线性（`INTER_LINEAR`，默认）
    Linear,
    /// 区域平均（`INTER_AREA`，缩小质量最好）
    Area,
    /// 双三次（`INTER_CUBIC`）
    Cubic,
}

impl Default for Interpolation {
    fn default() -> Self {
        Interpolation::Linear
    }
}

impl Interpolation {
    fn to_fir_alg(self) -> fast_image_resize::ResizeAlg {
        use fast_image_resize::{FilterType, ResizeAlg};
        match self {
            Interpolation::Nearest => ResizeAlg::Nearest,
            // Interpolation 系列使用固定核大小，行为与 OpenCV 一致
            Interpolation::Linear => ResizeAlg::Interpolation(FilterType::Bilinear),
            Interpolation::Cubic => ResizeAlg::Interpolation(FilterType::CatmullRom),
            Interpolation::Area => ResizeAlg::Convolution(FilterType::Box),
        }
    }

    fn to_fir_pixel_type(channels: usize) -> Option<fast_image_resize::PixelType> {
        use fast_image_resize::PixelType;
        match channels {
            1 => Some(PixelType::U8),
            2 => Some(PixelType::U8x2),
            3 => Some(PixelType::U8x3),
            4 => Some(PixelType::U8x4),
            _ => None,
        }
    }
}

/// 拉伸 resize 到目标尺寸（对应 `resize(src, dst, Size(w, h), INTER_x)`）。
pub fn resize(image: &Image, width: usize, height: usize, interp: Interpolation) -> Result<Image> {
    if width == 0 || height == 0 {
        return Err(VisionError::image("resize target size must be > 0"));
    }
    if image.is_empty() {
        return Err(VisionError::image("cannot resize empty image"));
    }
    let pixel_type = Interpolation::to_fir_pixel_type(image.channels()).ok_or_else(|| {
        VisionError::image(format!("unsupported channel count {}", image.channels()))
    })?;

    let src = fast_image_resize::images::Image::from_vec_u8(
        image.width() as u32,
        image.height() as u32,
        image.data().to_vec(),
        pixel_type,
    )
    .map_err(|e| VisionError::image(format!("resize source: {e}")))?;

    let mut dst = fast_image_resize::images::Image::new(
        width as u32,
        height as u32,
        pixel_type,
    );

    let options = fast_image_resize::ResizeOptions {
        algorithm: interp.to_fir_alg(),
        cropping: fast_image_resize::SrcCropping::None,
        // 无 alpha 通道语义；显式关闭避免对 2/4 通道做预乘处理
        mul_div_alpha: false,
    };

    let mut resizer = fast_image_resize::Resizer::new();
    resizer
        .resize(&src, &mut dst, &options)
        .map_err(|e| VisionError::image(format!("resize failed: {e}")))?;

    Image::from_raw(width, height, image.channels(), dst.into_vec())
}

/// 等比缩放到能放入 `width x height` 的最大尺寸（不放大，仅缩小；用于 SAHI 等）。
pub fn resize_within(image: &Image, width: usize, height: usize, interp: Interpolation) -> Result<Image> {
    let scale = (width as f64 / image.width() as f64)
        .min(height as f64 / image.height() as f64)
        .min(1.0);
    let nw = ((image.width() as f64 * scale).round() as usize).max(1);
    let nh = ((image.height() as f64 * scale).round() as usize).max(1);
    resize(image, nw, nh, interp)
}
