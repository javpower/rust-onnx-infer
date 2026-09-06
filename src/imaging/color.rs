//! 颜色空间转换（对应 `opencv_imgproc::cvtColor` 的常用子集）。
//!
//! 公式与 OpenCV 对齐：
//! - `BGR2GRAY`: Y = 0.299R + 0.587G + 0.114B（四舍五入）

use crate::error::{Result, VisionError};
use crate::imaging::image::Image;

/// 颜色转换码（对应 OpenCV `COLOR_*` 常量子集）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorConversion {
    Bgr2Rgb,
    Rgb2Bgr,
    Bgra2Rgb,
    Bgra2Bgr,
    Bgra2Gray,
    Bgr2Gray,
    Gray2Rgb,
    Gray2Bgr,
}

/// 通用颜色转换；输入通道数不符合要求时报错。
pub fn cvt_color(image: &Image, code: ColorConversion) -> Result<Image> {
    let (w, h) = (image.width(), image.height());
    let src = image.data();

    let convert = |channels: usize, f: &dyn Fn(&[u8]) -> Vec<u8>| -> Result<Image> {
        if image.channels() != channels {
            return Err(VisionError::image(format!(
                "cvtColor expects {channels} channels, got {}",
                image.channels()
            )));
        }
        let mut out = Vec::with_capacity(w * h * f(&vec![0u8; channels]).len());
        for px in src.chunks_exact(channels) {
            out.extend_from_slice(&f(px));
        }
        Ok(Image::from_raw(w, h, out.len() / (w * h), out)?)
    };

    match code {
        ColorConversion::Bgr2Rgb => convert(3, &|px| vec![px[2], px[1], px[0]]),
        ColorConversion::Rgb2Bgr => convert(3, &|px| vec![px[2], px[1], px[0]]),
        ColorConversion::Bgra2Rgb => convert(4, &|px| vec![px[2], px[1], px[0]]),
        ColorConversion::Bgra2Bgr => convert(4, &|px| vec![px[0], px[1], px[2]]),
        ColorConversion::Bgra2Gray => convert(4, &|px| {
            vec![bgr_to_gray(px[0], px[1], px[2])]
        }),
        ColorConversion::Bgr2Gray => convert(3, &|px| {
            vec![bgr_to_gray(px[0], px[1], px[2])]
        }),
        ColorConversion::Gray2Rgb | ColorConversion::Gray2Bgr => convert(1, &|px| {
            vec![px[0], px[0], px[0]]
        }),
    }
}

#[inline]
fn bgr_to_gray(b: u8, g: u8, r: u8) -> u8 {
    // OpenCV: round(0.299R + 0.587G + 0.114B)
    let y = 0.299_f32 * r as f32 + 0.587_f32 * g as f32 + 0.114_f32 * b as f32;
    y.round().clamp(0.0, 255.0) as u8
}

impl Image {
    /// BGR→RGB（in-place 交换）。
    pub fn bgr_to_rgb_in_place(&mut self) {
        if self.channels() == 3 {
            for px in self.data_mut().chunks_exact_mut(3) {
                px.swap(0, 2);
            }
        }
    }
}
