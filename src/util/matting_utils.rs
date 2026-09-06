//! 抠图结果可视化 / 合成工具。
//!
//! 原版为纯静态工具类（`public final class` + 私有构造器），Rust 侧以模块级
//! 函数提供等价能力。`Mat`（CV_32FC1 alpha）统一映射为 [`FloatMask`]，
//! `Mat`（BGR/BGRA 8U）映射为 [`Image`]。

use crate::error::{Result, VisionError};
use crate::imaging::{FloatMask, Image};

/// 将 alpha 应用到 BGR 图，生成 BGRA cutout（透明背景）。
///
/// 对应 `MattingUtils.cutoutBgra(bgr, alpha)`。
/// 新图由调用方持有。
///
/// # 参数
/// - `bgr`: 原图（1/3/4 通道，通道序 BGR/BGRA）
/// - `alpha`: CV_32FC1 [0,1]（对应 32F Mat），尺寸须与 `bgr` 一致
pub fn cutout_bgra(bgr: &Image, alpha: &FloatMask) -> Result<Image> {
    if bgr.is_empty() || alpha.is_empty() {
        return Err(VisionError::invalid_argument("bgr/alpha empty"));
    }
    if bgr.height() != alpha.height() || bgr.width() != alpha.width() {
        return Err(VisionError::invalid_argument("bgr and alpha size mismatch"));
    }
    let rows = bgr.height();
    let cols = bgr.width();
    let ch = bgr.channels();
    let mut out = Image::new(cols, rows, 4);
    {
        let src = bgr.data();
        let a_idx = alpha.data();
        let dst = out.data_mut();
        for y in 0..rows {
            for x in 0..cols {
                let i = y * cols + x;
                // a = max(0, min(1, a))
                let a = a_idx[i].max(0.0).min(1.0);
                let b = if ch >= 1 { src[i * ch] as i32 } else { 0 };
                let g = if ch >= 2 { src[i * ch + 1] as i32 } else { b };
                let r = if ch >= 3 { src[i * ch + 2] as i32 } else { b };
                let o = i * 4;
                dst[o] = b as u8;
                dst[o + 1] = g as u8;
                dst[o + 2] = r as u8;
                dst[o + 3] = (a * 255.0).round() as u8;
            }
        }
    }
    Ok(out)
}

/// 棋盘格背景合成预览（BGR），便于肉眼看半透明边缘。
///
/// 对应 `MattingUtils.compositeOnCheckerboard(bgr, alpha, blockSize)`。
/// `block_size < 2` 时自动改用默认值 16。
pub fn composite_on_checkerboard(bgr: &Image, alpha: &FloatMask, block_size: usize) -> Result<Image> {
    let block_size = if block_size < 2 { 16 } else { block_size };
    if bgr.is_empty() || alpha.is_empty() {
        return Err(VisionError::invalid_argument("bgr/alpha empty"));
    }
    if bgr.height() != alpha.height() || bgr.width() != alpha.width() {
        return Err(VisionError::invalid_argument("bgr and alpha size mismatch"));
    }
    let rows = bgr.height();
    let cols = bgr.width();
    let ch = bgr.channels();
    let mut out = Image::new(cols, rows, 3);
    {
        let src = bgr.data();
        let a_idx = alpha.data();
        let dst = out.data_mut();
        for y in 0..rows {
            for x in 0..cols {
                let i = y * cols + x;
                // 棋盘格：亮格 200，暗格 80
                let light = ((y / block_size) + (x / block_size)) % 2 == 0;
                let cb = if light { 200.0 } else { 80.0 };
                let a = a_idx[i].max(0.0).min(1.0) as f64;
                let b = if ch >= 1 { src[i * ch] as f64 } else { 0.0 };
                let g = if ch >= 2 { src[i * ch + 1] as f64 } else { b };
                let r = if ch >= 3 { src[i * ch + 2] as f64 } else { b };
                let o = i * 3;
                dst[o] = (b * a + cb * (1.0 - a)).round() as u8;
                dst[o + 1] = (g * a + cb * (1.0 - a)).round() as u8;
                dst[o + 2] = (r * a + cb * (1.0 - a)).round() as u8;
            }
        }
    }
    Ok(out)
}

/// alpha [0,1] → 灰度 0~255 预览（对应 `alphaToGrayU8`，
/// 即 `convertTo(gray, CV_8UC1, 255.0, 0)`）。
pub fn alpha_to_gray_u8(alpha: &FloatMask) -> Image {
    alpha.to_u8(255.0, 0.0)
}
