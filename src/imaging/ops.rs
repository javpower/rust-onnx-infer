//! 基础图像操作（threshold / morphologyEx / boundingRect / copyMakeBorder / 统计），
//! 对应 `opencv_imgproc` / `opencv_core` 的常用子集，主要服务于二值掩码处理。

use crate::error::{Result, VisionError};
use crate::imaging::image::Image;

/// 阈值化类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdType {
    /// `THRESH_BINARY`: dst = src > thresh ? maxval : 0
    Binary,
    /// `THRESH_BINARY_INV`: dst = src > thresh ? 0 : maxval
    BinaryInv,
}

/// 对单通道图像做阈值化（对应 `threshold`）。支持 8U 与 32F 数据由调用方另行处理；
/// 此处仅处理 8bit `Image`。
pub fn threshold(image: &Image, thresh: f64, maxval: f64, ty: ThresholdType) -> Result<Image> {
    if image.channels() != 1 {
        return Err(VisionError::image(format!(
            "threshold expects 1 channel, got {}",
            image.channels()
        )));
    }
    let maxv = maxval.clamp(0.0, 255.0) as u8;
    let t = thresh as f32;
    let mut out = Image::new(image.width(), image.height(), 1);
    let dst = out.data_mut();
    for (i, &v) in image.data().iter().enumerate() {
        let cond = (v as f32) > t;
        dst[i] = match ty {
            ThresholdType::Binary => {
                if cond {
                    maxv
                } else {
                    0
                }
            }
            ThresholdType::BinaryInv => {
                if cond {
                    0
                } else {
                    maxv
                }
            }
        };
    }
    Ok(out)
}

/// 形态学核形状。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelShape {
    Rect,
    Ellipse,
    Cross,
}

/// 形态学操作类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphType {
    Open,
    Close,
    Erode,
    Dilate,
}

/// 生成形态学结构元素（对应 `getStructuringElement`），返回 (ksize, kernel 位图行优先)。
pub fn get_structuring_element(shape: KernelShape, ksize: usize) -> Vec<bool> {
    let k = ksize;
    let mut kernel = vec![false; k * k];
    let r = (ksize as f64 - 1.0) / 2.0;
    let c = r;
    for i in 0..k {
        for j in 0..k {
            let inside = match shape {
                KernelShape::Rect => true,
                KernelShape::Cross => i == c as usize || j == c as usize,
                KernelShape::Ellipse => {
                    // OpenCV 椭圆方程: ((i-c)/(r+0.5))^2 + ((j-c)/(c+0.5))^2 <= 1
                    let di = (i as f64 - r) / (r + 0.5);
                    let dj = (j as f64 - c) / (c + 0.5);
                    di * di + dj * dj <= 1.0
                }
            };
            kernel[i * k + j] = inside;
        }
    }
    kernel
}

/// 对单通道图像做腐蚀 / 膨胀（边界按 OpenCV 默认 BORDER_CONSTANT 0 处理）。
fn erode_dilate(image: &Image, kernel: &[bool], ksize: usize, dilate: bool) -> Image {
    let (w, h) = (image.width(), image.height());
    let r = ksize as isize / 2;
    let mut out = Image::new(w, h, 1);
    {
        let src = image.data();
        let dst = out.data_mut();
        for y in 0..h {
            for x in 0..w {
                let mut acc = if dilate { 0u8 } else { 255u8 };
                'k: for ky in 0..ksize as isize {
                    for kx in 0..ksize as isize {
                        if !kernel[(ky * ksize as isize + kx) as usize] {
                            continue;
                        }
                        let sy = y as isize + ky - r;
                        let sx = x as isize + kx - r;
                        let v = if sy < 0 || sy >= h as isize || sx < 0 || sx >= w as isize {
                            0 // BORDER_CONSTANT with 0
                        } else {
                            src[(sy as usize * w + sx as usize) as usize]
                        };
                        if dilate {
                            if v > acc {
                                acc = v;
                            }
                        } else if v < acc {
                            acc = v;
                        }
                        if (!dilate && acc == 0) || (dilate && acc == 255) {
                            break 'k;
                        }
                    }
                }
                dst[y * w + x] = acc;
            }
        }
    }
    out
}

/// 形态学开闭运算（对应 `morphologyEx`）。
pub fn morphology_ex(image: &Image, ty: MorphType, shape: KernelShape, ksize: usize) -> Result<Image> {
    if image.channels() != 1 {
        return Err(VisionError::image(format!(
            "morphology expects 1 channel, got {}",
            image.channels()
        )));
    }
    let kernel = get_structuring_element(shape, ksize);
    match ty {
        MorphType::Erode => Ok(erode_dilate(image, &kernel, ksize, false)),
        MorphType::Dilate => Ok(erode_dilate(image, &kernel, ksize, true)),
        MorphType::Open => {
            let t = erode_dilate(image, &kernel, ksize, false);
            Ok(erode_dilate(&t, &kernel, ksize, true))
        }
        MorphType::Close => {
            let t = erode_dilate(image, &kernel, ksize, true);
            Ok(erode_dilate(&t, &kernel, ksize, false))
        }
    }
}

/// 二值掩码的前景外接矩形（对应 `boundingRect`；无前景返回 0 尺寸矩形）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

impl Rect {
    pub fn new(x: i32, y: i32, width: i32, height: i32) -> Self {
        Rect {
            x,
            y,
            width,
            height,
        }
    }

    #[inline]
    pub fn left(&self) -> i32 {
        self.x
    }

    #[inline]
    pub fn top(&self) -> i32 {
        self.y
    }

    #[inline]
    pub fn right(&self) -> i32 {
        self.x + self.width
    }

    #[inline]
    pub fn bottom(&self) -> i32 {
        self.y + self.height
    }

    pub fn area(&self) -> i64 {
        self.width as i64 * self.height as i64
    }
}

pub fn bounding_rect(image: &Image) -> Rect {
    if image.channels() != 1 {
        return Rect::default();
    }
    let (w, h) = (image.width(), image.height());
    let mut min_x = w as i32;
    let mut min_y = h as i32;
    let mut max_x = -1;
    let mut max_y = -1;
    for y in 0..h {
        for x in 0..w {
            if image.data()[y * w + x] > 0 {
                if (x as i32) < min_x {
                    min_x = x as i32;
                }
                if (x as i32) > max_x {
                    max_x = x as i32;
                }
                if (y as i32) < min_y {
                    min_y = y as i32;
                }
                if (y as i32) > max_y {
                    max_y = y as i32;
                }
            }
        }
    }
    if max_x < 0 {
        Rect::default()
    } else {
        Rect::new(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
    }
}

/// 边界填充类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorderType {
    /// 常量填充（默认 0）
    Constant(u8),
    /// `BORDER_REPLICATE`：复制边缘像素
    Replicate,
}

/// 边界填充（对应 `copyMakeBorder`）。
pub fn copy_make_border(image: &Image, top: usize, bottom: usize, left: usize, right: usize, border: BorderType) -> Result<Image> {
    let (w, h, c) = (image.width(), image.height(), image.channels());
    let nw = w + left + right;
    let nh = h + top + bottom;
    let mut out = Image::new(nw, nh, c);
    match border {
        BorderType::Constant(v) => {
            out.data_mut().fill(v);
        }
        BorderType::Replicate => {
            let clamp = |v: isize, max: usize| -> usize { v.clamp(0, max as isize) as usize };
            for dy in 0..nh {
                let sy = clamp(dy as isize - top as isize, h);
                for dx in 0..nw {
                    let sx = clamp(dx as isize - left as isize, w);
                    let src = &image.data()[(sy * w + sx) * c..(sy * w + sx) * c + c];
                    out.set_pixel(dx, dy, src);
                }
            }
            return Ok(out);
        }
    }
    // Constant：先填底色再贴原图
    out.paste(left, top, image);
    Ok(out)
}

/// `convertTo` 等价：dst = saturate(src * alpha + beta)（8bit）。
pub fn convert_to(image: &Image, alpha: f64, beta: f64) -> Image {
    let mut out = image.clone();
    for v in out.data_mut() {
        *v = (*v as f64 * alpha + beta).round().clamp(0.0, 255.0) as u8;
    }
    out
}

/// 32F 单通道掩码（对应 CV_32FC1 Mat，范围 [0,1]）。
#[derive(Clone, Debug, PartialEq)]
pub struct FloatMask {
    width: usize,
    height: usize,
    data: Vec<f32>,
}

impl FloatMask {
    pub fn new(width: usize, height: usize) -> Self {
        FloatMask {
            width,
            height,
            data: vec![0.0; width * height],
        }
    }

    pub fn from_raw(width: usize, height: usize, data: Vec<f32>) -> Result<Self> {
        if data.len() != width * height {
            return Err(VisionError::image(format!(
                "mask data length {} != {}x{}",
                data.len(),
                width,
                height
            )));
        }
        Ok(FloatMask {
            width,
            height,
            data,
        })
    }

    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    #[inline]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// 元素总数。
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: f32) {
        self.data[y * self.width + x] = v;
    }

    /// 二值化（> thresh → 255，否则 0）。
    pub fn binary_mask(&self, thresh: f32) -> Image {
        let mut out = Image::new(self.width, self.height, 1);
        for (i, &v) in self.data.iter().enumerate() {
            out.data_mut()[i] = if v > thresh { 255 } else { 0 };
        }
        out
    }

    /// 前景外接矩形。
    pub fn foreground_rect(&self, thresh: f32) -> Rect {
        bounding_rect(&self.binary_mask(thresh))
    }

    /// 像素总和。
    pub fn sum(&self) -> f64 {
        self.data.iter().map(|&v| v as f64).sum()
    }

    /// 缩放到目标尺寸（双线性）。
    pub fn resize(&self, width: usize, height: usize) -> FloatMask {
        let mut out = FloatMask::new(width, height);
        let sx = self.width as f32 / width as f32;
        let sy = self.height as f32 / height as f32;
        for y in 0..height {
            let fy = (y as f32 + 0.5) * sy - 0.5;
            let y0 = fy.floor().max(0.0) as usize;
            let y1 = (y0 + 1).min(self.height - 1);
            let wy = fy - fy.floor().max(0.0);
            for x in 0..width {
                let fx = (x as f32 + 0.5) * sx - 0.5;
                let x0 = fx.floor().max(0.0) as usize;
                let x1 = (x0 + 1).min(self.width - 1);
                let wx = fx - fx.floor().max(0.0);
                let v = self.get(x0, y0) * (1.0 - wx) * (1.0 - wy)
                    + self.get(x1, y0) * wx * (1.0 - wy)
                    + self.get(x0, y1) * (1.0 - wx) * wy
                    + self.get(x1, y1) * wx * wy;
                out.set(x, y, v);
            }
        }
        out
    }

    /// 转为 8bit 灰度图（按 alpha 缩放系数乘 alpha + beta）。
    pub fn to_u8(&self, alpha: f64, beta: f64) -> Image {
        let mut out = Image::new(self.width, self.height, 1);
        for (i, &v) in self.data.iter().enumerate() {
            out.data_mut()[i] = (v as f64 * alpha + beta).round().clamp(0.0, 255.0) as u8;
        }
        out
    }
}

/// `minMaxLoc` 等价（单通道 8bit）：返回 (min, max, min_loc, max_loc)。
pub fn min_max_loc(image: &Image) -> Option<(u8, u8, (usize, usize), (usize, usize))> {
    if image.channels() != 1 || image.is_empty() {
        return None;
    }
    let (w, h) = (image.width(), image.height());
    let mut min_v = u8::MAX;
    let mut max_v = 0u8;
    let mut min_loc = (0usize, 0usize);
    let mut max_loc = (0usize, 0usize);
    for y in 0..h {
        for x in 0..w {
            let v = image.data()[y * w + x];
            if v < min_v {
                min_v = v;
                min_loc = (x, y);
            }
            if v > max_v {
                max_v = v;
                max_loc = (x, y);
            }
        }
    }
    Some((min_v, max_v, min_loc, max_loc))
}
