//! 纯 Rust 图像容器。
//!
//! 与 原版约定一致：**3/4 通道图像的通道顺序为 BGR/BGRA**，
//! 引擎内部需要 RGB 时由 `cvtColor` 系列方法显式转换。
//! 像素始终为 8bit 无符号（对应 `CV_8U` 系）。

use crate::error::{Result, VisionError};

/// 8bit 图像（1/3/4 通道；3/4 通道为 BGR/BGRA 通道序，与 OpenCV Mat 对齐）。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    width: usize,
    height: usize,
    channels: usize,
    data: Vec<u8>,
}

impl Image {
    /// 创建全 0 图像。
    pub fn new(width: usize, height: usize, channels: usize) -> Self {
        Image {
            width,
            height,
            channels,
            data: vec![0; width * height * channels],
        }
    }

    /// 用指定值填充创建图像。
    pub fn filled(width: usize, height: usize, channels: usize, value: u8) -> Self {
        Image {
            width,
            height,
            channels,
            data: vec![value; width * height * channels],
        }
    }

    /// 从原始数据创建（数据长度必须等于 w*h*channels）。
    pub fn from_raw(width: usize, height: usize, channels: usize, data: Vec<u8>) -> Result<Self> {
        if data.len() != width * height * channels {
            return Err(VisionError::image(format!(
                "data length {} != {}x{}x{}",
                data.len(),
                width,
                height,
                channels
            )));
        }
        Ok(Image {
            width,
            height,
            channels,
            data,
        })
    }

    /// 从 RGB8 数据创建（内部转为 BGR）。
    pub fn from_rgb(width: usize, height: usize, rgb: &[u8]) -> Result<Self> {
        if rgb.len() != width * height * 3 {
            return Err(VisionError::image(format!(
                "rgb length {} != {}x{}x3",
                rgb.len(),
                width,
                height
            )));
        }
        let mut data = rgb.to_vec();
        for px in data.chunks_exact_mut(3) {
            px.swap(0, 2);
        }
        Image::from_raw(width, height, 3, data)
    }

    /// 从灰度数据创建单通道图像。
    pub fn from_gray(width: usize, height: usize, gray: Vec<u8>) -> Result<Self> {
        Image::from_raw(width, height, 1, gray)
    }

    /// 从 `image` crate 的 `DynamicImage` 创建（统一转 BGR3；灰度图转 3 通道）。
    pub fn from_dynamic(img: &image::DynamicImage) -> Self {
        match img {
            image::DynamicImage::ImageRgb8(rgb) => {
                Image::from_rgb(rgb.width() as usize, rgb.height() as usize, rgb.as_raw())
                    .expect("DynamicImage rgb buffer size always valid")
            }
            _ => {
                let rgb = img.to_rgb8();
                Image::from_rgb(rgb.width() as usize, rgb.height() as usize, rgb.as_raw())
                    .expect("DynamicImage rgb buffer size always valid")
            }
        }
    }

    /// 从文件加载图像（统一转 BGR3）。
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let img = image::open(path)?;
        Ok(Image::from_dynamic(&img))
    }

    /// 保存为图像文件（按扩展名推断格式；内部 BGR→RGB）。
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        self.to_dynamic()?.save(path)?;
        Ok(())
    }

    /// 转为 `image` crate 的 `DynamicImage`（BGR→RGB）。
    pub fn to_dynamic(&self) -> Result<image::DynamicImage> {
        match self.channels {
            1 => Ok(image::DynamicImage::ImageLuma8(image::GrayImage::from_raw(
                self.width as u32,
                self.height as u32,
                self.data.clone(),
            )
            .ok_or_else(|| VisionError::image("size mismatch"))?)),
            3 => {
                let mut rgb = self.data.clone();
                for px in rgb.chunks_exact_mut(3) {
                    px.swap(0, 2);
                }
                Ok(image::DynamicImage::ImageRgb8(
                    image::RgbImage::from_raw(
                        self.width as u32,
                        self.height as u32,
                        rgb,
                    )
                    .ok_or_else(|| VisionError::image("size mismatch"))?,
                ))
            }
            4 => {
                let mut rgba = self.data.clone();
                for px in rgba.chunks_exact_mut(4) {
                    px.swap(0, 2);
                }
                Ok(image::DynamicImage::ImageRgba8(
                    image::RgbaImage::from_raw(
                        self.width as u32,
                        self.height as u32,
                        rgba,
                    )
                    .ok_or_else(|| VisionError::image("size mismatch"))?,
                ))
            }
            c => Err(VisionError::image(format!("unsupported channels: {c}"))),
        }
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
    pub fn channels(&self) -> usize {
        self.channels
    }

    #[inline]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    #[inline]
    pub fn into_data(self) -> Vec<u8> {
        self.data
    }

    /// 是否为空图。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// 行数据切片。
    #[inline]
    pub fn row(&self, y: usize) -> &[u8] {
        &self.data[y * self.width * self.channels..(y + 1) * self.width * self.channels]
    }

    /// 读取像素。
    #[inline]
    pub fn pixel(&self, x: usize, y: usize) -> Option<&[u8]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let start = (y * self.width + x) * self.channels;
        Some(&self.data[start..start + self.channels])
    }

    /// 写入像素。
    #[inline]
    pub fn set_pixel(&mut self, x: usize, y: usize, value: &[u8]) {
        if x >= self.width || y >= self.height || value.len() != self.channels {
            return;
        }
        let start = (y * self.width + x) * self.channels;
        self.data[start..start + self.channels].copy_from_slice(value);
    }

    /// 裁剪 ROI（对应 `Mat(roi)` 子矩阵；越界部分报错）。
    pub fn crop(&self, x: usize, y: usize, w: usize, h: usize) -> Result<Image> {
        if x + w > self.width || y + h > self.height {
            return Err(VisionError::image(format!(
                "crop rect ({x},{y},{w},{h}) out of bounds {}x{}",
                self.width, self.height
            )));
        }
        let mut out = Image::new(w, h, self.channels);
        let row_bytes = w * self.channels;
        for dy in 0..h {
            let src_off = ((y + dy) * self.width + x) * self.channels;
            out.data[dy * row_bytes..dy * row_bytes + row_bytes]
                .copy_from_slice(&self.data[src_off..src_off + row_bytes]);
        }
        Ok(out)
    }

    /// 将一块图像粘贴到指定位置（越界部分裁掉，不做报错）。
    pub fn paste(&mut self, x: usize, y: usize, patch: &Image) {
        if patch.channels != self.channels {
            return;
        }
        let copy_w = patch.width.min(self.width.saturating_sub(x));
        let copy_h = patch.height.min(self.height.saturating_sub(y));
        let row_bytes = copy_w * self.channels;
        for dy in 0..copy_h {
            let dst_off = ((y + dy) * self.width + x) * self.channels;
            let src_off = dy * patch.width * self.channels;
            self.data[dst_off..dst_off + row_bytes]
                .copy_from_slice(&patch.data[src_off..src_off + row_bytes]);
        }
    }
}
