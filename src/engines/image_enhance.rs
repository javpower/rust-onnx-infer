//! 图像增强引擎。
//!
//! 去噪 / 低光增强 / 去雾，输入输出同尺寸 BGR 图像，可直接作为检测 / 分割引擎的前置预处理。
//!
//! # 模型
//!
//! 均由 `scripts/export_image_enhance.py` 从官方权重导出，
//! 统一 I/O 约定 `[1,3,H,W] float32 RGB [0,1]`（H/W 动态，输出与输入同尺寸）：
//! - DENOISE — DnCNN 彩色盲去噪（KAIR dncnn_color_blind，高斯 σ0~55，残差学习 x−noise）。
//!   针对高斯/传感器噪声训练，对 JPEG 压缩块效应效果有限。
//! - LOW_LIGHT — Zero-DCE（零参考低光增强，迭代曲线增强 ×8，无需成对训练数据，
//!   输出亮度提升明显）。
//! - DEHAZE — DehazeFormer-S（outdoor 版，U 形窗口注意力，输出 = K·x − B + x 的物理式复原）。
//!
//! # 分块推理
//!
//! 与 Real-ESRGAN 相同的重叠分块策略（默认 tile 512 + pad 32，replicate 边界填充），
//! 大图内存可控、拼接缝由重叠区吸收；图像小于 tile 时退化为单次整图推理。
//! 可通过 [`ImageEnhanceEngine::set_tile_size`] 调节（0 = 禁用）。
//!
//! 多步增强组合请使用 [`crate::engines::image_enhance_pipeline::ImageEnhancePipeline`]。

use ort::value::Tensor;

use crate::core::base::BaseOnnxEngine;
use crate::core::device_type::DeviceType;
use crate::error::{Result, VisionError};
use crate::imaging::{copy_make_border, cvt_color, BorderType, ColorConversion, Image};

/// 增强类型（决定配套模型与静态工厂）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageEnhanceType {
    /// DnCNN 彩色盲去噪（dncnn_color_blind.onnx）
    Denoise,
    /// Zero-DCE 低光增强（zerodce.onnx）
    LowLight,
    /// DehazeFormer-S 去雾（dehazeformer_s_outdoor.onnx）
    Dehaze,
}

impl ImageEnhanceType {
    /// 枚举名（DENOISE / LOW_LIGHT / DEHAZE）。
    pub fn as_str(&self) -> &'static str {
        match self {
            ImageEnhanceType::Denoise => "DENOISE",
            ImageEnhanceType::LowLight => "LOW_LIGHT",
            ImageEnhanceType::Dehaze => "DEHAZE",
        }
    }
}

impl std::fmt::Display for ImageEnhanceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// 图像增强引擎（去噪 / 低光增强 / 去雾）。
///
/// # 示例
///
/// ```ignore
/// let denoise = ImageEnhanceEngine::for_denoise("models/dncnn_color_blind.onnx", DeviceType::Cpu)?;
/// let cleaned = denoise.predict(&noisy_bgr)?;
/// ```
pub struct ImageEnhanceEngine {
    base: BaseOnnxEngine,
    /// 增强类型
    enhance_type: ImageEnhanceType,
    /// 分块尺寸（像素，0 = 禁用分块整图推理）；默认 512
    tile_size: i32,
    /// 分块重叠像素（吸收拼接缝）；默认 32（覆盖 DnCNN 41px 感受野的大半）
    tile_pad: i32,
}

impl ImageEnhanceEngine {
    pub fn new(
        model_path: impl AsRef<std::path::Path>,
        enhance_type: ImageEnhanceType,
        device_type: DeviceType,
    ) -> Result<Self> {
        let mut base = BaseOnnxEngine::new(model_path, device_type)?;
        base.set_normalization([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        Ok(ImageEnhanceEngine {
            base,
            enhance_type,
            tile_size: 512,
            tile_pad: 32,
        })
    }

    /// 去噪引擎（DnCNN 彩色盲，dncnn_color_blind.onnx）。
    pub fn for_denoise(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        Self::new(model_path, ImageEnhanceType::Denoise, device_type)
    }

    /// 低光增强引擎（Zero-DCE，zerodce.onnx）。
    pub fn for_low_light(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        Self::new(model_path, ImageEnhanceType::LowLight, device_type)
    }

    /// 去雾引擎（DehazeFormer-S outdoor，dehazeformer_s_outdoor.onnx）。
    pub fn for_dehaze(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        Self::new(model_path, ImageEnhanceType::Dehaze, device_type)
    }

    /// 增强类型。
    pub fn enhance_type(&self) -> ImageEnhanceType {
        self.enhance_type
    }

    /// 分块尺寸（像素，0 = 禁用分块整图推理）。
    pub fn tile_size(&self) -> i32 {
        self.tile_size
    }

    pub fn set_tile_size(&mut self, tile_size: i32) {
        self.tile_size = tile_size;
    }

    /// 分块重叠像素（吸收拼接缝）。
    pub fn tile_pad(&self) -> i32 {
        self.tile_pad
    }

    pub fn set_tile_pad(&mut self, tile_pad: i32) {
        self.tile_pad = tile_pad;
    }

    /// 增强：输入 BGR 图像，返回同尺寸 BGR 图像。
    pub fn predict_impl(&self, image: &Image) -> Result<Image> {
        if image.channels() != 3 {
            return Err(VisionError::invalid_argument(format!(
                "{} 增强需要 3 通道 BGR 输入，收到 {} 通道",
                self.enhance_type,
                image.channels()
            )));
        }
        let w = image.width() as i32;
        let h = image.height() as i32;
        if w == 0 || h == 0 {
            return Err(VisionError::invalid_argument("输入图像为空"));
        }

        if self.tile_size <= 0 || (w <= self.tile_size && h <= self.tile_size) {
            let result = self.process_tile(image)?;
            if result.width() != image.width() || result.height() != image.height() {
                return Err(VisionError::inference(format!(
                    "增强输出尺寸异常: {}x{}，期望 {}x{}",
                    result.width(),
                    result.height(),
                    w,
                    h
                )));
            }
            return Ok(result);
        }
        self.predict_tiled(image)
    }

    /// 重叠分块推理：replicate 填充 → 逐块推理 → 拼接 → 裁回原尺寸。
    fn predict_tiled(&self, image: &Image) -> Result<Image> {
        let w = image.width() as i32;
        let h = image.height() as i32;
        let tile_size = self.tile_size;
        let pad = self.tile_pad.max(0);

        let pad_u = pad as usize;
        let padded = copy_make_border(image, pad_u, pad_u, pad_u, pad_u, BorderType::Replicate)?;
        let pw = padded.width() as i32;
        let ph = padded.height() as i32;
        let tiles_x = (pw + tile_size - 1) / tile_size;
        let tiles_y = (ph + tile_size - 1) / tile_size;

        let mut out_padded = Image::new(pw as usize, ph as usize, 3);
        let start = std::time::Instant::now();
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                let sx = tx * tile_size;
                let sy = ty * tile_size;
                let tw = tile_size.min(pw - sx);
                let th = tile_size.min(ph - sy);
                let tile = padded.crop(sx as usize, sy as usize, tw as usize, th as usize)?;
                let tile_out = self.process_tile(&tile)?;
                out_padded.paste(sx as usize, sy as usize, &tile_out);
            }
        }

        let result = out_padded.crop(pad_u, pad_u, w as usize, h as usize)?;
        tracing::debug!(
            "{} 分块完成 {}x{}（{} 块，{} ms）",
            self.enhance_type,
            w,
            h,
            tiles_x * tiles_y,
            start.elapsed().as_millis()
        );
        Ok(result)
    }

    /// 单块（或整图）推理：BGR → RGB/255 CHW → ONNX → RGB [0,1] → HWC BGR u8。
    fn process_tile(&self, bgr: &Image) -> Result<Image> {
        let h = bgr.height();
        let w = bgr.width();

        // 1. BGR → RGB
        let rgb = cvt_color(bgr, ColorConversion::Bgr2Rgb)?;
        let pixels = rgb.data();

        // 2. CHW float32 /255
        let area = h * w;
        let mut chw = vec![0f32; 3 * area];
        for i in 0..area {
            chw[i] = pixels[i * 3] as f32 / 255.0;
            chw[i + area] = pixels[i * 3 + 1] as f32 / 255.0;
            chw[i + 2 * area] = pixels[i * 3 + 2] as f32 / 255.0;
        }

        // 3. 动态尺寸张量推理（基类 create_input_tensor 是固定尺寸，这里直接构造）
        let shape = vec![1i64, 3, h as i64, w as i64];
        let input_tensor = Tensor::from_array((shape, chw))?;
        let output = self.base.run_inference(input_tensor)?;
        if output.shape.len() < 4 {
            return Err(VisionError::inference(format!(
                "图像增强输出维度异常: {:?}",
                output.shape
            )));
        }
        let oh = output.shape[2] as usize;
        let ow = output.shape[3] as usize;
        let out_float = output.as_f32()?;

        // 4. [0,1] → u8，CHW → HWC（BGR）
        let mut out_bytes = vec![0u8; oh * ow * 3];
        for y in 0..oh {
            for x in 0..ow {
                let src = y * ow + x;
                out_bytes[src * 3] = clamp255(out_float[2 * oh * ow + src]);
                out_bytes[src * 3 + 1] = clamp255(out_float[oh * ow + src]);
                out_bytes[src * 3 + 2] = clamp255(out_float[src]);
            }
        }
        Image::from_raw(ow, oh, 3, out_bytes)
    }
}

/// 概率 [0,1] → u8（四舍五入并 clamp 0~255）。
fn clamp255(v: f32) -> u8 {
    let i = (v * 255.0 + 0.5) as i32;
    i.clamp(0, 255) as u8
}

crate::impl_engine_forward!(ImageEnhanceEngine, base, Image,
    fn predict(&self, image: &Image) -> Result<Image> { self.predict_impl(image) }
);
