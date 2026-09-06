//! Real-ESRGAN 图像超分引擎。
//!
//! # 模型
//!
//! 由 `scripts/export_realesrgan.py` 从官方权重导出（RealESRGAN_x2plus /
//! RealESRGAN_x4plus，RRDBNet 64feat×23block）。
//! 输入 `images [1,3,H,W] float32`（RGB、[0,1]，H/W 动态）、
//! 输出 `output0 [1,3,H*scale,W*scale]`（RGB、[0,1]）。
//!
//! # 分块超分
//!
//! RRDBNet 为全卷积、任意尺寸可推理，但大图整图推理显存/内存占用随面积平方级增长，
//! 且速度不可控。引擎默认启用<b>重叠分块</b>（tile 384 + 重叠 16，replicate 边界填充），
//! 逐块超分后拼接回原位，拼接缝由重叠区吸收；图像小于 tile 时自动退化为单次整图推理。
//! 可通过 [`RealEsrganEngine::set_tile_size`] 调节（0 = 禁用分块）。
//!
//! # 注意
//!
//! RRDBNet 计算量大（23 个 RRDB 残差块），CPU 上 384 分块单块约数秒，
//! 大图建议 GPU 或增大 tile。

use ort::value::Tensor;

use crate::core::base::BaseOnnxEngine;
use crate::core::device_type::DeviceType;
use crate::error::{Result, VisionError};
use crate::imaging::{copy_make_border, cvt_color, BorderType, ColorConversion, Image};

/// Real-ESRGAN 图像超分引擎（RRDBNet，支持 2x / 4x）。
///
/// # 示例
///
/// ```ignore
/// let engine = RealEsrganEngine::for_x4("models/realesrgan_x4.onnx", DeviceType::Cpu)?;
/// let upscaled = engine.predict(&small_image)?;   // 尺寸 = 输入 × 4
/// ```
pub struct RealEsrganEngine {
    base: BaseOnnxEngine,
    /// 放大倍数（2 或 4）
    scale: i32,
    /// 分块尺寸（像素，0 = 禁用分块整图推理）；默认 384
    tile_size: i32,
    /// 分块重叠像素（吸收拼接缝）
    tile_pad: i32,
}

impl RealEsrganEngine {
    pub fn new(model_path: impl AsRef<std::path::Path>, scale: i32, device_type: DeviceType) -> Result<Self> {
        let mut base = BaseOnnxEngine::new(model_path, device_type)?;
        if scale != 2 && scale != 4 {
            return Err(VisionError::invalid_argument(format!(
                "Real-ESRGAN scale 仅支持 2 或 4，收到: {scale}"
            )));
        }
        base.set_normalization([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        Ok(RealEsrganEngine {
            base,
            scale,
            tile_size: 384,
            tile_pad: 16,
        })
    }

    /// 2x 超分引擎（RealESRGAN_x2plus 导出模型）。
    pub fn for_x2(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        Self::new(model_path, 2, device_type)
    }

    /// 4x 超分引擎（RealESRGAN_x4plus 导出模型）。
    pub fn for_x4(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        Self::new(model_path, 4, device_type)
    }

    /// 放大倍数（2 或 4）。
    pub fn scale(&self) -> i32 {
        self.scale
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

    /// 超分：输入 BGR 图像，返回放大 [`RealEsrganEngine::scale`] 倍的 BGR 图像。
    pub fn predict_impl(&self, image: &Image) -> Result<Image> {
        if image.channels() != 3 {
            return Err(VisionError::invalid_argument(format!(
                "Real-ESRGAN 需要 3 通道 BGR 输入，收到 {} 通道",
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
            if result.width() as i32 != w * self.scale || result.height() as i32 != h * self.scale {
                return Err(VisionError::inference(format!(
                    "超分输出尺寸异常: {}x{}，期望 {}x{}",
                    result.width(),
                    result.height(),
                    w * self.scale,
                    h * self.scale
                )));
            }
            return Ok(result);
        }
        self.predict_tiled(image)
    }

    /// 重叠分块超分：replicate 填充 → 逐块推理 → 拼接 → 裁回原尺寸 ×scale。
    fn predict_tiled(&self, image: &Image) -> Result<Image> {
        let w = image.width() as i32;
        let h = image.height() as i32;
        let tile_size = self.tile_size;
        let scale = self.scale;
        let pad = self.tile_pad.max(0);

        // 1. 边界 replicate 填充，块与块之间因此天然有 2*pad 重叠
        let pad_u = pad as usize;
        let padded = copy_make_border(image, pad_u, pad_u, pad_u, pad_u, BorderType::Replicate)?;
        let pw = padded.width() as i32;
        let ph = padded.height() as i32;
        let tiles_x = (pw + tile_size - 1) / tile_size;
        let tiles_y = (ph + tile_size - 1) / tile_size;

        let mut out_padded = Image::new((pw * scale) as usize, (ph * scale) as usize, 3);
        let start = std::time::Instant::now();
        let mut done = 0;
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                let sx = tx * tile_size;
                let sy = ty * tile_size;
                let tw = tile_size.min(pw - sx);
                let th = tile_size.min(ph - sy);
                let tile = padded.crop(sx as usize, sy as usize, tw as usize, th as usize)?;
                let tile_up = self.process_tile(&tile)?;
                out_padded.paste((sx * scale) as usize, (sy * scale) as usize, &tile_up);
                done += 1;
                if done % 10 == 0 {
                    tracing::debug!("Real-ESRGAN 分块进度 {}/{}", done, tiles_x * tiles_y);
                }
            }
        }

        // 2. 裁掉填充区，回到原图尺寸 ×scale
        let result = out_padded.crop(
            (pad * scale) as usize,
            (pad * scale) as usize,
            (w * scale) as usize,
            (h * scale) as usize,
        )?;
        tracing::debug!(
            "Real-ESRGAN 完成 {}x{} → {}x{}（{} 块，{} ms）",
            w,
            h,
            w * scale,
            h * scale,
            tiles_x * tiles_y,
            start.elapsed().as_millis()
        );
        Ok(result)
    }

    /// 单块（或整图）推理：BGR → RGB/255 CHW → ONNX → 概率图 → BGR u8 ×scale。
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
                "Real-ESRGAN 输出维度异常: {:?}",
                output.shape
            )));
        }
        let oh = output.shape[2] as usize;
        let ow = output.shape[3] as usize;
        let out_float = output.as_f32()?;

        // 4. 概率 [0,1] → u8，CHW → HWC（BGR）
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

crate::impl_engine_forward!(RealEsrganEngine, base, Image,
    fn predict(&self, image: &Image) -> Result<Image> { self.predict_impl(image) }
);
