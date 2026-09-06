//! 图像增强管线。
//!
//! 把多种增强步骤按固定顺序串联成一次调用，输出可直接送入检测 / 分割引擎做质检检测。
//!
//! # 固定执行顺序（已启用的步骤才会执行）
//!
//! 1. 低光增强（Zero-DCE）— 先提亮；曲线增强会放大噪声，因此去噪排在其后
//! 2. 去雾（DehazeFormer）— 雾天/油烟场景先复原对比度
//! 3. 去噪（DnCNN）— 清理增强/复原后残留的噪声
//! 4. 超分（Real-ESRGAN）— 最后放大分辨率，供小目标检测
//!
//! # 注意
//!
//! 管线的每一步输出尺寸与输入相同（超分除外，为 ×scale），
//! 因此检测框坐标系无需任何换算即可直接使用。
//! Rust 版管线<b>持有注册引擎的所有权</b>（构造时移入），管线 drop 时引擎随之释放
//! （对应 `close()`）；如需在管线外继续使用引擎，请为其单独创建实例。

use crate::engines::image_enhance::{ImageEnhanceEngine, ImageEnhanceType};
use crate::engines::real_esrgan::RealEsrganEngine;
use crate::error::{Result, VisionError};
use crate::imaging::Image;

/// 图像增强管线：低光 → 去雾 → 去噪 → 超分 的固定顺序串联。
///
/// # 示例
///
/// ```ignore
/// let pipeline = ImageEnhancePipeline::builder()
///     .low_light(ImageEnhanceEngine::for_low_light("models/zerodce.onnx", DeviceType::Cpu)?)?
///     .denoise(ImageEnhanceEngine::for_denoise("models/dncnn_color_blind.onnx", DeviceType::Cpu)?)?
///     .build()?;
/// let enhanced = pipeline.process(&frame)?;
/// ```
pub struct ImageEnhancePipeline {
    low_light: Option<ImageEnhanceEngine>,
    dehaze: Option<ImageEnhanceEngine>,
    denoise: Option<ImageEnhanceEngine>,
    super_resolution: Option<RealEsrganEngine>,
}

impl ImageEnhancePipeline {
    fn new(
        low_light: Option<ImageEnhanceEngine>,
        dehaze: Option<ImageEnhanceEngine>,
        denoise: Option<ImageEnhanceEngine>,
        super_resolution: Option<RealEsrganEngine>,
    ) -> Result<Self> {
        if low_light.is_none() && dehaze.is_none() && denoise.is_none() && super_resolution.is_none() {
            return Err(VisionError::invalid_argument("至少需要注册一个增强步骤"));
        }
        Ok(ImageEnhancePipeline {
            low_light,
            dehaze,
            denoise,
            super_resolution,
        })
    }

    /// 管线构建器（步骤注册顺序无关，执行顺序固定：低光 → 去雾 → 去噪 → 超分）。
    pub fn builder() -> Builder {
        Builder::default()
    }

    /// 串联执行所有已启用步骤，返回最终 BGR 图像。
    /// 中间结果在管线内部随覆盖自动释放（对应 owned/release 逻辑）。
    pub fn process(&self, image: &Image) -> Result<Image> {
        let mut current: Option<Image> = None;
        for engine in [&self.low_light, &self.dehaze, &self.denoise] {
            if let Some(engine) = engine {
                let input = current.as_ref().unwrap_or(image);
                current = Some(engine.predict_impl(input)?);
            }
        }
        if let Some(super_resolution) = &self.super_resolution {
            let input = current.as_ref().unwrap_or(image);
            current = Some(super_resolution.predict_impl(input)?);
        }
        // 构造时已保证至少注册一个步骤，此分支仅为完整性兜底
        Ok(current.unwrap_or_else(|| image.clone()))
    }

    /// 批量处理：逐张独立执行管线。
    pub fn process_batch(&self, images: &[Image]) -> Result<Vec<Image>> {
        let mut results = Vec::with_capacity(images.len());
        for image in images {
            results.push(self.process(image)?);
        }
        Ok(results)
    }

    /// 关闭管线并释放其中注册的所有引擎（对应 `close()`；Rust 侧由 RAII 在
    /// drop 时自动完成，此方法仅显式消费管线）。
    pub fn close(self) {}
}

/// 管线构建器（步骤注册顺序无关，执行顺序固定：低光 → 去雾 → 去噪 → 超分）。
#[derive(Default)]
pub struct Builder {
    low_light: Option<ImageEnhanceEngine>,
    dehaze: Option<ImageEnhanceEngine>,
    denoise: Option<ImageEnhanceEngine>,
    super_resolution: Option<RealEsrganEngine>,
}

impl Builder {
    /// 注册低光增强（Zero-DCE）。
    pub fn low_light(mut self, engine: ImageEnhanceEngine) -> Result<Self> {
        self.low_light = Some(require_type(engine, ImageEnhanceType::LowLight)?);
        Ok(self)
    }

    /// 注册去雾（DehazeFormer）。
    pub fn dehaze(mut self, engine: ImageEnhanceEngine) -> Result<Self> {
        self.dehaze = Some(require_type(engine, ImageEnhanceType::Dehaze)?);
        Ok(self)
    }

    /// 注册去噪（DnCNN）。
    pub fn denoise(mut self, engine: ImageEnhanceEngine) -> Result<Self> {
        self.denoise = Some(require_type(engine, ImageEnhanceType::Denoise)?);
        Ok(self)
    }

    /// 注册超分（Real-ESRGAN），作为最后一步放大分辨率。
    pub fn super_resolution(mut self, engine: RealEsrganEngine) -> Self {
        self.super_resolution = Some(engine);
        self
    }

    /// 构建管线（未注册任何步骤时报错）。
    pub fn build(self) -> Result<ImageEnhancePipeline> {
        ImageEnhancePipeline::new(self.low_light, self.dehaze, self.denoise, self.super_resolution)
    }
}

/// 校验引擎类型（对应 `Builder.requireType`）。
fn require_type(engine: ImageEnhanceEngine, expected: ImageEnhanceType) -> Result<ImageEnhanceEngine> {
    if engine.enhance_type() != expected {
        return Err(VisionError::invalid_argument(format!(
            "期望 {} 引擎，收到 {}",
            expected.as_str(),
            engine.enhance_type().as_str()
        )));
    }
    Ok(engine)
}
