//! BiRefNet 显著性抠图 / 软边缘分割引擎。
//!
//! 对齐 birefnet_lab 中 ONNX 推理约定（onnx-community 标准算子导出）：
//! - 输入：`input_image` [1,3,1024,1024]，BGR→RGB，/255，ImageNet mean/std
//! - 输出：`output_image` [1,1,1024,1024] logits；若 max>1.5 则 sigmoid
//! - 后处理：alpha resize 回原图尺寸
//!
//! # 模型准备
//!
//! 使用社区已展开 DeformConv 的 ONNX，勿自行 torch.onnx.export 原始 BiRefNet
//! （会因 torchvision::deform_conv2d 失败）。
//! 推荐：onnx-community/BiRefNet-ONNX → `birefnet_onnx_community.onnx`

use crate::core::base::BaseOnnxEngine;
use crate::core::device_type::DeviceType;
use crate::error::{Result, VisionError};
use crate::imaging::{FloatMask, Image};
use crate::model::{MattingResult, Segmentation};

/// BiRefNet 显著性抠图 / 软边缘分割引擎。
///
/// # 示例
///
/// ```ignore
/// let eng = BiRefNetEngine::new("birefnet_onnx_community.onnx", DeviceType::Cuda)?;
/// let r = eng.predict(&image)?;
/// let alpha = &r.alpha;                     // f32 [0,1]，原图尺寸
/// let mask8 = r.binary_mask(0.5);           // 0/255，可喂 vision-measure
/// ```
pub struct BiRefNetEngine {
    base: BaseOnnxEngine,
    /// 二值化默认阈值（alpha 空间 0~1）
    mask_threshold: f32,
    model_size: i32,
}

impl BiRefNetEngine {
    /// 默认模型输入边长。
    pub const DEFAULT_INPUT_SIZE: i32 = 1024;

    const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

    /// 创建引擎（默认输入 1024）。
    pub fn new(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        Self::with_input_size(model_path, device_type, Self::DEFAULT_INPUT_SIZE)
    }

    /// 指定输入边长创建（<=0 时回退默认 1024）。
    pub fn with_input_size(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        input_size: i32,
    ) -> Result<Self> {
        Self::build(model_path, device_type, input_size, None)
    }

    fn build(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        input_size: i32,
        runtime_config: Option<crate::core::runtime_config::OnnxRuntimeConfig>,
    ) -> Result<Self> {
        let mut base = match runtime_config {
            Some(cfg) => BaseOnnxEngine::with_config(model_path, device_type, input_size, input_size, cfg)?,
            None => BaseOnnxEngine::with_input_size(model_path, device_type, input_size, input_size)?,
        };
        let model_size = if input_size > 0 { input_size } else { Self::DEFAULT_INPUT_SIZE };
        // 强制 ImageNet 归一化（与 lab preprocessor 一致）
        base.set_normalization(Self::IMAGENET_MEAN, Self::IMAGENET_STD);
        base.set_normalize(true);
        base.input_height = model_size;
        base.input_width = model_size;
        tracing::info!(
            "BiRefNetEngine ready: input={}x{}, in='{}', device={}",
            base.input_width,
            base.input_height,
            base.input_name(),
            device_type.name()
        );
        Ok(BiRefNetEngine {
            base,
            mask_threshold: 0.5,
            model_size,
        })
    }

    /// 指定运行参数创建（线程数 / GPU 设备 id）。
    pub fn with_config(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        input_size: i32,
        runtime_config: crate::core::runtime_config::OnnxRuntimeConfig,
    ) -> Result<Self> {
        Self::build(model_path, device_type, input_size, Some(runtime_config))
    }

    /// 二值化阈值（alpha 空间 0~1）。
    pub fn mask_threshold(&self) -> f32 {
        self.mask_threshold
    }

    /// 设置二值化阈值。
    pub fn set_mask_threshold(&mut self, threshold: f32) {
        self.mask_threshold = threshold;
    }

    /// 推理，返回原图尺寸 soft alpha。
    pub fn predict_impl(&self, image: &Image) -> Result<MattingResult> {
        if image.is_empty() {
            return Err(VisionError::invalid_argument("image is null or empty"));
        }
        let ow = image.width() as i32;
        let oh = image.height() as i32;
        let t0 = std::time::Instant::now();

        let input_data = self.base.preprocess(image)?;
        let input_tensor = self.base.create_input_tensor(input_data)?;
        let output = self.base.run_inference(input_tensor)?;
        let raw = output.as_f32()?;

        let alpha_model = Self::to_alpha01(raw);
        let alpha_small = Self::float_hw_to_mask(&alpha_model, self.model_size as usize, self.model_size as usize)?;
        // alpha resize 回原图尺寸（双线性，对应 INTER_LINEAR）
        let alpha_full = alpha_small.resize(ow as usize, oh as usize);

        let elapsed = t0.elapsed().as_millis() as u64;
        tracing::debug!("BiRefNet predict {}x{} -> alpha in {} ms", ow, oh, elapsed);

        MattingResult::try_new(alpha_full, ow, oh, self.model_size, elapsed)
    }

    /// 一步得到 0/255 二值 mask（原图尺寸）。
    pub fn predict_binary_mask(&self, image: &Image) -> Result<Image> {
        let r = self.predict_impl(image)?;
        Ok(r.binary_mask(self.mask_threshold))
    }

    /// 转为现有 [`Segmentation`] 列表（单前景实例），便于与检测管线统一。
    /// <p>返回的 Segmentation.mask 为 float [0,1]。
    pub fn predict_as_segmentations(&self, image: &Image) -> Result<Vec<Segmentation>> {
        let r = self.predict_impl(image)?;
        Ok(Self::to_segmentations(&r, self.mask_threshold))
    }

    /// MattingResult → Segmentation 列表（默认 1 个前景）。
    pub fn to_segmentations(result: &MattingResult, binary_threshold: f32) -> Vec<Segmentation> {
        if !result.has_alpha() {
            return Vec::new();
        }
        let alpha = result.alpha.clone(); // Segmentation 持有独立 mask
        let br = result.foreground_rect(binary_threshold);
        let conf = mean_alpha(&alpha) as f64;

        let x1 = br.x as f64;
        let y1 = br.y as f64;
        let x2 = (br.x + br.width) as f64;
        let y2 = (br.y + br.height) as f64;
        // 原版还计算中心点 cx/cy；Rust Segmentation 模型无对应字段，省略

        let mut seg = Segmentation::new("foreground", 0, x1, y1, x2, y2, conf, Some(alpha));
        seg.mask_threshold = binary_threshold;
        vec![seg]
    }

    // -------------------- 内部 --------------------

    /// logits 或已 sigmoid 的平面 → [0,1] alpha。
    /// 与 lab：max>1.5 则 sigmoid。
    pub fn to_alpha01(raw: &[f32]) -> Vec<f32> {
        if raw.is_empty() {
            return Vec::new();
        }
        let mut max = raw[0];
        let mut min = raw[0];
        for &v in raw {
            if v > max {
                max = v;
            }
            if v < min {
                min = v;
            }
        }
        let mut out = Vec::with_capacity(raw.len());
        if max > 1.5 || min < -0.01 {
            for &v in raw {
                out.push(sigmoid(v));
            }
        } else {
            for &v in raw {
                out.push(v.clamp(0.0, 1.0));
            }
        }
        out
    }

    /// 平面 float HWC 数据 → [`FloatMask`]（长度不足报错，多余截断，对齐上游 `floatHwToMat`）。
    fn float_hw_to_mask(data: &[f32], h: usize, w: usize) -> Result<FloatMask> {
        if data.len() < h * w {
            return Err(VisionError::invalid_argument(format!(
                "alpha length {} < {}",
                data.len(),
                h * w
            )));
        }
        FloatMask::from_raw(h, w, data[..h * w].to_vec())
    }
}

/// 防溢出 sigmoid。
pub fn sigmoid(x: f32) -> f32 {
    if x >= 20.0 {
        return 1.0;
    }
    if x <= -20.0 {
        return 0.0;
    }
    1.0 / (1.0 + (-x as f64).exp()) as f32
}

/// alpha 均值（对应 `meanAlpha`）。
fn mean_alpha(alpha: &FloatMask) -> f32 {
    if alpha.is_empty() {
        return 0.0;
    }
    let n = alpha.len() as f64;
    (alpha.sum() / n) as f32
}

crate::impl_engine_forward!(BiRefNetEngine, base, MattingResult,
    fn predict(&self, image: &Image) -> Result<MattingResult> { self.predict_impl(image) }
);
