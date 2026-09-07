//! fast-neural-style 风格迁移引擎（ONNX Model Zoo：candy / mosaic / udnie / pointilism / rain-princess）。
//!
//! # 模型
//!
//! ONNX Model Zoo `vision/style_transfer/fast_neural_style`（candy-9.onnx / mosaic-9.onnx 等，
//! 基于 *Perceptual Losses for Real-Time Style Transfer* + Instance Normalization）。
//! 输入 `[1,3,H,W]` float32 RGB；动态尺寸模型任意 H/W（输出与输入同尺寸），
//! 静态尺寸模型（探针实测 shape 含正数维度）则先 resize 到模型尺寸，输出后再还原原图尺寸。
//!
//! # 预处理约定
//!
//! ONNX Model Zoo 的官方定义（style-transfer-ort.ipynb、OpenCV `samples/dnn/fast_neural_style.py`）
//! 为 **RGB 原始 [0,255] 浮点，不做归一化**，经实拍图 A/B 验证：该约定下 candy-9 输出为正确的
//! candy 风格化；而 ImageNet mean/std 约定对同一文件会产生近乎全白的过曝输出，故默认采用
//! [`StylePreprocess::Raw255`]。两种流派因 InstanceNorm 的近似尺度/平移不变性对部分导出均可出图：
//! - [`StylePreprocess::Raw255`]（默认）：RGB 直接取 [0,255] 浮点 —— Model Zoo 官方约定，
//!   适用于 model zoo 的 candy-9 / mosaic-9 等文件；
//! - [`StylePreprocess::ImageNetNorm`]：x/255 → (x-mean)/std，mean=[0.485,0.456,0.406]、
//!   std=[0.229,0.224,0.225] —— Windows ML StyleTransfer 等实现的 fast-neural-style 标准预处理，
//!   仅适用于按此约定导出的模型文件。
//!
//! 输出按所选模式的逆过程反归一化，clamp 到 [0,255]，RGB→BGR，与输入图像同尺寸返回。

use ort::value::Tensor;

use crate::core::base::{BaseOnnxEngine, TensorOutput};
use crate::core::device_type::DeviceType;
use crate::error::{Result, VisionError};
use crate::imaging::{cvt_color, resize, ColorConversion, Image, Interpolation};

/// 预处理/反归一化模式（输入输出的数值约定必须配对使用）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StylePreprocess {
    /// 原始 [0,255] 浮点，不做归一化（默认；ONNX Model Zoo 官方 notebook / OpenCV 样例约定，
    /// 实测对 model zoo 的 candy-9 / mosaic-9 出图正确）。
    #[default]
    Raw255,
    /// x/255 → (x-mean)/std，ImageNet mean/std（fast-neural-style 的另一主流约定，
    /// 仅适用于按此约定导出的模型文件；对 model zoo 文件会输出近乎全白的过曝图）。
    ImageNetNorm,
}

/// fast-neural-style 风格迁移引擎。
///
/// # 示例
///
/// ```ignore
/// let engine = StyleTransferEngine::new("models/candy-9.onnx", DeviceType::Cpu)?;
/// let styled: Image = engine.predict(&bgr_image)?;
/// // 若模型按 x/255 + ImageNet mean/std 约定导出：
/// // StyleTransferEngine::with_preprocess(path, DeviceType::Cpu, StylePreprocess::ImageNetNorm)?
/// ```
pub struct StyleTransferEngine {
    base: BaseOnnxEngine,
    /// 预处理/反归一化模式
    preprocess: StylePreprocess,
    /// 模型 H/W 是否为动态维度（true：任意尺寸输入；false：固定尺寸，需先 resize）
    dynamic_hw: bool,
    /// 静态模型的输入 H/W（dynamic_hw = false 时有效）
    model_h: i32,
    model_w: i32,
}

impl StyleTransferEngine {
    /// 创建风格迁移引擎（自动探测模型输入 H/W 是否动态）。
    pub fn new(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        Self::with_preprocess(model_path, device_type, StylePreprocess::default())
    }

    /// 指定预处理约定创建引擎。
    pub fn with_preprocess(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        preprocess: StylePreprocess,
    ) -> Result<Self> {
        let mut base = BaseOnnxEngine::new(model_path, device_type)?;
        // ImageNet mean/std；Raw255 模式下反归一化不再使用 mean/std
        base.set_normalization([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
        base.set_normalize(preprocess == StylePreprocess::ImageNetNorm);

        // 从会话元信息实测 H/W 是否动态（基类对动态维度回退 640，这里不能依赖）。
        // 字段为 pub(crate)，库内可直接读取。
        let (dynamic_hw, model_h, model_w) = {
            let session = base.session.lock().unwrap();
            let mut dynamic = true;
            let (mut h, mut w) = (0i32, 0i32);
            if let Some(input) = session.inputs().first() {
                if let ort::value::ValueType::Tensor { shape, .. } = input.dtype() {
                    let dims: Vec<i64> = shape.iter().copied().collect();
                    if dims.len() >= 4 && dims[2] > 0 && dims[3] > 0 {
                        dynamic = false;
                        h = dims[2] as i32;
                        w = dims[3] as i32;
                    }
                }
            }
            (dynamic, h, w)
        };

        tracing::info!(
            "StyleTransferEngine initialized: preprocess={:?}, dynamic_hw={}, model_input={}x{}",
            preprocess,
            dynamic_hw,
            model_w,
            model_h
        );

        Ok(StyleTransferEngine {
            base,
            preprocess,
            dynamic_hw,
            model_h,
            model_w,
        })
    }

    /// 预处理/反归一化模式。
    pub fn preprocess_mode(&self) -> StylePreprocess {
        self.preprocess
    }

    /// 模型输入 H/W 是否为动态维度。
    pub fn is_dynamic_hw(&self) -> bool {
        self.dynamic_hw
    }

    /// 静态模型固定输入尺寸 (width, height)；动态模型返回 None。
    pub fn model_input_size(&self) -> Option<(i32, i32)> {
        if self.dynamic_hw {
            None
        } else {
            Some((self.model_w, self.model_h))
        }
    }

    /// 风格迁移：输入 BGR 图像，返回同尺寸的风格化 BGR 图像。
    pub fn predict_impl(&self, image: &Image) -> Result<Image> {
        if image.channels() != 3 {
            return Err(VisionError::invalid_argument(format!(
                "风格迁移需要 3 通道 BGR 输入，收到 {} 通道",
                image.channels()
            )));
        }
        if image.is_empty() {
            return Err(VisionError::invalid_argument("输入图像为空"));
        }

        // 1. 确定送入网络的尺寸：动态模型用原图尺寸，静态模型 resize 到模型尺寸
        let (in_w, in_h) = if self.dynamic_hw {
            (image.width(), image.height())
        } else {
            (self.model_w as usize, self.model_h as usize)
        };
        let resized = if in_w == image.width() && in_h == image.height() {
            image.clone()
        } else {
            resize(image, in_w, in_h, Interpolation::Linear)?
        };

        // 2. BGR → RGB
        let rgb = cvt_color(&resized, ColorConversion::Bgr2Rgb)?;

        // 3. 归一化 + HWC → CHW float32
        let chw = self.preprocess_chw(rgb.data(), in_w * in_h)?;

        // 4. [1,3,H,W] 张量推理
        let shape = vec![1i64, 3, in_h as i64, in_w as i64];
        let input_tensor = Tensor::from_array((shape, chw))?;
        let output = self.base.run_inference(input_tensor)?;

        // 5. 反归一化 + CHW → HWC + RGB → BGR
        let net_out = self.output_to_image(&output)?;

        // 6. 输出与原图尺寸对齐（动态模型输出可能相差数像素；静态模型还原原图尺寸）
        if net_out.width() == image.width() && net_out.height() == image.height() {
            Ok(net_out)
        } else {
            resize(&net_out, image.width(), image.height(), Interpolation::Linear)
        }
    }

    /// 归一化 + HWC(RGB u8) → CHW(f32)。
    fn preprocess_chw(&self, rgb: &[u8], area: usize) -> Result<Vec<f32>> {
        let mut chw = vec![0f32; 3 * area];
        match self.preprocess {
            // x/255 → (x-mean)/std
            StylePreprocess::ImageNetNorm => {
                let mean = self.base.mean;
                let std = self.base.std;
                for i in 0..area {
                    chw[i] = (rgb[i * 3] as f32 / 255.0 - mean[0]) / std[0];
                    chw[i + area] = (rgb[i * 3 + 1] as f32 / 255.0 - mean[1]) / std[1];
                    chw[i + 2 * area] = (rgb[i * 3 + 2] as f32 / 255.0 - mean[2]) / std[2];
                }
            }
            // 原始 [0,255] 浮点，不归一化
            StylePreprocess::Raw255 => {
                for i in 0..area {
                    chw[i] = rgb[i * 3] as f32;
                    chw[i + area] = rgb[i * 3 + 1] as f32;
                    chw[i + 2 * area] = rgb[i * 3 + 2] as f32;
                }
            }
        }
        Ok(chw)
    }

    /// 输出 [1,3,H,W] f32 → 反归一化 → clamp [0,255] → HWC BGR u8。
    fn output_to_image(&self, output: &TensorOutput) -> Result<Image> {
        if output.shape.len() < 4 {
            return Err(VisionError::inference(format!(
                "风格迁移输出维度异常: {:?}",
                output.shape
            )));
        }
        let (oh, ow) = (output.shape[2].max(1) as usize, output.shape[3].max(1) as usize);
        let area = oh * ow;
        let data = output.as_f32()?;
        if data.len() < 3 * area {
            return Err(VisionError::inference(format!(
                "风格迁移输出数据不足: {} < 3x{}x{}",
                data.len(),
                oh,
                ow
            )));
        }

        // 模型输出为 RGB 通道序（与输入约定一致），转成 BGR 图像
        let mut out_bytes = vec![0u8; area * 3];
        match self.preprocess {
            // x*std + mean → [0,1] → *255
            StylePreprocess::ImageNetNorm => {
                let mean = self.base.mean;
                let std = self.base.std;
                for i in 0..area {
                    out_bytes[i * 3] = clamp_u8((data[2 * area + i] * std[2] + mean[2]) * 255.0);
                    out_bytes[i * 3 + 1] = clamp_u8((data[area + i] * std[1] + mean[1]) * 255.0);
                    out_bytes[i * 3 + 2] = clamp_u8((data[i] * std[0] + mean[0]) * 255.0);
                }
            }
            // 已是 [0,255]，直接 clamp
            StylePreprocess::Raw255 => {
                for i in 0..area {
                    out_bytes[i * 3] = clamp_u8(data[2 * area + i]);
                    out_bytes[i * 3 + 1] = clamp_u8(data[area + i]);
                    out_bytes[i * 3 + 2] = clamp_u8(data[i]);
                }
            }
        }
        Image::from_raw(ow, oh, 3, out_bytes)
    }
}

/// 任意 f32 → u8（四舍五入并 clamp 0~255）。
fn clamp_u8(v: f32) -> u8 {
    let i = (v + 0.5) as i32;
    i.clamp(0, 255) as u8
}

crate::impl_engine_forward!(StyleTransferEngine, base, Image,
    fn predict(&self, image: &Image) -> Result<Image> { self.predict_impl(image) }
);
