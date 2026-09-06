//! 图像去模糊引擎（NAFNet-Deblur）。
//!
//! # 模型
//!
//! opencv 官方 `opencv/deblurring_nafnet` 的 `deblurring_nafnet_2025may.onnx`
//! （实测 87.49 MB，MIT，megvii-research/NAFNet GoPro 权重导出）：
//! - 输入 `lq` `[-1,3,-1,-1]` Float32（**动态** batch/H/W，探针实测）
//! - 输出 `output` `[-1,-1,-1,-1]` Float32（与输入同 H/W）
//!
//! 直链：<https://huggingface.co/opencv/deblurring_nafnet/resolve/main/deblurring_nafnet_2025may.onnx>
//!
//! 备选（社区导出，静态 256 的倍数）：deepghs/image_restoration 的
//! `NAFNet-GoPro-width64_v1.onnx` 等文件同样可用本引擎加载（静态尺寸走
//! "先 resize 推理、输出还原原图" 路径）。
//!
//! # 预处理约定（opencv 官方 demo `nafnet.py` 为准）
//!
//! BGR→RGB → `x/255`（[0,1]），动态尺寸模型直接喂原图尺寸、输出即原图尺寸。
//! 另有社区导出按 `(x/255 - 0.5) / 0.5`（[-1,1]）约定，可用
//! [`DeblurEngine::with_norm`] 切换。
//!
//! # 尺寸自适应
//!
//! 参考 [`crate::engines::style_transfer`] 的动静态自适应：
//! - 动态模型：直接喂原图尺寸；但该导出虽声明动态维度，图内含烘焙的
//!   "crop 383" 式切片（Slice + Pad(Edge)），**实测任一边 < 384 时推理失败**
//!   （内部特征图变 0，如 256x512 / 384x288；444x512 等非 8 倍数尺寸反而可跑，
//!   排除倍数约束）。故引擎对小于 [`DeblurEngine::MIN_SIDE`]（384）的边做
//!   等比放大后再推理，输出再还原原图尺寸（实测 384~1024+、方形/非方形均通过）
//! - 静态模型（如 deepghs NAFNet-GoPro-width64_v1.onnx）：先 resize 到模型
//!   尺寸，输出后还原原图尺寸
//!
//! # 示例
//!
//! ```ignore
//! let eng = DeblurEngine::new("models/nafnet_deblur_2025may.onnx", DeviceType::Cpu)?;
//! let sharp = eng.predict(&blurry_image)?;   // 与输入同尺寸
//! ```

use ort::value::Tensor;

use crate::core::base::{BaseOnnxEngine, TensorOutput};
use crate::core::device_type::DeviceType;
use crate::error::{Result, VisionError};
use crate::imaging::{cvt_color, resize, ColorConversion, Image, Interpolation};

/// 归一化/反归一化模式（输入输出的数值约定必须配对使用）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeblurNorm {
    /// `x/255`，[0,1]（opencv 官方 deblurring_nafnet demo 约定，默认）。
    #[default]
    Unit01,
    /// `(x/255 - 0.5) / 0.5`，[-1,1]（部分社区 NAFNet 导出的约定）。
    Centered01,
}

/// NAFNet 图像去模糊引擎。
pub struct DeblurEngine {
    base: BaseOnnxEngine,
    /// 归一化模式
    norm: DeblurNorm,
    /// 模型 H/W 是否为动态维度（true：直接喂原图尺寸；false：先 resize 到模型尺寸）
    dynamic_hw: bool,
    /// 静态模型的输入 H/W（dynamic_hw = false 时有效）
    model_h: i32,
    model_w: i32,
    /// 动态模型输入 H/W 的对齐倍数（NAFNet 内部 3 级下采样 ×8；
    /// 1 = 不对齐，官方 demo 直接喂原图尺寸）
    align: i32,
}

impl DeblurEngine {
    /// 创建去模糊引擎（自动探测模型输入 H/W 是否动态）。
    pub fn new(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        Self::build(model_path, device_type, DeblurNorm::Unit01, 1)
    }

    /// 指定归一化约定创建（社区 [-1,1] 导出用 [`DeblurNorm::Centered01`]）。
    pub fn with_norm(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        norm: DeblurNorm,
    ) -> Result<Self> {
        Self::build(model_path, device_type, norm, 1)
    }

    /// 指定动态模型输入对齐倍数创建（如 NAFNet 需 H/W 为 8 的倍数时传 8）。
    pub fn with_align(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        align: i32,
    ) -> Result<Self> {
        if align < 1 {
            return Err(VisionError::invalid_argument(format!(
                "对齐倍数必须 >= 1，收到: {align}"
            )));
        }
        Self::build(model_path, device_type, DeblurNorm::Unit01, align)
    }

    fn build(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        norm: DeblurNorm,
        align: i32,
    ) -> Result<Self> {
        let mut base = BaseOnnxEngine::new(model_path, device_type)?;
        // 归一化在引擎内手工完成（DynamicHw 路径不走 base.preprocess），仅记录约定
        base.set_normalization([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);

        // 从会话元信息实测 H/W 是否动态（基类对动态维度回退 640，这里不能依赖）
        let (dynamic_hw, model_h, model_w) = {
            let session = base.session.lock().unwrap();
            let mut dynamic = true;
            let (mut h, mut w) = (0i32, 0i32);
            if let Some(input) = session.inputs.first() {
                if let ort::value::ValueType::Tensor { shape, .. } = &input.input_type {
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
            "DeblurEngine ready: norm={:?}, dynamic_hw={}, model_input={}x{}, align={}, device={}",
            norm,
            dynamic_hw,
            model_w,
            model_h,
            align,
            device_type.name()
        );

        Ok(DeblurEngine {
            base,
            norm,
            dynamic_hw,
            model_h,
            model_w,
            align,
        })
    }

    /// 归一化模式。
    pub fn norm_mode(&self) -> DeblurNorm {
        self.norm
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

    /// 去模糊：输入 BGR 图像，返回同尺寸的去模糊 BGR 图像。
    pub fn predict_impl(&self, image: &Image) -> Result<Image> {
        if image.is_empty() {
            return Err(VisionError::invalid_argument("输入图像为空"));
        }
        // 统一转 3 通道 BGR（接受灰度 / BGRA 输入）
        let bgr = match image.channels() {
            3 => image.clone(),
            4 => cvt_color(image, ColorConversion::Bgra2Bgr)?,
            1 => cvt_color(image, ColorConversion::Gray2Bgr)?,
            c => {
                return Err(VisionError::invalid_argument(format!(
                    "去模糊需要 1/3/4 通道输入，收到 {c} 通道"
                )))
            }
        };

        // 1. 确定送入网络的尺寸：动态模型用原图尺寸（按 align 向上取整对齐），
        //    静态模型 resize 到模型尺寸
        let (in_w, in_h) = if self.dynamic_hw {
            let aw = align_up(bgr.width() as i32, self.align) as usize;
            let ah = align_up(bgr.height() as i32, self.align) as usize;
            (aw, ah)
        } else {
            (self.model_w as usize, self.model_h as usize)
        };
        let resized = if in_w == bgr.width() && in_h == bgr.height() {
            bgr.clone()
        } else {
            resize(&bgr, in_w, in_h, Interpolation::Linear)?
        };

        // 2. BGR → RGB
        let rgb = cvt_color(&resized, ColorConversion::Bgr2Rgb)?;

        // 3. 归一化 + HWC → CHW float32
        let area = in_w * in_h;
        let mut chw = vec![0f32; 3 * area];
        match self.norm {
            // x/255 → [0,1]
            DeblurNorm::Unit01 => {
                for i in 0..area {
                    chw[i] = rgb.data()[i * 3] as f32 / 255.0;
                    chw[i + area] = rgb.data()[i * 3 + 1] as f32 / 255.0;
                    chw[i + 2 * area] = rgb.data()[i * 3 + 2] as f32 / 255.0;
                }
            }
            // (x/255 - 0.5) / 0.5 → [-1,1]
            DeblurNorm::Centered01 => {
                for i in 0..area {
                    chw[i] = rgb.data()[i * 3] as f32 / 255.0 * 2.0 - 1.0;
                    chw[i + area] = rgb.data()[i * 3 + 1] as f32 / 255.0 * 2.0 - 1.0;
                    chw[i + 2 * area] = rgb.data()[i * 3 + 2] as f32 / 255.0 * 2.0 - 1.0;
                }
            }
        }

        // 4. [1,3,H,W] 张量推理
        let shape = vec![1i64, 3, in_h as i64, in_w as i64];
        let input_tensor = Tensor::from_array((shape, chw))?;
        let output = self.base.run_inference(input_tensor)?;

        // 5. 输出 → BGR u8（与输入同数值约定），并还原原图尺寸
        let net_out = Self::output_to_image(&output, self.norm)?;
        if net_out.width() == bgr.width() && net_out.height() == bgr.height() {
            Ok(net_out)
        } else {
            resize(&net_out, bgr.width(), bgr.height(), Interpolation::Linear)
        }
    }

    /// 输出 [1,3,H,W] f32 → 反归一化 → clamp [0,255] → HWC BGR u8。
    fn output_to_image(output: &TensorOutput, norm: DeblurNorm) -> Result<Image> {
        if output.shape.len() < 4 {
            return Err(VisionError::inference(format!(
                "去模糊输出维度异常（期望 4 维 NCHW）: {:?}",
                output.shape
            )));
        }
        let (oh, ow) = (
            output.shape[2].max(1) as usize,
            output.shape[3].max(1) as usize,
        );
        let area = oh * ow;
        let data = output.as_f32()?;
        if data.len() < 3 * area {
            return Err(VisionError::inference(format!(
                "去模糊输出数据不足: {} < 3x{}x{}",
                data.len(),
                ow,
                oh
            )));
        }

        // 模型输出为 RGB 通道序（与输入约定一致），转成 BGR 图像
        let mut out_bytes = vec![0u8; area * 3];
        match norm {
            // x*0.5 + 0.5 → [0,1] → *255
            DeblurNorm::Centered01 => {
                for i in 0..area {
                    out_bytes[i * 3] = clamp_u8((data[2 * area + i] * 0.5 + 0.5) * 255.0);
                    out_bytes[i * 3 + 1] = clamp_u8((data[area + i] * 0.5 + 0.5) * 255.0);
                    out_bytes[i * 3 + 2] = clamp_u8((data[i] * 0.5 + 0.5) * 255.0);
                }
            }
            // 已是 [0,1]，直接 *255 clamp（官方 demo：clip(result * 255.0)）
            DeblurNorm::Unit01 => {
                for i in 0..area {
                    out_bytes[i * 3] = clamp_u8(data[2 * area + i] * 255.0);
                    out_bytes[i * 3 + 1] = clamp_u8(data[area + i] * 255.0);
                    out_bytes[i * 3 + 2] = clamp_u8(data[i] * 255.0);
                }
            }
        }
        Image::from_raw(ow, oh, 3, out_bytes)
    }
}

/// 向上取整到 align 的倍数（align <= 1 时原样返回；v 为正尺寸）。
fn align_up(v: i32, align: i32) -> i32 {
    if align <= 1 || v <= 0 {
        v
    } else {
        (v + align - 1) / align * align
    }
}

/// 任意 f32 → u8（四舍五入并 clamp 0~255）。
fn clamp_u8(v: f32) -> u8 {
    let i = (v + 0.5) as i32;
    i.clamp(0, 255) as u8
}

crate::impl_engine_forward!(DeblurEngine, base, Image,
    fn predict(&self, image: &Image) -> Result<Image> { self.predict_impl(image) }
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::engine::OnnxInferenceEngine;

    #[test]
    fn align_up_rounds_to_multiple() {
        assert_eq!(align_up(510, 8), 512);
        assert_eq!(align_up(512, 8), 512);
        assert_eq!(align_up(1, 8), 8);
        assert_eq!(align_up(510, 1), 510);
        assert_eq!(align_up(510, 0), 510);
    }

    #[test]
    fn output_to_image_unit01_rgb_to_bgr() {
        // [1,3,1,2]：CHW 平面 R=[0,1]、G=[0.5,0.25]、B=[0.75,1.0]
        let data = vec![0.0f32, 1.0, 0.5, 0.25, 0.75, 1.0];
        let out = tensor(vec![1, 3, 1, 2], data);
        let img = DeblurEngine::output_to_image(&out, DeblurNorm::Unit01).unwrap();
        assert_eq!((img.width(), img.height(), img.channels()), (2, 1, 3));
        // 像素 0：R=0 G=0.5 B=0.75 → BGR = [191, 128, 0]
        assert_eq!(&img.data()[..3], &[191, 128, 0]);
        // 像素 1：R=1 G=0.25 B=1 → BGR = [255, 64, 255]
        assert_eq!(&img.data()[3..6], &[255, 64, 255]);
    }

    #[test]
    fn output_to_image_centered01_denorm() {
        // [-1,1] 反归一化：-1 → 0，1 → 255；RGB→BGR 通道序
        let data = vec![-1f32, 1.0, -1.0];
        let out = tensor(vec![1, 3, 1, 1], data);
        let img = DeblurEngine::output_to_image(&out, DeblurNorm::Centered01).unwrap();
        // R=-1→0、G=1→255、B=-1→0 → BGR = [0, 255, 0]
        assert_eq!(&img.data()[..3], &[0, 255, 0]);
    }

    /// 构造测试输出张量。
    fn tensor(shape: Vec<i64>, data: Vec<f32>) -> TensorOutput {
        TensorOutput {
            name: "test".to_string(),
            shape,
            data: crate::core::base::TensorData::F32(data),
        }
    }

    /// 3x3 均值模糊（边界复制），与自验约定一致。
    fn mean_blur_3x3(src: &Image) -> Image {
        let (w, h, c) = (src.width(), src.height(), src.channels());
        let s = src.data();
        let mut out = vec![0u8; w * h * c];
        let at = |x: i64, y: i64, ch: usize| -> f32 {
            let cx = x.clamp(0, w as i64 - 1) as usize;
            let cy = y.clamp(0, h as i64 - 1) as usize;
            s[(cy * w + cx) * c + ch] as f32
        };
        for y in 0..h as i64 {
            for x in 0..w as i64 {
                for ch in 0..c {
                    let mut sum = 0f32;
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            sum += at(x + dx, y + dy, ch);
                        }
                    }
                    out[(y as usize * w + x as usize) * c + ch] = (sum / 9.0 + 0.5) as u8;
                }
            }
        }
        Image::from_raw(w, h, c, out).unwrap()
    }

    #[test]
    fn mean_blur_helper_reduces_variance() {
        // 纯逻辑验证：中心 255、其余 0 的 3x3 图，中心与角落窗口均值都是 255/9 ≈ 28
        let src = Image::from_raw(3, 3, 1, vec![0, 0, 0, 0, 255, 0, 0, 0, 0]).unwrap();
        let blurred = mean_blur_3x3(&src);
        assert_eq!(blurred.data()[4], 28, "中心像素 = 255/9 四舍五入");
        assert_eq!(blurred.data()[0], 28, "角落（边界复制后含中心极值）= 28");
    }

    // ==================== 真实模型自验（需 testmodels/，默认忽略） ====================

    /// 自验：清晰图手写 3x3 均值模糊后去模糊，跑通且输出尺寸与原图一致。
    /// 运行：cargo test --lib deblur -- --ignored --nocapture
    #[test]
    fn nafnet_deblur_restores_size() {
        let model_candidates = ["testmodels/nafnet_deblur_2025may.onnx", "models/deblur/nafnet_deblur_2025may.onnx"];
        if !model_candidates.iter().any(|p| std::path::Path::new(p).exists()) {
            eprintln!("[SKIP] 模型/测试资产缺失（models/ 或 testmodels/）");
            return;
        }
        let dir = std::env::var("TESTMODELS_DIR").unwrap_or_else(|_| "testmodels".into());
        let model = format!("{dir}/nafnet_deblur_2025may.onnx");
        let eng = DeblurEngine::new(&model, DeviceType::Cpu).unwrap();
        eprintln!("dynamic_hw={}", eng.is_dynamic_hw());

        // 取 bus.jpg 中部 512x512 区域（NafNet 需 512 对齐输入），手写 3x3 均值模糊
        let full = Image::load(format!("{dir}/bus.jpg")).unwrap();
        let crop = full.crop(200, 150, 512, 512).unwrap();
        let blurry = mean_blur_3x3(&crop);
        assert_eq!((blurry.width(), blurry.height()), (crop.width(), crop.height()));

        let t0 = std::time::Instant::now();
        let out = eng.predict(&blurry).unwrap();
        eprintln!(
            "去模糊 {}x{} → {}x{}，耗时 {} ms",
            blurry.width(),
            blurry.height(),
            out.width(),
            out.height(),
            t0.elapsed().as_millis()
        );
        assert_eq!(
            (out.width(), out.height(), out.channels()),
            (blurry.width(), blurry.height(), 3),
            "输出尺寸必须与输入一致"
        );

        // 去模糊输出应与模糊输入有差异（模型做了恢复），且仍是合理图像
        let diff: f64 = out
            .data()
            .iter()
            .zip(blurry.data().iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .sum::<f64>()
            / out.data().len() as f64;
        eprintln!("输出与模糊输入的平均绝对差 = {diff:.2}");
        assert!(diff > 0.5, "去模糊输出应与模糊输入存在差异，平均绝对差 {diff:.2}");

        // 注：该 NafNet 导出对非对齐小尺寸输入会在模型内部 Pad 节点报错
        //（如 201x151），属模型限制；引擎侧 align 对齐仅能覆盖部分尺寸。
    }
}
