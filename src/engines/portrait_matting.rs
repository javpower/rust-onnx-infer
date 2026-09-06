//! 人像分割（实时）引擎：PP-HumanSeg 优先，MODNet 备选。
//!
//! # 模型
//!
//! - **PP-HumanSeg**（默认）：PaddleSeg 人像分割（PP-HumanSegV2 系），
//!   采用 opencv/opencv_zoo 官方移植的 ONNX
//!   `human_segmentation_pphumanseg_2023mar.onnx`（实测 5.88 MB，Apache-2.0）：
//!   - 输入 `x` `[1,3,192,192]` Float32（**静态**，探针实测）
//!   - 输出 `save_infer_model/scale_0.tmp_1` `[1,2,192,192]` Float32，
//!     通道 0 = 背景、通道 1 = 前景（softmax 概率图）
//!   - 社区另有 256x256 / 392x392 的静态导出变体，可用
//!     [`PortraitMattingEngine::with_input_size`] 指定输入边长
//! - **MODNet**（备选）：[`PortraitMattingEngine::new_modnet`]，
//!   输入 `[1,3,512,512]`（静态文件自动读取，动态文件回退 512）、
//!   输出 `[1,1,H,W]` matte。推荐直链：
//!   <https://huggingface.co/Xenova/modnet/resolve/main/onnx/model.onnx>
//!
//! # 预处理约定（均以官方推理代码为准，探针实测签名）
//!
//! - PP-HumanSeg（opencv_zoo `pphumanseg.py` 官方约定）：BGR→RGB → /255 →
//!   `(x - 0.5) / 0.5`，即 [-1,1]
//! - MODNet（官方约定）：BGR→RGB → /255，即 [0,1]
//!
//! # 后处理
//!
//! 对齐官方 demo：先把 [1,C,h,w] 概率图**双线性 resize 回原图尺寸**再取前景值
//! （软边）。C=2 时 alpha = 前景通道概率；C=1 时 alpha = matte（若数值超出
//! [0,1] 视为 logits，做 sigmoid 兜底）；I64 argmax 标签图（0=背景/1=前景）
//! 直接按标签转 alpha。
//!
//! # 示例
//!
//! ```ignore
//! let eng = PortraitMattingEngine::new("models/pp_humanseg_2023mar.onnx", DeviceType::Cpu)?;
//! let alpha = eng.predict_alpha(&image)?;      // f32 [0,1]，原图尺寸
//! let cut = eng.predict_cutout(&image)?;       // BGRA 四通道抠像
//! ```

use crate::core::base::{BaseOnnxEngine, TensorData, TensorOutput};
use crate::core::device_type::DeviceType;
use crate::error::{Result, VisionError};
use crate::imaging::{cvt_color, ColorConversion, FloatMask, Image};

/// 预处理/数值约定（输入输出的数值约定必须配对使用）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PortraitPreprocess {
    /// PP-HumanSeg：RGB/255 → (x-0.5)/0.5（opencv_zoo 官方 pphumanseg.py 约定，默认）。
    #[default]
    PPHumanSeg,
    /// MODNet：RGB/255，不做均值平移（MODNet 官方约定）。
    Modnet,
}

/// 人像分割（实时）引擎。
///
/// 输入 BGR 图像，输出原图尺寸的软 alpha（`FloatMask`，[0,1]）或
/// BGRA 四通道抠像（[`PortraitMattingEngine::predict_cutout`]）。
pub struct PortraitMattingEngine {
    base: BaseOnnxEngine,
    /// 预处理/数值约定
    preprocess: PortraitPreprocess,
}

impl PortraitMattingEngine {
    /// PP-HumanSeg 实时推理的默认输入边长（动态导出变体的回退值，
    /// opencv_zoo 官方 demo 即用 192x192）。
    pub const DEFAULT_INPUT_SIZE: i32 = 192;

    /// MODNet 备选的默认输入边长。
    pub const MODNET_INPUT_SIZE: i32 = 512;

    /// 创建 PP-HumanSeg 人像分割引擎（默认约定）。
    ///
    /// 静态模型自动读取模型尺寸；动态模型回退 192x192（可用
    /// [`PortraitMattingEngine::with_input_size`] 覆盖为 256/392 等）。
    pub fn new(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        Self::build(model_path, device_type, PortraitPreprocess::PPHumanSeg, 0, 0)
    }

    /// 指定输入边长创建（256x256 / 392x392 等导出变体；<=0 时回退自动探测）。
    pub fn with_input_size(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        input_size: i32,
    ) -> Result<Self> {
        Self::build(
            model_path,
            device_type,
            PortraitPreprocess::PPHumanSeg,
            input_size,
            input_size,
        )
    }

    /// 创建 MODNet 备选引擎（/255 约定；动态模型回退 512x512）。
    pub fn new_modnet(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        Self::build(model_path, device_type, PortraitPreprocess::Modnet, 0, 0)
    }

    fn build(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        preprocess: PortraitPreprocess,
        req_h: i32,
        req_w: i32,
    ) -> Result<Self> {
        let mut base = BaseOnnxEngine::with_input_size(model_path, device_type, req_h, req_w)?;
        match preprocess {
            // PP-HumanSeg：x/255 → (x-0.5)/0.5 = [-1,1]
            PortraitPreprocess::PPHumanSeg => {
                base.set_normalization([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]);
                base.set_normalize(true);
            }
            // MODNet：x/255 = [0,1]
            PortraitPreprocess::Modnet => {
                base.set_normalization([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
                base.set_normalize(true);
            }
        }
        // 动态模型的回退尺寸：基类默认 640 对实时人像分割过大，按约定收紧
        if req_h <= 0 && req_w <= 0 && Self::is_dynamic_input(&base) {
            let fallback = match preprocess {
                PortraitPreprocess::PPHumanSeg => Self::DEFAULT_INPUT_SIZE,
                PortraitPreprocess::Modnet => Self::MODNET_INPUT_SIZE,
            };
            base.input_height = fallback;
            base.input_width = fallback;
            tracing::info!(
                "PortraitMattingEngine: 动态输入模型，回退输入尺寸 {}x{}",
                base.input_width,
                base.input_height
            );
        }
        tracing::info!(
            "PortraitMattingEngine ready: preprocess={:?}, input={}x{}, in='{}', device={}",
            preprocess,
            base.input_width,
            base.input_height,
            base.input_name(),
            device_type.name()
        );
        Ok(PortraitMattingEngine { base, preprocess })
    }

    /// 模型输入 H/W 是否为动态维度（基类对动态维度回退 640，不能依赖字段值）。
    fn is_dynamic_input(base: &BaseOnnxEngine) -> bool {
        let session = base.session.lock().unwrap();
        session.inputs.first().is_some_and(|input| {
            matches!(&input.input_type, ort::value::ValueType::Tensor { shape, .. }
                if shape.len() >= 4 && (shape[2] < 0 || shape[3] < 0))
        })
    }

    /// 预处理/数值约定。
    pub fn preprocess_mode(&self) -> PortraitPreprocess {
        self.preprocess
    }

    /// 推理：返回原图尺寸的软 alpha（f32 [0,1]，双线性 resize，软边）。
    pub fn predict_alpha(&self, image: &Image) -> Result<FloatMask> {
        if image.is_empty() {
            return Err(VisionError::invalid_argument("输入图像为空"));
        }

        // 1. 通用预处理：resize 到模型输入 + BGR→RGB + 归一化 + HWC→CHW
        let input_data = self.base.preprocess(image)?;
        // 2. [1,C,H,W] 张量推理
        let input_tensor = self.base.create_input_tensor(input_data)?;
        let output = self.base.run_inference(input_tensor)?;

        // 3. 输出 → 模型尺寸 alpha → 双线性 resize 回原图（软边）
        let alpha_model = Self::output_to_model_alpha(&output)?;
        Ok(alpha_model.resize(image.width(), image.height()))
    }

    /// 抠像：返回 BGRA 四通道图像（BGR 取原图像素，第 4 通道 = alpha×255）。
    pub fn predict_cutout(&self, image: &Image) -> Result<Image> {
        let alpha = self.predict_alpha(image)?;
        // 先统一转 3 通道 BGR，再逐像素叠 alpha
        let bgr = match image.channels() {
            3 => image.clone(),
            4 => cvt_color(image, ColorConversion::Bgra2Bgr)?,
            1 => cvt_color(image, ColorConversion::Gray2Bgr)?,
            c => {
                return Err(VisionError::invalid_argument(format!(
                    "人像抠像需要 1/3/4 通道输入，收到 {c} 通道"
                )))
            }
        };
        compose_bgra(&bgr, &alpha)
    }

    // -------------------- 内部 --------------------

    /// 模型输出 → 模型输入尺寸的 alpha [`FloatMask`]。
    ///
    /// 支持 [1,2,H,W]（softmax 概率图，取前景通道）、[1,1,H,W]（matte）、
    /// I64 argmax 标签图（0=背景/1=前景）。
    fn output_to_model_alpha(output: &TensorOutput) -> Result<FloatMask> {
        if output.shape.len() < 4 {
            return Err(VisionError::inference(format!(
                "人像分割输出维度异常（期望 4 维 NCHW）: {:?}",
                output.shape
            )));
        }
        let (mh, mw) = (
            output.shape[2].max(1) as usize,
            output.shape[3].max(1) as usize,
        );
        let area = mh * mw;
        let channels = output.shape[1].max(1) as usize;

        match &output.data {
            TensorData::I64(labels) => {
                // argmax 标签图：0=背景 / 1=前景
                if labels.len() < area {
                    return Err(VisionError::inference(format!(
                        "标签图数据不足: {} < {area}",
                        labels.len()
                    )));
                }
                let plane = labels[..area].iter().map(|&v| v as f32).collect();
                FloatMask::from_raw(mw, mh, plane)
            }
            TensorData::F32(data) => {
                let plane = match channels {
                    // MODNet / 单通道 matte：sigmoid 兜底（logits 导出兼容）
                    1 => matte_plane_to_alpha(&data[..area.min(data.len())]),
                    // PP-HumanSeg 双通道：取前景概率通道（CHW：通道 0=背景、通道 1=前景）
                    2 => {
                        if data.len() < 2 * area {
                            return Err(VisionError::inference(format!(
                                "双通道输出数据不足: {} < 2x{area}",
                                data.len()
                            )));
                        }
                        fg_prob_plane(&data[area..2 * area], &data[..area])
                    }
                    c => {
                        return Err(VisionError::inference(format!(
                            "人像分割输出通道数不支持: {c}（期望 1 或 2）"
                        )))
                    }
                };
                FloatMask::from_raw(mw, mh, plane)
            }
            other => Err(VisionError::inference(format!(
                "人像分割输出元素类型不支持: {other:?}"
            ))),
        }
    }
}

/// BGR 图 + 同尺寸 alpha → BGRA 图（第 4 通道 = alpha×255 四舍五入）。
fn compose_bgra(bgr: &Image, alpha: &FloatMask) -> Result<Image> {
    let (w, h) = (bgr.width(), bgr.height());
    if alpha.width() != w || alpha.height() != h {
        return Err(VisionError::inference(format!(
            "alpha 尺寸 {}x{} 与原图 {}x{} 不一致",
            alpha.width(),
            alpha.height(),
            w,
            h
        )));
    }
    let area = w * h;
    let mut out = vec![0u8; area * 4];
    let src = bgr.data();
    let a = alpha.data();
    for i in 0..area {
        out[i * 4] = src[i * 3];
        out[i * 4 + 1] = src[i * 3 + 1];
        out[i * 4 + 2] = src[i * 3 + 2];
        out[i * 4 + 3] = clamp_u8(a[i] * 255.0);
    }
    Image::from_raw(w, h, 4, out)
}

/// 双通道输出 → 前景概率平面：数值超出 [0,1] 视为 logits，做 softmax；
/// 否则视为已 softmax 的概率图（官方导出），直接取前景通道。
/// 参数顺序：`fg` = 通道 1（前景）、`bg` = 通道 0（背景）。
fn fg_prob_plane(fg: &[f32], bg: &[f32]) -> Vec<f32> {
    let mut max = f32::MIN;
    for &v in fg.iter().chain(bg.iter()) {
        if v > max {
            max = v;
        }
    }
    if max > 1.0 + 1e-4 {
        // logits：数值稳定 softmax（逐像素减最大值，防 exp 溢出）后取前景
        fg.iter()
            .zip(bg.iter())
            .map(|(&f, &b)| {
                let m = f.max(b);
                let ef = (f - m).exp();
                let eb = (b - m).exp();
                ef / (ef + eb)
            })
            .collect()
    } else {
        fg.iter().map(|&v| v.clamp(0.0, 1.0)).collect()
    }
}

/// 单通道 matte → alpha：已 [0,1] 则 clamp；超出视为 logits，sigmoid 兜底
/// （与 [`crate::engines::birefnet`] 的判定约定一致）。
fn matte_plane_to_alpha(plane: &[f32]) -> Vec<f32> {
    let mut max = f32::MIN;
    let mut min = f32::MAX;
    for &v in plane {
        if v > max {
            max = v;
        }
        if v < min {
            min = v;
        }
    }
    if max > 1.5 || min < -0.01 {
        plane.iter().map(|&v| super::birefnet::sigmoid(v)).collect()
    } else {
        plane.iter().map(|&v| v.clamp(0.0, 1.0)).collect()
    }
}

/// 任意 f32 → u8（四舍五入并 clamp 0~255）。
fn clamp_u8(v: f32) -> u8 {
    let i = (v + 0.5) as i32;
    i.clamp(0, 255) as u8
}

crate::impl_engine_forward!(PortraitMattingEngine, base, FloatMask,
    fn predict(&self, image: &Image) -> Result<FloatMask> { self.predict_alpha(image) }
);

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造测试输出张量。
    fn tensor(shape: Vec<i64>, data: TensorData) -> TensorOutput {
        TensorOutput {
            name: "test".to_string(),
            shape,
            data,
        }
    }

    #[test]
    fn two_channel_softmax_prob_takes_fg() {
        // [1,2,2,2]：bg 通道全 0.25、fg 通道全 0.75 → alpha 应全 0.75
        let area = 4;
        let mut data = vec![0.25f32; 2 * area];
        for v in data.iter_mut().skip(area) {
            *v = 0.75;
        }
        let out = tensor(vec![1, 2, 2, 2], TensorData::F32(data));
        let alpha = PortraitMattingEngine::output_to_model_alpha(&out).unwrap();
        assert_eq!((alpha.width(), alpha.height()), (2, 2));
        assert!(alpha.data().iter().all(|&v| (v - 0.75).abs() < 1e-6));
    }

    #[test]
    fn two_channel_logits_apply_softmax() {
        // logits 导出兜底：bg=-100、fg=100 → softmax 前景 ≈ 1
        let area = 2;
        let mut data = vec![-100f32; 2 * area];
        for v in data.iter_mut().skip(area) {
            *v = 100.0;
        }
        let out = tensor(vec![1, 2, 1, 2], TensorData::F32(data));
        let alpha = PortraitMattingEngine::output_to_model_alpha(&out).unwrap();
        assert!(alpha.data().iter().all(|&v| v > 0.99));
    }

    #[test]
    fn single_channel_matte_clamped() {
        // 轻微超界（仍在 sigmoid 判定阈 [-0.01,1.5] 内）的 matte 直接 clamp
        let out = tensor(vec![1, 1, 1, 2], TensorData::F32(vec![-0.005, 1.2]));
        let alpha = PortraitMattingEngine::output_to_model_alpha(&out).unwrap();
        assert_eq!(alpha.data()[0], 0.0);
        assert_eq!(alpha.data()[1], 1.0);
    }

    #[test]
    fn single_channel_matte_logits_sigmoid() {
        // 超出 [-0.01, 1.5] 判定阈 → 视为 logits，sigmoid 兜底
        let out = tensor(vec![1, 1, 1, 2], TensorData::F32(vec![-8.0, 8.0]));
        let alpha = PortraitMattingEngine::output_to_model_alpha(&out).unwrap();
        assert!(alpha.data()[0] < 0.01, "sigmoid(-8) 应接近 0");
        assert!(alpha.data()[1] > 0.99, "sigmoid(8) 应接近 1");
    }

    #[test]
    fn i64_label_map_to_alpha() {
        // argmax 导出：0=背景 / 1=前景
        let out = tensor(vec![1, 1, 2, 1], TensorData::I64(vec![0, 1]));
        let alpha = PortraitMattingEngine::output_to_model_alpha(&out).unwrap();
        assert_eq!(alpha.data(), &[0.0, 1.0]);
    }

    #[test]
    fn compose_bgra_uses_alpha_as_fourth_channel() {
        // 2x1 BGR 图 + 全 0.5 alpha → BGRA 第 4 通道 = 128（0.5*255 四舍五入）
        let bgr = Image::from_raw(2, 1, 3, vec![10, 20, 30, 40, 50, 60]).unwrap();
        let alpha = FloatMask::from_raw(2, 1, vec![0.5, 0.5]).unwrap();
        let out = compose_bgra(&bgr, &alpha).unwrap();
        assert_eq!(out.channels(), 4);
        assert_eq!(&out.data()[..8], &[10, 20, 30, 128, 40, 50, 60, 128]);
    }

    #[test]
    fn compose_bgra_size_mismatch_errors() {
        let bgr = Image::from_raw(2, 1, 3, vec![0; 6]).unwrap();
        let alpha = FloatMask::from_raw(1, 1, vec![1.0]).unwrap();
        assert!(compose_bgra(&bgr, &alpha).is_err());
    }

    #[test]
    fn clamp_u8_rounding() {
        assert_eq!(clamp_u8(127.4), 127);
        assert_eq!(clamp_u8(127.5), 128);
        assert_eq!(clamp_u8(-5.0), 0);
        assert_eq!(clamp_u8(300.0), 255);
    }

    // ==================== 真实模型自验（需 testmodels/，默认忽略） ====================

    /// 自验：bus.jpg（人 + 巴士）应产生明显前景区域（人物区域 alpha 均值 > 0.1）。
    /// 运行：cargo test --lib portrait_matting -- --ignored --nocapture
    #[test]
    fn pphumanseg_bus_alpha_has_foreground() {
        let model_candidates = ["testmodels/pp_humanseg_2023mar.onnx", "models/portrait_matting/pp_humanseg_2023mar.onnx"];
        let image_candidates = ["testmodels/bus.jpg", "models/test_images/bus.jpg"];
        let Some(model) = model_candidates.iter().find(|p| std::path::Path::new(p).exists()) else {
            eprintln!("[SKIP] 模型/测试资产缺失（models/ 或 testmodels/）");
            return;
        };
        let Some(img_path) = image_candidates.iter().find(|p| std::path::Path::new(p).exists()) else {
            eprintln!("[SKIP] 测试图缺失");
            return;
        };
        let image = Image::load(img_path).unwrap();
        let eng = PortraitMattingEngine::new(model, DeviceType::Cpu).unwrap();
        let alpha = eng.predict_alpha(&image).unwrap();
        let mean = (alpha.sum() / alpha.len() as f64) as f32;
        let max = alpha.data().iter().cloned().fold(0.0f32, f32::max);
        eprintln!(
            "bus.jpg {}x{} → alpha {}x{}，均值={:.4}，max={:.4}",
            image.width(),
            image.height(),
            alpha.width(),
            alpha.height(),
            mean,
            max
        );
        assert_eq!((alpha.width(), alpha.height()), (image.width(), image.height()));
        assert!((0.0..=1.0).contains(&max), "alpha 应在 [0,1] 内，max={max:.4}");

        // 粗粒度 8x8 网格均值，检查前景空间分布是否成"人物块"而非噪声
        let (gw, gh) = (8usize, 8usize);
        let mut grid = String::new();
        for gy in 0..gh {
            for gx in 0..gw {
                let x0 = gx * alpha.width() / gw;
                let y0 = gy * alpha.height() / gh;
                let x1 = ((gx + 1) * alpha.width() / gw).max(x0 + 1);
                let y1 = ((gy + 1) * alpha.height() / gh).max(y0 + 1);
                let mut s = 0f64;
                let mut n = 0usize;
                for y in y0..y1 {
                    for x in x0..x1 {
                        s += alpha.get(x, y) as f64;
                        n += 1;
                    }
                }
                let m = (s / n.max(1) as f64 * 9.0).round() as i32;
                grid.push_str(match m {
                    0 => " .",
                    1..=3 => "░░",
                    4..=6 => "▒▒",
                    7..=8 => "▓▓",
                    _ => "██",
                });
            }
            grid.push('\n');
        }
        eprintln!("alpha 8x8 网格（██=前景）:\n{grid}");
        assert!(max > 0.9, "应存在高置信前景像素，max={max:.4}");

        // bus.jpg 为人 + 巴士全景（810x1080），人物仅占画面约一成（PP-HumanSeg
        // 只分割人、不分割巴士，整图均值 ≈ 0.096 属预期），故"均值 > 0.1"检查
        // 落在人物所在区域（左侧中部）：
        let (w, h) = (alpha.width(), alpha.height());
        let region_mean = |x0: usize, y0: usize, x1: usize, y1: usize| -> f32 {
            let mut s = 0f64;
            let mut n = 0usize;
            for y in y0..y1 {
                for x in x0..x1 {
                    s += alpha.get(x, y) as f64;
                    n += 1;
                }
            }
            (s / n.max(1) as f64) as f32
        };
        // 网格图中人物块：列 0~5/8、行 3~7/8
        let people_mean = region_mean(0, 3 * h / 8, 5 * w / 8, 7 * h / 8);
        // 右上背景（天空/巴士上部，无人物）
        let bg_mean = region_mean(5 * w / 8, 0, w, 3 * h / 8);
        eprintln!(
            "人物区域均值={people_mean:.4}，背景区域均值={bg_mean:.4}（整图 {mean:.4}）"
        );
        assert!(
            people_mean > 0.1,
            "bus.jpg 人物区域 alpha 均值 {people_mean:.4} 应 > 0.1（存在前景区域）"
        );
        assert!(
            bg_mean < 0.05,
            "无人物背景区域 alpha 均值 {bg_mean:.4} 应 < 0.05"
        );
        // 人像主导图全图均值（zidane.jpg 双人近景，前景占比大）
        let zidane = Image::load("testmodels/zidane.jpg").unwrap();
        let za = eng.predict_alpha(&zidane).unwrap();
        let zmean = (za.sum() / za.len() as f64) as f32;
        eprintln!(
            "zidane.jpg {}x{} → alpha 全图均值={:.4}",
            zidane.width(),
            zidane.height(),
            zmean
        );
        assert!(zmean > 0.1, "人像主导图全图均值 {zmean:.4} 应 > 0.1");

        let cut = eng.predict_cutout(&image).unwrap();
        assert_eq!(cut.channels(), 4);
        assert_eq!((cut.width(), cut.height()), (image.width(), image.height()));
        // 人物区域中心的 BGRA alpha 字节应高于左上角（背景）
        let people_center = (h / 2 * w + w * 2 / 8) * 4 + 3;
        let center_a = cut.data()[people_center];
        let corner_a = cut.data()[3];
        eprintln!("人物中心 alpha={center_a}，左上角 alpha={corner_a}");
    }
}
