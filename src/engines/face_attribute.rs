//! 人脸属性引擎：年龄 + 性别（[`AgeGenderEngine`]) 与 表情识别（[`ExpressionEngine`]）。
//!
//! 两个引擎均为「检测后属性分类」模式：上游由 YuNet 人脸检测
//! （[`crate::engines::face_detection::FaceDetectionEngine`]）给出人脸框
//! （`detection.bbox` 为 f64 的 x1/y1/x2/y2，可用 [`rect_from_xyxy`] 换算为本引擎
//! 接受的整型 [`Rect`]），本引擎按框裁脸（默认外扩 20%）后推理。
//!
//! # AgeGenderEngine — age-gender-recognition-retail-0013
//!
//! OpenVINO Open Model Zoo 模型（已对照官方模型卡
//! <https://docs.openvino.ai/2023.3/omz_models_model_age_gender_recognition_retail_0013.html>
//! 与仓库内真实模型逐项核实）：
//!
//! - **输入**：人脸 crop，`62x62`，**BGR 通道序**，0~255 原始像素值
//!   （官方未定义 mean/std 归一化，等价 mean=0）；
//! - **年龄输出**：标量回归值 = 实际年龄 / 100，即 **× 100 得到岁数**；
//!   官方口径训练数据覆盖 [18, 75] 岁，不适用于儿童；
//! - **性别输出**：`[female, male]` 两类 softmax 概率（**index 0 = female，
//!   index 1 = male**，官方导出已含 softmax，引擎对概率和 ≁ 1 的第三方导出兜底补 softmax）；
//! - **I/O 布局兼容**：官方权重没有 ONNX 直链发行，`testmodels/age_gender.onnx`
//!   为 PINTO model zoo #070 基于同一 OMZ 权重的 TF-ONNX 导出
//!   （下载：`https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/070_age-gender-recognition/resources.tar.gz`，
//!   权重规模 8,560,731 字节与官方 FP32 IR 的 8,552,076 字节一致）。该导出为
//!   **NHWC** `[1,62,62,3]`、输入名 `data:0`、输出名 `Identity`/`Identity_1`；
//!   而自行从 OMZ IR 转出的 ONNX 通常为 **NCHW** `[1,3,62,62]`、输入名 `data`、
//!   输出名 `age_conv3`/`prob`。引擎在构造时探测输入形状自动选择内存布局，
//!   输出则先按名字（含 `age` / `prob`|`gender`）匹配、失败后按
//!   「每样本 1 元素 = 年龄，2 元素 = 性别」的形状兜底，两种导出均可直接使用。
//!
//! # ExpressionEngine — emotion-ferplus
//!
//! ONNX Model Zoo 模型（已对照官方模型卡
//! <https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus>
//! 与仓库内真实模型核实）：
//!
//! - **输入**：`Input3`，`64x64` **灰度**单通道，float32 **0~255 原始值**
//!   （官方示例预处理仅「灰度 → resize 64x64」，不做归一化缩放）；
//! - **输出**：`Plus692_Output_0`，`[1,8]` logits，调用方做 softmax；
//! - **类别顺序**（官方 `emotion_table`，fer2013 的 7 类 + FER+ 扩展的 contempt）：
//!   `neutral, happiness, surprise, sadness, anger, disgust, fear, contempt`；
//! - `testmodels/emotion_ferplus.onnx` 下载自 HF 官方镜像
//!   `https://huggingface.co/onnxmodelzoo/emotion-ferplus-8`（ferplus-8/9 权重与
//!   输入输出签名完全一致，仅 opset 差异）。
//!
//! 两个模型的 batch 维均为静态 1：批量接口优先尝试真实打包批量，模型拒绝时
//! 自动退化为逐脸推理循环，对调用方透明。

use ort::value::{Tensor, ValueType};

use crate::core::base::{BaseOnnxEngine, TensorOutput};
use crate::core::device_type::DeviceType;
use crate::error::{Result, VisionError};
use crate::imaging::{cvt_color, resize, ColorConversion, Image, Interpolation, Rect};

// ============================================================================
// 公共定义
// ============================================================================

/// 年龄输出 = 实际年龄 / 100（官方模型卡口径），推理结果 × 100。
const AGE_SCALE: f32 = 100.0;

/// 性别 softmax 概率顺序：index 0 = female，index 1 = male（官方模型卡口径）。
pub const GENDER_LABELS: [&str; 2] = ["female", "male"];

/// FER+ 表情类别顺序（ONNX Model Zoo 官方 `emotion_table`：
/// fer2013 原 7 类 + FER+ 扩展的 contempt）。
pub const EXPRESSION_LABELS: [&str; 8] = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
];

/// 由 `(x1, y1, x2, y2)` 浮点框（如 YuNet [`crate::model::Detection`] 的
/// `bbox.x1/y1/x2/y2`）换算为本引擎接受的整型 [`Rect`]。
pub fn rect_from_xyxy(x1: f64, y1: f64, x2: f64, y2: f64) -> Rect {
    let left = x1.min(x2).round() as i32;
    let top = y1.min(y2).round() as i32;
    let width = (x2 - x1).abs().round() as i32;
    let height = (y2 - y1).abs().round() as i32;
    Rect::new(left, top, width, height)
}

/// 按人脸框裁剪并 resize 到模型输入尺寸。
///
/// 框先 clamp 到图内（检测框可能带负坐标 / 越界），再四边各外扩 `expand` 比例并
/// 重新 clamp；`to_gray=true` 时裁剪后转单通道灰度（表情模型输入）。
fn crop_and_resize(
    image: &Image,
    face_box: Rect,
    expand: f32,
    width: usize,
    height: usize,
    to_gray: bool,
) -> Result<Image> {
    if image.is_empty() {
        return Err(VisionError::image(
            "cannot extract face attribute from empty image",
        ));
    }
    let (img_w, img_h) = (image.width() as i32, image.height() as i32);

    // 1. clamp 到图内（x/y/width/height 语义：右 = x + width，下 = y + height）
    let mut x1 = face_box.x.max(0);
    let mut y1 = face_box.y.max(0);
    let mut x2 = face_box.right().min(img_w);
    let mut y2 = face_box.bottom().min(img_h);
    if x2 - x1 <= 0 || y2 - y1 <= 0 {
        return Err(VisionError::invalid_argument(format!(
            "face box {:?} has no overlap with image {}x{}",
            face_box, img_w, img_h
        )));
    }

    // 2. 四边外扩 expand 比例后重新 clamp（给属性模型更多上下文，缓解裁剪过紧）
    let dx = ((x2 - x1) as f32 * expand).round() as i32;
    let dy = ((y2 - y1) as f32 * expand).round() as i32;
    x1 = (x1 - dx).max(0);
    y1 = (y1 - dy).max(0);
    x2 = (x2 + dx).min(img_w);
    y2 = (y2 + dy).min(img_h);

    let crop = image.crop(x1 as usize, y1 as usize, (x2 - x1) as usize, (y2 - y1) as usize)?;

    // 3. 通道转换（Image 容器为 BGR/BGRA 通道序）
    let converted = match (to_gray, crop.channels()) {
        (true, 1) => crop,
        (true, 3) => cvt_color(&crop, ColorConversion::Bgr2Gray)?,
        (true, 4) => cvt_color(&crop, ColorConversion::Bgra2Gray)?,
        (true, c) => return Err(VisionError::image(format!("unsupported channel count {c}"))),
        (false, 3) => crop,
        (false, 4) => cvt_color(&crop, ColorConversion::Bgra2Bgr)?,
        (false, 1) => cvt_color(&crop, ColorConversion::Gray2Bgr)?,
        (false, c) => return Err(VisionError::image(format!("unsupported channel count {c}"))),
    };

    // 4. 拉伸 resize 到模型输入尺寸
    resize(&converted, width, height, Interpolation::Linear)
}

/// 数值稳定的 softmax（对齐 yolo_e_runtime 的风格）。
fn softmax(values: &[f32]) -> Vec<f32> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = values.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return vec![1.0 / values.len() as f32; values.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

/// 单个类别 argmax。
fn argmax(values: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &v) in values.iter().enumerate() {
        if v > values[best] {
            best = i;
        }
    }
    best
}

// ============================================================================
// AgeGenderEngine
// ============================================================================

/// 单人脸年龄 + 性别属性（age-gender-recognition-retail-0013）。
#[derive(Debug, Clone, PartialEq)]
pub struct FaceAttribute {
    /// 估计年龄（岁）。官方口径训练覆盖 [18, 75]，对儿童不可靠。
    pub age: f32,
    /// 是否男性（gender softmax 概率较大的一侧）。
    pub is_male: bool,
    /// 性别置信度：预测类别（`is_male` 对应一侧）的 softmax 概率。
    pub gender_score: f32,
}

/// 年龄 + 性别识别引擎（OpenVINO `age-gender-recognition-retail-0013`）。
///
/// 输入 62x62 BGR（0~255，无归一化）；输出年龄标量（× 100 为岁数）与
/// `[female, male]` softmax 概率。同时兼容 NCHW（OMZ IR 派生导出）与
/// NHWC（PINTO / TF 系导出）两种输入布局，输出按名字 / 形状自动定位
/// （详见模块注释）。
///
/// ```ignore
/// let eng = AgeGenderEngine::new("testmodels/age_gender.onnx", DeviceType::Cpu)?;
/// let faces = yunet.predict(&image)?;                       // 上游检测
/// let box_ = rect_from_xyxy(f.detection.x1(), f.detection.y1(),
///                           f.detection.x2(), f.detection.y2());
/// let attr = eng.predict_face(&image, box_)?;               // attr.age / attr.is_male
/// ```
pub struct AgeGenderEngine {
    /// 组合基类
    pub base: BaseOnnxEngine,
    /// 输入张量是否为 NHWC 布局（PINTO/TF 系导出；false = NCHW，OMZ IR 派生导出）
    nhwc: bool,
    /// 人脸框四边外扩比例（默认 0.2）
    expand_ratio: f32,
}

impl AgeGenderEngine {
    /// 模型输入边长（62x62，BGR 3 通道）。
    pub const INPUT_SIZE: i32 = 62;
    /// 人脸框默认外扩比例（20%）。
    pub const DEFAULT_EXPAND_RATIO: f32 = 0.2;

    /// 创建年龄 + 性别识别引擎。
    pub fn new(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        let mut base = BaseOnnxEngine::with_input_size(
            model_path,
            device_type,
            Self::INPUT_SIZE,
            Self::INPUT_SIZE,
        )?;
        // 输入为 0~255 原始 BGR 像素（官方无 mean/std 归一化），关闭基类归一化以表意
        base.set_normalize(false);
        // 性别类别名（trait labels() 暴露）
        base.set_labels(GENDER_LABELS.iter().map(|s| s.to_string()).collect());

        // 探测输入张量内存布局：len==4 且 C 在 dims[3]（==3）为 NHWC，否则按 NCHW 处理
        let nhwc = {
            let session = base.session.lock().unwrap();
            session
                .inputs()
                .first()
                .and_then(|input| match input.dtype() {
                    ValueType::Tensor { shape, .. } => {
                        let dims: Vec<i64> = shape.iter().copied().collect();
                        Some(dims.len() == 4 && dims.get(3) == Some(&3) && dims.get(1) != Some(&3))
                    }
                    _ => None,
                })
                .unwrap_or(false)
        };
        // NHWC 导出（[1,62,62,3]）时基类把 dims[1]=62 误解析为通道数、dims[3]=3 误解析
        // 为宽度，手工纠正为 NCHW 语义值（本引擎张量自行构造，不依赖基类 preprocess）
        if nhwc {
            base.input_channels = 3;
            base.input_height = Self::INPUT_SIZE;
            base.input_width = Self::INPUT_SIZE;
        }

        tracing::info!(
            "AgeGenderEngine initialized: input={}x62x62 {}, device={}, outputs={:?}",
            Self::INPUT_SIZE,
            if nhwc { "NHWC" } else { "NCHW" },
            base.device_type().name(),
            base.output_names()
        );

        Ok(AgeGenderEngine {
            base,
            nhwc,
            expand_ratio: Self::DEFAULT_EXPAND_RATIO,
        })
    }

    // ==================== 访问器 ====================

    /// 人脸框外扩比例。
    pub fn expand_ratio(&self) -> f32 {
        self.expand_ratio
    }

    /// 设置人脸框外扩比例（0 = 不外扩）。
    pub fn set_expand_ratio(&mut self, ratio: f32) {
        self.expand_ratio = ratio.max(0.0);
    }

    /// 输入张量是否为 NHWC 布局。
    pub fn is_nhwc(&self) -> bool {
        self.nhwc
    }

    // ==================== 推理 ====================

    /// 单人脸年龄 + 性别推理（人脸框来自上游人脸检测）。
    pub fn predict_face(&self, image: &Image, face_box: Rect) -> Result<FaceAttribute> {
        Ok(self.predict_batch(image, std::slice::from_ref(&face_box))?.remove(0))
    }

    /// 多人脸批量推理：一次裁出全部人脸，打包成 batch 张量推理
    /// （官方导出 batch 维静态为 1，此时自动退化为逐脸推理，结果一致）。
    pub fn predict_batch(&self, image: &Image, face_boxes: &[Rect]) -> Result<Vec<FaceAttribute>> {
        if face_boxes.is_empty() {
            return Ok(Vec::new());
        }

        // 1. 裁剪 + 预处理（62x62 BGR float 0~255）
        let crops: Vec<Image> = face_boxes
            .iter()
            .map(|&b| crop_and_resize(image, b, self.expand_ratio, 62, 62, false))
            .collect::<Result<_>>()?;

        // 2. 推理（优先打包批量；模型拒绝时退化为逐脸）
        match self.infer_packed(&crops) {
            Ok(outputs) => {
                let (age_idx, gender_idx) = Self::resolve_output_indices(&outputs)?;
                let age_data = outputs[age_idx].as_f32()?;
                let gender_data = outputs[gender_idx].as_f32()?;
                if age_data.len() < face_boxes.len() || gender_data.len() < face_boxes.len() * 2 {
                    return Err(VisionError::inference(format!(
                        "age-gender output size mismatch: age={}, gender={}, expect >= {}, {}",
                        age_data.len(),
                        gender_data.len(),
                        face_boxes.len(),
                        face_boxes.len() * 2
                    )));
                }
                let mut results = Vec::with_capacity(face_boxes.len());
                for k in 0..face_boxes.len() {
                    results.push(decode_face_attribute(
                        age_data[k],
                        &gender_data[k * 2..k * 2 + 2],
                    ));
                }
                tracing::info!("AgeGenderEngine inferred {} face(s) (batched)", results.len());
                Ok(results)
            }
            Err(batch_err) if crops.len() > 1 => {
                tracing::warn!(
                    "batched age-gender inference unavailable ({}), falling back to per-face loop",
                    batch_err
                );
                let mut results = Vec::with_capacity(crops.len());
                for crop in &crops {
                    let outputs = self.infer_packed(std::slice::from_ref(crop))?;
                    let (age_idx, gender_idx) = Self::resolve_output_indices(&outputs)?;
                    let age_data = outputs[age_idx].as_f32()?;
                    let gender_data = outputs[gender_idx].as_f32()?;
                    if age_data.is_empty() || gender_data.len() < 2 {
                        return Err(VisionError::inference(
                            "age-gender output is empty for single-face inference",
                        ));
                    }
                    results.push(decode_face_attribute(age_data[0], &gender_data[..2]));
                }
                tracing::info!("AgeGenderEngine inferred {} face(s) (loop)", results.len());
                Ok(results)
            }
            Err(e) => Err(e),
        }
    }

    /// 打包 batch 张量并推理（NCHW `[N,3,62,62]` / NHWC `[N,62,62,3]`）。
    fn infer_packed(&self, crops: &[Image]) -> Result<Vec<TensorOutput>> {
        let area = 62 * 62;
        let mut data = Vec::with_capacity(crops.len() * 3 * area);
        for crop in crops {
            if crop.width() != 62 || crop.height() != 62 || crop.channels() != 3 {
                return Err(VisionError::inference(format!(
                    "age-gender crop layout mismatch: got {}x{}x{}, expect 62x62x3",
                    crop.width(),
                    crop.height(),
                    crop.channels()
                )));
            }
            let px = crop.data();
            if self.nhwc {
                // NHWC：逐像素 BGR 平铺
                data.extend(px.iter().map(|&v| v as f32));
            } else {
                // NCHW：B/G/R 三平面
                let mut planes = vec![0f32; 3 * area];
                for i in 0..area {
                    planes[i] = px[i * 3] as f32;
                    planes[i + area] = px[i * 3 + 1] as f32;
                    planes[i + 2 * area] = px[i * 3 + 2] as f32;
                }
                data.extend_from_slice(&planes);
            }
        }
        let shape = if self.nhwc {
            vec![crops.len() as i64, 62, 62, 3]
        } else {
            vec![crops.len() as i64, 3, 62, 62]
        };
        let tensor = Tensor::from_array((shape, data))?;
        self.base.run_multi_output(tensor)
    }

    /// 定位（年龄, 性别）输出下标：先按名字（`age` / `prob`|`gender`）匹配，
    /// 失败后按「每样本 1 元素 = 年龄，2 元素 = 性别」的形状兜底
    /// （兼容 OMZ IR 派生导出 age_conv3/prob 与 PINTO 导出 Identity/Identity_1）。
    fn resolve_output_indices(outputs: &[TensorOutput]) -> Result<(usize, usize)> {
        let mut age = None;
        let mut gender = None;
        for (i, out) in outputs.iter().enumerate() {
            let name = out.name.to_lowercase();
            if name.contains("age") {
                age = Some(i);
            }
            if name.contains("prob") || name.contains("gender") {
                gender = Some(i);
            }
        }
        if let (Some(a), Some(g)) = (age, gender) {
            if a != g {
                return Ok((a, g));
            }
        }
        // 形状兜底（PINTO 导出输出名为 Identity/Identity_1，无语义信息）
        let batch = outputs
            .first()
            .and_then(|o| o.shape.first().copied())
            .map(|b| b.max(1) as usize)
            .unwrap_or(1);
        for (i, out) in outputs.iter().enumerate() {
            match out.element_count().checked_div(batch) {
                Some(1) if age.is_none() => age = Some(i),
                Some(2) if gender.is_none() => gender = Some(i),
                _ => {}
            }
        }
        match (age, gender) {
            (Some(a), Some(g)) if a != g => Ok((a, g)),
            _ => Err(VisionError::inference(format!(
                "cannot locate age/gender outputs among {:?}",
                outputs.iter().map(|o| (o.name.as_str(), o.shape.as_slice())).collect::<Vec<_>>()
            ))),
        }
    }
}

/// 由年龄标量与性别 `[female, male]` 概率组装结果（概率和 ≁ 1 时兜底补 softmax）。
fn decode_face_attribute(age_raw: f32, gender_prob: &[f32]) -> FaceAttribute {
    // 官方口径：输出 = 实际年龄 / 100
    let age = age_raw * AGE_SCALE;
    let mut prob = [gender_prob[0], gender_prob[1]];
    // 官方导出的 prob 已含 softmax；对未含 softmax 的第三方导出兜底
    if (prob[0] + prob[1] - 1.0).abs() > 1e-3 {
        let s = softmax(&prob);
        prob = [s[0], s[1]];
    }
    let is_male = prob[1] >= prob[0];
    FaceAttribute {
        age,
        is_male,
        gender_score: if is_male { prob[1] } else { prob[0] },
    }
}

crate::impl_engine_forward!(AgeGenderEngine, base, FaceAttribute,
    /// trait `predict` 不适用：人脸属性推理必须提供人脸框（来自 YuNet 等检测器），
    /// 请使用 [`AgeGenderEngine::predict_face`] / [`AgeGenderEngine::predict_batch`]
    /// （对齐 [`crate::engines::sam::SamEngine`] 的交互式引擎处理方式）。
    fn predict(&self, _image: &Image) -> Result<FaceAttribute> {
        Err(VisionError::Unsupported(
            "age-gender recognition needs a face box; use predict_face / predict_batch instead"
                .to_string(),
        ))
    },
    /// trait `predict_batch`（整图批量）同理不支持：批量粒度为「单图 + 多人脸框」。
    fn predict_batch(&self, _images: &[Image]) -> Result<Vec<FaceAttribute>> {
        Err(VisionError::Unsupported(
            "age-gender recognition needs face boxes; use predict_batch(image, &[Rect]) instead"
                .to_string(),
        ))
    }
);

// ============================================================================
// ExpressionEngine
// ============================================================================

/// 单人脸表情识别结果（emotion-ferplus）。
#[derive(Debug, Clone, PartialEq)]
pub struct Expression {
    /// 表情类别名（[`EXPRESSION_LABELS`] 之一）。
    pub label: String,
    /// 类别 ID（对应 [`EXPRESSION_LABELS`] 下标）。
    pub label_id: usize,
    /// 8 类 softmax 概率（顺序 = [`EXPRESSION_LABELS`]）。
    pub scores: Vec<f32>,
}

impl Expression {
    /// argmax 类别的概率。
    pub fn confidence(&self) -> f32 {
        self.scores.get(self.label_id).copied().unwrap_or(0.0)
    }
}

/// 表情识别引擎（ONNX Model Zoo `emotion-ferplus`，VGG-13）。
///
/// 输入 64x64 灰度（float32，0~255 原始值，无归一化缩放）；输出 `[1,8]` logits，
/// softmax 后按 [`EXPRESSION_LABELS`] 顺序取 argmax。
pub struct ExpressionEngine {
    /// 组合基类
    pub base: BaseOnnxEngine,
    /// 人脸框四边外扩比例（默认 0.2）
    expand_ratio: f32,
}

impl ExpressionEngine {
    /// 模型输入边长（64x64，单通道灰度）。
    pub const INPUT_SIZE: i32 = 64;
    /// 人脸框默认外扩比例（20%）。
    pub const DEFAULT_EXPAND_RATIO: f32 = 0.2;

    /// 创建表情识别引擎。
    pub fn new(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        let mut base = BaseOnnxEngine::with_input_size(
            model_path,
            device_type,
            Self::INPUT_SIZE,
            Self::INPUT_SIZE,
        )?;
        // 输入为 0~255 原始灰度值（官方示例不做归一化缩放），关闭基类归一化以表意
        base.set_normalize(false);
        // FER+ 8 类表情名（模型元数据无标签，手工注入供 trait labels() 暴露）
        base.set_labels(EXPRESSION_LABELS.iter().map(|s| s.to_string()).collect());

        tracing::info!(
            "ExpressionEngine initialized: input={}x64x64 (gray), device={}, outputs={:?}",
            Self::INPUT_SIZE,
            base.device_type().name(),
            base.output_names()
        );

        Ok(ExpressionEngine {
            base,
            expand_ratio: Self::DEFAULT_EXPAND_RATIO,
        })
    }

    // ==================== 访问器 ====================

    /// 人脸框外扩比例。
    pub fn expand_ratio(&self) -> f32 {
        self.expand_ratio
    }

    /// 设置人脸框外扩比例（0 = 不外扩）。
    pub fn set_expand_ratio(&mut self, ratio: f32) {
        self.expand_ratio = ratio.max(0.0);
    }

    // ==================== 推理 ====================

    /// 单人脸表情推理（人脸框来自上游人脸检测）。
    pub fn predict_expr(&self, image: &Image, face_box: Rect) -> Result<Expression> {
        Ok(self.predict_batch(image, std::slice::from_ref(&face_box))?.remove(0))
    }

    /// 多人脸批量推理：一次裁出全部人脸，打包成 batch 张量推理
    /// （官方导出 batch 维静态为 1，此时自动退化为逐脸推理，结果一致）。
    pub fn predict_batch(&self, image: &Image, face_boxes: &[Rect]) -> Result<Vec<Expression>> {
        if face_boxes.is_empty() {
            return Ok(Vec::new());
        }

        // 1. 裁剪 + 预处理（64x64 灰度 float 0~255，HWC → CHW 单平面）
        let crops: Vec<Image> = face_boxes
            .iter()
            .map(|&b| crop_and_resize(image, b, self.expand_ratio, 64, 64, true))
            .collect::<Result<_>>()?;

        // 2. 推理（优先打包批量；模型拒绝时退化为逐脸）
        match self.infer_packed(&crops) {
            Ok(output) => {
                let logits = output.as_f32()?;
                let n = face_boxes.len();
                if logits.len() < n * 8 {
                    return Err(VisionError::inference(format!(
                        "expression output size mismatch: got {} elements, expect >= {}",
                        logits.len(),
                        n * 8
                    )));
                }
                let width = logits.len() / n;
                let mut results = Vec::with_capacity(n);
                for k in 0..n {
                    results.push(decode_expression(&logits[k * width..(k + 1) * width]));
                }
                tracing::info!("ExpressionEngine inferred {} face(s) (batched)", results.len());
                Ok(results)
            }
            Err(batch_err) if crops.len() > 1 => {
                tracing::warn!(
                    "batched expression inference unavailable ({}), falling back to per-face loop",
                    batch_err
                );
                let mut results = Vec::with_capacity(crops.len());
                for crop in &crops {
                    let output = self.infer_packed(std::slice::from_ref(crop))?;
                    let logits = output.as_f32()?;
                    if logits.len() < 8 {
                        return Err(VisionError::inference(format!(
                            "expression output has {} elements, expect >= 8",
                            logits.len()
                        )));
                    }
                    results.push(decode_expression(&logits[..8]));
                }
                tracing::info!("ExpressionEngine inferred {} face(s) (loop)", results.len());
                Ok(results)
            }
            Err(e) => Err(e),
        }
    }

    /// 打包 batch 张量（`[N,1,64,64]` NCHW 灰度单平面）并推理。
    fn infer_packed(&self, crops: &[Image]) -> Result<TensorOutput> {
        let area = 64 * 64;
        let mut data = Vec::with_capacity(crops.len() * area);
        for crop in crops {
            if crop.width() != 64 || crop.height() != 64 || crop.channels() != 1 {
                return Err(VisionError::inference(format!(
                    "expression crop layout mismatch: got {}x{}x{}, expect 64x64x1",
                    crop.width(),
                    crop.height(),
                    crop.channels()
                )));
            }
            data.extend(crop.data().iter().map(|&v| v as f32));
        }
        let shape = vec![crops.len() as i64, 1, 64, 64];
        let tensor = Tensor::from_array((shape, data))?;
        self.base.run_inference(tensor)
    }
}

/// 由单行 logits 组装表情结果（softmax → argmax）。
fn decode_expression(logits: &[f32]) -> Expression {
    let scores = softmax(logits);
    let label_id = argmax(&scores);
    let label = EXPRESSION_LABELS
        .get(label_id)
        .copied()
        .unwrap_or("unknown");
    Expression {
        label: label.to_string(),
        label_id,
        scores,
    }
}

crate::impl_engine_forward!(ExpressionEngine, base, Expression,
    /// trait `predict` 不适用：表情推理必须提供人脸框（来自 YuNet 等检测器），
    /// 请使用 [`ExpressionEngine::predict_expr`] / [`ExpressionEngine::predict_batch`]。
    fn predict(&self, _image: &Image) -> Result<Expression> {
        Err(VisionError::Unsupported(
            "expression recognition needs a face box; use predict_expr / predict_batch instead"
                .to_string(),
        ))
    },
    /// trait `predict_batch`（整图批量）同理不支持：批量粒度为「单图 + 多人脸框」。
    fn predict_batch(&self, _images: &[Image]) -> Result<Vec<Expression>> {
        Err(VisionError::Unsupported(
            "expression recognition needs face boxes; use predict_batch(image, &[Rect]) instead"
                .to_string(),
        ))
    }
);
