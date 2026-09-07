//! Grounding DINO 开放词表目标检测引擎。
//!
//! 文本提示 + 图像 → 检测框（含类别名和置信度）。配合
//! [`Sam2Engine`](crate::engines::sam2::Sam2Engine) 即可组成 Grounded-SAM 流水线
//! （文本 → 检测 → 分割，见 [`crate::engines::grounded_sam::GroundedSamEngine`]）。
//!
//! # 模型准备
//!
//! ```text
//! # 从 Hugging Face 下载 onnx-community/grounding-dino-tiny-ONNX：
//! #   onnx/model.onnx      (~685 MB fp32)
//! #   tokenizer.json       (BERT WordPiece tokenizer)
//! #   vocab.txt
//! #   preprocessor_config.json
//! ```
//!
//! # ONNX I/O（onnx-community 实际导出，5 输入 + 2 输出）
//!
//! Inputs:
//!
//! - `pixel_values`:    [1, 3, 800, 800]  float32 — ImageNet 归一化，resize 到 800x800 正方形
//! - `input_ids`:       [1, seq_len]      int64   — BERT WordPiece + [CLS]...[SEP]（动态长度）
//! - `attention_mask`:  [1, seq_len]      int64   — 1=token, 0=pad
//! - `token_type_ids`:  [1, seq_len]      int64   — 全 0（单 prompt）
//! - `pixel_mask`:      [1, 800, 800]     int64   — 全 1（表示像素都有效）
//!
//! Outputs:
//!
//! - `logits`:     [1, 900, 256] float32 — sigmoid；class 255 = no-object
//! - `pred_boxes`: [1, 900, 4]   float32 — cxcywh，归一化 0-1
//!
//! # 示例
//!
//! ```ignore
//! let dino = GroundingDinoEngine::new(
//!     "/models/grounding_dino_tiny/grounding_dino_tiny.onnx",
//!     "/models/grounding_dino_tiny/tokenizer.json",
//!     DeviceType::Cpu,
//! )?;
//!
//! // 开放词表检测：类别用 "." 分隔（末尾可省略 "."，自动补）
//! let boxes = dino.predict_text(&image, "chair . table .")?;
//! // 或显式阈值
//! let boxes = dino.predict_text_with_thresholds(&image, "chair . table .", 0.25, 0.20)?;
//! ```

use ort::value::Tensor;
use tokenizers::{
    PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection,
    TruncationParams, TruncationStrategy,
};

use crate::core::base::BaseOnnxEngine;
use crate::core::device_type::DeviceType;
use crate::error::{Result, VisionError};
use crate::imaging::{cvt_color, resize, ColorConversion, Image, Interpolation};
use crate::model::Detection;

/// 默认检测框置信度阈值（对应上游 简化入口的 0.25）。
pub const DEFAULT_BOX_THRESHOLD: f32 = 0.25;
/// 默认文本匹配阈值（对应上游 简化入口的 0.20）。
pub const DEFAULT_TEXT_THRESHOLD: f32 = 0.20;

/// Grounding DINO 开放词表目标检测引擎（对应 `GroundingDinoEngine`）。
pub struct GroundingDinoEngine {
    /// 组合基类
    pub base: BaseOnnxEngine,

    /// BERT WordPiece 分词器（对应 `HuggingFaceTokenizer`；构造时配置
    /// max_length=256 截断 + pad 到定长，`encode` 时自动加 [CLS]/[SEP]，
    /// 默认 addSpecialTokens=true）。
    tokenizer: Tokenizer,
}

impl GroundingDinoEngine {
    /// 文本最大长度（对应 `MAX_TEXT_LEN`）。
    const MAX_TEXT_LEN: usize = 256;
    /// query 数量（对应 `NUM_QUERIES`）。
    const NUM_QUERIES: usize = 900;
    /// 图像输入边长（resize 到 800x800 正方形）。
    const IMAGE_SIZE: i32 = 800;
    /// ImageNet 归一化均值。
    const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    /// ImageNet 归一化标准差。
    const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

    // ---- BERT WordPiece 特殊 token id（上游实现魔数）----
    /// "." 的 BERT token id（phrase 分隔符）
    const BERT_DOT_ID: i64 = 1012;
    /// [CLS] token id
    const BERT_CLS_ID: i64 = 101;
    /// [SEP] token id
    const BERT_SEP_ID: i64 = 102;
    /// [PAD] token id
    const BERT_PAD_ID: i64 = 0;

    // ---- 模型 I/O 名称（onnx-community 导出）----
    const IN_PIXEL_VALUES: &'static str = "pixel_values";
    const IN_PIXEL_MASK: &'static str = "pixel_mask";
    const IN_INPUT_IDS: &'static str = "input_ids";
    const IN_ATTENTION_MASK: &'static str = "attention_mask";
    const IN_TOKEN_TYPE_IDS: &'static str = "token_type_ids";

    /// 创建引擎（对应 `GroundingDinoEngine(modelPath, tokenizerPath, deviceType)`）。
    ///
    /// - `model_path`: Grounding DINO ONNX 模型路径（输入固定 800x800）
    /// - `tokenizer_path`: `tokenizer.json` 路径（BERT WordPiece）
    pub fn new(
        model_path: impl AsRef<std::path::Path>,
        tokenizer_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
    ) -> Result<Self> {
        Self::with_config(model_path, tokenizer_path, device_type, Default::default())
    }

    /// 指定运行参数创建（线程数 / GPU 设备 id）。
    pub fn with_config(
        model_path: impl AsRef<std::path::Path>,
        tokenizer_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        runtime_config: crate::core::runtime_config::OnnxRuntimeConfig,
    ) -> Result<Self> {
        let base = BaseOnnxEngine::with_config(
            model_path,
            device_type,
            Self::IMAGE_SIZE,
            Self::IMAGE_SIZE,
            runtime_config,
        )?;
        let tokenizer = Self::build_tokenizer(tokenizer_path)?;

        tracing::info!(
            "Grounding DINO Engine initialized: model input='{}', outputs={}",
            base.input_name(),
            base.output_names().len()
        );

        Ok(GroundingDinoEngine { base, tokenizer })
    }

    /// 构建并配置 BERT tokenizer（max_length=256 截断 + pad 到固定 256，右填充）。
    fn build_tokenizer(tokenizer_path: impl AsRef<std::path::Path>) -> Result<Tokenizer> {
        // 对应上游：HuggingFaceTokenizer.builder()
        //     .optTokenizerPath(...).optMaxLength(256).optPadToMaxLength().build()
        let mut tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|e| VisionError::Tokenizer(format!("加载 tokenizer 失败: {e}")))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                direction: TruncationDirection::Right,
                max_length: Self::MAX_TEXT_LEN,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
            }))
            .map_err(|e| VisionError::Tokenizer(format!("配置 truncation 失败: {e}")))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(Self::MAX_TEXT_LEN),
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: 0, // [PAD] token id
            pad_type_id: 0,
            pad_token: "[PAD]".to_string(),
        }));
        Ok(tokenizer)
    }

    // ==================== Public API ====================

    /// 开放词表检测（对应上游 4 参 `predict(image, textPrompt, boxThreshold, textThreshold)`）。
    ///
    /// - `image`: 输入图像 (BGR)
    /// - `text_prompt`: 文本提示，类别用 "." 分隔（如 "chair . table ."）；末尾可省略 "."，自动补
    /// - `box_threshold`: 检测框置信度阈值（默认 0.25）
    /// - `text_threshold`: 文本匹配阈值（默认 0.20）
    pub fn predict_text_with_thresholds(
        &self,
        image: &Image,
        text_prompt: &str,
        box_threshold: f32,
        text_threshold: f32,
    ) -> Result<Vec<Detection>> {
        let orig_h = image.height() as i32;
        let orig_w = image.width() as i32;

        // 1. 图像预处理
        let pixel_data = self.preprocess_dino(image)?;

        // 2. 文本分词（自动补末尾 "."）
        let mut prompt = text_prompt.trim().to_string();
        if !prompt.ends_with('.') {
            prompt.push_str(" .");
        }
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| VisionError::Tokenizer(e.to_string()))?;
        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&v| v as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&v| v as i64)
            .collect();
        // 全 0（对应 `new long[inputIds.length]`，单 prompt 的 token_type_ids）
        let token_type_ids = vec![0i64; input_ids.len()];

        // 3. 推理
        let (logits, boxes) =
            self.run_inference(pixel_data, &input_ids, &attention_mask, &token_type_ids)?;

        // 4. 后处理
        self.parse_detections(
            &logits,
            &boxes,
            &input_ids,
            orig_h,
            orig_w,
            box_threshold,
            text_threshold,
        )
    }

    /// 简化入口（默认阈值 box=0.25 / text=0.20；对应上游 双参 `predict`）。
    pub fn predict_text(&self, image: &Image, text_prompt: &str) -> Result<Vec<Detection>> {
        self.predict_text_with_thresholds(
            image,
            text_prompt,
            DEFAULT_BOX_THRESHOLD,
            DEFAULT_TEXT_THRESHOLD,
        )
    }

    // ==================== 预处理 ====================

    /// Grounding DINO 图像预处理：resize 到 800x800 正方形 → BGR→RGB →
    /// ImageNet 归一化 → NCHW（对应 `preprocessDino`）。
    fn preprocess_dino(&self, image: &Image) -> Result<Vec<f32>> {
        let size = Self::IMAGE_SIZE as usize;

        // 输入统一到 3 通道 BGR（实现假定输入为 CV_8UC3）
        let bgr = match image.channels() {
            3 => image.clone(),
            4 => cvt_color(image, ColorConversion::Bgra2Bgr)?,
            _ => cvt_color(image, ColorConversion::Gray2Bgr)?,
        };

        // resize 到 800x800（OpenCV resize 默认双线性）
        let resized = resize(&bgr, size, size, Interpolation::Linear)?;

        // BGR → RGB
        let rgb = cvt_color(&resized, ColorConversion::Bgr2Rgb)?;

        // HWC → CHW with ImageNet normalize
        let area = size * size;
        let mut data = vec![0f32; 3 * area];
        let px = rgb.data();
        for i in 0..area {
            let r = px[i * 3] as f32 / 255.0;
            let g = px[i * 3 + 1] as f32 / 255.0;
            let b = px[i * 3 + 2] as f32 / 255.0;
            data[i] = (r - Self::IMAGENET_MEAN[0]) / Self::IMAGENET_STD[0];
            data[area + i] = (g - Self::IMAGENET_MEAN[1]) / Self::IMAGENET_STD[1];
            data[2 * area + i] = (b - Self::IMAGENET_MEAN[2]) / Self::IMAGENET_STD[2];
        }
        Ok(data)
    }

    // ==================== 推理 ====================

    /// 5 输入推理（
    /// lock 会话按名传入 —— 混合 f32/int64 输入无法走 `run_named`）。
    fn run_inference(
        &self,
        pixel_data: Vec<f32>,
        input_ids: &[i64],
        attention_mask: &[i64],
        token_type_ids: &[i64],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let seq_len = input_ids.len() as i64;
        let size = Self::IMAGE_SIZE as i64;

        // pixel_values [1, 3, 800, 800]
        let img_tensor = Tensor::from_array((vec![1, 3, size, size], pixel_data))?;

        // pixel_mask [1, 800, 800] int64 — 全 1
        let pixel_mask_arr = vec![1i64; (size * size) as usize];
        let pm_tensor = Tensor::from_array((vec![1, size, size], pixel_mask_arr))?;

        // input_ids [1, seq_len]
        let ids_tensor = Tensor::from_array((vec![1, seq_len], input_ids.to_vec()))?;

        // attention_mask [1, seq_len]
        let mask_tensor = Tensor::from_array((vec![1, seq_len], attention_mask.to_vec()))?;

        // token_type_ids [1, seq_len]
        let type_tensor = Tensor::from_array((vec![1, seq_len], token_type_ids.to_vec()))?;

        let mut session = self.base.session.lock().unwrap();
        let outputs = session.run(ort::inputs![
            Self::IN_PIXEL_VALUES  => img_tensor,
            Self::IN_PIXEL_MASK    => pm_tensor,
            Self::IN_INPUT_IDS     => ids_tensor,
            Self::IN_ATTENTION_MASK => mask_tensor,
            Self::IN_TOKEN_TYPE_IDS => type_tensor,
        ])?;

        // 按名称找 logits 和 pred_boxes（对应 toLowerCase().contains 匹配）
        let mut logits: Option<Vec<f32>> = None;
        let mut boxes: Option<Vec<f32>> = None;
        for name in self.base.output_names() {
            let lower = name.to_lowercase();
            if let Some(data) = snapshot_f32(&outputs, name)? {
                if lower.contains("logit") {
                    logits = Some(data);
                } else if lower.contains("box") {
                    boxes = Some(data);
                }
            }
        }

        match (logits, boxes) {
            (Some(l), Some(b)) => Ok((l, b)),
            _ => Err(VisionError::inference(format!(
                "Expected outputs 'logits' and 'pred_boxes'. Got: {:?}",
                self.base.output_names()
            ))),
        }
    }

    // ==================== 后处理 ====================

    /// 解析 logits/pred_boxes 为检测结果（对应 `parseDetections`）。
    #[allow(clippy::too_many_arguments)]
    fn parse_detections(
        &self,
        logits: &[f32],
        boxes: &[f32],
        input_ids: &[i64],
        orig_h: i32,
        orig_w: i32,
        box_threshold: f32,
        text_threshold: f32,
    ) -> Result<Vec<Detection>> {
        let mut results: Vec<Detection> = Vec::new();

        // 长度校验（上游实现越界抛异常，Rust 侧提前校验报错）
        if logits.len() < Self::NUM_QUERIES * Self::MAX_TEXT_LEN {
            return Err(VisionError::inference(format!(
                "logits size {} < {}x{}",
                logits.len(),
                Self::NUM_QUERIES,
                Self::MAX_TEXT_LEN
            )));
        }
        if boxes.len() < Self::NUM_QUERIES * 4 {
            return Err(VisionError::inference(format!(
                "pred_boxes size {} < {}x4",
                boxes.len(),
                Self::NUM_QUERIES
            )));
        }

        // logits shape: [900, 256]，每行是一个 query 对 256 个 token 的匹配分数
        // 最后一列（255）是 no-object
        for q in 0..Self::NUM_QUERIES {
            // 找最大激活的 token 位置（排除最后一个 no-object token）
            let mut max_idx = 0usize;
            let mut max_val = -1f32;
            for t in 0..Self::MAX_TEXT_LEN - 1 {
                let v = sigmoid(logits[q * Self::MAX_TEXT_LEN + t]);
                if v > max_val {
                    max_val = v;
                    max_idx = t;
                }
            }

            // 检测阈值过滤
            if max_val < box_threshold {
                continue;
            }
            //（上游的 maxIdx<0 / maxIdx>=MAX_TEXT_LEN 防御分支在此
            // usize 索引 + 下面 get() 校验下不可能命中，语义等价）

            // 验证该位置是真实 token（不是 PAD）；越界视为无效
            let token_id = match input_ids.get(max_idx) {
                Some(&id) => id,
                None => continue,
            };
            if token_id == Self::BERT_PAD_ID {
                continue; // PAD token id = 0
            }

            // 提取短语标签：从 maxIdx 往前找最近的 "."（phrase 边界），往后找下一个 "."
            let phrase = match self.extract_phrase(input_ids, max_idx) {
                Some(p) if !p.is_empty() => p,
                _ => continue,
            };

            // 文本阈值过滤（对短语本身的 token 取平均分数）
            if max_val < text_threshold {
                continue;
            }

            // box: cxcywh 归一化 → xyxy 像素坐标
            let cx = boxes[q * 4];
            let cy = boxes[q * 4 + 1];
            let bw = boxes[q * 4 + 2];
            let bh = boxes[q * 4 + 3];
            let mut x1 = (cx - bw / 2.0) * orig_w as f32;
            let mut y1 = (cy - bh / 2.0) * orig_h as f32;
            let mut x2 = (cx + bw / 2.0) * orig_w as f32;
            let mut y2 = (cy + bh / 2.0) * orig_h as f32;

            // clamp
            x1 = x1.max(0.0).min((orig_w - 1) as f32);
            y1 = y1.max(0.0).min((orig_h - 1) as f32);
            x2 = x2.max(0.0).min((orig_w - 1) as f32);
            y2 = y2.max(0.0).min((orig_h - 1) as f32);

            // 原版还计算 centerX/centerY 传入 Detection；Rust Detection 模型
            // 无对应字段，省略（对应 `new Detection(phrase, 0, bbox, cx, cy, maxVal)`）

            results.push(Detection::new(
                phrase,
                0,
                x1 as f64,
                y1 as f64,
                x2 as f64,
                y2 as f64,
                max_val as f64,
            ));
        }

        // 按置信度降序
        results.sort_by(|a, b| {
            b.confidence()
                .partial_cmp(&a.confidence())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(results)
    }

    /// 从 inputIds 提取短语：maxIdx 所在的 phrase（对应 `extractPhrase`）。
    ///
    /// Grounding DINO 用 "." 作 phrase 分隔符。
    fn extract_phrase(&self, input_ids: &[i64], max_idx: usize) -> Option<String> {
        // 往前找 "." 的下一个 token（phrase 开始）
        let mut start = max_idx;
        for i in (1..=max_idx).rev() {
            if input_ids[i] == Self::BERT_DOT_ID {
                // "." 的 BERT token id = 1012
                start = i + 1;
                break;
            }
            if input_ids[i] == Self::BERT_CLS_ID {
                // [CLS] = 101
                start = i + 1;
                break;
            }
            start = i;
        }
        // 往后找 "." 或 [SEP]（phrase 结束）
        let mut end = max_idx;
        for i in max_idx..input_ids.len() {
            if input_ids[i] == Self::BERT_DOT_ID
                || input_ids[i] == Self::BERT_SEP_ID
                || input_ids[i] == Self::BERT_PAD_ID
            {
                // "." [SEP] [PAD]
                end = i;
                break;
            }
            end = i + 1;
        }
        if start >= end {
            return None;
        }

        // 用 tokenizer.decode（对应上游 HuggingFaceTokenizer.decode，
        // 默认 skipSpecialTokens=false）
        let phrase_ids: Vec<u32> = input_ids[start..end].iter().map(|&v| v as u32).collect();
        match self.tokenizer.decode(&phrase_ids, false) {
            Ok(s) => Some(s.trim().to_string()),
            // Fallback: 简单字符串（对应上游 catch Exception → "object"）
            Err(_) => Some("object".to_string()),
        }
    }

    // 注：原版 close() 由基类释放会话；Rust 侧 ort Session / Tokenizer 均为
    // RAII（Drop 自动释放），无需显式 close。
}

/// sigmoid（对应 `GroundingDinoEngine.sigmoid`）。
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// 从会话输出按名取 f32 快照；非 f32 张量返回 `None`（跳过，对齐 上游实现
/// 不做类型校验的宽容行为；`snapshot_output` 语义的简化版）。
fn snapshot_f32(
    outputs: &ort::session::SessionOutputs<'_>,
    name: &str,
) -> Result<Option<Vec<f32>>> {
    let value = outputs
        .get(name)
        .ok_or_else(|| VisionError::inference(format!("missing output '{name}'")))?;
    match value.dtype() {
        ort::value::ValueType::Tensor {
            ty: ort::value::TensorElementType::Float32,
            ..
        } => {
            let (_, view) = value.try_extract_tensor::<f32>()?;
            Ok(Some(view.to_vec()))
        }
        _ => Ok(None),
    }
}

crate::impl_engine_forward!(GroundingDinoEngine, base, Vec<Detection>,
    /// 单图推理。开放词表检测需要文本提示，不支持整图 `predict`
    /// （对应上游 抛出 `UnsupportedOperationException`）。
    fn predict(&self, _image: &Image) -> Result<Vec<Detection>> {
        Err(VisionError::Unsupported(
            "Use predict_text/predict_text_with_thresholds instead".to_string(),
        ))
    },
    /// 批量推理同理不支持（对应 `predictBatch` 抛出异常）。
    fn predict_batch(&self, _images: &[Image]) -> Result<Vec<Vec<Detection>>> {
        Err(VisionError::Unsupported(
            "Use predict_text/predict_text_with_thresholds instead".to_string(),
        ))
    }
);
