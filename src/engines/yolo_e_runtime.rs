//! YOLOE 运行时视觉提示引擎（实验性）- 双模型架构。
//!
//! 与烘焙方式（[`crate::engines::segmentation::SegmentationEngine`] 承接的 YOLOE 导出模型）不同，
//! 本引擎把 YOLOE 拆成两个 ONNX，实现"换提示不用重新导出"：
//!
//! - **提示编码器**（由 export_yoloe.py --mode split 导出）：
//!   输入 (参考图 images[1,3,S,S], 提示掩码 prompt_masks[1,C,S/8,S/8]) → 类别嵌入 pe[1,C,512]。
//!   参考图 + 框在引擎侧前处理，每次 [`YoloERuntimeEngine::set_visual_prompts`] 重算一次 pe 并缓存；
//!   多张参考图时按类别聚合 embedding。
//! - **pe 输入检测器**：输入 (待检图 images[1,3,S,S], pe[1,C,512])
//!   → 输出 output0 [1, 4+C+32, anchors]（或 26 系列 End2End [1, 300, 4+1+1+32]）
//!   以及 output1 proto [1, 32, S/4, S/4]。引擎用 32 维掩码系数与 proto 解码实例掩码，
//!   返回 [`Segmentation`]（掩码已还原到原图坐标；旧版无 proto 的检测器返回 mask=None，兼容加载）。
//!
//! **使用示例**：
//! ```ignore
//! let engine = YoloERuntimeEngine::new(
//!     "models/yoloe_rt_encoder.onnx", "models/yoloe_rt_pe_detector.onnx", DeviceType::Cpu)?;
//! // 参考图上给 "person" 类别画框（可多框，坐标为参考图像素坐标）
//! let prompts = vec![
//!     ("person".to_string(), vec![vec![114.0, 197.0, 1114.0, 712.0]]),
//! ];
//! engine.set_visual_prompts(&reference_image, prompts)?;
//! let results = engine.predict(&image)?;
//! // 大图小目标：一行开启 SAHI 切片推理（返回类型不变，掩码自动跨切片平移/合并）
//! engine.set_sahi_config(Some(SahiConfig::of(512, 512, 0.2, 0.2)));
//! ```
//!
//! **注意事项**（生产环境请优先考虑烘焙方式，见 export_yoloe.py 头注释）：
//! - **分数与烘焙路径接近**（同一提示实测 0.883 vs 0.914，差异 ~0.03），
//!   阈值可与烘焙方式通用；引擎默认置信度阈值 0.01（偏保守，避免漏检，
//!   实际使用建议按效果上调）；
//! - **提示框请画紧**：单框覆盖多个目标会稀释嵌入质量（实测松散大框分数骤降），
//!   建议一个目标一个提示；
//! - **支持不规则多边形提示**：提示形状除矩形 [x1,y1,x2,y2] 外，
//!   还可以是多边形顶点平铺 [x1,y1,...,xn,yn]（≥3 点），引擎栅格化为掩码后编码，
//!   适合目标轮廓不规则、矩形框会带入大量背景的场景；
//! - 提示掩码分辨率 = imgsz/8（savpe 特征分辨率），由本引擎自动构建；
//! - 每条参考图都会重跑一次编码器（含完整主干网络），多张参考图的编码结果按类别平均；
//!   批量检测同一批提示时只需调用一次 `set_visual_prompts`；
//! - 原版实例非线程安全；Rust 侧提示缓存等内部可变字段用 `Mutex` 承载，
//!   letterbox 参数（ratio/dw/dh）按本 crate 惯例改为随调用链显式传递。
//!
//! **与 原版的实现差异**（数值逻辑保持逐行一致）：
//! - 26 系列旧版导出可能产生 DOUBLE 张量（elemType==11 分支）；
//!   legacy DOUBLE（elemType==11）输出由快照层统一转 f32 兼容；
//! - SAHI 切片推理已接入（`predict` 内部走 `predict_sliced_segmentation`）。

use std::collections::HashMap;
use std::sync::Mutex;

use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::core::base::{BaseOnnxEngine, TensorData, TensorOutput};
use crate::core::device_type::DeviceType;
use crate::error::{Result, VisionError};
use crate::imaging::{cvt_color, resize, ColorConversion, FloatMask, Image, Interpolation};
use crate::model::Segmentation;
use crate::sahi::config::SahiConfig;

/// 掩码系数维度（ultralytics seg 头固定 32；对应 `MASK_COEFFS`）。
const MASK_COEFFS: usize = 32;

/// 候选向量长度：`[x1,y1,x2,y2,conf,cls, c0..c31]`。
const CANDIDATE_LEN: usize = 6 + MASK_COEFFS;

/// letterbox 参数（对应上游 实例字段 ratio/dw/dh；Rust 侧随单次推理显式传递）。
#[derive(Debug, Clone, Copy)]
struct LetterboxParams {
    /// 缩放比例（min(imgsz/orig_h, imgsz/orig_w)）
    ratio: f32,
    /// 左右对称填充的一半宽度
    dw: f32,
    /// 上下对称填充的一半高度
    dh: f32,
}

impl LetterboxParams {
    /// 内容区宽 = imgsz - 2*pad，除以缩放比还原原图宽（对应 `w0()`）。
    fn content_width(&self, imgsz: usize) -> i32 {
        ((imgsz as f32 - 2.0 * self.dw) / self.ratio).round() as i32
    }

    /// 内容区高（对应 `h0()`）。
    fn content_height(&self, imgsz: usize) -> i32 {
        ((imgsz as f32 - 2.0 * self.dh) / self.ratio).round() as i32
    }
}

/// 掩码原型（对应 `protoAsMat` 生成的 `[C, H*W]` CV_32F Mat 布局，gemm 用；
/// CHW 连续布局直接按行落位）。
#[derive(Debug, Clone)]
struct ProtoMasks {
    /// 原型通道数（标准为 32）
    dim: usize,
    /// 原型掩码高
    height: usize,
    /// 原型掩码宽
    width: usize,
    /// 行主序数据：`data[c * h * w + y * w + x]`
    data: Vec<f32>,
}

/// 一条视觉提示参考：一张参考图，以及该图上每个类别对应的框。
///
/// 同一类别在同一张图上的多个框会合并成一个提示掩码；如果同一类别出现在多张
/// 参考图中，则每张图分别编码，最后对该类别的多个 embedding 求平均并重新归一化。
///
/// 提示形状（对应 `float[]`）：
/// - 矩形框：`[x1, y1, x2, y2]`
/// - 多边形：`[x1, y1, ..., xn, yn]`（顶点平铺，≥3 点）
pub struct VisualPromptReference<'a> {
    /// 参考图（任意尺寸，内部自动 letterbox）
    pub image: &'a Image,
    /// 类别名 → 该类别的提示形状列表（顺序 = 类别 id 顺序；每类多框会合并为一张提示掩码）
    pub class_prompts: Vec<(String, Vec<Vec<f32>>)>,
}

impl<'a> VisualPromptReference<'a> {
    /// 创建一条视觉提示参考。
    pub fn new(image: &'a Image, class_prompts: Vec<(String, Vec<Vec<f32>>)>) -> Self {
        VisualPromptReference {
            image,
            class_prompts,
        }
    }
}

/// 对一张参考图编码的结果：每个类别一个 embedding（对应上游内部
/// record `EncodedVisualPrompts`）。
struct EncodedVisualPrompts {
    labels: Vec<String>,
    values: Vec<f32>,
    dim: usize,
}

impl EncodedVisualPrompts {
    /// 取第 `class_index` 个类别的 embedding（对应 `embedding(classIndex)`）。
    fn embedding(&self, class_index: usize) -> &[f32] {
        &self.values[class_index * self.dim..(class_index + 1) * self.dim]
    }
}

/// 同名类别跨参考图的 embedding 累加器（对应上游内部类 `PromptAccumulator`）。
struct PromptAccumulator {
    sum: Vec<f64>,
    count: usize,
}

impl PromptAccumulator {
    fn new(dim: usize) -> Self {
        PromptAccumulator {
            sum: vec![0.0; dim],
            count: 0,
        }
    }

    fn add(&mut self, embedding: &[f32]) -> Result<()> {
        if embedding.len() != self.sum.len() {
            return Err(VisionError::invalid_argument("视觉 embedding 维度不一致"));
        }
        for (i, &v) in embedding.iter().enumerate() {
            self.sum[i] += v as f64;
        }
        self.count += 1;
        Ok(())
    }

    /// 求算术平均后做 L2 归一化（对应 `meanAndNormalize`）。
    // `!(norm > 1e-12)` 保留上游逐位语义：NaN 时同样报错（`norm <= 1e-12` 对 NaN 为 false）
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn mean_and_normalize(&self) -> Result<Vec<f32>> {
        if self.count == 0 {
            return Err(VisionError::invalid_argument("视觉提示类别没有有效 embedding"));
        }
        let mut result = vec![0f32; self.sum.len()];
        let mut squared_norm = 0.0f64;
        for (i, &s) in self.sum.iter().enumerate() {
            result[i] = (s / self.count as f64) as f32;
            squared_norm += result[i] as f64 * result[i] as f64;
        }
        let norm = squared_norm.sqrt();
        if !(norm > 1e-12) || !norm.is_finite() {
            return Err(VisionError::invalid_argument("视觉提示 embedding 范数无效"));
        }
        for v in result.iter_mut() {
            *v /= norm as f32;
        }
        Ok(result)
    }
}

/// 视觉提示缓存（对应 pe/peClasses/peDim/classNames 实例字段；
/// `set_visual_prompts` 成功后整体替换）。
#[derive(Debug, Clone, Default)]
struct VisualPrompts {
    /// 类别名（顺序 = 类别 id，由 set_visual_prompts 的类别顺序决定）
    class_names: Vec<String>,
    /// 缓存的类别嵌入 pe[1, C, D]
    pe: Vec<f32>,
    pe_classes: usize,
    pe_dim: usize,
}

/// YOLOE 运行时视觉提示引擎（实验性，双模型架构）。
pub struct YoloERuntimeEngine {
    /// 提示编码器（参考图 + 提示掩码 → pe 类别嵌入）
    encoder: BaseOnnxEngine,
    /// pe 输入检测器（待检图 + pe → output0 [+ output1 proto]）
    detector: BaseOnnxEngine,

    encoder_images_name: String,
    encoder_masks_name: String,
    detector_images_name: String,
    detector_pe_name: String,

    /// proto 输出名（output1）；旧版仅导出 output0 的检测器为 None，此时掩码不解码
    detector_proto_name: Option<String>,

    /// 检测器输入空间尺寸（savpe 掩码分辨率 = imgsz/8）
    imgsz: usize,
    mask_size: usize,

    /// NMS IoU 阈值（默认 0.65）
    nms_threshold: f32,
    /// 置信度阈值（默认 0.01 偏保守，避免漏检；可按效果上调）
    confidence_threshold: f32,

    /// SAHI 切片推理配置；None（默认）表示关闭。设置后 predict 内部自动切片推理，返回类型不变
    sahi_config: Option<SahiConfig>,

    /// 视觉提示缓存（类别名 + pe 嵌入）；对应 可变实例字段，用 Mutex 承载内部可变性
    prompts: Mutex<Option<VisualPrompts>>,

    /// 文本编码器（可选，attach_text_encoder 挂载）：token_ids [C,77] → features [C,512]
    /// （已 L2 归一化；由 YOLOE 的 mobileclip 文本塔导出）
    text_encoder: Option<BaseOnnxEngine>,
    /// 文本辅助头（可选）：features [1,C,512] → pe [1,C,512]（reprta + L2 归一化）
    tpe_head: Option<BaseOnnxEngine>,
    /// CLIP BPE 分词器（可选）：与 YOLOE text_model 的 tokenize 逐位一致
    /// （SOT 49406 / EOS 49407 / pad 0，上下文 77）
    clip_tokenizer: Option<Tokenizer>,
}

impl YoloERuntimeEngine {
    /// 创建 YOLOE 运行时引擎（对应上游 构造器）。
    pub fn new(
        encoder_path: impl AsRef<std::path::Path>,
        detector_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
    ) -> Result<Self> {
        let encoder_path = encoder_path.as_ref();
        let detector_path = detector_path.as_ref();
        let encoder = BaseOnnxEngine::new(encoder_path, device_type)?;
        let detector = BaseOnnxEngine::new(detector_path, device_type)?;

        // 输入名与尺寸（编码器/检测器各应有两个输入；对应 assert，此处返回错误）
        let enc_inputs = session_input_names(&encoder);
        if enc_inputs.len() != 2 {
            return Err(VisionError::inference(format!(
                "编码器应有两个输入, 实际 {} 个: {:?}",
                enc_inputs.len(),
                enc_inputs
            )));
        }
        let det_inputs = session_input_names(&detector);
        if det_inputs.len() != 2 {
            return Err(VisionError::inference(format!(
                "检测器应有两个输入, 实际 {} 个: {:?}",
                det_inputs.len(),
                det_inputs
            )));
        }
        let encoder_images_name = enc_inputs[0].clone();
        let encoder_masks_name = enc_inputs[1].clone();
        let detector_images_name = det_inputs[0].clone();
        let detector_pe_name = det_inputs[1].clone();

        // 对应上游 inputSpatialSize(detectorSession)：取输入 shape[2]，动态维度回退 640
        // （BaseOnnxEngine 已做同样解析：dims[2] > 0 时取之，否则 640）
        let imgsz = detector.input_height() as usize;
        let mask_size = imgsz / 8; // savpe 掩码分辨率 = stride-8 特征分辨率

        // proto 输出（output1）：存在则解码实例掩码；旧版单输出检测器返回 mask=None
        let detector_proto_name = detector.output_names().get(1).cloned();

        tracing::info!(
            "YoloERuntimeEngine 初始化: encoder={}, detector={}, imgsz={}, maskSize={}, proto={}",
            encoder_path.display(),
            detector_path.display(),
            imgsz,
            mask_size,
            if detector_proto_name.is_some() {
                "output1（掩码解码开启）"
            } else {
                "无（旧版导出，mask=null）"
            }
        );

        Ok(YoloERuntimeEngine {
            text_encoder: None,
            tpe_head: None,
            clip_tokenizer: None,
            encoder,
            detector,
            encoder_images_name,
            encoder_masks_name,
            detector_images_name,
            detector_pe_name,
            detector_proto_name,
            imgsz,
            mask_size,
            nms_threshold: 0.65,
            confidence_threshold: 0.01,
            sahi_config: None,
            prompts: Mutex::new(None),
        })
    }

    // ==================== 访问器（对应上游 @Getter/@Setter） ====================

    /// NMS IoU 阈值。
    pub fn nms_threshold(&self) -> f32 {
        self.nms_threshold
    }

    /// 设置 NMS IoU 阈值。
    pub fn set_nms_threshold(&mut self, threshold: f32) {
        self.nms_threshold = threshold;
    }

    /// 置信度阈值。
    pub fn confidence_threshold(&self) -> f32 {
        self.confidence_threshold
    }

    /// 设置置信度阈值。
    pub fn set_confidence_threshold(&mut self, threshold: f32) {
        self.confidence_threshold = threshold;
    }

    /// SAHI 是否已开启。
    pub fn is_sahi_enabled(&self) -> bool {
        self.sahi_config.is_some()
    }

    /// SAHI 切片推理配置。
    pub fn sahi_config(&self) -> Option<&SahiConfig> {
        self.sahi_config.as_ref()
    }

    /// 开启 SAHI 切片推理（大图小目标；掩码自动跨切片平移/并集合并）。
    pub fn set_sahi_config(&mut self, sahi_config: Option<SahiConfig>) {
        self.sahi_config = sahi_config;
    }

    /// 关闭 SAHI 切片推理。
    pub fn disable_sahi(&mut self) {
        self.sahi_config = None;
    }

    /// 获取当前视觉提示的类别名（顺序 = 类别 id；对应 `getLabels`）。
    pub fn get_labels(&self) -> Vec<String> {
        self.prompts
            .lock()
            .unwrap()
            .as_ref()
            .map(|p| p.class_names.clone())
            .unwrap_or_default()
    }

    // ==================== 视觉提示 ====================

    /// 设置单张参考图的视觉提示。
    ///
    /// 这是兼容旧版本的便捷重载，语义等同于传入只有一条元素的
    /// [`YoloERuntimeEngine::set_visual_prompts_multi`]。
    ///
    /// - `reference_image`: 参考图（任意尺寸，内部自动 letterbox）
    /// - `class_prompts`: 类别名 → 该类别的框/多边形提示列表
    ///   （顺序 = 类别 id 顺序；每类多框会合并为一张提示掩码）
    pub fn set_visual_prompts(
        &self,
        reference_image: &Image,
        class_prompts: Vec<(String, Vec<Vec<f32>>)>,
    ) -> Result<()> {
        let references = [VisualPromptReference::new(reference_image, class_prompts)];
        self.set_visual_prompts_multi(&references)
    }

    /// 设置多张参考图的视觉提示（对应 `setVisualPrompts(List)`）。
    ///
    /// 每张参考图会独立运行一次提示编码器。对于同名类别，先将各参考图得到的
    /// embedding 求算术平均，再做 L2 归一化，生成检测器使用的最终类别 embedding。
    /// 因此，多张参考图可以共同定义一个类别，而不是后一次调用覆盖前一次调用。
    ///
    /// 类别顺序取各参考图中首次出现的顺序；每张图中类别顺序取其 `class_prompts`
    /// 的迭代顺序，因此需要稳定类别 id 时请保持传入顺序稳定（对应 
    /// `LinkedHashMap` 约定）。
    ///
    /// # Errors
    /// 参数为空、框格式无效或各参考 embedding 维度不一致时返回
    /// [`VisionError::InvalidArgument`]（对应 `IllegalArgumentException`）。
    pub fn set_visual_prompts_multi(&self, references: &[VisualPromptReference<'_>]) -> Result<()> {
        Self::validate_references(references)?;

        let mut label_order: Vec<String> = Vec::new();
        let mut accumulators: HashMap<String, PromptAccumulator> = HashMap::new();
        let mut embedding_dim: Option<usize> = None;

        // 先完整计算，成功后再替换缓存，避免中途失败时破坏上一套可用提示。
        for reference in references {
            let encoded = self.encode_visual_prompts(reference.image, &reference.class_prompts)?;
            match embedding_dim {
                None => embedding_dim = Some(encoded.dim),
                Some(d) if d != encoded.dim => {
                    return Err(VisionError::invalid_argument(format!(
                        "不同参考图得到的视觉 embedding 维度不一致: {} vs {}",
                        d, encoded.dim
                    )));
                }
                _ => {}
            }

            for i in 0..encoded.labels.len() {
                let label = encoded.labels[i].clone();
                if !label_order.contains(&label) {
                    label_order.push(label.clone());
                }
                let dim = embedding_dim.expect("embedding_dim 已在上方初始化");
                let accumulator = accumulators
                    .entry(label)
                    .or_insert_with(|| PromptAccumulator::new(dim));
                accumulator.add(encoded.embedding(i))?;
            }
        }

        let embedding_dim = embedding_dim.unwrap_or(0);
        if label_order.is_empty() || embedding_dim == 0 {
            return Err(VisionError::invalid_argument("至少需要一个有效的视觉提示类别"));
        }

        let mut combined_pe = vec![0f32; label_order.len() * embedding_dim];
        for (class_index, label) in label_order.iter().enumerate() {
            let normalized = accumulators[label].mean_and_normalize()?;
            combined_pe[class_index * embedding_dim..(class_index + 1) * embedding_dim]
                .copy_from_slice(&normalized);
        }

        let prompts = VisualPrompts {
            class_names: label_order.clone(),
            pe_classes: label_order.len(),
            pe_dim: embedding_dim,
            pe: combined_pe,
        };
        *self.prompts.lock().unwrap() = Some(prompts);
        tracing::info!(
            "视觉提示已更新: references={}, classes={:?}, peDim={}",
            references.len(),
            label_order,
            embedding_dim
        );
        Ok(())
    }

    /// 挂载文本提示支持（YOLOE 文本编码器 + reprta 辅助头 + CLIP BPE 分词器）。
    ///
    /// 模型由 scripts 导出：`yoloe_text_encoder.onnx`（token_ids [C,77] → features [C,512]，
    /// 归一化内嵌）与 `yoloe_tpe_head.onnx`（features [1,C,512] → pe [1,C,512]，reprta+归一化），
    /// 分词器为 CLIP BPE（SOT 49406 / EOS 49407 / pad 0，上下文 77），与 YOLOE
    /// text_model 的 tokenize 逐位一致。挂载后可用 [`Self::set_text_prompts`] 运行时
    /// 更换文本类别，与视觉提示共用同一 pe 检测器。
    pub fn attach_text_encoder(
        &mut self,
        encoder_path: impl AsRef<std::path::Path>,
        tpe_head_path: impl AsRef<std::path::Path>,
        tokenizer_path: impl AsRef<std::path::Path>,
    ) -> Result<()> {
        let mut tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|e| VisionError::Tokenizer(format!("加载 CLIP 分词器失败: {e}")))?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                direction: tokenizers::TruncationDirection::Right,
                max_length: 77,
                strategy: tokenizers::TruncationStrategy::LongestFirst,
                stride: 0,
            }))
            .map_err(|e| VisionError::Tokenizer(format!("配置 truncation 失败: {e}")))?;
        // clip.tokenize 的填充为 0（非真实 token），上下文固定 77
        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(77),
            direction: tokenizers::PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: 0,
            pad_type_id: 0,
            pad_token: "[PAD]".to_string(),
        }));

        let encoder = BaseOnnxEngine::new(encoder_path, self.detector.device_type())?;
        let tpe_head = BaseOnnxEngine::new(tpe_head_path, self.detector.device_type())?;
        self.text_encoder = Some(encoder);
        self.tpe_head = Some(tpe_head);
        self.clip_tokenizer = Some(tokenizer);
        Ok(())
    }

    /// 运行时设置**文本提示**类别（与 [`Self::set_visual_prompts`] 互斥使用，后设者生效）。
    ///
    /// 流程：CLIP BPE 分词（77 上下文）→ 文本编码器逐条编码 → 拼成 [1,C,512] →
    /// reprta 辅助头 + L2 归一化 → pe。之后 `predict`/`predict_without_sahi` 直接
    /// 使用该 pe（与视觉提示同一下游）。
    pub fn set_text_prompts(&self, names: &[&str]) -> Result<()> {
        let (encoder, tpe_head, tokenizer) = match (
            &self.text_encoder,
            &self.tpe_head,
            &self.clip_tokenizer,
        ) {
            (Some(e), Some(h), Some(t)) => (e, h, t),
            _ => {
                return Err(VisionError::invalid_argument(
                    "未挂载文本编码器：请先调用 attach_text_encoder(encoder, tpe_head, tokenizer)",
                ))
            }
        };
        if names.is_empty() {
            return Err(VisionError::invalid_argument("至少需要一个文本提示类别"));
        }

        // 逐条编码（文本塔很小，逐条开销可忽略；编码器为静态 [1,77] 批）
        let mut dim = 0usize;
        let mut feats: Vec<f32> = Vec::with_capacity(names.len() * 512);
        for name in names {
            let enc = tokenizer
                .encode(*name, true)
                .map_err(|e| VisionError::Tokenizer(format!("分词失败: {e}")))?;
            let ids: Vec<i32> = enc.get_ids().iter().map(|&v| v as i32).collect();
            if ids.len() != 77 {
                return Err(VisionError::Tokenizer(format!(
                    "token 长度 {} != 77（分词器填充配置错误）",
                    ids.len()
                )));
            }
            // 文本编码器输入为 int64，走 session 直跑（run_inference 仅支持 f32）
            let out = {
                let mut sess = encoder.session.lock().unwrap();
                let outputs = sess.run(ort::inputs![encoder.input_name() => Tensor::from_array((vec![1, 77], ids))?])?;
                let (_, view) = outputs[0].try_extract_tensor::<f32>()?;
                view.to_vec()
            };
            let data: &[f32] = &out;
            if dim == 0 {
                dim = data.len();
                if dim == 0 {
                    return Err(VisionError::inference("文本编码器输出为空"));
                }
            } else if data.len() != dim {
                return Err(VisionError::inference(format!(
                    "文本编码器输出维度不一致: {} != {dim}",
                    data.len()
                )));
            }
            feats.extend_from_slice(data);
        }

        // [1,C,D] → reprta 头 + 归一化 → pe [1,C,D]
        let pe = tpe_head.run_inference(Tensor::from_array((
            vec![1, names.len() as i64, dim as i64],
            feats,
        ))?)?;
        let pe_data = pe.as_f32()?.to_vec();

        let prompts = VisualPrompts {
            class_names: names.iter().map(|s| s.to_string()).collect(),
            pe_classes: names.len(),
            pe_dim: dim,
            pe: pe_data,
        };
        *self.prompts.lock().unwrap() = Some(prompts);
        tracing::info!("文本提示已更新: classes={:?}, peDim={}", names, dim);
        Ok(())
    }

    fn validate_references(references: &[VisualPromptReference<'_>]) -> Result<()> {
        if references.is_empty() {
            return Err(VisionError::invalid_argument("至少需要一张参考图"));
        }
        for (i, reference) in references.iter().enumerate() {
            if reference.image.is_empty() {
                return Err(VisionError::invalid_argument(format!(
                    "参考图不能为空: index={i}"
                )));
            }
            Self::validate_class_prompts(&reference.class_prompts, i)?;
        }
        Ok(())
    }

    fn validate_class_prompts(
        class_prompts: &[(String, Vec<Vec<f32>>)],
        reference_index: usize,
    ) -> Result<()> {
        if class_prompts.is_empty() {
            return Err(VisionError::invalid_argument(format!(
                "参考图至少需要一个类别提示: index={reference_index}"
            )));
        }
        for (label, shapes) in class_prompts {
            if label.trim().is_empty() {
                return Err(VisionError::invalid_argument(format!(
                    "视觉提示类别名不能为空: referenceIndex={reference_index}"
                )));
            }
            if shapes.is_empty() {
                return Err(VisionError::invalid_argument(format!(
                    "类别至少需要一个提示形状: {label}"
                )));
            }
            for shape in shapes {
                Self::validate_prompt_shape(shape, label)?;
            }
        }
        Ok(())
    }

    /// 校验单个提示形状：`[x1,y1,x2,y2]`（矩形框）或 `[x1,y1,...,xn,yn]`
    /// （多边形顶点平铺，≥3 点），坐标必须为有限数值，框需满足 x2>x1 且 y2>y1。
    fn validate_prompt_shape(shape: &[f32], label: &str) -> Result<()> {
        for &coordinate in shape {
            if !coordinate.is_finite() {
                return Err(VisionError::invalid_argument(format!(
                    "提示坐标必须是有限数值: {label}"
                )));
            }
        }
        if shape.len() == 4 {
            if shape[2] <= shape[0] || shape[3] <= shape[1] {
                return Err(VisionError::invalid_argument(format!(
                    "框必须满足 x2>x1 且 y2>y1: {label}"
                )));
            }
        } else if shape.len() >= 6 && shape.len().is_multiple_of(2) {
            // 多边形：≥3 个顶点（x,y 平铺）
        } else {
            return Err(VisionError::invalid_argument(format!(
                "提示形状应为 [x1,y1,x2,y2]（框）或 [x1,y1,...,xn,yn]（多边形，≥3 点）: {label}"
            )));
        }
        Ok(())
    }

    /// 对一张参考图编码，每个类别输出一个 embedding（对应 `encodeVisualPrompts`）。
    fn encode_visual_prompts(
        &self,
        reference_image: &Image,
        class_prompts: &[(String, Vec<Vec<f32>>)],
    ) -> Result<EncodedVisualPrompts> {
        let labels: Vec<String> = class_prompts.iter().map(|(l, _)| l.clone()).collect();

        // 1. 参考图 letterbox
        let h0 = reference_image.height();
        let w0 = reference_image.width();
        let reference_ratio = (self.imgsz as f32 / h0 as f32).min(self.imgsz as f32 / w0 as f32);
        let new_w = (w0 as f32 * reference_ratio).round() as usize;
        let new_h = (h0 as f32 * reference_ratio).round() as usize;
        let reference_dw = (self.imgsz as f32 - new_w as f32) / 2.0;
        let reference_dh = (self.imgsz as f32 - new_h as f32) / 2.0;
        let ref_tensor = self.to_chw(
            reference_image,
            new_w,
            new_h,
            reference_dw.round() as i32,
            reference_dh.round() as i32,
        )?;

        // 2. 提示形状 → 掩码 [1, C, S/8, S/8]
        //    每个提示（矩形 [x1,y1,x2,y2] 或 多边形 [x1,y1,...,xn,yn]）填充为该类别
        //    掩码中的一个区域；同类多个提示取并集。掩码分辨率 = 特征分辨率（S/8）。
        let num_classes = labels.len();
        let mask_size = self.mask_size;
        let mut masks = vec![0f32; num_classes * mask_size * mask_size];
        let mask_scale = mask_size as f32 / self.imgsz as f32;
        for (c, (_label, shapes)) in class_prompts.iter().enumerate() {
            for shape in shapes {
                if shape.len() == 4 {
                    // 矩形框：letterbox 变换到 imgsz 画布后，再缩放到掩码分辨率
                    let x1 = (((shape[0] * reference_ratio + reference_dw) * mask_scale).floor())
                        .max(0.0) as i32;
                    let y1 = (((shape[1] * reference_ratio + reference_dh) * mask_scale).floor())
                        .max(0.0) as i32;
                    let x2 = (((shape[2] * reference_ratio + reference_dw) * mask_scale).ceil())
                        .min(mask_size as f32) as i32;
                    let y2 = (((shape[3] * reference_ratio + reference_dh) * mask_scale).ceil())
                        .min(mask_size as f32) as i32;
                    for y in y1..y2 {
                        let row_offset = c * mask_size * mask_size + y as usize * mask_size;
                        for x in x1..x2 {
                            masks[row_offset + x as usize] = 1.0;
                        }
                    }
                } else {
                    // 多边形：顶点做同样的坐标变换，然后逐格奇偶规则填充
                    let n = shape.len() / 2;
                    let mut xs = vec![0f64; n];
                    let mut ys = vec![0f64; n];
                    for i in 0..n {
                        xs[i] =
                            ((shape[2 * i] * reference_ratio + reference_dw) * mask_scale) as f64;
                        ys[i] = ((shape[2 * i + 1] * reference_ratio + reference_dh) * mask_scale)
                            as f64;
                    }
                    let class_offset = c * mask_size * mask_size;
                    for y in 0..mask_size {
                        for x in 0..mask_size {
                            if point_in_polygon(x as f64 + 0.5, y as f64 + 0.5, &xs, &ys) {
                                masks[class_offset + y * mask_size + x] = 1.0;
                            }
                        }
                    }
                }
            }
        }

        // 3. 编码器推理 → pe
        let image_shape = vec![1i64, 3, self.imgsz as i64, self.imgsz as i64];
        let mask_shape = vec![1i64, num_classes as i64, mask_size as i64, mask_size as i64];
        let image_tensor = Tensor::from_array((image_shape, ref_tensor))?;
        let mask_tensor = Tensor::from_array((mask_shape, masks))?;

        let outputs = self.encoder.run_named(vec![
            (self.encoder_images_name.as_str(), image_tensor),
            (self.encoder_masks_name.as_str(), mask_tensor),
        ])?;
        // 按名查找 "pe" 输出，兜底取第一个输出
        let pe_out = outputs
            .iter()
            .find(|o| o.name == "pe")
            .or_else(|| outputs.first())
            .ok_or_else(|| VisionError::inference("编码器没有输出"))?;
        let shape = &pe_out.shape;
        if shape.len() != 3 || shape[0] != 1 || shape[1] != num_classes as i64 || shape[2] <= 0 {
            return Err(VisionError::inference(format!(
                "编码器输出 pe 形状异常: {:?}, expected [1,{},D]",
                shape, num_classes
            )));
        }
        let dim = shape[2] as usize;
        let flat = require_f32_data(pe_out, "编码器 pe")?;
        if flat.len() < num_classes * dim {
            return Err(VisionError::inference(format!(
                "编码器 pe 元素数 {} < {}x{}",
                flat.len(),
                num_classes,
                dim
            )));
        }
        Ok(EncodedVisualPrompts {
            labels,
            values: flat[..num_classes * dim].to_vec(),
            dim,
        })
    }

    // ==================== 推理 ====================

    /// 对待检图推理：SAHI 开启时自动切片推理（返回类型不变，掩码跨切片平移/并集合并）。
    pub fn predict_impl(&self, image: &Image) -> Result<Vec<Segmentation>> {
        if self.is_sahi_enabled() {
            let config = self
                .sahi_config()
                .expect("sahi config must exist when SAHI enabled");
            return crate::sahi::sliced_predictor::predict_sliced_segmentation(self, self, image, config)
                .map(|result| result.detections);
        }
        self.predict_without_sahi(image)
    }

    /// 原始单图推理路径（SAHI 关闭时的 [`Self::predict_impl`] 行为；SAHI 开启时被逐切片调用）：
    /// letterbox → 检测器(images, pe) → letterbox 逆变换 → 类内 NMS → 掩码解码。
    pub fn predict_without_sahi(&self, image: &Image) -> Result<Vec<Segmentation>> {
        if image.is_empty() {
            return Err(VisionError::image("cannot predict on empty image"));
        }
        // 取出当前提示缓存（pe 很小，整段克隆后立即释放锁，不跨推理持锁）
        let (class_names, pe, pe_classes, pe_dim) = {
            let guard = self.prompts.lock().unwrap();
            match guard.as_ref() {
                None => {
                    return Err(VisionError::inference(
                        "请先调用 setVisualPrompts 设置视觉提示",
                    ));
                }
                Some(p) => (p.class_names.clone(), p.pe.clone(), p.pe_classes, p.pe_dim),
            }
        };

        let orig_w = image.width() as i32;
        let orig_h = image.height() as i32;
        let imgsz = self.imgsz;

        // 1. letterbox 参数（随调用链显式传递）
        let ratio = (imgsz as f32 / orig_h as f32).min(imgsz as f32 / orig_w as f32);
        let new_w = (orig_w as f32 * ratio).round() as usize;
        let new_h = (orig_h as f32 * ratio).round() as usize;
        let dw = (imgsz as f32 - new_w as f32) / 2.0;
        let dh = (imgsz as f32 - new_h as f32) / 2.0;
        let lb = LetterboxParams { ratio, dw, dh };
        let tensor = self.to_chw(image, new_w, new_h, dw.round() as i32, dh.round() as i32)?;

        // 2. 检测器推理：images + pe
        let im_tensor = Tensor::from_array((vec![1i64, 3, imgsz as i64, imgsz as i64], tensor))?;
        let pe_tensor = Tensor::from_array((vec![1i64, pe_classes as i64, pe_dim as i64], pe))?;
        let outputs = self.detector.run_named(vec![
            (self.detector_images_name.as_str(), im_tensor),
            (self.detector_pe_name.as_str(), pe_tensor),
        ])?;
        let out = outputs
            .first()
            .ok_or_else(|| VisionError::inference("检测器没有输出"))?;
        let shape = &out.shape;
        let flat = require_f32_data(out, "检测器 output0")?;

        // End2End 布局 [1, Q, V]（Q=查询数 > V=属性数，内置 NMS，
        // 每行 [x1,y1,x2,y2,conf,cls,(coeffs...)]，坐标为画布像素）；
        // 传统布局 [1, C, A]（C=4+nc+32 << A=anchors，需 NMS，xywh+c 布局）。
        // 候选 = [x1,y1,x2,y2,conf,cls, c0..c31]：掩码系数随候选过 NMS，保留者解码掩码。
        let end2end = shape.len() == 3 && shape[1] > shape[2];
        let mut candidates: Vec<[f32; CANDIDATE_LEN]> = Vec::new();

        if end2end {
            let queries = shape[1] as usize;
            let attrs = shape[2] as usize;
            if attrs < 6 {
                return Err(VisionError::inference(format!(
                    "End2End 输出属性数异常: {attrs}"
                )));
            }
            if flat.len() < queries * attrs {
                return Err(VisionError::inference(format!(
                    "End2End 输出元素数 {} < {}x{}",
                    flat.len(),
                    queries,
                    attrs
                )));
            }
            let has_proto = self.detector_proto_name.is_some();
            let with_coeffs = has_proto && attrs >= 6 + MASK_COEFFS;
            for r in 0..queries {
                let off = r * attrs;
                let conf = flat[off + 4];
                if conf <= self.confidence_threshold {
                    continue;
                }
                let cls_id = (flat[off + 5] as i32).max(0);
                Self::add_end2end_candidate(
                    &mut candidates, flat, off, cls_id, conf, with_coeffs, lb, imgsz,
                );
            }
        } else {
            if shape.len() != 3 {
                return Err(VisionError::inference(format!(
                    "检测器输出应为 3 维 [1, C, A], 实际 shape: {:?}",
                    shape
                )));
            }
            let channels = shape[1] as usize;
            let anchors = shape[2] as usize;
            let num_classes = channels as isize - 36; // 4 box + C scores + 32 mask coeffs
            if num_classes <= 0 {
                return Err(VisionError::inference(format!(
                    "检测器输出通道数 {channels} 无法解析（应 >= 36）"
                )));
            }
            let num_classes = num_classes as usize;
            if flat.len() < channels * anchors {
                return Err(VisionError::inference(format!(
                    "检测器输出元素数 {} < {}x{}",
                    flat.len(),
                    channels,
                    anchors
                )));
            }
            let has_proto = self.detector_proto_name.is_some();
            let with_coeffs = has_proto && channels >= 4 + num_classes + MASK_COEFFS;
            Self::collect_traditional(
                &mut candidates,
                flat,
                anchors,
                num_classes,
                with_coeffs,
                self.confidence_threshold,
                lb,
                imgsz,
            );
        }

        // proto → [C, H*W]（每次 predict 读取一次，供掩码解码）
        let proto = self.parse_proto_output(&outputs)?;

        // 按分数降序 + 类内 NMS（掩码系数随候选保留，保留者解码掩码）
        candidates.sort_by(|p, q| q[4].total_cmp(&p[4]));
        let mut kept: Vec<[f32; CANDIDATE_LEN]> = Vec::new();
        let mut results: Vec<Segmentation> = Vec::new();
        for cand in candidates {
            let mut suppressed = false;
            for k in &kept {
                if k[5] as i32 != cand[5] as i32 {
                    continue;
                }
                let xx1 = cand[0].max(k[0]);
                let yy1 = cand[1].max(k[1]);
                let xx2 = cand[2].min(k[2]);
                let yy2 = cand[3].min(k[3]);
                let inter = (xx2 - xx1).max(0.0) * (yy2 - yy1).max(0.0);
                let a1 = (cand[2] - cand[0]) * (cand[3] - cand[1]);
                let a2 = (k[2] - k[0]) * (k[3] - k[1]);
                if inter / (a1 + a2 - inter + 1e-9) > self.nms_threshold {
                    suppressed = true;
                    break;
                }
            }
            if suppressed {
                continue;
            }
            let cls_id = cand[5] as i32;

            let mask = match &proto {
                Some(p) => Some(self.decode_mask(
                    &cand[6..6 + MASK_COEFFS],
                    p,
                    lb,
                    orig_w,
                    orig_h,
                    cand[0],
                    cand[1],
                    cand[2],
                    cand[3],
                )?),
                // 旧版无 proto 的检测器返回 mask=null，兼容加载
                None => None,
            };
            results.push(Segmentation::new(
                label_name(&class_names, cls_id),
                cls_id,
                cand[0] as f64,
                cand[1] as f64,
                cand[2] as f64,
                cand[3] as f64,
                cand[4] as f64,
                mask,
            ));
            kept.push(cand);
        }

        tracing::debug!("YOLOE 推理完成: {} 个目标", results.len());
        Ok(results)
    }

    /// End2End 行 → 候选（行内为画布像素 xyxy，letterbox 逆变换到原图；
    /// 行尾掩码系数随候选保留；对应 `addEnd2EndCandidate`）。
    #[allow(clippy::too_many_arguments)]
    fn add_end2end_candidate(
        candidates: &mut Vec<[f32; CANDIDATE_LEN]>,
        flat: &[f32],
        off: usize,
        cls_id: i32,
        conf: f32,
        with_coeffs: bool,
        lb: LetterboxParams,
        imgsz: usize,
    ) {
        let w0 = lb.content_width(imgsz) as f32;
        let h0 = lb.content_height(imgsz) as f32;
        let x1 = ((flat[off] - lb.dw) / lb.ratio).max(0.0);
        let y1 = ((flat[off + 1] - lb.dh) / lb.ratio).max(0.0);
        let x2 = ((flat[off + 2] - lb.dw) / lb.ratio).min(w0);
        let y2 = ((flat[off + 3] - lb.dh) / lb.ratio).min(h0);
        candidates.push(build_candidate(
            flat, off + 6, with_coeffs, 1, x1, y1, x2, y2, conf, cls_id,
        ));
    }

    /// 传统布局候选生成：flat[0..3]=xywh（画布），flat[4+k]=类别 k 分数，
    /// 末尾 32 通道掩码系数（对应 `collectTraditional`）。
    #[allow(clippy::too_many_arguments)]
    fn collect_traditional(
        candidates: &mut Vec<[f32; CANDIDATE_LEN]>,
        flat: &[f32],
        anchors: usize,
        num_classes: usize,
        with_coeffs: bool,
        confidence_threshold: f32,
        lb: LetterboxParams,
        imgsz: usize,
    ) {
        let w0 = lb.content_width(imgsz) as f32;
        let h0 = lb.content_height(imgsz) as f32;
        for a in 0..anchors {
            let mut best = 0f32;
            let mut best_cls = -1i32;
            for k in 0..num_classes {
                let sc = flat[(4 + k) * anchors + a];
                if sc > best {
                    best = sc;
                    best_cls = k as i32;
                }
            }
            if best <= confidence_threshold {
                continue;
            }
            let cx = flat[a];
            let cy = flat[anchors + a];
            let bw = flat[2 * anchors + a];
            let bh = flat[3 * anchors + a];
            let x1 = ((cx - bw / 2.0 - lb.dw) / lb.ratio).max(0.0);
            let y1 = ((cy - bh / 2.0 - lb.dh) / lb.ratio).max(0.0);
            let x2 = ((cx + bw / 2.0 - lb.dw) / lb.ratio).min(w0);
            let y2 = ((cy + bh / 2.0 - lb.dh) / lb.ratio).min(h0);
            candidates.push(build_candidate(
                flat,
                (4 + num_classes) * anchors + a,
                with_coeffs,
                anchors,
                x1,
                y1,
                x2,
                y2,
                best,
                best_cls,
            ));
        }
    }

    /// 批量推理：共享当前提示，逐张处理（对应 `predictBatch`）。
    pub fn predict_batch_impl(&self, images: &[Image]) -> Result<Vec<Vec<Segmentation>>> {
        images.iter().map(|img| self.predict_impl(img)).collect()
    }

    // ==================== 内部辅助 ====================

    /// 解析 proto 输出（对应 `protoAsMat`：[1, C, H, W] → [C, H*W] 行主序数据，
    /// gemm 布局直接按行落位）。
    fn parse_proto_output(&self, outputs: &[TensorOutput]) -> Result<Option<ProtoMasks>> {
        let Some(name) = self.detector_proto_name.as_deref() else {
            return Ok(None);
        };
        let proto_out = outputs
            .get(1)
            .ok_or_else(|| VisionError::inference(format!("缺少 proto 输出 '{name}'")))?;
        let p_shape = &proto_out.shape;
        if p_shape.len() != 4 {
            return Err(VisionError::inference(format!(
                "proto 输出应为 4 维 [1, C, H, W], 实际 shape: {:?}",
                p_shape
            )));
        }
        let (c, h, w) = (p_shape[1] as usize, p_shape[2] as usize, p_shape[3] as usize);
        let flat = require_f32_data(proto_out, "检测器 proto")?;
        if flat.len() < c * h * w {
            return Err(VisionError::inference(format!(
                "proto 元素数 {} < {}x{}x{}",
                flat.len(),
                c,
                h,
                w
            )));
        }
        Ok(Some(ProtoMasks {
            dim: c,
            height: h,
            width: w,
            data: flat[..c * h * w].to_vec(),
        }))
    }

    /// 解码实例掩码（与 SegmentationEngine 同一数学，对应 ultralytics process_mask）：
    /// coeffs @ proto → sigmoid → resize 到画布 → 裁掉 letterbox 填充 → resize 到原图
    /// → 裁剪到 bbox（bbox 外为 0）。返回全图 CV_32F 概率图，由 Segmentation 管理。
    #[allow(clippy::too_many_arguments)]
    fn decode_mask(
        &self,
        coeffs: &[f32],
        proto: &ProtoMasks,
        lb: LetterboxParams,
        orig_w: i32,
        orig_h: i32,
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
    ) -> Result<FloatMask> {
        let hw = proto.height * proto.width;
        if coeffs.len() < proto.dim {
            return Err(VisionError::inference(format!(
                "mask coeffs length {} < proto dim {}",
                coeffs.len(),
                proto.dim
            )));
        }

        // coeffs [1,C] @ proto [C,H*W] → [1,H*W]（对应 gemm）
        let mut linear = vec![0f32; hw];
        for (d, &c) in coeffs.iter().take(proto.dim).enumerate() {
            let row = &proto.data[d * hw..(d + 1) * hw];
            for (j, &v) in row.iter().enumerate() {
                linear[j] += c * v;
            }
        }

        // sigmoid
        for v in linear.iter_mut() {
            *v = sigmoid(*v);
        }
        let mask_small = FloatMask::from_raw(proto.width, proto.height, linear)?;

        // resize 到画布尺寸
        let imgsz = self.imgsz;
        let mask_full = mask_small.resize(imgsz, imgsz);

        // 裁掉 letterbox 填充区域
        let pad_left = (lb.dw.round() as i32).clamp(0, imgsz as i32 - 1);
        let pad_top = (lb.dh.round() as i32).clamp(0, imgsz as i32 - 1);
        let valid_w = (imgsz as i32 - pad_left).min((imgsz as f32 - 2.0 * lb.dw).round() as i32);
        let valid_h = (imgsz as i32 - pad_top).min((imgsz as f32 - 2.0 * lb.dh).round() as i32);
        let mask_valid: FloatMask = if valid_w > 0 && valid_h > 0 {
            let vw = valid_w as usize;
            let vh = valid_h as usize;
            let mut data = vec![0f32; vw * vh];
            for y in 0..vh {
                for x in 0..vw {
                    data[y * vw + x] = mask_full.get(pad_left as usize + x, pad_top as usize + y);
                }
            }
            FloatMask::from_raw(vw, vh, data)?
        } else {
            mask_full
        };

        // resize 到原图
        let mask_orig = mask_valid.resize(orig_w as usize, orig_h as usize);

        // bbox 外为 0，bbox 内保留概率
        let bx1 = (x1.round() as i32).max(0);
        let by1 = (y1.round() as i32).max(0);
        let bx2 = (x2.round() as i32).min(orig_w);
        let by2 = (y2.round() as i32).min(orig_h);
        let mut mask_final = FloatMask::new(orig_w as usize, orig_h as usize);
        if bx2 > bx1 && by2 > by1 {
            // 边界检查：bbox 在图像内
            if bx1 >= 0
                && by1 >= 0
                && bx2 <= mask_orig.width() as i32
                && by2 <= mask_orig.height() as i32
            {
                for y in by1..by2 {
                    for x in bx1..bx2 {
                        mask_final
                            .set(x as usize, y as usize, mask_orig.get(x as usize, y as usize));
                    }
                }
            }
        }

        Ok(mask_final)
    }

    /// letterbox 到 newW×newH 再画布居中粘贴，输出 CHW /255 张量（RGB）。
    fn to_chw(
        &self,
        image: &Image,
        new_w: usize,
        new_h: usize,
        pad_x: i32,
        pad_y: i32,
    ) -> Result<Vec<f32>> {
        // 实现假定输入为 CV_8UC3（贴入 CV_8UC3 画布）；对 1/4 通道输入先转 BGR 保持兼容
        let bgr = match image.channels() {
            3 => image.clone(),
            4 => cvt_color(image, ColorConversion::Bgra2Bgr)?,
            1 => cvt_color(image, ColorConversion::Gray2Bgr)?,
            c => return Err(VisionError::image(format!("unsupported channel count: {c}"))),
        };
        let resized = resize(&bgr, new_w, new_h, Interpolation::Linear)?;

        // 画布 114 灰边，居中粘贴
        let mut canvas = Image::filled(self.imgsz, self.imgsz, 3, 114);
        canvas.paste(pad_x.max(0) as usize, pad_y.max(0) as usize, &resized);
        let rgb = cvt_color(&canvas, ColorConversion::Bgr2Rgb)?;

        // HWC → CHW，/255
        let imgsz = self.imgsz;
        let area = imgsz * imgsz;
        let px = rgb.data();
        let mut out = vec![0f32; 3 * area];
        for i in 0..area {
            out[i] = px[i * 3] as f32 / 255.0;
            out[i + area] = px[i * 3 + 1] as f32 / 255.0;
            out[i + 2 * area] = px[i * 3 + 2] as f32 / 255.0;
        }
        Ok(out)
    }
}

// ==================== 自由函数辅助 ====================

/// 奇偶规则（even-odd）点在多边形内判定，(px, py) 取掩码格中心
/// （对应上游 静态方法 `pointInPolygon`）。
fn point_in_polygon(px: f64, py: f64, xs: &[f64], ys: &[f64]) -> bool {
    let mut inside = false;
    let n = xs.len();
    let mut j = n - 1;
    for i in 0..n {
        if (ys[i] > py) != (ys[j] > py) {
            let x_at_y = (xs[j] - xs[i]) * (py - ys[i]) / (ys[j] - ys[i]) + xs[i];
            if px < x_at_y {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// 候选 = [x1,y1,x2,y2,conf,cls, c0..c31]。行优先布局 stride=1（End2End），
/// 通道优先 stride=anchors（传统）（对应 `buildCandidate`）。
#[allow(clippy::too_many_arguments)]
fn build_candidate(
    flat: &[f32],
    coeff_off: usize,
    with_coeffs: bool,
    stride: usize,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    conf: f32,
    cls_id: i32,
) -> [f32; CANDIDATE_LEN] {
    let mut cand = [0f32; CANDIDATE_LEN];
    cand[0] = x1;
    cand[1] = y1;
    cand[2] = x2;
    cand[3] = y2;
    cand[4] = conf;
    cand[5] = cls_id as f32;
    if with_coeffs {
        for (k, v) in cand[6..6 + MASK_COEFFS].iter_mut().enumerate() {
            *v = flat[coeff_off + k * stride];
        }
    }
    cand
}

/// 获取标签名称（越界时返回数字字符串；对应 `getLabelName`，
/// 作用于已克隆出的类别名列表）。
fn label_name(class_names: &[String], class_id: i32) -> String {
    if class_id >= 0 && (class_id as usize) < class_names.len() {
        return class_names[class_id as usize].clone();
    }
    class_id.to_string()
}

/// 读取张量快照中的 f32 数据（legacy DOUBLE 导出由快照层统一转 f32 兼容；
/// 仅 INT64 等其他类型返回明确错误）。
fn require_f32_data<'a>(output: &'a TensorOutput, what: &str) -> Result<&'a [f32]> {
    match &output.data {
        TensorData::F32(v) => Ok(v),
        TensorData::I64(_) => Err(VisionError::inference(format!(
            "{what} 输出为 INT64 类型, 期望 FLOAT32"
        ))),
        TensorData::Unsupported(t) => Err(VisionError::inference(format!(
            "{what} 输出元素类型 {t} 不受支持（原版兼容 legacy double 导出, Rust 侧仅支持 FLOAT32, 请重新以 float 导出）"
        ))),
    }
}

/// sigmoid。
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// 读取引擎会话的全部输入名（对应 `inputNames`；`session` 为 pub(crate) 字段，
/// crate 内可访问）。
fn session_input_names(engine: &BaseOnnxEngine) -> Vec<String> {
    engine
        .session
        .lock()
        .unwrap()
        .inputs()
        .iter()
        .map(|i| i.name().to_string())
        .collect()
}

/// 统一推理接口实现。
///
/// 原版为独立类（仅 `AutoCloseable`，未实现 `OnnxInferenceEngine`），此处接入
/// crate 统一 trait 以便 `Arc` 共享与异步包装；因类别名由视觉提示动态决定（存于
/// `Mutex` 内部），trait 的引用型 `labels()` 恒为 `None`，请用
/// [`YoloERuntimeEngine::get_labels`]。
#[::async_trait::async_trait]
impl crate::core::engine::OnnxInferenceEngine for YoloERuntimeEngine {
    type Output = Vec<Segmentation>;

    /// 单图推理（SAHI 开启时自动切片合并，返回类型不变）。
    fn predict(&self, image: &Image) -> Result<Vec<Segmentation>> {
        self.predict_impl(image)
    }

    /// 批量推理：共享当前提示，逐张处理。
    fn predict_batch(&self, images: &[Image]) -> Result<Vec<Vec<Segmentation>>> {
        self.predict_batch_impl(images)
    }

    fn input_size(&self) -> (i32, i32) {
        self.detector.input_size()
    }

    fn labels(&self) -> Option<&[String]> {
        None
    }

    /// 类别名由 `set_visual_prompts` 决定，本方法为兼容统一 trait 的空实现。
    fn set_labels(&mut self, _labels: Vec<String>) {
        tracing::warn!("YoloERuntimeEngine 的类别名由视觉提示决定, set_labels 无效");
    }

    fn set_confidence_threshold(&mut self, threshold: f32) {
        self.confidence_threshold = threshold;
    }

    fn confidence_threshold(&self) -> f32 {
        self.confidence_threshold
    }
}
