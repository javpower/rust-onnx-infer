//! DART 开放词汇检测分割引擎 v2（Meta SAM3 + DART，training-free，全部检出带实例掩码）。
//!
//! 移植自 Java `DartEngine`（vision-onnx-infer）。相对早期文本-only 版本的变化：
//!
//! - <b>仅 1008 单档</b>（v2 模型只导出 1008 档；v1 的 Tier/useTier 机制随
//!   tier_504 等资产一并移除）；
//! - <b>全部检出带 288×288 实例掩码</b>：enc-dec 换为 `enc_dec_hs.onnx`
//!   （额外输出解码器隐状态 hs 与编码特征 enc_hs），新增 `mask_head.onnx`
//!   解码掩码；返回 [`Segmentation`]（掩码为 f32 概率图，已还原到原图坐标，
//!   >0.5 即前景）；
//! - <b>逐类独立推理</b>：每类单独一次 enc-dec（slot 0 = 当前类，其余槽 padding），
//!   每类取 Top-`max_per_class` 查询数（默认 1 = 参考实现行为；
//!   同类多实例主要靠瓦片模式 + 跨瓦片去重）。
//!
//! # 两种提示方式
//!
//! 1. <b>文本提示</b> [`DartEngine::set_classes`]：英文类目列表；跨类目语义检测。
//! 2. <b>视觉示例（运行时，无需 Python）</b> [`DartEngine::set_visual_prompts`]：
//!    示例图上框选目标 + 英文概念名，引擎内部经 `geometry_encoder.onnx` 实时编码成
//!    概念 token；支持跨图迁移——换个品类只需换示例图重新框选。
//!
//! 两种方式共用同一检测头；"示例框 + 文本"联合的视觉提示跨图迁移最强
//! （官方实测 0.974）。文本 32 + 几何示例空间 16 共 48 token 槽位。
//!
//! # 模型目录（`models_dir` 标准布局，共 5 个 ONNX + 2 个资产）
//!
//! ```text
//!   modelsDir/
//!   ├── tokenizer.json            # 分词器（BPE 资产）
//!   ├── text_encoder.onnx         # 文本编码器（~1.4GB；换类目毫秒级、有缓存）
//!   ├── enc_dec_hs.onnx           # 编码-解码器（输出 hs/enc_hs 供掩码头使用）
//!   ├── mask_head.onnx            # 掩码头（288x288 掩码 logits）
//!   ├── geometry_encoder.onnx     # 几何编码器（运行时视觉提示；缺省可省，仅文本部署）
//!   ├── pos_1008.bin              # 位置编码常量（float32 小端，[P*P,1,256] 像素优先布局）
//!   └── tier_1008/
//!       └── hf_backbone.onnx      # 骨干网（~1.8GB .data 外置权重）
//! ```
//!
//! # ONNX I/O（对真实模型逐一探针确认）
//!
//! - `text_encoder.onnx`：`input_ids` INT64 [N,32] → `text_feats` FLOAT [32,N,256]
//!   + `text_mask` FLOAT [N,32]（1=padding）
//! - `tier_1008/hf_backbone.onnx`：`pixel_values` FLOAT [1,3,1008,1008] →
//!   `conv2d_2` [1,256,288,288]（=fpn_0）+ `conv2d_4` [1,256,144,144]（=fpn_1）
//!   + `conv2d_6` [1,256,72,72]（=fpn_2）
//! - `enc_dec_hs.onnx`：`img_feat` [4,256,72,72] + `img_pos` [4,256,72,72] +
//!   `text_feats` [48,4,256] + `text_mask` [4,48] → `scores` [4,200,1] +
//!   `boxes` [4,200,4] + `hs` [6,4,200,256] + `enc_hs` [5184,4,256]
//! - `mask_head.onnx`：`fpn_0` [1,256,288,288] + `fpn_1` [1,256,144,144] +
//!   `obj_queries` [6,1,1,256] + `enc_hs` [5184,1,256] + `text_feats` [32,1,256] +
//!   `text_mask` [1,32] → `pred_masks` [1,1,288,288]
//! - `geometry_encoder.onnx`：`conv_72` [1,256,72,72] + `text_feats` [32,1,256] +
//!   `text_mask` BOOL [1,32] + `boxes` [B,1,4] + `box_labels` BOOL [B,1] +
//!   `box_mask` BOOL [1,B] → `prompt` [L,1,256] + `prompt_mask` BOOL [1,L]
//!
//! # 示例
//!
//! ```ignore
//! use rust_onnx_infer::{DartEngine, DeviceType};
//!
//! let dart = DartEngine::new("/models/dart", DeviceType::Cpu)?;
//! // 方式一：文本提示（英文类目）
//! dart.set_classes(&["door trim panel", "door latch"])?;
//! // 方式二：运行时视觉示例（示例图上框选 + 英文概念名，无需 Python）
//! // dart.set_visual_prompt(&exemplar, &[Rect::new(1907, 781, 33, 72)], "door latch")?;
//!
//! let r1 = dart.predict(&image)?;        // 整图（含掩码）
//! let r2 = dart.predict_large(&image)?;  // 大图找小件：瓦片模式
//! let r3 = dart.predict_roi(&image, roi)?; // ROI 细查
//! ```
//!
//! # 重要规则（对应 Java javadoc）
//!
//! - <b>类名必须英文</b>（文本编码按英文训练，中文类名输出白噪声分数）；
//! - <b>小件必须两级</b>：1008 全图对 74px 级小目标会漏检，
//!   用 [`DartEngine::predict_large`]（瓦片）或 [`DartEngine::predict_roi`]；
//! - <b>掩码坐标</b>：`mask` 为原图尺寸 f32 概率图（sigmoid 后），>0.5 即前景；
//!   瓦片/ROI 模式掩码已平移回原图坐标；
//! - <b>性能</b>：逐类独立推理，类数越多越慢；CPU 上 1008 全图约 1-2 分钟
//!   （6 类），内存约 3.5GB（骨干 ~2GB + 文本编码器 ~1.4GB）。
//!
//! <p>权重许可：模型权重来自 Meta SAM3（SAM License，注意商用条款）；
//! DART 框架同为 SAM License。本引擎代码不含任何模型权重。
//!
//! 线程模型：非线程安全（与 Java 一致）；内部可变提示状态用 `Mutex` 承载，
//! 但请勿多线程并发调用同一实例的 predict。

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use ort::session::SessionOutputs;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::core::base::BaseOnnxEngine;
use crate::core::device_type::DeviceType;
use crate::error::{Result, VisionError};
use crate::imaging::{cvt_color, resize, ColorConversion, FloatMask, Image, Interpolation, Rect};
use crate::model::Segmentation;

// ============================ 固定尺寸（v2 模型约定） ============================

/// 模型输入边长（对应 `IMG_SZ`）。
const IMG_SZ: usize = 1008;
/// 骨干 FPN 网格 P×P（stride 14）（对应 `P`）。
const P: usize = 72;
/// 特征通道（对应 `C`）。
const C: usize = 256;
/// 提示槽位 = 文本 32 + 几何示例 16（对应 `TEXT_LEN`）。
const TEXT_LEN: usize = 48;
/// 文本部分行数（对应 `TEXT_PART`）。
const TEXT_PART: usize = 32;
/// enc-dec batch 容量（逐类只用 slot 0）（对应 `MAX_CLASSES`）。
const MAX_CLASSES: usize = 4;
/// enc-dec 查询数（对应 `QUERIES`）。
const QUERIES: usize = 200;
/// 解码器层数（hs 第一维）（对应 `DEC_LAYERS`）。
const DEC_LAYERS: usize = 6;
/// 掩码边长 4P（对应 `MASK_SZ`）。
const MASK_SZ: usize = 288;
/// enc_hs 展平长度 = P*P（对应 `ENC_HS_LEN`）。
const ENC_HS_LEN: usize = P * P;
/// 类/概念数上限（对应 `checkCount` 的 16）。
const MAX_PROMPTS: usize = 16;
/// 单概念示例框数上限（对应 `setVisualPrompts` 的 1~8 框约束）。
const MAX_BOXES_PER_PROMPT: usize = 8;

// ---- 瓦片模式魔数 ----
/// 瓦片重叠比例（对应 `TILE_OVERLAP`）。
const TILE_OVERLAP: f64 = 0.15;
/// 小于该边长的残余瓦片直接跳过（对应 `MIN_TILE_EDGE`）。
const MIN_TILE_EDGE: i32 = 200;

// ---- 模型 I/O 名称（对真实模型探针确认，对应 Java 各 *INPUTS/*OUTPUTS 常量）----
const IN_INPUT_IDS: &str = "input_ids";
const OUT_TEXT_FEATS: &str = "text_feats";
const OUT_TEXT_MASK: &str = "text_mask";

const IN_PIXEL_VALUES: &str = "pixel_values";
/// 模型导出的骨干网输出名 conv2d_2（形状语义 = fpn_0 [1,256,288,288]）。
const OUT_FPN0: &str = "conv2d_2";
/// conv2d_4（形状语义 = fpn_1 [1,256,144,144]）。
const OUT_FPN1: &str = "conv2d_4";
/// conv2d_6（形状语义 = fpn_2 [1,256,72,72]）。
const OUT_FPN2: &str = "conv2d_6";

const IN_IMG_FEAT: &str = "img_feat";
const IN_IMG_POS: &str = "img_pos";
const IN_TEXT_FEATS: &str = "text_feats";
const IN_TEXT_MASK: &str = "text_mask";
const OUT_SCORES: &str = "scores";
const OUT_BOXES: &str = "boxes";
const OUT_HS: &str = "hs";
const OUT_ENC_HS: &str = "enc_hs";

const IN_FPN0: &str = "fpn_0";
const IN_FPN1: &str = "fpn_1";
const IN_OBJ_QUERIES: &str = "obj_queries";
const IN_ENC_HS: &str = "enc_hs";
const OUT_PRED_MASKS: &str = "pred_masks";

const IN_CONV72: &str = "conv_72";
const IN_BOXES: &str = "boxes";
const IN_BOX_LABELS: &str = "box_labels";
const IN_BOX_MASK: &str = "box_mask";
const OUT_PROMPT: &str = "prompt";
const OUT_PROMPT_MASK: &str = "prompt_mask";

// ============================ 状态 ============================

/// 当前提示（对应 Java 的 currentNames/currentPrompt/currentPromptMask/currentN 四个字段；
/// feats 为 slot 主序 [TEXT_LEN*N*C]，mask 为 [N*TEXT_LEN]，1 = padding）。
#[derive(Debug, Clone)]
struct DartPrompt {
    /// 类目/概念名（顺序 = 类别 id）
    names: Vec<String>,
    /// [TEXT_LEN * N * C] 特征（slot 主序）
    feats: Vec<f32>,
    /// [N * TEXT_LEN] mask（1 = padding）
    mask: Vec<f32>,
}

/// 运行时视觉提示的单概念定义（对应 Java `VisualPromptSpec` record）：
/// 概念名（英文）+ 示例图上的框（像素坐标，1~8 个，框请画紧）。
#[derive(Debug, Clone, PartialEq)]
pub struct VisualPromptSpec {
    /// 英文概念名（如 "door latch"，作为检出结果的类名）
    pub name: String,
    /// 示例图上的目标框（像素坐标，1~8 个 = 同概念多实例）
    pub boxes: Vec<Rect>,
}

impl VisualPromptSpec {
    /// 创建单概念视觉提示定义。
    pub fn new(name: impl Into<String>, boxes: Vec<Rect>) -> Self {
        VisualPromptSpec {
            name: name.into(),
            boxes,
        }
    }
}

/// DART 开放词汇检测分割引擎 v2（对应 Java `DartEngine`）。
///
/// 多 BaseOnnxEngine 组合（参考 yolo_e_runtime 的双模型架构）：
/// 骨干网与编码-解码器用传入设备；文本编码器、掩码头与几何编码器恒为 CPU
/// （与 Java 一致，三者计算量小）。
pub struct DartEngine {
    /// 文本编码器：input_ids [N,32] int64 → text_feats [32,N,256] + text_mask [N,32]
    text_encoder: BaseOnnxEngine,
    /// 骨干网：pixel_values [1,3,1008,1008] → conv2d_2/4/6 三级 FPN
    backbone: BaseOnnxEngine,
    /// 编码-解码器：img_feat/img_pos/text_feats/text_mask → scores/boxes/hs/enc_hs
    enc_dec: BaseOnnxEngine,
    /// 掩码头：fpn_0/fpn_1/obj_queries/enc_hs/text_feats/text_mask → pred_masks
    mask_head: BaseOnnxEngine,
    /// 几何编码器（可选）：缺省不影响文本路径，set_visual_prompts 时才必须
    geometry: Option<BaseOnnxEngine>,
    /// 分词器（对应 `HuggingFaceTokenizer`；Java 未配置 padding，Rust 侧同样
    /// 手动按 TEXT_PART 截断/补 0）
    tokenizer: Tokenizer,
    /// 位置编码 NCHW 布局 [C, P, P]（构造时由 pos_1008.bin 的 [P*P,1,C]
    /// 像素优先布局转置一次；对应 Java `posEnc` + `posEncNchwBatch`，Java 在
    /// 每次区域推理时重算，数值等价，Rust 侧提前做）
    pos_enc_nchw: Vec<f32>,
    /// 文本嵌入缓存: 类名组合串 → ([32,N,C] 特征 + [N,32] mask)（对应 `textCache`）
    text_cache: Mutex<HashMap<String, (Vec<f32>, Vec<f32>)>>,
    /// 当前提示；None = 未设置（对应 Java currentPrompt == null）
    prompt: Mutex<Option<DartPrompt>>,

    // ---- 配置（对应 @Getter @Setter 字段）----
    /// 候选分数阈值（sigmoid 后；默认 0.1 与参考实现一致，偏保守防漏检）
    confidence_threshold: f32,
    /// 跨瓦片去重 IoU 阈值（同类）
    nms_threshold: f32,
    /// 每类每次推理保留的查询数；1 = 参考实现每类 Top-1，调大可得同类多实例
    max_per_class: usize,
}

impl DartEngine {
    // ============================ 生命周期 ============================

    /// 创建引擎（对应 `DartEngine(String modelsDir, DeviceType deviceType)`）。
    ///
    /// - `models_dir`: 模型目录（标准布局见 [crate 文档](self)）
    /// - `device_type`: 骨干网与编码-解码器的设备；文本编码器与掩码头恒为 CPU
    ///   （与参考实现一致，二者计算量小）
    pub fn new(models_dir: impl AsRef<Path>, device_type: DeviceType) -> Result<Self> {
        let models_dir = models_dir.as_ref();

        // 分词器（对应 HuggingFaceTokenizer.newInstance(tokenizer.json, {})）
        let tokenizer = Tokenizer::from_file(models_dir.join("tokenizer.json"))
            .map_err(|e| VisionError::Tokenizer(format!("加载 tokenizer 失败: {e}")))?;

        // 会话：文本编码器/掩码头恒为 CPU；骨干网与 enc-dec 用指定设备
        let text_encoder = BaseOnnxEngine::new(models_dir.join("text_encoder.onnx"), DeviceType::Cpu)?;
        let backbone = BaseOnnxEngine::new(
            models_dir.join("tier_1008").join("hf_backbone.onnx"),
            device_type,
        )?;
        let enc_dec = BaseOnnxEngine::new(models_dir.join("enc_dec_hs.onnx"), device_type)?;
        let mask_head = BaseOnnxEngine::new(models_dir.join("mask_head.onnx"), DeviceType::Cpu)?;
        // 几何编码器（运行时视觉提示）：缺省时不影响文本路径，set_visual_prompts 时才必须
        let geo_path = models_dir.join("geometry_encoder.onnx");
        let geometry = if geo_path.exists() {
            Some(BaseOnnxEngine::new(&geo_path, DeviceType::Cpu)?)
        } else {
            None
        };

        // pos bin 布局 [P*P,1,256]（像素优先），float32 小端（对应 readFloatsFile）
        let pos_enc = read_f32_le_file(&models_dir.join("pos_1008.bin"))?;
        if pos_enc.len() != P * P * C {
            return Err(VisionError::invalid_argument(format!(
                "pos_1008.bin 元素数 {} != {}（预期 [P*P,1,C] 布局）",
                pos_enc.len(),
                P * P * C
            )));
        }
        let pos_enc_nchw = pos_enc_nchw(&pos_enc);
        tracing::info!(
            "DartEngine(v2) 初始化: modelsDir={}, 1008 单档, geometryEncoder={}",
            models_dir.display(),
            if geometry.is_some() {
                "已加载"
            } else {
                "缺失（运行时视觉提示不可用）"
            }
        );

        Ok(DartEngine {
            text_encoder,
            backbone,
            enc_dec,
            mask_head,
            geometry,
            tokenizer,
            pos_enc_nchw,
            text_cache: Mutex::new(HashMap::new()),
            prompt: Mutex::new(None),
            confidence_threshold: 0.1,
            nms_threshold: 0.5,
            max_per_class: 1,
        })
    }

    // ============================ 访问器（对应 @Getter/@Setter） ============================

    /// 候选分数阈值（sigmoid 后）。
    pub fn confidence_threshold(&self) -> f32 {
        self.confidence_threshold
    }

    /// 设置候选分数阈值（默认 0.1）。
    pub fn set_confidence_threshold(&mut self, threshold: f32) {
        self.confidence_threshold = threshold;
    }

    /// 跨瓦片去重 IoU 阈值（同类）。
    pub fn nms_threshold(&self) -> f32 {
        self.nms_threshold
    }

    /// 设置跨瓦片去重 IoU 阈值（默认 0.5）。
    pub fn set_nms_threshold(&mut self, threshold: f32) {
        self.nms_threshold = threshold;
    }

    /// 每类每次推理保留的查询数（默认 1）。
    pub fn max_per_class(&self) -> usize {
        self.max_per_class
    }

    /// 设置每类保留查询数；1 = 参考实现每类 Top-1，调大可得同类多实例。
    pub fn set_max_per_class(&mut self, max_per_class: usize) {
        self.max_per_class = max_per_class.max(1);
    }

    /// 当前类目/概念名（顺序 = 类别 id；对应 `getLabels`）。
    pub fn get_labels(&self) -> Vec<String> {
        self.prompt
            .lock()
            .unwrap()
            .as_ref()
            .map(|p| p.names.clone())
            .unwrap_or_default()
    }

    /// 几何编码器是否可用（运行时视觉提示是否可用）。
    pub fn has_geometry_encoder(&self) -> bool {
        self.geometry.is_some()
    }

    // ============================ 提示设置 ============================

    /// 文本提示（英文类目，1~16 个；对应 `setClasses`）。
    ///
    /// 换类目只需重调此方法（文本嵌入有缓存，重复切换毫秒级）。
    /// 类名必须英文：文本编码按英文训练，中文类名输出白噪声分数。
    pub fn set_classes(&self, english_classes: &[&str]) -> Result<()> {
        check_count(english_classes.len())?;
        let names: Vec<String> = english_classes.iter().map(|s| s.to_string()).collect();

        // 文本嵌入缓存（对应 textCache，key = "\u0001".join(classes)）
        let (feats, mask) = {
            let key = english_classes.join("\u{1}");
            let mut cache = self.text_cache.lock().unwrap();
            match cache.get(&key) {
                Some(v) => v.clone(),
                None => {
                    let v = self.encode_text(&names)?;
                    cache.insert(key, v.clone());
                    v
                }
            }
        };
        self.pack_slots(&names, &feats, &mask);
        Ok(())
    }

    /// 从概念文件设置文本提示（对应 `OnnxEngineFactory` javadoc 提到的
    /// `setConcepts(概念文件)` 用法；Java v2 实现未内置该方法，Rust 侧补齐）。
    ///
    /// 文件为 UTF-8 文本，每行一个英文概念名；空行与 `#` 开头行忽略。
    pub fn set_concepts(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)
            .map_err(|e| VisionError::Io(format!("读取概念文件失败 {}: {e}", path.display())))?;
        let names: Vec<&str> = content
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect();
        self.set_classes(&names)
    }

    /// 运行时视觉提示（单概念便捷重载，对应 `setVisualPrompts(Mat, List<Rect>, String)`）：
    /// 在示例图上框选目标（可多框 = 同概念多实例），配合英文概念名实时编码。
    pub fn set_visual_prompt(&self, exemplar: &Image, boxes: &[Rect], text: &str) -> Result<()> {
        self.set_visual_prompts(exemplar, &[VisualPromptSpec::new(text, boxes.to_vec())])
    }

    /// 运行时视觉提示（多概念，对应 `setVisualPrompts(Mat, List<VisualPromptSpec>)`）：
    /// 同一张示例图上为多个概念各自框选目标。
    ///
    /// 骨干网只跑一次，逐概念过几何编码器；概念数 1~16，每概念 1~8 框。
    /// 概念 token = 文本（概念名，经 text_encoder）+ 几何（框，经 geometry_encoder），
    /// "文本+视觉"联合的跨图迁移最强。
    pub fn set_visual_prompts(&self, exemplar: &Image, concepts: &[VisualPromptSpec]) -> Result<()> {
        let geometry = self.geometry.as_ref().ok_or_else(|| {
            VisionError::invalid_argument("models 目录缺少 geometry_encoder.onnx（运行时视觉提示不可用）")
        })?;
        check_count(concepts.len())?;
        for spec in concepts {
            if spec.name.trim().is_empty() {
                return Err(VisionError::invalid_argument("概念名（英文）不能为空"));
            }
            if spec.boxes.is_empty() || spec.boxes.len() > MAX_BOXES_PER_PROMPT {
                return Err(VisionError::invalid_argument(format!(
                    "概念 {} 的示例框数量须为 1~{MAX_BOXES_PER_PROMPT}",
                    spec.name
                )));
            }
        }

        // 1) 示例图骨干特征（仅 72 级参与几何编码；多概念只跑一次骨干）
        let input = preprocess(exemplar, IMG_SZ)?;
        let (_fpn0, _fpn1, conv72) = self.run_backbone(input)?;

        // 2) 逐概念编码并打包进 slot
        let n = concepts.len();
        let mut names = Vec::with_capacity(n);
        let mut feats = vec![0f32; TEXT_LEN * n * C];
        let mut mask = vec![1f32; n * TEXT_LEN];
        for (i, spec) in concepts.iter().enumerate() {
            let (tfeats, tmask) = self.encode_text(std::slice::from_ref(&spec.name))?;
            let (prompt, pmask) = self.run_geometry_encoder(
                geometry,
                &conv72,
                &tfeats,
                &tmask,
                exemplar.width(),
                exemplar.height(),
                &spec.boxes,
            )?;
            let l = prompt.len() / C;
            if l > TEXT_LEN {
                return Err(VisionError::inference(format!(
                    "几何编码器输出 token 数 {l} 超过槽位 {TEXT_LEN}"
                )));
            }
            names.push(spec.name.clone());
            feats[i * TEXT_LEN * C..i * TEXT_LEN * C + l * C].copy_from_slice(&prompt[..l * C]);
            for s in 0..l {
                mask[i * TEXT_LEN + s] = pmask[s];
            }
            tracing::info!("概念 \"{}\" 已编码: {} token", spec.name, l);
        }

        tracing::info!("运行时视觉提示已设: {} 个概念（骨干 1 次）", n);
        *self.prompt.lock().unwrap() = Some(DartPrompt {
            names,
            feats,
            mask,
        });
        Ok(())
    }

    // ============================ 检测 API（全部带掩码） ============================

    /// 整图检测（对应 `predict(Mat)`）：整图缩放到 1008×1008 推理
    /// （大件场景；掩码为原图尺寸概率图）。
    pub fn predict(&self, image: &Image) -> Result<Vec<Segmentation>> {
        let prompt = self.require_prompt()?;
        if image.is_empty() {
            return Err(VisionError::image("cannot predict on empty image"));
        }
        let full_w = image.width() as i32;
        let full_h = image.height() as i32;
        self.detect_region(&prompt, image, (0, 0), (full_w, full_h))
    }

    /// 大图找小件（对应 `predict(Mat, smallParts=true)`）：超过 1008 的大图自动切
    /// 1008px 瓦片 1:1 采样（15% 重叠，跨瓦片同类 IoU 去重）；耗时按瓦片数放大。
    pub fn predict_large(&self, image: &Image) -> Result<Vec<Segmentation>> {
        let prompt = self.require_prompt()?;
        if image.is_empty() {
            return Err(VisionError::image("cannot predict on empty image"));
        }
        let w = image.width() as i32;
        let h = image.height() as i32;
        // 不超过 1008 的图直接整图推理（对应 max(w,h) <= IMG_SZ 分支）
        if w.max(h) <= IMG_SZ as i32 {
            return self.detect_region(&prompt, image, (0, 0), (w, h));
        }
        let step = (IMG_SZ as f64 * (1.0 - TILE_OVERLAP)) as i32;
        let mut all = Vec::new();
        let mut y = 0;
        while y < h {
            let mut x = 0;
            while x < w {
                let cw = (IMG_SZ as i32).min(w - x);
                let ch = (IMG_SZ as i32).min(h - y);
                // 残余小瓦片跳过（对应 continue）
                if cw >= MIN_TILE_EDGE && ch >= MIN_TILE_EDGE {
                    let tile = image.crop(x as usize, y as usize, cw as usize, ch as usize)?;
                    all.extend(self.detect_region(&prompt, &tile, (x, y), (w, h))?);
                }
                x += step;
            }
            y += step;
        }
        Ok(dedup_by_class(all, self.nms_threshold))
    }

    /// ROI 区域细查（小件推荐，对应 `predictROI`）：对 `roi` 子区域单独推理，
    /// 检测框与掩码均已映射回原图坐标。
    pub fn predict_roi(&self, image: &Image, roi: Rect) -> Result<Vec<Segmentation>> {
        let prompt = self.require_prompt()?;
        if image.is_empty() {
            return Err(VisionError::image("cannot predict on empty image"));
        }
        let w = image.width() as i32;
        let h = image.height() as i32;
        if roi.x < 0
            || roi.y < 0
            || roi.width <= 0
            || roi.height <= 0
            || roi.right() > w
            || roi.bottom() > h
        {
            return Err(VisionError::invalid_argument(format!(
                "ROI {:?} 越界（图像 {w}x{h}）",
                roi
            )));
        }
        let sub = image.crop(roi.x as usize, roi.y as usize, roi.width as usize, roi.height as usize)?;
        self.detect_region(&prompt, &sub, (roi.x, roi.y), (w, h))
    }

    /// 单区域完整推理（对应 `detectRegion`）：region 图缩放到 1008×1008 推理，
    /// 检测框与掩码映射回原图坐标（region 位于原图 (origin_x, origin_y) 处，
    /// 整图尺寸 full_w×full_h）。
    #[allow(clippy::too_many_arguments)]
    fn detect_region(
        &self,
        prompt: &DartPrompt,
        region: &Image,
        origin: (i32, i32),
        full: (i32, i32),
    ) -> Result<Vec<Segmentation>> {
        let (origin_x, origin_y) = origin;
        let (full_w, full_h) = full;
        let region_w = region.width() as i32;
        let region_h = region.height() as i32;
        let input = preprocess(region, IMG_SZ)?;

        // 骨干：region 一次，输出三级 FPN
        let t0 = std::time::Instant::now();
        let (f0, f1, f2) = self.run_backbone(input)?;
        let backbone_ms = t0.elapsed().as_millis();

        let mut res = Vec::new();
        for n in 0..prompt.names.len() {
            res.extend(self.detect_class(
                prompt, n, region_w, region_h, origin_x, origin_y, full_w, full_h, &f0, &f1, &f2,
            )?);
        }
        res.sort_by(|a, b| {
            b.confidence()
                .partial_cmp(&a.confidence())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        tracing::debug!(
            "区域 {}x{} 骨干耗时 {}ms, 检出 {} 条",
            region_w,
            region_h,
            backbone_ms,
            res.len()
        );
        Ok(res)
    }

    /// 逐类独立推理（对应 `detectClass`）：slot 0 = 当前类（其余 3 槽 padding），
    /// 取分数 Top-max_per_class 查询，逐查询解 288×288 掩码。
    #[allow(clippy::too_many_arguments)]
    fn detect_class(
        &self,
        prompt: &DartPrompt,
        n: usize,
        region_w: i32,
        region_h: i32,
        origin_x: i32,
        origin_y: i32,
        full_w: i32,
        full_h: i32,
        f0: &[f32],
        f1: &[f32],
        f2: &[f32],
    ) -> Result<Vec<Segmentation>> {
        let cls = prompt.names[n].clone();

        // 该类的单槽提示: slot 0 = 当前类，其余 3 槽 padding（feats=0, mask=1）
        let mut tf = vec![0f32; TEXT_LEN * MAX_CLASSES * C];
        let mut tm = vec![1f32; MAX_CLASSES * TEXT_LEN];
        for s in 0..TEXT_LEN {
            let src = (n * TEXT_LEN + s) * C;
            tf[(s * MAX_CLASSES) * C..(s * MAX_CLASSES) * C + C]
                .copy_from_slice(&prompt.feats[src..src + C]);
            tm[s] = prompt.mask[n * TEXT_LEN + s];
        }

        // 图像特征与位置编码沿 batch 复制 M 份（enc-dec 固定 batch = MAX_CLASSES）
        let single = C * P * P;
        let mut f2b = vec![0f32; MAX_CLASSES * single];
        let mut posb = vec![0f32; MAX_CLASSES * single];
        for r in 0..MAX_CLASSES {
            f2b[r * single..(r + 1) * single].copy_from_slice(f2);
            posb[r * single..(r + 1) * single].copy_from_slice(&self.pos_enc_nchw);
        }

        let f_t = Tensor::from_array((
            vec![MAX_CLASSES as i64, C as i64, P as i64, P as i64],
            f2b,
        ))?;
        let p_t = Tensor::from_array((
            vec![MAX_CLASSES as i64, C as i64, P as i64, P as i64],
            posb,
        ))?;
        let t_t = Tensor::from_array((
            vec![TEXT_LEN as i64, MAX_CLASSES as i64, C as i64],
            tf,
        ))?;
        let m_t = Tensor::from_array((vec![MAX_CLASSES as i64, TEXT_LEN as i64], tm))?;

        let (scores, boxes, hs, enc) = {
            let mut session = self.enc_dec.session.lock().unwrap();
            let outputs = session.run(ort::inputs![
                IN_IMG_FEAT   => f_t,
                IN_IMG_POS    => p_t,
                IN_TEXT_FEATS => t_t,
                IN_TEXT_MASK  => m_t,
            ])?;
            (
                snapshot_f32_output(&outputs, OUT_SCORES)?,
                snapshot_f32_output(&outputs, OUT_BOXES)?,
                snapshot_f32_output(&outputs, OUT_HS)?,
                snapshot_f32_output(&outputs, OUT_ENC_HS)?,
            )
        };
        let (_, scores) = scores;
        let (_, boxes) = boxes;
        let (_, hs) = hs;
        let (_, enc) = enc;
        expect_len(&scores, MAX_CLASSES * QUERIES, OUT_SCORES)?;
        expect_len(&boxes, MAX_CLASSES * QUERIES * 4, OUT_BOXES)?;
        expect_len(&hs, DEC_LAYERS * MAX_CLASSES * QUERIES * C, OUT_HS)?;
        expect_len(&enc, ENC_HS_LEN * MAX_CLASSES * C, OUT_ENC_HS)?;

        // slot 0 = 当前类：按分数取 Top-max_per_class 查询（默认 1 = 参考实现 Top-1）
        let queries = top_queries(&scores, self.max_per_class, self.confidence_threshold);
        let mut res = Vec::with_capacity(queries.len());
        for qi in queries {
            let score = sigmoid(scores[qi]);
            let cx = boxes[qi * 4];
            let cy = boxes[qi * 4 + 1];
            let bw = boxes[qi * 4 + 2];
            let bh = boxes[qi * 4 + 3];
            // cxcywh（region 归一化）→ xyxy（原图像素 + 原点平移）
            let x1 = (cx - bw / 2.0) * region_w as f32 + origin_x as f32;
            let y1 = (cy - bh / 2.0) * region_h as f32 + origin_y as f32;
            let x2 = (cx + bw / 2.0) * region_w as f32 + origin_x as f32;
            let y2 = (cy + bh / 2.0) * region_h as f32 + origin_y as f32;

            let hs1 = slice_hs(&hs, qi); // [6*C] 全 6 层
            let enc1 = slice_batch0(&enc); // [P*P*C]
            let tf1 = slice_text(&prompt.feats, n); // [TEXT_LEN*C]
            let tm1 = slice_text_mask(&prompt.mask, n); // [TEXT_LEN]
            let logits = self.run_mask_head(f0, f1, &hs1, &enc1, &tf1, &tm1)?;
            let mask = mask_to_original(
                &logits,
                region_w,
                region_h,
                origin_x,
                origin_y,
                full_w,
                full_h,
            )?;

            res.push(Segmentation::new(
                cls.clone(),
                n as i32,
                x1 as f64,
                y1 as f64,
                x2 as f64,
                y2 as f64,
                score as f64,
                Some(mask),
            ));
        }
        Ok(res)
    }

    /// 掩码头（对应 `runMaskHead`）：fpn_0/fpn_1 + 单查询 hs/enc_hs + 该类提示
    /// （前 32 文本行）→ 288×288 logits。
    fn run_mask_head(
        &self,
        f0: &[f32],
        f1: &[f32],
        hs1: &[f32],
        enc1: &[f32],
        tf1: &[f32],
        tm1: &[f32],
    ) -> Result<Vec<f32>> {
        // text_feats/text_mask 只取前 32 文本行
        let tf_head = &tf1[..TEXT_PART * C];
        let tm_head = &tm1[..TEXT_PART];

        let f0_t = Tensor::from_array((
            vec![1, C as i64, (4 * P) as i64, (4 * P) as i64],
            f0.to_vec(),
        ))?;
        let f1_t = Tensor::from_array((
            vec![1, C as i64, (2 * P) as i64, (2 * P) as i64],
            f1.to_vec(),
        ))?;
        let o_t = Tensor::from_array((
            vec![DEC_LAYERS as i64, 1, 1, C as i64],
            hs1.to_vec(),
        ))?;
        let e_t = Tensor::from_array((vec![ENC_HS_LEN as i64, 1, C as i64], enc1.to_vec()))?;
        let t_t = Tensor::from_array((vec![TEXT_PART as i64, 1, C as i64], tf_head.to_vec()))?;
        let m_t = Tensor::from_array((vec![1, TEXT_PART as i64], tm_head.to_vec()))?;

        let (logits,) = {
            let mut session = self.mask_head.session.lock().unwrap();
            let outputs = session.run(ort::inputs![
                IN_FPN0        => f0_t,
                IN_FPN1        => f1_t,
                IN_OBJ_QUERIES => o_t,
                IN_ENC_HS      => e_t,
                IN_TEXT_FEATS  => t_t,
                IN_TEXT_MASK   => m_t,
            ])?;
            (snapshot_f32_output(&outputs, OUT_PRED_MASKS)?,)
        };
        let (_, logits) = logits;
        expect_len(&logits, MASK_SZ * MASK_SZ, OUT_PRED_MASKS)?;
        Ok(logits)
    }

    /// 骨干网推理（region 一次）：[1,3,1008,1008] → (fpn_0 [288], fpn_1 [144], fpn_2 [72])。
    fn run_backbone(&self, chw: Vec<f32>) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let tensor = Tensor::from_array((vec![1, 3, IMG_SZ as i64, IMG_SZ as i64], chw))?;
        let (f0, f1, f2) = {
            let mut session = self.backbone.session.lock().unwrap();
            let outputs = session.run(ort::inputs![IN_PIXEL_VALUES => tensor])?;
            (
                snapshot_f32_output(&outputs, OUT_FPN0)?,
                snapshot_f32_output(&outputs, OUT_FPN1)?,
                snapshot_f32_output(&outputs, OUT_FPN2)?,
            )
        };
        let (_, f0) = f0;
        let (_, f1) = f1;
        let (_, f2) = f2;
        expect_len(&f0, C * 4 * P * 4 * P, OUT_FPN0)?;
        expect_len(&f1, C * 2 * P * 2 * P, OUT_FPN1)?;
        expect_len(&f2, C * P * P, OUT_FPN2)?;
        Ok((f0, f1, f2))
    }

    // ============================ 提示打包/加载 ============================

    /// 把 [TEXT_PART,N,C] 特征 + [N,TEXT_PART] mask 打包进 TEXT_LEN 槽位
    /// （其余行 padding=1）（对应 `packSlots`）。
    ///
    /// <b>与 Java 的偏差说明</b>：text_encoder 实测输出为 [32,N,256]（seq 主序，
    /// 经 onnxruntime 逐位验证：batch 内第 n 类的行块 = 输出 [:, n, :]）。
    /// Java `packSlots` 的 `arraycopy(feats, n*textPart*C, ...)` 按连续块拷贝，
    /// 仅在 N=1 时与该布局一致（N>1 时会跨类混行，疑为移植笔误；Java 侧
    /// javadoc 亦标注输入为 [textPart,N,C]）。Rust 侧按真实模型契约做跨步拷贝，
    /// N=1 时与 Java 逐位一致。
    fn pack_slots(&self, names: &[String], feats: &[f32], mask: &[f32]) {
        let n = names.len();
        let mut f = vec![0f32; TEXT_LEN * n * C];
        let mut m = vec![1f32; n * TEXT_LEN];
        for i in 0..n {
            for s in 0..TEXT_PART {
                f[(i * TEXT_LEN + s) * C..(i * TEXT_LEN + s) * C + C]
                    .copy_from_slice(&feats[(s * n + i) * C..(s * n + i) * C + C]);
                m[i * TEXT_LEN + s] = mask[i * TEXT_PART + s];
            }
            // s in TEXT_PART..TEXT_LEN：feats=0（已初始化），mask=1（已填充）
        }
        *self.prompt.lock().unwrap() = Some(DartPrompt {
            names: names.to_vec(),
            feats: f,
            mask: m,
        });
    }

    // ============================ 文本编码 ============================

    /// 编码英文类目（对应 `encodeText`）。
    ///
    /// # Returns
    /// - `feats`: text_feats [TEXT_PART, N, C]（seq 主序，模型实测布局）
    /// - `mask`: text_mask [N, TEXT_PART]（1 = padding）
    fn encode_text(&self, classes: &[String]) -> Result<(Vec<f32>, Vec<f32>)> {
        let n = classes.len();
        // 对应 tokenize：逐类分词，取前 TEXT_PART 个 id，不足补 0
        let mut flat = vec![0i64; n * TEXT_PART];
        for (i, class) in classes.iter().enumerate() {
            let encoding = self
                .tokenizer
                .encode(class.as_str(), true)
                .map_err(|e| VisionError::Tokenizer(format!("分词失败: {e}")))?;
            for (j, &id) in encoding.get_ids().iter().enumerate().take(TEXT_PART) {
                flat[i * TEXT_PART + j] = id as i64;
            }
        }

        // int64 输入张量直接 lock 会话跑（run_named 仅支持 f32）
        let ids_tensor = Tensor::from_array((vec![n as i64, TEXT_PART as i64], flat))?;
        let (feats, mask) = {
            let mut session = self.text_encoder.session.lock().unwrap();
            let outputs = session.run(ort::inputs![IN_INPUT_IDS => ids_tensor])?;
            (
                snapshot_f32_output(&outputs, OUT_TEXT_FEATS)?,
                snapshot_f32_output(&outputs, OUT_TEXT_MASK)?,
            )
        };
        let (_, feats) = feats;
        let (_, mask) = mask;
        expect_len(&feats, TEXT_PART * n * C, OUT_TEXT_FEATS)?;
        expect_len(&mask, n * TEXT_PART, OUT_TEXT_MASK)?;
        Ok((feats, mask))
    }

    // ============================ 几何编码 ============================

    /// 运行几何编码器（对应 `runGeometryEncoder`）：conv72 特征 + 文本特征 + 框
    /// → 概念 token。
    ///
    /// # Returns
    /// - `prompt`: [L*C]（L = 32 文本 + 几何 token 数）
    /// - `prompt_mask`: [L]（f32，1 = padding；模型输出为 BOOL）
    fn run_geometry_encoder(
        &self,
        geometry: &BaseOnnxEngine,
        conv72: &[f32],
        text_feats: &[f32],
        text_mask_float: &[f32],
        exemplar_w: usize,
        exemplar_h: usize,
        boxes: &[Rect],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let b = boxes.len();
        let mut box_arr = vec![0f32; b * 4];
        let label_arr = vec![true; b];
        let mask_arr = vec![true; b];
        for (i, r) in boxes.iter().enumerate() {
            // 框 → 归一化 cxcywh（示例图像素坐标）
            box_arr[i * 4] = (r.x as f32 + r.width as f32 / 2.0) / exemplar_w as f32;
            box_arr[i * 4 + 1] = (r.y as f32 + r.height as f32 / 2.0) / exemplar_h as f32;
            box_arr[i * 4 + 2] = r.width as f32 / exemplar_w as f32;
            box_arr[i * 4 + 3] = r.height as f32 / exemplar_h as f32;
        }
        // text_mask 在导出图中为 bool（True=padding），与文本编码器输出的 float 掩码按 >0.5 转换
        let text_mask_bool: Vec<bool> = text_mask_float.iter().map(|&v| v > 0.5).collect();

        let c_t = Tensor::from_array((vec![1, C as i64, P as i64, P as i64], conv72.to_vec()))?;
        let t_t = Tensor::from_array((
            vec![TEXT_PART as i64, 1, C as i64],
            text_feats.to_vec(),
        ))?;
        let m_t = Tensor::from_array((vec![1, TEXT_PART as i64], text_mask_bool))?;
        let b_t = Tensor::from_array((vec![b as i64, 1, 4], box_arr))?;
        let l_t = Tensor::from_array((vec![b as i64, 1], label_arr))?;
        let k_t = Tensor::from_array((vec![1, b as i64], mask_arr))?;

        let (prompt, pmask_bool) = {
            let mut session = geometry.session.lock().unwrap();
            let outputs = session.run(ort::inputs![
                IN_CONV72     => c_t,
                IN_TEXT_FEATS => t_t,
                IN_TEXT_MASK  => m_t,
                IN_BOXES      => b_t,
                IN_BOX_LABELS => l_t,
                IN_BOX_MASK   => k_t,
            ])?;
            (
                snapshot_f32_output(&outputs, OUT_PROMPT)?,
                snapshot_bool_output(&outputs, OUT_PROMPT_MASK)?,
            )
        };
        let (_, prompt) = prompt;
        let (_, pmask_bool) = pmask_bool;
        // prompt_mask 输出为 BOOL [1,L]（True=padding）；按导出契约读取为 bool
        // 再转 f32（Java readFloatsAuto 以 float 读 bool 张量为潜在缺陷，Rust 侧修正）
        let pmask = pmask_bool
            .iter()
            .map(|&v| if v { 1.0f32 } else { 0.0f32 })
            .collect();
        Ok((prompt, pmask))
    }

    // ============================ 基础设施 ============================

    /// 取出当前提示（无提示时返回错误，对应 `requirePrompt`）。
    fn require_prompt(&self) -> Result<DartPrompt> {
        self.prompt
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| VisionError::inference("先 set_classes(英文类名) 或 set_visual_prompts(...)"))
    }
}

// ============================ 自由函数辅助 ============================

/// 读取 float32 小端二进制文件为 f32 数组（对应 `readFloatsFile`）。
fn read_f32_le_file(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(VisionError::invalid_argument(format!(
            "{} 长度 {} 不是 4 的倍数（预期 float32 小端）",
            path.display(),
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// pos bin 布局 [P*P,1,C]（像素优先）→ [1,C,P,P] NCHW 单批（对应
/// `posEncNchwBatch` 的转置部分；沿 batch 复制 M 份在 detect_class 内做）。
fn pos_enc_nchw(pos_enc: &[f32]) -> Vec<f32> {
    let mut one = vec![0f32; C * P * P];
    for i in 0..P * P {
        for ch in 0..C {
            one[ch * P * P + i] = pos_enc[i * C + ch];
        }
    }
    one
}

/// 预处理（对应 Java `preprocess`）：缩放到 S×S（双线性）→ BGR→RGB →
/// (v/255 - 0.5) → CHW。
fn preprocess(image: &Image, size: usize) -> Result<Vec<f32>> {
    // 拉伸 resize（对应 resize(image, resized, Size(S,S), INTER_LINEAR)）
    let resized = resize(image, size, size, Interpolation::Linear)?;
    let rgb = match resized.channels() {
        3 => cvt_color(&resized, ColorConversion::Bgr2Rgb)?,
        4 => cvt_color(&resized, ColorConversion::Bgra2Rgb)?,
        _ => cvt_color(&resized, ColorConversion::Gray2Rgb)?,
    };

    let area = size * size;
    let mut out = vec![0f32; 3 * area];
    let px = rgb.data();
    for i in 0..area {
        out[i] = px[i * 3] as f32 / 255.0 - 0.5;
        out[area + i] = px[i * 3 + 1] as f32 / 255.0 - 0.5;
        out[2 * area + i] = px[i * 3 + 2] as f32 / 255.0 - 0.5;
    }
    Ok(out)
}

/// hs 布局 [6,M,200,C]：取全部 6 层 (batch 0, 查询 q) → [6*C]（对应 `sliceHs`）。
fn slice_hs(hs: &[f32], q: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(DEC_LAYERS * C);
    for l in 0..DEC_LAYERS {
        let src = ((l * MAX_CLASSES) * QUERIES + q) * C;
        out.extend_from_slice(&hs[src..src + C]);
    }
    out
}

/// enc_hs 布局 [P*P,M,C]：取 batch 0 → [P*P,C]（对应 `sliceBatch(enc, 0)`）。
fn slice_batch0(enc: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(ENC_HS_LEN * C);
    for i in 0..ENC_HS_LEN {
        let src = (i * MAX_CLASSES) * C;
        out.extend_from_slice(&enc[src..src + C]);
    }
    out
}

/// 取某类槽位的 [TEXT_LEN,C] 文本块（对应 `sliceText`；slot 主序布局下为连续块）。
fn slice_text(prompt: &[f32], slot: usize) -> Vec<f32> {
    let base = slot * TEXT_LEN * C;
    prompt[base..base + TEXT_LEN * C].to_vec()
}

/// 取某类槽位的 [TEXT_LEN] 掩码（对应 `sliceTextMask`）。
fn slice_text_mask(mask: &[f32], slot: usize) -> Vec<f32> {
    let base = slot * TEXT_LEN;
    mask[base..base + TEXT_LEN].to_vec()
}

/// slot 内按分数取 Top-K（≥ 阈值），K = min(max_keep, 达标数)
/// （对应 `topQueries`；固定 slot 0，索引即 0..QUERIES）。
fn top_queries(scores: &[f32], max_keep: usize, conf_threshold: f32) -> Vec<usize> {
    let mut order: Vec<usize> = (0..QUERIES).collect();
    order.sort_by(|&a, &b| {
        sigmoid(scores[b])
            .partial_cmp(&sigmoid(scores[a]))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut kept = Vec::new();
    for &idx in &order {
        if kept.len() >= max_keep {
            break;
        }
        if sigmoid(scores[idx]) < conf_threshold {
            break;
        }
        kept.push(idx);
    }
    kept
}

/// 跨瓦片同类去重（对应 `dedupByClass`）：按分数降序贪心，IoU > 阈值视为重复。
fn dedup_by_class(all: Vec<Segmentation>, iou_threshold: f32) -> Vec<Segmentation> {
    let mut sorted = all;
    sorted.sort_by(|a, b| {
        b.confidence()
            .partial_cmp(&a.confidence())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut kept: Vec<Segmentation> = Vec::new();
    for c in sorted {
        let mut drop = false;
        for k in &kept {
            if k.class_name() != c.class_name() {
                continue;
            }
            let (kx1, ky1, kx2, ky2) = (k.detection.x1(), k.detection.y1(), k.detection.x2(), k.detection.y2());
            let (cx1, cy1, cx2, cy2) = (c.detection.x1(), c.detection.y1(), c.detection.x2(), c.detection.y2());
            let ux1 = kx1.max(cx1);
            let uy1 = ky1.max(cy1);
            let ux2 = kx2.min(cx2);
            let uy2 = ky2.min(cy2);
            let inter = (ux2 - ux1).max(0.0) * (uy2 - uy1).max(0.0);
            let a1 = (kx2 - kx1) * (ky2 - ky1);
            let a2 = (cx2 - cx1) * (cy2 - cy1);
            if inter / (a1 + a2 - inter + 1e-9) > iou_threshold as f64 {
                drop = true;
                break;
            }
        }
        if !drop {
            kept.push(c);
        }
    }
    kept
}

/// 288×288 logits → 整图尺寸 f32 概率图（sigmoid），region 内有效、区域外为 0
/// （对应 `maskToOriginalMat`）。region 图在推理前被整体缩放到 1008×1008，
/// 因此掩码先放大到 region 尺寸再平移到原图 (origin_x, origin_y)。
#[allow(clippy::too_many_arguments)]
fn mask_to_original(
    logits: &[f32],
    region_w: i32,
    region_h: i32,
    origin_x: i32,
    origin_y: i32,
    full_w: i32,
    full_h: i32,
) -> Result<FloatMask> {
    let mut prob = vec![0f32; MASK_SZ * MASK_SZ];
    for (p, &l) in prob.iter_mut().zip(logits.iter()) {
        *p = sigmoid(l);
    }
    let small = FloatMask::from_raw(MASK_SZ, MASK_SZ, prob)?;
    // 288×288 → region 尺寸（双线性）
    let region_mask = small.resize(region_w as usize, region_h as usize);

    let mut full = FloatMask::new(full_w as usize, full_h as usize);
    let dx = origin_x.max(0);
    let dy = origin_y.max(0);
    let sx = (dx - origin_x) as usize;
    let sy = (dy - origin_y) as usize;
    let w = (region_w - (dx - origin_x)).min(full_w - dx);
    let h = (region_h - (dy - origin_y)).min(full_h - dy);
    if w > 0 && h > 0 {
        for yy in 0..h as usize {
            for xx in 0..w as usize {
                full.set(
                    dx as usize + xx,
                    dy as usize + yy,
                    region_mask.get(sx + xx, sy + yy),
                );
            }
        }
    }
    Ok(full)
}

/// 类/概念数校验（对应 `checkCount`）。
fn check_count(n: usize) -> Result<()> {
    if n == 0 || n > MAX_PROMPTS {
        return Err(VisionError::invalid_argument(format!(
            "类/概念数 1~{MAX_PROMPTS}（每次 enc-dec 内部按 slot 0 逐类循环）"
        )));
    }
    Ok(())
}

/// 元素数校验（输出快照长度不足时报错）。
fn expect_len(data: &[f32], expected: usize, what: &str) -> Result<()> {
    if data.len() < expected {
        return Err(VisionError::inference(format!(
            "{what} 元素数 {} < {expected}",
            data.len()
        )));
    }
    Ok(())
}

/// sigmoid（对应 `sigmoid`）。
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// 从会话输出按名取 f32 张量快照（复制出 native 内存，避免生命周期问题）。
fn snapshot_f32_output(
    outputs: &SessionOutputs<'_>,
    name: &str,
) -> Result<(Vec<i64>, Vec<f32>)> {
    let value = outputs
        .get(name)
        .ok_or_else(|| VisionError::inference(format!("missing output '{name}'")))?;
    let (shape, ty) = match value.dtype() {
        ort::value::ValueType::Tensor { ty, shape, .. } => {
            (shape.iter().copied().collect::<Vec<i64>>(), ty)
        }
        other => {
            return Err(VisionError::inference(format!(
                "输出 '{name}' 非张量类型: {other:?}"
            )))
        }
    };
    match ty {
        ort::tensor::TensorElementType::Float32 => {
            let (_, view) = value.try_extract_tensor::<f32>()?;
            Ok((shape, view.to_vec()))
        }
        other => Err(VisionError::inference(format!(
            "输出 '{name}' 元素类型 {other:?} 不受支持（预期 FLOAT32）"
        ))),
    }
}

/// 从会话输出按名取 bool 张量快照（几何编码器的 prompt_mask 用）。
fn snapshot_bool_output(
    outputs: &SessionOutputs<'_>,
    name: &str,
) -> Result<(Vec<i64>, Vec<bool>)> {
    let value = outputs
        .get(name)
        .ok_or_else(|| VisionError::inference(format!("missing output '{name}'")))?;
    let (shape, ty) = match value.dtype() {
        ort::value::ValueType::Tensor { ty, shape, .. } => {
            (shape.iter().copied().collect::<Vec<i64>>(), ty)
        }
        other => {
            return Err(VisionError::inference(format!(
                "输出 '{name}' 非张量类型: {other:?}"
            )))
        }
    };
    match ty {
        ort::tensor::TensorElementType::Bool => {
            let (_, view) = value.try_extract_tensor::<bool>()?;
            Ok((shape, view.to_vec()))
        }
        other => Err(VisionError::inference(format!(
            "输出 '{name}' 元素类型 {other:?} 不受支持（预期 BOOL）"
        ))),
    }
}

// ============================ 统一推理接口 ============================

/// 统一推理接口实现（参考 yolo_e_runtime 的做法；类别名由 set_classes /
/// set_visual_prompts 动态决定，trait 的引用型 `labels()` 恒为 None，
/// 请用 [`DartEngine::get_labels`]）。
#[::async_trait::async_trait]
impl crate::core::engine::OnnxInferenceEngine for DartEngine {
    type Output = Vec<Segmentation>;

    /// 单图推理（整图 1008 推理）。
    fn predict(&self, image: &Image) -> Result<Vec<Segmentation>> {
        DartEngine::predict(self, image)
    }

    /// 批量推理：逐张处理。
    fn predict_batch(&self, images: &[Image]) -> Result<Vec<Vec<Segmentation>>> {
        images.iter().map(|img| DartEngine::predict(self, img)).collect()
    }

    fn input_size(&self) -> (i32, i32) {
        self.backbone.input_size()
    }

    fn labels(&self) -> Option<&[String]> {
        None
    }

    /// 类别名由 `set_classes`/`set_visual_prompts` 决定，本方法为兼容统一 trait 的空实现。
    fn set_labels(&mut self, _labels: Vec<String>) {
        tracing::warn!("DartEngine 的类别名由 set_classes/set_visual_prompts 决定, set_labels 无效");
    }

    fn set_confidence_threshold(&mut self, threshold: f32) {
        self.confidence_threshold = threshold;
    }

    fn confidence_threshold(&self) -> f32 {
        self.confidence_threshold
    }
}

// 注：Java close() 释放 5 个会话 + tokenizer + env；Rust 侧 ort Session /
// Tokenizer 均为 RAII（Drop 自动释放），无需显式 close。

// ============================ 单元测试 ============================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_matches_java() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!((sigmoid(2.0) - 0.8807971).abs() < 1e-6);
        assert!((sigmoid(-2.0) - 0.11920292).abs() < 1e-6);
    }

    #[test]
    fn top_queries_filters_and_limits() {
        // 200 个查询，只有 3 个达标（sigmoid 后 >= 0.5）
        let mut scores = vec![-10f32; QUERIES];
        scores[7] = 3.0; // sigmoid ≈ 0.953
        scores[3] = 1.0; // sigmoid ≈ 0.731
        scores[100] = 0.0; // sigmoid = 0.5
        scores[50] = 0.5; // sigmoid ≈ 0.622
        let kept = top_queries(&scores, 10, 0.5);
        assert_eq!(kept.len(), 4);
        assert_eq!(kept[0], 7); // 分数降序
        assert_eq!(kept[1], 3);
        // max_keep 截断
        assert_eq!(top_queries(&scores, 2, 0.5).len(), 2);
        // 阈值截断（0.9 → 只留 sigmoid(3.0)）
        assert_eq!(top_queries(&scores, 10, 0.9), vec![7]);
    }

    #[test]
    fn slice_hs_takes_all_layers_of_query() {
        // [6,4,200,256]，batch 0 / 查询 1：((l*4+0)*200+1)*256
        let mut hs = vec![0f32; DEC_LAYERS * MAX_CLASSES * QUERIES * C];
        for l in 0..DEC_LAYERS {
            hs[((l * MAX_CLASSES) * QUERIES + 1) * C] = l as f32;
        }
        let out = slice_hs(&hs, 1);
        assert_eq!(out.len(), DEC_LAYERS * C);
        for l in 0..DEC_LAYERS {
            assert_eq!(out[l * C], l as f32);
        }
    }

    #[test]
    fn pos_enc_nchw_transposes_pixel_major() {
        // 输入 [P*P,1,C] 像素优先：pos[i*C+ch]；输出 NCHW：one[ch*P*P+i]
        let mut pos = vec![0f32; P * P * C];
        let (i, ch) = (5usize, 7usize);
        pos[i * C + ch] = 42.0;
        let nchw = pos_enc_nchw(&pos);
        assert_eq!(nchw[ch * P * P + i], 42.0);
        assert_eq!(nchw.iter().filter(|&&v| v != 0.0).count(), 1);
    }

    #[test]
    fn check_count_bounds() {
        assert!(check_count(1).is_ok());
        assert!(check_count(16).is_ok());
        assert!(check_count(0).is_err());
        assert!(check_count(17).is_err());
    }

    #[test]
    fn dedup_by_class_same_class_iou() {
        let a = Segmentation::new("cat", 0, 0.0, 0.0, 100.0, 100.0, 0.9, None);
        let b = Segmentation::new("cat", 0, 10.0, 10.0, 110.0, 110.0, 0.8, None); // IoU ≈ 0.68
        let c = Segmentation::new("dog", 1, 10.0, 10.0, 110.0, 110.0, 0.7, None);
        let kept = dedup_by_class(vec![a, b, c], 0.5);
        // 同类 cat 高 IoU 去重 → 保 a；dog 不同类保留
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].class_name(), "cat");
        assert_eq!(kept[1].class_name(), "dog");
    }

    #[test]
    fn preprocess_layout_is_chw_rgb_minus_half() {
        // 1x1 纯蓝图（BGR: [255,0,0]）→ RGB [0,0,255] → R=0-0.5, G=-0.5, B=1-0.5
        let img = Image::from_raw(1, 1, 3, vec![255, 0, 0]).unwrap();
        let out = preprocess(&img, 1).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], -0.5); // R
        assert_eq!(out[1], -0.5); // G
        assert!((out[2] - 0.5).abs() < 1e-6); // B
    }

    #[test]
    fn mask_to_original_offsets_into_full_canvas() {
        // 288 全 1 logits（sigmoid≈1）→ region 100x100 全 1 → 平移到 (10,20)
        let logits = vec![10f32; MASK_SZ * MASK_SZ];
        let mask = mask_to_original(&logits, 100, 100, 10, 20, 200, 200).unwrap();
        assert_eq!(mask.width(), 200);
        assert_eq!(mask.height(), 200);
        assert!(mask.get(15, 25) > 0.99);
        assert_eq!(mask.get(5, 25), 0.0); // 区域外为 0
        assert_eq!(mask.get(15, 10), 0.0);
    }

    // ==================== 端到端（需模型资产，耗时数分钟） ====================

    /// 模型目录（对应 Java 测试的 DART_MODELS_DIR 约定）。
    fn dart_models_dir() -> std::path::PathBuf {
        std::env::var("DART_MODELS_DIR")
            .unwrap_or_else(|_| "/Volumes/macEx/AI/vision-commons/models/dart".to_string())
            .into()
    }

    fn assets_available() -> bool {
        let dir = dart_models_dir();
        ["tokenizer.json", "text_encoder.onnx", "enc_dec_hs.onnx", "mask_head.onnx"]
            .iter()
            .all(|f| dir.join(f).exists())
            && dir.join("tier_1008/hf_backbone.onnx").exists()
    }

    /// 端到端冒烟：set_classes(["bottle"]) + predict（管线完整性与掩码尺寸校验）。
    ///
    /// ```text
    /// DART_MODELS_DIR=/path/to/dart cargo test --lib dart::tests::e2e -- --ignored --nocapture
    /// ```
    #[test]
    fn e2e_set_classes_then_predict() {
        if !assets_available() {
            eprintln!("跳过：DART 模型资产缺失（{}）", dart_models_dir().display());
            return;
        }
        // 日志（失败忽略：可能已被其他测试初始化）
        tracing_subscriber::fmt().try_init().ok();

        let engine = DartEngine::new(dart_models_dir(), DeviceType::Cpu).unwrap();
        assert!(engine.has_geometry_encoder());
        engine.set_classes(&["bottle"]).unwrap();
        assert_eq!(engine.get_labels(), vec!["bottle".to_string()]);

        let image = Image::load("testmodels/bus.jpg").unwrap();
        let results = engine.predict(&image).unwrap();
        println!("检出 {} 条", results.len());
        for r in &results {
            println!("{r}");
        }
        // 管线完整性：检出（可为 0 条）且掩码为原图尺寸概率图
        for r in &results {
            assert_eq!(r.class_name(), "bottle");
            let mask = r.mask.as_ref().expect("检出必须带掩码");
            assert_eq!(mask.width(), image.width());
            assert_eq!(mask.height(), image.height());
        }
    }

    /// 多类端到端（对应 Java `DartEngineTest.fullImageDetectsAllSixClassesWithMasks`）：
    /// door_right.jpg（2448x2048）+ 6 类，参考结果（Python 参考实现 full_run.log，
    /// Windows+CUDA，每类 Top-1 含掩码）：door latch 0.784 / rubber stopper 0.774 /
    /// storage box 0.726 / black wire harness 0.678 / door trim panel 0.641 /
    /// check link 0.605（CPU 与 CUDA 及插值实现的正常差异在 ±0.05 内）。
    ///
    /// 本用例同时验证 pack_slots 对 [32,N,256] 真实布局的跨步打包（N=6 时
    /// 逐类分数仍与参考一致）。
    #[test]
    fn e2e_six_classes_reference_scores() {
        if !assets_available() {
            eprintln!("跳过：DART 模型资产缺失（{}）", dart_models_dir().display());
            return;
        }
        tracing_subscriber::fmt().try_init().ok();

        let test_image = std::env::var("DART_TEST_IMAGE")
            .unwrap_or_else(|_| {
                dart_models_dir()
                    .join("test_images/door_right.jpg")
                    .display()
                    .to_string()
            });
        let image = Image::load(&test_image).unwrap();

        let classes = [
            "door trim panel",
            "storage box",
            "door latch",
            "rubber stopper",
            "black wire harness",
            "check link",
        ];
        let engine = DartEngine::new(dart_models_dir(), DeviceType::Cpu).unwrap();
        engine.set_classes(&classes).unwrap();
        assert_eq!(engine.get_labels().len(), 6);

        let results = engine.predict(&image).unwrap();
        println!("检出 {} 条", results.len());
        for r in &results {
            println!("{r}");
        }
        // 每个类目至少一条检出（Java 测试断言语义），掩码为原图尺寸
        for name in classes {
            let hit = results.iter().find(|r| r.class_name() == name);
            assert!(hit.is_some(), "类目 {name} 无检出");
            let hit = hit.unwrap();
            let mask = hit.mask.as_ref().expect("检出必须带掩码");
            assert_eq!(mask.width(), image.width());
            assert_eq!(mask.height(), image.height());
        }
    }
}
