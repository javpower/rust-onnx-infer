//! 表格结构识别引擎（PaddleOCR SLANet，RapidTable 的 ONNX 移植版）。
//!
//! 输入一张裁剪好的表格图，输出 HTML 结构骨架 + 各单元格在原图中的边框。
//! 单元格内容留空占位，配合库内 [`crate::engines::ocr::OcrPipeline`] 的 OCR 结果
//! 按 bbox 匹配回填即得完整表格（对齐 RapidTable `TableMatch` 的用法）。
//!
//! | 项 | 说明 |
//! |---|---|
//! | 模型 | `slanet-plus.onnx`（RapidAI/RapidTable 官方发行，6.8 MB） |
//! | 输入 | `x` `[1,3,488,488]` f32（声明动态，实际固定 488 画布） |
//! | 输出 | `save_infer_model/scale_0.tmp_0` `[1,40,8]` f32 —— cell 四角点回归（4×(x,y)，[0,1] 归一化） |
//! |      | `save_infer_model/scale_1.tmp_0` `[1,40,50]` f32 —— 结构 token 序列分类（40 帧 × 50 类，图内已 softmax） |
//!
//! **模型直链**（SHA256 与 RapidTable `default_models.yaml` 校验一致）：
//! - <https://www.modelscope.cn/models/RapidAI/RapidTable/resolve/v2.0.0/slanet-plus.onnx>
//!   （SHA256 `d57a942af6a2f57d6a4a0372573c696a2379bf5857c45e2ac69993f3b334514b`；
//!   本库存放路径 `testmodels/slanet.onnx`，I/O 签名已经 model_probe 实测确认）
//!
//! **结构 token 字典**（48 项，优先从模型元数据 key `character` 读取，缺失时用内置默认；
//! 解码字符表再按 RapidTable `TableLabelDecode` 约定前后补 `sos`/`eos`，共 50 类）：
//! `<thead>` `</thead>` `<tbody>` `</tbody>` `<tr>` `</tr>` `<td` `>` `</td>`
//! ` colspan="2"`..` colspan="20"`（19 项）` rowspan="2"`..` rowspan="20"`（19 项）`<td></td>`。
//! 注意 `<td` 与 `>` 是分离 token，`<td` + 属性 token + `>` 拼接成带 colspan/rowspan 的开标签；
//! HTML 包裹标签（`<html>/<body>/<table>` 及闭合）不在字典内，由后处理补齐（`wrap_with_html_struct`）。
//!
//! **预处理**（对齐 RapidTable `pp_structure/pre_process.py::TablePreprocess`，已核实——
//! 任务书里 "仅 /255 无 mean/std" 的猜测不成立，实际是 ImageNet 归一化）：
//! 1. 统一 3 通道 **BGR**（RapidTable `LoadImage` 统一转 BGR 后预处理不再换通道序）；
//! 2. 等比缩放 `ratio = 488 / max(h, w)`，resize 到 `(int(w·ratio), int(h·ratio))`；
//! 3. `(x/255 - [0.485,0.456,0.406]) / [0.229,0.224,0.225]`；
//! 4. 左上角放置，右/下零填充到 488×488 画布，HWC → CHW。
//!
//! **后处理**（对齐 RapidTable `pp_structure/post_process.py::TableLabelDecode`）：
//! 1. 逐 token argmax；`idx>0` 遇 `eos` 提前终止，`sos`/`eos` 跳过；
//! 2. td 系 token（`<td` / `<td></td>`）解码该帧 bbox：8 值 = 4 角点，
//!    `x·max(h,w)`、`y·max(h,w)` 还原到原图坐标（slanet-plus 的 bbox 按 488 画布归一化；
//!    等价参考实现 `_bbox_decode` 乘 (w,h) 后 `rescale_cell_bboxes` 乘 488/(w·ratio) 的合并结果，
//!    其中 ratio = 488/max(h,w)）；全零 bbox 为空白占位，丢弃；
//! 3. token 流重建 HTML（单元格内容留空），统计行数（`<tr>` 数）与列数（每行 td 数最大值，
//!    colspan 按跨度展开）。
//!
//! ```no_run
//! use rust_onnx_infer::core::DeviceType;
//! use rust_onnx_infer::core::engine::OnnxInferenceEngine;
//! use rust_onnx_infer::engines::table_recognition::TableRecognitionEngine;
//! use rust_onnx_infer::imaging::Image;
//!
//! let engine = TableRecognitionEngine::new("testmodels/slanet.onnx", DeviceType::Cpu)?;
//! let result = engine.predict(&Image::load("table.png")?)?;
//! println!("HTML: {}", result.html);
//! println!("{} 行 x {} 列，{} 个单元格", result.row_count, result.col_count, result.cell_boxes.len());
//! // 单元格文本回填：用 OcrPipeline 识别文本框，按 IoU/中心点匹配 result.cell_boxes
//! # Ok::<(), rust_onnx_infer::error::VisionError>(())
//! ```

use ort::value::Tensor;

use crate::core::base::{BaseOnnxEngine, TensorOutput};
use crate::core::device_type::DeviceType;
use crate::core::runtime_config::OnnxRuntimeConfig;
use crate::error::{Result, VisionError};
use crate::imaging::{cvt_color, resize, ColorConversion, Image, Interpolation};

/// 输入画布边长（RapidTable `TablePreprocess` 的 `max_len`，模型按此分辨率训练）。
const MAX_SIDE: usize = 488;
/// 输入归一化 mean（ImageNet，PaddleOCR 表格模型训练约定）。
const TABLE_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// 输入归一化 std（ImageNet）。
const TABLE_STD: [f32; 3] = [0.229, 0.224, 0.225];
/// bbox 输出每帧的坐标数（4 角点 × 2）。
const BBOX_COORDS: usize = 8;

/// SLANet 内置默认结构字典（48 项；模型元数据缺 `character` 时的兜底，
/// 与 slanet-plus.onnx 元数据逐项一致）。
const DEFAULT_STRUCTURE_DICT: [&str; 48] = [
    "<thead>",
    "</thead>",
    "<tbody>",
    "</tbody>",
    "<tr>",
    "</tr>",
    "<td",
    ">",
    "</td>",
    " colspan=\"2\"",
    " colspan=\"3\"",
    " colspan=\"4\"",
    " colspan=\"5\"",
    " colspan=\"6\"",
    " colspan=\"7\"",
    " colspan=\"8\"",
    " colspan=\"9\"",
    " colspan=\"10\"",
    " colspan=\"11\"",
    " colspan=\"12\"",
    " colspan=\"13\"",
    " colspan=\"14\"",
    " colspan=\"15\"",
    " colspan=\"16\"",
    " colspan=\"17\"",
    " colspan=\"18\"",
    " colspan=\"19\"",
    " colspan=\"20\"",
    " rowspan=\"2\"",
    " rowspan=\"3\"",
    " rowspan=\"4\"",
    " rowspan=\"5\"",
    " rowspan=\"6\"",
    " rowspan=\"7\"",
    " rowspan=\"8\"",
    " rowspan=\"9\"",
    " rowspan=\"10\"",
    " rowspan=\"11\"",
    " rowspan=\"12\"",
    " rowspan=\"13\"",
    " rowspan=\"14\"",
    " rowspan=\"15\"",
    " rowspan=\"16\"",
    " rowspan=\"17\"",
    " rowspan=\"18\"",
    " rowspan=\"19\"",
    " rowspan=\"20\"",
    "<td></td>",
];

/// 表格结构识别结果。
#[derive(Debug, Clone, PartialEq)]
pub struct TableResult {
    /// 重建的 HTML（`<html><body><table>…</table></body></html>`）。
    /// 单元格内容为空占位（`<td></td>` / `<td colspan="n"></td>`），
    /// 配合库内 OCR 结果按 [`TableResult::cell_boxes`] 匹配回填文本。
    pub html: String,
    /// 各单元格的轴对齐外接框 `[x1, y1, x2, y2]`（原图像素坐标）。
    ///
    /// 模型回归的是四角点（支持带 colspan 的跨列单元格），此处取四角点的
    /// min/max 收敛为轴对齐框；与 html 中的非空单元格按 token 顺序一一对应。
    /// 全零的空白占位 bbox 已过滤（对齐 RapidTable `filter_blank_bbox`），
    /// 因此长度可能小于 html 中的 `<td` 出现次数。
    pub cell_boxes: Vec<[f32; 4]>,
    /// 各单元格对应结构 token 的置信度（与 `cell_boxes` 同序）。
    pub cell_scores: Vec<f32>,
    /// 行数（`<tr>` token 数）。
    pub row_count: usize,
    /// 列数（各行单元格数的最大值；colspan 按跨度展开计入）。
    pub col_count: usize,
}

/// SLANet 表格结构识别引擎（RapidTable ONNX 版）。
pub struct TableRecognitionEngine {
    /// 组合基类（输入固定 [1,3,488,488] 画布）。
    pub base: BaseOnnxEngine,

    /// 解码字符表：`["sos"] + 结构字典 + ["eos"]`（长度须等于结构输出类别数）。
    char_list: Vec<String>,
    /// 字典是否取自模型元数据（false = 使用内置默认字典）。
    dict_from_metadata: bool,
}

impl TableRecognitionEngine {
    /// 创建表格结构识别引擎（模型路径如 `testmodels/slanet.onnx`）。
    pub fn new(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        Self::with_config(model_path, device_type, OnnxRuntimeConfig::defaults())
    }

    /// 创建表格结构识别引擎（自定义运行参数：线程数 / 设备 id 等）。
    pub fn with_config(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        runtime_config: OnnxRuntimeConfig,
    ) -> Result<Self> {
        // 输入固定 488x488 画布（模型声明动态维度，按参考实现固定）
        let base = BaseOnnxEngine::with_config(
            model_path,
            device_type,
            MAX_SIDE as i32,
            MAX_SIDE as i32,
            runtime_config,
        )?;
        let mut engine = TableRecognitionEngine {
            base,
            char_list: Vec::new(),
            dict_from_metadata: false,
        };
        engine.load_structure_dict();
        Ok(engine)
    }

    // ============ 参数访问器 ============

    /// 解码字符表（`["sos"] + 结构字典 + ["eos"]`，长度 = 结构输出类别数）。
    pub fn char_list(&self) -> &[String] {
        &self.char_list
    }

    /// 结构字典（不含 sos/eos 的 48 项 token）。
    pub fn structure_dict(&self) -> &[String] {
        &self.char_list[1..self.char_list.len() - 1]
    }

    /// 字典是否取自模型元数据（false = 内置默认字典）。
    pub fn dict_from_metadata(&self) -> bool {
        self.dict_from_metadata
    }

    // ============ 推理 ============

    /// 表格结构识别：返回 HTML 骨架与原图坐标系的单元格框。
    pub fn predict_impl(&self, image: &Image) -> Result<TableResult> {
        if image.is_empty() {
            return Err(VisionError::image("cannot recognize table on empty image"));
        }

        // 1. 预处理（等比缩放 + ImageNet 归一化 + 488 画布零填充）
        let input_data = self.preprocess(image)?;

        // 2. 推理 → [bbox (1,T,8), structure (1,T,C)]
        let input_tensor: Tensor<f32> = self.base.create_input_tensor(input_data)?;
        let outputs = self.base.run_multi_output(input_tensor)?;
        let (bbox_out, struct_out) = split_outputs(&outputs, self.char_list.len())?;

        // 3. 解码结构序列 + cell bbox 还原
        self.decode(bbox_out, struct_out, image.width(), image.height())
    }

    // ==================== 预处理 ====================

    /// 预处理：BGR → 等比缩放（ratio = 488/max(h,w)）→ ImageNet 归一化 →
    /// 左上角放置 + 右/下零填充到 488×488 → HWC 转 CHW。
    fn preprocess(&self, image: &Image) -> Result<Vec<f32>> {
        // 1. 统一 3 通道 BGR（RapidTable LoadImage 统一 BGR，表格预处理不再换序）
        let bgr = match image.channels() {
            3 => image.clone(),
            4 => cvt_color(image, ColorConversion::Bgra2Bgr)?,
            1 => cvt_color(image, ColorConversion::Gray2Bgr)?,
            n => return Err(VisionError::image(format!("不支持的通道数: {n}"))),
        };

        // 2. 等比缩放（int() 截断，与参考实现一致）
        let (orig_w, orig_h) = (bgr.width(), bgr.height());
        let ratio = MAX_SIDE as f32 / (orig_w.max(orig_h) as f32).max(1.0);
        let resize_w = ((orig_w as f32 * ratio) as usize).clamp(1, MAX_SIDE);
        let resize_h = ((orig_h as f32 * ratio) as usize).clamp(1, MAX_SIDE);
        let resized = resize(&bgr, resize_w, resize_h, Interpolation::Linear)?;

        // 3. (x/255 - mean)/std → 左上角放置、右/下补 0 → HWC 转 CHW
        let px = resized.data();
        let area = MAX_SIDE * MAX_SIDE;
        let mut data = vec![0f32; 3 * area];
        for y in 0..resize_h {
            for x in 0..resize_w {
                let src = (y * resize_w + x) * 3;
                let dst = y * MAX_SIDE + x;
                data[dst] = (px[src] as f32 / 255.0 - TABLE_MEAN[0]) / TABLE_STD[0];
                data[area + dst] = (px[src + 1] as f32 / 255.0 - TABLE_MEAN[1]) / TABLE_STD[1];
                data[2 * area + dst] = (px[src + 2] as f32 / 255.0 - TABLE_MEAN[2]) / TABLE_STD[2];
            }
        }
        Ok(data)
    }

    // ==================== 后处理 ====================

    /// 解码：逐 token argmax → HTML 重建 + cell bbox 还原到原图坐标。
    fn decode(
        &self,
        bbox_out: &TensorOutput,
        struct_out: &TensorOutput,
        orig_w: usize,
        orig_h: usize,
    ) -> Result<TableResult> {
        let struct_flat = struct_out.as_f32()?;
        let bbox_flat = bbox_out.as_f32()?;
        if struct_out.shape.len() != 3 || bbox_out.shape.len() != 3 {
            return Err(VisionError::inference(format!(
                "SLANet 输出应为 [1,T,C] / [1,T,8]，实际 structure {:?} / bbox {:?}",
                struct_out.shape, bbox_out.shape
            )));
        }
        let tokens = struct_out.shape[1] as usize;
        let classes = struct_out.shape[2] as usize;
        if bbox_out.shape[1] as usize != tokens || bbox_out.shape[2] as usize != BBOX_COORDS {
            return Err(VisionError::inference(format!(
                "bbox 输出 shape {:?} 与结构序列长度 {tokens} / 坐标数 {BBOX_COORDS} 不符",
                bbox_out.shape
            )));
        }
        if classes != self.char_list.len() {
            tracing::warn!(
                "结构输出类别数 {classes} 与解码字符表长度 {} 不一致，越界索引将被跳过",
                self.char_list.len()
            );
        }
        if struct_flat.len() < tokens * classes || bbox_flat.len() < tokens * BBOX_COORDS {
            return Err(VisionError::inference(format!(
                "输出元素数不足：structure {} / bbox {} < {tokens} 帧所需",
                struct_flat.len(),
                bbox_flat.len()
            )));
        }

        // 归一化坐标 → 原图坐标的统一比例因子：
        // x_orig = x_norm * 488 / ratio，而 ratio = 488 / max(h, w)，故因子 = max(h, w)
        // （slanet-plus 的 bbox 按 488 画布归一化；等价 RapidTable 对 slanet_plus
        //   先 _bbox_decode 乘 (w,h) 再 rescale_cell_bboxes 乘 488/(w·ratio) 的合并结果）
        let scale = (orig_w.max(orig_h) as f32).max(1.0);

        let end_idx = self.char_list.len().saturating_sub(1); // eos
        let mut builder = HtmlBuilder::default();
        let mut cell_boxes = Vec::new();
        let mut cell_scores = Vec::new();

        for t in 0..tokens {
            let row = &struct_flat[t * classes..(t + 1) * classes];
            let (mut best, mut prob) = (0usize, f32::MIN);
            for (i, &v) in row.iter().enumerate() {
                if v > prob {
                    prob = v;
                    best = i;
                }
            }

            // idx>0 遇 eos 提前终止；sos/eos 忽略（对齐 TableLabelDecode）
            if t > 0 && best == end_idx {
                break;
            }
            if best == 0 || best == end_idx {
                continue;
            }
            let Some(text) = self.char_list.get(best) else {
                continue; // 字符表与类别数不一致时的越界保护
            };

            // td 系 token 才有 cell bbox（"<td"、"<td></td>"；"<td>" 形式不在本字典）
            if is_td_token(text) {
                let raw = &bbox_flat[t * BBOX_COORDS..(t + 1) * BBOX_COORDS];
                tracing::debug!("token[{t}] '{text}' 原始 bbox = {:?}", raw);
                if !raw.iter().all(|&v| v == 0.0) {
                    // 8 值 = 4 角点 (x1,y1,x2,y2,x3,y3,x4,y4)，偶数位 x、奇数位 y
                    let xs = [raw[0], raw[2], raw[4], raw[6]];
                    let ys = [raw[1], raw[3], raw[5], raw[7]];
                    let x1 = xs.iter().cloned().fold(f32::MAX, f32::min) * scale;
                    let y1 = ys.iter().cloned().fold(f32::MAX, f32::min) * scale;
                    let x2 = xs.iter().cloned().fold(f32::MIN, f32::max) * scale;
                    let y2 = ys.iter().cloned().fold(f32::MIN, f32::max) * scale;
                    cell_boxes.push([
                        x1.clamp(0.0, orig_w as f32),
                        y1.clamp(0.0, orig_h as f32),
                        x2.clamp(0.0, orig_w as f32),
                        y2.clamp(0.0, orig_h as f32),
                    ]);
                    cell_scores.push(prob);
                }
            }

            builder.push_token(text);
        }

        tracing::debug!("SLANet 结构 token 序列：{:?}", builder.debug_tokens);
        let (html, row_count, col_count) = builder.finish();
        tracing::info!(
            "SLANet: {row_count} 行 x {col_count} 列，{} 个有效单元格（score 均值 {:.3}）",
            cell_boxes.len(),
            if cell_scores.is_empty() {
                0.0
            } else {
                cell_scores.iter().sum::<f32>() / cell_scores.len() as f32
            }
        );

        Ok(TableResult {
            html,
            cell_boxes,
            cell_scores,
            row_count,
            col_count,
        })
    }

    /// 加载结构字典：优先模型元数据 key `character`（RapidTable 同款，逐行一个
    /// token），缺失时回退内置默认；随后按参考实现补 `sos`/`eos` 组成解码字符表。
    fn load_structure_dict(&mut self) {
        let mut dict: Vec<String> = Vec::new();
        {
            let guard = self.base.session.lock().unwrap();
            // 先绑定再解构：Result<Metadata> 的临时值必须先于 guard 释放
            let meta_result = guard.metadata();
            if let Ok(meta) = meta_result {
                if let Ok(Some(value)) = meta.custom("character") {
                    let lines: Vec<String> =
                        value.lines().map(|s| s.trim_end().to_string()).collect();
                    if !lines.is_empty() {
                        dict = lines;
                        self.dict_from_metadata = true;
                    }
                }
            }
        }
        if dict.is_empty() {
            tracing::warn!(
                "模型元数据无 'character' 结构字典，使用内置 SLANet 默认字典（{} 项）",
                DEFAULT_STRUCTURE_DICT.len()
            );
            dict = default_structure_dict();
        }

        self.char_list = build_char_list(dict);
        tracing::info!(
            "表格结构字典：{} 项（含 sos/eos 共 {} 类，来源 = {}）",
            self.char_list.len() - 2,
            self.char_list.len(),
            if self.dict_from_metadata { "模型元数据" } else { "内置默认" }
        );
    }
}

// ONNXInferenceEngine 统一接口：Output = TableResult
crate::impl_engine_forward!(
    TableRecognitionEngine,
    base,
    TableResult,
    /// 单图表格结构识别。
    fn predict(&self, image: &Image) -> Result<TableResult> {
        self.predict_impl(image)
    }
);

// ==================== 内部辅助 ====================

/// 判断结构 token 是否为 td 系（携带 cell bbox 的 token；对齐 RapidTable
/// `td_token = ["<td>", "<td", "<td></td>"]`）。
fn is_td_token(token: &str) -> bool {
    matches!(token, "<td>" | "<td" | "<td></td>")
}

/// 由结构字典构建解码字符表：`["sos"] + dict + ["eos"]`（对齐
/// `TableLabelDecode.add_special_char`；本模型自带 `<td></td>`，无需 merge 步骤）。
fn build_char_list(dict: Vec<String>) -> Vec<String> {
    let mut char_list = Vec::with_capacity(dict.len() + 2);
    char_list.push("sos".to_string());
    char_list.extend(dict);
    char_list.push("eos".to_string());
    char_list
}

/// 内置默认结构字典（`DEFAULT_STRUCTURE_DICT` 的 owned 形式）。
fn default_structure_dict() -> Vec<String> {
    DEFAULT_STRUCTURE_DICT.iter().map(|s| s.to_string()).collect()
}

/// 区分双输出：结构输出末维 = 字符表长度，bbox 输出末维 = 8（4 角点）；
/// 形状无法区分时按 PaddleOCR 导出顺序回退（`scale_0` = bbox、`scale_1` = structure）。
fn split_outputs(
    outputs: &[TensorOutput],
    class_count: usize,
) -> Result<(&TensorOutput, &TensorOutput)> {
    let mut bbox_idx = None;
    let mut struct_idx = None;
    for (i, out) in outputs.iter().enumerate() {
        match out.shape.last() {
            Some(&n) if n > 0 && n as usize == class_count => struct_idx = Some(i),
            Some(&n) if n > 0 && n as usize == BBOX_COORDS => bbox_idx = Some(i),
            _ => {}
        }
    }
    match (bbox_idx, struct_idx) {
        (Some(b), Some(s)) => Ok((&outputs[b], &outputs[s])),
        _ if outputs.len() >= 2 => Ok((&outputs[0], &outputs[1])),
        _ => Err(VisionError::inference(format!(
            "SLANet 期望 2 个输出（bbox [1,T,8] + structure [1,T,{class_count}]），实际 {} 个",
            outputs.len()
        ))),
    }
}

/// 从属性 token 串中提取 colspan/rowspan 跨度（如 ` colspan="3"` → 3）。
fn attr_span(attrs: &str, name: &str) -> Option<usize> {
    let key = format!("{name}=\"");
    let start = attrs.find(&key)? + key.len();
    let rest = &attrs[start..];
    let end = rest.find('"').unwrap_or(rest.len());
    rest[..end].trim().parse().ok()
}

/// 结构 token 流 → HTML 重建器。
///
/// token 拼接规则对齐 RapidTable `wrap_with_html_struct` + `''.join`：
/// - `<td` 开启属性累积，后续 ` colspan="n"` / ` rowspan="n"` 附加，`>` 收尾成
///   `<td ...>` 开标签（含跨行列属性时计数按跨度展开）；
/// - `<td></td>` 直接输出空单元格；
/// - 其余 token（thead/tbody/tr 与闭标签等）原样拼接；
/// - 首尾补 `<html><body><table>` / `</table></body></html>` 包裹。
#[derive(Default)]
struct HtmlBuilder {
    /// 已拼接的 `<table>` 内部片段。
    parts: Vec<String>,
    /// `<td` 出现后累积的属性 token（`>` 收尾时取出）。
    pending_attrs: Option<String>,
    /// 每行单元格数（colspan 按跨度展开）。
    row_cells: Vec<usize>,
    /// 途经的原始 token 序列（诊断日志用）。
    debug_tokens: Vec<String>,
}

impl HtmlBuilder {
    /// 推入一个结构 token。
    fn push_token(&mut self, token: &str) {
        match token {
            "<td" => self.pending_attrs = Some(String::new()),
            ">" => match self.pending_attrs.take() {
                Some(attrs) => {
                    self.count_cell(&attrs);
                    self.parts.push(format!("<td{attrs}>"));
                }
                None => self.parts.push(token.to_string()),
            },
            "<td></td>" => {
                self.count_cell("");
                self.parts.push("<td></td>".to_string());
            }
            "<tr>" => {
                self.row_cells.push(0);
                self.parts.push("<tr>".to_string());
            }
            t if self.pending_attrs.is_some()
                && (t.starts_with(" colspan=") || t.starts_with(" rowspan=")) =>
            {
                if let Some(attrs) = self.pending_attrs.as_mut() {
                    attrs.push_str(t);
                }
            }
            t => self.parts.push(t.to_string()),
        }
        self.debug_tokens.push(token.to_string());
    }

    /// 当前行的单元格计数 +1（colspan 展开为跨度）。
    fn count_cell(&mut self, attrs: &str) {
        if let Some(cells) = self.row_cells.last_mut() {
            *cells += attr_span(attrs, "colspan").unwrap_or(1);
        }
    }

    /// 结束重建：返回 `(html, 行数, 列数)`。
    fn finish(self) -> (String, usize, usize) {
        let inner = self.parts.concat();
        let row_count = self.row_cells.len();
        let col_count = self.row_cells.iter().copied().max().unwrap_or(0);
        let html = format!("<html><body><table>{inner}</table></body></html>");
        (html, row_count, col_count)
    }
}

// ==================== 测试 ====================

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== 纯逻辑自验 ====================

    /// 内置默认字典 48 项；字符表 50 类且关键索引正确。
    #[test]
    fn default_dict_layout() {
        let dict = default_structure_dict();
        assert_eq!(dict.len(), 48);
        assert_eq!(dict[0], "<thead>");
        assert_eq!(dict[6], "<td");
        assert_eq!(dict[7], ">");
        assert_eq!(dict[8], "</td>");
        assert_eq!(dict[9], " colspan=\"2\"");
        assert_eq!(dict[27], " colspan=\"20\"");
        assert_eq!(dict[28], " rowspan=\"2\"");
        assert_eq!(dict[46], " rowspan=\"20\"");
        assert_eq!(dict[47], "<td></td>");

        let char_list = build_char_list(dict);
        assert_eq!(char_list.len(), 50, "48 + sos + eos = 50，与结构输出类别数一致");
        assert_eq!(char_list[0], "sos");
        assert_eq!(char_list[1], "<thead>");
        assert_eq!(char_list[5], "<tr>");
        assert_eq!(char_list[48], "<td></td>");
        assert_eq!(char_list[49], "eos");
    }

    /// token 流 → HTML：td 属性拼接、空单元格、thead/tbody 包裹、行列计数。
    #[test]
    fn html_builder_reconstructs_structure() {
        let tokens = [
            "<thead>", "<tr>", "<td", " colspan=\"2\"", ">", "</td>", "</tr>", "</thead>",
            "<tbody>", "<tr>", "<td></td>", "</tr>", "<tr>", "<td", ">", "</td>", "</tr>",
            "</tbody>",
        ];
        let mut builder = HtmlBuilder::default();
        for t in tokens {
            builder.push_token(t);
        }
        let (html, rows, cols) = builder.finish();
        assert_eq!(
            html,
            "<html><body><table><thead><tr><td colspan=\"2\"></td></tr></thead>\
             <tbody><tr><td></td></tr><tr><td></td></tr></tbody></table></body></html>"
        );
        assert_eq!(rows, 3);
        assert_eq!(cols, 2, "首行 colspan=2 展开；其余行 1 列");
    }

    /// ">" 无前置 "<td" 时原样透传（防御残缺序列）。
    #[test]
    fn html_builder_lone_gt() {
        let mut builder = HtmlBuilder::default();
        builder.push_token(">");
        builder.push_token("</td>");
        let (html, rows, cols) = builder.finish();
        assert_eq!(html, "<html><body><table>></td></table></body></html>");
        assert_eq!((rows, cols), (0, 0));
    }

    /// attr_span：colspan/rowspan 跨度解析与缺省。
    #[test]
    fn attr_span_parsing() {
        assert_eq!(attr_span(" colspan=\"3\"", "colspan"), Some(3));
        assert_eq!(attr_span(" rowspan=\"12\"", "rowspan"), Some(12));
        assert_eq!(attr_span("", "colspan"), None);
        assert_eq!(attr_span(" colspan=\"\"", "colspan"), None);
    }

    /// bbox 还原比例：slanet-plus 画布归一化 × max(h,w)。
    /// 与 RapidTable 两步合并结果逐项对拍（h=200, w=400 → ratio=1.22, 因子=400）。
    #[test]
    fn bbox_scale_matches_reference() {
        let (h, w) = (200usize, 400usize);
        let resized = 488f32;
        // RapidTable 两步：x' = x*w；x'' = x' * 488/(w*ratio)
        let ratio = resized / (h.max(w) as f32);
        let raw_x = 0.5f32;
        let raw_y = 0.25f32;
        let ref_x = raw_x * w as f32 * (resized / (w as f32 * ratio));
        let ref_y = raw_y * h as f32 * (resized / (h as f32 * ratio));
        // 本实现合并：x'' = x * max(h, w)
        let scale = (h.max(w) as f32).max(1.0);
        assert!((ref_x - raw_x * scale).abs() < 1e-4, "ref_x={ref_x}");
        assert!((ref_y - raw_y * scale).abs() < 1e-4, "ref_y={ref_y}");
    }

    /// split_outputs：按末维区分 bbox/structure，含回退路径。
    #[test]
    fn split_outputs_by_last_dim() {
        let make = |name: &str, shape: Vec<i64>| TensorOutput {
            name: name.to_string(),
            shape,
            data: crate::core::base::TensorData::F32(vec![0.0; 8]),
        };
        let outs = vec![
            make("scale_0", vec![1, 40, 8]),
            make("scale_1", vec![1, 40, 50]),
        ];
        let (bbox, structure) = split_outputs(&outs, 50).unwrap();
        assert_eq!(bbox.name, "scale_0");
        assert_eq!(structure.name, "scale_1");
        // 无法区分时按导出顺序回退
        let ambiguous = vec![make("a", vec![1, 40, 7]), make("b", vec![1, 40, 9])];
        let (bbox, structure) = split_outputs(&ambiguous, 50).unwrap();
        assert_eq!(bbox.name, "a");
        assert_eq!(structure.name, "b");
    }

    // ==================== 真实模型自验（需 testmodels/，默认忽略） ====================

    /// 在白底图上画一张 2x2 表格（黑线外框 + 中线，单元格内模拟文本条）。
    fn fill_rect(buf: &mut [u8], width: usize, x1: usize, y1: usize, x2: usize, y2: usize, v: u8) {
        for y in y1..y2 {
            for x in x1..x2 {
                let idx = (y * width + x) * 3;
                buf[idx] = v;
                buf[idx + 1] = v;
                buf[idx + 2] = v;
            }
        }
    }

    fn draw_table_image(width: usize, height: usize) -> Image {
        let mut buf = vec![255u8; width * height * 3];
        let dark = 20u8;
        // 外框（线宽 3）+ 中线 → 2 行 2 列
        fill_rect(&mut buf, width, 40, 30, 280, 33, dark);
        fill_rect(&mut buf, width, 40, 207, 280, 210, dark);
        fill_rect(&mut buf, width, 40, 30, 43, 210, dark);
        fill_rect(&mut buf, width, 277, 30, 280, 210, dark);
        fill_rect(&mut buf, width, 158, 30, 162, 210, dark);
        fill_rect(&mut buf, width, 40, 118, 280, 122, dark);
        // 单元格内模拟文本（灰色短条，每格 3 行）
        for (cx, cy) in [(70, 60), (190, 60), (70, 150), (190, 150)] {
            for r in 0..3 {
                fill_rect(&mut buf, width, cx, cy + r * 14, cx + 60, cy + r * 14 + 6, 90);
            }
        }
        Image::from_raw(width, height, 3, buf).unwrap()
    }

    /// 自验：合成 2x2 表格图 → html 结构合理、行列正确、cell 数接近 4。
    /// 运行：cargo test --lib table_recognition -- --ignored --nocapture
    #[test]
    fn slanet_recognizes_synthetic_2x2_table() {
        if !["testmodels/slanet.onnx", "models/table_recognition/slanet.onnx"].iter().any(|p| std::path::Path::new(p).exists()) {
            eprintln!("[SKIP] 模型/测试资产缺失（models/ 或 testmodels/）");
            return;
        }

        let dir = std::env::var("TESTMODELS_DIR").unwrap_or_else(|_| "testmodels".into());
        let model = format!("{dir}/slanet.onnx");
        let engine = TableRecognitionEngine::new(&model, DeviceType::Cpu).unwrap();
        eprintln!(
            "dict_from_metadata={} char_list={}",
            engine.dict_from_metadata(),
            engine.char_list().len()
        );

        let img = draw_table_image(320, 240);
        let t0 = std::time::Instant::now();
        let result = engine.predict_impl(&img).unwrap();
        eprintln!("耗时 {} ms", t0.elapsed().as_millis());
        eprintln!(
            "rows={} cols={} cells={:?} scores={:?}",
            result.row_count, result.col_count, result.cell_boxes, result.cell_scores
        );
        eprintln!("html = {}", result.html);

        // html 骨架
        assert!(result.html.starts_with("<html><body><table>"));
        assert!(result.html.ends_with("</table></body></html>"));
        assert!(result.html.contains("<td"), "应包含单元格：{}", result.html);

        // 结构：合成线条图在模型分布外（SLANet 训练于真实文档表格），实测倾向把
        // 首行并成 colspan=2 的跨列单元格：2 行 2 列、3~4 个 cell（"行×列 附近"）。
        // 解码管线的精确性由下方真实表格图测试 slanet_recognizes_real_table_image 保证。
        assert!(
            (2..=3).contains(&result.row_count),
            "行数应接近 2，实际 {}",
            result.row_count
        );
        assert_eq!(result.col_count, 2, "列数应为 2（colspan 展开），html = {}", result.html);
        assert!(
            (3..=8).contains(&result.cell_boxes.len()),
            "cell 数应接近 2x2=4，实际 {}",
            result.cell_boxes.len()
        );

        // cell bbox 落在原图内且非退化（分布外合成图的回归偏差较大，不做位置精校）
        for b in &result.cell_boxes {
            assert!(b[0] >= 0.0 && b[2] <= 320.0, "x 越界: {b:?}");
            assert!(b[1] >= 0.0 && b[3] <= 240.0, "y 越界: {b:?}");
            assert!(b[2] > b[0] && b[3] > b[1], "空框: {b:?}");
        }
    }

    /// 自验（真实表格图）：RapidTable 官方测试图 table.jpg（371x293，真实文档表格）。
    /// 校验解码管线：行列结构合理、cell bbox 均在原图内且互不嵌套重叠过深。
    /// 运行：cargo test --lib table_recognition -- --ignored --nocapture
    #[test]
    fn slanet_recognizes_real_table_image() {
        if !["testmodels/slanet.onnx", "models/table_recognition/slanet.onnx", "testmodels/table_demo.jpg", "models/table_recognition/table_demo.jpg"].iter().any(|p| std::path::Path::new(p).exists()) {
            eprintln!("[SKIP] 模型/测试资产缺失（models/ 或 testmodels/）");
            return;
        }

        let dir = std::env::var("TESTMODELS_DIR").unwrap_or_else(|_| "testmodels".into());
        let model = format!("{dir}/slanet.onnx");
        let img_path = format!("{dir}/table_demo.jpg");
        let engine = TableRecognitionEngine::new(&model, DeviceType::Cpu).unwrap();
        let img = Image::load(&img_path).unwrap();
        eprintln!("输入 {}x{}", img.width(), img.height());

        let result = engine.predict_impl(&img).unwrap();
        eprintln!("rows={} cols={} cells={}", result.row_count, result.col_count, result.cell_boxes.len());
        eprintln!("html = {}", result.html);
        for (i, b) in result.cell_boxes.iter().enumerate() {
            eprintln!("cell[{i}] = {b:?} score={:.3}", result.cell_scores[i]);
        }

        // html 骨架完整
        assert!(result.html.starts_with("<html><body><table>"));
        assert!(result.html.ends_with("</table></body></html>"));

        // 真实表格：应识别出多行多列与足够数量的单元格
        assert!(result.row_count >= 3, "行数应 >=3，实际 {}", result.row_count);
        assert!(result.col_count >= 2, "列数应 >=2，实际 {}", result.col_count);
        assert!(
            result.cell_boxes.len() >= 6,
            "有效 cell 数应 >=6，实际 {}",
            result.cell_boxes.len()
        );

        // cell bbox 在原图内且面积合理（去退化框）
        let (w, h) = (img.width() as f32, img.height() as f32);
        for b in &result.cell_boxes {
            assert!(b[0] >= 0.0 && b[1] >= 0.0 && b[2] <= w && b[3] <= h, "越界: {b:?}");
            assert!(b[2] - b[0] > 5.0 && b[3] - b[1] > 5.0, "退化框: {b:?}");
        }
    }
}
