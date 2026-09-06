//! OCR 引擎（PaddleOCR 检测 + 识别两件套）。
//!
//! | 引擎 | 模型 | 输入 | 输出 |
//! |---|---|---|---|
//! | [`OcrDetector`] | DBNet 文本检测（PP-OCRv4/v3/v2 det onnx） | `x` `[1,3,H,W]` f32（H/W 为 32 倍数，动态） | `sigmoid_0.tmp_0` `[1,1,H,W]` f32 概率图 |
//! | [`OcrRecognizer`] | SVTR/CRNN 文本识别（PP-OCRv4/v3 rec onnx + 字典） | `x` `[1,3,48,W']` f32（W' 按行宽高比动态） | `softmax_11.tmp_0` `[1,T,C]` f32（T 为时间帧，C = 字典大小 + blank + 空格） |
//!
//! **预处理约定**：
//! - 检测：BGR→RGB、/255、减 mean `[0.485,0.456,0.406]` 除 std `[0.229,0.224,0.225]`；
//!   官方 `DetResizeForTest` 的 limit 缩放 + 取整到 32 倍数（保持宽高比）
//! - 识别：BGR→RGB、/255、`(x - 0.5) / 0.5`；行高归一 48，右侧 0 填充
//!   （归一化后 0 = 灰 127.5，与官方 `resize_norm_img` 一致）
//!
//! **检测后处理**（DB postprocess，对齐 PaddleOCR `db_postprocess.py` 的轴对齐简化版）：
//! 概率图 > `thresh`(0.3) 二值化 → 8 连通域 BFS（代替官方 findContours）→
//! 最小边 < `min_size`(3) 过滤 → 连通域内概率均值（box score）< `box_thresh`(0.5) 过滤 →
//! unclip 轴对齐外扩（offset = area × `unclip_ratio`(2.0) / perimeter，与官方
//! `poly.area * ratio / poly.length` 等价）→ 概率图坐标还原到原图。
//!
//! **识别后处理**（CTC 解码）：逐帧 argmax → 相邻帧去重 → 去 blank（index 0）→
//! 查字典（`['blank'] + ppocr_keys_v1.txt(6623) + [' ']`，v4 输出维 C=6625 一一对应）→
//! `score` = 非 blank 帧概率均值。
//!
//! **模型直链**（RapidOCR 社区 ONNX，HuggingFace SWHL/RapidOCR）：
//! - 检测：<https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv4/ch_PP-OCRv4_det_infer.onnx>
//! - 识别：<https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv4/ch_PP-OCRv4_rec_infer.onnx>
//! - 字典：<https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/ppocr_keys_v1.txt>
//!
//! ```no_run
//! use rust_onnx_infer::core::DeviceType;
//! use rust_onnx_infer::engines::ocr::OcrPipeline;
//! use rust_onnx_infer::imaging::Image;
//!
//! let ocr = OcrPipeline::new(
//!     "testmodels/ch_PP-OCRv4_det_infer.onnx",
//!     "testmodels/ch_PP-OCRv4_rec_infer.onnx",
//!     "testmodels/ppocr_keys_v1.txt",
//!     DeviceType::Cpu,
//! )?;
//! let results = ocr.recognize(&Image::load("/tmp/ocr_test.png")?)?;
//! for (region, line) in &results {
//!     println!("{} (score={:.3})", line.text, line.score);
//! }
//! # Ok::<(), rust_onnx_infer::error::VisionError>(())
//! ```

use std::cmp::Ordering;
use std::collections::VecDeque;

use ort::value::Tensor;

use crate::core::base::{BaseOnnxEngine, TensorOutput};
use crate::core::device_type::DeviceType;
use crate::core::engine::OnnxInferenceEngine;
use crate::core::runtime_config::OnnxRuntimeConfig;
use crate::error::{Result, VisionError};
use crate::imaging::{cvt_color, resize, ColorConversion, Image, Interpolation};

/// 检测输入归一化 mean（ImageNet，PP-OCR det 训练约定）。
const DET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// 检测输入归一化 std（ImageNet）。
const DET_STD: [f32; 3] = [0.229, 0.224, 0.225];
/// 识别输入归一化 mean（官方 rec 默认 (x/255 - 0.5) / 0.5）。
const REC_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
/// 识别输入归一化 std。
const REC_STD: [f32; 3] = [0.5, 0.5, 0.5];

/// CTC blank 字符固定在字典第 0 位。
const CTC_BLANK_INDEX: usize = 0;
/// Pipeline 里从原图裁文本行时的四周 padding（像素）。
const REGION_CROP_PADDING: i32 = 3;

/// 文本区域（DBNet 检测结果）。
///
/// 轴对齐实现下 `quad` 为矩形四角，顺序：左上 → 右上 → 右下 → 左下。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextRegion {
    /// 四角坐标（原图像素坐标）。
    pub quad: [[f32; 2]; 4],
    /// 区域置信度（连通域内概率图均值）。
    pub score: f32,
}

impl TextRegion {
    /// 轴对齐外接框 (x1, y1, x2, y2)。
    pub fn axis_aligned_bbox(&self) -> (f32, f32, f32, f32) {
        let mut x1 = f32::MAX;
        let mut y1 = f32::MAX;
        let mut x2 = f32::MIN;
        let mut y2 = f32::MIN;
        for p in &self.quad {
            x1 = x1.min(p[0]);
            y1 = y1.min(p[1]);
            x2 = x2.max(p[0]);
            y2 = y2.max(p[1]);
        }
        (x1, y1, x2, y2)
    }

    /// 从轴对齐矩形构造（左上、右上、右下、左下）。
    pub fn from_bbox(x1: f32, y1: f32, x2: f32, y2: f32, score: f32) -> Self {
        TextRegion {
            quad: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            score,
        }
    }
}

/// 单行文本识别结果。
#[derive(Debug, Clone, PartialEq)]
pub struct TextLine {
    /// CTC 解码出的文本。
    pub text: String,
    /// 置信度（非 blank 帧概率均值）。
    pub score: f32,
}

/// 检测预处理 limit 缩放基准（官方 `DetResizeForTest` 的 `limit_type`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DetLimitType {
    /// 以长边为基准：长边超过 limit 时整体缩到 limit 内（PP-OCR 官方推理默认 960/max）。
    #[default]
    Max,
    /// 以短边为基准：短边缩放到 limit（小文本上采样，召回更好；RapidOCR 默认 736/min）。
    Min,
}

/// DBNet 文本检测引擎。
pub struct OcrDetector {
    /// 组合基类（张量形状按推理结果动态构建，不经基类的固定形状辅助）。
    pub base: BaseOnnxEngine,

    /// limit 缩放基准边长（默认 960，配合 [`DetLimitType::Max`]）。
    det_limit_side_len: usize,
    /// limit 缩放类型（默认 Max）。
    det_limit_type: DetLimitType,
    /// 概率图二值化阈值（默认 0.3）。
    det_thresh: f32,
    /// box score 阈值（默认 0.5；官方 DBPostProcess 默认 0.7，RapidOCR 用 0.5）。
    det_box_thresh: f32,
    /// unclip 扩张比例（默认 2.0）。
    det_unclip_ratio: f32,
    /// 连通域最小边长过滤（默认 3，官方 min_size）。
    det_min_size: usize,
    /// 最大候选连通域数（默认 1000）。
    det_max_candidates: usize,
    /// 二值化后是否做 2x2 膨胀（小文本召回更好，默认关闭）。
    det_use_dilation: bool,
}

impl OcrDetector {
    /// 创建检测引擎（动态输入模型，输入尺寸在推理时按原图计算）。
    pub fn new(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        Self::with_config(model_path, device_type, OnnxRuntimeConfig::defaults())
    }

    /// 创建检测引擎（自定义运行参数：线程数 / 设备 id 等）。
    pub fn with_config(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        runtime_config: OnnxRuntimeConfig,
    ) -> Result<Self> {
        // 输入 H/W 为动态维度，显式给 960 仅作为基类元信息兜底
        let base = BaseOnnxEngine::with_config(model_path, device_type, 960, 960, runtime_config)?;
        Ok(OcrDetector {
            base,
            det_limit_side_len: 960,
            det_limit_type: DetLimitType::Max,
            det_thresh: 0.3,
            det_box_thresh: 0.5,
            det_unclip_ratio: 2.0,
            det_min_size: 3,
            det_max_candidates: 1000,
            det_use_dilation: false,
        })
    }

    // ============ 参数访问器 ============

    /// 概率图二值化阈值。
    pub fn det_thresh(&self) -> f32 {
        self.det_thresh
    }

    /// 设置概率图二值化阈值。
    pub fn set_det_thresh(&mut self, thresh: f32) {
        self.det_thresh = thresh;
    }

    /// box score 阈值。
    pub fn box_thresh(&self) -> f32 {
        self.det_box_thresh
    }

    /// 设置 box score 阈值。
    pub fn set_box_thresh(&mut self, thresh: f32) {
        self.det_box_thresh = thresh;
    }

    /// unclip 扩张比例。
    pub fn unclip_ratio(&self) -> f32 {
        self.det_unclip_ratio
    }

    /// 设置 unclip 扩张比例。
    pub fn set_unclip_ratio(&mut self, ratio: f32) {
        self.det_unclip_ratio = ratio;
    }

    /// limit 缩放基准边长。
    pub fn limit_side_len(&self) -> usize {
        self.det_limit_side_len
    }

    /// 设置 limit 缩放基准边长。
    pub fn set_limit_side_len(&mut self, side_len: usize) {
        self.det_limit_side_len = side_len;
    }

    /// limit 缩放类型。
    pub fn limit_type(&self) -> DetLimitType {
        self.det_limit_type
    }

    /// 设置 limit 缩放类型。
    pub fn set_limit_type(&mut self, limit_type: DetLimitType) {
        self.det_limit_type = limit_type;
    }

    /// 是否开启二值化后膨胀。
    pub fn use_dilation(&self) -> bool {
        self.det_use_dilation
    }

    /// 设置二值化后膨胀（小文本场景建议开启）。
    pub fn set_use_dilation(&mut self, use_dilation: bool) {
        self.det_use_dilation = use_dilation;
    }

    /// 最大候选连通域数。
    pub fn max_candidates(&self) -> usize {
        self.det_max_candidates
    }

    /// 设置最大候选连通域数。
    pub fn set_max_candidates(&mut self, max_candidates: usize) {
        self.det_max_candidates = max_candidates;
    }

    // ============ 推理 ============

    /// 文本检测：返回按阅读顺序（先上后下、先左后右）排列的文本区域。
    pub fn detect(&self, image: &Image) -> Result<Vec<TextRegion>> {
        if image.is_empty() {
            return Err(VisionError::image("cannot detect on empty image"));
        }
        // 1. 预处理（limit 缩放 + 32 取整 + RGB 归一化）
        let (input_data, resize_w, resize_h) = self.preprocess(image)?;

        // 2. 动态形状输入张量 [1,3,H,W]（H/W 随原图变化，不能走基类固定形状辅助）
        let input_tensor = Tensor::from_array((
            vec![1i64, 3, resize_h as i64, resize_w as i64],
            input_data,
        ))?;

        // 3. 推理 → [1,1,H,W] 概率图（DB 输出与输入同分辨率）
        let output = self.base.run_inference(input_tensor)?;

        // 4. DB 后处理 → 原图坐标
        self.postprocess(&output, image.width(), image.height())
    }

    // ==================== 预处理 ====================

    /// 检测预处理：limit 缩放 + 取整到 32 倍数 + BGR→RGB + /255 + mean/std，HWC → CHW。
    /// 返回 `(CHW 数据, 宽, 高)`。
    fn preprocess(&self, image: &Image) -> Result<(Vec<f32>, usize, usize)> {
        let (orig_w, orig_h) = (image.width(), image.height());

        // 官方 DetResizeForTest：按 limit 基准算缩放比例
        let ratio = match self.det_limit_type {
            DetLimitType::Max => {
                let long_side = orig_w.max(orig_h) as f32;
                if long_side > self.det_limit_side_len as f32 {
                    self.det_limit_side_len as f32 / long_side
                } else {
                    1.0
                }
            }
            DetLimitType::Min => {
                let short_side = (orig_w.min(orig_h) as f32).max(1.0);
                self.det_limit_side_len as f32 / short_side
            }
        };

        // 保持宽高比，取整到 32 倍数（最小 32，动态模型对任意 32 倍数尺寸均可推理）
        let resize_w = (((orig_w as f32 * ratio / 32.0).round() as usize) * 32).max(32);
        let resize_h = (((orig_h as f32 * ratio / 32.0).round() as usize) * 32).max(32);

        let bgr = to_bgr3(image)?;
        let resized = resize(&bgr, resize_w, resize_h, Interpolation::Linear)?;
        let rgb = cvt_color(&resized, ColorConversion::Bgr2Rgb)?;

        // /255 → 减 mean 除 std → HWC 转 CHW
        let px = rgb.data();
        let area = resize_w * resize_h;
        let mut data = vec![0f32; 3 * area];
        for i in 0..area {
            data[i] = (px[i * 3] as f32 / 255.0 - DET_MEAN[0]) / DET_STD[0];
            data[i + area] = (px[i * 3 + 1] as f32 / 255.0 - DET_MEAN[1]) / DET_STD[1];
            data[i + 2 * area] = (px[i * 3 + 2] as f32 / 255.0 - DET_MEAN[2]) / DET_STD[2];
        }
        Ok((data, resize_w, resize_h))
    }

    // ==================== DB 后处理 ====================

    /// DB 后处理：二值化 → 连通域 → 过滤 → unclip → 坐标还原。
    fn postprocess(
        &self,
        output: &TensorOutput,
        orig_w: usize,
        orig_h: usize,
    ) -> Result<Vec<TextRegion>> {
        let flat = output.as_f32()?;
        let shape = &output.shape;
        if shape.len() != 4 || shape[1] != 1 {
            return Err(VisionError::inference(format!(
                "DB det 模型期望 [1,1,H,W] 概率图输出，实际 shape: {:?}",
                shape
            )));
        }
        let (map_h, map_w) = (shape[2] as usize, shape[3] as usize);
        if flat.len() < map_h * map_w {
            return Err(VisionError::inference(format!(
                "det 输出元素数 {} < {}x{}",
                flat.len(),
                map_h,
                map_w
            )));
        }

        // 1. 概率图按 det_thresh 二值化（{0,1} 位图）
        let map_len = map_h * map_w;
        let mut bitmap = vec![0u8; map_len];
        for i in 0..map_len {
            if flat[i] > self.det_thresh {
                bitmap[i] = 1;
            }
        }
        // 可选 2x2 膨胀（连通断裂的小文本召回更好）
        if self.det_use_dilation {
            dilate_bitmap(&mut bitmap, map_w, map_h);
        }

        // 2. 8 连通域（等价官方 findContours 的连通语义）
        let components = find_connected_components(&bitmap, map_w, map_h, true);

        // 3. 逐连通域：min_size / box score 过滤 → unclip 外扩 → 还原原图坐标
        let scale_x = orig_w as f32 / map_w as f32;
        let scale_y = orig_h as f32 / map_h as f32;
        let mut regions = Vec::new();
        for comp in components.iter().take(self.det_max_candidates) {
            let comp_w = comp.max_x - comp.min_x + 1;
            let comp_h = comp.max_y - comp.min_y + 1;

            // 最小边长过滤（官方 sside < min_size）
            if comp_w.min(comp_h) < self.det_min_size {
                continue;
            }

            // box score：连通域内概率均值（官方 box_score_fast 的连通域简化，
            // 比整框均值更贴近多边形内均值）
            let mut prob_sum = 0f32;
            for &idx in &comp.pixels {
                prob_sum += flat[idx as usize];
            }
            let score = prob_sum / comp.pixels.len() as f32;
            if score < self.det_box_thresh {
                continue;
            }

            // unclip 轴对齐外扩：offset = area * ratio / perimeter
            // （与官方 poly.area * unclip_ratio / poly.length 等价，周长 = 2*(w+h)）
            let offset = (comp_w * comp_h) as f32 * self.det_unclip_ratio
                / (2.0 * (comp_w + comp_h) as f32);

            // 概率图坐标 → 原图坐标，并 clamp 到图内
            let x1 = ((comp.min_x as f32 - offset) * scale_x).clamp(0.0, orig_w as f32);
            let y1 = ((comp.min_y as f32 - offset) * scale_y).clamp(0.0, orig_h as f32);
            let x2 = (((comp.max_x + 1) as f32 + offset) * scale_x).clamp(0.0, orig_w as f32);
            let y2 = (((comp.max_y + 1) as f32 + offset) * scale_y).clamp(0.0, orig_h as f32);
            if x2 - x1 < 1.0 || y2 - y1 < 1.0 {
                continue;
            }

            regions.push(TextRegion::from_bbox(x1, y1, x2, y2, score));
        }

        // 阅读顺序排序：先上后下、先左后右（官方不做排序，识别输入更稳定）
        regions.sort_by(|a, b| {
            let ka = (a.quad[0][1], a.quad[0][0]);
            let kb = (b.quad[0][1], b.quad[0][0]);
            ka.partial_cmp(&kb).unwrap_or(Ordering::Equal)
        });

        tracing::info!("OCR det: {} text regions", regions.len());
        Ok(regions)
    }
}

/// SVTR/CRNN 文本识别引擎（PP-OCRv4/v3 rec onnx + 字典）。
pub struct OcrRecognizer {
    /// 组合基类（行宽动态，张量形状按批内最大宽高比动态构建）。
    pub base: BaseOnnxEngine,

    /// 原始字典（ppocr_keys_v1.txt 逐行，共 6623 项）。
    dict: Vec<String>,
    /// 解码字符表：`['blank'] + dict + [' ']`（use_space_char 开启时）。
    char_list: Vec<String>,
    /// 是否在字符表尾部追加空格类别（PP-OCR 中文模型默认开启）。
    use_space_char: bool,
    /// 识别输入行高（默认 48，v3/v4；v2 老模型为 32）。
    rec_image_height: usize,
    /// 识别输入最大行宽（默认 320；更宽的行横向压缩）。
    rec_max_width: usize,
}

impl OcrRecognizer {
    /// 创建识别引擎（从文件加载字典）。
    pub fn new(
        model_path: impl AsRef<std::path::Path>,
        dict_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
    ) -> Result<Self> {
        Self::with_config(
            model_path,
            dict_path,
            device_type,
            OnnxRuntimeConfig::defaults(),
        )
    }

    /// 创建识别引擎（自定义运行参数）。
    pub fn with_config(
        model_path: impl AsRef<std::path::Path>,
        dict_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        runtime_config: OnnxRuntimeConfig,
    ) -> Result<Self> {
        let dict = Self::load_dict(dict_path)?;
        // 输入 [N,3,48,W]，48/320 仅作基类元信息兜底
        let base = BaseOnnxEngine::with_config(model_path, device_type, 48, 320, runtime_config)?;
        let mut rec = OcrRecognizer {
            base,
            dict,
            char_list: Vec::new(),
            use_space_char: true,
            rec_image_height: 48,
            rec_max_width: 320,
        };
        rec.rebuild_char_list();
        Ok(rec)
    }

    // ============ 参数访问器 ============

    /// 字典条目数。
    pub fn dict_len(&self) -> usize {
        self.dict.len()
    }

    /// 解码字符表（含 blank 与空格）。
    pub fn char_list(&self) -> &[String] {
        &self.char_list
    }

    /// 是否追加空格类别。
    pub fn use_space_char(&self) -> bool {
        self.use_space_char
    }

    /// 设置是否追加空格类别（重建字符表）。
    pub fn set_use_space_char(&mut self, use_space_char: bool) {
        self.use_space_char = use_space_char;
        self.rebuild_char_list();
    }

    /// 识别输入行高。
    pub fn rec_image_height(&self) -> usize {
        self.rec_image_height
    }

    /// 设置识别输入行高（v3/v4 为 48，v2 老模型为 32）。
    pub fn set_rec_image_height(&mut self, height: usize) {
        self.rec_image_height = height.max(1);
    }

    /// 识别输入最大行宽。
    pub fn rec_max_width(&self) -> usize {
        self.rec_max_width
    }

    /// 设置识别输入最大行宽。
    pub fn set_rec_max_width(&mut self, max_width: usize) {
        self.rec_max_width = max_width.max(1);
    }

    // ============ 推理 ============

    /// 识别原图中的单个文本区域（内部完成裁剪 + padding + 行高归一 48）。
    pub fn recognize(&self, image: &Image, region: &TextRegion) -> Result<TextLine> {
        let crop = self.crop_region(image, region)?;
        let mut crops = vec![crop];
        Ok(self.recognize_crops(&mut crops)?.remove(0))
    }

    /// 直接识别已裁剪的文本行图像（免检测用法）。
    pub fn recognize_line(&self, line: &Image) -> Result<TextLine> {
        let mut crops = vec![line.clone()];
        Ok(self.recognize_crops(&mut crops)?.remove(0))
    }

    /// 批量识别原图中的多个文本区域（批内按最大宽高比 pad 到统一宽度，一次前向）。
    pub fn recognize_batch(
        &self,
        image: &Image,
        regions: &[TextRegion],
    ) -> Result<Vec<TextLine>> {
        if regions.is_empty() {
            return Ok(Vec::new());
        }
        let mut crops = Vec::with_capacity(regions.len());
        for region in regions {
            crops.push(self.crop_region(image, region)?);
        }
        self.recognize_crops(&mut crops)
    }

    // ==================== 内部实现 ====================

    /// 从原图裁剪文本区域（轴对齐外接框 + 少量 padding，clamp 到图内）。
    fn crop_region(&self, image: &Image, region: &TextRegion) -> Result<Image> {
        let (fx1, fy1, fx2, fy2) = region.axis_aligned_bbox();
        let pad = REGION_CROP_PADDING as f32;
        let left = ((fx1 - pad).floor() as i32).clamp(0, image.width() as i32 - 1);
        let top = ((fy1 - pad).floor() as i32).clamp(0, image.height() as i32 - 1);
        let right = ((fx2 + pad).ceil() as i32).clamp(left + 1, image.width() as i32);
        let bottom = ((fy2 + pad).ceil() as i32).clamp(top + 1, image.height() as i32);
        image.crop(
            left as usize,
            top as usize,
            (right - left) as usize,
            (bottom - top) as usize,
        )
    }

    /// 识别一批文本行图像（批内 pad 到统一宽度，一次前向，逐行 CTC 解码）。
    fn recognize_crops(&self, crops: &mut [Image]) -> Result<Vec<TextLine>> {
        if crops.is_empty() {
            return Ok(Vec::new());
        }
        let (data, target_w) = self.preprocess_crops(crops)?;
        let batch = crops.len();
        let input_tensor = Tensor::from_array((
            vec![
                batch as i64,
                3,
                self.rec_image_height as i64,
                target_w as i64,
            ],
            data,
        ))?;
        let output = self.base.run_inference(input_tensor)?;

        // 期望 [N, T, C]（T 为时间帧，C 为类别数 = 字典 + blank + 空格）
        let flat = output.as_f32()?;
        let shape = &output.shape;
        let (frames, classes) = match shape.len() {
            3 => (shape[1] as usize, shape[2] as usize),
            2 if batch == 1 => (shape[0] as usize, shape[1] as usize), // 单样本 [T,C] 导出
            _ => {
                return Err(VisionError::inference(format!(
                    "rec 模型期望 [N,T,C] 输出，实际 shape: {:?}",
                    shape
                )))
            }
        };
        if classes != self.char_list.len() {
            tracing::warn!(
                "rec 模型输出类别数 {} 与字符表长度 {} 不一致，越界索引将被跳过",
                classes,
                self.char_list.len()
            );
        }

        let mut lines = Vec::with_capacity(batch);
        let stride = frames * classes;
        for b in 0..batch {
            let start = b * stride;
            let end = (start + stride).min(flat.len());
            lines.push(self.ctc_decode(&flat[start..end], frames, classes));
        }
        Ok(lines)
    }

    /// 识别预处理：行高归一 + 等比缩放 + (x/255 - 0.5)/0.5 + 批内右 pad 0。
    /// 返回 `(NCHW 数据, 批内统一宽度)`。
    fn preprocess_crops(&self, crops: &mut [Image]) -> Result<(Vec<f32>, usize)> {
        let height = self.rec_image_height;

        // 批内统一宽度 = imgH × 最大宽高比（官方 imgW = int(48 * max_wh_ratio)），上限 rec_max_width
        let mut max_ratio = 1.0f32;
        for crop in crops.iter() {
            if crop.is_empty() {
                return Err(VisionError::image("识别行图像为空"));
            }
            let ratio = crop.width() as f32 / crop.height() as f32;
            if ratio > max_ratio {
                max_ratio = ratio;
            }
        }
        let target_w = ((height as f32 * max_ratio).ceil() as usize).clamp(1, self.rec_max_width);

        let mut batch = vec![0f32; crops.len() * 3 * height * target_w];
        for (b, crop) in crops.iter_mut().enumerate() {
            // 单行等比缩放到行高，行宽超批宽时横向压缩（官方 resized_w = min(imgW, ceil(48*ratio))）
            let ratio = crop.width() as f32 / crop.height() as f32;
            let resized_w =
                (((height as f32 * ratio).ceil() as usize).min(target_w)).max(1);

            let bgr = to_bgr3(crop)?;
            let resized = resize(&bgr, resized_w, height, Interpolation::Linear)?;
            let rgb = cvt_color(&resized, ColorConversion::Bgr2Rgb)?;

            let px = rgb.data();
            // 通道平面步长必须是整平面大小（height * 批内统一宽度），右侧 pad 区保持 0
            let plane = height * target_w;
            let base_off = b * 3 * plane;
            for y in 0..height {
                for x in 0..resized_w {
                    let src = (y * resized_w + x) * 3;
                    let dst = base_off + y * target_w + x;
                    batch[dst] = (px[src] as f32 / 255.0 - REC_MEAN[0]) / REC_STD[0];
                    batch[dst + plane] = (px[src + 1] as f32 / 255.0 - REC_MEAN[1]) / REC_STD[1];
                    batch[dst + 2 * plane] = (px[src + 2] as f32 / 255.0 - REC_MEAN[2]) / REC_STD[2];
                }
            }
            // 右侧 padding 保持 0（归一化后 0 = 灰 127.5，与官方 zeros padding 一致）
        }
        Ok((batch, target_w))
    }

    /// CTC 解码：逐帧 argmax → 相邻去重 → 去 blank → 查字典；score = 非 blank 帧概率均值。
    fn ctc_decode(&self, frames: &[f32], frame_count: usize, classes: usize) -> TextLine {
        let mut text = String::new();
        let (mut prob_sum, mut kept) = (0f32, 0usize);
        let mut last_index = CTC_BLANK_INDEX;

        for t in 0..frame_count {
            let row = &frames[t * classes..(t + 1) * classes];
            let (mut best, mut prob) = (0usize, f32::MIN);
            for (i, &v) in row.iter().enumerate() {
                if v > prob {
                    prob = v;
                    best = i;
                }
            }
            // 与上一帧重复的字符跳过（CTC 去重），blank 跳过
            if best != last_index && best != CTC_BLANK_INDEX {
                if let Some(ch) = self.char_list.get(best) {
                    text.push_str(ch);
                    prob_sum += prob;
                    kept += 1;
                }
            }
            last_index = best;
        }

        TextLine {
            text,
            score: if kept > 0 {
                prob_sum / kept as f32
            } else {
                0.0
            },
        }
    }

    /// 构建解码字符表：`['blank'] + dict + [' ']`。
    fn rebuild_char_list(&mut self) {
        self.char_list.clear();
        self.char_list.push(String::from("blank")); // CTC blank 固定 index 0
        self.char_list.extend(self.dict.iter().cloned());
        if self.use_space_char {
            self.char_list.push(String::from(" "));
        }
    }

    /// 加载 PaddleOCR 文本字典（每行一个字符/词；忽略行尾空行）。
    fn load_dict(dict_path: impl AsRef<std::path::Path>) -> Result<Vec<String>> {
        let path = dict_path.as_ref();
        let content = std::fs::read_to_string(path).map_err(|e| {
            VisionError::Io(format!("读取 OCR 字典失败 {}: {e}", path.display()))
        })?;
        let mut lines: Vec<String> = content.lines().map(|l| l.to_string()).collect();
        while lines.last().map(|l| l.is_empty()).unwrap_or(false) {
            lines.pop();
        }
        if lines.is_empty() {
            return Err(VisionError::invalid_argument(format!(
                "OCR 字典为空: {}",
                path.display()
            )));
        }
        tracing::info!("Loaded OCR dict: {} entries from {}", lines.len(), path.display());
        Ok(lines)
    }
}

/// OCR 流水线：DBNet 检测 → 逐区域裁剪 → CTC 识别。
pub struct OcrPipeline {
    /// 文本检测引擎。
    pub detector: OcrDetector,
    /// 文本识别引擎。
    pub recognizer: OcrRecognizer,
}

impl OcrPipeline {
    /// 创建 OCR 流水线（det onnx + rec onnx + 字典 + 设备）。
    pub fn new(
        det_model_path: impl AsRef<std::path::Path>,
        rec_model_path: impl AsRef<std::path::Path>,
        dict_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
    ) -> Result<Self> {
        Ok(OcrPipeline {
            detector: OcrDetector::new(det_model_path, device_type)?,
            recognizer: OcrRecognizer::new(rec_model_path, dict_path, device_type)?,
        })
    }

    /// 创建 OCR 流水线（自定义运行参数，det/rec 共用）。
    pub fn with_config(
        det_model_path: impl AsRef<std::path::Path>,
        rec_model_path: impl AsRef<std::path::Path>,
        dict_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        runtime_config: OnnxRuntimeConfig,
    ) -> Result<Self> {
        Ok(OcrPipeline {
            detector: OcrDetector::with_config(det_model_path, device_type, runtime_config)?,
            recognizer: OcrRecognizer::with_config(rec_model_path, dict_path, device_type, runtime_config)?,
        })
    }

    /// 端到端识别：检测文本区域 → 从原图裁剪 → 逐区域识别。
    /// 返回按阅读顺序排列的 `(区域, 文本行)` 对。
    pub fn recognize(&self, image: &Image) -> Result<Vec<(TextRegion, TextLine)>> {
        let regions = self.detector.detect(image)?;
        if regions.is_empty() {
            return Ok(Vec::new());
        }
        let lines = self.recognizer.recognize_batch(image, &regions)?;
        Ok(regions.into_iter().zip(lines).collect())
    }
}

// ONNXInferenceEngine 统一接口：Output = Vec<(TextRegion, TextLine)>
// det/rec 单引擎为多输入形状协作关系，不单独实现 trait。
#[::async_trait::async_trait]
impl OnnxInferenceEngine for OcrPipeline {
    type Output = Vec<(TextRegion, TextLine)>;

    fn predict(&self, image: &Image) -> Result<Self::Output> {
        self.recognize(image)
    }

    fn predict_batch(&self, images: &[Image]) -> Result<Vec<Self::Output>> {
        images.iter().map(|img| self.recognize(img)).collect()
    }

    fn input_size(&self) -> (i32, i32) {
        self.detector.base.input_size()
    }

    fn labels(&self) -> Option<&[String]> {
        None
    }

    fn set_labels(&mut self, _labels: Vec<String>) {
        // OCR 无检测类别标签语义；识别字符表由字典文件决定
    }

    fn set_confidence_threshold(&mut self, threshold: f32) {
        // 语义对齐：置信度阈值映射为文本区域 box score 阈值
        self.detector.set_box_thresh(threshold);
    }

    fn confidence_threshold(&self) -> f32 {
        self.detector.box_thresh()
    }
}

// ==================== 通用辅助 ====================

/// 统一转 3 通道 BGR（库内 Image 3/4 通道为 BGR/BGRA 序，1 通道为灰度）。
fn to_bgr3(image: &Image) -> Result<Image> {
    match image.channels() {
        3 => Ok(image.clone()),
        4 => cvt_color(image, ColorConversion::Bgra2Bgr),
        1 => cvt_color(image, ColorConversion::Gray2Bgr),
        n => Err(VisionError::image(format!("不支持的通道数: {n}"))),
    }
}

/// 连通域（像素线性索引 + 外接框，左上闭右下闭）。
struct Component {
    /// 域内像素的线性索引（y * width + x）。
    pixels: Vec<u32>,
    min_x: usize,
    min_y: usize,
    max_x: usize,
    max_y: usize,
}

/// BFS 连通域标记（支持 4/8 连通；`bitmap` 为 {0,1} 位图）。
fn find_connected_components(
    bitmap: &[u8],
    width: usize,
    height: usize,
    connectivity8: bool,
) -> Vec<Component> {
    let mut visited = vec![false; bitmap.len()];
    let mut queue: VecDeque<usize> = VecDeque::with_capacity(256);
    let mut components = Vec::new();

    for start in 0..bitmap.len() {
        if bitmap[start] == 0 || visited[start] {
            continue;
        }
        visited[start] = true;
        queue.clear();
        queue.push_back(start);

        let mut comp = Component {
            pixels: Vec::new(),
            min_x: width,
            min_y: height,
            max_x: 0,
            max_y: 0,
        };
        while let Some(idx) = queue.pop_front() {
            let x = idx % width;
            let y = idx / width;
            comp.pixels.push(idx as u32);
            if x < comp.min_x {
                comp.min_x = x;
            }
            if x > comp.max_x {
                comp.max_x = x;
            }
            if y < comp.min_y {
                comp.min_y = y;
            }
            if y > comp.max_y {
                comp.max_y = y;
            }

            // 邻域入队（4 连通 + 可选对角 4 邻域 = 8 连通）
            let left = x > 0;
            let right = x + 1 < width;
            let up = y > 0;
            let down = y + 1 < height;
            for nidx in neighbors(idx, width, left, right, up, down, connectivity8) {
                if !visited[nidx] && bitmap[nidx] != 0 {
                    visited[nidx] = true;
                    queue.push_back(nidx);
                }
            }
        }
        components.push(comp);
    }
    components
}

/// 生成像素 idx 的邻域线性索引（保证不越界、不跨行回绕）。
fn neighbors(
    idx: usize,
    width: usize,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    connectivity8: bool,
) -> Vec<usize> {
    let mut out = Vec::with_capacity(if connectivity8 { 8 } else { 4 });
    if left {
        out.push(idx - 1);
    }
    if right {
        out.push(idx + 1);
    }
    if up {
        out.push(idx - width);
    }
    if down {
        out.push(idx + width);
    }
    if connectivity8 {
        if up && left {
            out.push(idx - width - 1);
        }
        if up && right {
            out.push(idx - width + 1);
        }
        if down && left {
            out.push(idx + width - 1);
        }
        if down && right {
            out.push(idx + width + 1);
        }
    }
    out
}

/// 2x2 核膨胀（对齐官方 use_dilation 的 kernel=[[1,1],[1,1]]）。
fn dilate_bitmap(bitmap: &mut [u8], width: usize, height: usize) {
    let src = bitmap.to_vec();
    for y in 0..height {
        for x in 0..width {
            let mut v = src[y * width + x];
            if x + 1 < width {
                v = v.max(src[y * width + x + 1]);
            }
            if y + 1 < height {
                v = v.max(src[(y + 1) * width + x]);
            }
            if x + 1 < width && y + 1 < height {
                v = v.max(src[(y + 1) * width + x + 1]);
            }
            bitmap[y * width + x] = v;
        }
    }
}
