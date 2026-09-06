//! 车牌检测 + 识别引擎（中国车牌为主，兼容单行牌 / 新能源绿牌）。
//!
//! 两级流水线：**车牌检测**（定位车牌 + 四角关键点）→ **车牌识别**（LPRNet CTC 解码车牌字符）。
//!
//! | 阶段 | 模型 | 输入 | 输出 |
//! |---|---|---|---|
//! | 检测（[`PlateDetectorKind::Mnet`]，默认） | RetinaPL 风格 mnet 单类车牌检测器 | `input.1` `[1,3,640,640]` f32（BGR - mean(104,117,123)，**不除 255**） | `loc [16800,4]` + `conf [16800,2]` + `landms [16800,8]`（SSD anchor 解码） |
//! | 检测（[`PlateDetectorKind::Yolo`]） | YOLOv8/v11 系单类车牌检测器 | `images` `[1,3,640,640]` f32（letterbox 114 + /255 + RGB） | 传统 `[1, 4+nc, anchors]`（nc=1）或 End2End `[1,M,6]`（自动识别） |
//! | 识别 | LPRNet 中文车牌识别 | `input.1` `[1,3,24,94]` f32（BGR，`(px-127.5)/128`） | `126` `[68,18]`（68 类字符 × T 帧，CTC 解码） |
//!
//! **识别字符表**（68 类，与 LPRNet 训练一致）：31 省份简称 + 10 数字 + 24 字母（无 I/O）
//! + I + O + `-`；CTC blank 固定为**最后一类**（`-` 仅作 blank 占位，不会出现在解码文本中），
//!
//! 解码规则：逐帧 softmax argmax → 相邻帧去重 → 去 blank。
//!
//! **车牌矫正**：mnet 检测输出车牌四角关键点（raw 顺序：右下/左下/左上/右上），通过单应矩阵
//! 透视变换矫正到 `94x24` 基准尺寸后再识别（斜拍车牌识别率显著优于直接裁剪拉伸）；
//! YOLO 检测器无关键点，回退为外接框裁剪 + 拉伸。
//!
//! **车牌类型**：模型不含颜色/类型头，`plate_type` 由车牌区域 BGR 均值的启发式规则判定
//! （蓝牌 / 新能源绿牌 / 黄牌 / 白牌 / 黑牌 / 未知）。
//!
//! **OCR 备选识别**：库内 [`crate::engines::ocr::OcrRecognizer`]（PP-OCRv4 rec）同样可读车牌
//! 文字，提供 [`LicensePlateEngine::recognize_with_ocr`] 组合方案（LPRNet 专用模型不可用时切换）。
//!
//! **模型直链**（已核实可下载）：
//! - 检测（mnet）：<https://raw.githubusercontent.com/hpc203/license-plate-detect-recoginition-opencv/main/mnet_plate.onnx>
//! - 识别（LPRNet）：<https://raw.githubusercontent.com/hpc203/license-plate-detect-recoginition-opencv/main/Final_LPRNet_model.onnx>
//! - 检测（YOLOv11n 备选）：<https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1n.onnx>
//!
//! ```no_run
//! use rust_onnx_infer::core::DeviceType;
//! use rust_onnx_infer::engines::license_plate::LicensePlateEngine;
//! use rust_onnx_infer::imaging::Image;
//!
//! let engine = LicensePlateEngine::new(
//!     "testmodels/mnet_plate.onnx",
//!     "testmodels/Final_LPRNet_model.onnx",
//!     DeviceType::Cpu,
//! )?;
//! for plate in engine.recognize(&Image::load("testmodels/plate_test.jpg")?)? {
//!     println!("{:?} {} score={:.3}", plate.r#box, plate.text, plate.score);
//! }
//! # Ok::<(), rust_onnx_infer::error::VisionError>(())
//! ```

use ort::value::Tensor;

use crate::core::base::BaseOnnxEngine;
use crate::core::device_type::DeviceType;
use crate::core::runtime_config::OnnxRuntimeConfig;
use crate::error::{Result, VisionError};
use crate::imaging::{cvt_color, resize, ColorConversion, Image, Interpolation};

// ==================== 常量 ====================

/// 检测输入边长（mnet_plate.onnx 固定 640x640）。
const DET_INPUT_SIZE: usize = 640;
/// 检测输入 BGR 均值（cv2 `blobFromImage(mean=(104,117,123))`，不除 255、不换通道序）。
const DET_MEAN_BGR: [f32; 3] = [104.0, 117.0, 123.0];
/// SSD anchor 每层最小框尺寸（与 RetinaPL / zeusees 车牌检测器训练一致）。
const PRIOR_MIN_SIZES: [[f32; 2]; 3] = [[24.0, 48.0], [96.0, 192.0], [384.0, 768.0]];
/// SSD anchor 对应的下采样步长。
const PRIOR_STEPS: [usize; 3] = [8, 16, 32];
/// anchor 解码方差（RetinaFace 约定 [0.1, 0.2]）。
const PRIOR_VARIANCE: [f32; 2] = [0.1, 0.2];
/// YOLO letterbox 灰边填充值（Ultralytics 约定 114）。
const YOLO_PAD_VALUE: u8 = 114;

/// LPRNet 识别输入宽。
const REC_WIDTH: usize = 94;
/// LPRNet 识别输入高。
const REC_HEIGHT: usize = 24;
/// 识别归一化：(px - 127.5) / 128（cv2 `blobFromImage(scale=1/128, mean=127.5)`）。
const REC_MEAN: f32 = 127.5;
const REC_SCALE: f32 = 1.0 / 128.0;

/// 中国车牌字符表（68 类，顺序与 LPRNet 训练一致；最后一类 `-` 为 CTC blank 占位符）。
const LPR_CHARS: [&str; 68] = [
    "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁",
    "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", //
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", //
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U",
    "V", "W", "X", "Y", "Z", "I", "O", "-", // '-' = CTC blank（不参与解码输出）
];

/// 透视矫正基准四角（左上、右上、左下、右下），对应 94x24 识别输入。
const REF_CORNERS: [[f32; 2]; 4] = [[0.0, 0.0], [94.0, 0.0], [0.0, 24.0], [94.0, 24.0]];

// ==================== 公开类型 ====================

/// 单块车牌的检测 + 识别结果。
#[derive(Debug, Clone, PartialEq)]
pub struct PlateResult {
    /// 车牌外接框（原图像素坐标 xyxy）。
    ///
    /// `box` 为 Rust 保留字，按规范使用 raw identifier。
    pub r#box: [f32; 4],
    /// CTC 解码出的车牌文本（如 `皖AH712X`；新能源牌 8 位）。
    pub text: String,
    /// 综合置信度 = 检测置信度 × 识别字符平均 softmax 概率（0~1）。
    pub score: f64,
    /// 车牌类型（颜色启发式：`蓝牌` / `新能源绿牌` / `黄牌` / `白牌` / `黑牌` / `未知`）。
    pub plate_type: Option<String>,
}

/// 车牌检测模型类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PlateDetectorKind {
    /// RetinaPL 风格 mnet 单类检测器（SSD anchor + 四角关键点，推荐，与 LPRNet 配套）。
    #[default]
    Mnet,
    /// YOLO 系单类车牌检测器（letterbox + 传统 / End2End 输出自动识别，无关键点）。
    Yolo,
}

/// 内部车牌候选框（[`LicensePlateEngine::detect_plates`] 仅检测模式的产物）。
#[derive(Debug, Clone, Copy)]
pub struct PlateBox {
    /// 原图坐标 xyxy。
    pub xyxy: [f32; 4],
    /// 检测置信度。
    pub score: f32,
    /// 四角关键点（原图坐标，顺序：左上、右上、左下、右下）。
    pub landmarks: Option<[[f32; 2]; 4]>,
}

/// 预处理坐标还原参数：`原图坐标 = (模型输入坐标 - pad) / ratio`。
#[derive(Debug, Clone, Copy)]
struct PadRestore {
    ratio: f32,
    pad_left: f32,
    pad_top: f32,
}

// ==================== 车牌识别子引擎 ====================

/// LPRNet 中文车牌识别引擎（`Final_LPRNet_model.onnx`）。
pub struct PlateRecognizer {
    /// 组合基类。
    pub base: BaseOnnxEngine,
}

impl PlateRecognizer {
    /// 创建识别引擎（输入固定 [1,3,24,94]）。
    pub fn new(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        let base = BaseOnnxEngine::with_input_size(model_path, device_type, REC_HEIGHT as i32, REC_WIDTH as i32)?;
        Ok(PlateRecognizer { base })
    }

    /// 识别一块车牌图像（任意尺寸，内部归一到 94x24）。
    ///
    /// 返回 `(车牌文本, 字符平均 softmax 概率)`。
    pub fn recognize_crop(&self, crop: &Image) -> Result<(String, f32)> {
        let data = preprocess_rec(crop)?;
        let input_tensor = Tensor::from_array((
            vec![1i64, 3, REC_HEIGHT as i64, REC_WIDTH as i64],
            data,
        ))?;
        let output = self.base.run_inference(input_tensor)?;

        // 输出兼容 [1,68,T] / [68,T] / [1,T,68] / [T,68]：定位 68 维为字符类别轴。
        // 注意行主序张量中 [68,T] 表示元素 (c,t) 位于 flat[c*T+t]（类别主序），
        // 与本实现参考的 Python 侧 `reshape(68,-1).argmax(axis=0)` 一致。
        let flat = output.as_f32()?;
        let shape = &output.shape;
        let (classes, frames, class_major) = match shape.len() {
            3 if shape[1] as usize == LPR_CHARS.len() => (shape[1] as usize, shape[2] as usize, true),
            3 if shape[2] as usize == LPR_CHARS.len() => (shape[2] as usize, shape[1] as usize, false),
            2 if shape[0] as usize == LPR_CHARS.len() => (shape[0] as usize, shape[1] as usize, true),
            2 if shape[1] as usize == LPR_CHARS.len() => (shape[1] as usize, shape[0] as usize, false),
            _ => {
                return Err(VisionError::inference(format!(
                    "LPRNet 期望输出含 {} 类别维（如 [68,T]），实际 shape: {:?}",
                    LPR_CHARS.len(),
                    shape
                )))
            }
        };
        if flat.len() < classes * frames {
            return Err(VisionError::inference(format!(
                "LPRNet 输出元素数 {} < {}x{}",
                flat.len(),
                classes,
                frames
            )));
        }
        // 类别主序（[68,T]，类别维在前）先重排为帧主序再解码
        let data = if class_major {
            let mut reordered = vec![0f32; classes * frames];
            for t in 0..frames {
                for c in 0..classes {
                    reordered[t * classes + c] = flat[c * frames + t];
                }
            }
            reordered
        } else {
            flat.to_vec()
        };
        Ok(ctc_decode(&data, classes, frames))
    }
}

// ==================== 车牌检测 + 识别引擎 ====================

/// 车牌检测 + 识别引擎（中国车牌为主，兼容单行牌 / 新能源绿牌）。
pub struct LicensePlateEngine {
    /// 检测模型基类（trait 接口转发字段）。
    pub base: BaseOnnxEngine,
    /// LPRNet 识别子引擎。
    pub recognizer: PlateRecognizer,

    /// 检测模型类型。
    detector_kind: PlateDetectorKind,
    /// NMS IoU 阈值。
    nms_threshold: f32,
    /// SSD anchor 表（mnet 专用，640x640 输入固定不变，构造时生成一次）。
    priors: Vec<[f32; 4]>,
}

impl LicensePlateEngine {
    /// 创建车牌引擎（mnet 检测 + LPRNet 识别，推荐配置）。
    pub fn new(
        det_model_path: impl AsRef<std::path::Path>,
        rec_model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
    ) -> Result<Self> {
        Self::for_mnet(det_model_path, rec_model_path, device_type)
    }

    /// 创建车牌引擎（mnet 检测器：anchor 解码 + 四角关键点透视矫正）。
    pub fn for_mnet(
        det_model_path: impl AsRef<std::path::Path>,
        rec_model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
    ) -> Result<Self> {
        Self::with_config(det_model_path, rec_model_path, device_type, PlateDetectorKind::Mnet, OnnxRuntimeConfig::defaults())
    }

    /// 创建车牌引擎（YOLO 系单类检测器：letterbox + 传统 / End2End 自动识别）。
    pub fn for_yolo(
        det_model_path: impl AsRef<std::path::Path>,
        rec_model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
    ) -> Result<Self> {
        Self::with_config(det_model_path, rec_model_path, device_type, PlateDetectorKind::Yolo, OnnxRuntimeConfig::defaults())
    }

    /// 创建车牌引擎（自定义运行参数与检测器类型）。
    pub fn with_config(
        det_model_path: impl AsRef<std::path::Path>,
        rec_model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        detector_kind: PlateDetectorKind,
        runtime_config: OnnxRuntimeConfig,
    ) -> Result<Self> {
        let base = BaseOnnxEngine::with_config(
            det_model_path,
            device_type,
            DET_INPUT_SIZE as i32,
            DET_INPUT_SIZE as i32,
            runtime_config,
        )?;
        let recognizer = PlateRecognizer::new(rec_model_path, device_type)?;
        let priors = match detector_kind {
            PlateDetectorKind::Mnet => generate_priors(),
            PlateDetectorKind::Yolo => Vec::new(),
        };
        tracing::info!(
            "LicensePlateEngine initialized: detector={:?}, priors={}",
            detector_kind,
            priors.len()
        );
        Ok(LicensePlateEngine {
            base,
            recognizer,
            detector_kind,
            nms_threshold: 0.4,
            priors,
        })
    }

    // ============ 参数访问器 ============

    /// 检测模型类型。
    pub fn detector_kind(&self) -> PlateDetectorKind {
        self.detector_kind
    }

    /// NMS IoU 阈值。
    pub fn nms_threshold(&self) -> f32 {
        self.nms_threshold
    }

    /// 设置 NMS IoU 阈值。
    pub fn set_nms_threshold(&mut self, threshold: f32) {
        self.nms_threshold = threshold;
    }

    // ============ 组合 API ============

    /// 单图核心实现（trait `predict` 转发到这里）：
    /// 车牌检测 → 逐牌裁剪矫正 → LPRNet 识别 → 颜色启发式定类型。
    pub fn recognize_impl(&self, image: &Image) -> Result<Vec<PlateResult>> {
        let plates = self.detect_plates(image)?;
        let mut results = Vec::with_capacity(plates.len());
        for plate in &plates {
            let crop = crop_plate_box(image, plate)?;
            let (text, rec_score) = self.recognizer.recognize_crop(&crop)?;
            results.push(PlateResult {
                r#box: plate.xyxy,
                text,
                score: (plate.score * rec_score) as f64,
                plate_type: Some(classify_plate_color(&crop)),
            });
        }
        log_results(&results);
        Ok(results)
    }

    /// OCR 备选识别：车牌检测 + 裁剪 + 库内 PP-OCRv4 rec 读牌。
    ///
    /// LPRNet 专用模型不可用时的组合方案——检测复用本引擎，文字改由
    /// [`crate::engines::ocr::OcrRecognizer`]（PP-OCRv4 rec + 字典）识别。
    pub fn recognize_with_ocr(
        &self,
        ocr: &crate::engines::ocr::OcrRecognizer,
        image: &Image,
    ) -> Result<Vec<PlateResult>> {
        let plates = self.detect_plates(image)?;
        let mut results = Vec::with_capacity(plates.len());
        for plate in &plates {
            // OCR rec 习惯留少量边距（其自身 crop 带 padding，这里手动外扩）
            let crop = crop_with_padding(image, plate.xyxy, 6.0)?;
            let line = ocr.recognize_line(&crop)?;
            results.push(PlateResult {
                r#box: plate.xyxy,
                text: line.text,
                score: (plate.score * line.score) as f64,
                plate_type: Some(classify_plate_color(&crop)),
            });
        }
        log_results(&results);
        Ok(results)
    }

    /// 仅车牌检测（不识别），返回原图坐标 xyxy 与置信度。
    pub fn detect_plates(&self, image: &Image) -> Result<Vec<PlateBox>> {
        if image.is_empty() {
            return Err(VisionError::image("cannot detect plates on empty image"));
        }
        match self.detector_kind {
            PlateDetectorKind::Mnet => self.detect_mnet(image),
            PlateDetectorKind::Yolo => self.detect_yolo(image),
        }
    }

    // ============ mnet 检测路径（SSD anchor 解码） ============

    /// mnet 检测：等比缩放 + 居中补 0 边 → loc/conf/landms → anchor 解码 → NMS → 坐标还原。
    fn detect_mnet(&self, image: &Image) -> Result<Vec<PlateBox>> {
        // 1. 预处理（等比缩放 + 补边，BGR 减均值，不除 255）
        let (input_data, restore) = preprocess_mnet(image)?;

        // 2. 推理（三输出 loc / conf / landms）
        let input_tensor = self.base.create_input_tensor(input_data)?;
        let outputs = self.base.run_multi_output(input_tensor)?;
        let loc = find_output(&outputs, &["loc"], 4)?;
        let conf = find_output(&outputs, &["conf", "scores"], 2)?;
        let landms = find_output(&outputs, &["landms", "landm", "keypoints"], 8)?;
        let loc = read_as_2d(loc, 4)?;
        let conf = read_as_2d(conf, 2)?;
        let landms = read_as_2d(landms, 8)?;
        let num_anchors = loc.rows;
        if conf.rows < num_anchors || landms.rows < num_anchors {
            return Err(VisionError::inference(format!(
                "mnet 输出行数不一致: loc={}, conf={}, landms={}",
                loc.rows,
                conf.rows,
                landms.rows
            )));
        }

        // 3. anchor 解码（框为 xywh，关键点为 4 个 (x,y)；均为 640 输入空间像素坐标）
        let threshold = self.base.confidence_threshold();
        let mut candidates = Vec::new();
        for i in 0..num_anchors {
            let score = conf.get(i, 1); // conf[:, 1] 为车牌置信度
            if score < threshold {
                continue;
            }
            let prior = self.priors[i];
            // 中心：p.xy + loc.xy * var0 * p.wh；宽高：p.wh * exp(loc.wh * var1)
            let cx = prior[0] + loc.get(i, 0) * PRIOR_VARIANCE[0] * prior[2];
            let cy = prior[1] + loc.get(i, 1) * PRIOR_VARIANCE[0] * prior[3];
            let w = prior[2] * (loc.get(i, 2) * PRIOR_VARIANCE[1]).exp();
            let h = prior[3] * (loc.get(i, 3) * PRIOR_VARIANCE[1]).exp();
            // 归一化 → 640 输入空间 xyxy
            let x1 = (cx - w / 2.0) * DET_INPUT_SIZE as f32;
            let y1 = (cy - h / 2.0) * DET_INPUT_SIZE as f32;
            let x2 = (cx + w / 2.0) * DET_INPUT_SIZE as f32;
            let y2 = (cy + h / 2.0) * DET_INPUT_SIZE as f32;

            // 四角关键点（raw 顺序：右下、左下、左上、右上 → 重排为 左上、右上、左下、右下）
            let pt = |k: usize| -> [f32; 2] {
                let px = prior[0] + landms.get(i, k * 2) * PRIOR_VARIANCE[0] * prior[2];
                let py = prior[1] + landms.get(i, k * 2 + 1) * PRIOR_VARIANCE[0] * prior[3];
                [px * DET_INPUT_SIZE as f32, py * DET_INPUT_SIZE as f32]
            };
            let corners = [pt(2), pt(3), pt(1), pt(0)];

            candidates.push(PlateBox {
                xyxy: restore.restore([x1, y1, x2, y2], image),
                score,
                landmarks: Some(restore.restore_corners(corners, image)),
            });
        }

        Ok(nms_plates(candidates, self.nms_threshold))
    }

    // ============ YOLO 检测路径（letterbox + 传统 / End2End） ============

    /// YOLO 检测：letterbox(114) + /255 → 传统 `[1,4+nc,A]` 或 End2End `[1,M,6]` 自动分派。
    fn detect_yolo(&self, image: &Image) -> Result<Vec<PlateBox>> {
        // 1. 预处理（letterbox + RGB + /255）
        let (input_data, restore) = preprocess_yolo_letterbox(image)?;
        let input_tensor = self.base.create_input_tensor(input_data)?;
        let outputs = self.base.run_multi_output(input_tensor)?;

        // 2. 找检测输出（跳过 shape=[1] 等元数据输出，取元素最多者）
        let output = outputs
            .iter()
            .max_by_key(|o| o.element_count())
            .ok_or_else(|| VisionError::inference("YOLO 车牌检测模型无输出"))?;

        let flat = output.as_f32()?;
        let shape = &output.shape;
        let threshold = self.base.confidence_threshold();
        let mut candidates = Vec::new();

        // End2End 判定（对齐 detection.rs 规则）：dim1 为检测数（~300），dim2 = 6 列
        let end2end = shape.len() == 3
            && (100..=400).contains(&shape[1])
            && (4..=100).contains(&shape[2]);

        if end2end {
            // [1, M, 6]: x1,y1,x2,y2,conf,cls（内置 NMS，仅阈值过滤）
            let (m, cols) = (shape[1] as usize, shape[2] as usize);
            for i in 0..m {
                let off = i * cols;
                if off + 5 >= flat.len() {
                    break;
                }
                let score = flat[off + 4];
                if score < threshold {
                    continue;
                }
                let xyxy = [flat[off], flat[off + 1], flat[off + 2], flat[off + 3]];
                candidates.push(PlateBox {
                    xyxy: restore.restore(xyxy, image),
                    score,
                    landmarks: None,
                });
            }
        } else {
            // 传统 [1, 4+nc, anchors]（nc=1）：cx,cy,w,h + 类别分数
            let (channels, anchors) = match shape.len() {
                3 => (shape[1] as usize, shape[2] as usize),
                2 => (shape[0] as usize, shape[1] as usize),
                _ => {
                    return Err(VisionError::inference(format!(
                        "YOLO 车牌检测输出维度不支持: {:?}",
                        shape
                    )))
                }
            };
            if channels < 5 {
                return Err(VisionError::inference(format!(
                    "YOLO 车牌检测输出 channels={} < 5（应为 4+nc, nc=1）",
                    channels
                )));
            }
            for i in 0..anchors {
                let mut best = 0f32;
                for c in 4..channels {
                    let s = flat[c * anchors + i];
                    if s > best {
                        best = s;
                    }
                }
                if best < threshold {
                    continue;
                }
                let cx = flat[i];
                let cy = flat[anchors + i];
                let w = flat[2 * anchors + i];
                let h = flat[3 * anchors + i];
                candidates.push(PlateBox {
                    xyxy: restore.restore([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], image),
                    score: best,
                    landmarks: None,
                });
            }
        }

        Ok(nms_plates(candidates, self.nms_threshold))
    }
}

// ==================== 预处理 ====================

/// mnet 预处理：等比缩放 + 居中补 0 边到 640x640，BGR 减均值（不除 255），HWC → CHW。
fn preprocess_mnet(image: &Image) -> Result<(Vec<f32>, PadRestore)> {
    let (orig_w, orig_h) = (image.width(), image.height());
    let hw_scale = orig_h as f32 / orig_w as f32;
    let (new_w, new_h, pad_left, pad_top) = if hw_scale > 1.0 {
        // 高图：高度贴满 640，左右居中补边
        let new_w = (DET_INPUT_SIZE as f32 / hw_scale) as usize;
        (new_w.max(1), DET_INPUT_SIZE, (DET_INPUT_SIZE - new_w) / 2, 0)
    } else {
        // 宽图：宽度贴满 640，上下居中补边
        let new_h = (DET_INPUT_SIZE as f32 * hw_scale) as usize;
        (DET_INPUT_SIZE, new_h.max(1), 0, (DET_INPUT_SIZE - new_h) / 2)
    };

    let bgr = to_bgr3(image)?;
    let resized = resize(&bgr, new_w, new_h, Interpolation::Area)?;
    let mut padded = Image::filled(DET_INPUT_SIZE, DET_INPUT_SIZE, 3, 0);
    padded.paste(pad_left, pad_top, &resized);

    let px = padded.data();
    let area = DET_INPUT_SIZE * DET_INPUT_SIZE;
    let mut data = vec![0f32; 3 * area];
    for i in 0..area {
        data[i] = px[i * 3] as f32 - DET_MEAN_BGR[0];
        data[i + area] = px[i * 3 + 1] as f32 - DET_MEAN_BGR[1];
        data[i + 2 * area] = px[i * 3 + 2] as f32 - DET_MEAN_BGR[2];
    }
    let restore = PadRestore {
        ratio: new_w as f32 / orig_w as f32,
        pad_left: pad_left as f32,
        pad_top: pad_top as f32,
    };
    Ok((data, restore))
}

/// YOLO 预处理：letterbox（114 灰边）+ BGR→RGB + /255，HWC → CHW。
fn preprocess_yolo_letterbox(image: &Image) -> Result<(Vec<f32>, PadRestore)> {
    let (orig_w, orig_h) = (image.width(), image.height());
    let input = DET_INPUT_SIZE;
    let ratio = (input as f32 / orig_w as f32).min(input as f32 / orig_h as f32);
    let new_w = (orig_w as f32 * ratio).round() as usize;
    let new_h = (orig_h as f32 * ratio).round() as usize;
    let dw = (input - new_w) as f32 / 2.0;
    let dh = (input - new_h) as f32 / 2.0;

    let bgr = to_bgr3(image)?;
    let resized = resize(&bgr, new_w, new_h, Interpolation::Linear)?;
    let mut padded = Image::filled(input, input, 3, YOLO_PAD_VALUE);
    padded.paste((dw - 0.1).round().max(0.0) as usize, (dh - 0.1).round().max(0.0) as usize, &resized);
    let rgb = cvt_color(&padded, ColorConversion::Bgr2Rgb)?;

    let px = rgb.data();
    let area = input * input;
    let mut data = vec![0f32; 3 * area];
    for i in 0..area {
        data[i] = px[i * 3] as f32 / 255.0;
        data[i + area] = px[i * 3 + 1] as f32 / 255.0;
        data[i + 2 * area] = px[i * 3 + 2] as f32 / 255.0;
    }
    Ok((
        data,
        PadRestore {
            ratio,
            pad_left: dw,
            pad_top: dh,
        },
    ))
}

/// 识别预处理：缩放到 94x24（BGR，不换通道序），(px - 127.5) / 128，HWC → CHW。
fn preprocess_rec(crop: &Image) -> Result<Vec<f32>> {
    if crop.is_empty() {
        return Err(VisionError::image("识别车牌图像为空"));
    }
    let bgr = to_bgr3(crop)?;
    let resized = resize(&bgr, REC_WIDTH, REC_HEIGHT, Interpolation::Linear)?;
    let px = resized.data();
    let area = REC_WIDTH * REC_HEIGHT;
    let mut data = vec![0f32; 3 * area];
    for i in 0..area {
        data[i] = (px[i * 3] as f32 - REC_MEAN) * REC_SCALE;
        data[i + area] = (px[i * 3 + 1] as f32 - REC_MEAN) * REC_SCALE;
        data[i + 2 * area] = (px[i * 3 + 2] as f32 - REC_MEAN) * REC_SCALE;
    }
    Ok(data)
}

impl PadRestore {
    /// 将模型输入空间的 xyxy 框还原并 clamp 到原图范围。
    fn restore(&self, xyxy: [f32; 4], image: &Image) -> [f32; 4] {
        let (w, h) = (image.width() as f32, image.height() as f32);
        [
            ((xyxy[0] - self.pad_left) / self.ratio).clamp(0.0, w),
            ((xyxy[1] - self.pad_top) / self.ratio).clamp(0.0, h),
            ((xyxy[2] - self.pad_left) / self.ratio).clamp(0.0, w),
            ((xyxy[3] - self.pad_top) / self.ratio).clamp(0.0, h),
        ]
    }

    /// 还原四角关键点坐标。
    fn restore_corners(&self, corners: [[f32; 2]; 4], image: &Image) -> [[f32; 2]; 4] {
        let (w, h) = (image.width() as f32, image.height() as f32);
        let mut out = [[0f32; 2]; 4];
        for (i, [x, y]) in corners.iter().enumerate() {
            out[i] = [
                ((x - self.pad_left) / self.ratio).clamp(0.0, w),
                ((y - self.pad_top) / self.ratio).clamp(0.0, h),
            ];
        }
        out
    }
}

// ==================== 裁剪与矫正 ====================

/// 裁剪车牌用于识别：有关键点时透视矫正到 94x24（对齐参考实现），否则退化为裁剪 + 拉伸。
fn crop_plate_box(image: &Image, plate: &PlateBox) -> Result<Image> {
    // 关键点缺失或退化时退回普通裁剪
    if let Some(corners) = plate.landmarks {
        if let Some(crop) = warp_plate(image, plate.xyxy, corners) {
            return Ok(crop);
        }
    }
    crop_with_padding(image, plate.xyxy, 0.0)
}

/// 带外扩 padding 的车牌裁剪（clamp 到图内，至少 1x1 像素）。
fn crop_with_padding(image: &Image, xyxy: [f32; 4], padding: f32) -> Result<Image> {
    let left = ((xyxy[0] - padding).floor() as i32).clamp(0, image.width() as i32 - 1);
    let top = ((xyxy[1] - padding).floor() as i32).clamp(0, image.height() as i32 - 1);
    let right = ((xyxy[2] + padding).ceil() as i32).clamp(left + 1, image.width() as i32);
    let bottom = ((xyxy[3] + padding).ceil() as i32).clamp(top + 1, image.height() as i32);
    image.crop(left as usize, top as usize, (right - left) as usize, (bottom - top) as usize)
}

/// 四角关键点透视矫正：车牌四边形 → 94x24 基准矩形（失败返回 None，调用方回退普通裁剪）。
fn warp_plate(image: &Image, xyxy: [f32; 4], corners: [[f32; 2]; 4]) -> Option<Image> {
    // 裁剪外接框（整数化、clamp），关键点转为框内局部坐标
    let left = ((xyxy[0].floor() as i32).max(0)) as usize;
    let top = ((xyxy[1].floor() as i32).max(0)) as usize;
    let width = ((xyxy[2].ceil() as i32).clamp(left as i32 + 1, image.width() as i32) - left as i32) as usize;
    let height = ((xyxy[3].ceil() as i32).clamp(top as i32 + 1, image.height() as i32) - top as i32) as usize;
    let crop = image.crop(left, top, width, height).ok()?;

    let mut src = [[0f32; 2]; 4];
    for (i, [x, y]) in corners.iter().enumerate() {
        src[i] = [x - left as f32, y - top as f32];
    }
    // H：车牌四边形 → 基准矩形；采样需用其逆（矩形 → 车牌）
    let h_fwd = solve_homography(src, REF_CORNERS)?;
    let h_inv = invert3x3(&h_fwd)?;

    let mut out = Image::filled(REC_WIDTH, REC_HEIGHT, image.channels(), 0);
    for dy in 0..REC_HEIGHT {
        for dx in 0..REC_WIDTH {
            let w = h_inv[2][0] * dx as f32 + h_inv[2][1] * dy as f32 + h_inv[2][2];
            if w.abs() < 1e-9 {
                continue;
            }
            let sx = (h_inv[0][0] * dx as f32 + h_inv[0][1] * dy as f32 + h_inv[0][2]) / w;
            let sy = (h_inv[1][0] * dx as f32 + h_inv[1][1] * dy as f32 + h_inv[1][2]) / w;
            if let Some(pixel) = bilinear_sample(&crop, sx, sy) {
                out.set_pixel(dx, dy, &pixel);
            }
        }
    }
    Some(out)
}

/// 双线性采样（越界返回 None，对应边界填 0）。
fn bilinear_sample(image: &Image, x: f32, y: f32) -> Option<Vec<u8>> {
    let (w, h, ch) = (image.width(), image.height(), image.channels());
    if x < -1.0 || y < -1.0 || x > w as f32 || y > h as f32 {
        return None;
    }
    let x0 = x.floor();
    let y0 = y.floor();
    let (fx, fy) = (x - x0, y - y0);
    let mut acc = vec![0f32; ch];
    for (dy, wy) in [(0.0, 1.0 - fy), (1.0, fy)] {
        for (dx, wx) in [(0.0, 1.0 - fx), (1.0, fx)] {
            let xi = (x0 + dx) as i64;
            let yi = (y0 + dy) as i64;
            if xi < 0 || yi < 0 || xi >= w as i64 || yi >= h as i64 {
                continue;
            }
            let off = (yi as usize * w + xi as usize) * ch;
            let weight = wx * wy;
            for c in 0..ch {
                acc[c] += image.data()[off + c] as f32 * weight;
            }
        }
    }
    let out = acc.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect();
    Some(out)
}

/// 解 4 点对单应矩阵（h33=1，8 元线性方程组 + 高斯列主元消元）；奇异时返回 None。
fn solve_homography(src: [[f32; 2]; 4], dst: [[f32; 2]; 4]) -> Option<[[f32; 3]; 3]> {
    let mut a = [[0f64; 9]; 8]; // 8x9 增广矩阵
    for i in 0..4 {
        let [sx, sy] = src[i];
        let [dx, dy] = dst[i];
        let (sx, sy, dx, dy) = (sx as f64, sy as f64, dx as f64, dy as f64);
        a[i * 2] = [sx, sy, 1.0, 0.0, 0.0, 0.0, -dx * sx, -dx * sy, dx];
        a[i * 2 + 1] = [0.0, 0.0, 0.0, sx, sy, 1.0, -dy * sx, -dy * sy, dy];
    }

    // 高斯消元（列主元）
    for col in 0..8 {
        let pivot = (col..8).fold(col, |best, r| if a[r][col].abs() > a[best][col].abs() { r } else { best });
        if a[pivot][col].abs() < 1e-10 {
            return None;
        }
        a.swap(col, pivot);
        for r in 0..8 {
            if r == col {
                continue;
            }
            let factor = a[r][col] / a[col][col];
            if factor == 0.0 {
                continue;
            }
            for c in col..9 {
                a[r][c] -= factor * a[col][c];
            }
        }
    }
    let h = [
        [a[0][8] / a[0][0], a[1][8] / a[1][1], a[2][8] / a[2][2]],
        [a[3][8] / a[3][3], a[4][8] / a[4][4], a[5][8] / a[5][5]],
        [a[6][8] / a[6][6], a[7][8] / a[7][7], 1.0],
    ];
    Some([
        [h[0][0] as f32, h[0][1] as f32, h[0][2] as f32],
        [h[1][0] as f32, h[1][1] as f32, h[1][2] as f32],
        [h[2][0] as f32, h[2][1] as f32, h[2][2] as f32],
    ])
}

/// 3x3 矩阵求逆（伴随矩阵法；行列式为 0 返回 None）。
fn invert3x3(m: &[[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ])
}

// ==================== 后处理 ====================

/// SSD anchor 表（640x640 输入，steps [8,16,32] × 2 尺寸，共 16800 个；
/// 生成顺序与训练导出严格一致：行外列内、小尺寸在前）。
fn generate_priors() -> Vec<[f32; 4]> {
    let mut priors = Vec::with_capacity(16800);
    let size = DET_INPUT_SIZE as f32;
    for (k, &step) in PRIOR_STEPS.iter().enumerate() {
        // ceil(640 / step)
        let cells = DET_INPUT_SIZE.div_ceil(step);
        for i in 0..cells {
            for j in 0..cells {
                let cx = (j as f32 + 0.5) * step as f32 / size;
                let cy = (i as f32 + 0.5) * step as f32 / size;
                for min_size in PRIOR_MIN_SIZES[k] {
                    priors.push([cx, cy, min_size / size, min_size / size]);
                }
            }
        }
    }
    priors
}

/// 行主序二维 float 矩阵视图。
struct Matrix2D {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Matrix2D {
    /// 将输出张量按 [rows, cols] 取出（兼容 3D [1,rows,cols] 与 2D 布局）。
    fn from_output(output: &crate::core::base::TensorOutput, cols_hint: usize) -> Result<Self> {
        let flat = output.as_f32()?;
        let shape = &output.shape;
        let (rows, cols) = match shape.len() {
            3 => (shape[1] as usize, shape[2] as usize),
            2 => (shape[0] as usize, shape[1] as usize),
            1 if cols_hint > 0 && flat.len() % cols_hint == 0 => (flat.len() / cols_hint, cols_hint),
            _ => {
                return Err(VisionError::inference(format!(
                    "mnet 车牌检测输出维度不支持: {:?}",
                    shape
                )))
            }
        };
        if cols != cols_hint {
            return Err(VisionError::inference(format!(
                "mnet 车牌检测输出末维 {} 与期望 {} 不符",
                cols, cols_hint
            )));
        }
        if flat.len() < rows * cols {
            return Err(VisionError::inference(format!(
                "mnet 车牌检测输出元素数 {} < {}x{}",
                flat.len(),
                rows,
                cols
            )));
        }
        Ok(Matrix2D {
            rows,
            cols,
            data: flat[..rows * cols].to_vec(),
        })
    }

    #[inline]
    fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }
}

/// 按名字定位 mnet 输出（忽略大小写包含匹配），失败时按末维尺寸兜底。
fn find_output<'a>(
    outputs: &'a [crate::core::base::TensorOutput],
    name_parts: &[&str],
    last_dim: usize,
) -> Result<&'a crate::core::base::TensorOutput> {
    for o in outputs {
        let name = o.name.to_lowercase();
        if name_parts.iter().any(|p| name.contains(p)) {
            return Ok(o);
        }
    }
    outputs
        .iter()
        .find(|o| o.shape.last().copied().unwrap_or(0) as usize == last_dim)
        .ok_or_else(|| {
            VisionError::inference(format!(
                "未找到 mnet 输出（候选名 {:?}，末维 {}）；实际: {:?}",
                name_parts,
                last_dim,
                outputs.iter().map(|o| o.name.as_str()).collect::<Vec<_>>()
            ))
        })
}

/// 读取输出为 [N, cols] 矩阵。
fn read_as_2d(output: &crate::core::base::TensorOutput, cols: usize) -> Result<Matrix2D> {
    Matrix2D::from_output(output, cols)
}

/// 车牌候选 NMS（单类，按分数降序贪心抑制）。
fn nms_plates(mut candidates: Vec<PlateBox>, iou_threshold: f32) -> Vec<PlateBox> {
    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let mut keep: Vec<PlateBox> = Vec::new();
    'outer: for cand in &candidates {
        for kept in &keep {
            if iou(&kept.xyxy, &cand.xyxy) > iou_threshold as f64 {
                continue 'outer;
            }
        }
        keep.push(*cand);
    }
    keep
}

/// 两 xyxy 框交并比。
fn iou(a: &[f32; 4], b: &[f32; 4]) -> f64 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    if inter <= 0.0 {
        return 0.0;
    }
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    (inter / (area_a + area_b - inter + 1e-9)) as f64
}

/// LPRNet CTC 解码：逐帧 softmax argmax → 相邻去重 → 去 blank（最后一类）。
/// 返回 `(车牌文本, 保留字符的平均 softmax 概率)`。
fn ctc_decode(frames: &[f32], classes: usize, frame_count: usize) -> (String, f32) {
    let blank = classes - 1; // LPRNet blank 固定为最后一类
    let mut text = String::new();
    let (mut prob_sum, mut kept) = (0f32, 0usize);
    let mut last = blank;

    for t in 0..frame_count {
        let row = &frames[t * classes..(t + 1) * classes];
        // argmax + softmax 概率
        let (mut best, mut best_logit) = (0usize, f32::MIN);
        let mut max_logit = f32::MIN;
        for (i, &v) in row.iter().enumerate() {
            if v > best_logit {
                best_logit = v;
                best = i;
            }
            if v > max_logit {
                max_logit = v;
            }
        }
        let mut sum_exp = 0f32;
        for &v in row {
            sum_exp += (v - max_logit).exp();
        }
        let prob = (best_logit - max_logit).exp() / sum_exp;

        if best != last && best != blank {
            if let Some(ch) = LPR_CHARS.get(best) {
                text.push_str(ch);
                prob_sum += prob;
                kept += 1;
            }
        }
        last = best;
    }

    (text, if kept > 0 { prob_sum / kept as f32 } else { 0.0 })
}

/// 车牌类型启发式：按车牌区域 BGR 均值判定颜色（蓝牌/绿牌/黄牌/白牌/黑牌）。
fn classify_plate_color(crop: &Image) -> String {
    if crop.is_empty() || crop.channels() < 3 {
        return "未知".to_string();
    }
    let px = crop.data();
    let n = px.len() / crop.channels();
    let mut sum = [0f64; 3];
    for i in 0..n {
        sum[0] += px[i * 3] as f64;
        sum[1] += px[i * 3 + 1] as f64;
        sum[2] += px[i * 3 + 2] as f64;
    }
    // Image 3 通道为 BGR 序
    let (b, g, r) = (sum[0] / n as f64, sum[1] / n as f64, sum[2] / n as f64);
    let mean_v = (b + g + r) / 3.0;
    let spread = b.max(g).max(r) - b.min(g).min(r);

    if g >= b + 15.0 && g >= r + 15.0 {
        "新能源绿牌".to_string()
    } else if b >= r + 15.0 && b >= g + 10.0 {
        "蓝牌".to_string()
    } else if r >= b + 30.0 && g >= b + 20.0 {
        "黄牌".to_string()
    } else if spread < 30.0 {
        if mean_v < 70.0 {
            "黑牌".to_string()
        } else {
            "白牌".to_string()
        }
    } else {
        "未知".to_string()
    }
}

/// 统一转 3 通道 BGR（库内 Image 3/4 通道为 BGR/BGRA 序，1 通道为灰度）。
fn to_bgr3(image: &Image) -> Result<Image> {
    match image.channels() {
        3 => Ok(image.clone()),
        4 => cvt_color(image, ColorConversion::Bgra2Bgr),
        1 => cvt_color(image, ColorConversion::Gray2Bgr),
        n => Err(VisionError::image(format!("不支持的通道数: {n}"))),
    }
}

/// 打印车牌识别结果日志。
fn log_results(results: &[PlateResult]) {
    tracing::info!(
        "LicensePlate: {} plate(s){}",
        results.len(),
        if results.is_empty() { "" } else { ":" }
    );
    for p in results {
        tracing::info!(
            "  text={:?}, box={:?}, score={:.3}, type={:?}",
            p.text,
            p.r#box,
            p.score,
            p.plate_type
        );
    }
}

// ==================== 统一接口 ====================

crate::impl_engine_forward!(LicensePlateEngine, base, Vec<PlateResult>,
    /// 单图车牌检测 + 识别。
    fn predict(&self, image: &Image) -> Result<Vec<PlateResult>> {
        self.recognize_impl(image)
    },
    /// 批量车牌检测 + 识别（逐张推理）。
    fn predict_batch(&self, images: &[Image]) -> Result<Vec<Vec<PlateResult>>> {
        images.iter().map(|img| self.recognize_impl(img)).collect()
    }
);

// ==================== 组合 API 别名 ====================

impl LicensePlateEngine {
    /// 车牌检测 + 识别（`recognize_impl` 的语义化别名，task 约定 API 名）。
    pub fn recognize(&self, image: &Image) -> Result<Vec<PlateResult>> {
        self.recognize_impl(image)
    }
}

// ==================== 自验测试 ====================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::device_type::DeviceType;

    const DET_MODEL: &str = "testmodels/mnet_plate.onnx";
    const REC_MODEL: &str = "testmodels/Final_LPRNet_model.onnx";
    const TEST_IMG: &str = "testmodels/plate_test.jpg";

    fn models_ready() -> bool {
        [DET_MODEL, REC_MODEL, TEST_IMG].iter().all(|p| std::path::Path::new(p).exists())
    }

    /// 端到端自验：检测框位置合理 + 文本非空 + 置信度过滤 + plate_type 非空。
    #[test]
    fn test_mnet_end_to_end() {
        if !models_ready() {
            eprintln!("skip: 模型或测试图不存在");
            return;
        }
        let engine = LicensePlateEngine::new(DET_MODEL, REC_MODEL, DeviceType::Cpu).unwrap();
        let image = Image::load(TEST_IMG).unwrap();
        let results = engine.recognize(&image).unwrap();
        println!("mnet 端到端结果:");
        for p in &results {
            println!("  text={:?} box={:?} score={:.3} type={:?}", p.text, p.r#box, p.score, p.plate_type);
        }
        assert!(!results.is_empty(), "至少应检测到一块车牌");
        let (w, h) = (image.width() as f32, image.height() as f32);
        for p in &results {
            // 框在图内且面积合理
            assert!(p.r#box[0] >= 0.0 && p.r#box[1] >= 0.0 && p.r#box[2] <= w && p.r#box[3] <= h);
            assert!(p.r#box[2] > p.r#box[0] && p.r#box[3] > p.r#box[1]);
            assert!(p.score > 0.1, "综合置信度过低: {}", p.score);
            assert!(!p.text.is_empty(), "识别文本不应为空");
            assert!(p.plate_type.is_some());
        }
    }

    /// 仅检测：验证置信度过滤与 NMS（框数量 == 高分去重数量）。
    #[test]
    fn test_detect_only_and_nms() {
        if !models_ready() {
            eprintln!("skip: 模型或测试图不存在");
            return;
        }
        let engine = LicensePlateEngine::new(DET_MODEL, REC_MODEL, DeviceType::Cpu).unwrap();
        let image = Image::load(TEST_IMG).unwrap();
        let plates = engine.detect_plates(&image).unwrap();
        println!("检测到 {} 块候选车牌", plates.len());
        for (i, p) in plates.iter().enumerate() {
            println!("  #{} score={:.3} box={:?}", i, p.score, p.xyxy);
            assert!(p.score >= engine.base.confidence_threshold());
        }
        // NMS 后不应有高度重叠的框
        for i in 0..plates.len() {
            for j in (i + 1)..plates.len() {
                assert!(
                    iou(&plates[i].xyxy, &plates[j].xyxy) <= engine.nms_threshold() as f64,
                    "NMS 后仍存在 IoU 超阈值的框"
                );
            }
        }
    }

    /// anchor 表规模与顺序校验（16800 个，行外列内、小尺寸在前）。
    #[test]
    fn test_priors_generation() {
        let priors = generate_priors();
        assert_eq!(priors.len(), 16800);
        // step=8 第一行第一列第一个 anchor：cx=0.5*8/640，尺寸 24/640
        assert!((priors[0][0] - 0.5 * 8.0 / 640.0).abs() < 1e-6);
        assert!((priors[0][2] - 24.0 / 640.0).abs() < 1e-6);
        // 同一 cell 内第二个 anchor 为 48/640
        assert!((priors[1][2] - 48.0 / 640.0).abs() < 1e-6);
        // 最后一层（step=32）末尾 anchor 为 768/640
        assert!((priors[16799][3] - 768.0 / 640.0).abs() < 1e-6);
    }

    /// CTC 解码单元测试：blank=最后一类、相邻去重、softmax 置信度。
    #[test]
    fn test_ctc_decode() {
        // 构造 3 帧：皖 / blank / A —— 类别索引：皖=12, A=41, blank=67
        let classes = LPR_CHARS.len();
        let mut frames = vec![-10f32; 3 * classes];
        frames[12] = 5.0; // 帧 0：皖
        frames[classes + 67] = 5.0; // 帧 1：blank
        frames[2 * classes + 41] = 5.0; // 帧 2：A
        let (text, score) = ctc_decode(&frames, classes, 3);
        assert_eq!(text, "皖A");
        assert!(score > 0.99, "softmax 置信度应接近 1，实际 {}", score);

        // 相邻重复去重：皖 皖 A → 皖A
        let mut frames2 = vec![-10f32; 3 * classes];
        frames2[12] = 5.0;
        frames2[classes + 12] = 5.0;
        frames2[2 * classes + 41] = 5.0;
        let (text2, _) = ctc_decode(&frames2, classes, 3);
        assert_eq!(text2, "皖A");
    }

    /// 单应矩阵求解与求逆往返一致性。
    #[test]
    fn test_homography() {
        // 单位映射（基准四角 → 自身）应为恒等矩阵
        let ident = solve_homography(REF_CORNERS, REF_CORNERS).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((ident[i][j] - expect).abs() < 1e-4);
            }
        }
        // 任意四边形 → 基准矩形：正向映射四角应精确落在基准上
        let src = [[10.0, 20.0], [140.0, 24.0], [14.0, 70.0], [150.0, 80.0]];
        let h = solve_homography(src, REF_CORNERS).unwrap();
        let apply = |m: &[[f32; 3]; 3], x: f32, y: f32| -> (f32, f32) {
            let w = m[2][0] * x + m[2][1] * y + m[2][2];
            (
                (m[0][0] * x + m[0][1] * y + m[0][2]) / w,
                (m[1][0] * x + m[1][1] * y + m[1][2]) / w,
            )
        };
        for (i, &[x, y]) in src.iter().enumerate() {
            let (px, py) = apply(&h, x, y);
            assert!((px - REF_CORNERS[i][0]).abs() < 1e-3, "正向 x 失败: {px}");
            assert!((py - REF_CORNERS[i][1]).abs() < 1e-3, "正向 y 失败: {py}");
        }
        // 逆映射：基准四角应还原回源四边形
        let h_inv = invert3x3(&h).unwrap();
        for (i, corner) in REF_CORNERS.iter().enumerate() {
            let (qx, qy) = apply(&h_inv, corner[0], corner[1]);
            assert!((qx - src[i][0]).abs() < 1e-2, "逆向 x 失败: {qx}");
            assert!((qy - src[i][1]).abs() < 1e-2, "逆向 y 失败: {qy}");
        }
    }

    /// 车牌颜色启发式：蓝牌 / 绿牌判定。
    #[test]
    fn test_plate_color() {
        // 蓝底白字（BGR: 蓝 180）
        let mut blue = Image::filled(94, 24, 3, 0);
        for y in 0..24 {
            for x in 0..94 {
                blue.set_pixel(x, y, &[180, 90, 40]);
            }
        }
        assert_eq!(classify_plate_color(&blue), "蓝牌");
        // 绿底（BGR: 绿 140）
        let mut green = Image::filled(94, 24, 3, 0);
        for y in 0..24 {
            for x in 0..94 {
                green.set_pixel(x, y, &[80, 140, 60]);
            }
        }
        assert_eq!(classify_plate_color(&green), "新能源绿牌");
    }
}
