//! 实例分割推理引擎。
//!
//! **支持的模型与后处理路径**（按 [`SegModelType`] 分派，互不影响）：
//!
//! | 模型 | 工厂方法 | 预处理 | 输出布局 | 掩码机制 |
//! |---|---|---|---|---|
//! | YOLOv8/v11-seg（传统格式） | [`Self::for_yolo`]（自动识别） | letterbox（114 灰边）+ /255 | `[1, 4+nc+32, anchors]` + proto `[1, 32, 160, 160]` | 32 维系数 · 32 通道原型，系数随检测输出 |
//! | YOLO26-seg（End2End NMS-Free） | [`Self::for_yolo_end2end`] 或 `for_yolo` 自动识别 | 同上 | `[1, 300, 4+1+1+32]` + proto | 同上，系数拼接在每行检测末尾 |
//! | RF-DETR-Seg（Roboflow 导出） | [`Self::for_rfdetr_seg`] | 拉伸 resize + ImageNet 归一化 | dets `[1,Q,4]` + labels `[1,Q,nc]`(logits) + masks `[1,Q,H,W]`(logits) | 每 query 直出掩码 logits，sigmoid 后上采样到原图 |
//! | YOLOE（文本/视觉提示；上游实现经 `OnnxEngineFactory#createYOLOEEngine` 复用本引擎） | `new` / `for_yolo` | letterbox + /255 | 同 YOLO 对应格式（11/v8 传统布局，26 系列 End2End 布局） | 同 YOLO 对应格式；类别由导出时烘焙的提示决定 |
//!
//! **路径选择规则**：`seg_model_type == Yolo`（默认）时，首次推理按检测输出 shape
//! 自动识别传统/End2End 格式：End2End 要求 dim1∈[100,400] **且** dim2∈[4,100]
//! （两者结合判断 —— 80 类传统模型 channels=4+80+32=116 恰好落在 [100,400]，
//! 仅判 dim1 会误判为 End2End 导致解析出垃圾结果，此为历史 bug，已修复）。
//! 传统格式的类别数按 `channels - 4 - 32` 推断。
//! `seg_model_type == RfDetrSeg` 走 [`Self::predict_rf_detr_seg`]：拉伸 resize 预处理，
//! logits 过 sigmoid → Q×C 扁平 top-k（与官方 RFDETRSeg.predict 对齐），掩码按 query 分段
//! 读取、sigmoid 后直接上采样到原图。
//!
//! **掩码约定**（与所有路径一致）：[`Segmentation::mask`] 为 float32 概率图（对应 CV_32F），
//! 尺寸等于原图，bbox 区域外为 0；`binary_mask()` 按阈值（默认 0.35）二值化。
//!
//! **注意事项**：
//! - RT-DETR（百度/Ultralytics）官方只有检测模型，不存在分割变体；
//!   transformer 系分割由 RF-DETR-Seg 与 YOLOE 覆盖
//! - RF-DETR-Seg / YOLOE 导出的 ONNX 若不含类别名元数据，用 `set_labels` 指定
//!   （RF-DETR-Seg 为 COCO 91 类原始 id：person=1、bus=6）
//! - NMS 阈值：Ultralytics 默认 0.7；阈值过小（如 0.45）会得到比官方更多的重叠检出
//! - 实例非线程安全语义与原实现一致；Rust 侧因 `&self` 推理接口改为无锁只读状态
//!   （letterbox 参数随单次推理传递，自动检测结果用 `OnceLock` 缓存）

use std::sync::OnceLock;

use crate::core::base::{BaseOnnxEngine, TensorOutput};
use crate::core::device_type::DeviceType;
use crate::error::{Result, VisionError};
use crate::imaging::{FloatMask, Image, Interpolation, cvt_color, resize, ColorConversion};
use crate::model::BoundingBox;
use crate::model::Segmentation;

/// ImageNet 归一化均值（对应 `DetectionEngine.IMAGENET_MEAN`，RF-DETR-Seg 预处理用）。
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// ImageNet 归一化标准差（对应 `DetectionEngine.IMAGENET_STD`）。
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// 分割模型类型（对应 `SegmentationEngine.SegModelType`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegModelType {
    /// YOLO-Seg（默认，含传统与 End2End 格式自动识别）
    Yolo,
    /// RF-DETR-Seg（Roboflow）
    RfDetrSeg,
}

/// Letterbox 参数（随单次推理传递，
/// 语义不变：仅在同一帧的预处理→后处理链路内使用）。
#[derive(Debug, Clone, Copy)]
struct LetterboxParams {
    /// 缩放比例
    ratio: f32,
    /// 宽度方向 padding 的一半
    dw: f32,
    /// 高度方向 padding 的一半
    dh: f32,
}

/// 传统路径候选框（对应 `float[7+maskProtoDim]`：
/// `[x1,y1,x2,y2,conf,cls,anchorIdx, maskCoeffs...]`）。
#[derive(Debug, Clone)]
struct Candidate {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    conf: f32,
    class_id: i32,
    /// anchor 索引（上游中存储但未参与后续逻辑，保留字段）
    #[allow(dead_code)]
    anchor_idx: usize,
    /// 32 维掩码系数
    coeffs: Vec<f32>,
}

impl Candidate {
    /// 框坐标数组 [x1, y1, x2, y2]。
    fn box_array(&self) -> [f32; 4] {
        [self.x1, self.y1, self.x2, self.y2]
    }
}

/// 掩码原型（对应 `protoArrayToMat` 生成的 `[dim, h*w]` CV_32F Mat 布局）。
#[derive(Debug, Clone)]
struct ProtoMasks {
    /// 原型通道数（标准为 32）
    dim: usize,
    /// 原型掩码高
    height: usize,
    /// 原型掩码宽
    width: usize,
    /// 行主序数据：`data[d * h * w + y * w + x]`
    data: Vec<f32>,
}

/// 输出缓存（对应上游内部类 `OutputCache`）。
///
/// 上游实现为避免重复 native 调用导致 crash 一次性缓存 Value 与 shape；
/// Rust 侧 [`TensorOutput`] 本身就是整段拷贝的 owned 快照，这里仅提供查找辅助。
struct OutputCache<'a> {
    outputs: &'a [TensorOutput],
}

impl<'a> OutputCache<'a> {
    fn new(outputs: &'a [TensorOutput]) -> Self {
        OutputCache { outputs }
    }

    fn print_debug(&self) {
        for (i, o) in self.outputs.iter().enumerate() {
            tracing::debug!("  output[{}]: shape={:?}, elementCount={}", i, o.shape, o.element_count());
        }
    }

    /// 找 4D 输出（掩码原型）。
    fn find_4d(&self) -> Option<usize> {
        self.outputs.iter().position(|o| o.shape.len() == 4)
    }

    /// 找 3D 输出（检测输出），跳过已选为 proto 的。
    fn find_3d(&self, exclude: Option<usize>) -> Option<usize> {
        self.outputs
            .iter()
            .enumerate()
            .position(|(i, o)| Some(i) != exclude && o.shape.len() == 3)
    }

    /// 找最大的非 `[1]` 输出。
    fn find_largest(&self, exclude: Option<usize>) -> Option<usize> {
        let mut best = None;
        let mut best_count = 1usize;
        for (i, o) in self.outputs.iter().enumerate() {
            if Some(i) == exclude {
                continue;
            }
            if o.element_count() > best_count {
                best_count = o.element_count();
                best = Some(i);
            }
        }
        best
    }

    fn shape_of(&self, idx: usize) -> &[i64] {
        &self.outputs[idx].shape
    }
}

/// 实例分割推理引擎。
pub struct SegmentationEngine {
    /// 组合基类（对应 基础引擎 继承）
    base: BaseOnnxEngine,

    /// NMS IoU 阈值（默认 0.45）
    nms_threshold: f32,
    /// 掩码阈值（引擎侧仅存取；二值化在 `Segmentation` 模型侧按其默认 0.35 执行）
    mask_threshold: f32,
    /// 是否为 end2end 模型（构造时指定；实际类型以自动检测为准）
    end2end: bool,
    /// 分割模型类型（默认 Yolo）
    seg_model_type: SegModelType,
    /// Transformer 输出保留的最大检测数（对应官方 top-k num_select，默认 300）
    max_detections: usize,

    /// 自动检测的模型类型缓存 `(is_end2end, num_classes)`（首次推理时检测，
    /// 上游中为 `autoDetectedEnd2End`/`autoDetectedNumClasses` 字段）
    auto_detected: OnceLock<(bool, i32)>,
}

impl SegmentationEngine {
    /// 掩码原型维度（对应上游 字段 `maskProtoDim = 32`）。
    const MASK_PROTO_DIM: usize = 32;

    // ==================== 构造 ====================

    /// 创建 YOLO 分割引擎（自动识别传统 / End2End 格式）。
    ///
    /// 对应 `forYOLO(modelPath, deviceType, confThreshold, nmsThreshold)`。
    pub fn for_yolo(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        conf_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Self> {
        Self::for_yolo_with_input_size(model_path, device_type, conf_threshold, nms_threshold, -1, -1)
    }

    /// 创建 YOLO 分割引擎（指定输入尺寸）。
    ///
    /// 对应 `forYOLO(modelPath, deviceType, confThreshold, nmsThreshold, inputHeight, inputWidth)`。
    pub fn for_yolo_with_input_size(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        conf_threshold: f32,
        nms_threshold: f32,
        input_height: i32,
        input_width: i32,
    ) -> Result<Self> {
        let mut engine = Self::with_input_size(model_path, device_type, false, input_height, input_width)?;
        engine.base.set_confidence_threshold(conf_threshold);
        engine.nms_threshold = nms_threshold;
        Ok(engine)
    }

    /// 创建 YOLO End2End 分割引擎。
    ///
    /// 对应 `forYOLOEnd2End(modelPath, deviceType, confThreshold)`。
    pub fn for_yolo_end2end(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        conf_threshold: f32,
    ) -> Result<Self> {
        Self::for_yolo_end2end_with_input_size(model_path, device_type, conf_threshold, -1, -1)
    }

    /// 创建 YOLO End2End 分割引擎（指定输入尺寸）。
    ///
    /// 对应 `forYOLOEnd2End(modelPath, deviceType, confThreshold, inputHeight, inputWidth)`。
    pub fn for_yolo_end2end_with_input_size(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        conf_threshold: f32,
        input_height: i32,
        input_width: i32,
    ) -> Result<Self> {
        let mut engine = Self::with_input_size(model_path, device_type, true, input_height, input_width)?;
        engine.base.set_confidence_threshold(conf_threshold);
        Ok(engine)
    }

    /// 创建分割引擎（对应上游 构造器 `SegmentationEngine(modelPath, deviceType, end2End)`）。
    pub fn new(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        end2end: bool,
    ) -> Result<Self> {
        Self::with_input_size(model_path, device_type, end2end, -1, -1)
    }

    /// 创建分割引擎（指定输入尺寸，对应上游 构造器
    /// `SegmentationEngine(modelPath, deviceType, end2End, inputHeight, inputWidth)`）。
    pub fn with_input_size(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        end2end: bool,
        input_height: i32,
        input_width: i32,
    ) -> Result<Self> {
        let base = BaseOnnxEngine::with_input_size(model_path, device_type, input_height, input_width)?;
        let mut engine = SegmentationEngine {
            base,
            nms_threshold: 0.45,
            mask_threshold: 0.35,
            end2end,
            seg_model_type: SegModelType::Yolo,
            max_detections: 300,
            auto_detected: OnceLock::new(),
        };
        // 上游构造器显式设置 mean={0,0,0} / std={1,1,1}（与基类默认一致，保留）
        engine.base.set_normalization([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        Ok(engine)
    }

    /// 创建 RF-DETR-Seg 分割引擎（Roboflow，ONNX 由 RFDETRSeg.export 导出）。
    ///
    /// 输入：拉伸 resize + ImageNet 归一化；输出 dets/labels/masks（后两者为 logits）。
    /// RF-DETR 导出的 ONNX 通常不含类别名元数据，可用 `set_labels` 手动设置。
    pub fn for_rfdetr_seg(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        conf_threshold: f32,
    ) -> Result<Self> {
        let mut engine = Self::with_input_size(model_path, device_type, false, -1, -1)?;
        engine.base.set_confidence_threshold(conf_threshold);
        engine.base.set_normalization(IMAGENET_MEAN, IMAGENET_STD);
        engine.seg_model_type = SegModelType::RfDetrSeg;
        Ok(engine)
    }

    // ==================== 访问器 ====================

    /// NMS IoU 阈值。
    pub fn nms_threshold(&self) -> f32 {
        self.nms_threshold
    }

    /// 设置 NMS IoU 阈值。
    pub fn set_nms_threshold(&mut self, threshold: f32) {
        self.nms_threshold = threshold;
    }

    /// 掩码阈值。
    pub fn mask_threshold(&self) -> f32 {
        self.mask_threshold
    }

    /// 设置掩码阈值。
    pub fn set_mask_threshold(&mut self, threshold: f32) {
        self.mask_threshold = threshold;
    }

    /// Transformer 输出保留的最大检测数。
    pub fn max_detections(&self) -> usize {
        self.max_detections
    }

    /// 设置 Transformer 输出保留的最大检测数。
    pub fn set_max_detections(&mut self, max_detections: usize) {
        self.max_detections = max_detections;
    }

    /// 是否为 end2end 模型（对应 `isEnd2End()`；为构造时指定值，
    /// 实际处理后缀以首次推理的自动检测为准）。
    pub fn is_end2end(&self) -> bool {
        self.end2end
    }

    /// 分割模型类型。
    pub fn seg_model_type(&self) -> SegModelType {
        self.seg_model_type
    }

    // ==================== 推理入口 ====================

    /// 单图推理（对应 `predict`，由 [`crate::impl_engine_forward!`] 转发 trait）。
    pub fn predict_impl(&self, image: &Image) -> Result<Vec<Segmentation>> {
        // SAHI 开启时：切片 → 逐片调用原始单图路径 → 掩码平移/并集合并（返回类型不变）
        if self.base.is_sahi_enabled() {
            let config = self
                .base
                .sahi_config()
                .expect("sahi config must exist when SAHI enabled");
            return crate::sahi::sliced_predictor::predict_sliced_segmentation(self, self, image, config)
                .map(|result| result.detections);
        }
        self.predict_without_sahi(image)
    }

    /// 原始单图推理路径（SAHI 关闭时的 [`Self::predict_impl`] 行为；SAHI 开启时被逐切片调用）。
    pub fn predict_without_sahi(&self, image: &Image) -> Result<Vec<Segmentation>> {
        // RF-DETR-Seg：拉伸 resize 预处理 + 独立后处理路径
        if self.seg_model_type == SegModelType::RfDetrSeg {
            return self.predict_rf_detr_seg(image);
        }

        let orig_width = image.width() as i32;
        let orig_height = image.height() as i32;

        // 1. 预处理
        let (input_data, lb) = self.preprocess_with_letterbox(image)?;

        // 2. 创建 Tensor
        let input_tensor = self.base.create_input_tensor(input_data)?;

        // 3. 运行推理（获取所有输出）
        let outputs = self.base.run_multi_output(input_tensor)?;

        // 4. 缓存所有输出信息（Rust 侧 TensorOutput 已是 owned 快照，避免生命周期问题）
        let cache = OutputCache::new(&outputs);

        // 5. 自动检测模型类型（首次推理时，基于实际输出 shape；结果缓存）
        let (is_end2end, num_classes) = self.detected_model_type(&cache);

        // 6. 后处理：优先使用自动检测结果，覆盖构造时指定的类型
        let results = if is_end2end {
            self.postprocess_end2_end(&cache, lb, (is_end2end, num_classes), orig_width, orig_height)?
        } else {
            self.postprocess_standard(&cache, lb, num_classes, orig_width, orig_height)?
        };

        self.log_results(&results);
        Ok(results)
    }

    // ==================== RF-DETR-Seg 路径 ====================

    /// RF-DETR-Seg 推理（Roboflow）。
    ///
    /// 预处理：拉伸 resize + ImageNet 归一化（与官方 RFDETRSeg.predict 一致）。
    /// 后处理：labels logits 过 sigmoid → Q×C 扁平 top-k → 阈值过滤 →
    /// 框 cxcywh 反归一化到原图，mask logits 过 sigmoid 后直接上采样到原图（掩码与输入同为拉伸空间）。
    /// 掩码只对通过阈值的检测做解码与上采样（上游实现按 query 分段读取大张量控制内存，
    /// Rust 侧 TensorOutput 已整段快照，这里按 query 偏移切片）。
    pub fn predict_rf_detr_seg(&self, image: &Image) -> Result<Vec<Segmentation>> {
        let orig_width = image.width() as i32;
        let orig_height = image.height() as i32;

        let input_data = self.base.preprocess(image)?;
        let input_tensor = self.base.create_input_tensor(input_data)?;
        let outputs = self.base.run_multi_output(input_tensor)?;

        // 1. 按名字/shape 定位三个输出：dets [1,Q,4]、labels [1,Q,nc]、masks [1,Q,H,W]
        //    （循环内允许后续匹配覆盖前面的索引：最后一个匹配生效）
        let mut det_idx: Option<usize> = None;
        let mut label_idx: Option<usize> = None;
        let mut mask_idx: Option<usize> = None;
        for i in 0..outputs.len() {
            let name = self
                .base
                .output_names()
                .get(i)
                .map(|s| s.to_lowercase())
                .unwrap_or_default();
            let shape = &outputs[i].shape;
            if shape.len() == 4 || name.contains("mask") {
                mask_idx = Some(i);
            } else if (shape.len() == 3 && shape.get(2) == Some(&4)) || name.contains("det") {
                det_idx = Some(i);
            } else if shape.len() == 3
                || name.contains("label")
                || name.contains("logit")
                || name.contains("score")
            {
                label_idx = Some(i);
            }
        }
        let (Some(det_idx), Some(label_idx)) = (det_idx, label_idx) else {
            let shapes: Vec<&[i64]> = outputs.iter().map(|o| o.shape.as_slice()).collect();
            return Err(VisionError::inference(format!(
                "Cannot locate dets/labels outputs for RF-DETR-Seg. Shapes: {shapes:?}"
            )));
        };

        // 2. 读取框与分数
        let boxes = read_rank3_as_2d(&outputs[det_idx])?;
        let mut scores = read_rank3_as_2d(&outputs[label_idx])?;
        let num_queries = boxes.len();
        if scores.is_empty() {
            return Err(VisionError::inference("RF-DETR-Seg labels output has 0 rows"));
        }
        if scores.len() != num_queries {
            return Err(VisionError::inference(format!(
                "RF-DETR-Seg dets rows ({}) != labels rows ({})",
                num_queries,
                scores.len()
            )));
        }
        let num_classes = scores[0].len();

        // 3. logits → sigmoid
        for row in scores.iter_mut() {
            for v in row.iter_mut() {
                *v = sigmoid(*v);
            }
        }

        // 4. Q×C 扁平 top-k（稳定排序）+ 阈值过滤，与官方 PostProcess._select_topk 对齐
        let mut pairs: Vec<[f32; 3]> = Vec::new(); // {conf, q, c}
        for q in 0..num_queries {
            for c in 0..num_classes {
                if scores[q][c] > self.base.confidence_threshold() {
                    pairs.push([scores[q][c], q as f32, c as f32]);
                }
            }
        }
        pairs.sort_by(|a, b| b[0].total_cmp(&a[0]));

        // 5. 掩码输出信息：masks 布局 [batch, Q, H, W]，本引擎固定 batch=1，
        //    query 维度按检测索引直接寻址
        let (mask_h, mask_w, mask_data): (usize, usize, Option<&[f32]>) = match mask_idx {
            Some(mi) => {
                let ms = &outputs[mi].shape;
                let mh = if ms.len() == 4 { ms[2].max(0) as usize } else { 1 };
                let mw = if ms.len() == 4 { ms[3].max(0) as usize } else { 1 };
                (mh, mw, Some(outputs[mi].as_f32()?))
            }
            None => (1, 1, None),
        };

        // 6. 解码
        let mut results: Vec<Segmentation> = Vec::new();
        for pair in &pairs {
            if results.len() >= self.max_detections {
                break;
            }
            let q = pair[1] as usize;
            let cls_id = pair[2] as i32;
            let conf = pair[0];

            // 框：归一化 cxcywh → xyxy → 原图像素（clamp 到图内）
            let (cx, cy, bw, bh) = (boxes[q][0], boxes[q][1], boxes[q][2], boxes[q][3]);
            let x1 = ((cx - bw / 2.0) * orig_width as f32).max(0.0);
            let y1 = ((cy - bh / 2.0) * orig_height as f32).max(0.0);
            let x2 = ((cx + bw / 2.0) * orig_width as f32).min(orig_width as f32);
            let y2 = ((cy + bh / 2.0) * orig_height as f32).min(orig_height as f32);

            let mask = match mask_data {
                Some(data) => Some(self.decode_rf_detr_mask(
                    data, q, mask_h, mask_w, orig_width, orig_height, x1, y1, x2, y2,
                )?),
                // 全 0 掩码画布（origHeight × origWidth，CV_32F）
                None => Some(FloatMask::new(orig_width as usize, orig_height as usize)),
            };

            results.push(Segmentation::new(
                self.base.get_label_name(cls_id),
                cls_id,
                x1 as f64,
                y1 as f64,
                x2 as f64,
                y2 as f64,
                conf as f64,
                mask,
            ));
        }

        self.log_results(&results);
        Ok(results)
    }

    /// 解码单个 query 的 RF-DETR 掩码：按偏移分段读取 `[H,W]` logits → sigmoid →
    /// 上采样到原图尺寸 → 裁剪到 bbox（与 YOLO-Seg 路径相同的掩码约定）。
    #[allow(clippy::too_many_arguments)]
    fn decode_rf_detr_mask(
        &self,
        flat: &[f32],
        query: usize,
        mh: usize,
        mw: usize,
        orig_width: i32,
        orig_height: i32,
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
    ) -> Result<FloatMask> {
        // 1. 分段读取该 query 的 mask logits（对应 `getFloatArrayRange`，
        //    Rust 侧输出已整段快照，这里按偏移切片拷贝）
        let start = query * mh * mw;
        let end = start + mh * mw;
        if end > flat.len() {
            return Err(VisionError::inference(format!(
                "RF-DETR-Seg mask range [{}, {}) out of bounds {}",
                start,
                end,
                flat.len()
            )));
        }
        let data: Vec<f32> = flat[start..end].iter().map(|&v| sigmoid(v)).collect();
        let mask_small = FloatMask::from_raw(mw, mh, data)?;

        // 2. 上采样到原图
        let mask_orig = mask_small.resize(orig_width as usize, orig_height as usize);

        // 3. 裁剪到 bbox，bbox 外为 0
        Ok(Self::crop_mask_to_bbox(mask_orig, orig_width, orig_height, x1, y1, x2, y2))
    }

    // ==================== 预处理 ====================

    /// 带 Letterbox 的预处理（对应 `preprocessWithLetterbox`）。
    ///
    /// 返回 `(CHW float 数据, letterbox 参数)`；letterbox 参数供后处理还原坐标使用。
    fn preprocess_with_letterbox(&self, image: &Image) -> Result<(Vec<f32>, LetterboxParams)> {
        let orig_width = image.width() as i32;
        let orig_height = image.height() as i32;
        let input_width = self.base.input_width();
        let input_height = self.base.input_height();

        let ratio = (input_width as f32 / orig_width as f32)
            .min(input_height as f32 / orig_height as f32);
        let new_width = (orig_width as f32 * ratio).round() as i32;
        let new_height = (orig_height as f32 * ratio).round() as i32;

        let dw = (input_width - new_width) as f32 / 2.0;
        let dh = (input_height - new_height) as f32 / 2.0;

        // 实现假定输入为 3 通道 BGR（贴入 CV_8UC3 画布）；对 1/4 通道输入先转 BGR 保持兼容
        let bgr = match image.channels() {
            3 => image.clone(),
            4 => cvt_color(image, ColorConversion::Bgra2Bgr)?,
            1 => cvt_color(image, ColorConversion::Gray2Bgr)?,
            c => return Err(VisionError::image(format!("unsupported channel count: {c}"))),
        };
        let resized = resize(&bgr, new_width as usize, new_height as usize, Interpolation::Linear)?;

        // 114 灰边画布（inputHeight × inputWidth，CV_8UC3）
        let mut padded = Image::filled(input_width as usize, input_height as usize, 3, 114);
        let top = (dh - 0.1).round().max(0.0) as usize;
        let left = (dw - 0.1).round().max(0.0) as usize;
        padded.paste(left, top, &resized);

        let rgb = cvt_color(&padded, ColorConversion::Bgr2Rgb)?;

        // HWC → CHW，/255（letterbox 路径固定 x/255，不走 mean/std）
        let input_channels = self.base.input_channels() as usize;
        let mut float_data = vec![0f32; input_channels * input_height as usize * input_width as usize];
        let area = (input_height * input_width) as usize;
        let px = rgb.data();

        for i in 0..area {
            float_data[i] = px[i * 3] as f32 / 255.0;
            float_data[i + area] = px[i * 3 + 1] as f32 / 255.0;
            float_data[i + 2 * area] = px[i * 3 + 2] as f32 / 255.0;
        }

        Ok((
            float_data,
            LetterboxParams { ratio, dw, dh },
        ))
    }

    // ==================== 模型类型自动检测 ====================

    /// 自动检测模型类型（首次推理时基于输出 tensor 的实际 shape，之后缓存）。
    ///
    /// End2End (YOLO26):        `[1, ~300, ~38]` — dim1 较小 (100~400)，dim2 = 4+1+1+32
    /// Traditional (YOLOv8/11): `[1, ~116, ~8400]` — dim1 是 channels，dim2 是 anchors（较大）
    fn detected_model_type(&self, cache: &OutputCache<'_>) -> (bool, i32) {
        *self.auto_detected.get_or_init(|| self.auto_detect_model_type(cache))
    }

    fn auto_detect_model_type(&self, cache: &OutputCache<'_>) -> (bool, i32) {
        // 找 3D 检测输出
        let proto_idx = cache.find_4d();
        let det_idx = match proto_idx {
            Some(p) => cache.find_3d(Some(p)),
            None => cache.find_largest(None),
        };

        let Some(det_idx) = det_idx else {
            return (false, 1);
        };

        let det_shape = cache.shape_of(det_idx);
        if det_shape.len() != 3 {
            return (false, 1);
        }

        let dim1 = det_shape[1]; // channels 或 num_detections
        let dim2 = det_shape[2]; // anchors 或 per_det_size

        // End2End (YOLO26): dim1 是检测数量（如 300），dim2 是每个检测属性数（如 38）
        // Traditional: dim1 是 channels（如 4+80+32=116），dim2 是 anchors（如 8400，较大）
        // 注意：仅判 dim1 不够 —— 80 类传统模型 channels=116 恰好落在 [100,400]，需结合 dim2 区分
        let is_end2end = (100..=400).contains(&dim1) && (4..=100).contains(&dim2);
        let num_classes = if is_end2end {
            self.base.labels().map(|l| l.len()).unwrap_or(1) as i32
        } else {
            // channels = 4 + numClasses + 32
            dim1 as i32 - 4 - Self::MASK_PROTO_DIM as i32
        };

        tracing::info!(
            "Model type auto-detected: {}, detShape=[{}, {}, {}], numClasses={}",
            if is_end2end { "End2End" } else { "Traditional" },
            det_shape[0],
            dim1,
            dim2,
            num_classes
        );

        (is_end2end, num_classes)
    }

    // ==================== End2End 后处理 ====================

    /// End2End 模型后处理（YOLO26-seg 等，内置 NMS）。
    /// 输出格式: `[1, 300, 38]` -> 每行 `[x1, y1, x2, y2, conf, clsId, mask0..mask31]`
    fn postprocess_end2_end(
        &self,
        cache: &OutputCache<'_>,
        lb: LetterboxParams,
        detected: (bool, i32),
        orig_width: i32,
        orig_height: i32,
    ) -> Result<Vec<Segmentation>> {
        tracing::debug!("=== End2End Segmentation PostProcess ===");
        cache.print_debug();

        // 找检测输出和掩码原型
        let proto_idx = cache.find_4d();
        let det_idx = match proto_idx {
            Some(p) => cache.find_3d(Some(p)),
            None => cache.find_largest(None),
        };

        let Some(det_idx) = det_idx else {
            tracing::error!("ERROR: No valid detection output found");
            return Ok(Vec::new());
        };

        // 解析检测输出
        let detections = match self.parse_detection_output(&cache.outputs[det_idx])? {
            Some(d) => d,
            None => return Ok(Vec::new()),
        };

        // 模型类型已在 predict() 中自动检测
        let (is_end2end, num_classes) = detected;
        if !is_end2end {
            // 实际是 Traditional 模型，切换处理
            tracing::info!("Auto-switching to Traditional processing");
            let proto = self.parse_proto_output(cache, proto_idx)?;
            return self.postprocess_traditional_from_parsed(
                &detections,
                proto.as_ref(),
                lb,
                num_classes,
                orig_width,
                orig_height,
            );
        }

        // 解析掩码原型
        let proto = self.parse_proto_output(cache, proto_idx)?;

        let mut results: Vec<Segmentation> = Vec::new();

        // 布局判定（每次推理一次，开销可忽略）：
        // - 标准 End2End（YOLO26-seg 等）: [x1,y1,x2,y2, conf, clsId, mask0..N-1]
        // - YOLOE-26 导出:         [x1,y1,x2,y2, cls0分..clsN-1分, mask0..N-1]
        // 依据：行长 - 4 - 掩码通道数 N 的剩余列数 == 元数据类别数；rem==2 的二义
        // 场景用 clsId 列是否为近似整数消歧（类别 id 必为整数，逐类分数为连续值）。
        let nm = proto_idx
            .and_then(|idx| cache.outputs.get(idx))
            .and_then(|o| o.shape.get(1).copied())
            .filter(|&c| c > 0)
            .unwrap_or(Self::MASK_PROTO_DIM as i64) as usize;
        let labels_len = self.base.labels().map(|l| l.len()).unwrap_or(0);
        let row_len = detections.first().map(|d| d.len()).unwrap_or(0);
        let rem = row_len as i64 - 4 - nm as i64;
        let per_class_candidate = rem > 0 && labels_len as i64 == rem;
        let mut per_class = per_class_candidate && rem != 2;
        if per_class_candidate && rem == 2 {
            // rem==2 二义（[conf, clsId] vs [cls0分, cls1分]）：两种解释各数一次
            // 低阈值检出数。标准解释完全检不出而逐类解释有检出 → YOLOE-26 布局
            // （其 sigmoid 分数饱和后恰为 0/1，与整数类别 id 无法用数值区分）。
            let low = 0.05f32;
            let conf_cls_hits = detections
                .iter()
                .filter(|d| d.len() >= 6 && d[4] >= low)
                .count();
            let per_class_hits = detections
                .iter()
                .filter(|d| d.len() >= 6 && (d[4] >= low || d[5] >= low))
                .count();
            if conf_cls_hits == 0 && per_class_hits > 0 {
                per_class = true;
            }
            tracing::debug!(
                "End2End rem==2 disambiguation: conf_cls_hits={conf_cls_hits}, per_class_hits={per_class_hits}, per_class={per_class}"
            );
        }
        tracing::debug!(
            "End2End layout: row_len={row_len}, nm={nm}, rem={rem}, labels={labels_len}, per_class={per_class}"
        );
        let coeff_off = if per_class { 4 + rem as usize } else { 6 };

        for det in &detections {
            if det.len() < coeff_off + nm {
                continue;
            }

            let (conf, cls_id) = if per_class {
                // 逐类分数布局：取分数最高的类别
                let scores = &det[4..coeff_off];
                let mut best = 0usize;
                let mut best_score = f32::NEG_INFINITY;
                for (i, &v) in scores.iter().enumerate() {
                    if v > best_score {
                        best_score = v;
                        best = i;
                    }
                }
                (best_score, best as i32)
            } else {
                (det[4], det[5] as i32)
            };
            if conf < self.base.confidence_threshold() {
                continue;
            }

            // End2End 输出已经是 xyxy 格式
            let x1 = ((det[0] - lb.dw) / lb.ratio).clamp(0.0, (orig_width - 1) as f32);
            let y1 = ((det[1] - lb.dh) / lb.ratio).clamp(0.0, (orig_height - 1) as f32);
            let x2 = ((det[2] - lb.dw) / lb.ratio).clamp(0.0, orig_width as f32);
            let y2 = ((det[3] - lb.dh) / lb.ratio).clamp(0.0, orig_height as f32);

            // 提取 mask 系数 [coeff_off..coeff_off+nm]
            let coeffs: Option<&[f32]> = if det.len() >= coeff_off + nm {
                Some(&det[coeff_off..coeff_off + nm])
            } else {
                None
            };

            // 生成掩码
            let mask = match (proto.as_ref(), coeffs) {
                (Some(p), Some(c)) => Some(self.decode_mask(
                    c, p, lb, orig_width, orig_height, x1, y1, x2, y2,
                )?),
                // 全 0 掩码画布（origHeight × origWidth，CV_32F）
                _ => Some(FloatMask::new(orig_width as usize, orig_height as usize)),
            };

            results.push(Segmentation::new(
                self.base.get_label_name(cls_id),
                cls_id,
                x1 as f64,
                y1 as f64,
                x2 as f64,
                y2 as f64,
                conf as f64,
                mask,
            ));
        }

        Ok(results)
    }

    // ==================== Traditional 后处理 ====================

    /// 标准模型后处理（YOLOv8/v11-seg，需要 NMS）。
    fn postprocess_standard(
        &self,
        cache: &OutputCache<'_>,
        lb: LetterboxParams,
        num_classes: i32,
        orig_width: i32,
        orig_height: i32,
    ) -> Result<Vec<Segmentation>> {
        tracing::debug!("=== Standard Segmentation PostProcess ===");
        cache.print_debug();

        // 找检测输出和掩码原型
        let proto_idx = cache.find_4d();
        let det_idx = match proto_idx {
            Some(p) => cache.find_3d(Some(p)),
            None => cache.find_largest(None),
        };

        let Some(det_idx) = det_idx else {
            tracing::error!("ERROR: No valid detection output found");
            return Ok(Vec::new());
        };

        if proto_idx.is_none() {
            tracing::error!("ERROR: No valid mask prototype output (4D) found");
            return Ok(Vec::new());
        }

        // 解析检测输出
        let detections = match self.parse_detection_output(&cache.outputs[det_idx])? {
            Some(d) => d,
            None => return Ok(Vec::new()),
        };

        // 解析掩码原型（proto_idx 已确认存在）
        let proto = self.parse_proto_output(cache, proto_idx)?;

        self.postprocess_traditional_from_parsed(
            &detections,
            proto.as_ref(),
            lb,
            num_classes,
            orig_width,
            orig_height,
        )
    }

    /// 传统格式的处理逻辑（需要 NMS）。
    fn postprocess_traditional_from_parsed(
        &self,
        detections: &[Vec<f32>],
        proto: Option<&ProtoMasks>,
        lb: LetterboxParams,
        num_classes: i32,
        orig_width: i32,
        orig_height: i32,
    ) -> Result<Vec<Segmentation>> {
        // 解析掩码原型（null 时返回空结果）
        let Some(proto) = proto else {
            return Ok(Vec::new());
        };

        // 模型类型已在 predict() 中自动检测
        let num_classes = num_classes.max(0) as usize;

        // detections 形状: [channels, anchors] 其中 channels = 4 + numClasses + maskProtoDim
        let channels = detections.len();
        if channels == 0 {
            return Ok(Vec::new());
        }
        let num_anchors = detections[0].len();

        tracing::debug!(
            "Detections: [{} x {}], numClasses={}, maskProtoDim={}",
            channels,
            num_anchors,
            num_classes,
            Self::MASK_PROTO_DIM
        );

        // 打印前几个 anchor 样本
        for i in 0..num_anchors.min(3) {
            let bbox: String = (0..4)
                .map(|b| detections.get(b).and_then(|r| r.get(i)).map(|v| format!("{:.3} ", v)).unwrap_or_default())
                .collect();
            let scores: String = (0..num_classes.min(3))
                .map(|c| detections.get(4 + c).and_then(|r| r.get(i)).map(|v| format!("{:.3} ", v)).unwrap_or_default())
                .collect();
            tracing::debug!("  anchor[{}]: bbox={} scores={}", i, bbox, scores);
        }

        let coeff_start = 4 + num_classes;

        // 收集候选框
        let mut candidates: Vec<Candidate> = Vec::new();

        for i in 0..num_anchors {
            let mut max_score = 0f32;
            let mut best_class = -1i32;

            for c in 0..num_classes {
                // 防御性越界保护（合法模型 channels = 4 + nc + 32 恒满足）
                if 4 + c >= channels {
                    break;
                }
                let score = detections[4 + c][i];
                if score > max_score {
                    max_score = score;
                    best_class = c as i32;
                }
            }

            if max_score < self.base.confidence_threshold() {
                continue;
            }

            let cx = detections[0][i];
            let cy = detections[1][i];
            let bw = detections[2][i];
            let bh = detections[3][i];

            // xywh -> xyxy 并还原坐标
            let x1 = (cx - bw / 2.0 - lb.dw) / lb.ratio;
            let y1 = (cy - bh / 2.0 - lb.dh) / lb.ratio;
            let x2 = (cx + bw / 2.0 - lb.dw) / lb.ratio;
            let y2 = (cy + bh / 2.0 - lb.dh) / lb.ratio;

            // 提取 mask 系数
            let mut mask_coeffs = vec![0f32; Self::MASK_PROTO_DIM];
            for (j, coeff) in mask_coeffs.iter_mut().enumerate() {
                if coeff_start + j < channels {
                    *coeff = detections[coeff_start + j][i];
                }
            }

            candidates.push(Candidate {
                x1,
                y1,
                x2,
                y2,
                conf: max_score,
                class_id: best_class,
                anchor_idx: i,
                coeffs: mask_coeffs,
            });
        }

        // NMS 并生成掩码
        self.nms_and_create_masks(candidates, proto, lb, orig_width, orig_height)
    }

    // ==================== 输出解析辅助 ====================

    /// 解析检测输出为二维数组 `[dim1][dim2]`（对应 `parseDetectionOutput`）。
    ///
    /// `Ok(None)` 表示维度不符合预期（返回空结果）。
    fn parse_detection_output(&self, output: &TensorOutput) -> Result<Option<Vec<Vec<f32>>>> {
        let det_flat = output.as_f32()?;
        tracing::debug!(
            "Detection output: shape={:?}, elements={}",
            output.shape,
            det_flat.len()
        );

        match output.shape.len() {
            3 => {
                // [1, channels, anchors] → 取 batch 0（对应 getFloatArray3D(...)[0]）
                let rows = output.shape[1].max(0) as usize;
                let cols = output.shape[2].max(0) as usize;
                if det_flat.len() < rows * cols {
                    return Err(VisionError::inference(format!(
                        "Detection output element count {} < {}x{}",
                        det_flat.len(),
                        rows,
                        cols
                    )));
                }
                Ok(Some(
                    (0..rows)
                        .map(|r| det_flat[r * cols..(r + 1) * cols].to_vec())
                        .collect(),
                ))
            }
            2 => {
                let rows = output.shape[0].max(0) as usize;
                let cols = output.shape[1].max(0) as usize;
                if det_flat.len() < rows * cols {
                    return Err(VisionError::inference(format!(
                        "Detection output element count {} < {}x{}",
                        det_flat.len(),
                        rows,
                        cols
                    )));
                }
                Ok(Some(
                    (0..rows)
                        .map(|r| det_flat[r * cols..(r + 1) * cols].to_vec())
                        .collect(),
                ))
            }
            1 => Ok(Some(self.reshape_1d_output(det_flat)?)),
            n => {
                tracing::error!("ERROR: Unexpected detection output dims: {n}");
                Ok(None)
            }
        }
    }

    /// 解析掩码原型输出（对应 `parseProtoOutput`），
    /// 返回 `[dim][h*w]` 线性布局（对应 `protoArrayToMat` 的 Mat 布局）。
    fn parse_proto_output(
        &self,
        cache: &OutputCache<'_>,
        proto_idx: Option<usize>,
    ) -> Result<Option<ProtoMasks>> {
        let Some(idx) = proto_idx else {
            tracing::error!("ERROR: No proto output");
            return Ok(None);
        };

        let proto = &cache.outputs[idx];
        let shape = &proto.shape;
        let flat = proto.as_f32()?;
        tracing::debug!("Proto output: shape={:?}", shape);

        let (dim, h, w) = match shape.len() {
            // [1, 32, H, W] → 取 batch 0（对应 getFloatArray4D(...)[0]）
            4 => (
                shape[1].max(0) as usize,
                shape[2].max(0) as usize,
                shape[3].max(0) as usize,
            ),
            3 => (
                shape[0].max(0) as usize,
                shape[1].max(0) as usize,
                shape[2].max(0) as usize,
            ),
            n => {
                tracing::error!("ERROR: Unexpected proto output dims: {n}");
                return Ok(None);
            }
        };

        let Some(need) = dim.checked_mul(h).and_then(|v| v.checked_mul(w)) else {
            return Err(VisionError::inference("proto dims overflow"));
        };
        if flat.len() < need {
            return Err(VisionError::inference(format!(
                "Proto element count {} < {}x{}x{}",
                flat.len(),
                dim,
                h,
                w
            )));
        }

        Ok(Some(ProtoMasks {
            dim,
            height: h,
            width: w,
            data: flat[..need].to_vec(),
        }))
    }

    // ==================== NMS 与掩码 ====================

    /// NMS 并创建掩码（对应 `nmsAndCreateMasks`）。
    fn nms_and_create_masks(
        &self,
        mut candidates: Vec<Candidate>,
        proto: &ProtoMasks,
        lb: LetterboxParams,
        orig_width: i32,
        orig_height: i32,
    ) -> Result<Vec<Segmentation>> {
        let mut results: Vec<Segmentation> = Vec::new();
        if candidates.is_empty() {
            return Ok(results);
        }

        // 按置信度降序（稳定排序）
        candidates.sort_by(|a, b| b.conf.total_cmp(&a.conf));
        let mut suppressed = vec![false; candidates.len()];

        for i in 0..candidates.len() {
            if suppressed[i] {
                continue;
            }

            let curr = &candidates[i];
            let conf = curr.conf;
            let cls_id = curr.class_id;

            // 边界裁剪
            let x1 = curr.x1.max(0.0).min(orig_width as f32);
            let y1 = curr.y1.max(0.0).min(orig_height as f32);
            let x2 = curr.x2.max(0.0).min(orig_width as f32);
            let y2 = curr.y2.max(0.0).min(orig_height as f32);

            // 生成掩码
            let mask = self.decode_mask(
                &curr.coeffs, proto, lb, orig_width, orig_height, x1, y1, x2, y2,
            )?;

            results.push(Segmentation::new(
                self.base.get_label_name(cls_id),
                cls_id,
                x1 as f64,
                y1 as f64,
                x2 as f64,
                y2 as f64,
                conf as f64,
                Some(mask),
            ));

            // 抑制同类重叠框
            for j in (i + 1)..candidates.len() {
                if suppressed[j] {
                    continue;
                }
                if candidates[j].class_id != cls_id {
                    continue;
                }
                if Self::compute_iou(&curr.box_array(), &candidates[j].box_array())
                    > self.nms_threshold as f64
                {
                    suppressed[j] = true;
                }
            }
        }

        Ok(results)
    }

    /// 解码掩码（参考 YOLO26SegmentationEngine 修复版；对应 `decodeMask`，
    /// 上游实现为 protected，此处因参数为内部类型而收窄为私有）。
    ///
    /// # 参数
    /// - `coeffs`: mask 系数 [32]
    /// - `proto`: 掩码原型 `[32, h*w]`（对应 CV_32F Mat）
    /// - `lb`: letterbox 参数（ratio/dw/dh）
    /// - `orig_width`/`orig_height`: 原始图像宽高
    /// - `x1,y1,x2,y2`: 边界框坐标（已还原到原图）
    #[allow(clippy::too_many_arguments)]
    fn decode_mask(
        &self,
        coeffs: &[f32],
        proto: &ProtoMasks,
        lb: LetterboxParams,
        orig_width: i32,
        orig_height: i32,
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

        // 2. 矩阵乘法: coeffs [1,32] @ proto [32, h*w] -> [1, h*w]（对应 gemm）
        let mut linear = vec![0f32; hw];
        for d in 0..proto.dim {
            let c = coeffs[d];
            let row = &proto.data[d * hw..(d + 1) * hw];
            for (j, v) in row.iter().enumerate() {
                linear[j] += c * v;
            }
        }

        // 3. Reshape 为 h×w（上游硬编码 reshape(1, 160)；标准 proto 为 160x160，等价）
        // 4. Sigmoid
        for v in linear.iter_mut() {
            *v = sigmoid(*v);
        }
        let mask_small = FloatMask::from_raw(proto.width, proto.height, linear)?;

        // 5. Resize 到 inputSize x inputSize
        let mask_full = mask_small.resize(
            self.base.input_width() as usize,
            self.base.input_height() as usize,
        );

        // 6. 裁剪 Letterbox 填充区域
        let input_width = self.base.input_width();
        let input_height = self.base.input_height();
        let valid_w = (input_width as f32 - 2.0 * lb.dw).round() as i32;
        let valid_h = (input_height as f32 - 2.0 * lb.dh).round() as i32;

        let mask_valid: FloatMask = if valid_w > 0 && valid_h > 0 {
            let pad_left = (lb.dw.round() as i32).clamp(0, input_width - 1);
            let pad_top = (lb.dh.round() as i32).clamp(0, input_height - 1);
            let vw = (valid_w as usize).min(input_width as usize - pad_left as usize);
            let vh = (valid_h as usize).min(input_height as usize - pad_top as usize);
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

        // 7. Resize 到原图尺寸
        let mask_orig = mask_valid.resize(orig_width as usize, orig_height as usize);

        // 8. 裁剪到 BBox 区域，保留浮点概率值
        Ok(Self::crop_mask_to_bbox(
            mask_orig, orig_width, orig_height, x1, y1, x2, y2,
        ))
    }

    /// 裁剪到 BBox 区域：bbox 外为 0，bbox 内保留浮点概率值
    /// （`decode_mask` / `decode_rfdetr_mask` 共用逻辑）。
    fn crop_mask_to_bbox(
        mask_orig: FloatMask,
        orig_width: i32,
        orig_height: i32,
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
    ) -> FloatMask {
        let bx1 = (x1.round() as i32).max(0);
        let by1 = (y1.round() as i32).max(0);
        let bx2 = (x2.round() as i32).min(orig_width);
        let by2 = (y2.round() as i32).min(orig_height);

        let mut mask_final = FloatMask::new(orig_width as usize, orig_height as usize);

        if bx2 > bx1 && by2 > by1 {
            // 边界检查：bbox 在图像内
            if bx1 >= 0
                && by1 >= 0
                && bx2 <= mask_orig.width() as i32
                && by2 <= mask_orig.height() as i32
            {
                for y in by1..by2 {
                    for x in bx1..bx2 {
                        mask_final.set(x as usize, y as usize, mask_orig.get(x as usize, y as usize));
                    }
                }
            }
        }

        mask_final
    }

    // ==================== 辅助方法 ====================

    /// 将 1D flat 检测输出重塑为 `[channels, anchors]`（对应 `reshape1DOutput`）。
    fn reshape_1d_output(&self, flat: &[f32]) -> Result<Vec<Vec<f32>>> {
        let total_elements = flat.len();
        let num_classes = self.base.labels().map(|l| l.len()).unwrap_or(0);

        if num_classes > 0 {
            let channels = 4 + num_classes + Self::MASK_PROTO_DIM;
            if total_elements % channels == 0 {
                let num_anchors = total_elements / channels;
                tracing::info!(
                    "Reshaping 1D seg output [{}] to [{}, {}]",
                    total_elements,
                    channels,
                    num_anchors
                );
                let mut result = vec![vec![0f32; num_anchors]; channels];
                let mut idx = 0;
                for c in 0..channels {
                    for a in 0..num_anchors {
                        result[c][a] = flat[idx];
                        idx += 1;
                    }
                }
                return Ok(result);
            }
        }

        Err(VisionError::inference(format!(
            "Cannot reshape 1D segmentation output of {} elements. Model labels: {}",
            total_elements,
            self.base.labels().map(|l| l.len().to_string()).unwrap_or_else(|| "none".to_string())
        )))
    }

    /// 计算两个框的 IoU（对应 `computeIoU`，坐标格式 `[x1,y1,x2,y2]`，
    /// 委托 [`BoundingBox::iou`]）。
    pub fn compute_iou(box_a: &[f32; 4], box_b: &[f32; 4]) -> f64 {
        BoundingBox::new(
            box_a[0] as f64,
            box_a[1] as f64,
            box_a[2] as f64,
            box_a[3] as f64,
            0.0,
        )
        .iou(&BoundingBox::new(
            box_b[0] as f64,
            box_b[1] as f64,
            box_b[2] as f64,
            box_b[3] as f64,
            0.0,
        ))
    }

    /// 打印分割结果日志（对应 `logResults`）。
    fn log_results(&self, results: &[Segmentation]) {
        tracing::info!(
            "Segmented {} objects{}",
            results.len(),
            if results.is_empty() { "" } else { ":" }
        );
        for s in results {
            tracing::info!("  {}", s);
        }
    }
}

/// sigmoid（对应 `sigmoid`，double 精度计算后截断为 f32）。
fn sigmoid(x: f32) -> f32 {
    (1.0 / (1.0 + (-(x as f64)).exp())) as f32
}

/// 将 `[1, Q, D]` 或 `[Q, D]` 张量读为 `[Q][D]` 二维数组
/// （对应 `readRank3As2D`）。
fn read_rank3_as_2d(tensor: &TensorOutput) -> Result<Vec<Vec<f32>>> {
    let flat = tensor.as_f32()?;
    let (rows, cols) = match tensor.shape.len() {
        3 => (
            tensor.shape[1].max(0) as usize,
            tensor.shape[2].max(0) as usize,
        ),
        2 => (
            tensor.shape[0].max(0) as usize,
            tensor.shape[1].max(0) as usize,
        ),
        n => {
            return Err(VisionError::inference(format!(
                "Expected rank-2/3 tensor, got rank-{n}, shape: {:?}",
                tensor.shape
            )));
        }
    };
    if flat.len() < rows * cols {
        return Err(VisionError::inference(format!(
            "Tensor element count {} < {}x{}",
            flat.len(),
            rows,
            cols
        )));
    }
    Ok((0..rows)
        .map(|r| flat[r * cols..(r + 1) * cols].to_vec())
        .collect())
}

crate::impl_engine_forward!(SegmentationEngine, base, Vec<Segmentation>,
    fn predict(&self, image: &Image) -> Result<Vec<Segmentation>> { self.predict_impl(image) }
);
