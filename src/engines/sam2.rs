//! SAM2 交互式分割引擎（SAM2.0 / SAM2.1，onnx-community 导出格式）。
//!
//! 使用单文件合并 ONNX（vision_encoder + prompt_encoder + mask_decoder 合一）。
//! 该 schema 是 onnx-community 在 Hugging Face 发布的官方 ONNX 导出格式。
//!
//! # 模型准备
//!
//! ```text
//! # 下载并合并 SAM2.1 ONNX（用 scripts/sam2_merge_onnx.py）：
//! python sam2_merge_onnx.py --repo onnx-community/sam2.1-hiera-tiny-ONNX \
//!     --out ./sam2_fused.onnx
//! # 产出：sam2_fused.onnx（单文件，~148 MB for tiny）
//! ```
//!
//! # 模型 I/O（4 输入 + 3 输出）
//!
//! ## Inputs
//!
//! - `pixel_values`:     [1, 3, 1024, 1024] float32 — 预处理后图像
//! - `dec_input_points`: [1, 1, N, 2]       float32 — 1024 空间像素坐标
//! - `dec_input_labels`: [1, 1, N]          int64   — 1=FG, 0=BG, -1=pad
//! - `dec_input_boxes`:  [1, M, 4]          float32 — M=0 表示无 box（需传 [1, 0, 4]）
//!
//! ## Outputs
//!
//! - `dec_iou_scores`:          [1, 1, 3]          float32 — 3 个候选 mask 的 IoU 分数
//! - `dec_pred_masks`:          [1, 1, 3, 256, 256] float32 — 3 个 256x256 mask logits
//! - `dec_object_score_logits`: [1, 1, 1]          float32 — 对象存在分数（一般忽略）
//!
//! # 关键行为
//!
//! - 模型**固定输出 3 个候选 mask**（whole / part / subpart），按 IoU 分数排序；
//!   与 SAM1 的 "multimask_output" 行为不同——本引擎没有该参数，3 mask 始终返回。
//! - mask 输出尺寸 256x256（不是 SAM1 的 1024x1024），按面积缩放回原图分辨率。
//! - box 与 point 是**独立输入**，不再是 SAM1 的"用 label=2/3 把 box 伪装成点"约定。
//! - 单点 / 多点分割时，input_points 末位需追加 padding point (0,0, label=-1)，
//!   与 SAM1 约定一致。
//!
//! # 示例
//!
//! ```ignore
//! let engine = Sam2Engine::new("sam2_fused.onnx", DeviceType::Cpu)?;
//!
//! // 单点前景分割（返回 3 mask，按 IoU 排序）
//! let segs = engine.predict_point(&image, image.width() as f32 / 2.0, image.height() as f32 / 2.0, 1)?;
//!
//! // 多点（前景+背景）
//! let segs2 = engine.predict_points(&image, &[x1, y1, x2, y2], &[1, 0])?;
//!
//! // 框分割
//! let segs3 = engine.predict_box(&image, 50.0, 50.0, 200.0, 300.0)?;
//! ```

use std::sync::Mutex;

use ort::value::Tensor;

use crate::core::base::BaseOnnxEngine;
use crate::core::device_type::DeviceType;
use crate::error::{Result, VisionError};
use crate::imaging::{cvt_color, resize, ColorConversion, FloatMask, Image, Interpolation};
use crate::model::Segmentation;

/// SAM2 交互式分割引擎（对应 `Sam2Engine`）。
pub struct Sam2Engine {
    /// 组合基类
    pub base: BaseOnnxEngine,

    /// 模型输入边长（默认 1024）
    input_size: i32,

    /// 上次推理输出缓存（对应上游 字段 lastMaskData / lastMaskShape / lastIouData /
    /// lastIouShape；"先取出再关闭" 模式的 Rust 快照等价）。
    last_output: Mutex<Option<Sam2OutputCache>>,
}

/// 上次推理的输出快照（对应 last* 实例字段）。
#[derive(Debug, Clone)]
struct Sam2OutputCache {
    /// dec_pred_masks [1, 1, 3, 256, 256] float logits
    mask_data: Vec<f32>,
    mask_shape: Vec<i64>,
    /// dec_iou_scores [1, 1, 3]
    iou_data: Vec<f32>,
    /// 形状（解析时未使用，保留字段）
    #[allow(dead_code)]
    iou_shape: Vec<i64>,
}

/// SAM2 预处理结果（对应上游内部类 `PreprocessResult`）。
struct PreprocessResult {
    pixel_data: Vec<f32>,
    orig_h: i32,
    orig_w: i32,
    /// resize 到 inputSize 后（补边到 inputSize×inputSize 之前）的高度
    padded_h: i32,
    /// resize 到 inputSize 后的宽度
    padded_w: i32,
}

impl Sam2Engine {
    /// ImageNet 归一化均值。
    const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    /// ImageNet 归一化标准差。
    const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
    /// 默认输入边长。
    pub const DEFAULT_INPUT_SIZE: i32 = 1024;
    /// 低分辨率 mask 边长（由模型输出 shape 读取，常量仅供外部参考）。
    pub const LOW_RES_MASK_SIZE: i32 = 256;

    // model I/O names (must match the merged ONNX)
    const IN_PIXEL: &'static str = "pixel_values";
    const IN_POINTS: &'static str = "dec_input_points";
    const IN_LABELS: &'static str = "dec_input_labels";
    const IN_BOXES: &'static str = "dec_input_boxes";
    const OUT_IOU: &'static str = "dec_iou_scores";
    const OUT_MASKS: &'static str = "dec_pred_masks";

    /// 创建引擎（默认输入 1024；对应上游 单参构造器）。
    pub fn new(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        Self::with_input_size(model_path, device_type, Self::DEFAULT_INPUT_SIZE)
    }

    /// 指定输入边长创建（对应上游 双参构造器）。
    pub fn with_input_size(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        input_size: i32,
    ) -> Result<Self> {
        let mut base =
            BaseOnnxEngine::with_input_size(model_path, device_type, input_size, input_size)?;
        base.set_normalization(Self::IMAGENET_MEAN, Self::IMAGENET_STD);
        tracing::info!(
            "SAM2 Engine initialized: input={}x{}, model inputs='{}', outputs={}",
            input_size,
            input_size,
            base.input_name(),
            base.output_names().len()
        );
        Ok(Sam2Engine {
            base,
            input_size,
            last_output: Mutex::new(None),
        })
    }

    // 注：原版覆写 createSessionOptions 使用 CudaProviderMode.V2；
    // Rust ort 统一走 CUDA V2 API（见 core::session_factory::CudaProviderMode 注释），
    // 基类默认创建路径语义一致，无需覆写。

    // ==================== Public API ====================

    /// 单点前景/背景分割。固定返回 3 个候选 mask（按 IoU 降序）。
    ///
    /// - `image`: 输入图像 (BGR)
    /// - `x` / `y`: 点坐标（原图像素坐标）
    /// - `label`: 1=前景, 0=背景
    pub fn predict_point(&self, image: &Image, x: f32, y: f32, label: i64) -> Result<Vec<Segmentation>> {
        let pre = self.preprocess_sam2(image)?;
        let scale = self.input_size as f32 / pre.orig_h.max(pre.orig_w) as f32;

        // input_points: [1, 1, 2, 2] — (click_x, click_y) + padding (0, 0)
        let point_coords = [x * scale, y * scale, 0.0f32, 0.0f32];
        // input_labels: [1, 1, 2] — click label + pad
        let point_labels = [label, -1i64];

        self.run_decoder(&pre.pixel_data, &point_coords, &point_labels, pre.orig_h, pre.orig_w)?;
        self.parse_sam2_output(pre.orig_h, pre.orig_w, pre.padded_h, pre.padded_w)
    }

    /// 多点分割。
    ///
    /// - `points`: [x0, y0, x1, y1, ...] 原图像素坐标
    /// - `labels`: 长度 = points.len() / 2；1=FG, 0=BG
    pub fn predict_points(
        &self,
        image: &Image,
        points: &[f32],
        labels: &[i64],
    ) -> Result<Vec<Segmentation>> {
        if points.len() != labels.len() * 2 {
            return Err(VisionError::invalid_argument(
                "points length must be labels.length * 2",
            ));
        }
        let pre = self.preprocess_sam2(image)?;
        let scale = self.input_size as f32 / pre.orig_h.max(pre.orig_w) as f32;

        let n = labels.len();
        // input_points: [1, 1, n+1, 2]
        let mut point_coords = vec![0f32; (n + 1) * 2];
        let mut point_labels = vec![0i64; n + 1];
        for i in 0..n {
            point_coords[i * 2] = points[i * 2] * scale;
            point_coords[i * 2 + 1] = points[i * 2 + 1] * scale;
            point_labels[i] = labels[i];
        }
        point_coords[n * 2] = 0.0;
        point_coords[n * 2 + 1] = 0.0;
        point_labels[n] = -1;

        self.run_decoder(&pre.pixel_data, &point_coords, &point_labels, pre.orig_h, pre.orig_w)?;
        self.parse_sam2_output(pre.orig_h, pre.orig_w, pre.padded_h, pre.padded_w)
    }

    /// 框分割（box 走独立 input_boxes 输入，不再伪装成点）。
    pub fn predict_box(
        &self,
        image: &Image,
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
    ) -> Result<Vec<Segmentation>> {
        let pre = self.preprocess_sam2(image)?;
        let scale = self.input_size as f32 / pre.orig_h.max(pre.orig_w) as f32;

        // 单 box: [1, 1, 4]
        let box_data = [x1 * scale, y1 * scale, x2 * scale, y2 * scale];
        // 单 padding point
        let point_coords = [0.0f32, 0.0f32];
        let point_labels = [-1i64];

        self.run_decoder_with_box(
            &pre.pixel_data,
            &point_coords,
            &point_labels,
            Some(&box_data),
            pre.orig_h,
            pre.orig_w,
        )?;
        self.parse_sam2_output(pre.orig_h, pre.orig_w, pre.padded_h, pre.padded_w)
    }

    /// 全图自动分割（网格点采样）。
    ///
    /// - `grid_step`: 网格步长（像素），建议 80
    pub fn predict_everything(&self, image: &Image, grid_step: i32) -> Result<Vec<Segmentation>> {
        // 上游实现 gridStep<=0 会死循环，此处显式拒绝
        if grid_step < 1 {
            return Err(VisionError::invalid_argument(
                "gridStep must be a positive integer",
            ));
        }
        let pre = self.preprocess_sam2(image)?;
        let scale = self.input_size as f32 / pre.orig_h.max(pre.orig_w) as f32;

        let mut all_results: Vec<Segmentation> = Vec::new();
        let mut y = grid_step;
        while y < pre.orig_h {
            let mut x = grid_step;
            while x < pre.orig_w {
                let point_coords = [x as f32 * scale, y as f32 * scale, 0.0f32, 0.0f32];
                let point_labels = [1i64, -1i64];

                self.run_decoder(
                    &pre.pixel_data,
                    &point_coords,
                    &point_labels,
                    pre.orig_h,
                    pre.orig_w,
                )?;
                let segs =
                    self.parse_sam2_output(pre.orig_h, pre.orig_w, pre.padded_h, pre.padded_w)?;

                for seg in segs {
                    if seg.mask.is_some() && seg.mask_area() > 200.0 && seg.confidence() > 0.3 {
                        all_results.push(seg);
                    }
                    // else：上游调用 releaseMask() 释放；Rust Drop 自动回收
                }
                x += grid_step;
            }
            y += grid_step;
        }

        if all_results.len() > 30 {
            all_results.sort_by(|a, b| {
                b.confidence()
                    .partial_cmp(&a.confidence())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            all_results.truncate(30);
        }
        Ok(all_results)
    }

    // ==================== Preprocessing ====================

    /// SAM2 预处理：resize longest side to inputSize → pad to inputSize×inputSize →
    /// BGR→RGB → ImageNet normalize → NCHW
    fn preprocess_sam2(&self, image: &Image) -> Result<PreprocessResult> {
        let orig_h = image.height() as i32;
        let orig_w = image.width() as i32;

        let scale = self.input_size as f32 / orig_h.max(orig_w) as f32;
        let new_h = (orig_h as f32 * scale).round() as usize;
        let new_w = (orig_w as f32 * scale).round() as usize;

        // 输入统一到 3 通道 BGR（实现假定输入为 CV_8UC3）
        let bgr = match image.channels() {
            3 => image.clone(),
            4 => cvt_color(image, ColorConversion::Bgra2Bgr)?,
            _ => cvt_color(image, ColorConversion::Gray2Bgr)?,
        };

        let resized = resize(&bgr, new_w, new_h, Interpolation::Linear)?;

        let rgb = cvt_color(&resized, ColorConversion::Bgr2Rgb)?;

        // Pad to inputSize×inputSize（右下角补黑边）
        let size = self.input_size as usize;
        let mut padded = Image::new(size, size, 3);
        padded.paste(0, 0, &rgb);

        // HWC → CHW with ImageNet normalize
        let area = size * size;
        let mut data = vec![0f32; 3 * area];

        let px = padded.data();
        for i in 0..area {
            let r = px[i * 3] as f32 / 255.0;
            let g = px[i * 3 + 1] as f32 / 255.0;
            let b = px[i * 3 + 2] as f32 / 255.0;
            data[i] = (r - Self::IMAGENET_MEAN[0]) / Self::IMAGENET_STD[0];
            data[area + i] = (g - Self::IMAGENET_MEAN[1]) / Self::IMAGENET_STD[1];
            data[2 * area + i] = (b - Self::IMAGENET_MEAN[2]) / Self::IMAGENET_STD[2];
        }

        Ok(PreprocessResult {
            pixel_data: data,
            orig_h,
            orig_w,
            padded_h: new_h as i32,
            padded_w: new_w as i32,
        })
    }

    // ==================== Inference ====================

    /// 点提示推理入口。box 走 [1, 0, 4]（空 box）— ONNX 强制要求这个输入存在
    /// （对应上游 4 参 `runDecoder` 重载）。
    #[allow(clippy::too_many_arguments)]
    fn run_decoder(
        &self,
        image_data: &[f32],
        point_coords: &[f32],
        point_labels: &[i64],
        orig_h: i32,
        orig_w: i32,
    ) -> Result<()> {
        self.run_decoder_with_box(image_data, point_coords, point_labels, None, orig_h, orig_w)
    }

    /// 框提示推理入口。point 走 [1, 1, 1, 2] = padding point
    /// （对应上游 6 参 `runDecoder` 重载）。
    #[allow(clippy::too_many_arguments)]
    fn run_decoder_with_box(
        &self,
        image_data: &[f32],
        point_coords: &[f32],
        point_labels: &[i64],
        box_data: Option<&[f32]>,
        _orig_h: i32,
        _orig_w: i32,
    ) -> Result<()> {
        // 1) pixel_values: [1, 3, 1024, 1024]
        let image_tensor = Tensor::from_array((
            vec![1, 3, self.input_size as i64, self.input_size as i64],
            image_data.to_vec(),
        ))?;

        // 2) dec_input_points: [1, 1, N, 2]
        let num_points = point_labels.len() as i64;
        let points_tensor = Tensor::from_array((
            vec![1, 1, num_points, 2],
            point_coords.to_vec(),
        ))?;

        // 3) dec_input_labels: [1, 1, N] int64
        let labels_tensor =
            Tensor::from_array((vec![1, 1, num_points], point_labels.to_vec()))?;

        // 4) dec_input_boxes: [1, M, 4]
        let (boxes_data, boxes_shape): (Vec<f32>, Vec<i64>) = match box_data {
            Some(b) if !b.is_empty() => {
                let num_boxes = (b.len() / 4) as i64;
                (b.to_vec(), vec![1, num_boxes, 4])
            }
            _ => (Vec::new(), vec![1, 0, 4]),
        };
        let boxes_tensor = Tensor::from_array((boxes_shape, boxes_data))?;

        let mut session = self.base.session.lock().unwrap();
        let outputs = session.run(ort::inputs![
            Self::IN_PIXEL  => image_tensor,
            Self::IN_POINTS => points_tensor,
            Self::IN_LABELS => labels_tensor,
            Self::IN_BOXES  => boxes_tensor,
        ])?;

        // outputs 按 outputNames 顺序返回；按名缓存（对应 名称匹配循环）
        let mut mask_out: Option<(Vec<i64>, Vec<f32>)> = None;
        let mut iou_out: Option<(Vec<i64>, Vec<f32>)> = None;
        for name in self.base.output_names() {
            if name == Self::OUT_MASKS {
                mask_out = Some(snapshot_f32(&outputs, name)?);
            } else if name == Self::OUT_IOU {
                iou_out = Some(snapshot_f32(&outputs, name)?);
            }
        }

        match (mask_out, iou_out) {
            (Some((mask_shape, mask_data)), Some((iou_shape, iou_data))) => {
                *self.last_output.lock().unwrap() = Some(Sam2OutputCache {
                    mask_data,
                    mask_shape,
                    iou_data,
                    iou_shape,
                });
            }
            _ => {
                return Err(VisionError::inference(format!(
                    "Expected outputs {} and {} not found. Got: {:?}",
                    Self::OUT_MASKS,
                    Self::OUT_IOU,
                    self.base.output_names()
                )));
            }
        }
        Ok(())
    }

    // ==================== Postprocessing ====================

    /// 解析 decoder 输出，构造 [`Segmentation`] 列表（对应 `parseSam2Output`）。
    ///
    /// - pred_masks shape: [1, 1, 3, 256, 256] — 3 个 256x256 mask logits
    /// - iou_scores shape:  [1, 1, 3]          — 3 个 mask 的 IoU 分数
    /// - 按 IoU 降序返回所有 3 个 Segmentation。
    ///
    /// 关键：模型在 1024×1024（含 padding）空间预测 mask。必须先把 256→1024、再
    /// crop 到 padded 区域 (paddedW × paddedH)、最后 resize 到 origW × origH，
    /// 否则非方形图的 mask 会偏移 padding 距离。
    fn parse_sam2_output(
        &self,
        orig_h: i32,
        orig_w: i32,
        padded_h: i32,
        padded_w: i32,
    ) -> Result<Vec<Segmentation>> {
        let mut results = Vec::new();
        let cache = self.last_output.lock().unwrap();
        let Some(cache) = cache.as_ref() else {
            return Ok(results);
        };

        // mask shape: [1, 1, 3, 256, 256]
        if cache.mask_shape.len() < 5 {
            return Err(VisionError::inference(format!(
                "dec_pred_masks must be rank-5, got shape {:?}",
                cache.mask_shape
            )));
        }
        let num_masks = cache.mask_shape[2] as usize;
        let mask_h = cache.mask_shape[3] as usize;
        let mask_w = cache.mask_shape[4] as usize;
        let mask_area = mask_h * mask_w;

        // 按 IoU 降序排序 mask 索引（稳定排序）
        let mut order: Vec<usize> = (0..num_masks).collect();
        order.sort_by(|&a, &b| {
            let fa = cache.iou_data.get(a).copied().unwrap_or(f32::NEG_INFINITY);
            let fb = cache.iou_data.get(b).copied().unwrap_or(f32::NEG_INFINITY);
            // 按 b 在前降序
            fb.partial_cmp(&fa).unwrap_or(std::cmp::Ordering::Equal)
        });

        for idx in order {
            let iou = cache.iou_data.get(idx).copied().unwrap_or(0.0);

            // 取出第 idx 个 mask 的 2D 区域 [maskH, maskW]
            if (idx + 1) * mask_area > cache.mask_data.len() {
                return Err(VisionError::inference(format!(
                    "mask data length {} < mask #{} area {}",
                    cache.mask_data.len(),
                    idx,
                    mask_area
                )));
            }
            let mask_flat = &cache.mask_data[idx * mask_area..(idx + 1) * mask_area];
            let mask_mat = FloatMask::from_raw(mask_w, mask_h, mask_flat.to_vec())?;

            // logit > 0 = foreground (与 SAM1/SamEngine 行为一致)
            let mut binary = FloatMask::new(mask_w, mask_h);
            for (dst, &v) in binary.data_mut().iter_mut().zip(mask_mat.data()) {
                *dst = if v > 0.0 { 1.0 } else { 0.0 };
            }

            // Step 1: resize 256x256 → 1024x1024 (回到模型预测的 input 空间)
            let mask_1024 =
                binary.resize(self.input_size as usize, self.input_size as usize);

            // Step 2: crop 到 padded 区域 (paddedW × paddedH, 即真实图像在 1024 中的区域)
            let pw = padded_w as usize;
            let ph = padded_h as usize;
            if ph == 0 || pw == 0 || ph > self.input_size as usize || pw > self.input_size as usize {
                return Err(VisionError::image(format!(
                    "invalid padded region {}x{}",
                    pw, ph
                )));
            }
            let mut cropped_data = Vec::with_capacity(ph * pw);
            for y in 0..ph {
                cropped_data.extend_from_slice(&mask_1024.data()[y * self.input_size as usize..y * self.input_size as usize + pw]);
            }
            let cropped = FloatMask::from_raw(pw, ph, cropped_data)?;

            // Step 3: resize 回原图
            let resized_mask = cropped.resize(orig_w as usize, orig_h as usize);

            // 原版还由 bbox 计算中心点 cx/cy；Rust Segmentation 模型无对应字段，省略
            if let Some(bbox) = Self::extract_bbox(&resized_mask) {
                results.push(Segmentation::new(
                    "object",
                    0,
                    bbox[0] as f64,
                    bbox[1] as f64,
                    bbox[2] as f64,
                    bbox[3] as f64,
                    iou as f64,
                    Some(resized_mask),
                ));
            }
        }

        Ok(results)
    }

    /// 全分辨率 mask 的前景包围盒（对应 `extractBBox`）。
    ///
    /// 原实现为逐像素扫描（原图 ~5MP × 3 mask）；这里先按 0.5 阈值化
    /// （resize 插值会产生 0~1 中间值，必须先二值化），再取前景外接矩形。
    /// 语义与旧实现一致（>0.5 的像素，含边界）。
    /// 返回 `[x, y, x+width-1, y+height-1]`；无前景返回 `None`。
    fn extract_bbox(mask: &FloatMask) -> Option<[f32; 4]> {
        let rect = mask.foreground_rect(0.5);
        if rect.width == 0 || rect.height == 0 {
            return None;
        }
        Some([
            rect.x as f32,
            rect.y as f32,
            (rect.x + rect.width - 1) as f32,
            (rect.y + rect.height - 1) as f32,
        ])
    }

    // 注：原版 close() 由基类统一释放会话；Rust 侧 ort Session 为 RAII（Drop 自动释放），
    // 无需显式 close。
}

/// 从会话输出中按名取 f32 张量快照（对齐 base.rs `snapshot_output` 的语义）。
fn snapshot_f32(
    outputs: &ort::session::SessionOutputs<'_>,
    name: &str,
) -> Result<(Vec<i64>, Vec<f32>)> {
    let value = outputs
        .get(name)
        .ok_or_else(|| VisionError::inference(format!("missing output '{name}'")))?;
    let shape = match value.dtype() {
        ort::value::ValueType::Tensor { shape, .. } => shape.iter().copied().collect::<Vec<i64>>(),
        other => {
            return Err(VisionError::inference(format!(
                "output '{name}' is not a tensor: {other:?}"
            )))
        }
    };
    let (_, view) = value.try_extract_tensor::<f32>()?;
    Ok((shape, view.to_vec()))
}

crate::impl_engine_forward!(Sam2Engine, base, Vec<Segmentation>,
    /// 单图推理。SAM2 为交互式分割引擎，不支持整图 `predict`
    /// （对应上游 抛出 `UnsupportedOperationException`）。
    fn predict(&self, _image: &Image) -> Result<Vec<Segmentation>> {
        Err(VisionError::Unsupported(
            "Use predict_point/predict_points/predict_box/predict_everything instead".to_string(),
        ))
    },
    /// 批量推理同理不支持（对应 `predictBatch` 抛出异常）。
    fn predict_batch(&self, _images: &[Image]) -> Result<Vec<Vec<Segmentation>>> {
        Err(VisionError::Unsupported(
            "Use predict_point/predict_points/predict_box/predict_everything instead".to_string(),
        ))
    }
);
