//! MobileSAM 交互式分割引擎。
//!
//! 使用合并的 ONNX 模型（encoder + decoder 合一），支持点/框/全图分割。
//!
//! # 模型输入
//!
//! - `image`: [1, 3, 1024, 1024] float32 — 预处理后的图像 (resize+pad+normalize)
//! - `point_coords`: [1, N, 2] float32 — 点坐标（1024 空间像素坐标）
//! - `point_labels`: [1, N] int64 — 1=前景, 0=背景, -1=padding, 2=box左上角, 3=box右下角
//! - `orig_im_size`: [2] float32 — 原始图像 [H, W]
//!
//! # 模型输出
//!
//! - `masks`: [1, 1, 1024, 1024] float — mask logits（正值=前景，负值=背景）
//! - `scores`: [1, 1] float — IoU 置信度
//! - `prepadded`: [1, 2] float — 裁剪尺寸 [crop_h, crop_w]，用于后处理 crop+resize 到原图
//!
//! # 注意事项
//!
//! - point_labels 必须以 int64 张量传入（上游实现需显式 `LongPointer`，Rust 侧为 `Tensor<i64>`）
//! - 单点分割时需手动添加 padding point (0,0, label=-1)，ONNX 模型未内置 SAM 的 pad 逻辑
//! - 后处理（crop/threshold/resize）在宿主侧完成，ONNX 模型仅输出 1024x1024 logits

use std::sync::Mutex;

use ort::value::Tensor;

use crate::core::base::BaseOnnxEngine;
use crate::core::device_type::DeviceType;
use crate::error::{Result, VisionError};
use crate::imaging::{cvt_color, resize, ColorConversion, FloatMask, Image, Interpolation};
use crate::model::Segmentation;

/// MobileSAM 交互式分割引擎（对应 `SamEngine`）。
///
/// # 示例
///
/// ```ignore
/// let eng = SamEngine::new("mobile_sam_fused.onnx", DeviceType::Cpu)?;
/// let segs = eng.predict_point(&image, x, y, 1)?;       // 单点前景分割
/// let segs = eng.predict_box(&image, x1, y1, x2, y2)?;  // 框分割
/// let segs = eng.predict_everything(&image, 80)?;       // 全图自动分割
/// ```
pub struct SamEngine {
    /// 组合基类
    pub base: BaseOnnxEngine,

    /// 上次推理输出缓存（对应上游 字段 lastMaskData / lastScoreData / lastMaskShape /
    /// lastPrepadded；上游实现是 IoBinding "先取出再关闭" 的临时存储，Rust 侧直接
    /// 保存推理输出的快照）。`&self` 不可变语义下用 `Mutex` 提供内部可变性。
    last_output: Mutex<Option<SamOutputCache>>,
}

/// 上次推理的输出快照（对应 last* 实例字段）。
#[derive(Debug, Clone)]
struct SamOutputCache {
    /// masks [1, 1, 1024, 1024] float logits
    mask_data: Vec<f32>,
    /// masks 形状
    #[allow(dead_code)]
    mask_shape: Vec<i64>,
    /// scores [1, 1] — IoU 置信度
    score_data: Vec<f32>,
    /// prepadded [1, 2] → [crop_h, crop_w]
    prepadded: [f32; 2],
}

/// SAM 预处理结果（对应上游内部类 `PreprocessResult`）。
struct PreprocessResult {
    pixel_data: Vec<f32>,
    orig_h: i32,
    orig_w: i32,
}

impl SamEngine {
    /// SAM 输入边长。
    pub const SAM_SIZE: i32 = 1024;

    const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

    /// 创建引擎（对应上游 构造器；输入固定 1024x1024）。
    pub fn new(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        let mut base = BaseOnnxEngine::with_input_size(
            model_path,
            device_type,
            Self::SAM_SIZE,
            Self::SAM_SIZE,
        )?;
        base.set_normalization(Self::IMAGENET_MEAN, Self::IMAGENET_STD);

        tracing::info!(
            "SAM Engine initialized: {} input nodes, {} output nodes",
            // 上游记录 numInputNodes；基类以首输入名 + 输出数呈现
            base.input_name(),
            base.output_names().len()
        );

        Ok(SamEngine {
            base,
            last_output: Mutex::new(None),
        })
    }

    // ==================== 公开 API ====================

    /// 单点分割。
    ///
    /// - `image`: 输入图像 (BGR)
    /// - `x` / `y`: 点的坐标（原图像素坐标）
    /// - `label`: 1=前景, 0=背景
    pub fn predict_point(&self, image: &Image, x: f32, y: f32, label: i64) -> Result<Vec<Segmentation>> {
        let pre = self.preprocess_sam(image)?;
        let scale = Self::SAM_SIZE as f32 / pre.orig_h.max(pre.orig_w) as f32;

        // SAM 的 PromptEncoder 在没有 box 时会添加一个 padding point (label=-1)
        // ONNX 模型没有内置这个逻辑，所以需要手动添加
        let point_coords = [x * scale, y * scale, 0.0f32, 0.0f32];
        let point_labels = [label, -1i64];
        let orig_size = [pre.orig_h as f32, pre.orig_w as f32];

        tracing::info!(
            "[SAM] predictPoint: x={}, y={}, label={}, scale={}, coords=[{},{},{},{}], labels=[{},{}], origSize=[{},{}]",
            x, y, label, scale,
            point_coords[0], point_coords[1], point_coords[2], point_coords[3],
            point_labels[0], point_labels[1], orig_size[0], orig_size[1]
        );

        self.run_sam_inference(&pre.pixel_data, &point_coords, &point_labels, &orig_size, 2)?;
        self.parse_sam_output(pre.orig_h, pre.orig_w)
    }

    /// 多点分割。
    ///
    /// - `points`: 点坐标，[x0, y0, x1, y1, ...]（原图像素坐标）
    /// - `labels`: 标签列表，1=前景, 0=背景
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
        let pre = self.preprocess_sam(image)?;
        let scale = Self::SAM_SIZE as f32 / pre.orig_h.max(pre.orig_w) as f32;

        // 多点时也需要添加 padding point（与 SAM PromptEncoder 行为一致）
        let n = labels.len();
        let mut point_coords = vec![0f32; (n + 1) * 2];
        let mut point_labels = vec![0i64; n + 1];
        for i in 0..n {
            point_coords[i * 2] = points[i * 2] * scale;
            point_coords[i * 2 + 1] = points[i * 2 + 1] * scale;
            point_labels[i] = labels[i];
        }
        // padding point
        point_coords[n * 2] = 0.0;
        point_coords[n * 2 + 1] = 0.0;
        point_labels[n] = -1;

        let orig_size = [pre.orig_h as f32, pre.orig_w as f32];

        self.run_sam_inference(&pre.pixel_data, &point_coords, &point_labels, &orig_size, n + 1)?;
        self.parse_sam_output(pre.orig_h, pre.orig_w)
    }

    /// 框分割（使用框的两个角点作为提示）。
    ///
    /// label=2 对应 SAM 的 top-left box corner, label=3 对应 bottom-right box corner。
    pub fn predict_box(
        &self,
        image: &Image,
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
    ) -> Result<Vec<Segmentation>> {
        let pre = self.preprocess_sam(image)?;
        let scale = Self::SAM_SIZE as f32 / pre.orig_h.max(pre.orig_w) as f32;

        let point_coords = [x1 * scale, y1 * scale, x2 * scale, y2 * scale];
        let point_labels = [2i64, 3i64];
        let orig_size = [pre.orig_h as f32, pre.orig_w as f32];

        self.run_sam_inference(&pre.pixel_data, &point_coords, &point_labels, &orig_size, 2)?;
        self.parse_sam_output(pre.orig_h, pre.orig_w)
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
        let pre = self.preprocess_sam(image)?;
        let scale = Self::SAM_SIZE as f32 / pre.orig_h.max(pre.orig_w) as f32;
        let orig_size = [pre.orig_h as f32, pre.orig_w as f32];

        let mut all_results: Vec<Segmentation> = Vec::new();
        let mut y = grid_step;
        while y < pre.orig_h {
            let mut x = grid_step;
            while x < pre.orig_w {
                // 添加 padding point
                let point_coords = [x as f32 * scale, y as f32 * scale, 0.0f32, 0.0f32];
                let point_labels = [1i64, -1i64];

                self.run_sam_inference(&pre.pixel_data, &point_coords, &point_labels, &orig_size, 2)?;
                let segs = self.parse_sam_output(pre.orig_h, pre.orig_w)?;

                for seg in segs {
                    if seg.mask.is_some() && seg.mask_area() > 200.0 && seg.confidence() > 0.5 {
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

    // ==================== 预处理 ====================

    /// SAM 预处理: resize longest side to 1024 → pad to 1024x1024 → ImageNet normalize → NCHW
    fn preprocess_sam(&self, image: &Image) -> Result<PreprocessResult> {
        let orig_h = image.height() as i32;
        let orig_w = image.width() as i32;

        let scale = Self::SAM_SIZE as f32 / orig_h.max(orig_w) as f32;
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

        // Pad to 1024x1024 (copy resized into top-left corner of black canvas)
        let size = Self::SAM_SIZE as usize;
        let mut padded = Image::new(size, size, 3);
        padded.paste(0, 0, &rgb);

        // HWC → CHW with normalization
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
        })
    }

    // ==================== 推理 ====================

    /// 组装 4 个输入并运行 SAM 推理，把输出快照存入 [`Self::last_output`]
    /// （对应 `runSamInference` 的 IoBinding + "先取出再关闭" 缓存模式）。
    fn run_sam_inference(
        &self,
        image_data: &[f32],
        point_coords: &[f32],
        point_labels: &[i64],
        orig_size: &[f32],
        num_points: usize,
    ) -> Result<()> {
        let num_points = num_points as i64;

        // image [1, 3, 1024, 1024]
        let image_tensor = Tensor::from_array((
            vec![1, 3, Self::SAM_SIZE, Self::SAM_SIZE],
            image_data.to_vec(),
        ))?;

        // point_coords [1, N, 2]
        let coords_tensor = Tensor::from_array((vec![1, num_points, 2], point_coords.to_vec()))?;

        // point_labels [1, N] int64 — 显式使用 int64 张量（对应上游 LongPointer 约定）
        let labels_tensor = Tensor::from_array((vec![1, num_points], point_labels.to_vec()))?;

        // orig_im_size [2]
        let size_tensor = Tensor::from_array((vec![2i64], orig_size.to_vec()))?;

        let mut session = self.base.session.lock().unwrap();
        let outputs = session.run(ort::inputs![
            "image" => image_tensor,
            "point_coords" => coords_tensor,
            "point_labels" => labels_tensor,
            "orig_im_size" => size_tensor,
        ])?;

        let names = self.base.output_names();
        if names.len() < 3 {
            return Err(VisionError::inference(format!(
                "Expected 3 outputs, got {}",
                names.len()
            )));
        }

        // output[0]: masks [1, 1, 1024, 1024] float logits
        let (mask_shape, mask_data) = snapshot_f32(&outputs, &names[0])?;

        // output[1]: scores [1, 1]
        let (_, score_data) = snapshot_f32(&outputs, &names[1])?;

        // output[2]: prepadded [1, 2] → [crop_h, crop_w]
        let (_, pp_data) = snapshot_f32(&outputs, &names[2])?;
        if pp_data.len() < 2 {
            return Err(VisionError::inference(format!(
                "output '{}' must have 2 elements, got {}",
                names[2],
                pp_data.len()
            )));
        }
        let prepadded = [pp_data[0], pp_data[1]];

        tracing::info!(
            "[SAM] inference result: maskShape=[{},{},{},{}], score={}, prepadded=[{},{}]",
            mask_shape.first().copied().unwrap_or(-1),
            mask_shape.get(1).copied().unwrap_or(-1),
            mask_shape.get(2).copied().unwrap_or(-1),
            mask_shape.get(3).copied().unwrap_or(-1),
            score_data.first().copied().unwrap_or(f32::NAN),
            prepadded[0],
            prepadded[1]
        );

        *self.last_output.lock().unwrap() = Some(SamOutputCache {
            mask_data,
            mask_shape,
            score_data,
            prepadded,
        });
        Ok(())
    }

    // ==================== 后处理 ====================

    /// 解析最近一次推理输出，构造 [`Segmentation`] 列表（对应 `parseSamOutput`）。
    ///
    /// mask 来自 ONNX: [1, 1, 1024, 1024] logits；
    /// 后处理：crop → threshold → resize to original。
    fn parse_sam_output(&self, orig_h: i32, orig_w: i32) -> Result<Vec<Segmentation>> {
        let mut results = Vec::new();

        let cache = self.last_output.lock().unwrap();
        let Some(cache) = cache.as_ref() else {
            return Ok(results);
        };

        let score = cache.score_data.first().copied().unwrap_or(0.0);

        let size = Self::SAM_SIZE as usize;
        if cache.mask_data.len() < size * size {
            return Err(VisionError::inference(format!(
                "mask element count {} < {}x{}",
                cache.mask_data.len(),
                size,
                size
            )));
        }

        // crop 区域 [crop_h, crop_w]（整数强转向零截断）
        let crop_h = cache.prepadded[0] as usize;
        let crop_w = cache.prepadded[1] as usize;
        if crop_h == 0 || crop_w == 0 || crop_h > size || crop_w > size {
            return Err(VisionError::image(format!(
                "invalid prepadded crop size {}x{}",
                crop_w, crop_h
            )));
        }

        // 取 [0:cropH, 0:cropW] 区域
        let mut cropped_data = Vec::with_capacity(crop_h * crop_w);
        for y in 0..crop_h {
            let row = &cache.mask_data[y * size..y * size + crop_w];
            cropped_data.extend_from_slice(row);
        }
        let cropped = FloatMask::from_raw(crop_w, crop_h, cropped_data)?;

        // threshold: logit > 0.0 → mask（0/1 浮点掩码，对应 CV_32F THRESH_BINARY maxval=1.0）
        let mut binary = FloatMask::new(crop_w, crop_h);
        for (dst, &v) in binary.data_mut().iter_mut().zip(cropped.data()) {
            *dst = if v > 0.0 { 1.0 } else { 0.0 };
        }

        // resize to original image size（双线性，对应 INTER_LINEAR 默认插值）
        let resized_mask = binary.resize(orig_w as usize, orig_h as usize);

        // 原版还由 bbox 计算中心点 cx/cy；Rust Segmentation 模型无对应字段，省略
        if let Some(bbox) = extract_bbox(&resized_mask) {
            results.push(Segmentation::new(
                "object",
                0,
                bbox[0] as f64,
                bbox[1] as f64,
                bbox[2] as f64,
                bbox[3] as f64,
                score as f64,
                Some(resized_mask),
            ));
        }

        Ok(results)
    }
}

/// 全分辨率 mask 的前景包围盒（对应 `extractBBox`：逐像素扫描 > 0.5）。
///
/// 返回 `[minX, minY, maxX, maxY]`；无前景返回 `None`。
fn extract_bbox(mask: &FloatMask) -> Option<[f32; 4]> {
    let (w, h) = (mask.width(), mask.height());
    let mut min_x = w as i32;
    let mut min_y = h as i32;
    let mut max_x = 0i32;
    let mut max_y = 0i32;
    let mut found = false;

    for y in 0..h {
        for x in 0..w {
            if mask.get(x, y) > 0.5 {
                found = true;
                if (x as i32) < min_x {
                    min_x = x as i32;
                }
                if (x as i32) > max_x {
                    max_x = x as i32;
                }
                if (y as i32) < min_y {
                    min_y = y as i32;
                }
                if (y as i32) > max_y {
                    max_y = y as i32;
                }
            }
        }
    }

    if !found {
        None
    } else {
        Some([min_x as f32, min_y as f32, max_x as f32, max_y as f32])
    }
}

/// 从会话输出中按名取 f32 张量快照（对齐 base.rs `snapshot_output` 的语义）。
fn snapshot_f32(outputs: &ort::session::SessionOutputs<'_>, name: &str) -> Result<(Vec<i64>, Vec<f32>)> {
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

crate::impl_engine_forward!(SamEngine, base, Vec<Segmentation>,
    /// 单图推理。SAM 为交互式分割引擎，不支持整图 `predict`
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
