//! 姿态估计推理引擎（YOLOv8n-pose / YOLO11n-pose，Ultralytics ONNX 导出）。
//!
//! **支持的模型与输出布局**（首次推理时按输出 shape 自动识别，互不影响）：
//!
//! | 布局 | 输出 shape | 解码 |
//! |---|---|---|
//! | 传统（YOLOv8/v11-pose 默认导出） | `[1, 4+nc+3*nk, anchors]`（COCO 时 `[1, 56, 8400]`） | 逐 anchor 取最大类分 → 置信度过滤 → 类内 NMS → 关键点还原 |
//! | End2End（NMS-Free，如 YOLO26-pose） | `[1, M, 6+3*nk]`（COCO 时 `[1, 300, 57]`） | 按行解析 `[x1,y1,x2,y2,conf,cls,kpts...]`，无需 NMS |
//!
//! 其中 `nk` 为关键点数（COCO 17 点，动态推导，不硬编码）；`nc` 优先取模型元数据
//! 标签数、回退 1（YOLO-pose 默认单类 person），对应 `channels = 4 + nc + 3*nk` 的动态拆分。
//!
//! **布局自动判定**（同源 [`crate::engines::detection`] 的 auto_detect 思路）：
//! `dim1∈[100,400] && dim2∈[4,100]` 判为 End2End —— End2End 的 dim1 是最大检测数 M（默认 300）、
//! dim2 是 `6+3*nk`；传统布局 dim1 是 channels（COCO 为 56），dim2 是 anchors（8400，远大于 100）。
//! 传统布局 channels 随类别数动态变化（大 nc 时可能落入 [100,400]），此时 dim2=anchors 仍会
//! 排除误判，因此两个条件需同时满足。
//!
//! **关键点解码**：坐标为 letterbox 输入分辨率像素，`kpt = (kpt_input - dw/dh) / ratio` 还原到
//! 原图（与 Ultralytics 官方一致不做边界裁剪）；置信度规格为 sigmoid(kpt_conf)，但官方
//! Ultralytics 导出已在计算图内过 sigmoid（`decode_kpts` 的 `.sigmoid()`），二次 sigmoid 会把
//! 分数压向 0.5 —— 因此与 [`crate::engines::detection`] Transformer 分数的自适应策略一致：
//! 仅当置信度区间出现负值或 >1（即原始 logits）时才补 sigmoid，否则原值使用。
//!
//! **默认阈值**：置信度 0.25、关键点分数过滤 0.5（可配，低于阈值的关键点仍保留原值，
//! 见 [`PoseResult`] 约定）、NMS IoU 0.45。
//!
//! 注意：SAHI 切片推理暂不支持（行人关键点跨切片拼接无意义），开启 SAHI 配置时仍走整图路径。

use std::sync::Mutex;

use crate::core::base::{BaseOnnxEngine, TensorOutput};
use crate::core::device_type::DeviceType;
use crate::error::{Result, VisionError};
use crate::imaging::{cvt_color, resize, ColorConversion, Image, Interpolation};
use crate::model::{BoundingBox, Detection, Keypoint, PoseResult};

/// letterbox 预处理的坐标还原参数（对应 detection 引擎的 `LetterboxParams` 模式：
/// Rust `&self` 不可变，参数随调用链显式传递而非写在实例字段上）。
#[derive(Debug, Clone, Copy, Default)]
struct LetterboxParams {
    /// 缩放比例（min(input_w/orig_w, input_h/orig_h)）
    ratio: f32,
    /// 左右对称填充的一半宽度
    dw: f32,
    /// 上下对称填充的一半高度
    dh: f32,
}

/// 行主序二维 float 矩阵（传统布局 [channels, anchors] 的只读视图）。
#[derive(Debug, Clone)]
struct Matrix2D {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Matrix2D {
    /// 从 flat 数据按 shape 解释为 [rows][cols]（长度不足时报错）。
    fn from_flat(flat: &[f32], rows: usize, cols: usize) -> Result<Self> {
        if flat.len() < rows * cols {
            return Err(VisionError::inference(format!(
                "Tensor element count {} < {}x{}",
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

/// 传统布局的候选实例（NMS 中间结构；坐标为 letterbox 输入空间的 xyxy）。
#[derive(Debug, Clone, Copy)]
struct PoseCandidate {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    /// 最大类别分数
    score: f32,
    class_id: i32,
    /// 该候选对应的 anchor 列索引（用于 NMS 后回读关键点通道）
    anchor: usize,
}

/// 姿态估计推理引擎。
pub struct PoseEngine {
    /// 组合基类
    pub base: BaseOnnxEngine,

    /// NMS IoU 阈值（默认 0.45）
    nms_threshold: f32,
    /// 关键点分数过滤阈值（默认 0.5；低于阈值的关键点仍保留原值，供下游自行过滤）
    keypoint_threshold: f32,
    /// 是否为 end2end 模型（内置 NMS）；None 前由自动检测决定
    end2_end: bool,
    /// 自动检测结果缓存（首次推理时检测；对应 detection 引擎的 `Mutex<Option<bool>>` 模式）
    auto_detected_end2_end: Mutex<Option<bool>>,
}

impl PoseEngine {
    /// 创建姿态估计引擎（默认阈值：conf 0.25 / kpt 0.5 / nms 0.45）。
    pub fn new(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        Self::new_with_input_size(model_path, device_type, -1, -1)
    }

    /// 创建姿态估计引擎（指定输入尺寸；<=0 时从模型读取）。
    pub fn new_with_input_size(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        input_height: i32,
        input_width: i32,
    ) -> Result<Self> {
        let base =
            BaseOnnxEngine::with_input_size(model_path, device_type, input_height, input_width)?;
        Ok(Self::build(base))
    }

    /// 创建姿态估计引擎（显式指定置信度与 NMS 阈值）。
    pub fn for_yolo(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        confidence_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Self> {
        Self::for_yolo_with_input_size(
            model_path,
            device_type,
            confidence_threshold,
            nms_threshold,
            -1,
            -1,
        )
    }

    /// 创建姿态估计引擎（显式阈值 + 指定输入尺寸）。
    pub fn for_yolo_with_input_size(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        confidence_threshold: f32,
        nms_threshold: f32,
        input_height: i32,
        input_width: i32,
    ) -> Result<Self> {
        let mut engine =
            Self::new_with_input_size(model_path, device_type, input_height, input_width)?;
        engine.base.set_confidence_threshold(confidence_threshold);
        engine.nms_threshold = nms_threshold;
        Ok(engine)
    }

    /// 创建 End2End（NMS-Free）姿态估计引擎，跳过输出布局自动检测。
    pub fn for_yolo_end2_end(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        confidence_threshold: f32,
    ) -> Result<Self> {
        let mut engine = Self::new(model_path, device_type)?;
        engine.base.set_confidence_threshold(confidence_threshold);
        engine.end2_end = true;
        *engine.auto_detected_end2_end.lock().unwrap() = Some(true);
        Ok(engine)
    }

    /// 内部构造（统一默认阈值：conf 0.25 / kpt 0.5 / nms 0.45）。
    fn build(base: BaseOnnxEngine) -> Self {
        let mut engine = PoseEngine {
            base,
            nms_threshold: 0.45,
            keypoint_threshold: 0.5,
            end2_end: false,
            auto_detected_end2_end: Mutex::new(None),
        };
        engine.base.set_confidence_threshold(0.25);
        engine
    }

    // ============ 访问器 ============

    /// NMS IoU 阈值。
    pub fn nms_threshold(&self) -> f32 {
        self.nms_threshold
    }

    /// 设置 NMS IoU 阈值。
    pub fn set_nms_threshold(&mut self, threshold: f32) {
        self.nms_threshold = threshold;
    }

    /// 关键点分数过滤阈值。
    pub fn keypoint_threshold(&self) -> f32 {
        self.keypoint_threshold
    }

    /// 设置关键点分数过滤阈值（低于阈值的关键点仍保留原值，见 [`PoseResult`] 约定）。
    pub fn set_keypoint_threshold(&mut self, threshold: f32) {
        self.keypoint_threshold = threshold;
    }

    /// 是否为 end2end 模型（内置 NMS；自动检测已缓存时优先返回检测结果）。
    pub fn is_end2_end(&self) -> bool {
        self.auto_detected_end2_end
            .lock()
            .unwrap()
            .unwrap_or(self.end2_end)
    }

    // ============ 推理路径 ============

    /// 单图推理核心实现（trait `predict` 转发到这里）。
    pub fn predict_impl(&self, image: &Image) -> Result<Vec<PoseResult>> {
        self.predict_without_sahi(image)
    }

    /// 原始单图推理路径（预留与 detection 引擎一致的 SAHI 扩展点；当前 SAHI 不适用姿态任务）。
    pub fn predict_without_sahi(&self, image: &Image) -> Result<Vec<PoseResult>> {
        if image.is_empty() {
            return Err(VisionError::image("cannot run pose on empty image"));
        }

        // 1. 记录原始尺寸
        let orig_width = image.width();
        let orig_height = image.height();

        // 2. 预处理（letterbox + /255 + RGB + HWC→CHW，同时记录坐标还原参数）
        let (input_data, lb) = self.preprocess_with_letterbox(image)?;

        // 3. 创建 Tensor 并推理
        let input_tensor = self.base.create_input_tensor(input_data)?;
        let all_outputs = self.base.run_multi_output(input_tensor)?;

        // 4. 定位输出张量（跳过 shape=[1] 等元数据输出）并自动识别布局
        let output_tensor = Self::find_pose_output(&all_outputs)?;
        let end2end = self.resolve_end2_end(output_tensor)?;

        // 5. 后处理
        let results = if end2end {
            self.postprocess_end2_end(output_tensor, orig_width, orig_height, lb)?
        } else {
            self.postprocess_standard(output_tensor, orig_width, orig_height, lb)?
        };

        self.log_results(&results);
        Ok(results)
    }

    /// 批量推理核心实现（trait `predict_batch` 转发到这里）。
    ///
    /// 动态 batch 模型走真实批量路径（逐图携带各自的 letterbox 参数）；
    /// 输出 batch 维与输入不符时回退为逐张完整推理。
    pub fn predict_batch_impl(&self, images: &[Image]) -> Result<Vec<Vec<PoseResult>>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        if images.len() == 1 {
            return Ok(vec![self.predict_impl(&images[0])?]);
        }

        let batch_size = images.len();

        // 1. 批量预处理（逐图记录还原参数）
        let (batch_data, params) = self.preprocess_batch_with_letterbox(images)?;

        // 2. 创建批量 Tensor 并推理
        let input_tensor = self.base.create_batch_input_tensor(batch_data, batch_size)?;
        let all_outputs = self.base.run_multi_output(input_tensor)?;
        let output_tensor = Self::find_pose_output(&all_outputs)?;
        let end2end = self.resolve_end2_end(output_tensor)?;

        let flat = output_tensor.as_f32()?;
        let shape = &output_tensor.shape;
        let mut all_results = Vec::with_capacity(batch_size);

        if end2end {
            // [batch, M, 6+3*nk]
            if shape.len() == 3 && shape[0] == batch_size as i64 {
                let rows = shape[1] as usize;
                let cols = shape[2] as usize;
                let stride = rows * cols;
                for b in 0..batch_size {
                    all_results.push(self.decode_end2_end_rows(
                        &flat[b * stride..(b + 1) * stride],
                        rows,
                        cols,
                        images[b].width(),
                        images[b].height(),
                        params[b],
                    )?);
                }
                return Ok(all_results);
            }
        } else {
            // [batch, channels, anchors]
            if shape.len() == 3 && shape[0] == batch_size as i64 {
                let channels = shape[1] as usize;
                let anchors = shape[2] as usize;
                let stride = channels * anchors;
                let (nc, nk) = self.split_channels(channels)?;
                for b in 0..batch_size {
                    let m = Matrix2D::from_flat(&flat[b * stride..(b + 1) * stride], channels, anchors)?;
                    all_results.push(self.decode_standard(
                        &m,
                        nc,
                        nk,
                        images[b].width(),
                        images[b].height(),
                        params[b],
                    )?);
                }
                return Ok(all_results);
            }
        }

        // 输出 batch 维与输入不一致（如固定 batch=1 导出）：回退逐张完整推理
        tracing::debug!(
            "Batched output shape {:?} does not match batch={}, falling back to per-image inference",
            shape,
            batch_size
        );
        for image in images {
            all_results.push(self.predict_impl(image)?);
        }
        Ok(all_results)
    }

    // ==================== 预处理 ====================

    /// 带 Letterbox 的预处理（保持宽高比，114 灰边填充）+ BGR→RGB + /255 + HWC→CHW。
    ///
    /// 返回 `(CHW 浮点数据, 坐标还原参数)`；参数随返回值显式传递（对齐 detection 引擎模式）。
    fn preprocess_with_letterbox(&self, image: &Image) -> Result<(Vec<f32>, LetterboxParams)> {
        if image.is_empty() {
            return Err(VisionError::image("cannot letterbox empty image"));
        }
        let orig_width = image.width();
        let orig_height = image.height();
        let input_width = self.base.input_width() as usize;
        let input_height = self.base.input_height() as usize;

        // 1. 计算缩放比例
        let ratio = (input_width as f32 / orig_width as f32)
            .min(input_height as f32 / orig_height as f32);
        let new_width = (orig_width as f32 * ratio).round() as usize;
        let new_height = (orig_height as f32 * ratio).round() as usize;

        // 2. 计算填充
        let dw = (input_width as f32 - new_width as f32) / 2.0;
        let dh = (input_height as f32 - new_height as f32) / 2.0;

        // 输入统一到 3 通道 BGR
        let bgr = match image.channels() {
            3 => image.clone(),
            4 => cvt_color(image, ColorConversion::Bgra2Bgr)?,
            _ => cvt_color(image, ColorConversion::Gray2Bgr)?,
        };

        // 3. Resize
        let resized = resize(&bgr, new_width, new_height, Interpolation::Linear)?;

        // 4. 画布填充（114 灰边）
        let mut padded = Image::filled(input_width, input_height, 3, 114);
        let top = ((dh - 0.1).round() as i32).max(0) as usize;
        let left = ((dw - 0.1).round() as i32).max(0) as usize;
        padded.paste(left, top, &resized);

        // 5. BGR -> RGB
        let rgb = cvt_color(&padded, ColorConversion::Bgr2Rgb)?;

        // 6. /255 归一化 + HWC → CHW
        let px = rgb.data();
        let area = input_height * input_width;
        let mut float_data = vec![0f32; 3 * area];
        for i in 0..area {
            float_data[i] = px[i * 3] as f32 / 255.0;
            float_data[i + area] = px[i * 3 + 1] as f32 / 255.0;
            float_data[i + 2 * area] = px[i * 3 + 2] as f32 / 255.0;
        }

        Ok((float_data, LetterboxParams { ratio, dw, dh }))
    }

    /// 批量预处理（带 Letterbox），返回拼接的 CHW 数据与每张图的还原参数。
    fn preprocess_batch_with_letterbox(
        &self,
        images: &[Image],
    ) -> Result<(Vec<f32>, Vec<LetterboxParams>)> {
        let mut batch_data = Vec::new();
        let mut params = Vec::with_capacity(images.len());
        for image in images {
            let (data, lb) = self.preprocess_with_letterbox(image)?;
            params.push(lb);
            batch_data.extend_from_slice(&data);
        }
        Ok((batch_data, params))
    }

    // ==================== 传统布局后处理 ====================

    /// 传统布局后处理：`[channels, anchors]`（channels = 4 + nc + 3*nk）。
    ///
    /// 逐 anchor 取最大类分 → 置信度过滤 → 类内 NMS → 关键点解码还原。
    fn postprocess_standard(
        &self,
        output_tensor: &TensorOutput,
        orig_width: usize,
        orig_height: usize,
        lb: LetterboxParams,
    ) -> Result<Vec<PoseResult>> {
        let flat = output_tensor.as_f32()?;
        let shape = &output_tensor.shape;

        tracing::debug!("Pose output tensor shape: {:?}, total elements: {}", shape, flat.len());

        // 解释为 [channels, anchors]；支持 3D [1,C,A] / 2D [C,A] / 1D flat
        let (channels, anchors) = match shape.len() {
            3 => (shape[1] as usize, shape[2] as usize),
            2 => (shape[0] as usize, shape[1] as usize),
            1 => self.reshape_1d_dims(flat.len())?,
            _ => {
                return Err(VisionError::inference(format!(
                    "Unsupported pose output tensor dimensions: {}, shape: {:?}",
                    shape.len(),
                    shape
                )))
            }
        };
        if channels == 0 || anchors == 0 {
            return Err(VisionError::inference(format!(
                "Pose output has zero channels/anchors, shape: {:?}",
                shape
            )));
        }

        let m = Matrix2D::from_flat(flat, channels, anchors)?;
        let (nc, nk) = self.split_channels(channels)?;
        self.decode_standard(&m, nc, nk, orig_width, orig_height, lb)
    }

    /// 传统布局解码主体（单图 / 批量切片共用）。
    fn decode_standard(
        &self,
        m: &Matrix2D,
        nc: usize,
        nk: usize,
        orig_width: usize,
        orig_height: usize,
        lb: LetterboxParams,
    ) -> Result<Vec<PoseResult>> {
        let anchors = m.cols;
        let kpt_base = 4 + nc; // 关键点通道起始行：前 4 行是 box cx,cy,w,h，随后 nc 行类别分

        tracing::debug!(
            "decode_standard: [rows={}, anchors={}], nc={}, nk={}",
            m.rows,
            anchors,
            nc,
            nk
        );

        // 关键点置信度是否为原始 logits（决定是否补 sigmoid，见模块文档）
        let need_sigmoid = kpt_conf_need_sigmoid_standard(m, kpt_base, nk, anchors);

        // 1. 逐 anchor 生成候选（坐标保持在输入空间，NMS 后再还原）
        let mut candidates: Vec<PoseCandidate> = Vec::new();
        for a in 0..anchors {
            // 取最大类别分数
            let mut max_score = 0f32;
            let mut best_class = -1i32;
            for c in 0..nc {
                let score = m.get(4 + c, a);
                if score > max_score {
                    max_score = score;
                    best_class = c as i32;
                }
            }

            if max_score < self.base.confidence_threshold() {
                continue;
            }

            // box：输入空间 xywh → xyxy
            let cx = m.get(0, a);
            let cy = m.get(1, a);
            let w = m.get(2, a);
            let h = m.get(3, a);

            candidates.push(PoseCandidate {
                x1: cx - w / 2.0,
                y1: cy - h / 2.0,
                x2: cx + w / 2.0,
                y2: cy + h / 2.0,
                score: max_score,
                class_id: best_class,
                anchor: a,
            });
        }

        // 2. 类内 NMS（IoU 复用 `BoundingBox::iou`，在输入空间比较）
        let kept = self.nms_select(candidates);

        // 3. 保留项还原坐标 + 解码关键点
        let mut results = Vec::with_capacity(kept.len());
        for cand in kept {
            let x1 = ((cand.x1 - lb.dw) / lb.ratio).max(0.0).min(orig_width as f32);
            let y1 = ((cand.y1 - lb.dh) / lb.ratio).max(0.0).min(orig_height as f32);
            let x2 = ((cand.x2 - lb.dw) / lb.ratio).max(0.0).min(orig_width as f32);
            let y2 = ((cand.y2 - lb.dh) / lb.ratio).max(0.0).min(orig_height as f32);

            let detection = Detection::new(
                self.base.get_label_name(cand.class_id),
                cand.class_id,
                x1 as f64,
                y1 as f64,
                x2 as f64,
                y2 as f64,
                cand.score as f64,
            );

            // 关键点解码：输入空间像素 → 原图像素（与 Ultralytics 官方一致不裁剪，
            // 越界值交由下游处理）；置信度 kpt_score = sigmoid(kpt_conf)（自适应，见模块文档）
            let mut keypoints = Vec::with_capacity(nk);
            for k in 0..nk {
                let col = kpt_base + k * 3;
                let kx = (m.get(col, cand.anchor) - lb.dw) / lb.ratio;
                let ky = (m.get(col + 1, cand.anchor) - lb.dh) / lb.ratio;
                let mut score = m.get(col + 2, cand.anchor);
                if need_sigmoid {
                    score = sigmoid(score);
                }
                keypoints.push(Keypoint::new(kx, ky, score));
            }

            results.push(PoseResult::new(detection, keypoints));
        }

        Ok(results)
    }

    // ==================== End2End 布局后处理 ====================

    /// End2End（NMS-Free）后处理：`[M, 6+3*nk]`，每行 `[x1,y1,x2,y2,conf,cls,kpts...]`。
    fn postprocess_end2_end(
        &self,
        output_tensor: &TensorOutput,
        orig_width: usize,
        orig_height: usize,
        lb: LetterboxParams,
    ) -> Result<Vec<PoseResult>> {
        let flat = output_tensor.as_f32()?;
        let shape = &output_tensor.shape;

        // [M, 6+3*nk]
        let (rows, cols) = match shape.len() {
            3 => (shape[1] as usize, shape[2] as usize),
            2 => (shape[0] as usize, shape[1] as usize),
            _ => {
                return Err(VisionError::inference(format!(
                    "End2End pose model expects 2D/3D output, got {}D, shape: {:?}",
                    shape.len(),
                    shape
                )))
            }
        };

        self.decode_end2_end_rows(flat, rows, cols, orig_width, orig_height, lb)
    }

    /// End2End 行解析主体（单图 / 批量切片共用）：按行取框 + 类别 + 关键点，无需 NMS。
    fn decode_end2_end_rows(
        &self,
        flat: &[f32],
        rows: usize,
        cols: usize,
        orig_width: usize,
        orig_height: usize,
        lb: LetterboxParams,
    ) -> Result<Vec<PoseResult>> {
        if rows * cols > flat.len() {
            return Err(VisionError::inference(format!(
                "End2End pose tensor element count {} < {}x{}",
                flat.len(),
                rows,
                cols
            )));
        }
        // 行宽 = 4(box) + 1(conf) + 1(cls) + 3*nk
        let rem = cols as isize - 6;
        if rem < 3 || rem % 3 != 0 {
            return Err(VisionError::inference(format!(
                "End2End pose row width {} cannot be split into 6 + 3*nk",
                cols
            )));
        }
        let nk = (rem / 3) as usize;

        // 关键点置信度是否为原始 logits（决定是否补 sigmoid，见模块文档）
        let need_sigmoid = kpt_conf_need_sigmoid_end2end(flat, rows, cols, nk);

        let mut results = Vec::new();
        for r in 0..rows {
            let row_off = r * cols;
            let confidence = flat[row_off + 4];
            if confidence < self.base.confidence_threshold() {
                continue;
            }
            let class_id = flat[row_off + 5] as i32;

            // 框：End2End 导出为输入分辨率像素 xyxy，letterbox 反算 + 边界裁剪
            let x1 = ((flat[row_off] - lb.dw) / lb.ratio).max(0.0).min(orig_width as f32);
            let y1 = ((flat[row_off + 1] - lb.dh) / lb.ratio).max(0.0).min(orig_height as f32);
            let x2 = ((flat[row_off + 2] - lb.dw) / lb.ratio).max(0.0).min(orig_width as f32);
            let y2 = ((flat[row_off + 3] - lb.dh) / lb.ratio).max(0.0).min(orig_height as f32);

            let detection = Detection::new(
                self.base.get_label_name(class_id),
                class_id,
                x1 as f64,
                y1 as f64,
                x2 as f64,
                y2 as f64,
                confidence as f64,
            );

            // 关键点：第 6 列起每 3 个一组 (x, y, conf)
            let mut keypoints = Vec::with_capacity(nk);
            for k in 0..nk {
                let col = 6 + k * 3;
                let kx = (flat[row_off + col] - lb.dw) / lb.ratio;
                let ky = (flat[row_off + col + 1] - lb.dh) / lb.ratio;
                let mut score = flat[row_off + col + 2];
                if need_sigmoid {
                    score = sigmoid(score);
                }
                keypoints.push(Keypoint::new(kx, ky, score));
            }

            results.push(PoseResult::new(detection, keypoints));
        }

        Ok(results)
    }

    // ==================== 通道拆分与 NMS ====================

    /// 动态拆分传统布局 channels = 4 + nc + 3*nk。
    ///
    /// nc 优先取模型元数据标签数，回退 1（YOLO-pose 默认单类 person）；
    /// 拆分失败（余数非 3 的倍数）时报错并给出诊断信息。
    fn split_channels(&self, channels: usize) -> Result<(usize, usize)> {
        let mut class_candidates: Vec<usize> = Vec::new();
        if let Some(labels) = self.base.labels() {
            class_candidates.push(labels.len());
        }
        class_candidates.push(1);

        for nc in class_candidates {
            let rem = channels as isize - 4 - nc as isize;
            if rem >= 3 && rem % 3 == 0 {
                let nk = (rem / 3) as usize;
                tracing::debug!(
                    "Pose channels={} split as 4 + nc={} + 3*nk={} (nk={})",
                    channels,
                    nc,
                    nk * 3,
                    nk
                );
                return Ok((nc, nk));
            }
        }

        Err(VisionError::inference(format!(
            "Cannot split pose output channels={} into 4 + nc + 3*nk (labels={:?})",
            channels,
            self.base.labels().map(|l| l.len())
        )))
    }

    /// 类内 NMS（按分数降序，同类别 IoU > 阈值时抑制）。
    fn nms_select(&self, mut candidates: Vec<PoseCandidate>) -> Vec<PoseCandidate> {
        if candidates.is_empty() {
            return candidates;
        }

        // 按分数降序（稳定排序，等分保持 anchor 顺序）
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        let mut suppressed = vec![false; candidates.len()];
        let mut kept = Vec::new();

        for i in 0..candidates.len() {
            if suppressed[i] {
                continue;
            }
            let curr = candidates[i];
            kept.push(curr);

            // 抑制同类重叠候选（输入空间坐标）
            let curr_box = BoundingBox::new(
                curr.x1 as f64,
                curr.y1 as f64,
                curr.x2 as f64,
                curr.y2 as f64,
                curr.score as f64,
            );
            for j in (i + 1)..candidates.len() {
                if suppressed[j] {
                    continue;
                }
                let cand = candidates[j];
                if cand.class_id != curr.class_id {
                    continue;
                }
                let cand_box = BoundingBox::new(
                    cand.x1 as f64,
                    cand.y1 as f64,
                    cand.x2 as f64,
                    cand.y2 as f64,
                    cand.score as f64,
                );
                if curr_box.iou(&cand_box) > self.nms_threshold as f64 {
                    suppressed[j] = true;
                }
            }
        }

        kept
    }

    /// 1D flat 输出的维度推断：按 channels = 4 + nc + 3*17（假设 COCO 17 点）尝试。
    ///
    /// 1D 输出在实际导出中极少见，仅做 COCO 假设的兜底。
    fn reshape_1d_dims(&self, total_elements: usize) -> Result<(usize, usize)> {
        let mut class_candidates: Vec<usize> = Vec::new();
        if let Some(labels) = self.base.labels() {
            class_candidates.push(labels.len());
        }
        class_candidates.push(1);

        for nc in class_candidates {
            let channels = 4 + nc + 3 * 17;
            if total_elements >= channels && total_elements % channels == 0 {
                let anchors = total_elements / channels;
                tracing::info!(
                    "Reshaping 1D pose output [{}] to [{}, {}] (assuming nc={}, nk=17)",
                    total_elements,
                    channels,
                    anchors,
                    nc
                );
                return Ok((channels, anchors));
            }
        }

        Err(VisionError::inference(format!(
            "Cannot reshape 1D pose output of {} elements into [channels, anchors] (assumed 17 keypoints)",
            total_elements
        )))
    }

    // ==================== 输出定位与布局自动检测 ====================

    /// 从多个输出中找到姿态输出张量（跳过 shape=[1] 等元数据输出，取元素数最多者）。
    fn find_pose_output(outputs: &[TensorOutput]) -> Result<&TensorOutput> {
        if outputs.len() == 1 {
            return Ok(&outputs[0]);
        }

        let mut best = &outputs[0];
        let mut best_count = best.element_count();
        for o in &outputs[1..] {
            let count = o.element_count();
            if count > best_count {
                best_count = count;
                best = o;
            }
        }

        tracing::debug!(
            "Selected pose output tensor with shape={:?}, elements={}",
            best.shape,
            best_count
        );
        Ok(best)
    }

    /// 获取（必要时首次计算）布局自动检测结果（`Mutex<Option<bool>>` 缓存）。
    fn resolve_end2_end(&self, output_tensor: &TensorOutput) -> Result<bool> {
        let mut cache = self.auto_detected_end2_end.lock().unwrap();
        if cache.is_none() {
            *cache = Some(Self::auto_detect_end2_end(&output_tensor.shape, self.end2_end));
        }
        Ok(cache.unwrap())
    }

    /// 首次推理时自动检测输出布局（基于实际输出 shape；同源 detection 引擎 auto_detect 思路）。
    ///
    /// - End2End（NMS-Free，如 YOLO26-pose）：`[1, M, 6+3*nk]` —— dim1 为最大检测数 M
    ///   （Ultralytics 默认 300），dim2 为行宽（COCO 17 点时 57）
    /// - 传统（YOLOv8/v11-pose）：`[1, 4+nc+3*nk, anchors]` —— dim1 为 channels（COCO 为 56），
    ///   dim2 为 anchors（如 8400）
    ///
    /// channels 随类别数动态变化（nc 较大时 dim1 可能落入 [100,400]），但此时 dim2=anchors
    /// 远超 100，仍会被 dim2 条件排除；两条件同时满足才判为 End2End，避免大类别数传统模型误判。
    fn auto_detect_end2_end(shape: &[i64], constructor_end2end: bool) -> bool {
        if shape.len() != 3 {
            // 非 3D 输出默认沿用构造函数指定
            tracing::info!(
                "Pose output shape {:?} is not 3D, using constructor end2End={}",
                shape,
                constructor_end2end
            );
            return constructor_end2end;
        }

        let dim1 = shape[1]; // channels 或最大检测数 M
        let dim2 = shape[2]; // anchors 或行宽 6+3*nk

        let is_end2end = (100..=400).contains(&dim1) && (4..=100).contains(&dim2);

        tracing::info!(
            "Auto-detected pose output layout: {} (shape=[{},{},{}], dim1={}, dim2={})",
            if is_end2end { "End2End" } else { "Traditional" },
            shape[0],
            shape[1],
            shape[2],
            dim1,
            dim2
        );

        is_end2end
    }

    // ==================== 日志 ====================

    /// 打印姿态估计结果日志（实例明细为 debug 级别，避免高频推理刷屏）。
    fn log_results(&self, results: &[PoseResult]) {
        tracing::info!(
            "Detected {} pose instances{}",
            results.len(),
            if results.is_empty() { "" } else { ":" }
        );
        for r in results {
            tracing::debug!("  {}, keypoints={}", r.detection, r.keypoints.len());
        }
    }
}

/// 关键点置信度是否为原始 logits：区间出现负值或 >1（+容差）时判为需要补 sigmoid。
///
/// 与 detection 引擎 Transformer 分数的自适应策略一致：官方 Ultralytics 导出已在
/// 计算图内对 kpt conf 过 sigmoid（值域 [0,1]），此时不再二次 sigmoid。
fn kpt_conf_need_sigmoid_standard(m: &Matrix2D, kpt_base: usize, nk: usize, anchors: usize) -> bool {
    let mut min_score = f32::MAX;
    let mut max_score = -f32::MAX;
    for k in 0..nk {
        let row = kpt_base + k * 3 + 2; // 每组第 3 个值是置信度
        for a in 0..anchors {
            let v = m.get(row, a);
            if v < min_score {
                min_score = v;
            }
            if v > max_score {
                max_score = v;
            }
        }
    }
    min_score < 0.0 || max_score > 1.0 + 1e-3
}

/// End2End 布局的关键点置信度自适应判定（列偏移 `6 + k*3 + 2`）。
fn kpt_conf_need_sigmoid_end2end(flat: &[f32], rows: usize, cols: usize, nk: usize) -> bool {
    let mut min_score = f32::MAX;
    let mut max_score = -f32::MAX;
    for r in 0..rows {
        let row_off = r * cols;
        for k in 0..nk {
            let idx = row_off + 6 + k * 3 + 2;
            if idx >= flat.len() {
                return false;
            }
            let v = flat[idx];
            if v < min_score {
                min_score = v;
            }
            if v > max_score {
                max_score = v;
            }
        }
    }
    min_score < 0.0 || max_score > 1.0 + 1e-3
}

/// sigmoid（关键点置信度为原始 logits 时使用）。
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

crate::impl_engine_forward!(PoseEngine, base, Vec<PoseResult>,
    /// 单图推理。
    fn predict(&self, image: &Image) -> Result<Vec<PoseResult>> {
        self.predict_impl(image)
    },
    /// 批量推理（动态 batch 走真实批量路径；固定 batch 导出自动退化为逐张推理）。
    fn predict_batch(&self, images: &[Image]) -> Result<Vec<Vec<PoseResult>>> {
        self.predict_batch_impl(images)
    }
);
