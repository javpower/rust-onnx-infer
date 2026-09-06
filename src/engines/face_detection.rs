//! YuNet 人脸检测推理引擎（OpenCV Zoo `face_detection_yunet_2023mar.onnx`）。
//!
//! 实现细节均已对照 OpenCV 官方 `FaceDetectorYNImpl`
//! （modules/objdetect/src/face_detect.cpp）与仓库内真实模型
//! （testmodels/face_detection_yunet_2023mar.onnx，静态输入 [1,3,640,640]）核实：
//!
//! - **预处理**：图像拉伸 resize 到输入尺寸后，右/下补 0 至 32 的倍数（divisor=32，
//!   对应官方 `padWithDivisor`）；输入张量为 **BGR、0~255 原始像素值，不做归一化**
//!   （官方 `blobFromImage(pad_image)` 使用默认参数：scale=1、mean=0、swapRB=false，
//!   且该 ONNX 计算图以 Conv 直接消费输入，内部无归一化节点）；
//! - **官方模型输出布局**：12 个头，stride ∈ {8, 16, 32}，名字顺序
//!   `cls_8, cls_16, cls_32, obj_8, obj_16, obj_32, bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32`
//!   （cls/obj 形如 [1, N, 1]，bbox [1, N, 4]，kps [1, N, 10]，anchor 行优先，属性在最内维），
//!   逐 anchor 解码规则（与官方 postProcess 逐行一致）：
//!     - `score = sqrt(clamp(cls,0,1) * clamp(obj,0,1))`
//!     - `cx = (col + bbox[0]) * stride`，`cy = (row + bbox[1]) * stride`
//!     - `w  = exp(bbox[2]) * stride`，`h  = exp(bbox[3]) * stride`
//!     - 关键点 `x = (kps[2n] + col) * stride`，`y = (kps[2n+1] + row) * stride`
//! - **融合导出兼容**：部分第三方导出把解码融合进计算图，单输出 `[1, N, 15]`
//!   （每行 = x, y, w, h + 5 关键点 ×(x,y) + 置信度，坐标已为输入像素），自动识别分派；
//! - **关键点顺序**：YuNet 原始输出为（右眼, 左眼, 鼻尖, 右嘴角, 左嘴角）（以人脸自身
//!   为参照，即图像中偏左的眼在首位）；crate 约定 [`crate::model::face_landmarks::NAMES`]
//!   的 left_eye/right_eye 以观察者视角命名，图像偏左的眼同样是首位——两者按索引一一对应
//!   （OpenCV `FaceRecognizerSF::alignCrop` 即将 YuNet 关键点与 ArcFace 规范 5 点按索引
//!   直接配对，可证首点为图像左侧眼），故不做任何位置交换，仅做命名映射说明；
//! - **后处理**：score 阈值过滤（默认 0.6）→ top-k 截断（默认 5000）→ 贪心 NMS
//!   （默认 IoU 0.3，复用 [`BoundingBox::iou`]），与官方
//!   `dnn::NMSBoxes(..., scoreThreshold, nmsThreshold, keepIdx, 1.0, topK)` 语义一致；
//! - 坐标按拉伸 resize 比例还原到原图像素并 clamp 到图内。

use ort::value::Tensor;

use crate::core::base::{BaseOnnxEngine, TensorOutput};
use crate::core::device_type::DeviceType;
use crate::error::{Result, VisionError};
use crate::imaging::{copy_make_border, cvt_color, resize, BorderType, ColorConversion, Image, Interpolation};
use crate::model::{BoundingBox, Detection, Keypoint};

/// pad divisor：输入需补边到 32 的倍数（对应官方 divisor=32）。
const YUNET_DIVISOR: usize = 32;

/// YuNet 单张人脸检测结果：检测框 + 5 关键点（顺序 = [`crate::model::face_landmarks::NAMES`]）。
#[derive(Debug, Clone, PartialEq)]
pub struct FaceDetectResult {
    /// 基础检测信息（框 + 类别 "face" + 置信度）
    pub detection: Detection,
    /// 人脸 5 关键点（原图像素）：left_eye / right_eye / nose_tip / left_mouth_corner / right_mouth_corner
    pub landmarks: Vec<Keypoint>,
}

impl FaceDetectResult {
    pub fn new(detection: Detection, landmarks: Vec<Keypoint>) -> Self {
        FaceDetectResult { detection, landmarks }
    }

    /// 置信度。
    pub fn confidence(&self) -> f64 {
        self.detection.confidence()
    }
}

/// 解码后的候选脸（模型输入坐标系；关键点为 YuNet 原始顺序：图像左眼, 图像右眼, 鼻尖,
/// 图像左嘴角, 图像右嘴角——即官方注释的 re/le/nt/rcm/lcm，与人脸自身左右相反）。
#[derive(Debug, Clone, Copy)]
struct RawFace {
    /// 框左上角 x（输入像素）
    x: f32,
    /// 框左上角 y（输入像素）
    y: f32,
    /// 框宽（输入像素）
    w: f32,
    /// 框高（输入像素）
    h: f32,
    /// 置信度
    score: f32,
    /// 10 个坐标：YuNet 原始顺序 5 点 × (x, y)
    kps: [f32; 10],
}

/// YuNet 人脸检测引擎。
pub struct FaceDetectionEngine {
    /// 组合基类
    pub base: BaseOnnxEngine,
    /// NMS IoU 阈值（官方默认 0.3）
    nms_threshold: f32,
    /// NMS 前按分数保留的最大候选数（官方默认 5000）
    top_k: usize,
}

impl FaceDetectionEngine {
    /// 创建 YuNet 人脸检测引擎。
    ///
    /// 输入尺寸从模型元信息读取（官方导出为静态 640x640）；动态输入模型默认 320x320
    /// （opencv_zoo YuNet 参考实现的默认输入尺寸）。score 阈值默认 0.6。
    pub fn new(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        Self::with_input_size(model_path, device_type, -1, -1)
    }

    /// 创建 YuNet 人脸检测引擎（指定输入尺寸；<=0 时从模型读取，动态维度回退 320x320）。
    pub fn with_input_size(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        input_height: i32,
        input_width: i32,
    ) -> Result<Self> {
        let h = if input_height > 0 { input_height } else { 320 };
        let w = if input_width > 0 { input_width } else { 320 };
        let mut base = BaseOnnxEngine::with_input_size(model_path, device_type, h, w)?;
        // YuNet 输入为 0~255 原始像素（官方模型图内无归一化节点），关闭基类归一化以表意
        base.set_normalize(false);
        // 官方 demo / 任务约定 score 阈值 0.6（基类默认 0.5）
        base.set_confidence_threshold(0.6);
        Ok(FaceDetectionEngine {
            base,
            nms_threshold: 0.3,
            top_k: 5000,
        })
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

    /// NMS 前保留的最大候选数。
    pub fn top_k(&self) -> usize {
        self.top_k
    }

    /// 设置 NMS 前保留的最大候选数。
    pub fn set_top_k(&mut self, top_k: usize) {
        self.top_k = top_k;
    }

    // ============ 推理 ============

    /// 单图推理核心实现（trait `predict` 转发到这里）。
    pub fn predict_impl(&self, image: &Image) -> Result<Vec<FaceDetectResult>> {
        if image.is_empty() {
            return Err(VisionError::image("cannot detect faces on empty image"));
        }
        let orig_width = image.width();
        let orig_height = image.height();

        // 1. 统一转 3 通道 BGR（YuNet 输入约定 BGR 通道序）
        let bgr = match image.channels() {
            3 => image.clone(),
            4 => cvt_color(image, ColorConversion::Bgra2Bgr)?,
            _ => cvt_color(image, ColorConversion::Gray2Bgr)?,
        };

        // 2. 拉伸 resize 到输入尺寸（对应官方约定：调用方先把图 resize 到 inputSize）
        let input_width = self.base.input_width() as usize;
        let input_height = self.base.input_height() as usize;
        let resized = resize(&bgr, input_width, input_height, Interpolation::Linear)?;

        // 3. 右/下补 0 到 32 的倍数（对应官方 padWithDivisor）
        let pad_width = ((input_width - 1) / YUNET_DIVISOR + 1) * YUNET_DIVISOR;
        let pad_height = ((input_height - 1) / YUNET_DIVISOR + 1) * YUNET_DIVISOR;
        let padded = if pad_width != input_width || pad_height != input_height {
            copy_make_border(
                &resized,
                0,
                pad_height - input_height,
                0,
                pad_width - input_width,
                BorderType::Constant(0),
            )?
        } else {
            resized
        };

        // 4. HWC(BGR, 0~255) → CHW，不做归一化（官方 blobFromImage 默认参数）
        let px = padded.data();
        let area = pad_width * pad_height;
        let mut data = vec![0f32; 3 * area];
        for i in 0..area {
            data[i] = px[i * 3] as f32;
            data[i + area] = px[i * 3 + 1] as f32;
            data[i + 2 * area] = px[i * 3 + 2] as f32;
        }

        // 5. 推理（输入形状为补边后的实际尺寸；官方导出静态 640x640 时补边为空操作）
        let tensor = Tensor::from_array((vec![1i64, 3, pad_height as i64, pad_width as i64], data))?;
        let outputs = self.base.run_multi_output(tensor)?;

        // 6. 解码（官方 12 头布局或融合 [1,N,15] 布局自动分派）
        let faces = if outputs.len() == 1 {
            self.decode_fused(&outputs[0])?
        } else {
            self.decode_by_stride(&outputs, pad_width, pad_height)?
        };

        // 7. 后处理：过滤 / top-k / NMS / 坐标还原
        let results = self.postprocess(faces, orig_width, orig_height);
        tracing::info!("YuNet detected {} face(s)", results.len());
        Ok(results)
    }

    // ==================== 解码 ====================

    /// 官方 12 头布局解码（对应 FaceDetectorYNImpl::postProcess）。
    fn decode_by_stride(
        &self,
        outputs: &[TensorOutput],
        pad_width: usize,
        pad_height: usize,
    ) -> Result<Vec<RawFace>> {
        let indices = Self::head_indices(outputs)?;
        let strides = [8usize, 16, 32];
        let score_threshold = self.base.confidence_threshold();
        let mut faces = Vec::new();

        for (si, &s) in strides.iter().enumerate() {
            // 特征图网格：pad 尺寸 / stride（与官方一致）
            let cols = pad_width / s;
            let rows = pad_height / s;
            let n = rows * cols;

            let cls = outputs[indices[si]].as_f32()?;
            let obj = outputs[indices[si + 3]].as_f32()?;
            let bbox = outputs[indices[si + 6]].as_f32()?;
            let kps = outputs[indices[si + 9]].as_f32()?;

            // 长度校验（cls/obj 每 anchor 1 个值；bbox 4 个；kps 10 个）
            if cls.len() < n || obj.len() < n || bbox.len() < n * 4 || kps.len() < n * 10 {
                return Err(VisionError::inference(format!(
                    "YuNet stride-{s} head size mismatch: cls={}, obj={}, bbox={}, kps={}; expected >= {n}, {n}, {}, {}",
                    cls.len(),
                    obj.len(),
                    bbox.len(),
                    kps.len(),
                    n * 4,
                    n * 10
                )));
            }

            for r in 0..rows {
                for c in 0..cols {
                    let idx = r * cols + c;
                    // 分数 = sqrt(cls * obj)，先 clamp 到 [0,1]（与官方一致）
                    let cls_score = cls[idx].clamp(0.0, 1.0);
                    let obj_score = obj[idx].clamp(0.0, 1.0);
                    let score = (cls_score * obj_score).sqrt();
                    if score < score_threshold {
                        continue;
                    }
                    // 框：anchor 中心 + 回归偏移，再按 stride 还原到输入像素
                    let cx = (c as f32 + bbox[idx * 4]) * s as f32;
                    let cy = (r as f32 + bbox[idx * 4 + 1]) * s as f32;
                    let w = bbox[idx * 4 + 2].exp() * s as f32;
                    let h = bbox[idx * 4 + 3].exp() * s as f32;
                    // 关键点：网格偏移 + stride 还原（x 配 col，y 配 row）
                    let mut kp = [0f32; 10];
                    for (k, item) in kp.iter_mut().enumerate() {
                        let offset = if k % 2 == 0 { c as f32 } else { r as f32 };
                        *item = (kps[idx * 10 + k] + offset) * s as f32;
                    }
                    faces.push(RawFace { x: cx - w / 2.0, y: cy - h / 2.0, w, h, score, kps: kp });
                }
            }
        }
        Ok(faces)
    }

    /// 融合单输出布局解码：`[1, N, 15]`（或 `[N, 15]`），每行
    /// `x, y, w, h, 5 关键点 ×(x,y), score`，坐标已为输入像素值。
    fn decode_fused(&self, output: &TensorOutput) -> Result<Vec<RawFace>> {
        const ROW: usize = 15;
        let flat = output.as_f32()?;
        let shape = &output.shape;
        let (rows, cols) = match shape.len() {
            3 => (shape[1].max(0) as usize, shape[2].max(0) as usize),
            2 => (shape[0].max(0) as usize, shape[1].max(0) as usize),
            _ => (0, 0),
        };
        // 取第一个 batch（引擎为单图推理）
        let (rows, _cols) = if cols == ROW && rows > 0 {
            (rows, cols)
        } else if flat.len() >= ROW && flat.len() % ROW == 0 {
            let n = flat.len() / ROW;
            tracing::info!("Reshaping YuNet fused output (shape={shape:?}, elements={}) to [{n}, 15]", flat.len());
            (n, ROW)
        } else {
            return Err(VisionError::inference(format!(
                "YuNet fused output shape {shape:?} ({} elements) is not [batch, N, 15]",
                flat.len()
            )));
        };

        let score_threshold = self.base.confidence_threshold();
        let mut faces = Vec::with_capacity(rows);
        for r in 0..rows {
            let off = r * ROW;
            let score = flat[off + 14];
            if score < score_threshold {
                continue;
            }
            let mut kp = [0f32; 10];
            kp.copy_from_slice(&flat[off + 4..off + 14]);
            faces.push(RawFace {
                x: flat[off],
                y: flat[off + 1],
                w: flat[off + 2],
                h: flat[off + 3],
                score,
                kps: kp,
            });
        }
        Ok(faces)
    }

    /// 在输出列表中定位 12 个头：优先按名字匹配（cls_8/obj_8/.../kps_32），
    /// 名字不可用时按官方导出顺序兜底（要求恰好 12 个输出）。
    fn head_indices(outputs: &[TensorOutput]) -> Result<[usize; 12]> {
        let frags = [
            "cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32", "bbox_8", "bbox_16",
            "bbox_32", "kps_8", "kps_16", "kps_32",
        ];
        let mut idx = [0usize; 12];
        let mut all_matched = true;
        for (k, frag) in frags.iter().enumerate() {
            match outputs.iter().position(|o| o.name.to_lowercase().contains(frag)) {
                Some(p) => idx[k] = p,
                None => {
                    all_matched = false;
                    break;
                }
            }
        }
        if !all_matched {
            if outputs.len() != 12 {
                return Err(VisionError::inference(format!(
                    "YuNet model outputs are neither the official 12-head layout nor a single fused output (got {} outputs: {:?})",
                    outputs.len(),
                    outputs.iter().map(|o| o.name.as_str()).collect::<Vec<_>>()
                )));
            }
            for (k, item) in idx.iter_mut().enumerate() {
                *item = k;
            }
        }
        Ok(idx)
    }

    // ==================== 后处理 ====================

    /// 按分数降序 top-k → 贪心 NMS → 坐标还原到原图 → 组装结果。
    fn postprocess(
        &self,
        mut faces: Vec<RawFace>,
        orig_width: usize,
        orig_height: usize,
    ) -> Vec<FaceDetectResult> {
        if faces.is_empty() {
            return Vec::new();
        }

        // 按分数降序 + top-k 截断（对应官方 NMSBoxes 的 top_k）
        faces.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        faces.truncate(self.top_k);

        // 拉伸 resize 的比例还原系数
        let input_width = self.base.input_width() as f32;
        let input_height = self.base.input_height() as f32;
        let scale_x = orig_width as f32 / input_width;
        let scale_y = orig_height as f32 / input_height;
        let orig_w = orig_width as f32;
        let orig_h = orig_height as f32;

        // 贪心 NMS（IoU 复用 BoundingBox::iou，坐标尺度不影响 IoU）
        let mut suppressed = vec![false; faces.len()];
        let mut results = Vec::new();
        for i in 0..faces.len() {
            if suppressed[i] {
                continue;
            }
            let cur = faces[i];
            let cur_box = BoundingBox::new(
                cur.x as f64,
                cur.y as f64,
                (cur.x + cur.w) as f64,
                (cur.y + cur.h) as f64,
                cur.score as f64,
            );
            for j in (i + 1)..faces.len() {
                if suppressed[j] {
                    continue;
                }
                let other = faces[j];
                let other_box = BoundingBox::new(
                    other.x as f64,
                    other.y as f64,
                    (other.x + other.w) as f64,
                    (other.y + other.h) as f64,
                    other.score as f64,
                );
                if cur_box.iou(&other_box) > self.nms_threshold as f64 {
                    suppressed[j] = true;
                }
            }

            // 坐标还原到原图并 clamp 到图内
            let x1 = (cur.x * scale_x).clamp(0.0, orig_w);
            let y1 = (cur.y * scale_y).clamp(0.0, orig_h);
            let x2 = ((cur.x + cur.w) * scale_x).clamp(0.0, orig_w);
            let y2 = ((cur.y + cur.h) * scale_y).clamp(0.0, orig_h);

            // YuNet 原始顺序与 crate 约定按索引一一对应（均为“图像中偏左”的点在首位，
            // 详见模块注释），此处仅按索引取点。
            let landmarks = (0..5)
                .map(|k| {
                    Keypoint::new(
                        (cur.kps[k * 2] * scale_x).clamp(0.0, orig_w),
                        (cur.kps[k * 2 + 1] * scale_y).clamp(0.0, orig_h),
                        cur.score,
                    )
                })
                .collect();

            results.push(FaceDetectResult::new(
                Detection::new("face", 0, x1 as f64, y1 as f64, x2 as f64, y2 as f64, cur.score as f64),
                landmarks,
            ));
        }
        results
    }
}

crate::impl_engine_forward!(FaceDetectionEngine, base, Vec<FaceDetectResult>,
    /// 单图推理：返回全部人脸（score 阈值默认 0.6，框与关键点均为原图像素坐标）。
    fn predict(&self, image: &Image) -> Result<Vec<FaceDetectResult>> {
        self.predict_impl(image)
    }
);
