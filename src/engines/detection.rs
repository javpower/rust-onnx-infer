//! 目标检测推理引擎。
//!
//! **支持的模型与后处理路径**（按 [`DetectionModelType`] 分派，互不影响）：
//!
//! | 模型 | 工厂方法 | 预处理 | 输出布局 | 后处理 |
//! |---|---|---|---|---|
//! | YOLOv5/v8/v9/v10/v11（传统格式） | [`for_yolo`] | letterbox（114 灰边）+ /255 | `[1, 4+nc, anchors]` | 阈值过滤 + 类内 NMS + 坐标 letterbox 反算 |
//! | YOLO26 / v10（End2End NMS-Free） | [`for_yolo_end2_end`] 或 `for_yolo` 自动识别 | 同上 | `[1, 300, 6]`（x1,y1,x2,y2,conf,cls） | 阈值过滤，无需 NMS |
//! | RT-DETR / DETR（Ultralytics 导出） | [`for_rtdetr`] / [`for_detr`] | 拉伸 resize（scaleFill）+ /255 | `[1, num_queries, 4+nc]`，框为归一化 cxcywh，分数为概率 | 每 query 类别 argmax，坐标 ×原图宽高（无 NMS） |
//! | RF-DETR（Roboflow 导出） | [`for_rfdetr`] | 拉伸 resize + ImageNet 归一化 | `dets=[1,Q,4] + labels=[1,Q,nc]`（logits） | sigmoid → Q×C 扁平 top-k（多标签，与官方 predict 对齐） |
//!
//! **路径选择规则**：`DetectionModelType::Yolo`（默认）时保持历史行为 —— letterbox 预处理 +
//! 首次推理时按输出 shape 自动识别传统/End2End 格式（dim1∈[100,400] 且 dim2∈[4,100] 判为
//! End2End，两者结合判断以避免 80 类传统模型 channels=116 被误判）；
//! 非 YOLO 类型走 [`DetectionEngine::predict_transformer`]，预处理改为拉伸 resize，跳过 YOLO 自动识别。
//!
//! **分数与框格式自适应**（Transformer 路径）：分数张量存在负值或 >1 时自动过 sigmoid
//! （兼容"导出已 sigmoid"与"导出原始 logits"两种导出习惯）；框坐标按最大值 >2 区分
//! 归一化 cxcywh 与输入分辨率像素 xyxy。
//!
//! 与 原版的实现差异（数值逻辑保持逐行一致）：
//! - letterbox 参数（ratio/dw/dh）随调用链显式传递
//!   （Rust `&self` 不可变，且避免多线程共享时的字段竞争）；
//! - End2End 自动检测结果用 `Mutex<Option<bool>>` 缓存（对应 可空 `Boolean` 字段）；
//! - SAHI 切片推理：开启时 `predict_impl` 走 [`crate::sahi::sliced_predictor`] 完整流程。

use std::sync::Mutex;

use crate::core::base::{BaseOnnxEngine, TensorOutput};
use crate::core::device_type::DeviceType;
use crate::error::{Result, VisionError};
use crate::imaging::{cvt_color, resize, ColorConversion, Image, Interpolation};
use crate::model::{BoundingBox, Detection};

/// ImageNet 归一化参数（RF-DETR / 原版 DETR 等 ViT backbone 模型使用）。
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// ImageNet 归一化标准差。
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// 检测模型类型（对应 `DetectionEngine.ModelType`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DetectionModelType {
    /// YOLO 系列（letterbox 预处理 + 传统/End2End 输出自动识别）
    #[default]
    Yolo,
    /// RT-DETR（百度实时 Transformer 检测器）
    RtDetr,
    /// DETR（标准 Transformer 检测模型）
    Detr,
    /// RF-DETR（Roboflow，双输出 dets/labels）
    RfDetr,
}

/// letterbox 预处理的坐标还原参数（对应上游 实例字段 ratio/dw/dh）。
#[derive(Debug, Clone, Copy, Default)]
struct LetterboxParams {
    /// 缩放比例（min(input_w/orig_w, input_h/orig_h)）
    ratio: f32,
    /// 左右对称填充的一半宽度
    dw: f32,
    /// 上下对称填充的一半高度
    dh: f32,
}

/// 行主序二维 float 矩阵。
#[derive(Debug, Clone)]
struct Matrix2D {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Matrix2D {
    /// 从 flat 数据按 shape 解释为 [rows][cols]（越界报错，对齐 上游实现长度校验）。
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

/// 目标检测推理引擎。
pub struct DetectionEngine {
    /// 组合基类
    pub base: BaseOnnxEngine,

    /// NMS IoU 阈值（默认 0.45）
    nms_threshold: f32,
    /// 是否为 end2end 模型（内置 NMS）
    end2_end: bool,
    /// 自动检测结果缓存（首次推理时检测；对应 `Boolean autoDetectedEnd2End`）
    auto_detected_end2_end: Mutex<Option<bool>>,
    /// 模型类型
    model_type: DetectionModelType,
    /// Transformer 输出保留的最大检测数（对应官方 top-k num_select / max_det）
    max_detections: usize,
}

impl DetectionEngine {
    /// 创建 YOLO 检测引擎。
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

    /// 创建 YOLO 检测引擎（指定输入尺寸）。
    pub fn for_yolo_with_input_size(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        confidence_threshold: f32,
        nms_threshold: f32,
        input_height: i32,
        input_width: i32,
    ) -> Result<Self> {
        let mut engine = Self::new_with_input_size(
            model_path,
            device_type,
            false,
            input_height,
            input_width,
        )?;
        engine.base.set_confidence_threshold(confidence_threshold);
        engine.nms_threshold = nms_threshold;
        Ok(engine)
    }

    /// 创建 YOLO End2End 检测引擎（无需 NMS）。
    pub fn for_yolo_end2_end(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        confidence_threshold: f32,
    ) -> Result<Self> {
        Self::for_yolo_end2_end_with_input_size(model_path, device_type, confidence_threshold, -1, -1)
    }

    /// 创建 YOLO End2End 检测引擎（指定输入尺寸）。
    pub fn for_yolo_end2_end_with_input_size(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        confidence_threshold: f32,
        input_height: i32,
        input_width: i32,
    ) -> Result<Self> {
        let mut engine =
            Self::new_with_input_size(model_path, device_type, true, input_height, input_width)?;
        engine.base.set_confidence_threshold(confidence_threshold);
        Ok(engine)
    }

    /// 创建 RT-DETR 检测引擎（Transformer 模型）。
    ///
    /// RT-DETR 是百度开发的实时 Transformer 检测器。
    pub fn for_rtdetr(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        confidence_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Self> {
        Self::for_rtdetr_with_input_size(
            model_path,
            device_type,
            confidence_threshold,
            nms_threshold,
            -1,
            -1,
        )
    }

    /// 创建 RT-DETR 检测引擎（指定输入尺寸）。
    pub fn for_rtdetr_with_input_size(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        confidence_threshold: f32,
        nms_threshold: f32,
        input_height: i32,
        input_width: i32,
    ) -> Result<Self> {
        let mut engine =
            Self::new_with_input_size(model_path, device_type, true, input_height, input_width)?;
        engine.base.set_confidence_threshold(confidence_threshold);
        engine.nms_threshold = nms_threshold;
        engine.model_type = DetectionModelType::RtDetr;
        Ok(engine)
    }

    /// 创建 DETR 检测引擎（标准 Transformer 检测模型）。
    pub fn for_detr(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        confidence_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Self> {
        Self::for_detr_with_input_size(
            model_path,
            device_type,
            confidence_threshold,
            nms_threshold,
            -1,
            -1,
        )
    }

    /// 创建 DETR 检测引擎（指定输入尺寸）。
    pub fn for_detr_with_input_size(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        confidence_threshold: f32,
        nms_threshold: f32,
        input_height: i32,
        input_width: i32,
    ) -> Result<Self> {
        let mut engine =
            Self::new_with_input_size(model_path, device_type, true, input_height, input_width)?;
        engine.base.set_confidence_threshold(confidence_threshold);
        engine.nms_threshold = nms_threshold;
        engine.model_type = DetectionModelType::Detr;
        Ok(engine)
    }

    /// 创建 RF-DETR 检测引擎（Roboflow，ONNX 由 RFDETR.export 导出）。
    ///
    /// 输入：拉伸 resize 到 resolution×resolution + ImageNet 归一化（DINOv2 backbone）；
    /// 输出：dets=[1,Q,4]（归一化 cxcywh）+ labels=[1,Q,nc]（logits，需 sigmoid）。
    /// 后处理与官方 RFDETR.predict 对齐：每类独立 sigmoid → Q×C 扁平 top-k（默认 300）→ 阈值过滤。
    ///
    /// RF-DETR 导出的 ONNX 通常不含类别名元数据，可用 `set_labels` 手动设置。
    pub fn for_rfdetr(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        confidence_threshold: f32,
    ) -> Result<Self> {
        let mut engine = Self::new_with_input_size(model_path, device_type, true, -1, -1)?;
        engine.base.set_confidence_threshold(confidence_threshold);
        engine.base.set_normalization(IMAGENET_MEAN, IMAGENET_STD);
        engine.model_type = DetectionModelType::RfDetr;
        Ok(engine)
    }

    /// 创建检测引擎（默认输入尺寸从模型读取，动态维度回退 640）。
    pub fn new(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        end2_end: bool,
    ) -> Result<Self> {
        Self::new_with_input_size(model_path, device_type, end2_end, -1, -1)
    }

    /// 创建检测引擎（指定输入尺寸；<=0 时从模型读取）。
    pub fn new_with_input_size(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        end2_end: bool,
        input_height: i32,
        input_width: i32,
    ) -> Result<Self> {
        let base =
            BaseOnnxEngine::with_input_size(model_path, device_type, input_height, input_width)?;
        Ok(DetectionEngine {
            base,
            nms_threshold: 0.45,
            end2_end,
            auto_detected_end2_end: Mutex::new(None),
            model_type: DetectionModelType::Yolo,
            max_detections: 300,
        })
    }

    /// 创建检测引擎（自定义运行参数：线程数 / GPU 设备 id）。
    pub fn for_yolo_with_config(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        confidence_threshold: f32,
        nms_threshold: f32,
        runtime_config: crate::core::runtime_config::OnnxRuntimeConfig,
    ) -> Result<Self> {
        let mut engine = Self::new_with_config(model_path, device_type, false, runtime_config)?;
        engine.base.set_confidence_threshold(confidence_threshold);
        engine.nms_threshold = nms_threshold;
        Ok(engine)
    }

    /// 创建检测引擎（指定运行参数）。
    pub fn new_with_config(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        end2_end: bool,
        runtime_config: crate::core::runtime_config::OnnxRuntimeConfig,
    ) -> Result<Self> {
        let base =
            BaseOnnxEngine::with_config(model_path, device_type, -1, -1, runtime_config)?;
        Ok(DetectionEngine {
            base,
            nms_threshold: 0.45,
            end2_end,
            auto_detected_end2_end: Mutex::new(None),
            model_type: DetectionModelType::Yolo,
            max_detections: 300,
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

    /// 是否为 end2end 模型（内置 NMS）。
    pub fn is_end2_end(&self) -> bool {
        self.end2_end
    }

    /// 获取模型类型。
    pub fn model_type(&self) -> DetectionModelType {
        self.model_type
    }

    /// Transformer 输出保留的最大检测数。
    pub fn max_detections(&self) -> usize {
        self.max_detections
    }

    /// 设置 Transformer 输出保留的最大检测数。
    pub fn set_max_detections(&mut self, max_detections: usize) {
        self.max_detections = max_detections;
    }

    // ============ 推理路径 ============

    /// 单图推理核心实现（trait `predict` 转发到这里）。
    pub fn predict_impl(&self, image: &Image) -> Result<Vec<Detection>> {
        // SAHI 开启时：切片 → 逐片调用原始单图路径 → 坐标映射 → 整图标准预测 → 合并（返回类型不变）
        if self.base.is_sahi_enabled() {
            let config = self
                .base
                .sahi_config()
                .expect("sahi config must exist when SAHI enabled");
            return crate::sahi::sliced_predictor::predict_sliced_detection(self, self, image, config)
                .map(|result| result.detections);
        }
        self.predict_without_sahi(image)
    }

    /// 原始单图推理路径（SAHI 关闭时的 `predict` 行为；SAHI 开启时被逐切片调用）。
    pub fn predict_without_sahi(&self, image: &Image) -> Result<Vec<Detection>> {
        // Transformer 系列（RT-DETR / DETR / RF-DETR）：拉伸 resize 预处理 + 独立后处理路径
        if self.model_type != DetectionModelType::Yolo {
            return self.predict_transformer(image);
        }

        // 1. 记录原始尺寸
        let orig_width = image.width();
        let orig_height = image.height();

        // 2. 预处理（letterbox + /255，同时记录坐标还原参数）
        let (input_data, lb) = self.preprocess_with_letterbox(image)?;

        // 3. 创建 Tensor
        let input_tensor = self.base.create_input_tensor(input_data)?;

        // 4. 运行推理（获取所有输出）
        let all_outputs = self.base.run_multi_output(input_tensor)?;

        // 5. 找到正确的检测输出张量（跳过 shape=[1] 等元数据输出）
        let output_tensor = Self::find_detection_output(&all_outputs)?;

        // 6. 自动检测模型类型（首次推理时，基于实际输出 shape）
        let end2end = self.resolve_end2_end(output_tensor)?;

        // 7. 后处理：使用自动检测结果
        let detections = if end2end {
            self.postprocess_end2_end(output_tensor, orig_width, orig_height, lb)?
        } else {
            self.postprocess_standard(output_tensor, orig_width, orig_height, lb)?
        };

        self.log_results(&detections);
        Ok(detections)
    }

    /// Transformer 检测模型推理（RT-DETR / DETR / RF-DETR）。
    ///
    /// 预处理为拉伸 resize（scaleFill，与 Ultralytics RT-DETR / RF-DETR 训练行为一致），
    /// 后处理按输出张量布局自动分派：
    /// - 单输出 `[1, Q, 4+nc]`：RT-DETR / DETR 风格
    /// - 双输出 `dets=[1, Q, 4] + labels=[1, Q, nc]`：RF-DETR 风格
    fn predict_transformer(&self, image: &Image) -> Result<Vec<Detection>> {
        let orig_width = image.width();
        let orig_height = image.height();

        // 拉伸 resize + mean/std 归一化（基类 preprocess，RF-DETR 已设 ImageNet 参数）
        let input_data = self.base.preprocess(image)?;
        let input_tensor = self.base.create_input_tensor(input_data)?;
        let outputs = self.base.run_multi_output(input_tensor)?;

        let detections = self.postprocess_transformer(&outputs, orig_width, orig_height)?;
        self.log_results(&detections);
        Ok(detections)
    }

    /// 批量推理核心实现（trait `predict_batch` 转发到这里）。
    pub fn predict_batch_impl(&self, images: &[Image]) -> Result<Vec<Vec<Detection>>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        // SAHI 开启时逐张走 predict（每张内部切片）
        if self.base.is_sahi_enabled() {
            return self.base.predict_batch_via_sahi(images, |img| self.predict_impl(img));
        }

        let batch_size = images.len();

        // 如果是单张图，直接调用普通 predict
        if batch_size == 1 {
            return Ok(vec![self.predict_impl(&images[0])?]);
        }

        // Transformer 导出模型固定 batch=1，逐张处理
        if self.model_type != DetectionModelType::Yolo {
            return images.iter().map(|img| self.predict_impl(img)).collect();
        }

        // 1. 批量预处理（带 Letterbox，逐图记录还原参数）
        let (batch_data, params) = self.preprocess_batch_with_letterbox(images)?;

        // 2. 创建批量 Tensor
        let input_tensor = self.base.create_batch_input_tensor(batch_data, batch_size)?;

        // 3. 运行推理
        let all_outputs = self.base.run_multi_output(input_tensor)?;
        let output_tensor = Self::find_detection_output(&all_outputs)?;

        // 4. 自动检测模型类型（首次推理时）
        let end2end = self.resolve_end2_end(output_tensor)?;

        // 5. 批量后处理
        let all_detections = if end2end {
            self.postprocess_batch_end2_end(output_tensor, batch_size, images, &params)?
        } else {
            self.postprocess_batch_standard(output_tensor, batch_size, images, &params)?
        };

        for detections in &all_detections {
            self.log_results(detections);
        }

        Ok(all_detections)
    }

    // ==================== 预处理 ====================

    /// 带 Letterbox 的预处理（保持宽高比，114 灰边填充）。
    ///
    /// 返回 `(CHW 浮点数据, 坐标还原参数)`；原版把参数写在实例字段上，
    /// Rust 侧显式返回以保证 `&self` 不可变语义。
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

        // 输入统一到 3 通道 BGR（实现假定输入为 CV_8UC3）
        let bgr = match image.channels() {
            3 => image.clone(),
            4 => cvt_color(image, ColorConversion::Bgra2Bgr)?,
            _ => cvt_color(image, ColorConversion::Gray2Bgr)?,
        };

        // 3. Resize
        let resized = resize(&bgr, new_width, new_height, Interpolation::Linear)?;

        // 4. 创建画布并填充（114 灰边）
        let mut padded = Image::filled(input_width, input_height, 3, 114);
        let top = ((dh - 0.1).round() as i32).max(0) as usize;
        let left = ((dw - 0.1).round() as i32).max(0) as usize;
        padded.paste(left, top, &resized);

        // 5. 颜色转换 BGR -> RGB
        let rgb = cvt_color(&padded, ColorConversion::Bgr2Rgb)?;

        // 6. 提取像素并归一化（/255，HWC → CHW）
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

    // ==================== YOLO 后处理 ====================

    /// End2End 模型后处理（NMS-Free）。
    ///
    /// 输出格式: `[batch, num_detections, 6]` -> `[x1, y1, x2, y2, confidence, class_id]`。
    fn postprocess_end2_end(
        &self,
        output_tensor: &TensorOutput,
        orig_width: usize,
        orig_height: usize,
        lb: LetterboxParams,
    ) -> Result<Vec<Detection>> {
        let flat = output_tensor.as_f32()?;
        let shape = &output_tensor.shape;

        // [num_detections, 6]
        let (rows, cols): (usize, usize) = match shape.len() {
            3 => (shape[1] as usize, shape[2] as usize),
            2 => (shape[0] as usize, shape[1] as usize),
            1 if flat.len() >= 6 && flat.len() % 6 == 0 => {
                // 1D flat output with [x1,y1,x2,y2,conf,cls, ...] layout
                let num_dets = flat.len() / 6;
                tracing::info!("Reshaping 1D End2End output [{}] to [{}, 6]", flat.len(), num_dets);
                (num_dets, 6)
            }
            _ => {
                return Err(VisionError::inference(format!(
                    "End2End model expects 2D/3D output, got {}D, shape: {:?}, total elements: {}",
                    shape.len(),
                    shape,
                    flat.len()
                )))
            }
        };
        let data = Matrix2D::from_flat(flat, rows, cols)?;

        let mut detections = Vec::new();
        for r in 0..rows {
            if cols < 6 {
                continue;
            }

            let confidence = data.get(r, 4);
            if confidence < self.base.confidence_threshold() {
                continue;
            }

            let class_id = data.get(r, 5) as i32;

            // 坐标还原
            let mut x1 = (data.get(r, 0) - lb.dw) / lb.ratio;
            let mut y1 = (data.get(r, 1) - lb.dh) / lb.ratio;
            let mut x2 = (data.get(r, 2) - lb.dw) / lb.ratio;
            let mut y2 = (data.get(r, 3) - lb.dh) / lb.ratio;

            // 边界检查
            x1 = x1.max(0.0).min(orig_width as f32);
            y1 = y1.max(0.0).min(orig_height as f32);
            x2 = x2.max(0.0).min(orig_width as f32);
            y2 = y2.max(0.0).min(orig_height as f32);

            detections.push(Detection::new(
                self.base.get_label_name(class_id),
                class_id,
                x1 as f64,
                y1 as f64,
                x2 as f64,
                y2 as f64,
                confidence as f64,
            ));
        }

        Ok(detections)
    }

    /// 标准模型后处理（需要 NMS）。
    ///
    /// 输出格式: `[batch, channels, anchors]` 或 `[batch, anchors, channels]`，
    /// 也支持 1D/2D 输出。
    fn postprocess_standard(
        &self,
        output_tensor: &TensorOutput,
        orig_width: usize,
        orig_height: usize,
        lb: LetterboxParams,
    ) -> Result<Vec<Detection>> {
        let flat = output_tensor.as_f32()?;
        let shape = &output_tensor.shape;

        tracing::debug!("Output tensor shape: {:?}, total elements: {}", shape, flat.len());

        // [channels, anchors] format for further processing
        let transposed = match shape.len() {
            3 => {
                // [batch, channels, anchors] - standard YOLOv8 format
                Matrix2D::from_flat(flat, shape[1] as usize, shape[2] as usize)?
            }
            2 => {
                // [channels, anchors] or [anchors, channels]
                Matrix2D::from_flat(flat, shape[0] as usize, shape[1] as usize)?
            }
            1 => {
                // 1D flat output - try to reshape to [channels, anchors]
                self.reshape_1d_output(flat)?
            }
            _ => {
                return Err(VisionError::inference(format!(
                    "Unsupported output tensor dimensions: {}, shape: {:?}",
                    shape.len(),
                    shape
                )))
            }
        };

        // 上游实现 transposed[0] 在 rows==0 时数组越界，对应 Rust 返回错误
        if transposed.rows == 0 {
            return Err(VisionError::inference("output tensor has zero channels"));
        }
        let num_classes = transposed.rows as isize - 4;
        let num_anchors = transposed.cols;

        // 收集所有候选框
        let mut candidates: Vec<[f32; 6]> = Vec::new();

        for i in 0..num_anchors {
            // 找出最大类别分数
            let mut max_score = 0f32;
            let mut best_class = -1i32;

            if num_classes > 0 {
                for c in 0..num_classes as usize {
                    let score = transposed.get(4 + c, i);
                    if score > max_score {
                        max_score = score;
                        best_class = c as i32;
                    }
                }
            }

            if max_score < self.base.confidence_threshold() {
                continue;
            }

            // 读取 bbox (xywh 格式)
            let cx = transposed.get(0, i);
            let cy = transposed.get(1, i);
            let w = transposed.get(2, i);
            let h = transposed.get(3, i);

            // 转换为 xyxy
            let x1 = cx - w / 2.0;
            let y1 = cy - h / 2.0;
            let x2 = cx + w / 2.0;
            let y2 = cy + h / 2.0;

            candidates.push([x1, y1, x2, y2, max_score, best_class as f32]);
        }

        // 执行 NMS
        self.nms(candidates, orig_width, orig_height, lb)
    }

    /// 非极大值抑制（类内 NMS，IoU 复用 `BoundingBox::iou`）。
    fn nms(
        &self,
        mut candidates: Vec<[f32; 6]>,
        orig_width: usize,
        orig_height: usize,
        lb: LetterboxParams,
    ) -> Result<Vec<Detection>> {
        let mut results = Vec::new();
        if candidates.is_empty() {
            return Ok(results);
        }

        // 按分数降序排序（稳定排序）
        candidates.sort_by(|a, b| b[4].partial_cmp(&a[4]).unwrap_or(std::cmp::Ordering::Equal));

        let mut suppressed = vec![false; candidates.len()];

        for i in 0..candidates.len() {
            if suppressed[i] {
                continue;
            }

            let curr = candidates[i];

            // 坐标还原
            let mut x1 = (curr[0] - lb.dw) / lb.ratio;
            let mut y1 = (curr[1] - lb.dh) / lb.ratio;
            let mut x2 = (curr[2] - lb.dw) / lb.ratio;
            let mut y2 = (curr[3] - lb.dh) / lb.ratio;

            // 边界检查
            x1 = x1.max(0.0).min(orig_width as f32);
            y1 = y1.max(0.0).min(orig_height as f32);
            x2 = x2.max(0.0).min(orig_width as f32);
            y2 = y2.max(0.0).min(orig_height as f32);

            let class_id = curr[5] as i32;
            results.push(Detection::new(
                self.base.get_label_name(class_id),
                class_id,
                x1 as f64,
                y1 as f64,
                x2 as f64,
                y2 as f64,
                curr[4] as f64,
            ));

            // 抑制重叠框（同类）
            let curr_box =
                BoundingBox::new(curr[0] as f64, curr[1] as f64, curr[2] as f64, curr[3] as f64, curr[4] as f64);
            for j in (i + 1)..candidates.len() {
                if suppressed[j] {
                    continue;
                }
                let cand = candidates[j];
                if cand[5] as i32 != class_id {
                    continue;
                }

                let cand_box = BoundingBox::new(
                    cand[0] as f64,
                    cand[1] as f64,
                    cand[2] as f64,
                    cand[3] as f64,
                    cand[4] as f64,
                );
                if curr_box.iou(&cand_box) > self.nms_threshold as f64 {
                    suppressed[j] = true;
                }
            }
        }

        Ok(results)
    }

    /// 将 1D flat 输出重塑为 [channels, anchors] 二维数组。
    ///
    /// channels = 4 + numClasses，优先从标签数量推断，否则尝试常见类别数。
    fn reshape_1d_output(&self, flat: &[f32]) -> Result<Matrix2D> {
        let total_elements = flat.len();

        // 从标签推断类别数
        let num_classes = self.base.labels().map(|l| l.len()).unwrap_or(0);

        if num_classes > 0 {
            let channels = 4 + num_classes;
            if total_elements % channels == 0 {
                let num_anchors = total_elements / channels;
                tracing::info!("Reshaping 1D output [{}] to [{}, {}]", total_elements, channels, num_anchors);
                return Ok(Matrix2D {
                    rows: channels,
                    cols: num_anchors,
                    data: flat.to_vec(),
                });
            }
            tracing::warn!(
                "Total elements {} not divisible by channels (4+{}={}), trying anchor-first layout",
                total_elements,
                num_classes,
                channels
            );

            // 尝试按 [numAnchors, channels] 解释再转置（上游实现中此分支实际不可达，保留语义）
            let num_anchors = total_elements / channels;
            if num_anchors * channels == total_elements {
                let mut data = vec![0f32; total_elements];
                let mut idx = 0;
                for a in 0..num_anchors {
                    for c in 0..channels {
                        data[c * num_anchors + a] = flat[idx];
                        idx += 1;
                    }
                }
                return Ok(Matrix2D {
                    rows: channels,
                    cols: num_anchors,
                    data,
                });
            }
        }

        // 无标签或无法推断 - 尝试常见类别数
        for try_classes in [1usize, 2, 3, 4, 5, 10, 20, 80] {
            let channels = 4 + try_classes;
            if total_elements % channels == 0 {
                let num_anchors = total_elements / channels;
                tracing::warn!(
                    "Guessing {} classes to reshape 1D output [{}] to [{}, {}]",
                    try_classes,
                    total_elements,
                    channels,
                    num_anchors
                );
                return Ok(Matrix2D {
                    rows: channels,
                    cols: num_anchors,
                    data: flat.to_vec(),
                });
            }
        }

        Err(VisionError::inference(format!(
            "Cannot reshape 1D output of {} elements. Model labels: {}",
            total_elements,
            match self.base.labels() {
                Some(l) => l.len().to_string(),
                None => "none".to_string(),
            }
        )))
    }

    // ==================== 批量后处理 ====================

    /// 批量后处理 - End2End。
    fn postprocess_batch_end2_end(
        &self,
        output_tensor: &TensorOutput,
        batch_size: usize,
        images: &[Image],
        params: &[LetterboxParams],
    ) -> Result<Vec<Vec<Detection>>> {
        let mut all_detections = Vec::with_capacity(batch_size);

        let flat = output_tensor.as_f32()?;
        let shape = &output_tensor.shape;

        // 形状可能是 [batch, num_det, 6] 或 [batch, channels, anchors]
        if shape.len() == 3 && shape[0] == batch_size as i64 {
            // [batch, num_det, 6] 格式
            let dim1 = shape[1] as usize;
            let dim2 = shape[2] as usize;
            let stride = dim1 * dim2;

            for b in 0..batch_size {
                let offset = b * stride;
                let lb = params[b];
                let orig_width = images[b].width();
                let orig_height = images[b].height();

                let mut batch_results = Vec::new();
                for i in 0..dim1 {
                    if i * 6 + 5 + offset >= flat.len() {
                        break;
                    }

                    let base_idx = offset + i * 6;
                    let x1 = flat[base_idx];
                    let y1 = flat[base_idx + 1];
                    let x2 = flat[base_idx + 2];
                    let y2 = flat[base_idx + 3];
                    let conf = flat[base_idx + 4];
                    let cls = flat[base_idx + 5];

                    if conf < self.base.confidence_threshold() {
                        continue;
                    }

                    // 坐标还原
                    let x1 = ((x1 - lb.dw) / lb.ratio).max(0.0).min(orig_width as f32);
                    let y1 = ((y1 - lb.dh) / lb.ratio).max(0.0).min(orig_height as f32);
                    let x2 = ((x2 - lb.dw) / lb.ratio).max(0.0).min(orig_width as f32);
                    let y2 = ((y2 - lb.dh) / lb.ratio).max(0.0).min(orig_height as f32);

                    let class_id = cls as i32;
                    batch_results.push(Detection::new(
                        self.base.get_label_name(class_id),
                        class_id,
                        x1 as f64,
                        y1 as f64,
                        x2 as f64,
                        y2 as f64,
                        conf as f64,
                    ));
                }
                all_detections.push(batch_results);
            }
        } else {
            // 回退到逐个处理（与原实现一致：复用实例字段 dw/dh/ratio，即最后一张图的 letterbox 参数）
            let lb = params[batch_size - 1];
            for b in 0..batch_size {
                all_detections.push(self.postprocess_end2_end(
                    output_tensor,
                    images[b].width(),
                    images[b].height(),
                    lb,
                )?);
            }
        }

        Ok(all_detections)
    }

    /// 批量后处理 - Standard。
    ///
    /// Standard 模式的批量处理涉及 NMS，暂时回退到逐个处理
    /// （与原实现一致：复用实例字段 dw/dh/ratio，即最后一张图的 letterbox 参数）。
    fn postprocess_batch_standard(
        &self,
        output_tensor: &TensorOutput,
        batch_size: usize,
        images: &[Image],
        params: &[LetterboxParams],
    ) -> Result<Vec<Vec<Detection>>> {
        let mut all_detections = Vec::with_capacity(batch_size);
        let lb = params[batch_size - 1];

        for b in 0..batch_size {
            all_detections.push(self.postprocess_standard(
                output_tensor,
                images[b].width(),
                images[b].height(),
                lb,
            )?);
        }

        Ok(all_detections)
    }

    // ==================== Transformer 后处理（RT-DETR / DETR / RF-DETR） ====================

    /// Transformer 系列后处理，按输出张量布局分派：
    /// - 单输出 `[1, Q, 4+nc]`：RT-DETR / DETR 风格，每 query 取最高类别分
    /// - 双输出 `dets=[1, Q, 4] + labels=[1, Q, nc]`：RF-DETR 风格，
    ///   logits 过 sigmoid 后做 Q×C 扁平 top-k（同一 query 可命中多个类别）
    ///
    /// 框格式自动识别：归一化 cxcywh（Ultralytics RT-DETR / RF-DETR 导出）或
    /// 输入分辨率像素 xyxy（部分旧版导出），通过坐标范围区分。
    /// 分数若为 logits（存在负值）自动过 sigmoid。
    fn postprocess_transformer(
        &self,
        outputs: &[TensorOutput],
        orig_width: usize,
        orig_height: usize,
    ) -> Result<Vec<Detection>> {
        let n = outputs.len();
        let shapes: Vec<&[i64]> = outputs.iter().map(|o| o.shape.as_slice()).collect();

        // 1. 定位框输出与分数输出：优先按名字（RF-DETR: dets/labels），再按 shape 兜底
        let mut boxes_idx: isize = -1;
        let mut scores_idx: isize = -1;
        for i in 0..n {
            if shapes[i].len() != 3 {
                continue;
            }
            let name = self
                .base
                .output_names()
                .get(i)
                .map(|s| s.to_lowercase())
                .unwrap_or_default();
            if name.contains("det") {
                boxes_idx = i as isize;
            } else if name.contains("label") || name.contains("score") || name.contains("logit") {
                scores_idx = i as isize;
            }
        }
        let mut combined = false;
        if boxes_idx < 0 && scores_idx < 0 {
            // 单输出 [1, Q, 4+nc]
            for i in 0..n {
                if shapes[i].len() == 3 && shapes[i][2] > 4 {
                    boxes_idx = i as isize;
                    combined = true;
                    break;
                }
            }
        } else {
            if boxes_idx < 0 {
                for i in 0..n {
                    if shapes[i].len() == 3 && shapes[i][2] == 4 {
                        boxes_idx = i as isize;
                        break;
                    }
                }
            }
            if scores_idx < 0 {
                for i in 0..n {
                    if i as isize == boxes_idx || shapes[i].len() != 3 {
                        continue;
                    }
                    scores_idx = i as isize;
                    break;
                }
            }
        }
        if boxes_idx < 0 || (!combined && scores_idx < 0) {
            return Err(VisionError::inference(format!(
                "Cannot locate detection/score outputs for Transformer model. Shapes: {:?}",
                shapes
            )));
        }

        // 2. 读取 [Q, 4] 框与 [Q, nc] 分数
        let mut boxes: Vec<[f32; 4]>;
        let mut scores: Vec<Vec<f32>>;
        if combined {
            let shape = shapes[boxes_idx as usize];
            let flat = outputs[boxes_idx as usize].as_f32()?;
            let q = shape[1] as usize;
            let v = shape[2] as usize;
            let nc = v - 4;
            boxes = Vec::with_capacity(q);
            scores = Vec::with_capacity(q);
            for i in 0..q {
                let row_off = i * v;
                boxes.push([flat[row_off], flat[row_off + 1], flat[row_off + 2], flat[row_off + 3]]);
                scores.push(flat[row_off + 4..row_off + 4 + nc].to_vec());
            }
        } else {
            let bm = Self::read_rank_as_2d(&outputs[boxes_idx as usize])?;
            let sm = Self::read_rank_as_2d(&outputs[scores_idx as usize])?;
            boxes = (0..bm.rows)
                .map(|r| [bm.get(r, 0), bm.get(r, 1), bm.get(r, 2), bm.get(r, 3)])
                .collect();
            scores = (0..sm.rows)
                .map(|r| (0..sm.cols).map(|c| sm.get(r, c)).collect())
                .collect();
        }

        // 3. 分数归一化：logits（有负值）→ sigmoid；已 sigmoid 的保持不变
        let mut min_score = f32::MAX;
        let mut max_score = -f32::MAX;
        for row in &scores {
            for &s in row {
                if s < min_score {
                    min_score = s;
                }
                if s > max_score {
                    max_score = s;
                }
            }
        }
        let need_sigmoid = min_score < 0.0 || max_score > 1.0 + 1e-3;
        if need_sigmoid {
            for row in &mut scores {
                for s in row.iter_mut() {
                    *s = sigmoid(*s);
                }
            }
        }

        // 4. 框格式：归一化 cxcywh（值 ≤ ~1.x）或输入分辨率像素 xyxy（部分旧版导出）
        let mut max_coord = 0f32;
        for b in &boxes {
            for &c in b {
                if c > max_coord {
                    max_coord = c;
                }
            }
        }
        let pixel_boxes = max_coord > 2.0;
        let scale_x = if pixel_boxes {
            orig_width as f32 / self.base.input_width() as f32
        } else {
            orig_width as f32
        };
        let scale_y = if pixel_boxes {
            orig_height as f32 / self.base.input_height() as f32
        } else {
            orig_height as f32
        };

        // 5. 候选生成：RT-DETR/DETR 每 query 一个候选（互斥 argmax）；RF-DETR Q×C 扁平（多标签）
        let num_queries = boxes.len();
        let num_classes = scores.first().map(|r| r.len()).unwrap_or(0);
        let mut candidates: Vec<[f32; 6]> = Vec::new(); // {x1,y1,x2,y2, conf, clsId}

        if self.model_type == DetectionModelType::RfDetr {
            for q in 0..num_queries {
                for c in 0..num_classes {
                    let conf = scores[q][c];
                    if conf <= self.base.confidence_threshold() {
                        continue;
                    }
                    candidates.push(Self::make_candidate(
                        &boxes[q],
                        conf,
                        c as f32,
                        scale_x,
                        scale_y,
                        orig_width,
                        orig_height,
                    ));
                }
            }
        } else {
            for q in 0..num_queries {
                let mut best = 0f32;
                let mut best_cls = -1.0f32;
                for c in 0..num_classes {
                    if scores[q][c] > best {
                        best = scores[q][c];
                        best_cls = c as f32;
                    }
                }
                if best <= self.base.confidence_threshold() {
                    continue;
                }
                candidates.push(Self::make_candidate(
                    &boxes[q],
                    best,
                    best_cls,
                    scale_x,
                    scale_y,
                    orig_width,
                    orig_height,
                ));
            }
        }

        // 6. 按分数降序（稳定排序，等分保持 index 顺序，与官方实现一致）+ top-k 截断
        candidates.sort_by(|a, b| b[4].partial_cmp(&a[4]).unwrap_or(std::cmp::Ordering::Equal));
        let mut results = Vec::with_capacity(self.max_detections.min(candidates.len()));
        for cand in &candidates {
            if results.len() >= self.max_detections {
                break;
            }
            let cls_id = cand[5] as i32;
            results.push(Detection::new(
                self.base.get_label_name(cls_id),
                cls_id,
                cand[0] as f64,
                cand[1] as f64,
                cand[2] as f64,
                cand[3] as f64,
                cand[4] as f64,
            ));
        }
        Ok(results)
    }

    /// 归一化 cxcywh → xyxy → 原图像素坐标（clamp 到图内）。
    fn make_candidate(
        b: &[f32; 4],
        conf: f32,
        cls_id: f32,
        scale_x: f32,
        scale_y: f32,
        orig_width: usize,
        orig_height: usize,
    ) -> [f32; 6] {
        let (cx, cy, w, h) = (b[0], b[1], b[2], b[3]);
        let x1 = ((cx - w / 2.0) * scale_x).max(0.0);
        let y1 = ((cy - h / 2.0) * scale_y).max(0.0);
        let x2 = ((cx + w / 2.0) * scale_x).min(orig_width as f32);
        let y2 = ((cy + h / 2.0) * scale_y).min(orig_height as f32);
        [x1, y1, x2, y2, conf, cls_id]
    }

    /// 将 [1, Q, D] 或 [Q, D] 张量读为 [Q][D] 二维矩阵。
    fn read_rank_as_2d(tensor: &TensorOutput) -> Result<Matrix2D> {
        let flat = tensor.as_f32()?;
        let shape = &tensor.shape;
        let (rows, cols) = match shape.len() {
            3 => (shape[1] as usize, shape[2] as usize),
            2 => (shape[0] as usize, shape[1] as usize),
            _ => {
                return Err(VisionError::inference(format!(
                    "Expected rank-2/3 tensor, got rank-{}, shape: {:?}",
                    shape.len(),
                    shape
                )))
            }
        };
        Matrix2D::from_flat(flat, rows, cols)
    }

    // ==================== 输出定位与自动检测 ====================

    /// 从多个输出中找到检测输出张量。
    ///
    /// 跳过 shape=[1] 等小维度元数据输出，选择元素数最多的输出。
    fn find_detection_output(outputs: &[TensorOutput]) -> Result<&TensorOutput> {
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
            "Selected output tensor with shape={:?}, elements={}",
            best.shape,
            best_count
        );

        Ok(best)
    }

    /// 获取（必要时首次计算）End2End 自动检测结果。
    fn resolve_end2_end(&self, output_tensor: &TensorOutput) -> Result<bool> {
        let mut cache = self.auto_detected_end2_end.lock().unwrap();
        if cache.is_none() {
            *cache = Some(Self::auto_detect_end2_end(&output_tensor.shape, self.end2_end));
        }
        Ok(cache.unwrap())
    }

    /// 首次推理时自动检测模型类型（基于输出 tensor 的实际 shape）。
    ///
    /// - End2End (YOLO26): `[1, ~300, 6]` — dim1 较小（100~400），dim2 = 4+1+1 = 6
    /// - Traditional (YOLOv8/v11): `[1, ~84, ~8400]` — dim1 是 channels，dim2 是 anchors（较大）
    fn auto_detect_end2_end(shape: &[i64], constructor_end2end: bool) -> bool {
        if shape.len() != 3 {
            // 非 3D 输出默认当 Traditional 处理
            tracing::info!(
                "Output shape {:?} is not 3D, using constructor end2End={}",
                shape,
                constructor_end2end
            );
            return constructor_end2end;
        }

        let dim1 = shape[1]; // channels 或 num_detections
        let dim2 = shape[2]; // anchors 或 per_det_size

        // End2End: dim1 是检测数量（如 300），dim2 是属性数（如 6 = 4+1+1）
        // Traditional: dim1 是 channels（如 84），dim2 是 anchors（如 8400，较大）
        let is_end2end = (100..=400).contains(&dim1) && (4..=100).contains(&dim2);

        tracing::info!(
            "Auto-detected model type: {} (shape=[{},{},{}], dim1={}, dim2={})",
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

    /// 打印检测结果日志。
    fn log_results(&self, detections: &[Detection]) {
        tracing::info!(
            "Detected {} objects{}",
            detections.len(),
            if detections.is_empty() { "" } else { ":" }
        );
        for d in detections {
            tracing::info!("  {}", d);
        }
    }
}

/// sigmoid（对应 `DetectionEngine.sigmoid`）。
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

crate::impl_engine_forward!(DetectionEngine, base, Vec<Detection>,
    /// 单图推理（SAHI 开启时自动切片合并，返回类型不变）。
    fn predict(&self, image: &Image) -> Result<Vec<Detection>> {
        self.predict_impl(image)
    },
    /// 批量推理（YOLO 动态 batch 走真实批量路径；Transformer / SAHI 自动退化为逐张推理）。
    fn predict_batch(&self, images: &[Image]) -> Result<Vec<Vec<Detection>>> {
        self.predict_batch_impl(images)
    }
);
