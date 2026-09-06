//! 图像分类推理引擎。
//!
//! 支持多种 ONNX 分类模型：
//! - YOLO 分类系列：yolov8n-cls.onnx、yolov11n-cls.onnx 等（pixel/255）
//! - ImageNet 预训练模型：ResNet、MobileNet、EfficientNet、ViT 等
//! - 自定义归一化模型
//!
//! 与 原版的差异约定：
//! - `predict` 低于阈值返回 `None`
//!   （trait 输出类型为 `Option<ClassificationResult>`）
//! - `predict_batch` 列表中对应 `None`
//! - `predict_top_k` 在 `predict` 返回 `None` 时返回空列表，
//!   Rust 侧对应情形（低于阈值）返回空列表

use crate::core::base::{BaseOnnxEngine, TensorOutput};
use crate::core::device_type::DeviceType;
use crate::error::{Result, VisionError};
use crate::imaging::Image;
use crate::model::ClassificationResult;

/// 分类模型类型（对应 `ClassificationModelType`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClassificationModelType {
    /// YOLO 分类模型 (pixel/255)
    Yolo,
    /// ImageNet 预训练模型 ((pixel/255 - mean) / std)
    ImageNet,
    /// 自定义归一化
    Custom,
    /// 自动检测归一化方式
    #[default]
    Auto,
}

/// 图像分类推理引擎。
pub struct ClassificationEngine {
    /// 组合基类
    pub base: BaseOnnxEngine,
    /// 模型类型
    model_type: ClassificationModelType,
}

impl ClassificationEngine {
    /// 创建 YOLO 分类引擎。
    pub fn for_yolo(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
    ) -> Result<Self> {
        Self::for_yolo_with_input_size(model_path, device_type, -1, -1)
    }

    /// 创建 YOLO 分类引擎（指定输入尺寸）。
    pub fn for_yolo_with_input_size(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        input_height: i32,
        input_width: i32,
    ) -> Result<Self> {
        let mut engine = Self::new_with_input_size(
            model_path,
            device_type,
            ClassificationModelType::Yolo,
            input_height,
            input_width,
        )?;
        engine
            .base
            .set_normalization([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        Ok(engine)
    }

    /// 创建 ImageNet 分类引擎。
    pub fn for_image_net(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
    ) -> Result<Self> {
        Self::for_image_net_with_input_size(model_path, device_type, -1, -1)
    }

    /// 创建 ImageNet 分类引擎（指定输入尺寸）。
    pub fn for_image_net_with_input_size(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        input_height: i32,
        input_width: i32,
    ) -> Result<Self> {
        let mut engine = Self::new_with_input_size(
            model_path,
            device_type,
            ClassificationModelType::ImageNet,
            input_height,
            input_width,
        )?;
        engine.base.set_normalization(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        );
        Ok(engine)
    }

    /// 创建自动检测分类引擎。
    ///
    /// 使用安全的默认归一化（pixel/255），适用于大多数 ONNX 分类模型。
    /// 如需 ImageNet 归一化，请使用 [`ClassificationEngine::for_image_net`]。
    pub fn for_auto(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
    ) -> Result<Self> {
        Self::for_auto_with_input_size(model_path, device_type, -1, -1)
    }

    /// 创建自动检测分类引擎（指定输入尺寸）。
    pub fn for_auto_with_input_size(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        input_height: i32,
        input_width: i32,
    ) -> Result<Self> {
        Self::new_with_input_size(
            model_path,
            device_type,
            ClassificationModelType::Auto,
            input_height,
            input_width,
        )
    }

    /// 创建自定义归一化分类引擎。
    pub fn for_custom(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        mean: [f32; 3],
        std: [f32; 3],
    ) -> Result<Self> {
        Self::for_custom_with_input_size(model_path, device_type, -1, -1, mean, std)
    }

    /// 创建自定义归一化分类引擎（指定输入尺寸）。
    pub fn for_custom_with_input_size(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        input_height: i32,
        input_width: i32,
        mean: [f32; 3],
        std: [f32; 3],
    ) -> Result<Self> {
        let mut engine = Self::new_with_input_size(
            model_path,
            device_type,
            ClassificationModelType::Custom,
            input_height,
            input_width,
        )?;
        engine.base.set_normalization(mean, std);
        Ok(engine)
    }

    /// 创建分类引擎（默认输入尺寸从模型读取）。
    pub fn new(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        model_type: ClassificationModelType,
    ) -> Result<Self> {
        Self::new_with_input_size(model_path, device_type, model_type, -1, -1)
    }

    /// 创建分类引擎（指定输入尺寸；<=0 时从模型读取）。
    pub fn new_with_input_size(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        model_type: ClassificationModelType,
        input_height: i32,
        input_width: i32,
    ) -> Result<Self> {
        let base = BaseOnnxEngine::with_input_size(model_path, device_type, input_height, input_width)?;
        // AUTO 模式：使用安全的默认归一化（pixel/255）
        if model_type == ClassificationModelType::Auto {
            tracing::info!("AUTO mode: using default normalization (pixel/255)");
        }
        Ok(ClassificationEngine { base, model_type })
    }

    /// 获取模型类型。
    pub fn model_type(&self) -> ClassificationModelType {
        self.model_type
    }

    /// 单图推理核心实现（trait `predict` 转发到这里）。
    pub fn predict_impl(&self, image: &Image) -> Result<Option<ClassificationResult>> {
        // 1. 预处理
        let input_data = self.base.preprocess(image)?;

        // 2. 创建 Tensor
        let input_tensor = self.base.create_input_tensor(input_data)?;

        // 3. 运行推理
        let output_tensor = self.base.run_inference(input_tensor)?;

        // 4. 后处理
        let classification = self.postprocess(&output_tensor)?;

        match &classification {
            Some(r) => tracing::info!("{}", r),
            None => tracing::info!(
                "Classification: no result above threshold ({})",
                self.base.confidence_threshold()
            ),
        }

        Ok(classification)
    }

    /// 批量推理核心实现（trait `predict_batch` 转发到这里）。
    pub fn predict_batch_impl(&self, images: &[Image]) -> Result<Vec<Option<ClassificationResult>>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        // 1. 批量预处理
        let batch_data = self.base.preprocess_batch(images)?;

        // 2. 创建批量 Tensor
        let input_tensor = self.base.create_batch_input_tensor(batch_data, images.len())?;

        // 3. 运行推理
        let output_tensor = self.base.run_inference(input_tensor)?;

        // 4. 批量后处理
        self.postprocess_batch(&output_tensor, images.len())
    }

    /// 预测 Top-K 结果。
    ///
    /// 原版在 `predict` 返回 `null`（低于阈值）时会 NPE，此处对应返回空列表；
    /// 结果不含全部分数（`all_scores == None`）时退化为单元素列表。
    pub fn predict_top_k(&self, image: &Image, k: usize) -> Result<Vec<ClassificationResult>> {
        let Some(result) = self.predict_impl(image)? else {
            return Ok(Vec::new());
        };

        let Some(all_scores) = result.all_scores.as_deref() else {
            return Ok(vec![result]);
        };

        Ok(ClassificationResult::from_scores(
            all_scores,
            self.base.labels(),
            k,
        ))
    }

    /// 后处理单图结果（低于置信度阈值返回 `None`，原实现返回 `null`）。
    fn postprocess(&self, output_tensor: &TensorOutput) -> Result<Option<ClassificationResult>> {
        let scores = output_tensor.as_f32()?;
        if scores.is_empty() {
            // 上游实现此处会数组越界，对应 Rust 返回错误
            return Err(VisionError::inference("classification output tensor is empty"));
        }

        // 找出最大分数（严格大于，同分取最先出现，与原实现一致）
        let mut max_idx = 0usize;
        let mut max_score = scores[0];
        for (i, &s) in scores.iter().enumerate().skip(1) {
            if s > max_score {
                max_score = s;
                max_idx = i;
            }
        }

        // 置信度过滤
        if max_score < self.base.confidence_threshold() {
            return Ok(None);
        }

        let mut result = ClassificationResult::new(
            self.base.get_label_name(max_idx as i32),
            max_idx,
            max_score as f64,
        );
        result.all_scores = Some(scores.to_vec());
        Ok(Some(result))
    }

    /// 批量后处理（低于阈值的位置为 `None`，对应上游 列表中的 `null`）。
    fn postprocess_batch(
        &self,
        output_tensor: &TensorOutput,
        batch_size: usize,
    ) -> Result<Vec<Option<ClassificationResult>>> {
        let output = output_tensor.as_f32()?;

        // 假设输出形状为 [batch, num_classes]
        let num_classes = if output_tensor.shape.len() >= 2 {
            output_tensor.shape[1] as usize
        } else {
            output.len() / batch_size
        };
        if num_classes == 0 {
            return Err(VisionError::inference("classification output has zero classes"));
        }

        let mut results = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let start = b * num_classes;
            let end = start + num_classes;
            if end > output.len() {
                // 上游此处会数组越界
                return Err(VisionError::inference(format!(
                    "classification output size {} < batch {} x num_classes {}",
                    output.len(),
                    b + 1,
                    num_classes
                )));
            }
            let scores = &output[start..end];

            let mut max_idx = 0usize;
            let mut max_score = scores[0];
            for (i, &s) in scores.iter().enumerate().skip(1) {
                if s > max_score {
                    max_score = s;
                    max_idx = i;
                }
            }

            if max_score >= self.base.confidence_threshold() {
                let mut r = ClassificationResult::new(
                    self.base.get_label_name(max_idx as i32),
                    max_idx,
                    max_score as f64,
                );
                r.all_scores = Some(scores.to_vec());
                results.push(Some(r));
            } else {
                results.push(None);
            }
        }

        Ok(results)
    }
}

crate::impl_engine_forward!(ClassificationEngine, base, Option<ClassificationResult>,
    /// 单图推理（低于置信度阈值返回 `None`，原实现返回 `null`）。
    fn predict(&self, image: &Image) -> Result<Option<ClassificationResult>> {
        self.predict_impl(image)
    },
    /// 批量推理（真实批量路径；低于阈值的位置为 `None`，对应上游 列表中的 `null`）。
    fn predict_batch(&self, images: &[Image]) -> Result<Vec<Option<ClassificationResult>>> {
        self.predict_batch_impl(images)
    }
);
