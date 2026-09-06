//! ONNX 推理引擎基类。
//!
//! 封装与具体任务无关的通用能力，具体引擎只需实现业务前/后处理：
//! - 会话管理：经 [`crate::core::session_factory`] 创建（支持 CPU/CUDA/CoreML 与线程配置）、RAII 释放
//! - 输入输出元信息：输入名、输出名、输入尺寸（动态维度回退默认 640）
//! - 标签：自动解析模型元数据（names/labels/categories，支持 YOLO 格式与逗号分隔）
//! - 通用预处理：拉伸 resize + BGR→RGB + (x/255 - mean)/std（HWC → CHW）
//! - 张量辅助：输入/批量张量创建、多输出推理、float/int64 输出读取与形状获取
//! - SAHI 切片推理配置

use std::sync::Mutex;

use ort::session::Session;
use ort::value::Tensor;

use crate::core::device_type::DeviceType;
use crate::core::runtime_config::OnnxRuntimeConfig;
use crate::core::session_factory::create_session_builder_with;
use crate::error::{Result, VisionError};
use crate::imaging::{resize, ColorConversion, Image, Interpolation};
use crate::sahi::config::SahiConfig;

/// 输出张量的统一快照（复制出 native 内存，避免生命周期问题；对齐 上游实现
/// "取出 Value 引用并复用" 的约定，Rust 侧直接拷贝更安全）。
#[derive(Debug, Clone)]
pub struct TensorOutput {
    pub name: String,
    pub shape: Vec<i64>,
    pub data: TensorData,
}

/// 输出张量数据（f32 / i64；其他类型以错误形式记录）。
#[derive(Debug, Clone)]
pub enum TensorData {
    F32(Vec<f32>),
    I64(Vec<i64>),
    Unsupported(String),
}

impl TensorData {
    /// 取 f32 数据（类型不符时报错）。
    pub fn as_f32(&self) -> Result<&[f32]> {
        match self {
            TensorData::F32(v) => Ok(v),
            TensorData::I64(_) => Err(VisionError::inference("tensor is i64, expected f32")),
            TensorData::Unsupported(t) => Err(VisionError::inference(format!(
                "unsupported tensor element type: {t}"
            ))),
        }
    }

    /// 取 i64 数据（类型不符时报错）。
    pub fn as_i64(&self) -> Result<&[i64]> {
        match self {
            TensorData::I64(v) => Ok(v),
            TensorData::F32(_) => Err(VisionError::inference("tensor is f32, expected i64")),
            TensorData::Unsupported(t) => Err(VisionError::inference(format!(
                "unsupported tensor element type: {t}"
            ))),
        }
    }

    /// 元素总数。
    pub fn element_count(&self) -> usize {
        match self {
            TensorData::F32(v) => v.len(),
            TensorData::I64(v) => v.len(),
            TensorData::Unsupported(_) => 0,
        }
    }
}

impl TensorOutput {
    /// 元素总数。
    pub fn element_count(&self) -> usize {
        self.data.element_count()
    }

    /// 取 f32 数据。
    pub fn as_f32(&self) -> Result<&[f32]> {
        self.data.as_f32()
    }

    /// 取 i64 数据。
    pub fn as_i64(&self) -> Result<&[i64]> {
        self.data.as_i64()
    }

    /// 按 shape 取二维 [dim0][dim1] 视图所需的 dim1（对应 `getFloatArray2D` 的语义）。
    pub fn dim1(&self) -> usize {
        if self.shape.len() >= 2 {
            self.shape[1] as usize
        } else {
            self.data.element_count() / self.shape.first().copied().unwrap_or(1).max(1) as usize
        }
    }
}

/// ONNX 推理引擎基类。
pub struct BaseOnnxEngine {
    pub(crate) session: Mutex<Session>,
    pub(crate) input_name: String,
    pub(crate) output_names: Vec<String>,
    pub(crate) labels: Option<Vec<String>>,

    pub(crate) input_height: i32,
    pub(crate) input_width: i32,
    pub(crate) input_channels: i32,

    pub(crate) confidence_threshold: f32,

    pub(crate) mean: [f32; 3],
    pub(crate) std: [f32; 3],
    pub(crate) normalize: bool,

    pub(crate) device_type: DeviceType,
    pub(crate) runtime_config: OnnxRuntimeConfig,

    pub(crate) sahi_config: Option<SahiConfig>,
}

impl BaseOnnxEngine {
    /// 创建基础引擎（输入尺寸从模型读取，动态维度回退 640）。
    pub fn new(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        Self::with_config(model_path, device_type, -1, -1, OnnxRuntimeConfig::defaults())
    }

    /// 指定输入尺寸创建（<=0 时从模型读取）。
    pub fn with_input_size(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        input_height: i32,
        input_width: i32,
    ) -> Result<Self> {
        Self::with_config(model_path, device_type, input_height, input_width, OnnxRuntimeConfig::defaults())
    }

    /// 指定输入尺寸与运行参数创建。
    pub fn with_config(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        input_height: i32,
        input_width: i32,
        runtime_config: OnnxRuntimeConfig,
    ) -> Result<Self> {
        let model_path = model_path.as_ref();
        let mut input_height_v = 640;
        let mut input_width_v = 640;
        if input_height > 0 {
            input_height_v = input_height;
        }
        if input_width > 0 {
            input_width_v = input_width;
        }

        let session = create_session_builder_with(device_type, runtime_config, Default::default())?
            .commit(model_path)?;

        let mut base = BaseOnnxEngine {
            input_name: session
                .inputs
                .first()
                .map(|i| i.name.clone())
                .unwrap_or_else(|| "images".to_string()),
            output_names: session.outputs.iter().map(|o| o.name.clone()).collect(),
            labels: None,
            input_height: input_height_v,
            input_width: input_width_v,
            input_channels: 3,
            confidence_threshold: 0.5,
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
            normalize: true,
            device_type,
            runtime_config,
            sahi_config: None,
            session: Mutex::new(session),
        };

        base.load_input_info();
        base.load_labels();
        tracing::info!(
            "ONNX Engine initialized: model={}, input={}x{}, device={}, config={}",
            model_path.display(),
            base.input_width,
            base.input_height,
            device_type.name(),
            base.runtime_config
        );
        Ok(base)
    }

    /// 从会话元信息解析输入尺寸（NCHW；动态维度警告并保留默认值）。
    fn load_input_info(&mut self) {
        let session = self.session.lock().unwrap();
        let Some(input) = session.inputs.first() else {
            return;
        };
        if let Some(name) = session.inputs.first().map(|i| i.name.clone()) {
            self.input_name = name;
        }
        if let ort::value::ValueType::Tensor { shape, .. } = &input.input_type {
            let dims: Vec<i64> = shape.iter().copied().collect();
            if dims.len() >= 4 {
                if dims[1] > 0 {
                    self.input_channels = dims[1] as i32;
                }
                if dims[2] > 0 {
                    self.input_height = dims[2] as i32;
                }
                if dims[3] > 0 {
                    self.input_width = dims[3] as i32;
                }
                tracing::info!(
                    "Model input: name='{}', shape={:?} -> parsed inputSize={}x{}",
                    self.input_name,
                    dims,
                    self.input_width,
                    self.input_height
                );
                if dims[2] < 0 || dims[3] < 0 {
                    tracing::warn!(
                        "Model has dynamic input H/W dimensions. Using default {}x{}. If wrong, set input size explicitly.",
                        self.input_width,
                        self.input_height
                    );
                }
            }
        }
    }

    /// 从模型元数据加载标签（keys: names/labels/categories）。
    fn load_labels(&mut self) {
        let guard = self.session.lock().unwrap();
        let meta = match guard.metadata() {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!("Failed to load labels from metadata: {}", e);
                return;
            }
        };
        for key in ["names", "labels", "categories"] {
            let value = match meta.custom(key) {
                Ok(Some(v)) if !v.is_empty() => v,
                _ => continue,
            };
            let labels = parse_labels(&value);
            tracing::info!("Loaded {} labels from metadata key '{}'", labels.len(), key);
            self.labels = Some(labels);
            return;
        }
        tracing::debug!("No labels found in model metadata, labels will need to be set manually");
    }

    // ============ 访问器 ============

    pub fn input_name(&self) -> &str {
        &self.input_name
    }

    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    pub fn input_height(&self) -> i32 {
        self.input_height
    }

    pub fn input_width(&self) -> i32 {
        self.input_width
    }

    pub fn input_channels(&self) -> i32 {
        self.input_channels
    }

    /// 输入尺寸 (width, height)。
    pub fn input_size(&self) -> (i32, i32) {
        (self.input_width, self.input_height)
    }

    pub fn labels(&self) -> Option<&[String]> {
        self.labels.as_deref()
    }

    /// 手动设置类别名。
    pub fn set_labels(&mut self, labels: Vec<String>) {
        self.labels = Some(labels);
    }

    pub fn set_confidence_threshold(&mut self, threshold: f32) {
        self.confidence_threshold = threshold;
    }

    pub fn confidence_threshold(&self) -> f32 {
        self.confidence_threshold
    }

    pub fn device_type(&self) -> DeviceType {
        self.device_type
    }

    pub fn runtime_config(&self) -> &OnnxRuntimeConfig {
        &self.runtime_config
    }

    /// 设置归一化参数。
    pub fn set_normalization(&mut self, mean: [f32; 3], std: [f32; 3]) {
        self.mean = mean;
        self.std = std;
    }

    /// 设置是否归一化。
    pub fn set_normalize(&mut self, normalize: bool) {
        self.normalize = normalize;
    }

    // ============ SAHI ============

    /// 设置 SAHI 切片推理配置；None 等价于 disable。
    pub fn set_sahi_config(&mut self, config: Option<SahiConfig>) {
        self.sahi_config = config;
    }

    /// 关闭 SAHI。
    pub fn disable_sahi(&mut self) {
        self.sahi_config = None;
    }

    /// SAHI 是否已开启。
    pub fn is_sahi_enabled(&self) -> bool {
        self.sahi_config.is_some()
    }

    pub fn sahi_config(&self) -> Option<&SahiConfig> {
        self.sahi_config.as_ref()
    }

    /// SAHI 开启时的批量推理退化路径：逐张走 predict。
    pub fn predict_batch_via_sahi<T, F>(&self, images: &[Image], mut f: F) -> Result<Vec<T>>
    where
        F: FnMut(&Image) -> Result<T>,
    {
        images.iter().map(|img| f(img)).collect()
    }

    // ============ 预处理 ============

    /// 通用预处理：拉伸 resize + BGR/BGRA/GRAY→RGB + (x/255 - mean)/std，HWC → CHW。
    pub fn preprocess(&self, image: &Image) -> Result<Vec<f32>> {
        let width = self.input_width as usize;
        let height = self.input_height as usize;
        let channels = self.input_channels as usize;

        // 1. Resize（拉伸）
        let resized = resize(image, width, height, Interpolation::Linear)?;

        // 2. 颜色空间转换 → RGB
        let rgb = match resized.channels() {
            4 => crate::imaging::cvt_color(&resized, ColorConversion::Bgra2Rgb)?,
            3 => crate::imaging::cvt_color(&resized, ColorConversion::Bgr2Rgb)?,
            _ => crate::imaging::cvt_color(&resized, ColorConversion::Gray2Rgb)?,
        };

        // 3. 归一化 + HWC → CHW
        let area = height * width;
        let mut float_data = vec![0f32; channels * area];
        let px = rgb.data();
        for i in 0..area {
            let r = px[i * 3] as f32 / 255.0;
            let g = px[i * 3 + 1] as f32 / 255.0;
            let b = px[i * 3 + 2] as f32 / 255.0;
            float_data[i] = if self.normalize {
                (r - self.mean[0]) / self.std[0]
            } else {
                r
            };
            float_data[i + area] = if self.normalize {
                (g - self.mean[1]) / self.std[1]
            } else {
                g
            };
            float_data[i + 2 * area] = if self.normalize {
                (b - self.mean[2]) / self.std[2]
            } else {
                b
            };
        }
        Ok(float_data)
    }

    /// 批量预处理。
    pub fn preprocess_batch(&self, images: &[Image]) -> Result<Vec<f32>> {
        let single = self.input_channels as usize * self.input_height as usize * self.input_width as usize;
        let mut batch = Vec::with_capacity(images.len() * single);
        for img in images {
            batch.extend_from_slice(&self.preprocess(img)?);
        }
        Ok(batch)
    }

    // ============ 张量辅助 ============

    /// 创建输入 Tensor（shape = [1, C, H, W]）。
    pub fn create_input_tensor(&self, data: Vec<f32>) -> Result<Tensor<f32>> {
        self.create_batch_input_tensor(data, 1)
    }

    /// 创建批量输入 Tensor（shape = [N, C, H, W]）。
    pub fn create_batch_input_tensor(&self, data: Vec<f32>, batch_size: usize) -> Result<Tensor<f32>> {
        let shape = vec![
            batch_size as i64,
            self.input_channels as i64,
            self.input_height as i64,
            self.input_width as i64,
        ];
        Ok(Tensor::from_array((shape, data))?)
    }

    /// 运行推理，返回全部输出的快照（对应 `runInferenceMultiOutput`）。
    pub fn run_multi_output(&self, input_tensor: Tensor<f32>) -> Result<Vec<TensorOutput>> {
        let mut session = self.session.lock().unwrap();
        let outputs = session.run(ort::inputs![self.input_name.as_str() => input_tensor])?;

        let mut result = Vec::with_capacity(self.output_names.len());
        for name in &self.output_names {
            let value = outputs
                .get(name.as_str())
                .ok_or_else(|| VisionError::inference(format!("missing output '{name}'")))?;
            result.push(snapshot_output(name, value)?);
        }
        Ok(result)
    }

    /// 运行推理并返回第一个输出（对应 `runInference`）。
    pub fn run_inference(&self, input_tensor: Tensor<f32>) -> Result<TensorOutput> {
        let mut outs = self.run_multi_output(input_tensor)?;
        outs.pop()
            .ok_or_else(|| VisionError::inference("model returned no outputs"))
    }

    /// 以任意命名输入运行推理（多输入模型，如 SAM / GroundingDINO / LoMaR）。
    pub fn run_named(&self, inputs: Vec<(&str, Tensor<f32>)>) -> Result<Vec<TensorOutput>> {
        let mut session = self.session.lock().unwrap();
        let outputs = session.run(inputs)?;

        let mut result = Vec::with_capacity(self.output_names.len());
        for name in &self.output_names {
            let value = outputs
                .get(name.as_str())
                .ok_or_else(|| VisionError::inference(format!("missing output '{name}'")))?;
            result.push(snapshot_output(name, value)?);
        }
        Ok(result)
    }

    /// 获取标签名称（越界时返回数字字符串）。
    pub fn get_label_name(&self, class_id: i32) -> String {
        if let Some(labels) = &self.labels {
            let idx = class_id;
            if idx >= 0 && (idx as usize) < labels.len() {
                return labels[idx as usize].clone();
            }
        }
        class_id.to_string()
    }

    /// 获取输入张量 shape（[1, C, H, W]）。
    pub fn input_tensor_shape(&self, batch: i64) -> Vec<i64> {
        vec![batch, self.input_channels as i64, self.input_height as i64, self.input_width as i64]
    }
}

/// 把 ort 输出 Value 复制为 owned 快照。
fn snapshot_output(name: &str, value: &ort::value::DynValue) -> Result<TensorOutput> {
    let (shape, ty) = match value.dtype() {
        ort::value::ValueType::Tensor { ty, shape, .. } => {
            (shape.iter().copied().collect::<Vec<i64>>(), ty)
        }
        other => {
            return Ok(TensorOutput {
                name: name.to_string(),
                shape: Vec::new(),
                data: TensorData::Unsupported(format!("{other:?}")),
            })
        }
    };

    let data = match ty {
        ort::tensor::TensorElementType::Float32 => {
            let (_, view) = value.try_extract_tensor::<f32>()?;
            TensorData::F32(view.to_vec())
        }
        // legacy f64 导出兼容：快照时转 f32（数值等价），下游统一按 f32 消费
        ort::tensor::TensorElementType::Float64 => {
            let (_, view) = value.try_extract_tensor::<f64>()?;
            TensorData::F32(view.iter().map(|&v| v as f32).collect())
        }
        ort::tensor::TensorElementType::Int64 => {
            let (_, view) = value.try_extract_tensor::<i64>()?;
            TensorData::I64(view.to_vec())
        }
        other => TensorData::Unsupported(format!("{other:?}")),
    };

    Ok(TensorOutput {
        name: name.to_string(),
        shape,
        data,
    })
}

/// 解析标签字符串：支持 YOLO 格式 `{0: 'person', 1: 'car', ...}` 与逗号分隔。
pub fn parse_labels(names_str: &str) -> Vec<String> {
    let mut label_list = Vec::new();
    let s = names_str.trim();
    if s.starts_with('{') {
        let inner = &s[1..s.len().saturating_sub(1)];
        // 提取所有 'xxx' 或 "xxx"
        let mut chars = inner.chars().peekable();
        while let Some(&c) = chars.peek() {
            if c == '\'' || c == '"' {
                let quote = c;
                chars.next();
                let mut label = String::new();
                for ch in chars.by_ref() {
                    if ch == quote {
                        break;
                    }
                    label.push(ch);
                }
                if !label.is_empty() {
                    label_list.push(label);
                }
            } else {
                chars.next();
            }
        }
    } else {
        for part in s.split(',') {
            label_list.push(part.trim().to_string());
        }
    }
    label_list
}
