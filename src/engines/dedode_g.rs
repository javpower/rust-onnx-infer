//! DeDoDe-G 关键点 + 描述子检测引擎。
//!
//! 适配 davnords/storage 提供的 LoMa-R 兼容 ONNX。
//!
//! 两个独立 ONNX：
//! 1. **detector**：输入 `image[1,3,H,W]` + `num_keypoints[1]` (int64)，
//!    输出 `keypoints[1,n,2]`（范围 [-1, 1]） + `keypoint_probs[1,n]`（logits，需 sigmoid）
//! 2. **descriptor**：输入 `image[1,3,784,784]` + `keypoints[1,n,2]`（范围 [-1, 1]），
//!    输出 `descriptions[1,n,256]`
//!
//! 统一用 letterbox resize 到 784×784（DINOv2-G 主干标准），
//! detector/descriptor 共享同一图像预处理（descriptor 必须 784，detector 用同尺寸避免重复 resize）。
//! detector 输出 keypoints 反归一化到 [-1,1] → 原图像素。
//!
//! 用法：
//! ```ignore
//! let det = DedodeGEngine::new(
//!     "models/dedode_g_detector.onnx",
//!     "models/dedode_g_descriptor.onnx",
//!     DeviceType::Cpu)?;
//! let r = det.detect(&img_a)?;
//! // r.keypoints : 原图像素坐标
//! // r.descriptors : [N][256]
//! ```
//!
//! 与 原版的实现差异（数值逻辑保持逐行一致）：
//! - descriptor 使用独立的 [`BaseOnnxEngine`]；
//! - 上游用 IoBinding + 显式生命周期管理防 native buffer 提前回收，
//!   Rust 侧 `session.run(ort::inputs![...])` 的所有权语义天然避免该问题；
//! - 上游非 letterbox 模式下对小于 784 的图会因缓冲区越界失败，
//!   Rust 侧该情形退化为拉伸 resize 到 784×784（其余路径与原实现一致）。

use ort::session::SessionOutputs;
use ort::value::{DynValue, Tensor};

use crate::core::base::{BaseOnnxEngine, TensorData, TensorOutput};
use crate::core::device_type::DeviceType;
use crate::error::{Result, VisionError};
use crate::imaging::{cvt_color, resize, ColorConversion, Image, Interpolation};

/// descriptor 固定输入尺寸 784×784（davnords/storage 的 ONNX hardcode）。
const DEDODE_INPUT_SIZE: usize = 784;

/// 描述子维度（原版 `Detection.descriptors` 固定按 256 列分配）。
const DESCRIPTOR_DIM: usize = 256;

/// DeDoDe-G 检测引擎。
pub struct DedodeGEngine {
    /// 组合基类
    pub base: BaseOnnxEngine,

    /// descriptor 独立 Session（detector 由 `base` 管理；对应 `descriptorSession`）
    descriptor_base: BaseOnnxEngine,

    /// 关键点置信度阈值（0..1），低于此分数的关键点会被丢弃（sigmoid 后的概率）
    confidence_threshold: f32,

    /// 期望保留的最多关键点数（topK），实际 N 取决于 threshold + topK 较小者
    num_keypoints: i32,

    /// descriptor 的输入名（image + keypoints）
    descriptor_image_name: String,
    descriptor_kpts_name: String,

    /// detector 第二输入 num_keypoints 名（如果输入数 >= 2）
    detector_num_kpts_name: Option<String>,

    /// detector 概率输出名（keypoint_probs / confidence / score）
    detector_probs_name: Option<String>,

    /// 是否使用 letterbox（true=只缩不放到 784）
    letterbox: bool,
}

/// DeDoDe-G 单图检测结果（对应 `DedodeGEngine.Detection`）。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Detection {
    /// 原图像素坐标的关键点 [N][2]（供可视化 / 几何验证使用）
    pub keypoints: Vec<[f32; 2]>,
    /// 归一化坐标的关键点 [N][2]，范围 [-1, 1]，对应 PyTorch
    /// `get_normalized_grid`（与 `grid_sample` align_corners=False 对齐）。
    ///
    /// matcher（如 LoMa-R）的 Fourier 位置编码必须接收此坐标空间，不能是像素坐标。
    pub normalized_keypoints: Vec<[f32; 2]>,
    /// 每个关键点的置信度（sigmoid 后的概率） [N]
    pub scores: Vec<f32>,
    /// 每个关键点处的描述子 [N][256]
    pub descriptors: Vec<Vec<f32>>,
    /// 原图宽度
    pub image_width: i32,
    /// 原图高度
    pub image_height: i32,
}

impl Detection {
    /// 实际关键点数。
    pub fn size(&self) -> usize {
        self.keypoints.len()
    }
}

/// letterbox 预处理中间信息（对应 `PreprocessInfo`；pixelData 随调用链显式返回）。
#[derive(Debug, Clone, Copy, Default)]
struct PreprocessInfo {
    scale: f32,
    pad_top: i32,
    pad_left: i32,
    #[allow(dead_code)]
    original_h: i32,
    #[allow(dead_code)]
    original_w: i32,
}

/// detector 输出解析结果（对应 `DetectorOut`）。
struct DetectorOut {
    /// 扁平 keypoints [n*2]，范围 [-1, 1]
    kpts: Vec<f32>,
    /// sigmoid 后的概率 [n]
    probs: Vec<f32>,
    num_keypoints: usize,
}

impl DedodeGEngine {
    /// 创建 DeDoDe-G 引擎（默认参数：confidenceThreshold=0.0、numKeypoints=2048、letterbox=true）。
    ///
    /// numKeypoints=2048 对齐 PyTorch LoMa-R 默认值（cfg.num_keypoints=2048）；
    /// 关键点数过少会让 matcher 缺少候选，显著降低匹配数与 RANSAC inlier。
    pub fn new(
        detector_model_path: impl AsRef<std::path::Path>,
        descriptor_model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
    ) -> Result<Self> {
        Self::with_options(
            detector_model_path,
            descriptor_model_path,
            device_type,
            0.0,
            2048,
            true,
        )
    }

    /// 完整参数构造。
    ///
    /// detector 基类用 784×784 输入（与 descriptor 一致）。
    pub fn with_options(
        detector_model_path: impl AsRef<std::path::Path>,
        descriptor_model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        confidence_threshold: f32,
        num_keypoints: i32,
        letterbox: bool,
    ) -> Result<Self> {
        let detector_model_path = detector_model_path.as_ref();
        let descriptor_model_path = descriptor_model_path.as_ref();
        let base = BaseOnnxEngine::with_input_size(
            detector_model_path,
            device_type,
            DEDODE_INPUT_SIZE as i32,
            DEDODE_INPUT_SIZE as i32,
        )?;
        // descriptor 用独立 Session（参数与 detector 一致，默认行为不变）
        let descriptor_base = BaseOnnxEngine::with_input_size(
            descriptor_model_path,
            device_type,
            DEDODE_INPUT_SIZE as i32,
            DEDODE_INPUT_SIZE as i32,
        )?;

        // ---- descriptor 输入名（image + keypoints）----
        let (descriptor_image_name, descriptor_kpts_name) = {
            let session = descriptor_base.session.lock().unwrap();
            if session.inputs.len() < 2 {
                return Err(VisionError::invalid_argument(format!(
                    "DeDoDe descriptor ONNX 至少需要 2 个输入，实际: {}",
                    session.inputs.len()
                )));
            }
            (
                session.inputs[0].name.clone(),
                session.inputs[1].name.clone(),
            )
        };

        // ---- detector 第二输入 num_keypoints 名（如果输入数 >= 2）----
        let detector_num_kpts_name = {
            let session = base.session.lock().unwrap();
            session.inputs.get(1).map(|i| i.name.clone())
        };

        // ---- 找 detector 概率输出名（keypoint_probs / confidence / score）----
        let detector_probs_name = {
            let mut found: Option<String> = None;
            for n in base.output_names() {
                let lower = n.to_lowercase();
                if (lower.contains("prob") || lower.contains("confidence") || lower.contains("score"))
                    && (lower.contains("keypoint") || lower.contains("conf") || lower.contains("score"))
                {
                    found = Some(n.clone());
                    break;
                }
            }
            if found.is_none() && base.output_names().len() >= 2 {
                // 兜底：第二个输出当作概率
                found = Some(base.output_names()[1].clone());
            }
            found
        };

        let detector_num_inputs = base.session.lock().unwrap().inputs.len();
        tracing::info!(
            "DedodeG Engine initialized: detector='{}', descriptor='{}', inputSize={}x{}, numInput={}, detectorProbsName={:?}, descriptorOutputs={:?}, confThreshold={}, numKeypoints={}",
            detector_model_path.display(),
            descriptor_model_path.display(),
            DEDODE_INPUT_SIZE,
            DEDODE_INPUT_SIZE,
            detector_num_inputs,
            detector_probs_name,
            descriptor_base.output_names(),
            confidence_threshold,
            num_keypoints
        );

        Ok(DedodeGEngine {
            base,
            descriptor_base,
            confidence_threshold,
            num_keypoints,
            descriptor_image_name,
            descriptor_kpts_name,
            detector_num_kpts_name,
            detector_probs_name,
            letterbox,
        })
    }

    // ============ 访问器 ============

    /// 关键点置信度阈值（0..1）。
    pub fn confidence_threshold(&self) -> f32 {
        self.confidence_threshold
    }

    /// 设置关键点置信度阈值。
    pub fn set_confidence_threshold(&mut self, confidence_threshold: f32) {
        self.confidence_threshold = confidence_threshold;
    }

    /// 期望保留的最多关键点数（topK）。
    pub fn num_keypoints(&self) -> i32 {
        self.num_keypoints
    }

    /// 设置期望保留的最多关键点数（topK）。
    pub fn set_num_keypoints(&mut self, num_keypoints: i32) {
        self.num_keypoints = num_keypoints;
    }

    /// 是否使用 letterbox（true=只缩不放到 784）。
    pub fn letterbox(&self) -> bool {
        self.letterbox
    }

    // ============ 检测 ============

    /// 检测单张图像，返回关键点 + 描述子。
    ///
    /// # 参数
    /// - `image`：BGR 图像
    ///
    /// # 返回
    /// Detection 结果（关键点已是原图像素坐标）
    pub fn detect(&self, image: &Image) -> Result<Detection> {
        if image.is_empty() {
            return Err(VisionError::invalid_argument(
                "Input image cannot be null or empty",
            ));
        }
        let orig_w = image.width() as i32;
        let orig_h = image.height() as i32;

        // 1. 预处理 —— 统一 letterbox 到 784×784
        let (pixel_data, pre) = self.preprocess_dedode(image)?;
        let h = DEDODE_INPUT_SIZE as i64;

        // 2. detector 推理（image + 可选 num_keypoints）
        let det_outputs: Vec<TensorOutput> = {
            let mut session = self.base.session.lock().unwrap();
            let img_tensor = Tensor::from_array((vec![1, 3, h, h], pixel_data.clone()))?;
            let outputs = if let Some(nk_name) = &self.detector_num_kpts_name {
                let nk_tensor =
                    Tensor::from_array((vec![1i64], vec![self.num_keypoints as i64]))?;
                session.run(ort::inputs![
                    self.base.input_name() => img_tensor,
                    nk_name.as_str() => nk_tensor
                ])?
            } else {
                session.run(ort::inputs![self.base.input_name() => img_tensor])?
            };
            snapshot_outputs(&outputs, self.base.output_names())?
        };

        // 3. 解析 detector 输出
        let dout = self.parse_detector_output(&det_outputs)?;

        // 4. 过滤：probs sigmoid 后 >= threshold，取 topK
        let n = dout.num_keypoints;
        let mut order: Vec<usize> = (0..n).collect();
        // 按概率降序（稳定排序）
        order.sort_by(|&a, &b| {
            dout.probs[b]
                .partial_cmp(&dout.probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let keep = (self.num_keypoints as usize).min(n);
        let mut kpts_norm_list: Vec<[f32; 2]> = Vec::with_capacity(keep); // [-1, 1] 归一化坐标
        let mut score_list: Vec<f32> = Vec::with_capacity(keep);
        for &i in &order {
            if kpts_norm_list.len() >= keep {
                break;
            }
            if dout.probs[i] < self.confidence_threshold {
                break;
            }
            kpts_norm_list.push([dout.kpts[i * 2], dout.kpts[i * 2 + 1]]);
            score_list.push(dout.probs[i]);
        }

        // 5. 用过滤后的关键点跑 descriptor（descriptor 输入 kpts 是 [-1,1]）
        let kept = kpts_norm_list.len();
        let mut descriptors = vec![vec![0f32; DESCRIPTOR_DIM]; kept];
        let mut kpts_pixel_list: Vec<[f32; 2]> = Vec::with_capacity(kept); // 还原到原图像素
        if kept > 0 {
            let mut kpts_norm = Vec::with_capacity(kept * 2);
            for k in &kpts_norm_list {
                kpts_norm.push(k[0]);
                kpts_norm.push(k[1]);
            }
            // descriptor 要求 [1, n, 2] (rank 3)，不是 [n, 2]
            let kpts_tensor = Tensor::from_array((vec![1, kept as i64, 2], kpts_norm))?;

            // descriptor 用独立的 image tensor，不与 detector 复用同一个 OrtValue：
            // 一个 Value 横跨两个 session 的绑定/释放序列时，Windows+CUDA 下曾被
            // 上游提前释放 native 对象；这里用 clone 的数据独立创建
            let desc_img_tensor = Tensor::from_array((vec![1, 3, h, h], pixel_data.clone()))?;

            let desc_outputs: Vec<TensorOutput> = {
                let mut session = self.descriptor_base.session.lock().unwrap();
                let outputs = session.run(ort::inputs![
                    self.descriptor_image_name.as_str() => desc_img_tensor,
                    self.descriptor_kpts_name.as_str() => kpts_tensor
                ])?;
                snapshot_outputs(&outputs, self.descriptor_base.output_names())?
            };

            let desc_tensor = self
                .descriptor_base
                .output_names()
                .iter()
                .position(|name| name == "descriptions")
                .map(|idx| &desc_outputs[idx])
                .unwrap_or(&desc_outputs[0]);
            let desc_flat = tensor_data_as_f32(&desc_tensor.data)?;
            let desc_shape = &desc_tensor.shape;
            let desc_dim = if desc_shape.len() >= 3 {
                desc_shape[desc_shape.len() - 1] as usize
            } else {
                DESCRIPTOR_DIM
            };
            let copy_len = desc_dim.min(DESCRIPTOR_DIM);
            for (i, row) in descriptors.iter_mut().enumerate().take(kept) {
                let start = i * desc_dim;
                let end = start + copy_len;
                if end <= desc_flat.len() {
                    row[..copy_len].copy_from_slice(&desc_flat[start..end]);
                }
            }

            // 6. 把 detector 输出的 [-1,1] 坐标还原到原图像素
            //    反归一化公式: pixel = (norm + 1) / 2 * letterbox_input_dim
            //    letterbox 还原: (letterbox_pixel - pad) / scale
            for k in &kpts_norm_list {
                let (nx, ny) = (k[0], k[1]);
                // 784 空间像素
                let lbx = (nx + 1.0) / 2.0 * DEDODE_INPUT_SIZE as f32;
                let lby = (ny + 1.0) / 2.0 * DEDODE_INPUT_SIZE as f32;
                // letterbox 还原
                let ox = (lbx - pre.pad_left as f32) / pre.scale;
                let oy = (lby - pre.pad_top as f32) / pre.scale;
                let ox = ox.max(0.0).min(orig_w as f32 - 1.0);
                let oy = oy.max(0.0).min(orig_h as f32 - 1.0);
                kpts_pixel_list.push([ox, oy]);
            }
        }

        // 7. 组装 Detection
        let mut keypoints = Vec::with_capacity(kept);
        for i in 0..kept {
            keypoints.push(kpts_pixel_list.get(i).copied().unwrap_or([0.0, 0.0]));
        }
        Ok(Detection {
            keypoints,
            // 保留归一化坐标 [-1,1]，供 matcher 的位置编码使用（像素坐标会让 Fourier 编码饱和）
            normalized_keypoints: kpts_norm_list,
            scores: score_list,
            descriptors,
            image_width: orig_w,
            image_height: orig_h,
        })
    }

    // ============ 预处理 ============

    /// 预处理：BGR→RGB + letterbox（只缩不放）/ 拉伸到 784×784 + /255，HWC → CHW。
    ///
    /// 返回 `(CHW 浮点数据, 坐标还原参数)`。
    fn preprocess_dedode(&self, img: &Image) -> Result<(Vec<f32>, PreprocessInfo)> {
        // BGR→RGB；兼容 4/1 通道输入
        let rgb = match img.channels() {
            3 => cvt_color(img, ColorConversion::Bgr2Rgb)?,
            4 => cvt_color(img, ColorConversion::Bgra2Rgb)?,
            _ => cvt_color(img, ColorConversion::Gray2Rgb)?,
        };

        let orig_h = rgb.height();
        let orig_w = rgb.width();
        let h_size = DEDODE_INPUT_SIZE;

        let scale;
        let new_w;
        let new_h;

        if self.letterbox {
            let mut s = (h_size as f32 / orig_w as f32).min(h_size as f32 / orig_h as f32);
            s = s.min(1.0); // 只缩不放
            scale = s;
            new_w = (orig_w as f32 * s).round() as usize;
            new_h = (orig_h as f32 * s).round() as usize;
        } else {
            scale = (h_size as f32 / orig_w as f32).min(h_size as f32 / orig_h as f32);
            new_w = h_size;
            new_h = h_size;
        }

        let resized = if scale < 1.0 {
            resize(&rgb, new_w, new_h, Interpolation::Linear)?
        } else if self.letterbox {
            rgb.clone()
        } else {
            // 上游此分支直接 clone（保持原尺寸），随后按 784×784 读缓冲会越界；
            // Rust 侧退化为拉伸 resize 保证安全（数值上等价于"铺满画布"的意图）
            resize(&rgb, new_w, new_h, Interpolation::Linear)?
        };

        let (padded, pad_top, pad_left) = if self.letterbox {
            let pad_top = (h_size - new_h) / 2;
            let pad_left = (h_size - new_w) / 2;
            // Scalar(0,0,0,0) 黑边画布 + ROI copyTo
            let mut canvas = Image::new(h_size, h_size, rgb.channels());
            canvas.paste(pad_left, pad_top, &resized);
            (canvas, pad_top, pad_left)
        } else {
            (resized, 0, 0)
        };

        let total = h_size * h_size;
        let raw = padded.data();
        if raw.len() < total * 3 {
            return Err(VisionError::image(format!(
                "preprocessed buffer {} < {}x{}x3",
                raw.len(),
                h_size,
                h_size
            )));
        }

        // ONNX 期望 NCHW 布局 [1, 3, H, W]，但 data 是 HWC [H, W, 3]。
        // 必须做 HWC → CHW 转换，否则 detector 看到的是 channel 错位的图。
        let mut pixels = vec![0f32; 3 * total];
        // R 通道
        for i in 0..total {
            pixels[i] = raw[i * 3] as f32 / 255.0;
        }
        // G 通道
        for i in 0..total {
            pixels[total + i] = raw[i * 3 + 1] as f32 / 255.0;
        }
        // B 通道
        for i in 0..total {
            pixels[2 * total + i] = raw[i * 3 + 2] as f32 / 255.0;
        }

        Ok((
            pixels,
            PreprocessInfo {
                scale,
                pad_top: pad_top as i32,
                pad_left: pad_left as i32,
                original_h: orig_h as i32,
                original_w: orig_w as i32,
            },
        ))
    }

    // ============ detector 输出解析 ============

    /// 解析 detector 输出：keypoints（默认 0 号输出）+ probs（按名字定位，sigmoid）。
    fn parse_detector_output(&self, outputs: &[TensorOutput]) -> Result<DetectorOut> {
        // keypoints 输出（默认 0 号）
        let kpts_output = outputs
            .first()
            .ok_or_else(|| VisionError::inference("detector returned no outputs"))?;
        let kpts_flat = tensor_data_as_f32(&kpts_output.data)?;
        let kpts_shape = &kpts_output.shape;
        let n = match kpts_shape.len() {
            3 => kpts_shape[1] as usize, // [1, n, 2]
            2 => kpts_shape[0] as usize, // [n, 2]
            _ => {
                return Err(VisionError::inference(format!(
                    "Unexpected keypoints shape: {kpts_shape:?}"
                )))
            }
        };

        // probs 输出（keypoint_probs / confidence / score）
        let mut probs = vec![0f32; n];
        let probs_output = self
            .detector_probs_name
            .as_ref()
            .and_then(|name| outputs.iter().find(|o| &o.name == name));
        match probs_output {
            Some(pt) => {
                let p_flat = tensor_data_as_f32(&pt.data)?;
                let p_shape = &pt.shape;
                let np = match p_shape.len() {
                    2 => p_shape[1] as usize,
                    1 => p_shape[0] as usize,
                    _ => n,
                };
                let count = np.min(probs.len()).min(p_flat.len());
                probs[..count].copy_from_slice(&p_flat[..count]);
                // sigmoid
                for p in probs.iter_mut() {
                    *p = sigmoid(*p);
                }
            }
            None => probs.fill(1.0),
        }

        Ok(DetectorOut {
            kpts: kpts_flat,
            probs,
            num_keypoints: n,
        })
    }
}

/// sigmoid（`parse_detector_output` 内的 1/(1+exp(-x))）。
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// 张量数据转 f32（对应 `readTensorAsFloatLocal` 的 FLOAT/INT64/INT32 转换分支；
/// 其他类型报错）。
fn tensor_data_as_f32(data: &TensorData) -> Result<Vec<f32>> {
    match data {
        TensorData::F32(v) => Ok(v.clone()),
        TensorData::I64(v) => Ok(v.iter().map(|&x| x as f32).collect()),
        TensorData::Unsupported(t) => Err(VisionError::inference(format!(
            "unsupported tensor element type: {t}"
        ))),
    }
}

/// 把 session 输出按 `names` 顺序复制为 owned 快照（对齐 `BaseOnnxEngine::snapshot_output`）。
fn snapshot_outputs(outputs: &SessionOutputs<'_>, names: &[String]) -> Result<Vec<TensorOutput>> {
    let mut result = Vec::with_capacity(names.len());
    for name in names {
        let value: &DynValue = outputs
            .get(name.as_str())
            .ok_or_else(|| VisionError::inference(format!("missing output '{name}'")))?;
        let (shape, ty) = match value.dtype() {
            ort::value::ValueType::Tensor { ty, shape, .. } => {
                (shape.iter().copied().collect::<Vec<i64>>(), ty)
            }
            other => {
                result.push(TensorOutput {
                    name: name.clone(),
                    shape: Vec::new(),
                    data: TensorData::Unsupported(format!("{other:?}")),
                });
                continue;
            }
        };
        let data = match ty {
            ort::tensor::TensorElementType::Float32 => {
                let (_, view) = value.try_extract_tensor::<f32>()?;
                TensorData::F32(view.to_vec())
            }
            ort::tensor::TensorElementType::Int64 => {
                let (_, view) = value.try_extract_tensor::<i64>()?;
                TensorData::I64(view.to_vec())
            }
            ort::tensor::TensorElementType::Int32 => {
                // 上游 readTensorAsFloat 直接转 f32
                let (_, view) = value.try_extract_tensor::<i32>()?;
                TensorData::F32(view.iter().map(|&x| x as f32).collect())
            }
            other => TensorData::Unsupported(format!("{other:?}")),
        };
        result.push(TensorOutput {
            name: name.clone(),
            shape,
            data,
        });
    }
    Ok(result)
}

crate::impl_engine_forward!(DedodeGEngine, base, Detection,
    /// 单图推理（对应 `predict(Mat)` → `detect`）。
    fn predict(&self, image: &Image) -> Result<Detection> {
        self.detect(image)
    }
);
