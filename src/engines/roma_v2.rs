//! RoMaV2（Robust Dense Feature Matching v2）推理引擎
//! 。
//!
//! 支持 RoMaV2 密集特征匹配 ONNX 模型，用于两张图像间的**密集对应场**估计。
//! A 图中每个像素都映射到 B 图的坐标，并给出对应的置信度。
//!
//! **模型格式**
//! - 双输入: img_A `[batch,3,640,640]` + img_B `[batch,3,640,640]`，float32 RGB
//! - 仅 /255 归一化到 [0,1]，**无 ImageNet mean/std**（DINOv3 主干在内部处理归一化）
//! - stretch resize 到输入尺寸（无 letterbox padding）
//!
//! **输出**
//! - warp_AB: `[batch,H,W,2]` — A 中每像素映射到 B 的归一化坐标 [-1,1]
//!   （align_corners=False，±1 为 B 边界）
//! - overlap_AB: `[batch,H,W,1]` — 每个对应的置信度 [0,1]（sigmoid 已烘焙入图）
//!
//! **采样策略**
//! 引擎除保留完整密集对应场外，还按固定步长（[`SAMPLE_STRIDE`]）采样稀疏匹配，
//! 过滤 `overlap >= overlap_threshold` 且在 B 图边界内（`|warp| <= 1`）的对应，
//! 坐标按官方 `to_pixel` 公式还原到 A/B 原图像素空间，便于 findHomography 等
//! 稀疏几何估计。
//!
//! **推理实现**
//! 原版使用 IoBinding API 按名称绑定多输入/输出；Rust 侧 ort 的 `session.run`
//! 本身即按名称绑定输入并返回全部输出，语义一致。
//!
//! 与 原版的其他差异（数值逻辑保持逐行一致）：
//! - 上游的 `predict` / `predictBatch` 抛 `UnsupportedOperationException`，
//!   不适用统一推理 trait，因此本引擎不实现
//!   [`crate::core::engine::OnnxInferenceEngine`]，仅提供固有方法 `match_images`。

use ort::value::TensorElementType;
use ort::value::{DynValue, Tensor, ValueType};

use crate::core::base::BaseOnnxEngine;
use crate::core::device_type::DeviceType;
use crate::error::{Result, VisionError};
use crate::imaging::{cvt_color, resize, ColorConversion, Image, Interpolation};
use crate::model::{DenseMatch, RomaV2MatchResult};

/// 采样步长（像素），按此间隔在 dense 网格上采样稀疏匹配点。
const SAMPLE_STRIDE: usize = 16;

/// RoMaV2 密集特征匹配推理引擎。
pub struct RomaV2Engine {
    /// 组合基类
    pub base: BaseOnnxEngine,

    /// 置信度阈值，用于过滤采样稀疏匹配
    overlap_threshold: f32,

    /// 第二个输入的名称（img_B），按名称绑定
    input1_name: String,
}

impl RomaV2Engine {
    /// 创建 RoMaV2 引擎（overlap 阈值默认 0.3，输入尺寸从模型读取，动态维度回退 640）。
    pub fn new(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        Self::with_overlap_threshold(model_path, device_type, 0.3)
    }

    /// 指定置信度阈值创建。
    pub fn with_overlap_threshold(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        overlap_threshold: f32,
    ) -> Result<Self> {
        Self::with_input_size(model_path, device_type, overlap_threshold, -1, -1)
    }

    /// 指定置信度阈值与输入尺寸创建（input_height/input_width <=0 时从模型读取）。
    pub fn with_input_size(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        overlap_threshold: f32,
        input_height: i32,
        input_width: i32,
    ) -> Result<Self> {
        let base =
            BaseOnnxEngine::with_input_size(model_path, device_type, input_height, input_width)?;

        // 校验模型为双输入（img_A + img_B）
        let input1_name = {
            let session = base.session.lock().unwrap();
            if session.inputs().len() < 2 {
                return Err(VisionError::inference(format!(
                    "RoMaV2 模型应为双输入（img_A + img_B），当前输入数: {}",
                    session.inputs().len()
                )));
            }
            // 获取第二个输入名（按名称绑定）
            session.inputs()[1].name().to_string()
        };

        tracing::info!(
            "RoMaV2 Engine initialized: inputs=[{}, {}], inputSize={}x{}, overlapThreshold={}",
            base.input_name(),
            input1_name,
            base.input_width(),
            base.input_height(),
            overlap_threshold
        );

        Ok(RomaV2Engine {
            base,
            overlap_threshold,
            input1_name,
        })
    }

    // ============ 访问器 ============

    /// 置信度阈值，用于过滤采样稀疏匹配。
    pub fn overlap_threshold(&self) -> f32 {
        self.overlap_threshold
    }

    /// 设置置信度阈值。
    pub fn set_overlap_threshold(&mut self, threshold: f32) {
        self.overlap_threshold = threshold;
    }

    // ============ 推理入口 ============

    /// 匹配两张图像（密集对应场）。
    ///
    /// - `img_a`: A 图（源图，warp_AB 描述其每像素到 B 的映射）
    /// - `img_b`: B 图（目标图）
    ///
    /// 返回密集匹配结果（含密集对应场 + 采样稀疏匹配）。
    pub fn match_images(&self, img_a: &Image, img_b: &Image) -> Result<RomaV2MatchResult> {
        if img_a.is_empty() || img_b.is_empty() {
            return Err(VisionError::invalid_argument(
                "Input images cannot be null or empty",
            ));
        }

        let orig_wa = img_a.width() as i32;
        let orig_ha = img_a.height() as i32;
        let orig_wb = img_b.width() as i32;
        let orig_hb = img_b.height() as i32;

        // 1. 预处理（stretch resize 到输入尺寸，/255，NCHW）
        let pre_a = self.preprocess_rgb(img_a)?;
        let pre_b = self.preprocess_rgb(img_b)?;

        // 2. 双输入推理
        let outputs = self.run_dual_input(&pre_a.pixel_data, &pre_b.pixel_data)?;

        // 3. 解析输出（密集对应场 + 采样稀疏匹配）
        self.parse_output(&outputs, &pre_a, &pre_b, orig_wa, orig_ha, orig_wb, orig_hb)
    }

    // ==================== 预处理 ====================

    /// RGB 化 -> stretch resize 到输入尺寸 -> /255 归一化到 [0,1] -> NCHW float32
    ///
    /// RoMaV2 不使用 ImageNet 归一化（官方 normalizers.py 为空），仅做 [0,1] 缩放，
    /// 因为 DINOv3 主干网络在内部处理 patch embedding 归一化。
    fn preprocess_rgb(&self, img: &Image) -> Result<PreprocessInfo> {
        // 1. 转换为 RGB
        let rgb = match img.channels() {
            4 => cvt_color(img, ColorConversion::Bgra2Rgb)?,
            3 => cvt_color(img, ColorConversion::Bgr2Rgb)?,
            _ => cvt_color(img, ColorConversion::Gray2Rgb)?,
        };

        let orig_h = rgb.height() as i32;
        let orig_w = rgb.width() as i32;
        let input_w = self.base.input_width() as usize;
        let input_h = self.base.input_height() as usize;

        // 2. stretch resize 到输入尺寸（无 letterbox，保持 RoMaV2 官方做法）
        let resized = resize(&rgb, input_w, input_h, Interpolation::Linear)?;

        // 3. 提取像素并归一化到 [0,1]，HWC -> CHW
        let area = input_h * input_w;
        let mut pixel_data = vec![0f32; 3 * area];
        let px = resized.data();
        for i in 0..area {
            // R 通道
            pixel_data[i] = px[i * 3] as f32 / 255.0;
            // G 通道
            pixel_data[i + area] = px[i * 3 + 1] as f32 / 255.0;
            // B 通道
            pixel_data[i + 2 * area] = px[i * 3 + 2] as f32 / 255.0;
        }

        Ok(PreprocessInfo {
            pixel_data,
            original_h: orig_h,
            original_w: orig_w,
            scale_x: input_w as f32 / orig_w as f32,
            scale_y: input_h as f32 / orig_h as f32,
        })
    }

    // ==================== 推理 ====================

    /// 双输入推理（两个张量 shape 均为 `[1, 3, H, W]`，按名称绑定）。
    fn run_dual_input(&self, data_a: &[f32], data_b: &[f32]) -> Result<Vec<RawTensor>> {
        let shape = vec![
            1,
            3,
            self.base.input_height() as i64,
            self.base.input_width() as i64,
        ];
        let tensor_a = Tensor::from_array((shape.clone(), data_a.to_vec()))?;
        let tensor_b = Tensor::from_array((shape, data_b.to_vec()))?;

        let mut session = self.base.session.lock().unwrap();
        let outputs = session.run(ort::inputs![
            self.base.input_name() => tensor_a,
            self.input1_name.as_str() => tensor_b,
        ])?;
        snapshot_outputs(&outputs)
    }

    // ==================== 输出解析 ====================

    /// 解析输出：密集对应场 + 采样稀疏匹配。
    ///
    /// RoMaV2 标准输出:
    /// - warp_AB: `[1, H, W, 2]` — A 中每像素映射到 B 的归一化坐标 [-1,1]
    /// - overlap_AB: `[1, H, W, 1]` — 每个对应的置信度 [0,1]（sigmoid 已烘焙）
    ///
    /// 采样坐标还原（官方 `to_pixel` 公式，用 `*W` 不用 `*(W-1)`）：
    /// - A 图输入空间像素 `(j, i)` -> 原图: `xA = (j + 0.5) / inputW * origWA`
    /// - B 图归一化 `(nx, ny)` -> 原图: `xB = (nx + 1) / 2 * origWB`
    fn parse_output(
        &self,
        outputs: &[RawTensor],
        pre_a: &PreprocessInfo,
        pre_b: &PreprocessInfo,
        orig_wa: i32,
        orig_ha: i32,
        orig_wb: i32,
        orig_hb: i32,
    ) -> Result<RomaV2MatchResult> {
        if outputs.is_empty() {
            return Err(VisionError::inference("model returned no outputs"));
        }

        // 按名称找到输出张量；如果名称匹配失败，按位置取
        let warp_tensor = find_output(outputs, "warp_AB").unwrap_or(&outputs[0]);
        let overlap_tensor = find_output(outputs, "overlap_AB")
            .or_else(|| if outputs.len() > 1 { outputs.get(1) } else { None });

        // 解析 warp_AB: [1, H, W, 2]，取 H、W
        let warp_flat = &warp_tensor.data;
        let warp_shape = &warp_tensor.shape;
        if warp_shape.len() < 3 {
            return Err(VisionError::inference(format!(
                "Unexpected warp_AB shape dims: {}, shape={warp_shape:?}",
                warp_shape.len()
            )));
        }
        let dense_h = warp_shape[warp_shape.len() - 3] as usize;
        let dense_w = warp_shape[warp_shape.len() - 2] as usize;

        // 解析 overlap_AB: [1, H, W, 1] -> 压掉最后一维，长度 = H*W
        let overlap_flat: Vec<f32> = match overlap_tensor {
            Some(t) => t.data.clone(),
            None => vec![0.0; dense_h * dense_w],
        };

        // 采样稀疏匹配
        let sampled_matches = self.sample_dense_matches(
            warp_flat,
            &overlap_flat,
            dense_h,
            dense_w,
            orig_wa,
            orig_ha,
            orig_wb,
            orig_hb,
        );

        tracing::info!(
            "RoMaV2 result: dense={}x{}, sampled matches={} (threshold={})",
            dense_w,
            dense_h,
            sampled_matches.len(),
            self.overlap_threshold
        );

        Ok(RomaV2MatchResult {
            dense_height: dense_h,
            dense_width: dense_w,
            warp_ab: Some(warp_flat.clone()),
            overlap_ab: Some(overlap_flat),
            sampled_matches,
            image_a_width: orig_wa,
            image_a_height: orig_ha,
            image_b_width: orig_wb,
            image_b_height: orig_hb,
            scale_a: pre_a.scale_x,
            scale_b: pre_b.scale_x,
        })
    }

    /// 按 [`SAMPLE_STRIDE`] 步长在 dense 网格上采样，过滤低置信度和越界对应，
    /// 坐标还原到 A/B 原图像素空间。
    fn sample_dense_matches(
        &self,
        warp: &[f32],
        overlap: &[f32],
        dense_h: usize,
        dense_w: usize,
        orig_wa: i32,
        orig_ha: i32,
        orig_wb: i32,
        orig_hb: i32,
    ) -> Vec<DenseMatch> {
        let mut matches = Vec::new();

        // 归一化坐标 [-1,1] -> 原图像素 的转换系数
        let half_w = orig_wb as f32 / 2.0;
        let half_h = orig_hb as f32 / 2.0;

        // A 图输入空间 -> 原图 的转换系数（中心像素对齐）
        let scale_x_a = orig_wa as f32 / dense_w as f32;
        let scale_y_a = orig_ha as f32 / dense_h as f32;

        let mut y = 0;
        while y < dense_h {
            let mut x = 0;
            while x < dense_w {
                let idx = y * dense_w + x;
                let Some(conf) = overlap.get(idx).copied() else {
                    break;
                };
                if conf < self.overlap_threshold {
                    x += SAMPLE_STRIDE;
                    continue;
                }

                let warp_idx = idx * 2;
                let Some(nx) = warp.get(warp_idx).copied() else {
                    break;
                };
                let Some(ny) = warp.get(warp_idx + 1).copied() else {
                    break;
                };

                // in-frame 过滤：归一化坐标超出 [-1,1] 表示映射到 B 图外
                if nx.abs() > 1.0 || ny.abs() > 1.0 {
                    x += SAMPLE_STRIDE;
                    continue;
                }

                // 还原到原图像素空间（官方 to_pixel: (n + 1) / 2 * W）
                let x_b = (nx + 1.0) * half_w;
                let y_b = (ny + 1.0) * half_h;

                // A 图输入空间像素 (x, y) 中心对齐到原图
                let x_a = (x as f32 + 0.5) * scale_x_a;
                let y_a = (y as f32 + 0.5) * scale_y_a;

                matches.push(DenseMatch {
                    x_a,
                    y_a,
                    x_b,
                    y_b,
                    score: conf,
                });

                x += SAMPLE_STRIDE;
            }
            y += SAMPLE_STRIDE;
        }
        matches
    }
}

// ==================== 张量解析辅助 ====================

/// 根据名称查找输出张量（按绑定顺序返回，对应 `outputNames`）。
fn find_output<'a>(outputs: &'a [RawTensor], name: &str) -> Option<&'a RawTensor> {
    outputs.iter().find(|t| t.name == name)
}

// ==================== 内部数据结构 ====================

/// 预处理信息（对应 `PreprocessInfo`）。
// original_h/original_w/scale_y 与原实现一致仅作记录（结果仅使用 scale_x）。
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
struct PreprocessInfo {
    /// /255 归一化后的 NCHW 数据（长度 3*H*W）
    pixel_data: Vec<f32>,
    original_h: i32,
    original_w: i32,
    scale_x: f32,
    scale_y: f32,
}

/// 输出张量快照（名称 + 形状 + 已转为 f32 的数据）。
#[derive(Debug, Clone)]
struct RawTensor {
    name: String,
    shape: Vec<i64>,
    data: Vec<f32>,
}

/// 按 `output_names` 顺序把会话输出快照为 f32 张量。
///
/// 对应 `readTensorAsFloat` 的多类型兼容：float32 / int64 / int32
/// （RoMaV2 的 warp_AB / overlap_AB 通常都是 float32，但保留多类型兼容）；
/// 未知类型尝试按 float 读取（类型不符时由 ort 报错）。
fn snapshot_outputs(outputs: &ort::session::SessionOutputs<'_>) -> Result<Vec<RawTensor>> {
    let mut result = Vec::new();
    for (name, value) in outputs.iter() {
        result.push(snapshot_output(name, &value)?);
    }
    Ok(result)
}

/// 把单个输出 Value 复制并转换为 f32 快照。
fn snapshot_output(name: &str, value: &DynValue) -> Result<RawTensor> {
    let (shape, ty) = match value.dtype() {
        ValueType::Tensor { ty, shape, .. } => (shape.iter().copied().collect::<Vec<i64>>(), ty),
        other => {
            return Err(VisionError::inference(format!(
                "output '{name}' is not a tensor: {other:?}"
            )));
        }
    };

    let data = match ty {
        TensorElementType::Float32 => {
            let (_, view) = value.try_extract_tensor::<f32>()?;
            view.to_vec()
        }
        TensorElementType::Int64 => {
            let (_, view) = value.try_extract_tensor::<i64>()?;
            view.iter().map(|&v| v as f32).collect()
        }
        TensorElementType::Int32 => {
            let (_, view) = value.try_extract_tensor::<i32>()?;
            view.iter().map(|&v| v as f32).collect()
        }
        // 未知类型，尝试 float
        _ => {
            let (_, view) = value.try_extract_tensor::<f32>()?;
            view.to_vec()
        }
    };

    Ok(RawTensor {
        name: name.to_string(),
        shape,
        data,
    })
}
