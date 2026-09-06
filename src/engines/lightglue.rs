//! LightGlue 端到端 Pipeline 推理引擎。
//!
//! 支持 LightGlue 特征匹配 ONNX 模型，用于两张图像间的关键点检测与匹配。
//!
//! **模型格式**
//! - 双输入模式: image0 `[1,1,H,W]` + image1 `[1,1,H,W]`
//! - 单输入模式: images `[2,1,H,W]` — 两张图像合并为一个 batch
//!
//! **输出**
//! - keypoints: `(2, N, 2)` — 两张图像的关键点坐标
//! - matches: `(M, 2)` 或 `(M, 3)` — 匹配索引对
//! - mscores: `(M,)` — 匹配分数
//!
//! 与 原版的实现差异（数值逻辑保持逐行一致）：
//! - 上游的 `predict` / `predictBatch` 抛 `UnsupportedOperationException`，
//!   不适用统一推理 trait，因此本引擎不实现
//!   [`crate::core::engine::OnnxInferenceEngine`]，仅提供固有方法
//!   `match_images` / `match_batch`；
//! - 双输入推理按名称经 `session.run` 传参
//!   绑定输入（语义一致）；
//! - 模型层的 `LightGlueBatchResult` 未从 `crate::model` 重导出（引擎层无法命名
//!   该类型），故 `match_batch` 直接返回 `Vec<LightGlueMatchResult>`，
//!   返回值长度即批次大小。

use std::collections::HashSet;

use ort::tensor::TensorElementType;
use ort::value::{DynValue, Tensor, ValueType};

use crate::core::base::BaseOnnxEngine;
use crate::core::device_type::DeviceType;
use crate::error::{Result, VisionError};
use crate::imaging::{cvt_color, resize, ColorConversion, Image, Interpolation};
use crate::model::{LightGlueFeatureMatch, LightGlueKeyPoint, LightGlueMatchResult};

/// LightGlue 特征匹配推理引擎。
pub struct LightGlueEngine {
    /// 组合基类
    pub base: BaseOnnxEngine,

    /// 匹配分数阈值（默认 0.0 表示不过滤）
    match_threshold: f32,

    /// 是否为双输入模型（image0 + image1）
    dual_input: bool,

    /// 第二个输入的名称（双输入模式）
    input1_name: Option<String>,
}

impl LightGlueEngine {
    /// 创建 LightGlue 引擎（匹配阈值默认 0.0，输入尺寸从模型读取，动态维度回退 640）。
    pub fn new(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        Self::with_match_threshold(model_path, device_type, 0.0)
    }

    /// 指定匹配分数阈值创建。
    pub fn with_match_threshold(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        match_threshold: f32,
    ) -> Result<Self> {
        Self::with_input_size(model_path, device_type, match_threshold, -1, -1)
    }

    /// 指定匹配分数阈值与输入尺寸创建（input_height/input_width <=0 时从模型读取）。
    pub fn with_input_size(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        match_threshold: f32,
        input_height: i32,
        input_width: i32,
    ) -> Result<Self> {
        let base =
            BaseOnnxEngine::with_input_size(model_path, device_type, input_height, input_width)?;

        // 检测模型输入格式
        let (dual_input, input1_name) = {
            let session = base.session.lock().unwrap();
            if session.inputs.len() >= 2 {
                // 获取第二个输入名
                let name = session.inputs[1].name.clone();
                (true, Some(name))
            } else {
                (false, None)
            }
        };

        tracing::info!(
            "LightGlue Engine initialized: dualInput={}, inputSize={}x{}",
            dual_input,
            base.input_width(),
            base.input_height()
        );

        Ok(LightGlueEngine {
            base,
            match_threshold,
            dual_input,
            input1_name,
        })
    }

    // ============ 访问器 ============

    /// 匹配分数阈值。
    pub fn match_threshold(&self) -> f32 {
        self.match_threshold
    }

    /// 是否为双输入模型（image0 + image1）。
    pub fn is_dual_input(&self) -> bool {
        self.dual_input
    }

    // ============ 推理入口 ============

    /// 匹配两张图像（单对）。
    ///
    /// - `img0`: 模板图像
    /// - `img1`: 场景图像
    pub fn match_images(&self, img0: &Image, img1: &Image) -> Result<LightGlueMatchResult> {
        if img0.is_empty() || img1.is_empty() {
            return Err(VisionError::invalid_argument(
                "Input images cannot be null or empty",
            ));
        }

        let orig_w0 = img0.width() as i32;
        let orig_h0 = img0.height() as i32;
        let orig_w1 = img1.width() as i32;
        let orig_h1 = img1.height() as i32;

        // 1. 预处理
        let pre0 = self.preprocess_gray(img0)?;
        let pre1 = self.preprocess_gray(img1)?;

        // 2. 运行推理
        let outputs = if self.dual_input {
            self.run_dual_input(&pre0.pixel_data, &pre1.pixel_data)?
        } else {
            self.run_single_input(&pre0.pixel_data, &pre1.pixel_data)?
        };

        // 3. 解析输出
        self.parse_output(&outputs, &pre0, &pre1, orig_w0, orig_h0, orig_w1, orig_h1)
    }

    /// 批量匹配多对图像。
    ///
    /// - `image_pairs`: 图像对列表 `[img0_0, img1_0, img0_1, img1_1, ...]`
    ///
    /// 双输入模式逐对推理；单输入模式合并为 `[2B, 1, H, W]` 批量推理。
    /// 返回值长度即批次大小（对应 `BatchResult.results`）。
    pub fn match_batch(&self, image_pairs: &[Image]) -> Result<Vec<LightGlueMatchResult>> {
        if image_pairs.len() % 2 != 0 {
            return Err(VisionError::invalid_argument(
                "imagePairs must contain even number of images",
            ));
        }
        let batch_size = image_pairs.len() / 2;
        if batch_size == 0 {
            return Ok(Vec::new());
        }

        if self.dual_input {
            // 双输入模式：逐对推理
            let mut results = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                results.push(self.match_images(&image_pairs[2 * i], &image_pairs[2 * i + 1])?);
            }
            return Ok(results);
        }

        // 单输入模式：批量推理 [2B, 1, H, W]
        let input_h = self.base.input_height() as usize;
        let input_w = self.base.input_width() as usize;
        let hw = input_h * input_w;

        let mut pre_results = Vec::with_capacity(2 * batch_size);
        let mut orig_w = Vec::with_capacity(2 * batch_size);
        let mut orig_h = Vec::with_capacity(2 * batch_size);
        let mut batch_data = Vec::with_capacity(2 * batch_size * hw);

        for img in image_pairs {
            orig_w.push(img.width() as i32);
            orig_h.push(img.height() as i32);
            let pre = self.preprocess_gray(img)?;
            batch_data.extend_from_slice(&pre.pixel_data);
            pre_results.push(pre);
        }

        let shape = vec![2 * batch_size as i64, 1, input_h as i64, input_w as i64];
        let input_tensor = Tensor::from_array((shape, batch_data))?;

        let mut session = self.base.session.lock().unwrap();
        let outputs = session.run(ort::inputs![self.base.input_name() => input_tensor])?;
        let snapshots = snapshot_outputs(&outputs)?;

        self.parse_batch_output(&snapshots, &pre_results, batch_size, &orig_w, &orig_h)
    }

    // ==================== 预处理 ====================

    /// 灰度化 -> Letterbox Resize（只缩小不放大） -> 归一化到 [0, 1]
    ///
    /// 关键策略：scale 上限为 1.0，原图比 inputSize 小的保持原尺寸居中 + padding，
    /// 避免放大产生插值伪影导致特征不可靠。
    fn preprocess_gray(&self, img: &Image) -> Result<PreprocessInfo> {
        // 1. 灰度化
        let gray = match img.channels() {
            3 => cvt_color(img, ColorConversion::Bgr2Gray)?,
            4 => cvt_color(img, ColorConversion::Bgra2Gray)?,
            _ => img.clone(),
        };

        let orig_h = gray.height() as i32;
        let orig_w = gray.width() as i32;
        let input_w = self.base.input_width();
        let input_h = self.base.input_height();

        // 2. Letterbox resize — scale 上限 1.0，只缩小不放大
        let scale = (input_w as f32 / orig_w as f32)
            .min(input_h as f32 / orig_h as f32)
            .min(1.0);
        let new_w = (orig_w as f32 * scale).round() as i32;
        let new_h = (orig_h as f32 * scale).round() as i32;

        let resized = if scale < 1.0 {
            resize(&gray, new_w as usize, new_h as usize, Interpolation::Linear)?
        } else {
            gray.clone()
        };

        // 3. 填充到 inputSize x inputSize
        let pad_top = (input_h - new_h) / 2;
        let pad_left = (input_w - new_w) / 2;

        let mut padded = Image::new(input_w as usize, input_h as usize, 1);
        padded.paste(pad_left.max(0) as usize, pad_top.max(0) as usize, &resized);

        // 4. 提取像素并归一化到 [0, 1]
        let area = input_h as usize * input_w as usize;
        let mut pixel_data = vec![0f32; area];
        for (i, &v) in padded.data().iter().enumerate() {
            pixel_data[i] = v as f32 / 255.0;
        }

        Ok(PreprocessInfo {
            pixel_data,
            original_h: orig_h,
            original_w: orig_w,
            scale,
            pad_top,
            pad_left,
        })
    }

    // ==================== 推理 ====================

    /// 双输入推理（两个张量 shape 均为 `[1, 1, H, W]`，按名称绑定）。
    fn run_dual_input(&self, data0: &[f32], data1: &[f32]) -> Result<Vec<RawTensor>> {
        let shape = vec![
            1,
            1,
            self.base.input_height() as i64,
            self.base.input_width() as i64,
        ];
        let tensor0 = Tensor::from_array((shape.clone(), data0.to_vec()))?;
        let tensor1 = Tensor::from_array((shape, data1.to_vec()))?;

        let input1_name = self.input1_name.clone().unwrap_or_default();
        let mut session = self.base.session.lock().unwrap();
        let outputs = session.run(ort::inputs![
            self.base.input_name() => tensor0,
            input1_name.as_str() => tensor1,
        ])?;
        snapshot_outputs(&outputs)
    }

    /// 单输入推理 `[2, 1, H, W]`。
    fn run_single_input(&self, data0: &[f32], data1: &[f32]) -> Result<Vec<RawTensor>> {
        let hw = self.base.input_height() as usize * self.base.input_width() as usize;
        let mut batch_data = Vec::with_capacity(2 * hw);
        batch_data.extend_from_slice(data0);
        batch_data.extend_from_slice(data1);

        let shape = vec![
            2,
            1,
            self.base.input_height() as i64,
            self.base.input_width() as i64,
        ];
        let input_tensor = Tensor::from_array((shape, batch_data))?;

        let mut session = self.base.session.lock().unwrap();
        let outputs = session.run(ort::inputs![self.base.input_name() => input_tensor])?;
        snapshot_outputs(&outputs)
    }

    // ==================== 输出解析 ====================

    /// 解析单对输出。
    ///
    /// LightGlue 标准输出:
    /// - keypoints: `(2, N, 2)` — 两图的关键点 `[x, y]`
    /// - matches: `(M, 2)` `[idx0, idx1]` 或 `(M, 3)` `[batch, idx0, idx1]`
    /// - mscores: `(M,)` — 匹配分数
    fn parse_output(
        &self,
        outputs: &[RawTensor],
        pre0: &PreprocessInfo,
        pre1: &PreprocessInfo,
        orig_w0: i32,
        orig_h0: i32,
        orig_w1: i32,
        orig_h1: i32,
    ) -> Result<LightGlueMatchResult> {
        if outputs.is_empty() {
            return Err(VisionError::inference("model returned no outputs"));
        }

        // 按名称找到输出张量；如果名称匹配失败，按位置取
        let kpts_tensor = find_output(outputs, "keypoints").unwrap_or(&outputs[0]);
        let matches_tensor = find_output(outputs, "matches").unwrap_or(
            outputs
                .get(1)
                .ok_or_else(|| VisionError::inference("model output count < 2"))?,
        );
        let scores_tensor = find_output(outputs, "mscores")
            .or_else(|| if outputs.len() > 2 { outputs.get(2) } else { None });

        // 解析 keypoints: (2, N, 2) — 支持 int64 和 float
        let kpd = parse_keypoints(kpts_tensor)?;

        // 解析 matches: (M, 2) 或 (M, 3)
        let md = parse_matches(Some(matches_tensor))?;

        // 解析 mscores: (M,)
        let scores = parse_scores(scores_tensor);

        // 构建关键点列表
        let mut keypoints0 = Vec::with_capacity(kpd.num_kpts0);
        let mut keypoints1 = Vec::with_capacity(kpd.num_kpts1);

        for i in 0..kpd.num_kpts0 {
            keypoints0.push(LightGlueKeyPoint::new(
                kpd.data0[i * 2],
                kpd.data0[i * 2 + 1],
                i as i32,
            ));
        }
        for i in 0..kpd.num_kpts1 {
            keypoints1.push(LightGlueKeyPoint::new(
                kpd.data1[i * 2],
                kpd.data1[i * 2 + 1],
                i as i32,
            ));
        }

        // 构建匹配列表（互斥匹配，确保一对一）
        let matches = self.build_matches(&keypoints0, &keypoints1, &md, scores.as_deref());

        Ok(LightGlueMatchResult {
            keypoints0,
            keypoints1,
            matches,
            image0_width: orig_w0,
            image0_height: orig_h0,
            image1_width: orig_w1,
            image1_height: orig_h1,
            scale0: pre0.scale,
            scale1: pre1.scale,
            pad_top0: pre0.pad_top,
            pad_left0: pre0.pad_left,
            pad_top1: pre1.pad_top,
            pad_left1: pre1.pad_left,
        })
    }

    /// 解析批量输出（keypoints: `(2B, N, 2)`）。
    fn parse_batch_output(
        &self,
        outputs: &[RawTensor],
        pre_results: &[PreprocessInfo],
        batch_size: usize,
        orig_w: &[i32],
        orig_h: &[i32],
    ) -> Result<Vec<LightGlueMatchResult>> {
        if outputs.is_empty() {
            return Err(VisionError::inference("model returned no outputs"));
        }
        let kpts_tensor = find_output(outputs, "keypoints").unwrap_or(&outputs[0]);
        let matches_tensor = find_output(outputs, "matches").unwrap_or(
            outputs
                .get(1)
                .ok_or_else(|| VisionError::inference("model output count < 2"))?,
        );
        let scores_tensor = find_output(outputs, "mscores")
            .or_else(|| if outputs.len() > 2 { outputs.get(2) } else { None });

        // keypoints: (2B, N, 2)
        let kpts_flat = &kpts_tensor.data;
        let kpts_shape = &kpts_tensor.shape;
        if kpts_shape.len() < 2 {
            return Err(VisionError::inference(format!(
                "Unexpected keypoints shape dims: {}, shape={kpts_shape:?}",
                kpts_shape.len()
            )));
        }
        let num_kpts = kpts_shape[1] as usize;

        // matches: (M, 2) 或 (M, 3)
        let md = parse_matches(Some(matches_tensor))?;

        // mscores: (M,)
        let scores = parse_scores(scores_tensor);

        let mut results = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let img0_idx = 2 * b;
            let img1_idx = 2 * b + 1;

            let mut kps0 = Vec::with_capacity(num_kpts);
            let mut kps1 = Vec::with_capacity(num_kpts);

            let offset0 = img0_idx * num_kpts * 2;
            let offset1 = img1_idx * num_kpts * 2;
            if kpts_flat.len() < offset1 + num_kpts * 2 {
                return Err(VisionError::inference(
                    "keypoints tensor too small for batch output",
                ));
            }

            for i in 0..num_kpts {
                kps0.push(LightGlueKeyPoint::new(
                    kpts_flat[offset0 + i * 2],
                    kpts_flat[offset0 + i * 2 + 1],
                    i as i32,
                ));
                kps1.push(LightGlueKeyPoint::new(
                    kpts_flat[offset1 + i * 2],
                    kpts_flat[offset1 + i * 2 + 1],
                    i as i32,
                ));
            }

            // 过滤当前 batch 的匹配
            let mut matches = Vec::new();
            let mut used0: HashSet<i32> = HashSet::new();
            let mut used1: HashSet<i32> = HashSet::new();

            for i in 0..md.num_matches {
                let batch_idx = md.batch_indices.as_ref().map(|bi| bi[i]).unwrap_or(0);
                if batch_idx != b as i32 {
                    continue;
                }

                let idx0 = md.indices0[i];
                let idx1 = md.indices1[i];
                let score = scores
                    .as_deref()
                    .and_then(|s| s.get(i).copied())
                    .unwrap_or(1.0);

                if used0.contains(&idx0) || used1.contains(&idx1) {
                    continue;
                }
                if idx0 >= 0
                    && (idx0 as usize) < kps0.len()
                    && idx1 >= 0
                    && (idx1 as usize) < kps1.len()
                    && score > self.match_threshold
                {
                    matches.push(LightGlueFeatureMatch {
                        kp0: kps0[idx0 as usize],
                        kp1: kps1[idx1 as usize],
                        score,
                    });
                    used0.insert(idx0);
                    used1.insert(idx1);
                }
            }

            let pre0 = pre_results
                .get(img0_idx)
                .ok_or_else(|| VisionError::inference("missing preprocess info for batch"))?;
            let pre1 = pre_results
                .get(img1_idx)
                .ok_or_else(|| VisionError::inference("missing preprocess info for batch"))?;

            results.push(LightGlueMatchResult {
                keypoints0: kps0,
                keypoints1: kps1,
                matches,
                image0_width: orig_w[img0_idx],
                image0_height: orig_h[img0_idx],
                image1_width: orig_w[img1_idx],
                image1_height: orig_h[img1_idx],
                scale0: pre0.scale,
                scale1: pre1.scale,
                pad_top0: pre0.pad_top,
                pad_left0: pre0.pad_left,
                pad_top1: pre1.pad_top,
                pad_left1: pre1.pad_left,
            });
        }

        Ok(results)
    }

    /// 构建互斥匹配列表（确保一对一匹配）。
    fn build_matches(
        &self,
        kps0: &[LightGlueKeyPoint],
        kps1: &[LightGlueKeyPoint],
        md: &MatchesData,
        scores: Option<&[f32]>,
    ) -> Vec<LightGlueFeatureMatch> {
        let mut matches = Vec::new();
        let mut used0: HashSet<i32> = HashSet::new();
        let mut used1: HashSet<i32> = HashSet::new();

        for i in 0..md.num_matches {
            let idx0 = md.indices0[i];
            let idx1 = md.indices1[i];
            let score = scores.and_then(|s| s.get(i).copied()).unwrap_or(1.0);

            if used0.contains(&idx0) || used1.contains(&idx1) {
                continue;
            }

            if idx0 >= 0
                && (idx0 as usize) < kps0.len()
                && idx1 >= 0
                && (idx1 as usize) < kps1.len()
                && score > self.match_threshold
            {
                matches.push(LightGlueFeatureMatch {
                    kp0: kps0[idx0 as usize],
                    kp1: kps1[idx1 as usize],
                    score,
                });
                used0.insert(idx0);
                used1.insert(idx1);
            }
        }

        matches
    }
}

// ==================== 张量解析辅助 ====================

/// 根据名称查找输出张量（对应 `findOutputByName`）。
fn find_output<'a>(outputs: &'a [RawTensor], name: &str) -> Option<&'a RawTensor> {
    outputs.iter().find(|t| t.name == name)
}

/// 打印张量调试信息（对应 `logTensorDebug`，以 debug 级别输出）。
fn log_tensor_debug(name: &str, t: &RawTensor) {
    tracing::debug!(
        "[LightGlue] {name}: shape={:?}, dtype={}, elements={}",
        t.shape,
        t.dtype,
        t.data.len()
    );

    // 打印前几个值用于调试
    let n = t.data.len().min(10);
    let sample: Vec<String> = t.data[..n].iter().map(|v| format!("{v:.1}")).collect();
    tracing::debug!("[LightGlue] {name} sample: {}", sample.join(" "));
}

/// 解析 keypoints 张量 — 支持 int64/float32，1D/3D/4D。
fn parse_keypoints(tensor: &RawTensor) -> Result<KeypointsData> {
    log_tensor_debug("keypoints", tensor);

    let mut kpd = KeypointsData::default();
    let shape = &tensor.shape;
    let flat = &tensor.data;

    match shape.len() {
        3 => {
            // (2, N, 2)
            let n = shape[1] as usize;
            kpd.num_kpts0 = n;
            kpd.num_kpts1 = n;
            if flat.len() < n * 2 {
                return Err(VisionError::inference(format!(
                    "keypoints tensor too small: {} elements, expect >= {}",
                    flat.len(),
                    n * 2
                )));
            }
            kpd.data0 = flat[..n * 2].to_vec();
            kpd.data1 = vec![0.0; n * 2];
            if flat.len() >= n * 4 {
                kpd.data1.copy_from_slice(&flat[n * 2..n * 4]);
            }
        }
        4 => {
            // (2, N, 1, 2) — squeeze dim2
            let n = shape[1] as usize;
            kpd.num_kpts0 = n;
            kpd.num_kpts1 = n;
            kpd.data0 = vec![0.0; n * 2];
            kpd.data1 = vec![0.0; n * 2];
            for i in 0..n {
                if i * 2 + 1 >= flat.len() {
                    break;
                }
                kpd.data0[i * 2] = flat[i * 2];
                kpd.data0[i * 2 + 1] = flat[i * 2 + 1];
            }
            let offset1 = n * 2;
            for i in 0..n {
                if offset1 + i * 2 + 1 >= flat.len() {
                    break;
                }
                kpd.data1[i * 2] = flat[offset1 + i * 2];
                kpd.data1[i * 2 + 1] = flat[offset1 + i * 2 + 1];
            }
        }
        1 => {
            // 1D flat: 假设 [img0_kpts..., img1_kpts...] 每图 N*2 个值
            let n = shape[0] as usize / 4; // 2 images * 2 coords
            kpd.num_kpts0 = n;
            kpd.num_kpts1 = n;
            kpd.data0 = vec![0.0; n * 2];
            kpd.data1 = vec![0.0; n * 2];
            let copy_len = (n * 2).min(flat.len());
            kpd.data0[..copy_len].copy_from_slice(&flat[..copy_len]);
            if flat.len() >= n * 4 {
                kpd.data1.copy_from_slice(&flat[n * 2..n * 4]);
            }
        }
        dims => {
            return Err(VisionError::inference(format!(
                "Unexpected keypoints shape dims: {dims}, shape={shape:?}"
            )));
        }
    }

    Ok(kpd)
}

/// 解析 matches 张量 — 支持 int64/float32，1D/(M,2)/(M,3) 格式。
fn parse_matches(tensor: Option<&RawTensor>) -> Result<MatchesData> {
    let mut md = MatchesData::default();
    let Some(t) = tensor else {
        tracing::debug!("[LightGlue] matches: null");
        return Ok(md);
    };
    log_tensor_debug("matches", t);

    let shape = &t.shape;
    let flat = &t.data;

    if shape.len() == 1 {
        // 1D flat — 需要判断是 (M,) 索引格式还是 (M*2)/(M*3) 展平格式
        if flat.is_empty() {
            return Ok(md);
        }

        // 检查是否是 (M, 3) 展平为 1D，即长度是 3 的倍数且值符合 [batch, idx0, idx1] 模式
        // 先尝试 (M, 3) 展平（上游优先按 %3 判断）
        if flat.len() % 3 == 0 {
            let m = flat.len() / 3;
            md.num_matches = m;
            let mut batch_indices = Vec::with_capacity(m);
            md.indices0 = Vec::with_capacity(m);
            md.indices1 = Vec::with_capacity(m);
            for i in 0..m {
                batch_indices.push(flat[i * 3] as i32);
                md.indices0.push(flat[i * 3 + 1] as i32);
                md.indices1.push(flat[i * 3 + 2] as i32);
            }
            md.batch_indices = Some(batch_indices);
        } else if flat.len() % 2 == 0 {
            // (M, 2)
            let m = flat.len() / 2;
            md.num_matches = m;
            md.batch_indices = None;
            for i in 0..m {
                md.indices0.push(flat[i * 2] as i32);
                md.indices1.push(flat[i * 2 + 1] as i32);
            }
        } else {
            // 单索引数组 (N,) — 每个值对应 image0 第 i 个关键点匹配到 image1 的索引，-1 不匹配
            md.num_matches = flat.iter().filter(|&&v| v >= 0.0).count();
            md.batch_indices = None;
            for (i, &v) in flat.iter().enumerate() {
                if v >= 0.0 {
                    md.indices0.push(i as i32);
                    md.indices1.push(v as i32);
                }
            }
        }
        return Ok(md);
    }

    // 2D: (M, 2) 或 (M, 3)
    let m = shape[0] as usize;
    let cols = if shape.len() >= 2 { shape[1] as usize } else { 2 };
    if flat.len() < m * cols {
        return Err(VisionError::inference(format!(
            "matches tensor too small: {} elements, expect >= {}x{}",
            flat.len(),
            m,
            cols
        )));
    }

    md.num_matches = m;
    md.indices0 = vec![0; m];
    md.indices1 = vec![0; m];
    md.batch_indices = if cols >= 3 { Some(vec![0; m]) } else { None };

    for i in 0..m {
        if cols == 2 {
            md.indices0[i] = flat[i * 2] as i32;
            md.indices1[i] = flat[i * 2 + 1] as i32;
        } else {
            md.batch_indices.as_mut().unwrap()[i] = flat[i * cols] as i32;
            md.indices0[i] = flat[i * cols + 1] as i32;
            md.indices1[i] = flat[i * cols + 2] as i32;
        }
    }

    Ok(md)
}

/// 解析 mscores 张量 — 支持 float32/int64。
fn parse_scores(tensor: Option<&RawTensor>) -> Option<Vec<f32>> {
    let t = match tensor {
        None => {
            tracing::debug!("[LightGlue] mscores: null");
            return None;
        }
        Some(t) => t,
    };
    log_tensor_debug("mscores", t);
    Some(t.data.clone())
}

// ==================== 内部数据结构 ====================

/// 预处理信息（对应 `PreprocessInfo`）。
// original_h/original_w 与原实现一致仅作记录（结果使用 scale/pad）。
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
struct PreprocessInfo {
    /// 归一化到 [0,1] 的像素数据（长度 H*W）
    pixel_data: Vec<f32>,
    original_h: i32,
    original_w: i32,
    scale: f32,
    pad_top: i32,
    pad_left: i32,
}

/// 关键点张量解析结果（对应 `KeypointsData`）。
#[derive(Debug, Default)]
struct KeypointsData {
    num_kpts0: usize,
    num_kpts1: usize,
    /// `[numKpts0 * 2]`
    data0: Vec<f32>,
    /// `[numKpts1 * 2]`
    data1: Vec<f32>,
}

/// 匹配张量解析结果（对应 `MatchesData`）。
#[derive(Debug, Default)]
struct MatchesData {
    num_matches: usize,
    indices0: Vec<i32>,
    indices1: Vec<i32>,
    /// 可空，仅 (M, 3) 格式
    batch_indices: Option<Vec<i32>>,
}

/// 输出张量快照（名称 + 形状 + 已转为 f32 的数据）。
#[derive(Debug, Clone)]
struct RawTensor {
    name: String,
    shape: Vec<i64>,
    /// 元素类型标签（用于调试日志）
    dtype: String,
    data: Vec<f32>,
}

/// 按 `output_names` 顺序把会话输出快照为 f32 张量。
///
/// 对应 `readTensorAsFloat` 的多类型兼容：float32 / int64 / int32
/// （LightGlue 模型的 keypoints/matches 通常是 int64，mscores 是 float）；
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

    let (dtype, data) = match ty {
        TensorElementType::Float32 => {
            let (_, view) = value.try_extract_tensor::<f32>()?;
            ("float32".to_string(), view.to_vec())
        }
        TensorElementType::Int64 => {
            let (_, view) = value.try_extract_tensor::<i64>()?;
            ("int64".to_string(), view.iter().map(|&v| v as f32).collect())
        }
        TensorElementType::Int32 => {
            let (_, view) = value.try_extract_tensor::<i32>()?;
            ("int32".to_string(), view.iter().map(|&v| v as f32).collect())
        }
        // 未知类型，尝试 float
        other => {
            let (_, view) = value.try_extract_tensor::<f32>()?;
            (format!("{other:?}"), view.to_vec())
        }
    };

    Ok(RawTensor {
        name: name.to_string(),
        shape,
        dtype,
        data,
    })
}
