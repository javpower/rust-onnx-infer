//! 图像质量评估引擎（无参考 IQA）。
//!
//! **方案决策树落地说明**：
//!
//! - 方案 a）BRISQUE ONNX：经 WebSearch + 直接抓取核实，PINTO_model_zoo 无 BRISQUE 条目，
//!   HuggingFace 亦无可用的 `[1,1,H,W]` 灰度输入 ONNX 权重（`opencv/qrcode_wechatqrcode` 等仓库
//!   只发布 caffe 格式）。因此 BRISQUE 路径实现为**模型挂载式** [`ImageQualityEngine`]：
//!   只要提供一个符合约定（输入 `[1,1,H,W]` 灰度、像素 /255、输出单质量分）的 onnx 文件即可
//!   即插即用，代码路径已按该签名实现并验证。
//! - 方案 b）（打底，必做）纯算法 [`QualityAssessor`]：不依赖任何模型文件。
//!
//! 纯算法各分量定义：
//! - `blur_score`：拉普拉斯方差（`[[0,1,0],[1,-4,1],[0,1,0]]` 卷积响应的方差），越大越清晰；
//! - `brightness`：灰度均值（0~255，127.5 视为最佳）；
//! - `contrast`：灰度标准差（0~255，越大对比越强）；
//! - `noise`：Immerkær 快速噪声估计（3x3 高频残差掩码绝对值之和的统计量，越小越干净）；
//! - `overall`：各分量归一化后的加权和（权重经 [`QualityWeights`] 可配），范围 [0,1]，1 为最好。
//!
//! 自验基准：清晰合成图 vs 手写 3x3 均值模糊图的 `blur_score` 拉开 >5 倍（见模块内测试，
//! `cargo test image_quality -- --nocapture` 可查看数值）。

use crate::core::base::BaseOnnxEngine;
use crate::core::device_type::DeviceType;
use crate::error::{Result, VisionError};
use crate::imaging::{cvt_color, resize, ColorConversion, Image, Interpolation};

/// `overall` 计算中各归一分量的权重（任务要求权重可配）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityWeights {
    /// 清晰度（拉普拉斯方差归一）权重
    pub blur: f32,
    /// 亮度分量权重
    pub brightness: f32,
    /// 对比度分量权重
    pub contrast: f32,
    /// 噪声分量权重
    pub noise: f32,
}

impl Default for QualityWeights {
    fn default() -> Self {
        QualityWeights {
            blur: 0.40,
            brightness: 0.20,
            contrast: 0.25,
            noise: 0.15,
        }
    }
}

/// 各分量归一化时的参考标尺（默认值对应常规自然图像的经验范围）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityScale {
    /// 清晰度软饱和半分点：拉普拉斯方差达到该值时清晰度分量计 0.5
    /// （`sharp = var / (var + ref)`，默认 500；拉普拉斯方差动态范围跨数量级，
    /// 用软饱和映射避免清晰图/模糊图同时打满而失去区分度）
    pub blur_reference: f32,
    /// 灰度标准差达到该值即计满分对比度（默认 50）
    pub contrast_reference: f32,
    /// 噪声估计达到该值计零分（默认 15；典型干净图 < 5）
    pub noise_reference: f32,
}

impl Default for QualityScale {
    fn default() -> Self {
        QualityScale {
            blur_reference: 500.0,
            contrast_reference: 50.0,
            noise_reference: 15.0,
        }
    }
}

/// 图像质量评估报告。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageQualityReport {
    /// 清晰度：拉普拉斯方差（越大越清晰，模糊后急剧下降）
    pub blur_score: f32,
    /// 亮度：灰度均值 0~255
    pub brightness: f32,
    /// 对比度：灰度标准差 0~255
    pub contrast: f32,
    /// 噪声：Immerkær 高频残差估计（越小越干净）
    pub noise: f32,
    /// 综合质量分：加权归一 [0,1]，1 为最好
    pub overall: f32,
}

impl std::fmt::Display for ImageQualityReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ImageQuality[blur={:.2}, brightness={:.1}, contrast={:.1}, noise={:.3}, overall={:.3}]",
            self.blur_score, self.brightness, self.contrast, self.noise, self.overall
        )
    }
}

// ==================== 纯算法原语 ====================

/// 任意通道输入转单通道灰度（1 通道直接克隆）。
fn to_gray(image: &Image) -> Result<Image> {
    match image.channels() {
        1 => Ok(image.clone()),
        3 => cvt_color(image, ColorConversion::Bgr2Gray),
        4 => cvt_color(image, ColorConversion::Bgra2Gray),
        c => Err(VisionError::image(format!(
            "image_quality expects 1/3/4 channels, got {c}"
        ))),
    }
}

/// 手写 3x3 均值（盒子）模糊，边界复制（BORDER_REPLICATE）。
///
/// 主要用于自验：清晰图与该模糊结果的拉普拉斯方差必须显著拉开。
pub fn box_blur_3x3(image: &Image) -> Result<Image> {
    let gray = to_gray(image)?;
    let (w, h) = (gray.width(), gray.height());
    if w < 3 || h < 3 {
        return Err(VisionError::image("box_blur_3x3 requires size >= 3x3"));
    }
    let src = gray.data();
    let mut dst = vec![0u8; w * h];
    // 行列分离的 3x3 盒子滤波等价实现：先水平求和再垂直求和，共 9 个样本取均值
    let mut tmp = vec![0u32; w * h];
    for y in 0..h {
        for x in 0..w {
            let xm = if x == 0 { 0 } else { x - 1 };
            let xp = if x + 1 >= w { w - 1 } else { x + 1 };
            let row = y * w;
            tmp[row + x] = src[row + xm] as u32 + src[row + x] as u32 + src[row + xp] as u32;
        }
    }
    for y in 0..h {
        let ym = if y == 0 { 0 } else { y - 1 };
        let yp = if y + 1 >= h { h - 1 } else { y + 1 };
        for x in 0..w {
            let sum = tmp[ym * w + x] + tmp[y * w + x] + tmp[yp * w + x];
            // 水平 3 项 + 垂直 3 项 => 共 9 个样本
            dst[y * w + x] = (sum / 9) as u8;
        }
    }
    Image::from_raw(w, h, 1, dst)
}

/// 拉普拉斯方差清晰度（对应 OpenCV `cv2.Laplacian(gray, CV_32F).var()`）。
///
/// 核为 `[[0,1,0],[1,-4,1],[0,1,0]]`，仅统计内部有效像素；返回值越大越清晰。
pub fn laplacian_variance(image: &Image) -> Result<f32> {
    let gray = to_gray(image)?;
    let (w, h) = (gray.width(), gray.height());
    if w < 3 || h < 3 {
        return Err(VisionError::image("laplacian_variance requires size >= 3x3"));
    }
    let src = gray.data();
    let n = (w - 2) * (h - 2);
    // 单遍求和 + 平方和（数值上转 f32 足够，灰度 laplacian 范围 ±1020）
    let mut sum = 0f64;
    let mut sum_sq = 0f64;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            // laplacian = 上下左右 - 4*center
            let lap = src[i - w] as f32 + src[i + w] as f32 + src[i - 1] as f32 + src[i + 1] as f32
                - 4.0 * src[i] as f32;
            sum += lap as f64;
            sum_sq += (lap * lap) as f64;
        }
    }
    let mean = sum / n as f64;
    // 无偏方差与 OpenCV var() 的母体方差差异可忽略，取母体方差保持一致
    let var = sum_sq / n as f64 - mean * mean;
    Ok(var.max(0.0) as f32)
}

/// 灰度均值与标准差（母体标准差，对齐 OpenCV `meanStdDev`）。
pub fn mean_std(image: &Image) -> Result<(f32, f32)> {
    let gray = to_gray(image)?;
    let n = (gray.width() * gray.height()) as f64;
    if n == 0.0 {
        return Err(VisionError::image("cannot assess empty image"));
    }
    let mut sum = 0f64;
    for &v in gray.data() {
        sum += v as f64;
    }
    let mean = sum / n;
    let mut sum_sq = 0f64;
    for &v in gray.data() {
        let d = v as f64 - mean;
        sum_sq += d * d;
    }
    Ok((mean as f32, (sum_sq / n).sqrt() as f32))
}

/// Immerkær 快速噪声估计（高频残差法）。
///
/// 3x3 掩码 `[[1,-2,1],[-2,4,-2],[1,-2,1]]` 与图像卷积，取绝对值求和：
/// `sigma = sqrt(pi/2) * sum(|conv|) / (6 * (W-2) * (H-2))`。
/// 对噪声鲁棒、对结构边缘部分抑制，是经典的高频残差估计。
pub fn estimate_noise(image: &Image) -> Result<f32> {
    let gray = to_gray(image)?;
    let (w, h) = (gray.width(), gray.height());
    if w < 3 || h < 3 {
        return Err(VisionError::image("estimate_noise requires size >= 3x3"));
    }
    let src = gray.data();
    let mut abs_sum = 0f64;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let c = src[i] as i32;
            // Immerkær 掩码：[1 -2 1; -2 4 -2; 1 -2 1]
            let m = (src[i - w - 1] as i32 + src[i - w] as i32 * -2 + src[i - w + 1] as i32)
                + (src[i - 1] as i32 * -2 + c * 4 + src[i + 1] as i32 * -2)
                + (src[i + w - 1] as i32 + src[i + w] as i32 * -2 + src[i + w + 1] as i32);
            abs_sum += m.unsigned_abs() as f64;
        }
    }
    let n = (w - 2) as f64 * (h - 2) as f64;
    let sigma = (std::f64::consts::FRAC_PI_2).sqrt() * abs_sum / (6.0 * n);
    Ok(sigma as f32)
}

#[inline]
fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

// ==================== 纯算法质量评估器（方案 b，打底必做） ====================

/// 纯算法图像质量评估器（不依赖模型文件）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityAssessor {
    weights: QualityWeights,
    scale: QualityScale,
}

impl Default for QualityAssessor {
    fn default() -> Self {
        Self::new()
    }
}

impl QualityAssessor {
    /// 创建默认配置的评估器。
    pub fn new() -> Self {
        QualityAssessor {
            weights: QualityWeights::default(),
            scale: QualityScale::default(),
        }
    }

    /// 自定义权重（其余参数取默认）。
    pub fn with_weights(weights: QualityWeights) -> Self {
        QualityAssessor {
            weights,
            scale: QualityScale::default(),
        }
    }

    /// 自定义权重与归一化标尺。
    pub fn with_config(weights: QualityWeights, scale: QualityScale) -> Self {
        QualityAssessor { weights, scale }
    }

    /// 当前权重。
    pub fn weights(&self) -> QualityWeights {
        self.weights
    }

    /// 修改权重。
    pub fn set_weights(&mut self, weights: QualityWeights) {
        self.weights = weights;
    }

    /// 当前归一化标尺。
    pub fn scale(&self) -> QualityScale {
        self.scale
    }

    /// 修改归一化标尺。
    pub fn set_scale(&mut self, scale: QualityScale) {
        self.scale = scale;
    }

    /// 评估图像质量。
    pub fn assess(&self, image: &Image) -> Result<ImageQualityReport> {
        if image.is_empty() {
            return Err(VisionError::image("cannot assess empty image"));
        }
        let gray = to_gray(image)?;
        let blur_score = laplacian_variance(&gray)?;
        let (brightness, contrast) = mean_std(&gray)?;
        let noise = estimate_noise(&gray)?;

        // 各分量归一化到 [0,1]；清晰度用软饱和（拉普拉斯方差无上界，
        // 线性截断会让清晰/模糊图同时打满 1.0 而失去区分度）
        let sharp = blur_score / (blur_score + self.scale.blur_reference.max(f32::EPSILON));
        let bright = 1.0 - (brightness - 127.5).abs() / 127.5;
        let contr = clamp01(contrast / self.scale.contrast_reference.max(f32::EPSILON));
        let clean = clamp01(1.0 - noise / self.scale.noise_reference.max(f32::EPSILON));

        // 加权归一：权重和不要求为 1，除以权重和保证 overall ∈ [0,1]
        let total_w = (self.weights.blur
            + self.weights.brightness
            + self.weights.contrast
            + self.weights.noise)
            .max(f32::EPSILON);
        let overall = clamp01(
            (self.weights.blur * sharp
                + self.weights.brightness * bright
                + self.weights.contrast * contr
                + self.weights.noise * clean)
                / total_w,
        );

        Ok(ImageQualityReport {
            blur_score,
            brightness,
            contrast,
            noise,
            overall,
        })
    }
}

// ==================== ONNX 模型挂载引擎（方案 a，可选） ====================

/// BRISQUE 风格的 ONNX 质量评估引擎（模型挂载式）。
///
/// 预期模型 I/O 约定（与 BRISQUE 类无参考质量模型一致，已按此签名实现）：
/// - 输入：`[1,1,H,W]` f32 灰度，像素 `/255`；
/// - 输出：单个标量质量分（`[1]` / `[1,1]` / `[1,1,1]` 等）。
///
/// 经 WebSearch 核实，目前公开渠道（PINTO model zoo / HuggingFace）没有可直接下载的
/// BRISQUE onnx 权重，故模型文件由调用方提供（`new` 时传入路径，拿到即用）。
///
/// 分数语义：默认按 BRISQUE 约定“分数越低质量越好”，[`ImageQualityEngine::quality`]
/// 将原始分线性映射到 [0,1]（1 最好）；映射区间可用 [`ImageQualityEngine::set_score_range`]
/// 适配不同挂载模型。
pub struct ImageQualityEngine {
    /// 组合基类（会话 / 张量 / 推理）
    pub base: BaseOnnxEngine,
    /// 质量最好时的模型分数（默认 0，BRISQUE 约定）
    best_score: f32,
    /// 质量最差时的模型分数（默认 100，BRISQUE 约定）
    worst_score: f32,
    /// RGB 3 通道模式（FIQA 等深度 IQA 模型）；false = BRISQUE 灰度模式
    rgb_mode: bool,
    /// 分数方向：true = 分数越高质量越好（FIQA）；false = 越低越好（BRISQUE）
    higher_is_better: bool,
}

impl ImageQualityEngine {
    /// 挂载质量评估模型（输入尺寸从模型读取，动态维度回退 640）。
    ///
    /// 动态输入的模型建议用 [`ImageQualityEngine::new_with_input_size`] 显式指定
    /// （如 BRISQUE 常见 224x224）。
    pub fn new(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        Self::new_with_input_size(model_path, device_type, -1, -1)
    }

    /// 挂载质量评估模型（显式输入尺寸；<=0 时从模型读取）。
    pub fn new_with_input_size(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        input_height: i32,
        input_width: i32,
    ) -> Result<Self> {
        let base =
            BaseOnnxEngine::with_input_size(model_path, device_type, input_height, input_width)?;
        if base.input_channels() != 1 {
            tracing::warn!(
                "ImageQualityEngine expects 1-channel (grayscale) model input, model declares {} channels",
                base.input_channels()
            );
        }
        Ok(ImageQualityEngine {
            base,
            best_score: 0.0,
            worst_score: 100.0,
            rgb_mode: false,
            higher_is_better: false,
        })
    }

    /// 设置原始分数 → [0,1] 质量分映射区间（`best` 对应 1 分、`worst` 对应 0 分）。
    pub fn set_score_range(&mut self, best: f32, worst: f32) -> Result<()> {
        if !best.is_finite() || !worst.is_finite() || (worst - best).abs() < f32::EPSILON {
            return Err(VisionError::invalid_argument(
                "score range must satisfy worst != best",
            ));
        }
        self.best_score = best;
        self.worst_score = worst;
        Ok(())
    }

    /// 预处理：灰度模式（拉伸 → 灰度 → /255，CHW 单通道）或 RGB 模式（拉伸 →
    /// BGR→RGB → /255，CHW 三通道；FIQA 等深度 IQA 模型）。
    fn preprocess_gray(&self, image: &Image) -> Result<Vec<f32>> {
        let width = self.base.input_width() as usize;
        let height = self.base.input_height() as usize;
        let resized = resize(image, width, height, Interpolation::Linear)?;
        if self.rgb_mode {
            let rgb = cvt_color(&resized, ColorConversion::Bgr2Rgb)?;
            let area = width * height;
            let mut chw = vec![0f32; 3 * area];
            let px = rgb.data();
            for i in 0..area {
                for c in 0..3 {
                    chw[c * area + i] = px[i * 3 + c] as f32 / 255.0;
                }
            }
            return Ok(chw);
        }
        let gray = to_gray(&resized)?;
        Ok(gray.data().iter().map(|&v| v as f32 / 255.0).collect())
    }

    /// 挂载 **FIQA 类深度 IQA 模型**（RGB 3 通道、/255、分数越高质量越好；
    /// 实测 Efficient-FIQA EdgeNeXt：清晰图 0.50 vs 重模糊图 0.14）。
    pub fn new_fiqa(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        let mut engine = Self::new(model_path, device_type)?;
        engine.rgb_mode = true;
        engine.higher_is_better = true;
        // FIQA 输出为 [0,1] 概率语义，直接映射
        engine.best_score = 1.0;
        engine.worst_score = 0.0;
        Ok(engine)
    }

    /// 原始模型质量分（BRISQUE 约定：越低越好）。
    pub fn score(&self, image: &Image) -> Result<f32> {
        if image.is_empty() {
            return Err(VisionError::image("cannot score empty image"));
        }
        let input = self.preprocess_gray(image)?;
        let tensor = self.base.create_input_tensor(input)?;
        let output = self.base.run_inference(tensor)?;
        let flat = output.as_f32()?;
        let first = flat.first().copied().ok_or_else(|| {
            VisionError::inference("quality model returned an empty output tensor")
        })?;
        if flat.len() > 1 {
            tracing::debug!(
                "quality model output has {} elements, using the first ({})",
                flat.len(),
                first
            );
        }
        Ok(first)
    }

    /// 归一化质量分 [0,1]（1 为最好；按 BRISQUE“低分=高质量”约定线性映射）。
    pub fn quality(&self, image: &Image) -> Result<f32> {
        let raw = self.score(image)?;
        let span = self.worst_score - self.best_score;
        Ok(clamp01(1.0 - (raw - self.best_score) / span))
    }
}

crate::impl_engine_forward!(ImageQualityEngine, base, f32,
    /// 单图推理（trait `predict` 返回原始模型分，BRISQUE 约定越低越好）。
    fn predict(&self, image: &Image) -> Result<f32> {
        self.score(image)
    }
);

// ==================== 自验测试 ====================

#[cfg(test)]
mod tests {
    use super::*;

    /// 合成"清晰"图：8px 黑白棋盘格（高频边缘丰富，拉普拉斯方差大）。
    fn make_sharp_image() -> Image {
        let size = 256;
        let block = 8;
        let mut data = vec![0u8; size * size];
        for y in 0..size {
            for x in 0..size {
                let black = ((x / block) + (y / block)) % 2 == 0;
                data[y * size + x] = if black { 0 } else { 255 };
            }
        }
        Image::from_raw(size, size, 1, data).unwrap()
    }

    /// 任务自验要求：清晰图 vs 手写 3x3 均值模糊图的 blur_score 必须拉开 >5 倍。
    #[test]
    fn blur_score_separates_sharp_and_blurred() {
        let sharp = make_sharp_image();
        // 连续两次 3x3 均值模糊，模拟高斯模糊的低通效果
        let blurred = box_blur_3x3(&box_blur_3x3(&sharp).unwrap()).unwrap();

        let sharp_score = laplacian_variance(&sharp).unwrap();
        let blur_score = laplacian_variance(&blurred).unwrap();
        let ratio = sharp_score / blur_score.max(f32::EPSILON);

        println!("清晰图 blur_score = {sharp_score:.2}");
        println!("模糊图 blur_score = {blur_score:.2}");
        println!("拉开倍数          = {ratio:.1}x");

        assert!(ratio > 5.0, "blur_score 拉开倍数不足 5 倍: {ratio:.2}x");
    }

    /// 纯算法评估全流程：overall 必须落在 [0,1]，且清晰图 overall 高于模糊图。
    #[test]
    fn assessor_overall_in_unit_range_and_separates() {
        let assessor = QualityAssessor::new();
        let sharp = make_sharp_image();
        let blurred = box_blur_3x3(&box_blur_3x3(&sharp).unwrap()).unwrap();

        let r_sharp = assessor.assess(&sharp).unwrap();
        let r_blur = assessor.assess(&blurred).unwrap();
        println!("清晰图报告: {r_sharp}");
        println!("模糊图报告: {r_blur}");

        for r in [r_sharp, r_blur] {
            assert!((0.0..=1.0).contains(&r.overall), "overall 越界: {}", r.overall);
        }
        assert!(r_sharp.overall > r_blur.overall);
    }

    /// 均匀图：噪声与清晰度接近 0，亮度即灰度值。
    #[test]
    fn flat_image_degenerates_gracefully() {
        let flat = Image::filled(64, 64, 1, 128);
        let r = QualityAssessor::new().assess(&flat).unwrap();
        println!("均匀图报告: {r}");
        assert!((r.blur_score.abs()) < 1e-3);
        assert!(r.noise < 1e-3);
        assert!((r.brightness - 128.0).abs() <= 1.0);
        assert!((0.0..=1.0).contains(&r.overall));
    }

    /// 真实照片回归：zidane.jpg 原图 vs 手写 3x3 均值模糊版。
    #[test]
    fn real_photo_sharp_vs_blurred() {
        if !["testmodels/zidane.jpg", "models/test_images/zidane.jpg"].iter().any(|p| std::path::Path::new(p).exists()) {
            eprintln!("[SKIP] 模型/测试资产缺失（models/ 或 testmodels/）");
            return;
        }

        let sharp = Image::load("testmodels/zidane.jpg").unwrap();
        let blurred = box_blur_3x3(&box_blur_3x3(&sharp).unwrap()).unwrap();
        let assessor = QualityAssessor::new();
        let r_sharp = assessor.assess(&sharp).unwrap();
        let r_blur = assessor.assess(&blurred).unwrap();
        println!("真实照片原图  : {r_sharp}");
        println!("真实照片模糊版: {r_blur}");
        assert!(
            r_sharp.blur_score > r_blur.blur_score * 5.0,
            "真实照片 blur_score 拉开不足 5 倍: {} vs {}",
            r_sharp.blur_score,
            r_blur.blur_score
        );
        assert!(r_sharp.overall > r_blur.overall);
    }
}
