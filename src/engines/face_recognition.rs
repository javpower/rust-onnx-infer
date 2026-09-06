//! ArcFace / MobileFaceNet 人脸识别引擎（insightface 风格 5 点对齐 + 512/128 维 embedding）。
//!
//! 预处理细节均已核实（两大家族归一化方式不同，用 [`FaceNormMode`] 区分）：
//!
//! - **InsightFace ArcFace / MobileFaceNet**（默认模式 [`FaceNormMode::InsightFace`]，
//!   对应 insightface 官方 `model_zoo/arcface_onnx.py`）：
//!   `blobFromImage(img, 1.0/127.5, (112,112), (127.5,127.5,127.5), swapRB=True)`
//!   即 **RGB 通道序 + (x - 127.5) / 127.5**，输出 embedding 做 L2 归一化；
//!   （常被写作 `(x-127.5)/128`，二者数值差异 <0.4%，本引擎按官方取 127.5）
//! - **OpenCV Zoo SFace**（[`FaceNormMode::SFaceGraph`]，对应仓库内
//!   `testmodels/face_recognition_sface_2021dec.onnx`）：归一化已折叠进计算图
//!   （首两层为 `(data - 127.5) * (1/128)`），因此输入为 **原始 0~255 RGB**；
//!   `blobFromImage(aligned, 1.0, (112,112), (0,0,0), swapRB=true)`，输出 128 维。
//!
//! **对齐**：按 5 关键点求解 2D 相似变换（旋转 + 等比缩放 + 平移，最小二乘闭式解，
//! 等价于 OpenCV `FaceRecognizerSF::alignCrop` 使用的 `getSimilarityTransformMatrix`），
//! 把人脸 warp 到 ArcFace 规范 5 点（112x112），逆向映射 + 双线性采样（边界复制）。
//! 关键点顺序为 crate 约定（left_eye/right_eye/nose_tip/left/right_mouth_corner，
//! 均以观察者视角、图像偏左者为先），与规范 5 点按索引一一对应。
//!
//! 输出 embedding 维度随模型（ArcFace [1,512]、SFace [1,128]），统做 L2 归一化；
//! 相似度用 [`FaceRecognitionEngine::cosine_similarity`]（或等价的向量点积）。

use crate::core::base::BaseOnnxEngine;
use crate::core::device_type::DeviceType;
use crate::error::{Result, VisionError};
use crate::imaging::{cvt_color, resize, ColorConversion, Image, Interpolation};
use crate::model::Keypoint;

/// 对齐后的规范输出边长。
const ALIGNED_SIZE: usize = 112;

/// ArcFace 规范 5 点（112x112；顺序 = crate 关键点约定：图像左眼/图像右眼/鼻尖/图像左嘴角/图像右嘴角）。
pub const ARCFACE_DST: [(f32, f32); 5] = [
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
];

/// 预处理模式（不同模型家族的归一化位置不同）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FaceNormMode {
    /// insightface ArcFace / MobileFaceNet：图外归一化，RGB + (x-127.5)/127.5（默认）
    #[default]
    InsightFace,
    /// OpenCV Zoo SFace：归一化 (x-127.5)/128 在计算图内，输入为原始 0~255 RGB
    SFaceGraph,
}

/// 人脸识别（特征提取）引擎。
pub struct FaceRecognitionEngine {
    /// 组合基类
    pub base: BaseOnnxEngine,
    /// 预处理模式
    norm_mode: FaceNormMode,
}

impl FaceRecognitionEngine {
    /// 创建 ArcFace / MobileFaceNet 人脸识别引擎（insightface 预处理约定）。
    ///
    /// 输入固定 112x112（模型为动态输入时也按 112x112 送入）。
    pub fn new(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        Self::with_norm_mode(model_path, device_type, FaceNormMode::InsightFace)
    }

    /// 创建 OpenCV Zoo SFace 人脸识别引擎（归一化在计算图内，输入原始 0~255 RGB）。
    ///
    /// 对应 `face_recognition_sface_2021dec.onnx`，输出 128 维 embedding。
    pub fn new_sface(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        Self::with_norm_mode(model_path, device_type, FaceNormMode::SFaceGraph)
    }

    /// 指定预处理模式创建。
    pub fn with_norm_mode(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        norm_mode: FaceNormMode,
    ) -> Result<Self> {
        let mut base = BaseOnnxEngine::with_input_size(model_path, device_type, ALIGNED_SIZE as i32, ALIGNED_SIZE as i32)?;
        // InsightFace 模式：基类 preprocess 为 (x/255 - mean)/std，
        // 取 mean=std=0.5 时恰为 (x - 127.5) / 127.5；SFace 模式走自定义原始值路径
        base.set_normalization([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]);
        Ok(FaceRecognitionEngine { base, norm_mode })
    }

    // ============ 访问器 ============

    /// 当前预处理模式。
    pub fn norm_mode(&self) -> FaceNormMode {
        self.norm_mode
    }

    /// 切换预处理模式。
    pub fn set_norm_mode(&mut self, mode: FaceNormMode) {
        self.norm_mode = mode;
    }

    // ============ 核心 API ============

    /// 按人脸 5 关键点对齐后提取特征（主 API）。
    ///
    /// `landmarks`：5 点，顺序 = crate 约定
    /// （left_eye / right_eye / nose_tip / left_mouth_corner / right_mouth_corner，
    /// 图像坐标，由 [`crate::engines::face_detection::FaceDetectionEngine`] 输出）。
    /// 返回 L2 归一化后的 embedding（维度随模型：ArcFace 512 / SFace 128）。
    pub fn extract(&self, image: &Image, landmarks: &[Keypoint]) -> Result<Vec<f32>> {
        if landmarks.len() < 5 {
            return Err(VisionError::invalid_argument(format!(
                "extract 需要 5 个人脸关键点，实际 {} 个",
                landmarks.len()
            )));
        }
        let warped = self.align_face(image, landmarks)?;
        self.embed(&warped)
    }

    /// 人脸对齐：按 5 点相似变换 warp 到 ArcFace 规范 5 点（112x112 BGR 图）。
    pub fn align_face(&self, image: &Image, landmarks: &[Keypoint]) -> Result<Image> {
        if image.is_empty() {
            return Err(VisionError::image("cannot align empty image"));
        }
        let mut src = [(0f32, 0f32); 5];
        for (dst_slot, kp) in src.iter_mut().zip(landmarks.iter().take(5)) {
            *dst_slot = (kp.x, kp.y);
        }
        // 求解 src → 规范点的相似变换（最小二乘闭式解）
        let (m, t) = solve_similarity_transform(&src, &ARCFACE_DST);
        let det = m[0] * m[3] - m[1] * m[2];
        if det.abs() < 1e-6 {
            return Err(VisionError::invalid_argument("人脸关键点共线，相似变换退化"));
        }
        // 逆矩阵（M = s·R，非奇异）
        let i00 = m[3] / det;
        let i01 = -m[1] / det;
        let i10 = -m[2] / det;
        let i11 = m[0] / det;

        let (w, h, c) = (image.width(), image.height(), image.channels());
        if c != 1 && c != 3 && c != 4 {
            return Err(VisionError::image(format!("unsupported channels: {c}")));
        }
        let px = image.data();
        let mut out = Image::new(ALIGNED_SIZE, ALIGNED_SIZE, 3);
        {
            let out_px = out.data_mut();
            for dy in 0..ALIGNED_SIZE {
                for dx in 0..ALIGNED_SIZE {
                    // 逆向映射：目标像素 → 源图坐标
                    let fx = i00 * (dx as f32 - t.0) + i01 * (dy as f32 - t.1);
                    let fy = i10 * (dx as f32 - t.0) + i11 * (dy as f32 - t.1);
                    // 双线性采样（边界复制）
                    let x0i = fx.floor() as isize;
                    let y0i = fy.floor() as isize;
                    let wx = fx - x0i as f32;
                    let wy = fy - y0i as f32;
                    let wmax = w as isize - 1;
                    let hmax = h as isize - 1;
                    let xa = x0i.clamp(0, wmax) as usize;
                    let xb = (x0i + 1).clamp(0, wmax) as usize;
                    let ya = y0i.clamp(0, hmax) as usize;
                    let yb = (y0i + 1).clamp(0, hmax) as usize;
                    for ch in 0..3usize {
                        // 灰度图时复制同一平面到 3 通道
                        let sc = ch.min(c - 1);
                        let v00 = px[(ya * w + xa) * c + sc] as f32;
                        let v01 = px[(ya * w + xb) * c + sc] as f32;
                        let v10 = px[(yb * w + xa) * c + sc] as f32;
                        let v11 = px[(yb * w + xb) * c + sc] as f32;
                        let v = v00 * (1.0 - wx) * (1.0 - wy)
                            + v01 * wx * (1.0 - wy)
                            + v10 * (1.0 - wx) * wy
                            + v11 * wx * wy;
                        out_px[(dy * ALIGNED_SIZE + dx) * 3 + ch] = v.round().clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
        Ok(out)
    }

    /// 整图无对齐特征提取（trait `predict` 语义；正式使用请优先 [`Self::extract`]）。
    pub fn predict_impl(&self, image: &Image) -> Result<Vec<f32>> {
        self.embed(image)
    }

    /// 计算两个 embedding 的余弦相似度（含维度校验，维度不一致时 panic）。
    ///
    /// 输入应为 [`Self::extract`] 返回的 L2 归一化向量；非归一化向量结果同样正确
    /// （内部会除以各自模长），范围 [-1, 1]。
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        assert!(
            a.len() == b.len(),
            "cosine_similarity: embedding 维度不一致 {} vs {}",
            a.len(),
            b.len()
        );
        let mut dot = 0f32;
        let mut norm_a = 0f32;
        let mut norm_b = 0f32;
        for (&va, &vb) in a.iter().zip(b.iter()) {
            dot += va * vb;
            norm_a += va * va;
            norm_b += vb * vb;
        }
        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom <= f32::EPSILON {
            return 0.0;
        }
        (dot / denom).clamp(-1.0, 1.0)
    }

    // ==================== 内部实现 ====================

    /// 预处理 + 推理 + L2 归一化。
    fn embed(&self, image: &Image) -> Result<Vec<f32>> {
        // 预处理（两种模式的归一化位置不同）
        let input_data = match self.norm_mode {
            FaceNormMode::InsightFace => {
                // 基类：拉伸 resize + BGR→RGB + (x/255 - 0.5)/0.5 = (x - 127.5)/127.5
                self.base.preprocess(image)?
            }
            FaceNormMode::SFaceGraph => {
                // 图内已做 (x-127.5)/128，输入为原始 0~255 RGB
                self.prepare_raw_rgb(image)?
            }
        };

        let input_tensor = self.base.create_input_tensor(input_data)?;
        let output = self.base.run_inference(input_tensor)?;
        let embedding = l2_normalize(output.as_f32()?.to_vec());
        Ok(embedding)
    }

    /// SFace 模式预处理：拉伸 resize 到 112x112 + BGR→RGB + 原始 0~255 CHW。
    fn prepare_raw_rgb(&self, image: &Image) -> Result<Vec<f32>> {
        let width = self.base.input_width() as usize;
        let height = self.base.input_height() as usize;
        let resized = resize(image, width, height, Interpolation::Linear)?;
        let rgb = match resized.channels() {
            4 => cvt_color(&resized, ColorConversion::Bgra2Rgb)?,
            3 => cvt_color(&resized, ColorConversion::Bgr2Rgb)?,
            _ => cvt_color(&resized, ColorConversion::Gray2Rgb)?,
        };
        let px = rgb.data();
        let area = height * width;
        let mut data = vec![0f32; 3 * area];
        for i in 0..area {
            data[i] = px[i * 3] as f32;
            data[i + area] = px[i * 3 + 1] as f32;
            data[i + 2 * area] = px[i * 3 + 2] as f32;
        }
        Ok(data)
    }
}

/// 求解 2D 相似变换（旋转 + 等比缩放 + 平移，无反射）最小二乘闭式解。
///
/// 对应 OpenCV `getSimilarityTransformMatrix` / skimage `SimilarityTransform` 的常规
/// （det>0）情形：把点视为复数，`c = Σ v·conj(u) / Σ|u|²`，`c = s·e^{iθ}`。
/// 返回 `(m, t)`：`dst = m·src + t`，`m = [m00, m01, m10, m11]`。
fn solve_similarity_transform(src: &[(f32, f32); 5], dst: &[(f32, f32); 5]) -> ([f32; 4], (f32, f32)) {
    let n = src.len();
    let mut src_mean = (0f32, 0f32);
    let mut dst_mean = (0f32, 0f32);
    for i in 0..n {
        src_mean.0 += src[i].0;
        src_mean.1 += src[i].1;
        dst_mean.0 += dst[i].0;
        dst_mean.1 += dst[i].1;
    }
    src_mean.0 /= n as f32;
    src_mean.1 /= n as f32;
    dst_mean.0 /= n as f32;
    dst_mean.1 /= n as f32;

    // 去质心后累计 a = Σ(v·conj(u)) 实部/虚部与 Σ|u|²
    let mut a = 0f32;
    let mut b = 0f32;
    let mut norm2 = 0f32;
    for i in 0..n {
        let (ux, uy) = (src[i].0 - src_mean.0, src[i].1 - src_mean.1);
        let (vx, vy) = (dst[i].0 - dst_mean.0, dst[i].1 - dst_mean.1);
        a += vx * ux + vy * uy;
        b += vy * ux - vx * uy;
        norm2 += ux * ux + uy * uy;
    }

    let r = (a * a + b * b).sqrt();
    if r < 1e-6 || norm2 < 1e-6 {
        // 退化（点重合/共线）：退化为纯平移
        return ([1.0, 0.0, 0.0, 1.0], (dst_mean.0 - src_mean.0, dst_mean.1 - src_mean.1));
    }
    let scale = r / norm2;
    let cos = a / r;
    let sin = b / r;
    let m = [scale * cos, -scale * sin, scale * sin, scale * cos];
    let t = (
        dst_mean.0 - (m[0] * src_mean.0 + m[1] * src_mean.1),
        dst_mean.1 - (m[2] * src_mean.0 + m[3] * src_mean.1),
    );
    (m, t)
}

/// L2 归一化（零向量保持为零）。
fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    v
}

crate::impl_engine_forward!(FaceRecognitionEngine, base, Vec<f32>,
    /// 单图推理：整图无对齐特征提取（拉伸 resize 到 112x112）；
    /// 5 点对齐提取请使用 [`FaceRecognitionEngine::extract`]。返回 L2 归一化 embedding。
    fn predict(&self, image: &Image) -> Result<Vec<f32>> {
        self.predict_impl(image)
    }
);
