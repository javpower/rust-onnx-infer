//! 静态手势分类引擎（基于手部 21 关键点）。
//!
//! 提供两条互补路线：
//!
//! **路线 a —— ONNX 模型（[`GestureEngine`]）**：加载"关键点 → 手势类别"的
//! 分类 ONNX（输入 `[1,63]` / `[1,42]` 展平向量或 `[1,21,3]` 逐点张量，
//! 输出各类别 logits）。截至 2026-09 全网检索未找到符合 MediaPipe 7 类约定
//! （Closed_Fist / Open_Palm / Pointing_Up / Thumb_Down / Thumb_Up / Victory /
//! I_Love_You）且可直连下载的 ONNX：
//! - MediaPipe 官方 `gesture_recognizer.task` 仅 TFLite 打包（需 tf2onnx 自行转换）；
//! - `qualcomm/MediaPipe-Hand-Gesture-Recognition`（HF）仓库无模型权重文件；
//! - `PINTO0309/hand-gesture-recognition-using-onnx` 的 keypoint_classifier 仅 3 类
//!   （Open/Close/Pointer），非 7 类约定；
//! - `osamajan90/gesture-recognition`（HF）为 42 维输入 / 18 类 HaGRID MLP，
//!   实测对绝对坐标位置过拟合（合成开掌被误判为 stop_inverted、握拳误判为 call），
//!   不满足 7 类约定且精度不可信。
//!
//! 因此本引擎按"通用关键点分类器"实现（输入布局自适应、标签可注入），
//!   用户拿到符合上述签名的 ONNX（如自行 tf2onnx 转换 MediaPipe 官方模型）即可直连；
//!   在此之前请优先使用路线 b。
//!
//! **路线 b —— 纯几何规则（[`GestureClassifier`]，零模型依赖、必选兜底）**：
//! 按手指伸屈判定：
//! - 四指（食/中/无名/小）：指尖（8/12/16/20）到腕点（0）的距离与第二关节
//!   （6/10/14/18）到腕点的距离之比 > 阈值判为伸直（尺度不变）；
//! - 拇指：综合"张开度"（指尖 4 到小指根 17 的距离 / 关节 3 到 17 的距离）与
//!   "腕向延展度"（d(4,0)/d(2,0)）双重判据；
//! - 拇指方向（图像坐标系 y 向下）：关节 3 → 指尖 4 的向量向上为 Thumb_Up、
//!   向下为 Thumb_Down。
//!
//! 支持类别：`Fist`（全屈）、`Open_Palm`（全伸）、`Victory`（食+中）、
//!   `Pointing_Up`（仅食指）、`Thumb_Up`（仅拇指伸且朝上）、`Thumb_Down`
//!   （仅拇指伸且朝下）、`Three`（3 指）、`Four`（4 指）；无法命中时返回
//!   `Unknown`。打分为各规则的"命中度"（0~1，逐指伸屈置信度与目标模式的贴合均值）。
//!
//! # 串联示例（与手部引擎组合，`ignore`：涉及本机模型路径）
//!
//! ```ignore
//! use rust_onnx_infer::core::device_type::DeviceType;
//! use rust_onnx_infer::engines::gesture::GestureClassifier;
//! use rust_onnx_infer::engines::hand_keypoint::{HandDetectionEngine, HandLandmarkEngine};
//! use rust_onnx_infer::imaging::{Rect, load_image};
//!
//! let image = load_image("photo.jpg")?;
//!
//! // 1. 手掌检测（SSD anchor 解码，输出手部框）
//! let detector = HandDetectionEngine::new("testmodels/hand_palm_lite.onnx", DeviceType::Cpu)?;
//! let hands = detector.detect(&image)?;
//!
//! // 2. 对每只手裁剪区域推理 21 关键点（RTMPose-m-hand，SimCC 解码）
//! let landmarker = HandLandmarkEngine::new("testmodels/rtmpose_m_hand.onnx", DeviceType::Cpu)?;
//! let classifier = GestureClassifier::new();
//! for hand in &hands {
//!     let landmarks = landmarker.extract(&image, Rect::new(hand.x, hand.y, hand.width, hand.height))?;
//!     // 3. 21 关键点 → 手势类别（纯几何规则，无需模型）
//!     let gesture = classifier.classify(&landmarks.points)?;
//!     println!("gesture: {} ({:.2})", gesture.label, gesture.score);
//! }
//! ```

use crate::core::base::BaseOnnxEngine;
use crate::core::device_type::DeviceType;
use crate::error::{Result, VisionError};
use crate::model::Keypoint;

/// MediaPipe 官方手势识别器 7 类标签约定（索引即模型输出顺序）。
pub const GESTURE_LABELS: [&str; 7] = [
    "Closed_Fist",
    "Open_Palm",
    "Pointing_Up",
    "Thumb_Down",
    "Thumb_Up",
    "Victory",
    "I_Love_You",
];

/// 未命中任何规则时的标签。
pub const GESTURE_UNKNOWN: &str = "Unknown";

/// 手部关键点数（MediaPipe / COCO hand 21 点约定）。
const NUM_HAND_POINTS: usize = 21;

/// 手势预测结果。
#[derive(Debug, Clone, PartialEq)]
pub struct GesturePrediction {
    /// 手势标签（如 `Open_Palm`；未命中为 [`GESTURE_UNKNOWN`]）
    pub label: String,
    /// 命中度 [0,1]（规则路线：目标伸屈模式的贴合程度；模型路线：softmax 概率）
    pub score: f32,
}

impl GesturePrediction {
    pub fn new(label: impl Into<String>, score: f32) -> Self {
        GesturePrediction {
            label: label.into(),
            score,
        }
    }
}

// ============================================================================
// 路线 b：纯几何规则分类器
// ============================================================================

/// 两点欧氏距离。
fn dist(a: &Keypoint, b: &Keypoint) -> f32 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
}

/// 数值夹到 [0,1]。
fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

/// 线性过渡斜坡：v < lo → 0，v > hi → 1，中间线性。
fn ramp(v: f32, lo: f32, hi: f32) -> f32 {
    clamp01((v - lo) / (hi - lo))
}

/// 纯几何规则手势分类器（路线 b，无模型依赖）。
///
/// 判定完全基于 21 点的相对几何关系，尺度 / 平移不变（但不含旋转不变性，
/// 手部大幅倾斜时四指判定仍成立，拇指方向判定以图像坐标系为准）。
#[derive(Debug, Clone, Copy, Default)]
pub struct GestureClassifier {
    _private: (),
}

/// 四指（食/中/无名/小）的（指尖索引, 第二关节索引）对。
const FINGER_TIP_PIP: [(usize, usize); 4] = [(8, 6), (12, 10), (16, 14), (20, 18)];

/// 伸屈判定斜坡区间（腕距比值）：比值 < 1.0 视为完全屈，> 1.35 视为完全伸。
const EXT_LO: f32 = 1.0;
const EXT_HI: f32 = 1.35;

/// 拇指判据斜坡区间（张开度与腕向延展度共用）。
const THUMB_LO: f32 = 1.15;
const THUMB_HI: f32 = 1.5;

/// 拇指方向判据斜坡区间（方向余弦：1 为竖直向上，0 为水平，负为向下）。
const DIR_LO: f32 = 0.2;
const DIR_HI: f32 = 0.9;

/// 命中度低于该阈值视为未识别（返回 [`GESTURE_UNKNOWN`]）。
const MIN_HIT: f32 = 0.55;

impl GestureClassifier {
    /// 创建规则分类器。
    pub fn new() -> Self {
        GestureClassifier { _private: () }
    }

    /// 对 21 关键点分类静态手势。
    ///
    /// `keypoints` 顺序须为 MediaPipe 21 点约定（0=腕，1~4 拇指，5~8 食指，
    /// 9~12 中指，13~16 无名指，17~20 小指）；坐标可为任意像素尺度（内部相对化）。
    pub fn classify(&self, keypoints: &[Keypoint]) -> Result<GesturePrediction> {
        if keypoints.len() != NUM_HAND_POINTS {
            return Err(VisionError::invalid_argument(format!(
                "手势分类需要 21 个手部关键点（MediaPipe 约定），实际 {} 个",
                keypoints.len()
            )));
        }
        let e = self.extension_degrees(keypoints);

        // 各目标模式的命中度 = 逐指 (伸→e，屈→1-e) 的均值
        // [拇指, 食指, 中指, 无名指, 小指]，1 = 目标伸直，0 = 目标弯曲
        let patterns: [(&str, [f32; 5]); 9] = [
            ("Fist", [0.0, 0.0, 0.0, 0.0, 0.0]),
            ("Open_Palm", [1.0, 1.0, 1.0, 1.0, 1.0]),
            ("Victory", [0.0, 1.0, 1.0, 0.0, 0.0]),
            ("Pointing_Up", [0.0, 1.0, 0.0, 0.0, 0.0]),
            // 拇指单伸的方向在下方乘方向因子
            ("Thumb_Up", [1.0, 0.0, 0.0, 0.0, 0.0]),
            ("Thumb_Down", [1.0, 0.0, 0.0, 0.0, 0.0]),
            // 数字 3 的两种常见形态：食+中+无名（拇指收）/ 拇+食+中
            ("Three", [0.0, 1.0, 1.0, 1.0, 0.0]),
            ("Three", [1.0, 1.0, 1.0, 0.0, 0.0]),
            ("Four", [0.0, 1.0, 1.0, 1.0, 1.0]),
        ];

        // 拇指方向余弦：图像坐标 y 向下，(3.y - 4.y) > 0 表示指尖高于关节（朝上）
        let thumb_len = dist(&keypoints[3], &keypoints[4]).max(1e-6);
        let dir_cos = (keypoints[3].y - keypoints[4].y) / thumb_len;

        let mut best: (&str, f32) = (GESTURE_UNKNOWN, 0.0);
        for (label, pattern) in patterns {
            let hit: f32 = pattern
                .iter()
                .zip(e.iter())
                .map(|(&p, &v)| if p > 0.5 { v } else { 1.0 - v })
                .sum::<f32>()
                / pattern.len() as f32;
            // 拇指单伸的两类：命中度乘方向因子（朝上/朝下程度）
            let hit = match label {
                "Thumb_Up" => hit * ramp(dir_cos, DIR_LO, DIR_HI),
                "Thumb_Down" => hit * ramp(-dir_cos, DIR_LO, DIR_HI),
                _ => hit,
            };
            if hit > best.1 {
                best = (label, hit);
            }
        }

        if best.1 < MIN_HIT {
            return Ok(GesturePrediction::new(GESTURE_UNKNOWN, best.1));
        }
        Ok(GesturePrediction::new(best.0, best.1))
    }

    /// 五指伸屈置信度 [拇指, 食指, 中指, 无名指, 小指]，1 = 完全伸直，0 = 完全弯曲。
    fn extension_degrees(&self, pts: &[Keypoint]) -> [f32; 5] {
        let wrist = &pts[0];
        let mut e = [0f32; 5];

        // 拇指：双判据取大
        // 1) 张开度：指尖到小指根距离 vs 拇指关节(3)到小指根距离（握拳时拇指横贯掌心，比值 < 1）
        let spread = dist(&pts[4], &pts[17]) / dist(&pts[3], &pts[17]).max(1e-6);
        // 2) 腕向延展度：指尖到腕距离 vs 掌根关节(2)到腕距离
        let wrist_ext = dist(&pts[4], wrist) / dist(&pts[2], wrist).max(1e-6);
        e[0] = ramp(spread.max(wrist_ext), THUMB_LO, THUMB_HI);

        // 四指：指尖-腕距离 / 第二关节-腕距离（伸直时指尖显著更远）
        for (i, (tip, pip)) in FINGER_TIP_PIP.iter().enumerate() {
            let ratio = dist(&pts[*tip], wrist) / dist(&pts[*pip], wrist).max(1e-6);
            e[i + 1] = ramp(ratio, EXT_LO, EXT_HI);
        }
        e
    }
}

// ============================================================================
// 路线 a：ONNX 关键点分类引擎
// ============================================================================

/// ONNX 关键点手势分类引擎（路线 a）。
///
/// 对模型签名的要求（与 MediaPipe 约定对齐）：
/// - 输入：`[1,63]` / `[1,42]` / `[1,21,3]` / `[1,21,2]`，坐标为归一化关键点；
/// - 输出：`[1, N]` logits（N=7 时自动套用 [`GESTURE_LABELS`] 标签，
///   其余类别数须通过 [`OnnxInferenceEngine::set_labels`] 注入）。
///
/// 坐标归一化约定：输入坐标在 [0,1] 内时原样使用（MediaPipe 归一化坐标约定）；
/// 为像素坐标时自动按手部包围盒映射到 [0.2, 0.8] 规范框（平移/尺度不变）。
/// 注意：不同模型的训练预处理不一，实际精度以模型文档为准；无合格模型时请用
/// [`GestureClassifier`]（路线 b）。
pub struct GestureEngine {
    /// 组合基类
    pub base: BaseOnnxEngine,
    /// 每点特征维（2 = x,y；3 = x,y,z）
    feature_dim: usize,
    /// 输入张量 shape（含 batch 维；展平 [1,63]/[1,42] 与逐点 [1,21,3]/[1,21,2]
    /// 的内存布局一致，仅 shape 不同，故无需单独记录布局类型）
    input_shape: Vec<i64>,
}

impl GestureEngine {
    /// 创建手势分类引擎（从模型元信息自动识别输入布局）。
    pub fn new(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        let base = BaseOnnxEngine::new(model_path, device_type)?;

        // 读取真实输入 shape（平面/逐点输入不是 NCHW，基类解析不适用）
        let dims: Vec<i64> = {
            let session = base.session.lock().unwrap();
            let input = session.inputs().first().ok_or_else(|| {
                VisionError::inference("手势分类模型缺少输入定义")
            })?;
            match input.dtype() {
                ort::value::ValueType::Tensor { shape, .. } => shape.iter().copied().collect(),
                other => {
                    return Err(VisionError::inference(format!(
                        "手势分类模型输入应为张量，实际 {other:?}"
                    )))
                }
            }
        };
        let total: usize = dims
            .iter()
            .filter(|&&d| d > 0)
            .map(|&d| d as usize)
            .product();
        let feature_dim = if dims.len() >= 2 && dims[dims.len() - 1] > 0 {
            dims[dims.len() - 1] as usize
        } else {
            0
        };

        let feature_dim = match (dims.len(), total, feature_dim) {
            // 展平 [1,63]（x,y,z）或 [1,42]（x,y）
            (2, 63, _) => 3,
            (2, 42, _) => 2,
            // 逐点 [1,21,3] / [1,21,2]
            (3, 63, 3) => 3,
            (3, 42, 2) => 2,
            _ => {
                return Err(VisionError::invalid_argument(format!(
                    "不支持的手势分类输入 shape {dims:?}（期望 [1,63]/[1,42]/[1,21,3]/[1,21,2]）"
                )))
            }
        };

        Ok(GestureEngine {
            base,
            feature_dim,
            input_shape: dims.iter().map(|&d| if d > 0 { d } else { 1 }).collect(),
        })
    }

    /// 对 21 关键点推理手势类别。
    pub fn classify(&self, keypoints: &[Keypoint]) -> Result<GesturePrediction> {
        if keypoints.len() != NUM_HAND_POINTS {
            return Err(VisionError::invalid_argument(format!(
                "手势分类需要 21 个手部关键点（MediaPipe 约定），实际 {} 个",
                keypoints.len()
            )));
        }

        // 坐标归一化：像素坐标（任一维 > 1.5）按包围盒映射到 [0.2,0.8]；
        // 已归一化的坐标原样使用
        let pixel = keypoints.iter().any(|p| p.x.abs() > 1.5 || p.y.abs() > 1.5);
        let (xs, ys) = if pixel {
            let min_x = keypoints.iter().map(|p| p.x).fold(f32::INFINITY, f32::min);
            let min_y = keypoints.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
            let span = keypoints
                .iter()
                .map(|p| (p.x - min_x).max(p.y - min_y))
                .fold(f32::NEG_INFINITY, f32::max)
                .max(1e-6);
            keypoints
                .iter()
                .map(|p| (0.2 + 0.6 * (p.x - min_x) / span, 0.2 + 0.6 * (p.y - min_y) / span))
                .unzip::<_, _, Vec<f32>, Vec<f32>>()
        } else {
            keypoints
                .iter()
                .map(|p| (p.x, p.y))
                .unzip::<_, _, Vec<f32>, Vec<f32>>()
        };

        // 组装输入（逐点存放：Flat 展平与 PerPoint 张量的内存布局一致）
        let mut data = Vec::with_capacity(NUM_HAND_POINTS * self.feature_dim);
        for i in 0..NUM_HAND_POINTS {
            data.push(xs[i]);
            data.push(ys[i]);
            if self.feature_dim == 3 {
                // 2D 关键点无深度信息，z 置 0
                data.push(0.0);
            }
        }

        let tensor = ort::value::Tensor::from_array((self.input_shape.clone(), data))?;
        let output = self.base.run_inference(tensor)?;
        let logits = output.as_f32()?;
        if logits.is_empty() {
            return Err(VisionError::inference("手势分类模型输出为空"));
        }

        // softmax（数值稳定）
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&v| (v - max_logit).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|&v| v / sum).collect();

        let mut best_idx = 0usize;
        let mut best_p = f32::NEG_INFINITY;
        for (i, &p) in probs.iter().enumerate() {
            if p > best_p {
                best_p = p;
                best_idx = i;
            }
        }
        Ok(GesturePrediction::new(
            self.label_of(best_idx, logits.len()),
            best_p,
        ))
    }

    /// 类别索引 → 标签名：优先模型元数据标签，7 类输出时回退 MediaPipe 约定。
    fn label_of(&self, idx: usize, num_classes: usize) -> String {
        if let Some(labels) = self.base.labels() {
            if idx < labels.len() {
                return labels[idx].clone();
            }
        }
        if num_classes == GESTURE_LABELS.len() {
            return GESTURE_LABELS[idx].to_string();
        }
        idx.to_string()
    }
}

crate::impl_engine_forward!(GestureEngine, base, GesturePrediction,
    /// 手势分类的输入为 21 关键点而非图像，`predict` 不适用：
    /// 请先用 [`crate::engines::hand_keypoint::HandLandmarkEngine`] 提取关键点
    /// （见模块文档串联示例），再调用 [`GestureEngine::classify`] /
    /// [`GestureClassifier::classify`]。
    fn predict(&self, _image: &crate::imaging::Image) -> Result<GesturePrediction> {
        Err(VisionError::Unsupported(
            "GestureEngine 消费 21 关键点而非图像：请用 HandLandmarkEngine 提取关键点后调用 classify()".to_string(),
        ))
    }
);

// ============================================================================
// 自验：合成手部关键点（手写 21 点坐标，图像坐标 y 向下）
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 由分段坐标构造 21 点（score 置 1.0）。
    fn hand(wrist: (f32, f32), thumb: [(f32, f32); 4], index: [(f32, f32); 4],
            middle: [(f32, f32); 4], ring: [(f32, f32); 4], pinky: [(f32, f32); 4]) -> Vec<Keypoint> {
        let mut pts = vec![Keypoint::new(wrist.0, wrist.1, 1.0)];
        for seg in [thumb, index, middle, ring, pinky] {
            for (x, y) in seg {
                pts.push(Keypoint::new(x, y, 1.0));
            }
        }
        assert_eq!(pts.len(), 21);
        pts
    }

    /// 从已构造的手中截取一段关键点坐标（复用基准手的手指段）。
    fn seg(hand: &[Keypoint], r: std::ops::Range<usize>) -> [(f32, f32); 4] {
        let mut it = hand[r].iter().map(|p| (p.x, p.y));
        [
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
        ]
    }

    /// 张开手掌：五指伸直，拇指向左张开。
    fn open_palm() -> Vec<Keypoint> {
        hand(
            (0.50, 0.85),
            [(0.36, 0.80), (0.28, 0.73), (0.23, 0.66), (0.19, 0.59)],
            [(0.41, 0.62), (0.39, 0.50), (0.38, 0.41), (0.37, 0.33)],
            [(0.50, 0.60), (0.50, 0.46), (0.50, 0.36), (0.50, 0.27)],
            [(0.59, 0.62), (0.61, 0.49), (0.62, 0.40), (0.63, 0.32)],
            [(0.67, 0.66), (0.70, 0.56), (0.72, 0.49), (0.73, 0.43)],
        )
    }

    /// 握拳：四指卷曲、拇指横贯掌心。
    fn fist() -> Vec<Keypoint> {
        hand(
            (0.50, 0.85),
            [(0.38, 0.81), (0.32, 0.75), (0.35, 0.70), (0.42, 0.68)],
            [(0.41, 0.62), (0.41, 0.51), (0.43, 0.54), (0.45, 0.60)],
            [(0.50, 0.60), (0.50, 0.48), (0.50, 0.52), (0.50, 0.58)],
            [(0.59, 0.62), (0.59, 0.51), (0.58, 0.55), (0.57, 0.60)],
            [(0.67, 0.66), (0.67, 0.57), (0.66, 0.60), (0.65, 0.64)],
        )
    }

    /// Victory：食指 + 中指伸直。
    fn victory() -> Vec<Keypoint> {
        let f = fist();
        let o = open_palm();
        hand(
            (0.50, 0.85),
            seg(&f, 1..5),
            seg(&o, 5..9),
            seg(&o, 9..13),
            seg(&f, 13..17),
            seg(&f, 17..21),
        )
    }

    /// 指向：仅食指伸直。
    fn pointing() -> Vec<Keypoint> {
        let f = fist();
        let o = open_palm();
        hand(
            (0.50, 0.85),
            seg(&f, 1..5),
            seg(&o, 5..9),
            seg(&f, 9..13),
            seg(&f, 13..17),
            seg(&f, 17..21),
        )
    }

    /// 竖大拇指：仅拇指伸直且朝上。
    fn thumb_up() -> Vec<Keypoint> {
        let f = fist();
        hand(
            (0.50, 0.85),
            [(0.44, 0.80), (0.42, 0.70), (0.41, 0.60), (0.40, 0.48)],
            seg(&f, 5..9),
            seg(&f, 9..13),
            seg(&f, 13..17),
            seg(&f, 17..21),
        )
    }

    /// 三指（食 + 中 + 无名）。
    fn three() -> Vec<Keypoint> {
        let f = fist();
        let o = open_palm();
        hand(
            (0.50, 0.85),
            seg(&f, 1..5),
            seg(&o, 5..9),
            seg(&o, 9..13),
            seg(&o, 13..17),
            seg(&f, 17..21),
        )
    }

    /// 四指（四指全伸、拇指收）。
    fn four() -> Vec<Keypoint> {
        let f = fist();
        let o = open_palm();
        hand(
            (0.50, 0.85),
            seg(&f, 1..5),
            seg(&o, 5..9),
            seg(&o, 9..13),
            seg(&o, 13..17),
            seg(&o, 17..21),
        )
    }

    /// 拇指朝下（拇指单伸、指向下方；腕位与握拳一致，四指保持卷曲）。
    fn thumb_down() -> Vec<Keypoint> {
        let f = fist();
        hand(
            (0.50, 0.85),
            [(0.46, 0.92), (0.44, 1.00), (0.43, 1.08), (0.42, 1.16)],
            seg(&f, 5..9),
            seg(&f, 9..13),
            seg(&f, 13..17),
            seg(&f, 17..21),
        )
    }

    fn classify(points: &[Keypoint]) -> GesturePrediction {
        GestureClassifier::new().classify(points).unwrap()
    }

    #[test]
    fn 开手掌识别为_open_palm() {
        let g = classify(&open_palm());
        assert_eq!(g.label, "Open_Palm", "实际 {g:?}");
        assert!(g.score > 0.75, "命中度过低: {g:?}");
    }

    #[test]
    fn 握拳识别为_fist() {
        let g = classify(&fist());
        assert_eq!(g.label, "Fist", "实际 {g:?}");
        assert!(g.score > 0.75, "命中度过低: {g:?}");
    }

    #[test]
    fn 食中指识别为_victory() {
        let g = classify(&victory());
        assert_eq!(g.label, "Victory", "实际 {g:?}");
        assert!(g.score > 0.75, "命中度过低: {g:?}");
    }

    #[test]
    fn 仅食指识别为_pointing_up() {
        let g = classify(&pointing());
        assert_eq!(g.label, "Pointing_Up", "实际 {g:?}");
        assert!(g.score > 0.75, "命中度过低: {g:?}");
    }

    #[test]
    fn 拇指朝上识别为_thumb_up() {
        let g = classify(&thumb_up());
        assert_eq!(g.label, "Thumb_Up", "实际 {g:?}");
        assert!(g.score > 0.7, "命中度过低: {g:?}");
    }

    #[test]
    fn 拇指朝下识别为_thumb_down() {
        let g = classify(&thumb_down());
        assert_eq!(g.label, "Thumb_Down", "实际 {g:?}");
        assert!(g.score > 0.7, "命中度过低: {g:?}");
    }

    #[test]
    fn 三指识别为_three() {
        let g = classify(&three());
        assert_eq!(g.label, "Three", "实际 {g:?}");
        assert!(g.score > 0.75, "命中度过低: {g:?}");
    }

    #[test]
    fn 四指识别为_four() {
        let g = classify(&four());
        assert_eq!(g.label, "Four", "实际 {g:?}");
        assert!(g.score > 0.75, "命中度过低: {g:?}");
    }

    #[test]
    fn 点数不足时报参数错误() {
        let err = GestureClassifier::new().classify(&open_palm()[..10]).unwrap_err();
        assert!(err.to_string().contains("21"));
    }

    #[test]
    fn 像素尺度坐标与归一化坐标结果一致() {
        let scale = 640.0;
        let norm = open_palm();
        let pixel: Vec<Keypoint> = norm
            .iter()
            .map(|p| Keypoint::new(p.x * scale, p.y * scale * 0.75, p.score))
            .collect();
        let a = classify(&norm);
        let b = classify(&pixel);
        assert_eq!(a.label, "Open_Palm");
        assert_eq!(b.label, "Open_Palm", "像素尺度下分类漂移: {b:?}");
    }
}
