//! 关键点模型（姿态估计 / 人脸地标共用）。

use super::detection::Detection;

/// 单个关键点：像素坐标 + 置信度。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Keypoint {
    /// x 坐标（原图像素）
    pub x: f32,
    /// y 坐标（原图像素）
    pub y: f32,
    /// 可见性/置信度 [0,1]
    pub score: f32,
}

impl Keypoint {
    pub fn new(x: f32, y: f32, score: f32) -> Self {
        Keypoint { x, y, score }
    }
}

/// 姿态估计结果：一个行人实例的检测框 + COCO 17 关键点。
#[derive(Debug, Clone, PartialEq)]
pub struct PoseResult {
    /// 基础检测信息（框 + 类别 + 置信度）
    pub detection: Detection,
    /// 关键点（顺序与 COCO 17 点一致；score <= 阈值的点仍保留原值）
    pub keypoints: Vec<Keypoint>,
}

impl PoseResult {
    pub fn new(detection: Detection, keypoints: Vec<Keypoint>) -> Self {
        PoseResult { detection, keypoints }
    }

    pub fn confidence(&self) -> f64 {
        self.detection.confidence()
    }

    /// 按索引取关键点（越界返回 None）。
    pub fn keypoint(&self, idx: usize) -> Option<Keypoint> {
        self.keypoints.get(idx).copied()
    }
}

/// COCO 17 关键点定义（YOLO-Pose 输出顺序）。
pub mod coco {
    /// 关键点名称（索引即模型输出顺序）。
    pub const NAMES: [&str; 17] = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
        "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
        "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    ];

    /// 骨架连接对（0-based 索引，用于可视化）。
    pub const SKELETON: [(usize, usize); 16] = [
        (0, 1), (0, 2), (1, 3), (2, 4),       // 头部
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), // 上肢
        (5, 11), (6, 12), (11, 12),           // 躯干
        (11, 13), (13, 15), (12, 14), (14, 16), // 下肢
    ];
}

/// 人脸 5 点地标定义（YuNet/SCRFD 输出顺序）。
pub mod face_landmarks {
    pub const NAMES: [&str; 5] = [
        "left_eye", "right_eye", "nose_tip", "left_mouth_corner", "right_mouth_corner",
    ];
}
