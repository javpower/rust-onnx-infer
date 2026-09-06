//! WholeBody 全身 133 关键点引擎（RTMPose-m-wholebody）。
//!
//! 133 点 = COCO body 17 + 足部 6 + 面部 68 + 双手 42（每手 21），一次前向
//! 覆盖人体/面部/双手关键点（对应上游"支持很多关键点的模型"需求）。
//! SimCC 解码与 mmpose `get_simcc_maximum` 语义一致。

use crate::core::base::BaseOnnxEngine;
use crate::core::device_type::DeviceType;
use crate::core::engine::OnnxInferenceEngine;
use crate::error::{Result, VisionError};
use crate::imaging::Image;
use crate::model::Keypoint;

/// 关键点总数。
pub const NUM_KEYPOINTS: usize = 133;
/// 模型输入 (宽 192, 高 256)。
pub const INPUT_WIDTH: i32 = 192;
pub const INPUT_HEIGHT: i32 = 256;

/// COCO-WholeBody 分段索引（供下游按部位取点）。
pub mod segments {
    /// body 17 点：[0, 17)
    pub const BODY: std::ops::Range<usize> = 0..17;
    /// 足部 6 点：[17, 23)
    pub const FOOT: std::ops::Range<usize> = 17..23;
    /// 面部 68 点：[23, 91)
    pub const FACE: std::ops::Range<usize> = 23..91;
    /// 左手 21 点：[91, 112)
    pub const LEFT_HAND: std::ops::Range<usize> = 91..112;
    /// 右手 21 点：[112, 133)
    pub const RIGHT_HAND: std::ops::Range<usize> = 112..133;
}

/// WholeBody 133 关键点结果（原图像素坐标）。
#[derive(Debug, Clone, PartialEq)]
pub struct WholeBodyKeypoints {
    /// 133 个关键点（顺序 = COCO-WholeBody 官方约定）
    pub points: Vec<Keypoint>,
}

impl WholeBodyKeypoints {
    /// 按部位切片取点。
    pub fn segment(&self, range: std::ops::Range<usize>) -> &[Keypoint] {
        &self.points[range.start..range.end.min(self.points.len())]
    }
}

/// WholeBody 全身关键点引擎。
pub struct WholeBodyPoseEngine {
    /// 组合基类
    pub base: BaseOnnxEngine,
}

impl WholeBodyPoseEngine {
    /// 创建引擎（输入固定 192×256）。
    pub fn new(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        let mut base = BaseOnnxEngine::with_input_size(
            model_path,
            device_type,
            INPUT_HEIGHT,
            INPUT_WIDTH,
        )?;
        // RTMPose 标准预处理：RGB + ImageNet mean/std
        base.set_normalization([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
        Ok(WholeBodyPoseEngine { base })
    }

    /// 对人体区域推理 133 点（`person_box`：检测框，内部拉伸到 192×256）。
    pub fn extract(&self, image: &Image, person_box: crate::imaging::Rect) -> Result<WholeBodyKeypoints> {
        let x0 = person_box.x.max(0) as usize;
        let y0 = person_box.y.max(0) as usize;
        let x1 = (person_box.x + person_box.width as i32).clamp(0, image.width() as i32) as usize;
        let y1 = (person_box.y + person_box.height as i32).clamp(0, image.height() as i32) as usize;
        if x1 <= x0 || y1 <= y0 {
            return Err(VisionError::invalid_argument("人体框无效"));
        }
        let crop = image.crop(x0, y0, x1 - x0, y1 - y0)?;
        let (cw, ch) = (crop.width() as f32, crop.height() as f32);

        let input = self.base.preprocess(&crop)?;
        let tensor = self.base.create_input_tensor(input)?;
        let outputs = self.base.run_multi_output(tensor)?;
        if outputs.len() < 2 {
            return Err(VisionError::inference("WholeBody 模型应有 simcc_x/simcc_y 两个输出"));
        }
        let sx = outputs[0].as_f32()?;
        let sy = outputs[1].as_f32()?;
        let k = outputs[0].shape[1] as usize;
        let n_bin_x = outputs[0].shape[2] as usize;
        let n_bin_y = outputs[1].shape[2] as usize;
        let bin_x = n_bin_x as f32;
        let bin_y = n_bin_y as f32;

        let mut points = Vec::with_capacity(k);
        for i in 0..k {
            let row_x = &sx[i * n_bin_x..(i + 1) * n_bin_x];
            let row_y = &sy[i * n_bin_y..(i + 1) * n_bin_y];
            let (bx, vx) = argmax(row_x);
            let (by, vy) = argmax(row_y);
            // mmpose get_simcc_maximum：双轴分数取 min；模型输出已过 sigmoid
            let score = vx.min(vy).clamp(0.0, 1.0);
            let mx = (bx as f32 / bin_x * cw + x0 as f32).clamp(0.0, image.width() as f32 - 1.0);
            let my = (by as f32 / bin_y * ch + y0 as f32).clamp(0.0, image.height() as f32 - 1.0);
            points.push(Keypoint::new(mx, my, score));
        }
        Ok(WholeBodyKeypoints { points })
    }
}

/// 一维 argmax。
fn argmax(v: &[f32]) -> (usize, f32) {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    (best, best_v)
}

#[async_trait::async_trait]
impl OnnxInferenceEngine for WholeBodyPoseEngine {
    type Output = crate::model::PoseResult;

    fn predict(&self, image: &Image) -> Result<Self::Output> {
        // trait 语义：整图当作单个人体区域
        let box_rect = crate::imaging::Rect::new(0, 0, image.width() as i32, image.height() as i32);
        let kpts = self.extract(image, box_rect)?;
        Ok(crate::model::PoseResult::new(
            crate::model::Detection::new("person", 0, 0.0, 0.0, image.width() as f64, image.height() as f64, 1.0),
            kpts.points,
        ))
    }

    fn input_size(&self) -> (i32, i32) {
        self.base.input_size()
    }

    fn labels(&self) -> Option<&[String]> {
        self.base.labels()
    }

    fn set_labels(&mut self, labels: Vec<String>) {
        self.base.set_labels(labels)
    }

    fn set_confidence_threshold(&mut self, _threshold: f32) {}

    fn confidence_threshold(&self) -> f32 {
        0.0
    }
}
