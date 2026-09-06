//! 人脸 106 精细关键点引擎（insightface 2d106det）。
//!
//! 输入 192×192 人脸区域（拉伸），输出 106 个关键点像素坐标（眉毛/眼睛/
//! 鼻梁/嘴形/轮廓），供活体检测、疲劳监测、AR 贴合等下游使用。

use crate::core::base::BaseOnnxEngine;
use crate::core::device_type::DeviceType;
use crate::core::engine::OnnxInferenceEngine;
use crate::error::{Result, VisionError};
use crate::imaging::{Image, Rect};
use crate::model::Keypoint;

/// 模型输入边长。
pub const INPUT_SIZE: usize = 192;
/// 关键点数。
pub const NUM_LANDMARKS: usize = 106;

/// 人脸 106 关键点结果（坐标已还原到原图像素）。
#[derive(Debug, Clone, PartialEq)]
pub struct FaceLandmarks106 {
    /// 106 个关键点（原图像素坐标，顺序 = insightface 2d106det 输出顺序）
    pub points: Vec<Keypoint>,
    /// 对齐用的源人脸框（原图坐标）
    pub face_box: Rect,
}

impl FaceLandmarks106 {
    /// 双眼区域中心（眼周区间均值近似；insightface 106 无官方逐点语义表，
    /// 精确瞳孔 index 需按模型版本标定）。
    pub fn eye_centers(&self) -> Option<(Keypoint, Keypoint)> {
        let left = region_mean(&self.points, 44, 56);
        let right = region_mean(&self.points, 74, 86);
        match (left, right) {
            (Some(l), Some(r)) => Some((l, r)),
            _ => None,
        }
    }
}

/// 区间内关键点均值。
fn region_mean(points: &[Keypoint], from: usize, to: usize) -> Option<Keypoint> {
    let seg = points.get(from..to)?;
    if seg.is_empty() {
        return None;
    }
    let n = seg.len() as f32;
    Some(Keypoint::new(
        seg.iter().map(|p| p.x).sum::<f32>() / n,
        seg.iter().map(|p| p.y).sum::<f32>() / n,
        seg.iter().map(|p| p.score).sum::<f32>() / n,
    ))
}

/// 人脸 106 关键点引擎。
pub struct FaceLandmark106Engine {
    /// 组合基类
    pub base: BaseOnnxEngine,
}

impl FaceLandmark106Engine {
    /// 创建引擎（输入固定 192×192）。
    pub fn new(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        let base = BaseOnnxEngine::with_input_size(
            model_path,
            device_type,
            INPUT_SIZE as i32,
            INPUT_SIZE as i32,
        )?;
        Ok(FaceLandmark106Engine { base })
    }

    /// 对单个人脸区域推理 106 点。
    ///
    /// `face_box`：原图上的人脸框（来自 YuNet 等）；内部按框外扩 20% 裁剪并
    /// 拉伸到 192×192（5 点相似变换对齐的精确版可先经 [`Self::extract_aligned`]）。
    pub fn extract(&self, image: &Image, face_box: Rect) -> Result<FaceLandmarks106> {
        // 外扩 20%（保持中心），clamp 到图内
        let exp = 0.2f32;
        let cx = face_box.x as f32 + face_box.width as f32 / 2.0;
        let cy = face_box.y as f32 + face_box.height as f32 / 2.0;
        let half_w = face_box.width as f32 * (0.5 + exp);
        let half_h = face_box.height as f32 * (0.5 + exp);
        let x0 = (cx - half_w).max(0.0).floor() as usize;
        let y0 = (cy - half_h).max(0.0).floor() as usize;
        let x1 = ((cx + half_w).ceil() as usize).min(image.width());
        let y1 = ((cy + half_h).ceil() as usize).min(image.height());
        if x1 <= x0 || y1 <= y0 {
            return Err(VisionError::invalid_argument("人脸框无效"));
        }
        let crop = image.crop(x0, y0, x1 - x0, y1 - y0)?;
        let crop_w = crop.width() as f32;
        let crop_h = crop.height() as f32;

        let points_model = self.forward(&crop)?;
        // 模型输出为 192 模型空间坐标 → 按 crop 缩放还原到原图
        let mut points = Vec::with_capacity(points_model.len());
        for (x, y) in points_model {
            points.push(Keypoint::new(
                x * crop_w / INPUT_SIZE as f32 + x0 as f32,
                y * crop_h / INPUT_SIZE as f32 + y0 as f32,
                1.0,
            ));
        }
        Ok(FaceLandmarks106 {
            points,
            face_box,
        })
    }

    /// 对已对齐的人脸图直接推理（坐标相对该图）。
    pub fn extract_aligned(&self, aligned: &Image) -> Result<Vec<Keypoint>> {
        let points_model = self.forward(aligned)?;
        let (w, h) = (aligned.width() as f32, aligned.height() as f32);
        Ok(points_model
            .into_iter()
            .map(|(x, y)| Keypoint::new(x * w / INPUT_SIZE as f32, y * h / INPUT_SIZE as f32, 1.0))
            .collect())
    }

    /// 内部前向：返回模型空间坐标 [(x,y); 106]。
    fn forward(&self, face: &Image) -> Result<Vec<(f32, f32)>> {
        // 模型输入：BGR 0~255（insightface 系约定），拉伸到 192×192
        let resized = crate::imaging::resize(
            face,
            INPUT_SIZE,
            INPUT_SIZE,
            crate::imaging::Interpolation::Linear,
        )?;
        let area = INPUT_SIZE * INPUT_SIZE;
        let mut chw = vec![0f32; 3 * area];
        let px = resized.data();
        for i in 0..area {
            for c in 0..3 {
                chw[c * area + i] = px[i * 3 + c] as f32;
            }
        }
        let tensor = self.base.create_input_tensor(chw)?;
        let output = self.base.run_inference(tensor)?;
        let data = output.as_f32()?;
        if data.len() != NUM_LANDMARKS * 2 {
            return Err(VisionError::inference(format!(
                "关键点输出元素数 {} != {}x2",
                data.len(),
                NUM_LANDMARKS
            )));
        }
        Ok((0..NUM_LANDMARKS)
            .map(|i| (data[i * 2], data[i * 2 + 1]))
            .collect())
    }
}

#[async_trait::async_trait]
impl OnnxInferenceEngine for FaceLandmark106Engine {
    type Output = Vec<Keypoint>;

    fn predict(&self, image: &Image) -> Result<Self::Output> {
        // trait 语义：整图当作单个人脸区域
        let box_rect = Rect::new(0, 0, image.width() as i32, image.height() as i32);
        Ok(self.extract(image, box_rect)?.points)
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
