//! 单目深度估计引擎（Depth Anything V2 Small / MiDaS small 兼容）。
//!
//! 细节均已核实：
//!
//! - **Depth Anything V2 Small**（仓库内 `testmodels/depth_anything_v2_small.onnx`，
//!   输入 `pixel_values` [b,3,H,W] 动态，输出 `predicted_depth` [b, H', W'] rank-3）：
//!   输入 518x518（ViT patch=14 的倍数，onnx-community 官方导出的标准分辨率），
//!   **ImageNet mean/std 归一化**：(x/255 - [0.485,0.456,0.406]) / [0.229,0.224,0.225]，RGB；
//! - **MiDaS small**（384x384）：归一化为 (x/255 - 0.5)/0.5 = (x-127.5)/127.5，RGB，
//!   用 [`DepthEstimationEngine::new_midas`] 构造；
//! - **输出语义**：相对深度（inverse depth），**值越大越近**——min-max 归一化到 [0,1]
//!   后即为"近=1、远=0"，与模型约定一致，无需取反；
//! - 深度图最终双线性还原到原图分辨率。

use crate::core::base::BaseOnnxEngine;
use crate::core::device_type::DeviceType;
use crate::error::{Result, VisionError};
use crate::imaging::{FloatMask, Image};

/// ImageNet 归一化均值（Depth Anything V2 系列使用）。
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// ImageNet 归一化标准差。
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// 单目深度估计引擎（相对深度）。
pub struct DepthEstimationEngine {
    /// 组合基类
    pub base: BaseOnnxEngine,
}

impl DepthEstimationEngine {
    /// 创建 Depth Anything V2 引擎（ImageNet 归一化）。
    ///
    /// 输入尺寸从模型读取；动态输入时默认 518x518。
    pub fn new(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        Self::with_input_size(model_path, device_type, -1, -1)
    }

    /// 创建 Depth Anything V2 引擎（指定输入尺寸；<=0 时从模型读取，动态维度回退 518x518）。
    pub fn with_input_size(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        input_height: i32,
        input_width: i32,
    ) -> Result<Self> {
        let h = if input_height > 0 { input_height } else { 518 };
        let w = if input_width > 0 { input_width } else { 518 };
        let mut base = BaseOnnxEngine::with_input_size(model_path, device_type, h, w)?;
        // ImageNet mean/std（基类 preprocess 自动完成 /255 + RGB 转换 + 归一化）
        base.set_normalization(IMAGENET_MEAN, IMAGENET_STD);
        Ok(DepthEstimationEngine { base })
    }

    /// 创建 MiDaS small 兼容引擎（384x384，归一化 (x/255-0.5)/0.5）。
    pub fn new_midas(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        Self::new_midas_with_input_size(model_path, device_type, -1, -1)
    }

    /// 创建 MiDaS small 兼容引擎（指定输入尺寸；<=0 时从模型读取，动态维度回退 384x384）。
    pub fn new_midas_with_input_size(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        input_height: i32,
        input_width: i32,
    ) -> Result<Self> {
        let h = if input_height > 0 { input_height } else { 384 };
        let w = if input_width > 0 { input_width } else { 384 };
        let mut base = BaseOnnxEngine::with_input_size(model_path, device_type, h, w)?;
        base.set_normalization([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]);
        Ok(DepthEstimationEngine { base })
    }

    // ============ 推理 ============

    /// 单图深度估计：返回**原图分辨率**的相对深度图。
    ///
    /// 模型输出为相对深度（inverse depth），经 min-max 归一化到 [0,1]：
    /// **近 = 1，远 = 0**（值越大越近，与 Depth Anything V2 / MiDaS 的模型约定一致）。
    /// 注意：逐图归一化，不同图片间的数值不可直接比较（相对深度无绝对尺度）。
    pub fn predict_depth(&self, image: &Image) -> Result<FloatMask> {
        if image.is_empty() {
            return Err(VisionError::image("cannot estimate depth on empty image"));
        }
        let orig_width = image.width();
        let orig_height = image.height();

        // 1. 预处理（基类：拉伸 resize + BGR→RGB + /255 + ImageNet mean/std，HWC→CHW）
        let input_data = self.base.preprocess(image)?;
        let input_tensor = self.base.create_input_tensor(input_data)?;

        // 2. 推理（模型单输出 [1,H,W] / [1,1,H,W] / [H,W]）
        let output = self.base.run_inference(input_tensor)?;
        let flat = output.as_f32()?;
        let shape = &output.shape;
        let (h, w) = match shape.len() {
            3 => (shape[1].max(0) as usize, shape[2].max(0) as usize),
            4 => (shape[2].max(0) as usize, shape[3].max(0) as usize),
            2 => (shape[0].max(0) as usize, shape[1].max(0) as usize),
            _ => {
                return Err(VisionError::inference(format!(
                    "depth model expects rank-2/3/4 output, got rank-{}, shape: {:?}",
                    shape.len(),
                    shape
                )))
            }
        };
        if h == 0 || w == 0 {
            return Err(VisionError::inference(format!(
                "depth output has invalid spatial dims: shape {shape:?}"
            )));
        }
        // 取第一个 batch 的平面
        if flat.len() < h * w {
            return Err(VisionError::inference(format!(
                "depth output element count {} < {h}x{w}",
                flat.len()
            )));
        }
        let plane = &flat[..h * w];

        // 3. min-max 归一化到 [0,1]（相对深度，近=1 远=0）
        let mut min_v = f32::MAX;
        let mut max_v = -f32::MAX;
        for &v in plane {
            if v < min_v {
                min_v = v;
            }
            if v > max_v {
                max_v = v;
            }
        }
        let range = max_v - min_v;
        let mut data = vec![0f32; h * w];
        if range > f32::EPSILON {
            for (dst, &v) in data.iter_mut().zip(plane.iter()) {
                *dst = (v - min_v) / range;
            }
        }
        let mut mask = FloatMask::from_raw(w, h, data)?;

        // 4. 双线性还原到原图分辨率
        if (w, h) != (orig_width, orig_height) {
            mask = mask.resize(orig_width, orig_height);
        }
        tracing::info!(
            "Depth estimated: model output {w}x{h} -> resized to {}x{}",
            orig_width,
            orig_height
        );
        Ok(mask)
    }

    /// 深度可视化：深度图转 8bit 灰度图（近 = 白(255)，远 = 黑(0)），原图分辨率。
    pub fn predict_vis(&self, image: &Image) -> Result<Image> {
        let depth = self.predict_depth(image)?;
        Ok(depth.to_u8(255.0, 0.0))
    }
}

crate::impl_engine_forward!(DepthEstimationEngine, base, FloatMask,
    /// 单图推理：返回原图分辨率的相对深度图（[0,1]，近=1 远=0）。
    fn predict(&self, image: &Image) -> Result<FloatMask> {
        self.predict_depth(image)
    }
);
