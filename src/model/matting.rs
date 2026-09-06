//! 抠图 / 显著性分割结果。

use crate::error::{Result, VisionError};
use crate::imaging::{bounding_rect, FloatMask, Image, Rect};

/// 抠图结果：持有原图尺寸的 soft alpha（float，[0,1]）。
#[derive(Debug, Clone, PartialEq)]
pub struct MattingResult {
    /// soft alpha，[0,1]，尺寸 = 原图
    pub alpha: FloatMask,
    /// 原图宽度
    pub original_width: i32,
    /// 原图高度
    pub original_height: i32,
    /// 模型输入边长（如 1024）
    pub model_input_size: i32,
    /// 推理耗时 ms
    pub elapsed_ms: u64,
}

impl MattingResult {
    /// 二值 mask（0/255）。
    pub fn binary_mask(&self, thresh: f32) -> Image {
        // alpha 是 0~1，thresh 同量纲；先得到 0/1 再扩到 0/255
        self.alpha.binary_mask(thresh)
    }

    /// 前景外接矩形（基于二值化）。
    pub fn foreground_rect(&self, thresh: f32) -> Rect {
        bounding_rect(&self.alpha.binary_mask(thresh))
    }

    /// 是否包含有效 alpha。
    pub fn has_alpha(&self) -> bool {
        !self.alpha.is_empty()
    }
}

impl MattingResult {
    /// 构造（内部校验 alpha 尺寸）。
    pub fn try_new(
        alpha: FloatMask,
        original_width: i32,
        original_height: i32,
        model_input_size: i32,
        elapsed_ms: u64,
    ) -> Result<Self> {
        if alpha.is_empty() {
            return Err(VisionError::image("matting alpha is empty"));
        }
        Ok(MattingResult {
            alpha,
            original_width,
            original_height,
            model_input_size,
            elapsed_ms,
        })
    }
}
