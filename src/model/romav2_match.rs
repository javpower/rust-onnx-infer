//! RoMaV2 密集特征匹配结果。
//!
//! 同时保留密集对应场（warp/overlap，模型输出空间）与采样稀疏匹配（原图像素空间）。

/// 单个密集匹配对（坐标已还原到 A/B 原图像素空间）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DenseMatch {
    /// A 图中的 x 坐标（原图像素）
    pub x_a: f32,
    /// A 图中的 y 坐标（原图像素）
    pub y_a: f32,
    /// B 图中的 x 坐标（原图像素）
    pub x_b: f32,
    /// B 图中的 y 坐标（原图像素）
    pub y_b: f32,
    /// 置信度（来自 overlap_AB，[0,1]）
    pub score: f32,
}

/// RoMaV2 密集匹配结果。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RomaV2MatchResult {
    /// 密集 warp 的高度（= 输入高度，通常 640）
    pub dense_height: usize,
    /// 密集 warp 的宽度（= 输入宽度，通常 640）
    pub dense_width: usize,

    /// 密集对应场：长度 `H*W*2`，布局 [y, x, 2]，归一化坐标 [-1,1]（align_corners=False）
    pub warp_ab: Option<Vec<f32>>,
    /// 重叠/置信度图：长度 `H*W`，范围 [0,1]
    pub overlap_ab: Option<Vec<f32>>,

    /// 按 overlap 阈值采样得到的稀疏匹配（坐标已还原到 A/B 原图像素空间）
    pub sampled_matches: Vec<DenseMatch>,

    /// A 图原始宽度
    pub image_a_width: i32,
    /// A 图原始高度
    pub image_a_height: i32,
    /// B 图原始宽度
    pub image_b_width: i32,
    /// B 图原始高度
    pub image_b_height: i32,

    /// A 图预处理缩放比例（RoMaV2 采用 stretch resize 无 padding）
    pub scale_a: f32,
    /// B 图预处理缩放比例
    pub scale_b: f32,
}

impl RomaV2MatchResult {
    /// 获取采样匹配数量。
    pub fn match_count(&self) -> usize {
        self.sampled_matches.len()
    }

    /// 获取置信度 ≥ 阈值的密集对应数。
    pub fn dense_count(&self, threshold: f32) -> usize {
        self.overlap_ab
            .as_ref()
            .map(|o| o.iter().filter(|&&v| v >= threshold).count())
            .unwrap_or(0)
    }

    /// 获取指定网格坐标 (x, y) 处的 B 图归一化坐标 [xB, yB]。
    pub fn warp_at(&self, x: usize, y: usize) -> [f32; 2] {
        match &self.warp_ab {
            None => [0.0, 0.0],
            Some(w) => {
                let idx = (y * self.dense_width + x) * 2;
                [w[idx], w[idx + 1]]
            }
        }
    }

    /// 获取指定网格坐标 (x, y) 处的置信度。
    pub fn overlap_at(&self, x: usize, y: usize) -> f32 {
        match &self.overlap_ab {
            None => 0.0,
            Some(o) => o[y * self.dense_width + x],
        }
    }

    /// 获取用于 findHomography 的点对（原图像素空间坐标）。
    pub fn matched_points(&self) -> (Vec<super::point2d::Point2D>, Vec<super::point2d::Point2D>) {
        let mut pts_a = Vec::with_capacity(self.sampled_matches.len());
        let mut pts_b = Vec::with_capacity(self.sampled_matches.len());
        for m in &self.sampled_matches {
            pts_a.push(super::point2d::Point2D::new(m.x_a as f64, m.y_a as f64));
            pts_b.push(super::point2d::Point2D::new(m.x_b as f64, m.y_b as f64));
        }
        (pts_a, pts_b)
    }
}
