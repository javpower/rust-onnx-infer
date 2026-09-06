//! LoMa-R 稀疏特征匹配结果。
//!
//! 匹配 pipeline：DeDoDe 检测器抽关键点 → 描述器采样描述子 → LoMa-R 输出 [M,N]
//! 分数矩阵 → 双向最近邻 + 阈值过滤得到稀疏匹配对。

/// 关键点（已还原到原图像素坐标）。
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct MatchKeyPoint {
    /// x 坐标（像素）
    pub x: f32,
    /// y 坐标（像素）
    pub y: f32,
    /// 在所属图关键点列表中的索引
    pub index: i32,
}

impl MatchKeyPoint {
    pub fn new(x: f32, y: f32, index: i32) -> Self {
        MatchKeyPoint { x, y, index }
    }
}

/// 特征匹配对。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeatureMatch {
    /// 图 A 中的关键点
    pub kp0: MatchKeyPoint,
    /// 图 B 中的关键点
    pub kp1: MatchKeyPoint,
    /// 匹配置信度（来自 LoMa-R scores 矩阵）
    pub score: f32,
}

/// LoMa-R 稀疏匹配结果。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LoMaRMatchResult {
    /// 图 A（模板）上的关键点，已还原到原图像素坐标
    pub keypoints0: Vec<MatchKeyPoint>,
    /// 图 B（场景）上的关键点，已还原到原图像素坐标
    pub keypoints1: Vec<MatchKeyPoint>,
    /// 双向最近邻 + 阈值过滤后的匹配对
    pub matches: Vec<FeatureMatch>,
    /// 图 A 原始宽度
    pub image0_width: i32,
    /// 图 A 原始高度
    pub image0_height: i32,
    /// 图 B 原始宽度
    pub image1_width: i32,
    /// 图 B 原始高度
    pub image1_height: i32,
    /// DeDoDe 检测器在图 A 上保留的关键点数
    pub detector_num_keypoints0: i32,
    /// DeDoDe 检测器在图 B 上保留的关键点数
    pub detector_num_keypoints1: i32,
    /// 互最近邻过滤使用的分数阈值
    pub filter_threshold: f32,
    /// 原始匹配分数矩阵 [M, N]（可选；为节省内存可置 None）
    pub score_matrix: Option<Vec<Vec<f32>>>,
}

impl LoMaRMatchResult {
    /// 获取匹配数量。
    pub fn match_count(&self) -> usize {
        self.matches.len()
    }

    /// 导出匹配点对（用于 findHomography / findFundamentalMat 等几何验证）。
    pub fn matched_points(&self) -> (Vec<super::point2d::Point2D>, Vec<super::point2d::Point2D>) {
        let mut pts0 = Vec::with_capacity(self.matches.len());
        let mut pts1 = Vec::with_capacity(self.matches.len());
        for m in &self.matches {
            pts0.push(super::point2d::Point2D::new(m.kp0.x as f64, m.kp0.y as f64));
            pts1.push(super::point2d::Point2D::new(m.kp1.x as f64, m.kp1.y as f64));
        }
        (pts0, pts1)
    }
}
