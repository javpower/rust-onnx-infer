//! LightGlue 特征匹配结果。

/// 关键点。
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct MatchKeyPoint {
    /// x 坐标
    pub x: f32,
    /// y 坐标
    pub y: f32,
    /// 在关键点列表中的索引
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
    /// 模板图中的关键点
    pub kp0: MatchKeyPoint,
    /// 场景图中的关键点
    pub kp1: MatchKeyPoint,
    /// 匹配置信度
    pub score: f32,
}

/// LightGlue 特征匹配结果：两张图的关键点、匹配对与坐标还原所需的预处理信息。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LightGlueMatchResult {
    /// 模板图关键点
    pub keypoints0: Vec<MatchKeyPoint>,
    /// 场景图关键点
    pub keypoints1: Vec<MatchKeyPoint>,
    /// 匹配对列表
    pub matches: Vec<FeatureMatch>,
    /// 模板图原始宽度
    pub image0_width: i32,
    /// 模板图原始高度
    pub image0_height: i32,
    /// 场景图原始宽度
    pub image1_width: i32,
    /// 场景图原始高度
    pub image1_height: i32,
    /// 模板图预处理缩放比例
    pub scale0: f32,
    /// 场景图预处理缩放比例
    pub scale1: f32,
    /// 模板图 Letterbox 上填充像素数
    pub pad_top0: i32,
    /// 模板图 Letterbox 左填充像素数
    pub pad_left0: i32,
    /// 场景图 Letterbox 上填充像素数
    pub pad_top1: i32,
    /// 场景图 Letterbox 左填充像素数
    pub pad_left1: i32,
}

impl LightGlueMatchResult {
    /// 获取匹配数量。
    pub fn match_count(&self) -> usize {
        self.matches.len()
    }

    /// 获取匹配点对（输入空间坐标，不还原到原图）。
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

/// 批量匹配结果。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LightGlueBatchResult {
    /// 每对图像的匹配结果
    pub results: Vec<LightGlueMatchResult>,
    /// 批次大小
    pub batch_size: usize,
}

impl LightGlueBatchResult {
    /// 获取指定 batch 的结果。
    pub fn result(&self, batch_index: usize) -> Option<&LightGlueMatchResult> {
        self.results.get(batch_index)
    }

    /// 获取匹配数 >= min_match_count 的成功数量。
    pub fn success_count(&self, min_match_count: usize) -> usize {
        self.results
            .iter()
            .filter(|r| r.match_count() >= min_match_count)
            .count()
    }
}
