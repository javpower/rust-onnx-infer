//! SAHI 切片推理配置与基础类型。
//!
//! 默认值与 sahi==0.12.6 `get_sliced_prediction` 签名默认值一致。

use std::collections::HashSet;

/// 合并策略（对应官方 postprocess_type）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PostprocessType {
    /// 非极大值抑制：丢弃重叠的低分框
    Nms,
    /// 贪心非极大值合并：重叠框并入最高分框（bbox 并集），SAHI 默认
    #[default]
    Greedynmm,
    /// 非极大值合并（传递式）：A 并 B、B 并 C 则三者合并
    Nmm,
    /// 局部敏感 NMS：官方实验性功能，未实现（等价按 Nms 处理）
    Lsnms,
}

/// 重叠度量（对应官方 match_metric）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MatchMetric {
    /// 交并比 intersection / union
    Iou,
    /// 交小比 intersection / min(area1, area2)，SAHI 默认
    #[default]
    Ios,
}

/// SAHI 切片推理配置，默认值与官方 `get_sliced_prediction` 一致。
#[derive(Debug, Clone, PartialEq)]
pub struct SahiConfig {
    /// 切片高；None 或 <=0 时按分辨率自动计算（auto_slice_resolution）
    pub slice_height: Option<i32>,
    /// 切片宽；None 或 <=0 时按分辨率自动计算
    pub slice_width: Option<i32>,
    /// 高度方向相邻切片重叠比例（0.2 = 重叠 20% 切片高度）
    pub overlap_height_ratio: f64,
    /// 宽度方向相邻切片重叠比例
    pub overlap_width_ratio: f64,
    /// 未指定切片尺寸时按图像分辨率自动计算切片参数（官方 auto_slice_resolution）
    pub auto_slice_resolution: bool,
    /// 切片数 > 1 时对整图再做一次标准预测并参与合并（官方 perform_standard_pred）
    pub perform_standard_prediction: bool,
    /// 合并策略（官方 postprocess_type；检测引擎置信度阈值 < 0.1 时自动切换为 NMS/IOU）
    pub postprocess_type: PostprocessType,
    /// 匹配度量（官方 postprocess_match_metric，SAHI 默认 IOS）
    pub match_metric: MatchMetric,
    /// 匹配阈值（官方 postprocess_match_threshold）
    pub match_threshold: f64,
    /// true 时合并/抑制忽略类别（官方 postprocess_class_agnostic）
    pub class_agnostic: bool,
    /// true 时禁用"低置信度阈值自动切换 NMS/IOU"（官方 force_postprocess_type）
    pub force_postprocess_type: bool,
    /// 低内存模式的合并缓冲长度：累积预测数超过该值时中途执行一次合并（官方 merge_buffer_length）
    pub merge_buffer_length: Option<i32>,
    /// 按类别名排除的预测（官方 exclude_classes_by_name）
    pub exclude_class_names: HashSet<String>,
    /// 按类别 id 排除的预测（官方 exclude_classes_by_id）
    pub exclude_class_ids: HashSet<i32>,
    /// 是否输出每个切片的预测数量日志
    pub verbose: bool,
}

impl Default for SahiConfig {
    fn default() -> Self {
        SahiConfig {
            slice_height: None,
            slice_width: None,
            overlap_height_ratio: 0.2,
            overlap_width_ratio: 0.2,
            auto_slice_resolution: true,
            perform_standard_prediction: true,
            postprocess_type: PostprocessType::Greedynmm,
            match_metric: MatchMetric::Ios,
            match_threshold: 0.5,
            class_agnostic: false,
            force_postprocess_type: false,
            merge_buffer_length: None,
            exclude_class_names: HashSet::new(),
            exclude_class_ids: HashSet::new(),
            verbose: false,
        }
    }
}

impl SahiConfig {
    /// 官方 LOW_MODEL_CONFIDENCE：引擎置信度阈值低于该值时 postprocess 自动切换 NMS/IOU
    pub const LOW_MODEL_CONFIDENCE: f32 = 0.1;

    /// 与官方默认一致的配置。
    pub fn defaults() -> Self {
        SahiConfig::default()
    }

    /// 指定切片尺寸与重叠比例的快捷配置（最常用）。
    pub fn of(slice_height: i32, slice_width: i32, overlap_height_ratio: f64, overlap_width_ratio: f64) -> Self {
        SahiConfig {
            slice_height: Some(slice_height),
            slice_width: Some(slice_width),
            overlap_height_ratio,
            overlap_width_ratio,
            ..SahiConfig::default()
        }
    }
}

/// SAHI 合并后处理使用的内部框表示（对应官方 ObjectPrediction 的核心字段）。
///
/// 坐标/分数全程以 **float32** 保存，保证度量矩阵阶段与官方逐位一致；
/// 合并复核阶段（has_match）官方用 float64，对应 `f64` 运算。
/// 坐标为**全图坐标系**（切片坐标已加上切片偏移）。
///
/// `payload` 承载引擎原始结果：检测场景为 None；分割场景承载掩码信息。
#[derive(Debug, Clone)]
pub struct SahiBox {
    pub min_x: f32,
    pub min_y: f32,
    pub max_x: f32,
    pub max_y: f32,
    pub score: f32,
    pub category_id: i32,
    pub category_name: String,
    /// 引擎原始结果负载（分割场景使用）
    pub payload: Option<SahiPayload>,
}

/// SAHI 分割场景的掩码负载（切片帧掩码 + 偏移，或已合并的全图掩码画布）。
#[derive(Debug, Clone)]
pub enum SahiPayload {
    /// 切片内掩码帧（掩码尺寸的 [0,1] 数据）+ 该切片在全图中的偏移与全图尺寸
    /// （对应 `SliceMask(sliceMask, shiftX, shiftY, fullWidth, fullHeight)`；
    /// 全图尺寸供物化全图画布时使用）
    SliceMask {
        data: Vec<f32>,
        width: usize,
        height: usize,
        offset_x: i32,
        offset_y: i32,
        full_width: i32,
        full_height: i32,
    },
    /// 已合并的全图掩码画布
    FullMask(crate::imaging::FloatMask),
}

impl SahiBox {
    pub fn new(
        min_x: f32,
        min_y: f32,
        max_x: f32,
        max_y: f32,
        score: f32,
        category_id: i32,
        category_name: impl Into<String>,
    ) -> Self {
        SahiBox {
            min_x,
            min_y,
            max_x,
            max_y,
            score,
            category_id,
            category_name: category_name.into(),
            payload: None,
        }
    }

    pub fn area(&self) -> f32 {
        (self.max_x - self.min_x) * (self.max_y - self.min_y)
    }

    /// 官方 merge_object_prediction_pair：bbox 并集 + 分数取最大 + 类别取分高者（同分取后者）。
    pub fn merge_into(&mut self, other: &SahiBox) {
        self.min_x = self.min_x.min(other.min_x);
        self.min_y = self.min_y.min(other.min_y);
        self.max_x = self.max_x.max(other.max_x);
        self.max_y = self.max_y.max(other.max_y);
        self.score = self.score.max(other.score);
        // get_merged_category: `if pred1.score.value > pred2.score.value: pred1 else pred2`
        if !(self.score > other.score) {
            self.category_id = other.category_id;
            self.category_name = other.category_name.clone();
        }
    }
}
