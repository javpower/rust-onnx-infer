//! SAHI 切片推理结果容器。

/// SAHI 切片推理结果：合并后的最终结果 + 过程信息（切片数、各阶段耗时）。
///
/// `T` 为合并后的结果列表类型（检测 `Vec<Detection>` / 分割 `Vec<Segmentation>`，
/// 对应原版 `List<Detection> detections` 字段）。
#[derive(Debug, Clone)]
pub struct SahiPredictionResult<T> {
    /// 合并后的最终结果列表（全图坐标，已按官方输出顺序排序）
    pub detections: T,

    /// 实际生成的切片数量
    pub num_slices: usize,

    /// 原图尺寸
    pub image_width: i32,
    pub image_height: i32,

    /// 各阶段耗时（毫秒）
    pub slice_millis: u64,
    pub predict_millis: u64,
    pub postprocess_millis: u64,
}

impl<T> SahiPredictionResult<T> {
    /// 创建结果容器（对应上游 构造器）。
    pub fn new(
        detections: T,
        num_slices: usize,
        image_width: i32,
        image_height: i32,
        slice_millis: u64,
        predict_millis: u64,
        postprocess_millis: u64,
    ) -> Self {
        SahiPredictionResult {
            detections,
            num_slices,
            image_width,
            image_height,
            slice_millis,
            predict_millis,
            postprocess_millis,
        }
    }

    /// 总耗时（毫秒）。
    pub fn total_millis(&self) -> u64 {
        self.slice_millis + self.predict_millis + self.postprocess_millis
    }
}
