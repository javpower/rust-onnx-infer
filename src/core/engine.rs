//! ONNX 推理引擎统一接口。

use async_trait::async_trait;

use crate::error::Result;
use crate::imaging::Image;

/// 在 tokio 运行时中执行阻塞推理：multi_thread 运行时用 `block_in_place`，
/// 否则（无运行时 / current_thread）直接同步执行。
pub async fn run_blocking<T, F>(f: F) -> T
where
    T: Send,
    F: FnOnce() -> T + Send,
{
    let in_multithread = tokio::runtime::Handle::try_current()
        .map(|h| h.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread)
        .unwrap_or(false);
    if in_multithread {
        tokio::task::block_in_place(f)
    } else {
        f()
    }
}

/// ONNX 推理引擎统一接口。
///
/// 所有引擎均为 `Send + Sync`，可放入 `Arc` 多线程共享（内部对 Session 加锁）。
#[async_trait]
pub trait OnnxInferenceEngine: Send + Sync {
    /// 单张图片的推理结果类型（如 `Vec<Detection>`、`ClassificationResult`）
    type Output: Send;

    /// 单图推理
    fn predict(&self, image: &Image) -> Result<Self::Output>;

    /// 批量推理（默认实现：逐张调用 `predict`；引擎可覆写为真实批量）
    fn predict_batch(&self, images: &[Image]) -> Result<Vec<Self::Output>> {
        images.iter().map(|img| self.predict(img)).collect()
    }

    /// 异步单图推理（对应 `predictAsync`）
    async fn predict_async(&self, image: &Image) -> Result<Self::Output> {
        run_blocking(|| self.predict(image)).await
    }

    /// 异步批量推理（对应 `predictBatchAsync`）
    async fn predict_batch_async(&self, images: &[Image]) -> Result<Vec<Self::Output>> {
        run_blocking(|| self.predict_batch(images)).await
    }

    /// 获取输入尺寸 (width, height)
    fn input_size(&self) -> (i32, i32);

    /// 获取类别标签
    fn labels(&self) -> Option<&[String]>;

    /// 手动设置类别名（模型元数据不含标签时使用）
    fn set_labels(&mut self, labels: Vec<String>);

    /// 设置置信度阈值
    fn set_confidence_threshold(&mut self, threshold: f32);

    /// 获取置信度阈值
    fn confidence_threshold(&self) -> f32;
}

/// 为具体引擎实现通用接口转发（把 trait 方法委托给内部的 [`crate::core::base::BaseOnnxEngine`]）。
///
/// `predict`（及可选的 `predict_batch` 等覆写）由引擎自带实现传入：
///
/// ```ignore
/// impl_engine_forward!(ClassificationEngine, base, ClassificationResult,
///     fn predict(&self, image: &Image) -> Result<ClassificationResult> { self.predict_impl(image) });
/// ```
#[macro_export]
macro_rules! impl_engine_forward {
    ($engine:ty, $field:ident, $output:ty, $predict:item $(, $extra:item)* $(,)?) => {
        #[::async_trait::async_trait]
        impl $crate::core::engine::OnnxInferenceEngine for $engine {
            type Output = $output;

            $predict

            $($extra)*

            fn input_size(&self) -> (i32, i32) {
                self.$field.input_size()
            }

            fn labels(&self) -> Option<&[String]> {
                self.$field.labels()
            }

            fn set_labels(&mut self, labels: Vec<String>) {
                self.$field.set_labels(labels)
            }

            fn set_confidence_threshold(&mut self, threshold: f32) {
                self.$field.set_confidence_threshold(threshold)
            }

            fn confidence_threshold(&self) -> f32 {
                self.$field.confidence_threshold()
            }
        }
    };
}
