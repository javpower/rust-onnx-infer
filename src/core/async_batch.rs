//! 异步批处理优化器。
//!
//! 通过动态批量聚合和预热机制提升推理吞吐量：
//! - 动态批量聚合：等待一定数量的图像后统一推理
//! - 超时触发：即使批量未满，超过超时时间也触发推理
//! - 预热机制：启动时预加载模型，减少首次推理延迟

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{mpsc, oneshot};

use crate::core::engine::{run_blocking, OnnxInferenceEngine};
use crate::error::{Result, VisionError};
use crate::imaging::Image;

/// 优化器配置（对应 `AsyncBatchOptimizer.Builder`）。
#[derive(Debug, Clone)]
pub struct AsyncBatchOptimizerBuilder {
    pub max_batch_size: usize,
    pub timeout_ms: u64,
    pub processing_threads: usize,
    pub queue_capacity: usize,
    pub enable_warmup: bool,
}

impl Default for AsyncBatchOptimizerBuilder {
    fn default() -> Self {
        AsyncBatchOptimizerBuilder {
            max_batch_size: 8,
            timeout_ms: 50,
            processing_threads: 2,
            queue_capacity: 100,
            enable_warmup: true,
        }
    }
}

impl AsyncBatchOptimizerBuilder {
    pub fn max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = size.max(1);
        self
    }

    pub fn timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = ms.max(1);
        self
    }

    pub fn processing_threads(mut self, threads: usize) -> Self {
        self.processing_threads = threads.max(1);
        self
    }

    pub fn queue_capacity(mut self, capacity: usize) -> Self {
        self.queue_capacity = capacity.max(1);
        self
    }

    pub fn enable_warmup(mut self, enable: bool) -> Self {
        self.enable_warmup = enable;
        self
    }
}

struct BatchRequest<T> {
    image: Image,
    reply: oneshot::Sender<Result<T>>,
}

/// 异步批处理优化器。
///
/// `submit_async` 返回的 `oneshot::Receiver` 等价于一个一次性 future 句柄。
pub struct AsyncBatchOptimizer<E>
where
    E: OnnxInferenceEngine + 'static,
    E::Output: Send + Clone,
{
    engine: Arc<E>,
    tx: mpsc::Sender<BatchRequest<E::Output>>,
    handle: Option<tokio::task::JoinHandle<()>>,
    config: AsyncBatchOptimizerBuilder,
}

impl<E> AsyncBatchOptimizer<E>
where
    E: OnnxInferenceEngine + 'static,
    E::Output: Send + Clone,
{
    /// 使用默认参数创建（maxBatchSize=8, timeoutMs=50, warmup=true）。
    pub fn new(engine: Arc<E>) -> Self {
        AsyncBatchOptimizer::with_builder(engine, AsyncBatchOptimizerBuilder::default())
    }

    /// 使用自定义参数创建并启动处理循环。
    pub fn with_builder(engine: Arc<E>, builder: AsyncBatchOptimizerBuilder) -> Self {
        let mut builder = builder;
        builder.max_batch_size = builder.max_batch_size.max(1);
        builder.timeout_ms = builder.timeout_ms.max(1);
        builder.queue_capacity = builder.queue_capacity.max(1);
        let (tx, rx) = mpsc::channel(builder.queue_capacity);

        if builder.enable_warmup {
            let engine = Arc::clone(&engine);
            tokio::spawn(async move {
                tracing::info!("Starting warmup...");
                let dummy = Image::new(640, 640, 1);
                match run_blocking(|| engine.predict(&dummy)).await {
                    Ok(_) => tracing::info!("Warmup completed"),
                    Err(e) => tracing::warn!("Warmup failed: {}", e),
                }
            });
        }

        tracing::info!(
            "AsyncBatchOptimizer started: maxBatchSize={}, timeoutMs={}",
            builder.max_batch_size,
            builder.timeout_ms
        );
        let handle = tokio::spawn(processing_loop(rx, Arc::clone(&engine), builder.clone()));

        AsyncBatchOptimizer {
            engine,
            tx,
            handle: Some(handle),
            config: builder,
        }
    }

    /// 引擎访问器。
    pub fn engine(&self) -> &Arc<E> {
        &self.engine
    }

    /// 配置访问器。
    pub fn config(&self) -> &AsyncBatchOptimizerBuilder {
        &self.config
    }

    /// 提交单图异步推理请求（队列满时立即返回 `Queue is full` 错误）。
    pub async fn submit_async(&self, image: Image) -> Result<oneshot::Receiver<Result<E::Output>>> {
        let (reply, rx) = oneshot::channel();
        let req = BatchRequest { image, reply };
        match self.tx.try_send(req) {
            Ok(()) => Ok(rx),
            Err(mpsc::error::TrySendError::Full(_)) => Err(VisionError::Other("Queue is full".to_string())),
            Err(mpsc::error::TrySendError::Closed(_)) => {
                Err(VisionError::Other("Batch optimizer queue is closed".to_string()))
            }
        }
    }

    /// 当前队列剩余容量（近似对应 `getQueueSize` 的反向指标）。
    pub fn queue_capacity_left(&self) -> usize {
        self.tx.capacity()
    }

    /// 处理循环是否仍在运行。
    pub fn is_running(&self) -> bool {
        self.handle.as_ref().map(|h| !h.is_finished()).unwrap_or(false)
    }
}

impl<E> Drop for AsyncBatchOptimizer<E>
where
    E: OnnxInferenceEngine + 'static,
    E::Output: Send + Clone,
{
    fn drop(&mut self) {
        if let Some(h) = self.handle.take() {
            h.abort();
        }
    }
}

/// 批处理主循环：批量聚合 + 超时触发（对应 `processingLoop`）。
async fn processing_loop<E>(
    mut rx: mpsc::Receiver<BatchRequest<E::Output>>,
    engine: Arc<E>,
    config: AsyncBatchOptimizerBuilder,
) where
    E: OnnxInferenceEngine + 'static,
    E::Output: Send + Clone,
{
    let timeout = Duration::from_millis(config.timeout_ms);
    let mut batch: Vec<BatchRequest<E::Output>> = Vec::with_capacity(config.max_batch_size);
    let mut last_process = Instant::now();

    loop {
        let first = if batch.is_empty() {
            match tokio::time::timeout(timeout, rx.recv()).await {
                Ok(Some(req)) => Some(req),
                Ok(None) => break, // 通道关闭
                Err(_) => None,    // 等待超时，触发积压处理
            }
        } else {
            let wait = timeout.saturating_sub(last_process.elapsed());
            match tokio::time::timeout(wait, rx.recv()).await {
                Ok(Some(req)) => Some(req),
                Ok(None) => break,
                Err(_) => None,
            }
        };

        if let Some(req) = first {
            batch.push(req);
            while batch.len() < config.max_batch_size {
                match rx.try_recv() {
                    Ok(r) => batch.push(r),
                    Err(_) => break,
                }
            }
        }

        let should_process = !batch.is_empty()
            && (batch.len() >= config.max_batch_size || last_process.elapsed() >= timeout);
        if should_process {
            process_batch(&engine, &mut batch).await;
            last_process = Instant::now();
        }
    }

    if !batch.is_empty() {
        process_batch(&engine, &mut batch).await;
    }
}

async fn process_batch<E>(engine: &Arc<E>, batch: &mut Vec<BatchRequest<E::Output>>)
where
    E: OnnxInferenceEngine + 'static,
    E::Output: Send + Clone,
{
    if batch.is_empty() {
        return;
    }
    let images: Vec<Image> = batch.iter().map(|r| r.image.clone()).collect();
    let results = run_blocking(|| engine.predict_batch(&images)).await;

    match results {
        Ok(outputs) => {
            for (i, req) in batch.drain(..).enumerate() {
                let reply = match outputs.get(i) {
                    Some(r) => Ok(r.clone()),
                    None => Err(VisionError::Other("Result index out of bounds".to_string())),
                };
                let _ = req.reply.send(reply);
            }
        }
        Err(e) => {
            for req in batch.drain(..) {
                let _ = req.reply.send(Err(e.clone()));
            }
        }
    }
}
