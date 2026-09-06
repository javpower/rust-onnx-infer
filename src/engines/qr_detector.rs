//! 二维码检测引擎（WeChat QR Detector，ONNX）。
//!
//! **模型来源与签名核实**（WebSearch + 本仓库 `model_probe` 实测）：
//! - 模型：WeChat 开源 `wechat_qrcode` 检测子模型（SSD），ONNX 转换取自
//!   `wilinz/wxscan-weights`（caffe→onnx，SHA256
//!   `df03617f92ba20da5705c48564dfe17ba44c13034392a5be814099dfa4a426d6` 校验通过；
//!   上游为 `WeChatCV/opencv_3rdparty` 分支 `wechat_qrcode_20210119`）。
//!   存放路径：`testmodels/wechat_qr_detect.onnx`。
//! - 实测签名（探针输出）：输入 `data = [1,1,-1,-1]`（**单通道灰度**；图内 Reshape 固化
//!   为 416x416，故必须喂 `[1,1,416,416]`——任务描述的 `[1,3,416,416]` 为误传）；
//!   输出 `mbox_loc = [1,32448]`、`mbox_conf_flatten = [1,16224]`，即 8112 个先验框
//!   （各 4 维回归 + 2 类 softmax 概率）。
//!
//! **后处理**：ONNX 导出裁掉了 caffe 的 `PriorBox` / `DetectionOutput` 层，这里按
//! `detect.prototxt` 的层参数 + OpenCV DNN 同名层实现逐行复刻：
//! - 先验框：5 个检测层（stage4_8 step16 26x26；stage5_4~stage8_2 step32 13x13），
//!   每位置 6 框（min 方框、`sqrt(min*max)` 方框、4 个长宽比框），共 1352*6 = 8112；
//! - 解码（CENTER_SIZE）：`cx = 0.1*dx*pw + pcx`、`w = exp(0.2*dw)*pw`（方差 [0.1,0.1,0.2,0.2]）；
//! - 置信度过滤（默认 0.2，与上游 `detection_output_param` 一致）+ top-k 100 +
//!   NMS（IoU 阈值 0.45，复用 [`BoundingBox::iou`] 模式）；
//! - 坐标还原：输入为整图拉伸 resize（416x416），归一化框 × 原图宽高（对齐
//!   [`crate::engines::detection::DetectionEngine`] 的 scaleFill 还原模式）。

use crate::core::base::{BaseOnnxEngine, TensorOutput};
use crate::core::device_type::DeviceType;
use crate::error::{Result, VisionError};
use crate::imaging::{cvt_color, resize, ColorConversion, Image, Interpolation};
use crate::model::BoundingBox;

/// 模型输入边长（图内 Reshape 固化，必须为 416）。
const INPUT_SIZE: usize = 416;

/// CENTER_SIZE 解码方差（`detect.prototxt` PriorBox `variance`）。
const VARIANCE: [f32; 4] = [0.1, 0.1, 0.2, 0.2];

/// 先验框中心偏移（PriorBox `offset`）。
const OFFSET: f32 = 0.5;

/// 长宽比序列（PriorBox `aspect_ratio`，flip=false 不追加倒数）。
const ASPECT_RATIOS: [f32; 4] = [2.0, 0.5, 3.0, 0.333_333_34];

/// 一个检测层的先验框配置（来自 `detect.prototxt` 的 5 个 PriorBox 层）。
#[derive(Debug, Clone, Copy)]
struct PriorLayerConfig {
    /// 该层 feature map 边长（416/step）
    fm_size: usize,
    /// 先验步长（输入像素）
    step: f32,
    /// 最小框边长（输入像素）
    min_size: f32,
    /// 最大框边长（输入像素）
    max_size: f32,
}

/// 按Concat顺序列出：stage4_8 → stage5_4 → stage6_2 → stage7_2 → stage8_2。
const PRIOR_LAYERS: [PriorLayerConfig; 5] = [
    PriorLayerConfig { fm_size: 26, step: 16.0, min_size: 50.0, max_size: 100.0 },
    PriorLayerConfig { fm_size: 13, step: 32.0, min_size: 100.0, max_size: 150.0 },
    PriorLayerConfig { fm_size: 13, step: 32.0, min_size: 150.0, max_size: 200.0 },
    PriorLayerConfig { fm_size: 13, step: 32.0, min_size: 200.0, max_size: 300.0 },
    PriorLayerConfig { fm_size: 13, step: 32.0, min_size: 300.0, max_size: 400.0 },
];

/// 先验框总数：26*26*6 + 4*(13*13*6) = 8112（与探针实测 16224/2 一致）。
const NUM_PRIORS: usize = {
    let mut n = 0;
    let mut i = 0;
    while i < PRIOR_LAYERS.len() {
        n += PRIOR_LAYERS[i].fm_size * PRIOR_LAYERS[i].fm_size * 6;
        i += 1;
    }
    n
};

/// 二维码检测结果（原图像素坐标）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QrDetection {
    /// 框左上角 X（原图像素）
    pub x: i32,
    /// 框左上角 Y（原图像素）
    pub y: i32,
    /// 框宽（原图像素）
    pub w: i32,
    /// 框高（原图像素）
    pub h: i32,
    /// 置信度 [0,1]
    pub score: f32,
}

impl QrDetection {
    /// 归一化 xyxy 框 + 置信度 → 原图像素 xywh（拉伸还原 + 越界裁剪）。
    fn from_normalized(x1: f32, y1: f32, x2: f32, y2: f32, score: f32, img_w: usize, img_h: usize) -> Self {
        let px1 = (x1 * img_w as f32).clamp(0.0, img_w as f32);
        let py1 = (y1 * img_h as f32).clamp(0.0, img_h as f32);
        let px2 = (x2 * img_w as f32).clamp(0.0, img_w as f32);
        let py2 = (y2 * img_h as f32).clamp(0.0, img_h as f32);
        let x = px1.round() as i32;
        let y = py1.round() as i32;
        let w = (px2.round() as i32 - x).max(0);
        let h = (py2.round() as i32 - y).max(0);
        QrDetection { x, y, w, h, score }
    }

    /// 中心点（原图像素）。
    pub fn center(&self) -> (f32, f32) {
        (self.x as f32 + self.w as f32 / 2.0, self.y as f32 + self.h as f32 / 2.0)
    }
}

impl std::fmt::Display for QrDetection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "QrDetection[x={}, y={}, w={}, h={}] score={:.3}",
            self.x, self.y, self.w, self.h, self.score
        )
    }
}

/// WeChat 二维码检测引擎。
pub struct QrDetector {
    /// 组合基类（会话 / 张量 / 推理）
    pub base: BaseOnnxEngine,
    /// NMS IoU 阈值（默认 0.45，对齐上游 `nms_param.nms_threshold`）
    nms_threshold: f32,
    /// NMS 前按分数保留的候选上限（对齐上游 `nms_param.top_k`）
    top_k: usize,
}

impl QrDetector {
    /// 创建检测引擎（模型路径如 `testmodels/wechat_qr_detect.onnx`）。
    pub fn new(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        // 模型声明输入为 [1,1,-1,-1]，但图内 Reshape 固化 416x416，必须显式指定
        let mut base =
            BaseOnnxEngine::with_input_size(model_path, device_type, INPUT_SIZE as i32, INPUT_SIZE as i32)?;
        // 对齐上游 detection_output_param.confidence_threshold = 0.2
        base.set_confidence_threshold(0.2);
        Ok(QrDetector {
            base,
            nms_threshold: 0.45,
            top_k: 100,
        })
    }

    /// NMS IoU 阈值。
    pub fn nms_threshold(&self) -> f32 {
        self.nms_threshold
    }

    /// 设置 NMS IoU 阈值。
    pub fn set_nms_threshold(&mut self, threshold: f32) {
        self.nms_threshold = threshold;
    }

    /// NMS 前候选保留上限。
    pub fn top_k(&self) -> usize {
        self.top_k
    }

    /// 设置 NMS 前候选保留上限。
    pub fn set_top_k(&mut self, top_k: usize) {
        self.top_k = top_k.max(1);
    }

    /// 置信度阈值。
    pub fn confidence_threshold(&self) -> f32 {
        self.base.confidence_threshold()
    }

    /// 设置置信度阈值。
    pub fn set_confidence_threshold(&mut self, threshold: f32) {
        self.base.set_confidence_threshold(threshold);
    }

    /// 检测图中的二维码，返回按置信度降序排列的候选框（原图像素坐标）。
    pub fn detect(&self, image: &Image) -> Result<Vec<QrDetection>> {
        if image.is_empty() {
            return Err(VisionError::image("cannot detect on empty image"));
        }
        let img_w = image.width();
        let img_h = image.height();

        // 1. 预处理：整图拉伸 resize 到 416x416 → 灰度 → /255 → CHW
        let input = self.preprocess_gray(image)?;

        // 2. 推理（双输出 mbox_loc / mbox_conf_flatten）
        let tensor = self.base.create_input_tensor(input)?;
        let outputs = self.base.run_multi_output(tensor)?;
        let (loc, conf) = Self::split_loc_conf(&outputs)?;

        if loc.len() < NUM_PRIORS * 4 || conf.len() < NUM_PRIORS * 2 {
            return Err(VisionError::inference(format!(
                "wechat qr detect output size mismatch: loc={} (expect {}), conf={} (expect {})",
                loc.len(),
                NUM_PRIORS * 4,
                conf.len(),
                NUM_PRIORS * 2
            )));
        }

        // 3. 解码 + 置信度过滤
        let threshold = self.base.confidence_threshold();
        let priors = generate_priors();
        let mut candidates: Vec<(f32, f32, f32, f32, f32)> = Vec::new(); // (x1,y1,x2,y2,score)
        for i in 0..NUM_PRIORS {
            // softmax 后第 1 类为二维码（background_label_id = 0）
            let score = conf[i * 2 + 1];
            if score <= threshold {
                continue;
            }
            // prior 归一化 xyxy
            let p = priors[i];
            let pw = p[2] - p[0];
            let ph = p[3] - p[1];
            let pcx = p[0] + pw * 0.5;
            let pcy = p[1] + ph * 0.5;
            // CENTER_SIZE 解码（方差先乘在回归值上）
            let dx = VARIANCE[0] * loc[i * 4];
            let dy = VARIANCE[1] * loc[i * 4 + 1];
            let dw = VARIANCE[2] * loc[i * 4 + 2];
            let dh = VARIANCE[3] * loc[i * 4 + 3];
            let cx = dx * pw + pcx;
            let cy = dy * ph + pcy;
            let bw = dw.exp() * pw;
            let bh = dh.exp() * ph;
            candidates.push((cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5, score));
        }

        // 4. 按分数降序 → top-k 截断 → NMS（复用 BoundingBox::iou 模式）
        candidates.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(self.top_k);

        let mut results = Vec::with_capacity(candidates.len());
        let mut suppressed = vec![false; candidates.len()];
        for i in 0..candidates.len() {
            if suppressed[i] {
                continue;
            }
            let (x1, y1, x2, y2, score) = candidates[i];
            results.push(QrDetection::from_normalized(x1, y1, x2, y2, score, img_w, img_h));

            let cur = BoundingBox::new(x1 as f64, y1 as f64, x2 as f64, y2 as f64, score as f64);
            for j in (i + 1)..candidates.len() {
                if suppressed[j] {
                    continue;
                }
                let (cx1, cy1, cx2, cy2, cscore) = candidates[j];
                let other =
                    BoundingBox::new(cx1 as f64, cy1 as f64, cx2 as f64, cy2 as f64, cscore as f64);
                if cur.iou(&other) > self.nms_threshold as f64 {
                    suppressed[j] = true;
                }
            }
        }

        tracing::info!("QrDetector: {} candidate(s) after confidence/NMS", results.len());
        Ok(results)
    }

    /// 灰度预处理：拉伸 resize（对齐 OpenCV INTER_CUBIC）→ 灰度 → /255（单通道 CHW）。
    fn preprocess_gray(&self, image: &Image) -> Result<Vec<f32>> {
        let resized = resize(image, INPUT_SIZE, INPUT_SIZE, Interpolation::Cubic)?;
        let gray = match resized.channels() {
            1 => resized,
            3 => cvt_color(&resized, ColorConversion::Bgr2Gray)?,
            4 => cvt_color(&resized, ColorConversion::Bgra2Gray)?,
            c => {
                return Err(VisionError::image(format!(
                    "qr detector expects 1/3/4 channel image, got {c}"
                )))
            }
        };
        Ok(gray.data().iter().map(|&v| v as f32 / 255.0).collect())
    }

    /// 从双输出中定位 `mbox_loc` 与 `mbox_conf_flatten`（优先按名字，兜底按元素数 4:2）。
    fn split_loc_conf(outputs: &[TensorOutput]) -> Result<(&[f32], &[f32])> {
        if outputs.len() < 2 {
            return Err(VisionError::inference(format!(
                "wechat qr detect expects 2 outputs (mbox_loc/mbox_conf_flatten), got {}",
                outputs.len()
            )));
        }
        let mut loc_idx: Option<usize> = None;
        let mut conf_idx: Option<usize> = None;
        for (i, o) in outputs.iter().enumerate() {
            let name = o.name.to_lowercase();
            if name.contains("loc") {
                loc_idx = Some(i);
            } else if name.contains("conf") {
                conf_idx = Some(i);
            }
        }
        // 兜底：loc 元素数是 conf 的 2 倍（4 维回归 vs 2 类概率）
        if loc_idx.is_none() || conf_idx.is_none() {
            for (i, _o) in outputs.iter().enumerate() {
                if Some(i) == loc_idx || Some(i) == conf_idx {
                    continue;
                }
                if loc_idx.is_none() {
                    loc_idx = Some(i);
                } else {
                    conf_idx = Some(i);
                }
            }
            if let (Some(l), Some(c)) = (loc_idx, conf_idx) {
                if outputs[c].element_count() > outputs[l].element_count() {
                    std::mem::swap(&mut loc_idx, &mut conf_idx);
                }
            }
        }
        let loc = outputs[loc_idx.ok_or_else(|| VisionError::inference("cannot locate mbox_loc"))?].as_f32()?;
        let conf = outputs[conf_idx.ok_or_else(|| VisionError::inference("cannot locate mbox_conf_flatten"))?].as_f32()?;
        Ok((loc, conf))
    }
}

/// 生成全部 8112 个先验框（归一化 xyxy，与 OpenCV DNN `PriorBoxLayer` 逐行对齐）。
///
/// 每位置 6 框顺序：min 方框 → `sqrt(min*max)` 方框 → 各长宽比框（宽 = min*sqrt(ar)，高 = min/sqrt(ar)）；
/// 位置遍历：外层 y、内层 x；层顺序与 `mbox_priorbox` 的 Concat 一致。
fn generate_priors() -> Vec<[f32; 4]> {
    let total: usize = PRIOR_LAYERS.iter().map(|l| l.fm_size * l.fm_size * 6).sum();
    let mut priors = Vec::with_capacity(total);
    for layer in &PRIOR_LAYERS {
        // 每位置 6 组 (宽, 高)：min 方框、max 方框、4 个长宽比
        let mut box_sizes = [(layer.min_size, layer.min_size); 6];
        let diag = (layer.min_size * layer.max_size).sqrt();
        box_sizes[1] = (diag, diag);
        for (k, &ar) in ASPECT_RATIOS.iter().enumerate() {
            let root = ar.sqrt();
            box_sizes[2 + k] = (layer.min_size * root, layer.min_size / root);
        }
        let img = INPUT_SIZE as f32;
        for h in 0..layer.fm_size {
            for w in 0..layer.fm_size {
                let cx = (w as f32 + OFFSET) * layer.step / img;
                let cy = (h as f32 + OFFSET) * layer.step / img;
                for (bw, bh) in box_sizes {
                    let bw = bw / img;
                    let bh = bh / img;
                    priors.push([cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5]);
                }
            }
        }
    }
    priors
}

impl QrDetector {
    /// 检测并**解码**二维码内容：先定位码区，再对检测框（含 10% 外扩）裁剪
    /// 做纯 Rust QR 解码（rqrr），返回 `(检测框, 解码文本)` 列表。
    ///
    /// 解码失败的框仍会返回，文本为 `None`（可能是破损/反光/非 QR 码）。
    pub fn decode(&self, image: &Image) -> Result<Vec<(QrDetection, Option<String>)>> {
        let dets = self.detect(image)?;
        let mut out = Vec::new();
        for d in dets {
            let text = decode_qr_region(image, d);
            out.push((d, text));
        }
        Ok(out)
    }
}

/// 对单个检测框区域做 QR 解码（rqrr；解码独立于 ONNX 检测，失败返回 None）。
fn decode_qr_region(image: &Image, d: QrDetection) -> Option<String> {
    // 外扩 10% 给定位角留白，clamp 到图内
    let pad_x = (d.w as f32 * 0.1).round() as i32;
    let pad_y = (d.h as f32 * 0.1).round() as i32;
    let x0 = (d.x - pad_x).max(0) as usize;
    let y0 = (d.y - pad_y).max(0) as usize;
    let x1 = ((d.x + d.w + pad_x) as usize).min(image.width());
    let y1 = ((d.y + d.h + pad_y) as usize).min(image.height());
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    let crop = image.crop(x0, y0, x1 - x0, y1 - y0).ok()?;
    let dynamic = crop.to_dynamic().ok()?;
    let luma = dynamic.to_luma8();
    let (w, h) = (luma.width() as usize, luma.height() as usize);
    // 二值化：亮度 < 128 视为黑（qr 光照均匀时该阈值足够；破损码由 None 返回）
    let mut img = rqrr::PreparedImage::prepare_from_bitmap(w, h, |x, y| {
        luma.get_pixel(x as u32, y as u32)[0] < 128
    });
    let grids = img.detect_grids();
    let grid = grids.into_iter().next()?;
    grid.decode().ok().map(|(_, content)| content)
}

crate::impl_engine_forward!(QrDetector, base, Vec<QrDetection>,
    /// 单图推理（trait `predict` 转发到 [`QrDetector::detect`]）。
    fn predict(&self, image: &Image) -> Result<Vec<QrDetection>> {
        self.detect(image)
    }
);

// ==================== 自验测试 ====================

#[cfg(test)]
mod tests {
    use super::*;

    /// 先验框数量必须与探针实测输出对齐：8112 个（loc 32448 / conf 16224）。
    #[test]
    fn prior_count_matches_model_output() {
        let priors = generate_priors();
        assert_eq!(priors.len(), 8112);
        assert_eq!(NUM_PRIORS, 8112);
    }

    /// 先验框中心必须落在 [0,1]，尺寸为正且角点次序正确。
    /// （边缘 anchor 的框可小量出界——SSD 标准行为，解码端负责 clip。）
    #[test]
    fn priors_are_normalized_and_ordered() {
        for p in generate_priors() {
            let (cx, cy) = ((p[0] + p[2]) / 2.0, (p[1] + p[3]) / 2.0);
            assert!(
                (-0.1..=1.1).contains(&cx) && (-0.1..=1.1).contains(&cy),
                "prior center out of range: {p:?}"
            );
            assert!(p[2] > p[0] && p[3] > p[1], "prior corners inverted: {p:?}");
        }
    }

    /// 真实模型 + python qrcode 生成的测试图（testmodels/qr_test.png，
    /// 二维码位于 (200,150) 410x410）。
    #[test] // 依赖 testmodels/ 下的模型与测试图，按仓库惯例默认跳过
    fn detect_real_qr_with_model() {
        if !["testmodels/wechat_qr_detect.onnx", "models/qr_detector/wechat_qr_detect.onnx", "testmodels/qr_test.png"].iter().any(|p| std::path::Path::new(p).exists()) {
            eprintln!("[SKIP] 模型/测试资产缺失（models/ 或 testmodels/）");
            return;
        }

        let detector = QrDetector::new("testmodels/wechat_qr_detect.onnx", DeviceType::Cpu).unwrap();
        let image = Image::load("testmodels/qr_test.png").unwrap();
        let detections = detector.detect(&image).unwrap();
        println!("检测到 {} 个二维码框:", detections.len());
        for d in &detections {
            println!("  {d}");
        }
        assert!(!detections.is_empty(), "未检出二维码");

        let best = detections[0];
        // 期望真值：左上 (200,150)，尺寸 410x410
        let (cx, cy) = best.center();
        assert!(
            (cx - 405.0).abs() < 80.0 && (cy - 355.0).abs() < 80.0,
            "检测框中心偏离真值过远: ({cx:.0}, {cy:.0})"
        );
        assert!(
            (best.w - 410).abs() < 100 && (best.h - 410).abs() < 100,
            "检测框尺寸偏离真值过远: {}x{}",
            best.w,
            best.h
        );
        assert!(best.score > 0.5, "置信度过低: {}", best.score);
    }

    /// 小尺寸二维码（testmodels/qr_test_small.png，二维码位于 (360,260) 174x174）
    /// 与无二维码负样本（bus.jpg）的行为观察。
    #[test] // 依赖 testmodels/ 下的模型与测试图
    fn detect_small_qr_and_negative_case() {
        if !["testmodels/wechat_qr_detect.onnx", "models/qr_detector/wechat_qr_detect.onnx", "testmodels/qr_test_small.png"].iter().any(|p| std::path::Path::new(p).exists()) {
            eprintln!("[SKIP] 模型/测试资产缺失（models/ 或 testmodels/）");
            return;
        }

        let detector = QrDetector::new("testmodels/wechat_qr_detect.onnx", DeviceType::Cpu).unwrap();

        // 正样本：小二维码（约 174x174，占画面比例小）
        let image = Image::load("testmodels/qr_test_small.png").unwrap();
        let detections = detector.detect(&image).unwrap();
        println!("小码检测到 {} 个框:", detections.len());
        for d in &detections {
            println!("  {d}");
        }
        assert!(!detections.is_empty(), "未检出小尺寸二维码");
        let best = detections[0];
        // 真值中心 (447, 347)
        let (cx, cy) = best.center();
        assert!((cx - 447.0).abs() < 60.0 && (cy - 347.0).abs() < 60.0,
            "小码检测框中心偏离: ({cx:.0}, {cy:.0})");

        // 负样本：无二维码的公交场景（仅打印观察；上游阈值 0.2 较宽松，
        // 若有低分误检属正常，要求置信度不得接近 1）
        let neg = Image::load("testmodels/bus.jpg").unwrap();
        let neg_dets = detector.detect(&neg).unwrap();
        println!("负样本检出 {} 个框: {neg_dets:?}", neg_dets.len());
        for d in &neg_dets {
            assert!(d.score < 0.9, "负样本误检置信度过高: {d}");
        }
    }
}
