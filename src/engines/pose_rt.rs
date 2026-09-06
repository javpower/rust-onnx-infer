//! RTMO 实时一阶段多人姿态估计引擎（open-mmlab/mmpose RTMO-S/M，YOLOX 风格单阶段）。
//!
//! 与 [`crate::engines::pose`]（YOLO-Pose 两阶段：先检人后检点）不同，RTMO 单模型
//! 一次前向同时输出检测框与关键点，输入为 640x640（模型 zoo 官方导出）的整图 letterbox。
//!
//! **支持的 ONNX 输出布局**（首次推理时按输出名/shape 自动识别，见 [`OutputLayout`]）：
//!
//! | 布局 | 输出张量 | 来源 |
//! |---|---|---|
//! | SimCC（mmpose head 原生导出） | `det_outputs [1,N,5]`（x1,y1,x2,y2,score）+ `simcc_x [1,N,K,W']` + `simcc_y [1,N,K,H']` | 社区/mmpose rtmo head 直接导出 |
//! | End2End（mmpose 官方 SDK 导出） | `dets [1,N,5]` + `keypoints [1,N,K,3]`（x,y,score） | download.openmmlab.com 的 `onnx_sdk/*.zip` |
//!
//! **预处理（已对官方模型查证）**：官方 RTMO 配置 `data_preprocessor mean=[0,0,0] std=[1,1,1]`、
//! 部署 pipeline `Normalize(to_rgb=false)`，且 ONNX 图输入侧无任何归一化/通道交换节点
//! （rtmlib 参考实现同样直接喂原始 BGR 0..255）—— 因此默认 **不做 /255、不做 BGR→RGB**。
//! 针对自带图内归一化的第三方导出，可经 [`RealtimePoseEngine::set_to_rgb`] /
//! [`RealtimePoseEngine::set_divide_255`] 打开对应开关。letterbox 用 114 灰边居中填充
//! （还原公式与 detection/pose 引擎一致：`orig = (input - d) / ratio`）。
//!
//! **SimCC 关键点解码（已对 mmpose 源码查证）**：`mmpose/codecs/utils/post_processing.py`
//! 的 `get_simcc_maximum` —— 坐标 `(argmax(simcc_x), argmax(simcc_y)) / simcc_split_ratio`
//! （RTMO 为 2.0，即 640 -> 320）；关键点分数取两轴最大响应中的 **较小者**
//! `min(max(simcc_x), max(simcc_y))`（两轴必须同时置信，并非取 max）。RTMO 的 SimCC 分支
//! 以 BCE-logits 训练，原生导出未在图内过 sigmoid，因此与 pose 引擎的自适应策略一致：
//! 仅当响应值出现负值或 >1（+容差）时补 sigmoid，否则视为已在 [0,1]。响应 <=0 的点按
//! mmpose `locs[vals<=0]=-1` 语义视为无效，分数记 0。
//!
//! **后处理**：det score >= 置信度阈值过滤 → [`BoundingBox::iou`] NMS（官方 SDK 导出图内
//! 已做 NMS，这里再过一遍为幂等操作，保证 SimCC 原生导出同样正确）→ 关键点/框按 letterbox
//! 参数还原到原图坐标。
//!
//! **默认阈值**（RTMO 常用值）：det 置信度 0.45、关键点分数阈值 0.5（低于阈值的关键点仍保留
//! 原值，见 [`PoseResult`] 约定）、NMS IoU 0.65。
//!
//! 注意：SAHI 切片推理不适用于一阶段姿态任务，开启 SAHI 配置时仍走整图路径。

use std::sync::Mutex;

use crate::core::base::{BaseOnnxEngine, TensorOutput};
use crate::core::device_type::DeviceType;
use crate::error::{Result, VisionError};
use crate::imaging::{cvt_color, resize, ColorConversion, Image, Interpolation};
use crate::model::{BoundingBox, Detection, Keypoint, PoseResult};

/// letterbox 预处理的坐标还原参数（与 [`crate::engines::pose`] 的同名模式一致：
/// Rust `&self` 不可变，参数随调用链显式传递而非写在实例字段上）。
#[derive(Debug, Clone, Copy, Default)]
struct LetterboxParams {
    /// 缩放比例（min(input_w/orig_w, input_h/orig_h)）
    ratio: f32,
    /// 左右对称填充的一半宽度
    dw: f32,
    /// 上下对称填充的一半高度
    dh: f32,
}

/// RTMO ONNX 输出布局（首次推理自动识别并缓存）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputLayout {
    /// 三张量：`det_outputs [1,N,5]` + `simcc_x [1,N,K,W']` + `simcc_y [1,N,K,H']`，
    /// 关键点需经 SimCC argmax 解码。
    Simcc,
    /// 两张量：`dets [1,N,5]` + `keypoints [1,N,K,3]`（官方 SDK 端到端导出，
    /// NMS 与 SimCC 解码已在计算图内完成）。
    End2End,
}

/// SimCC 张量的归一化视图：(实例数 N, 关键点数 K, 每轴 bin 数 S)，
/// 数据按 `[N][K][S]` 行主序展开访问。
struct SimccView {
    n_inst: usize,
    num_kpts: usize,
    bins: usize,
    data: Vec<f32>,
}

impl SimccView {
    /// 把 simcc 输出张量解释为 `[N, K, S]` 视图。
    ///
    /// 支持 `rank4 [1,N,K,S]` / `rank3 [N,K,S]`（N 为实例数）/ `rank3 [K,S]`（单实例）。
    /// `expected_inst` 为检测框数量，用于区分 rank3 的两种可能。
    fn from_tensor(t: &TensorOutput, expected_inst: usize) -> Result<Self> {
        let flat = t.as_f32()?;
        let d = &t.shape;
        let (n_inst, num_kpts, bins) = match d.len() {
            4 => (d[1] as usize, d[2] as usize, d[3] as usize),
            3 if expected_inst > 1 && d[0] as usize == expected_inst => {
                (d[0] as usize, d[1] as usize, d[2] as usize)
            }
            3 | 2 => (1, d[d.len() - 2] as usize, d[d.len() - 1] as usize),
            _ => {
                return Err(VisionError::inference(format!(
                    "SimCC tensor rank {} is not supported, shape: {:?}",
                    d.len(),
                    d
                )))
            }
        };
        if n_inst == 0 || num_kpts == 0 || bins == 0 {
            return Err(VisionError::inference(format!(
                "SimCC tensor has zero dimension, shape: {:?}",
                d
            )));
        }
        if flat.len() < n_inst * num_kpts * bins {
            return Err(VisionError::inference(format!(
                "SimCC tensor element count {} < {}x{}x{}",
                flat.len(),
                n_inst,
                num_kpts,
                bins
            )));
        }
        Ok(SimccView {
            n_inst,
            num_kpts,
            bins,
            data: flat[..n_inst * num_kpts * bins].to_vec(),
        })
    }

    #[inline]
    fn get(&self, inst: usize, kpt: usize, bin: usize) -> f32 {
        self.data[(inst * self.num_kpts + kpt) * self.bins + bin]
    }
}

/// 检测候选实例（NMS 中间结构；坐标为 letterbox 输入空间的 xyxy，
/// `index` 为该候选在模型输出中的实例行号，NMS 后据此回读关键点）。
#[derive(Debug, Clone, Copy)]
struct RtmoCandidate {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    score: f32,
    /// det_outputs / dets 中的行索引
    index: usize,
}

/// RTMO 实时姿态估计引擎。
pub struct RealtimePoseEngine {
    /// 组合基类
    pub base: BaseOnnxEngine,

    /// NMS IoU 阈值（默认 0.65）
    nms_threshold: f32,
    /// 关键点分数过滤阈值（默认 0.5；低于阈值的关键点仍保留原值，供下游自行过滤）
    keypoint_threshold: f32,
    /// 预处理是否做 BGR→RGB（默认 false：官方 RTMO ONNX 期望原始 BGR，见模块文档查证）
    to_rgb: bool,
    /// 预处理是否 /255（默认 false：官方 RTMO mean=0/std=1，原始 0..255，见模块文档查证）
    divide_255: bool,
    /// 自动识别的输出布局缓存（`Mutex<Option<_>>` 模式，首次推理时填充）
    layout_cache: Mutex<Option<OutputLayout>>,
}

impl RealtimePoseEngine {
    /// 创建 RTMO 实时姿态引擎（默认阈值：det 0.45 / kpt 0.5 / nms 0.65）。
    pub fn new(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        Self::with_input_size(model_path, device_type, -1, -1)
    }

    /// 创建 RTMO 实时姿态引擎（指定输入尺寸；<=0 时从模型读取，官方模型为 640x640）。
    pub fn with_input_size(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        input_height: i32,
        input_width: i32,
    ) -> Result<Self> {
        let base = BaseOnnxEngine::with_input_size(model_path, device_type, input_height, input_width)?;
        Ok(Self::build(base))
    }

    /// 内部构造（统一默认阈值：det 0.45 / kpt 0.5 / nms 0.65）。
    fn build(base: BaseOnnxEngine) -> Self {
        let mut engine = RealtimePoseEngine {
            base,
            nms_threshold: 0.65,
            keypoint_threshold: 0.5,
            to_rgb: false,
            divide_255: false,
            layout_cache: Mutex::new(None),
        };
        engine.base.set_confidence_threshold(0.45);
        engine
    }

    // ============ 访问器 ============

    /// NMS IoU 阈值。
    pub fn nms_threshold(&self) -> f32 {
        self.nms_threshold
    }

    /// 设置 NMS IoU 阈值。
    pub fn set_nms_threshold(&mut self, threshold: f32) {
        self.nms_threshold = threshold;
    }

    /// 关键点分数过滤阈值。
    pub fn keypoint_threshold(&self) -> f32 {
        self.keypoint_threshold
    }

    /// 设置关键点分数过滤阈值（低于阈值的关键点仍保留原值，见 [`PoseResult`] 约定）。
    pub fn set_keypoint_threshold(&mut self, threshold: f32) {
        self.keypoint_threshold = threshold;
    }

    /// 预处理是否做 BGR→RGB（默认 false）。
    pub fn to_rgb(&self) -> bool {
        self.to_rgb
    }

    /// 设置预处理是否做 BGR→RGB（仅图内**未**含通道转换的第三方导出需要打开）。
    pub fn set_to_rgb(&mut self, to_rgb: bool) {
        self.to_rgb = to_rgb;
    }

    /// 预处理是否 /255（默认 false）。
    pub fn divide_255(&self) -> bool {
        self.divide_255
    }

    /// 设置预处理是否 /255（仅图内**未**含归一化的第三方导出需要打开）。
    pub fn set_divide_255(&mut self, divide: bool) {
        self.divide_255 = divide;
    }

    // ============ 推理路径 ============

    /// 单图推理核心实现（trait `predict` 转发到这里）。
    pub fn predict_impl(&self, image: &Image) -> Result<Vec<PoseResult>> {
        if image.is_empty() {
            return Err(VisionError::image("cannot run realtime pose on empty image"));
        }

        // 1. 记录原始尺寸
        let orig_width = image.width();
        let orig_height = image.height();

        // 2. 预处理（letterbox 114 填充 + 原始 BGR 0..255 + HWC→CHW，同时记录还原参数）
        let (input_data, lb) = self.preprocess_with_letterbox(image)?;

        // 3. 创建 Tensor 并推理
        let input_tensor = self.base.create_input_tensor(input_data)?;
        let all_outputs = self.base.run_multi_output(input_tensor)?;

        // 4. 识别输出布局并定位各张量（结果缓存，后续推理直接复用）
        let (layout, det, others) = self.resolve_layout(&all_outputs)?;

        // 5. 解码（box/关键点均在还原参数传递下统一映射回原图坐标）
        let results = match layout {
            OutputLayout::Simcc => self.decode_simcc(det, &others, orig_width, orig_height, lb)?,
            OutputLayout::End2End => self.decode_end2end(det, &others, orig_width, orig_height, lb)?,
        };

        self.log_results(&results);
        Ok(results)
    }

    // ==================== 预处理 ====================

    /// 带 Letterbox 的预处理：保持宽高比缩放 + 114 灰边居中填充，
    /// 像素保持原始 0..255 BGR（官方模型图内自带处理，见模块文档），HWC → CHW。
    ///
    /// 返回 `(CHW 浮点数据, 坐标还原参数)`；参数随返回值显式传递（对齐 detection/pose 引擎模式）。
    fn preprocess_with_letterbox(&self, image: &Image) -> Result<(Vec<f32>, LetterboxParams)> {
        if image.is_empty() {
            return Err(VisionError::image("cannot letterbox empty image"));
        }
        let orig_width = image.width();
        let orig_height = image.height();
        let input_width = self.base.input_width() as usize;
        let input_height = self.base.input_height() as usize;

        // 1. 计算缩放比例
        let ratio = (input_width as f32 / orig_width as f32)
            .min(input_height as f32 / orig_height as f32);
        let new_width = (orig_width as f32 * ratio).round() as usize;
        let new_height = (orig_height as f32 * ratio).round() as usize;

        // 2. 计算居中填充
        let dw = (input_width as f32 - new_width as f32) / 2.0;
        let dh = (input_height as f32 - new_height as f32) / 2.0;

        // 输入统一到 3 通道 BGR
        let bgr = match image.channels() {
            3 => image.clone(),
            4 => cvt_color(image, ColorConversion::Bgra2Bgr)?,
            _ => cvt_color(image, ColorConversion::Gray2Bgr)?,
        };

        // 3. Resize
        let resized = resize(&bgr, new_width, new_height, Interpolation::Linear)?;

        // 4. 画布填充（114 灰边）
        let mut padded = Image::filled(input_width, input_height, 3, 114);
        let top = ((dh - 0.1).round() as i32).max(0) as usize;
        let left = ((dw - 0.1).round() as i32).max(0) as usize;
        padded.paste(left, top, &resized);

        // 5. 可选颜色空间转换（默认关闭：官方 RTMO 期望 BGR）
        let channel_data = if self.to_rgb {
            cvt_color(&padded, ColorConversion::Bgr2Rgb)?
        } else {
            padded
        };

        // 6. 原始像素（可选 /255）+ HWC → CHW
        let px = channel_data.data();
        let area = input_height * input_width;
        let scale = if self.divide_255 { 1.0 / 255.0 } else { 1.0 };
        let mut float_data = vec![0f32; 3 * area];
        for i in 0..area {
            float_data[i] = px[i * 3] as f32 * scale;
            float_data[i + area] = px[i * 3 + 1] as f32 * scale;
            float_data[i + 2 * area] = px[i * 3 + 2] as f32 * scale;
        }

        Ok((float_data, LetterboxParams { ratio, dw, dh }))
    }

    // ==================== 布局识别 ====================

    /// 获取（必要时首次识别）输出布局，并返回 det 张量与其余张量引用。
    ///
    /// 识别策略（名称优先，shape 兜底；结果缓存）：
    /// 1. 任一输出名含 "simcc" -> Simcc 布局（det 取名含 "det" 或末维=5 者）
    /// 2. 任一输出名含 "keypoint" -> End2End 布局
    /// 3. shape 兜底：存在 rank4 且末维=3 的张量 -> End2End；否则三张量布局 -> Simcc
    fn resolve_layout<'a>(
        &self,
        outputs: &'a [TensorOutput],
    ) -> Result<(OutputLayout, &'a TensorOutput, Vec<&'a TensorOutput>)> {
        if outputs.is_empty() {
            return Err(VisionError::inference("RTMO model returned no outputs"));
        }

        {
            let mut cache = self.layout_cache.lock().unwrap();
            if cache.is_none() {
                *cache = Some(Self::detect_layout(outputs)?);
            }
        }
        let layout = self
            .layout_cache
            .lock()
            .unwrap()
            .expect("layout cache must be filled");

        let det = Self::find_det_output(outputs)?;
        let others: Vec<&TensorOutput> = outputs.iter().filter(|o| !std::ptr::eq(*o, det)).collect();
        Ok((layout, det, others))
    }

    /// 首次推理时自动识别输出布局（依据输出名与 shape）。
    fn detect_layout(outputs: &[TensorOutput]) -> Result<OutputLayout> {
        let simcc_named = outputs.iter().any(|o| o.name.contains("simcc"));
        let kpt_named = outputs.iter().any(|o| o.name.contains("keypoint"));

        let layout = if simcc_named {
            OutputLayout::Simcc
        } else if kpt_named {
            OutputLayout::End2End
        } else {
            // shape 兜底：rank4 末维=3 为端到端 keypoints；否则按三张量 SimCC 处理
            let has_rank4_kpt = outputs
                .iter()
                .any(|o| o.shape.len() == 4 && o.shape.last() == Some(&3));
            if has_rank4_kpt {
                OutputLayout::End2End
            } else {
                OutputLayout::Simcc
            }
        };

        tracing::info!(
            "Auto-detected RTMO output layout: {:?} (outputs: {:?})",
            layout,
            outputs.iter().map(|o| (o.name.as_str(), o.shape.clone())).collect::<Vec<_>>()
        );
        Ok(layout)
    }

    /// 定位检测张量（x1,y1,x2,y2,score）。
    ///
    /// 名称优先（含 "det"）；否则取末维=5 的张量；再兜底取第一个 rank>=2 张量。
    fn find_det_output(outputs: &[TensorOutput]) -> Result<&TensorOutput> {
        if let Some(o) = outputs.iter().find(|o| o.name.contains("det")) {
            return Ok(o);
        }
        if let Some(o) = outputs
            .iter()
            .find(|o| o.shape.len() >= 2 && o.shape.last() == Some(&5))
        {
            return Ok(o);
        }
        outputs
            .first()
            .ok_or_else(|| VisionError::inference("RTMO model returned no outputs"))
    }

    // ==================== SimCC 布局解码 ====================

    /// SimCC 三张量布局解码：`det_outputs [1,N,5]` + `simcc_x/simcc_y`。
    ///
    /// 解码公式（对齐 mmpose `get_simcc_maximum`）：
    /// - 关键点坐标 = `(argmax(simcc_x), argmax(simcc_y)) / split_ratio`（输入空间，再经 letterbox 还原）
    /// - split_ratio 动态推导 = `input_size / simcc_bins`（RTMO 官方 640 -> 320，ratio 2.0）
    /// - 关键点分数 = `min(max(simcc_x), max(simcc_y))`（两轴同时置信），logits 时自适应补 sigmoid
    fn decode_simcc(
        &self,
        det: &TensorOutput,
        others: &[&TensorOutput],
        orig_width: usize,
        orig_height: usize,
        lb: LetterboxParams,
    ) -> Result<Vec<PoseResult>> {
        if others.len() < 2 {
            return Err(VisionError::inference(format!(
                "SimCC layout expects 3 outputs (det + simcc_x + simcc_y), got det + {}",
                others.len()
            )));
        }
        // 其余两个张量按名含 simcc_x/simcc_y 区分，名字不可用时按输出顺序 (x, y)
        let (sx_t, sy_t) = Self::pick_simcc_pair(others);

        // 1. 解析检测张量 [1,N,5] / [N,5]
        let det_flat = det.as_f32()?;
        let det_n = Self::det_instance_count(det)?;
        if det_n == 0 {
            return Ok(Vec::new());
        }

        // 2. 解析 SimCC 张量为 [N,K,S] 视图（x/y 两轴 K 必须一致）
        let sx = SimccView::from_tensor(sx_t, det_n)?;
        let sy = SimccView::from_tensor(sy_t, det_n)?;
        if sx.num_kpts != sy.num_kpts {
            return Err(VisionError::inference(format!(
                "SimCC x/y keypoint dims mismatch: {} vs {}",
                sx.num_kpts, sy.num_kpts
            )));
        }
        // split_ratio = 输入尺寸 / bin 数（x 轴对应输入宽、y 轴对应输入高）
        let ratio_x = self.base.input_width() as f32 / sx.bins as f32;
        let ratio_y = self.base.input_height() as f32 / sy.bins as f32;

        // 3. SimCC 响应是否为原始 logits（区间出现负值或 >1 时补 sigmoid，见模块文档）
        let need_sigmoid = simcc_need_sigmoid(&sx, &sy);

        // 4. 置信度过滤生成候选（坐标保持在输入空间，NMS 后再还原）
        let mut candidates: Vec<RtmoCandidate> = Vec::with_capacity(det_n);
        for i in 0..det_n {
            let base = i * 5;
            let score = det_flat[base + 4];
            if score < self.base.confidence_threshold() {
                continue;
            }
            candidates.push(RtmoCandidate {
                x1: det_flat[base],
                y1: det_flat[base + 1],
                x2: det_flat[base + 2],
                y2: det_flat[base + 3],
                score,
                index: i,
            });
        }

        // 5. NMS（官方 SDK 导出图内已 NMS，此处幂等；SimCC 原生导出必须）
        let kept = self.nms_select(candidates);

        // 6. 保留项还原框 + argmax 解码关键点
        let n_inst = sx.n_inst.min(sy.n_inst);
        let num_kpts = sx.num_kpts;
        let mut results = Vec::with_capacity(kept.len());
        for cand in kept {
            if cand.index >= n_inst {
                return Err(VisionError::inference(format!(
                    "det row {} out of SimCC instance range {n_inst}",
                    cand.index
                )));
            }

            let detection = Detection::new(
                self.base.get_label_name(0), // RTMO 单类 person
                0,
                self.map_to_orig(cand.x1, lb.dw, lb.ratio, orig_width) as f64,
                self.map_to_orig(cand.y1, lb.dh, lb.ratio, orig_height) as f64,
                self.map_to_orig(cand.x2, lb.dw, lb.ratio, orig_width) as f64,
                self.map_to_orig(cand.y2, lb.dh, lb.ratio, orig_height) as f64,
                cand.score as f64,
            );

            let mut keypoints = Vec::with_capacity(num_kpts);
            for k in 0..num_kpts {
                // 逐行 argmax + 行最大值（mmpose get_simcc_maximum 语义）
                let mut bx = 0usize;
                let mut vx = f32::MIN;
                for b in 0..sx.bins {
                    let v = sx.get(cand.index, k, b);
                    if v > vx {
                        vx = v;
                        bx = b;
                    }
                }
                let mut by = 0usize;
                let mut vy = f32::MIN;
                for b in 0..sy.bins {
                    let v = sy.get(cand.index, k, b);
                    if v > vy {
                        vy = v;
                        by = b;
                    }
                }
                // 关键点分数 = 两轴最大响应中的较小者（mmpose 语义），logits 自适应 sigmoid
                let mut kscore = vx.min(vy);
                if need_sigmoid {
                    kscore = sigmoid(kscore);
                }
                // mmpose: locs[vals <= 0] = -1 —— 无效点分数记 0（坐标仍按解码值返回）
                if kscore <= 0.0 {
                    kscore = 0.0;
                }

                // bin 中心 -> 输入空间坐标 -> letterbox 还原到原图
                let kx_input = bx as f32 * ratio_x;
                let ky_input = by as f32 * ratio_y;
                keypoints.push(Keypoint::new(
                    self.map_to_orig(kx_input, lb.dw, lb.ratio, orig_width),
                    self.map_to_orig(ky_input, lb.dh, lb.ratio, orig_height),
                    kscore,
                ));
            }

            results.push(PoseResult::new(detection, keypoints));
        }

        Ok(results)
    }

    // ==================== End2End 布局解码 ====================

    /// 官方 SDK 端到端布局解码：`dets [1,N,5]` + `keypoints [1,N,K,3]`。
    ///
    /// NMS 与 SimCC 解码已在计算图内完成，此处只做阈值过滤（+ 幂等 NMS）、
    /// letterbox 坐标还原与 [`PoseResult`] 组装。
    fn decode_end2end(
        &self,
        det: &TensorOutput,
        others: &[&TensorOutput],
        orig_width: usize,
        orig_height: usize,
        lb: LetterboxParams,
    ) -> Result<Vec<PoseResult>> {
        // keypoints 张量：rank4 且末维=3（[1,N,K,3]），按名含 "keypoint" 优先
        let kps_t = others
            .iter()
            .find(|o| o.name.contains("keypoint"))
            .or_else(|| {
                others
                    .iter()
                    .find(|o| o.shape.len() == 4 && o.shape.last() == Some(&3))
            })
            .ok_or_else(|| {
                VisionError::inference(format!(
                    "End2End layout expects a [1,N,K,3] keypoints output, got: {:?}",
                    others.iter().map(|o| (o.name.as_str(), o.shape.clone())).collect::<Vec<_>>()
                ))
            })?;

        let det_flat = det.as_f32()?;
        let det_n = Self::det_instance_count(det)?;
        if det_n == 0 {
            return Ok(Vec::new());
        }

        // keypoints 展平为 [N, K, 3]
        let kps_flat = kps_t.as_f32()?;
        let kps_shape = &kps_t.shape;
        let (kps_n, num_kpts) = match kps_shape.len() {
            4 => (kps_shape[1] as usize, kps_shape[2] as usize),
            3 => (1, kps_shape[0] as usize), // [K,3] 单实例兜底
            _ => {
                return Err(VisionError::inference(format!(
                    "keypoints tensor rank {} is not supported, shape: {:?}",
                    kps_shape.len(),
                    kps_shape
                )))
            }
        };
        if num_kpts == 0 {
            return Err(VisionError::inference(format!(
                "keypoints tensor has zero keypoint dim, shape: {:?}",
                kps_shape
            )));
        }
        let kps_stride = num_kpts * 3;
        if kps_flat.len() < kps_n * kps_stride {
            return Err(VisionError::inference(format!(
                "keypoints element count {} < {}x{}x3",
                kps_flat.len(),
                kps_n,
                num_kpts
            )));
        }

        // 1. 置信度过滤生成候选
        let mut candidates: Vec<RtmoCandidate> = Vec::with_capacity(det_n);
        for i in 0..det_n {
            let base = i * 5;
            let score = det_flat[base + 4];
            if score < self.base.confidence_threshold() {
                continue;
            }
            candidates.push(RtmoCandidate {
                x1: det_flat[base],
                y1: det_flat[base + 1],
                x2: det_flat[base + 2],
                y2: det_flat[base + 3],
                score,
                index: i,
            });
        }

        // 2. 幂等 NMS（图内已 NMS，重复执行无副作用）
        let kept = self.nms_select(candidates);

        // 3. 还原框与关键点到原图坐标
        let mut results = Vec::with_capacity(kept.len());
        for cand in kept {
            if cand.index >= kps_n {
                return Err(VisionError::inference(format!(
                    "det row {} out of keypoints instance range {kps_n}",
                    cand.index
                )));
            }

            let detection = Detection::new(
                self.base.get_label_name(0), // RTMO 单类 person
                0,
                self.map_to_orig(cand.x1, lb.dw, lb.ratio, orig_width) as f64,
                self.map_to_orig(cand.y1, lb.dh, lb.ratio, orig_height) as f64,
                self.map_to_orig(cand.x2, lb.dw, lb.ratio, orig_width) as f64,
                self.map_to_orig(cand.y2, lb.dh, lb.ratio, orig_height) as f64,
                cand.score as f64,
            );

            let row = cand.index * kps_stride;
            let mut keypoints = Vec::with_capacity(num_kpts);
            for k in 0..num_kpts {
                let base = row + k * 3;
                keypoints.push(Keypoint::new(
                    self.map_to_orig(kps_flat[base], lb.dw, lb.ratio, orig_width),
                    self.map_to_orig(kps_flat[base + 1], lb.dh, lb.ratio, orig_height),
                    kps_flat[base + 2].clamp(0.0, 1.0),
                ));
            }

            results.push(PoseResult::new(detection, keypoints));
        }

        Ok(results)
    }

    // ==================== 公共辅助 ====================

    /// 输入空间坐标 -> 原图坐标（letterbox 反算 + 边界裁剪）。
    #[inline]
    fn map_to_orig(&self, v: f32, d: f32, ratio: f32, limit: usize) -> f32 {
        ((v - d) / ratio).max(0.0).min(limit as f32)
    }

    /// det 张量的实例数 N（支持 [1,N,5] / [N,5]，1D 按行宽 5 切分兜底）。
    fn det_instance_count(det: &TensorOutput) -> Result<usize> {
        let flat = det.as_f32()?;
        let d = &det.shape;
        match d.len() {
            3 => Ok(d[1] as usize),
            2 => Ok(d[0] as usize),
            _ => {
                if flat.len() >= 5 {
                    Ok(flat.len() / 5)
                } else {
                    Err(VisionError::inference(format!(
                        "det tensor rank {} is not supported, shape: {:?}",
                        d.len(),
                        d
                    )))
                }
            }
        }
    }

    /// 从其余张量中挑选 simcc_x / simcc_y（名字优先，兜底按输出顺序 (x, y)）。
    fn pick_simcc_pair<'a>(others: &[&'a TensorOutput]) -> (&'a TensorOutput, &'a TensorOutput) {
        let x = others
            .iter()
            .find(|o| o.name.ends_with("_x") || o.name.contains("simcc_x"))
            .unwrap_or(&others[0]);
        let y = others
            .iter()
            .find(|o| o.name.ends_with("_y") || o.name.contains("simcc_y"))
            .unwrap_or_else(|| {
                others
                    .iter()
                    .find(|o| !std::ptr::eq(*o, x))
                    .unwrap_or(&others[others.len() - 1])
            });
        (x, y)
    }

    /// 按分数降序的贪心 NMS（同 [`crate::engines::pose`]，IoU 复用 `BoundingBox::iou`，
    /// 坐标在输入空间比较；RTMO 单类无类别分组）。
    fn nms_select(&self, mut candidates: Vec<RtmoCandidate>) -> Vec<RtmoCandidate> {
        if candidates.is_empty() {
            return candidates;
        }

        // 按分数降序（稳定排序，等分保持行序）
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        let mut suppressed = vec![false; candidates.len()];
        let mut kept = Vec::new();

        for i in 0..candidates.len() {
            if suppressed[i] {
                continue;
            }
            let curr = candidates[i];
            kept.push(curr);

            let curr_box = BoundingBox::new(
                curr.x1 as f64,
                curr.y1 as f64,
                curr.x2 as f64,
                curr.y2 as f64,
                curr.score as f64,
            );
            for j in (i + 1)..candidates.len() {
                if suppressed[j] {
                    continue;
                }
                let cand = candidates[j];
                let cand_box = BoundingBox::new(
                    cand.x1 as f64,
                    cand.y1 as f64,
                    cand.x2 as f64,
                    cand.y2 as f64,
                    cand.score as f64,
                );
                if curr_box.iou(&cand_box) > self.nms_threshold as f64 {
                    suppressed[j] = true;
                }
            }
        }

        kept
    }

    // ==================== 日志 ====================

    /// 打印推理结果日志（实例明细为 debug 级别，避免高频推理刷屏）。
    fn log_results(&self, results: &[PoseResult]) {
        tracing::info!(
            "RTMO detected {} pose instances{}",
            results.len(),
            if results.is_empty() { "" } else { ":" }
        );
        for r in results {
            tracing::debug!("  {}, keypoints={}", r.detection, r.keypoints.len());
        }
    }
}

/// SimCC 响应是否为原始 logits：任一值出现负值或 >1（+容差）时判为需要补 sigmoid。
///
/// 与 [`crate::engines::pose`] 的关键点置信度自适应策略一致：RTMO 的 SimCC 分支为
/// BCE-logits，原生导出未在图内过 sigmoid；部分导出（含官方 SDK）已在图内完成，
/// 值域 [0,1]，此时不再二次 sigmoid。
fn simcc_need_sigmoid(sx: &SimccView, sy: &SimccView) -> bool {
    let scan = |v: &SimccView| -> (f32, f32) {
        let mut min_v = f32::MAX;
        let mut max_v = -f32::MAX;
        for x in &v.data {
            if *x < min_v {
                min_v = *x;
            }
            if *x > max_v {
                max_v = *x;
            }
        }
        (min_v, max_v)
    };
    let (min_x, max_x) = scan(sx);
    let (min_y, max_y) = scan(sy);
    min_x.min(min_y) < 0.0 || max_x.max(max_y) > 1.0 + 1e-3
}

/// sigmoid（SimCC 响应为原始 logits 时使用）。
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

crate::impl_engine_forward!(RealtimePoseEngine, base, Vec<PoseResult>,
    /// 单图推理。
    fn predict(&self, image: &Image) -> Result<Vec<PoseResult>> {
        self.predict_impl(image)
    }
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::base::TensorData;

    /// 构造测试张量快照。
    fn tensor(name: &str, shape: Vec<i64>, data: Vec<f32>) -> TensorOutput {
        TensorOutput {
            name: name.to_string(),
            shape,
            data: TensorData::F32(data),
        }
    }

    #[test]
    fn simcc_view_rank4_indexing() {
        // [1, 2, 3, 4]：实例 1、关键点 2、bin 3 的值应为 1*12 + 2*4 + 3 = 23
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = tensor("simcc_x", vec![1, 2, 3, 4], data);
        let v = SimccView::from_tensor(&t, 2).unwrap();
        assert_eq!((v.n_inst, v.num_kpts, v.bins), (2, 3, 4));
        assert_eq!(v.get(1, 2, 3), 23.0);
    }

    #[test]
    fn simcc_view_rank3_single_instance() {
        // [17, 320]（单实例 [K,S] 布局）
        let data = vec![0.5f32; 17 * 320];
        let t = tensor("simcc_x", vec![17, 320], data);
        let v = SimccView::from_tensor(&t, 1).unwrap();
        assert_eq!((v.n_inst, v.num_kpts, v.bins), (1, 17, 320));
    }

    #[test]
    fn simcc_view_rank3_per_instance() {
        // [3, 17, 320]（det 有 3 实例时按 [N,K,S] 解释）
        let data = vec![0.5f32; 3 * 17 * 320];
        let t = tensor("simcc_x", vec![3, 17, 320], data);
        let v = SimccView::from_tensor(&t, 3).unwrap();
        assert_eq!((v.n_inst, v.num_kpts, v.bins), (3, 17, 320));
    }

    #[test]
    fn simcc_need_sigmoid_adaptive() {
        // 值域 [0,1]：已在图内过 sigmoid，不再二次 sigmoid
        let sx = SimccView {
            n_inst: 1,
            num_kpts: 1,
            bins: 2,
            data: vec![0.1, 0.9],
        };
        let sy = SimccView {
            n_inst: 1,
            num_kpts: 1,
            bins: 2,
            data: vec![0.2, 0.8],
        };
        assert!(!simcc_need_sigmoid(&sx, &sy));

        // 出现负值：原始 logits，需要补 sigmoid
        let sx_logit = SimccView {
            n_inst: 1,
            num_kpts: 1,
            bins: 2,
            data: vec![-1.0, 3.2],
        };
        assert!(simcc_need_sigmoid(&sx_logit, &sy));
    }

    #[test]
    fn layout_detection_by_name_and_shape() {
        // 官方 SDK 布局：dets + keypoints
        let end2end = vec![
            tensor("dets", vec![1, 50, 5], vec![0.0; 250]),
            tensor("keypoints", vec![1, 50, 17, 3], vec![0.0; 2550]),
        ];
        assert_eq!(RealtimePoseEngine::detect_layout(&end2end).unwrap(), OutputLayout::End2End);

        // SimCC 布局：det_outputs + simcc_x + simcc_y
        let simcc = vec![
            tensor("det_outputs", vec![1, 50, 5], vec![0.0; 250]),
            tensor("simcc_x", vec![1, 17, 320], vec![0.0; 5440]),
            tensor("simcc_y", vec![1, 17, 320], vec![0.0; 5440]),
        ];
        assert_eq!(RealtimePoseEngine::detect_layout(&simcc).unwrap(), OutputLayout::Simcc);

        // 无名兜底：rank4 末维 3 -> End2End
        let anonymous = vec![
            tensor("output0", vec![1, 50, 5], vec![0.0; 250]),
            tensor("output1", vec![1, 50, 17, 3], vec![0.0; 2550]),
        ];
        assert_eq!(RealtimePoseEngine::detect_layout(&anonymous).unwrap(), OutputLayout::End2End);
    }

    #[test]
    fn det_instance_count_shapes() {
        let t3 = tensor("dets", vec![1, 42, 5], vec![0.0; 210]);
        assert_eq!(RealtimePoseEngine::det_instance_count(&t3).unwrap(), 42);
        let t2 = tensor("dets", vec![7, 5], vec![0.0; 35]);
        assert_eq!(RealtimePoseEngine::det_instance_count(&t2).unwrap(), 7);
    }
}
