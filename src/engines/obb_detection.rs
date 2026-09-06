//! YOLO-OBB 旋转框检测引擎（YOLOv8/11-OBB 传统布局 + YOLO26-OBB End2End 布局）。
//!
//! 输出为带角度的旋转框（中心点 + 宽高 + 弧度角），后处理含旋转 IoU NMS
//! （多边形 Sutherland-Hodgman 精确求交）。DOTA 数据集 15 类（航拍），对自然
//! 场景图为跨域使用，检出分数偏低属正常。

use std::sync::Mutex;

use crate::core::base::BaseOnnxEngine;
use crate::core::device_type::DeviceType;
use crate::core::engine::OnnxInferenceEngine;
use crate::error::{Result, VisionError};
use crate::imaging::Image;

/// DOTA 1.0 十五类（Ultralytics OBB 默认类别表，index 即类别 id）。
pub const DOTA_CLASSES: [&str; 15] = [
    "plane",
    "ship",
    "storage-tank",
    "baseball-diamond",
    "tennis-court",
    "basketball-court",
    "ground-track-field",
    "harbor",
    "bridge",
    "large-vehicle",
    "small-vehicle",
    "helicopter",
    "roundabout",
    "soccer-ball-field",
    "swimming-pool",
];

/// 默认置信度阈值。
pub const DEFAULT_CONF_THRESHOLD: f32 = 0.25;
/// 默认旋转 NMS IoU 阈值。
pub const DEFAULT_NMS_THRESHOLD: f32 = 0.45;

/// 旋转框检测结果。
#[derive(Debug, Clone, PartialEq)]
pub struct ObbResult {
    /// 中心点 x（原图像素）
    pub cx: f32,
    /// 中心点 y（原图像素）
    pub cy: f32,
    /// 宽（原图像素，angle=0 方向）
    pub w: f32,
    /// 高（原图像素，垂直于 w 方向）
    pub h: f32,
    /// 旋转角（弧度，[0, π/2)，Ultralytics OBB 约定）
    pub angle_rad: f32,
    /// 四角点（原图像素，按顺时针顺序）
    pub corners: [[f32; 2]; 4],
    /// 类别名
    pub class_name: String,
    /// 类别 id
    pub class_id: i32,
    /// 置信度
    pub confidence: f64,
}

impl ObbResult {
    /// 旋转框在原图上的轴对齐外接矩形 (x1, y1, x2, y2)。
    pub fn aabb(&self) -> (f32, f32, f32, f32) {
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;
        for c in &self.corners {
            min_x = min_x.min(c[0]);
            min_y = min_y.min(c[1]);
            max_x = max_x.max(c[0]);
            max_y = max_y.max(c[1]);
        }
        (min_x, min_y, max_x, max_y)
    }
}

/// YOLO-OBB 旋转框检测引擎。
pub struct ObbDetectionEngine {
    /// 组合基类
    pub base: BaseOnnxEngine,
    /// 旋转 NMS IoU 阈值
    nms_threshold: f32,
    /// 布局自动检测结果（首次推理时解析并缓存）
    layout: Mutex<Option<ObbLayout>>,
}

/// 输出布局。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ObbLayout {
    /// 传统 [1, 4+1+nc, anchors]（YOLOv8/11-OBB）
    Traditional,
    /// End2End [1, M, 7]（YOLO26-OBB：cx,cy,w,h,angle,cls_id,conf）
    End2End,
}

impl ObbDetectionEngine {
    /// 创建 OBB 引擎（输入尺寸从模型读取，动态回退 1024）。
    pub fn new(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        Self::with_input_size(model_path, device_type, -1, -1)
    }

    /// 创建 OBB 引擎（指定输入尺寸；<=0 时从模型读取）。
    pub fn with_input_size(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        input_height: i32,
        input_width: i32,
    ) -> Result<Self> {
        let base =
            BaseOnnxEngine::with_input_size(model_path, device_type, input_height, input_width)?;
        // 类别表：模型元数据优先，缺省用 DOTA 15 类
        let mut base = base;
        if base.labels().is_none() {
            base.set_labels(DOTA_CLASSES.iter().map(|s| s.to_string()).collect());
        }
        tracing::info!(
            "OBB Engine initialized: input={}x{}, device={}",
            base.input_width(),
            base.input_height(),
            device_type.name()
        );
        Ok(ObbDetectionEngine {
            base,
            nms_threshold: DEFAULT_NMS_THRESHOLD,
            layout: Mutex::new(None),
        })
    }

    /// 旋转 NMS IoU 阈值。
    pub fn nms_threshold(&self) -> f32 {
        self.nms_threshold
    }

    /// 设置旋转 NMS IoU 阈值。
    pub fn set_nms_threshold(&mut self, threshold: f32) {
        self.nms_threshold = threshold;
    }

    /// 单图推理（SAHI 开启时自动切片，返回类型不变；合并以 aabb 近似）。
    pub fn predict_impl(&self, image: &Image) -> Result<Vec<ObbResult>> {
        if self.base.is_sahi_enabled() {
            let config = self
                .base
                .sahi_config()
                .expect("sahi config must exist when SAHI enabled");
            return crate::sahi::sliced_predictor::predict_sliced_obb(self, self, image, config)
                .map(|result| result.detections);
        }
        self.predict_without_sahi(image)
    }

    /// 原始单图推理路径（SAHI 关闭时的 `predict` 行为；开启时被逐切片调用）。
    pub fn predict_without_sahi(&self, image: &Image) -> Result<Vec<ObbResult>> {
        let (input, lb) = self.preprocess_with_letterbox(image)?;
        let tensor = self.base.create_input_tensor(input)?;
        let output = self.base.run_inference(tensor)?;

        // 布局判定（首次解析后缓存）
        let layout = {
            let mut guard = self.layout.lock().unwrap();
            match *guard {
                Some(l) => l,
                None => {
                    let l = detect_layout(&output);
                    tracing::info!("OBB 输出布局: {:?}（shape={:?}）", l, output.shape);
                    *guard = Some(l);
                    l
                }
            }
        };

        let orig_w = image.width() as f32;
        let orig_h = image.height() as f32;
        let mut results = match layout {
            ObbLayout::Traditional => self.postprocess_traditional(&output, lb, orig_w, orig_h)?,
            ObbLayout::End2End => self.postprocess_end2end(&output, lb, orig_w, orig_h)?,
        };
        results.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(results)
    }

    // ==================== 预处理 ====================

    /// letterbox 预处理（114 灰边 + /255 + BGR→RGB + HWC→CHW），参数随调用链返回。
    fn preprocess_with_letterbox(&self, image: &Image) -> Result<(Vec<f32>, Letterbox)> {
        if image.is_empty() {
            return Err(VisionError::image("cannot letterbox empty image"));
        }
        let orig_w = image.width() as usize;
        let orig_h = image.height() as usize;
        let in_w = self.base.input_width() as usize;
        let in_h = self.base.input_height() as usize;

        let ratio = (in_w as f32 / orig_w as f32).min(in_h as f32 / orig_h as f32);
        let new_w = (orig_w as f32 * ratio).round() as usize;
        let new_h = (orig_h as f32 * ratio).round() as usize;
        let dw = (in_w - new_w) as f32 / 2.0;
        let dh = (in_h - new_h) as f32 / 2.0;

        let resized =
            crate::imaging::resize(image, new_w, new_h, crate::imaging::Interpolation::Linear)?;
        let mut canvas = Image::filled(in_w, in_h, 3, 114);
        canvas.paste(dw.round() as usize, dh.round() as usize, &resized);

        // BGR→RGB + /255 + HWC→CHW
        let rgb = crate::imaging::cvt_color(&canvas, crate::imaging::ColorConversion::Bgr2Rgb)?;
        let area = in_w * in_h;
        let mut chw = vec![0f32; 3 * area];
        let px = rgb.data();
        for i in 0..area {
            chw[i] = px[i * 3] as f32 / 255.0;
            chw[i + area] = px[i * 3 + 1] as f32 / 255.0;
            chw[i + 2 * area] = px[i * 3 + 2] as f32 / 255.0;
        }
        Ok((chw, Letterbox { ratio, dw, dh }))
    }

    // ==================== 传统布局 [1, 4+1+nc, anchors] ====================

    fn postprocess_traditional(
        &self,
        output: &crate::core::base::TensorOutput,
        lb: Letterbox,
        orig_w: f32,
        orig_h: f32,
    ) -> Result<Vec<ObbResult>> {
        let data = output.as_f32()?;
        let shape = &output.shape;
        if shape.len() != 3 {
            return Err(VisionError::inference(format!(
                "OBB 传统布局应为 3D 输出，实际 shape={shape:?}"
            )));
        }
        let channels = shape[1] as usize;
        let anchors = shape[2] as usize;
        if channels < 6 {
            return Err(VisionError::inference(format!(
                "OBB 传统布局通道数 {channels} < 6（4 box + 1 angle + ≥1 类）"
            )));
        }
        let num_classes = channels - 5;

        // 行主序取 [channels][anchors]
        let row = |c: usize, a: usize| -> f32 { data[c * anchors + a] };

        // 布局：[cx,cy,w,h, cls_logits×nc, angle]（angle 在最后一通道，图内已做
        // (sigmoid-0.25)·π 变换，值域 [-π/4, 3π/4]；实测 class 通道 logits 值域
        // 可 >1，与官方 NMS 的 conf 语义一致）。
        let angle_ch = channels - 1;

        // 阈值语义与 ultralytics non_max_suppression 对齐：类分数是 logits，
        // 官方直接 `cls > conf_thres` 过滤并输出 logits 原值（不做 sigmoid）。
        let mut candidates: Vec<(f32, f32, f32, f32, f32, f32, i32)> = Vec::new();
        for a in 0..anchors {
            let mut best_cls = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            for c in 0..num_classes {
                let v = row(4 + c, a);
                if v > best_score {
                    best_score = v;
                    best_cls = c;
                }
            }
            if best_score < self.base.confidence_threshold() {
                continue;
            }
            candidates.push((
                row(0, a),
                row(1, a),
                row(2, a),
                row(3, a),
                row(angle_ch, a),
                best_score,
                best_cls as i32,
            ));
        }

        // 分数降序 + 旋转 NMS
        candidates.sort_by(|x, y| y.5.partial_cmp(&x.5).unwrap_or(std::cmp::Ordering::Equal));
        let mut results = Vec::new();
        let mut kept_corners: Vec<[[f32; 2]; 4]> = Vec::new();
        for (cx, cy, w, h, angle, score, cls) in candidates {
            let (ocx, ocy, ow, oh) = restore_wh(lb, cx, cy, w, h, orig_w, orig_h);
            let corners = corners_of(ocx, ocy, ow, oh, angle);
            let suppressed = kept_corners
                .iter()
                .any(|k| rotated_iou(&corners, k) > self.nms_threshold);
            if suppressed {
                continue;
            }
            kept_corners.push(corners);
            results.push(ObbResult {
                cx: ocx,
                cy: ocy,
                w: ow,
                h: oh,
                angle_rad: angle,
                corners,
                class_name: self.base.get_label_name(cls),
                class_id: cls,
                confidence: score as f64,
            });
        }
        Ok(results)
    }

    // ==================== End2End [1, M, 7] ====================

    fn postprocess_end2end(
        &self,
        output: &crate::core::base::TensorOutput,
        lb: Letterbox,
        orig_w: f32,
        orig_h: f32,
    ) -> Result<Vec<ObbResult>> {
        let data = output.as_f32()?;
        let shape = &output.shape;
        if shape.len() != 3 {
            return Err(VisionError::inference(format!(
                "OBB End2End 布局应为 3D 输出，实际 shape={shape:?}"
            )));
        }
        let rows = shape[1] as usize;
        let cols = shape[2] as usize;
        if cols < 7 {
            return Err(VisionError::inference(format!(
                "OBB End2End 每行应有 ≥7 列（cx,cy,w,h,angle,cls_id,conf），实际 {cols}"
            )));
        }

        let mut results = Vec::new();
        let mut kept_corners: Vec<[[f32; 2]; 4]> = Vec::new();
        for r in 0..rows {
            let row = &data[r * cols..(r + 1) * cols];
            // 官方 End2End OBB postprocess 注释：[x, y, w, h, max_class_prob, class_index, angle]
            let conf = adaptive_sigmoid(row[4]);
            if conf < self.base.confidence_threshold() {
                continue;
            }
            let cls_id = row[5].round() as i32;
            let (cx, cy, w, h) = restore_wh(lb, row[0], row[1], row[2], row[3], orig_w, orig_h);
            let angle = row[6];
            let corners = corners_of(cx, cy, w, h, angle);
            let suppressed = kept_corners
                .iter()
                .any(|k| rotated_iou(&corners, k) > self.nms_threshold);
            if suppressed {
                continue;
            }
            kept_corners.push(corners);
            results.push(ObbResult {
                cx,
                cy,
                w,
                h,
                angle_rad: angle,
                corners,
                class_name: self.base.get_label_name(cls_id),
                class_id: cls_id,
                confidence: conf as f64,
            });
        }
        Ok(results)
    }
}

/// letterbox 参数（随调用链传递）。
#[derive(Debug, Clone, Copy)]
struct Letterbox {
    ratio: f32,
    dw: f32,
    dh: f32,
}

/// 输出布局判定：[M(≤400), 7] → End2End；否则传统。
fn detect_layout(output: &crate::core::base::TensorOutput) -> ObbLayout {
    let s = &output.shape;
    if s.len() == 3 {
        let (d1, d2) = (s[1], s[2]);
        if (1..=400).contains(&d1) && d2 == 7 {
            return ObbLayout::End2End;
        }
    }
    ObbLayout::Traditional
}

/// sigmoid。
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// 自适应 sigmoid：值域已在 [0,1] 时原值返回，logits（有负值）补 sigmoid。
/// YOLO26-OBB End2End 的 conf 列实测存在负值（logits）。
fn adaptive_sigmoid(v: f32) -> f32 {
    if (0.0..=1.0).contains(&v) {
        v
    } else {
        sigmoid(v)
    }
}

/// 坐标还原：中心点减 padding 除 ratio；宽高只除 ratio；clamp 到原图。
fn restore_wh(
    lb: Letterbox,
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
    orig_w: f32,
    orig_h: f32,
) -> (f32, f32, f32, f32) {
    let cx = ((cx - lb.dw) / lb.ratio).clamp(0.0, orig_w);
    let cy = ((cy - lb.dh) / lb.ratio).clamp(0.0, orig_h);
    let w = (w / lb.ratio).clamp(1.0, orig_w * 2.0);
    let h = (h / lb.ratio).clamp(1.0, orig_h * 2.0);
    (cx, cy, w, h)
}

/// 中心 + 宽高 + 角度 → 四角点（顺时针；标准旋转矩阵 [[cos,-sin],[sin,cos]]）。
fn corners_of(cx: f32, cy: f32, w: f32, h: f32, angle: f32) -> [[f32; 2]; 4] {
    let (c, s) = (angle.cos(), angle.sin());
    let (hw, hh) = (w / 2.0, h / 2.0);
    // 局部坐标四角（顺时针：左上、右上、右下、左下）
    let local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)];
    let mut corners = [[0f32; 2]; 4];
    for (i, (lx, ly)) in local.iter().enumerate() {
        corners[i][0] = cx + lx * c - ly * s;
        corners[i][1] = cy + lx * s + ly * c;
    }
    corners
}

/// 多边形面积（鞋带公式，顶点有序；绝对值）。
fn polygon_area(poly: &[[f32; 2]]) -> f32 {
    shoelace_signed(poly).abs()
}

/// 有向面积（鞋带公式，带符号）。
fn shoelace_signed(poly: &[[f32; 2]]) -> f32 {
    let n = poly.len();
    let mut area = 0.0f32;
    for i in 0..n {
        let j = (i + 1) % n;
        area += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1];
    }
    area / 2.0
}

/// 两线段交点（p1p2 与 p3p4；平行/共线返回 None）。
fn line_intersection(p1: [f32; 2], p2: [f32; 2], p3: [f32; 2], p4: [f32; 2]) -> Option<[f32; 2]> {
    let d1x = p2[0] - p1[0];
    let d1y = p2[1] - p1[1];
    let d2x = p4[0] - p3[0];
    let d2y = p4[1] - p3[1];
    let denom = d1x * d2y - d1y * d2x;
    if denom.abs() < 1e-9 {
        return None;
    }
    let t = ((p3[0] - p1[0]) * d2y - (p3[1] - p1[1]) * d2x) / denom;
    Some([p1[0] + t * d1x, p1[1] + t * d1y])
}

/// 点相对有向边 a→b 的内侧判定（边统一为逆时针方向后 cross ≥ 0 为内侧）。
fn point_in_poly_edge(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> bool {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0.0
}

/// Sutherland–Hodgman 多边形裁剪：用凸四边形 `clip` 裁剪 `subject`，
/// 返回交集多边形（顶点数 0~8）。方向自动统一。
fn sutherland_hodgman(subject: &[[f32; 2]; 4], clip: &[[f32; 2]; 4]) -> Vec<[f32; 2]> {
    let mut clip_poly = *clip;
    if shoelace_signed(&clip_poly) < 0.0 {
        clip_poly.reverse();
    }
    let mut output: Vec<[f32; 2]> = subject.to_vec();
    if shoelace_signed(&output) < 0.0 {
        output.reverse();
    }

    for i in 0..4 {
        if output.is_empty() {
            return output;
        }
        let a = clip_poly[i];
        let b = clip_poly[(i + 1) % 4];
        let input = output.clone();
        output.clear();
        let n = input.len();
        for j in 0..n {
            let cur = input[j];
            let prev = input[(j + n - 1) % n];
            let cur_inside = point_in_poly_edge(cur, a, b);
            let prev_inside = point_in_poly_edge(prev, a, b);
            if cur_inside {
                if !prev_inside {
                    if let Some(x) = line_intersection(prev, cur, a, b) {
                        output.push(x);
                    }
                }
                output.push(cur);
            } else if prev_inside {
                if let Some(x) = line_intersection(prev, cur, a, b) {
                    output.push(x);
                }
            }
        }
    }
    output
}

/// 两个旋转框（四角点表示）的 IoU：多边形精确求交。
pub fn rotated_iou(a: &[[f32; 2]; 4], b: &[[f32; 2]; 4]) -> f32 {
    let area_a = polygon_area(a);
    let area_b = polygon_area(b);
    if area_a <= 0.0 || area_b <= 0.0 {
        return 0.0;
    }
    let inter_poly = sutherland_hodgman(a, b);
    let inter = polygon_area(&inter_poly);
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        (inter / union).clamp(0.0, 1.0)
    }
}

#[async_trait::async_trait]
impl OnnxInferenceEngine for ObbDetectionEngine {
    type Output = Vec<ObbResult>;

    fn predict(&self, image: &Image) -> Result<Self::Output> {
        self.predict_impl(image)
    }

    fn input_size(&self) -> (i32, i32) {
        self.base.input_size()
    }

    fn labels(&self) -> Option<&[String]> {
        self.base.labels()
    }

    fn set_labels(&mut self, labels: Vec<String>) {
        self.base.set_labels(labels)
    }

    fn set_confidence_threshold(&mut self, threshold: f32) {
        self.base.set_confidence_threshold(threshold)
    }

    fn confidence_threshold(&self) -> f32 {
        self.base.confidence_threshold()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 同一框 IoU = 1。
    #[test]
    fn rotated_iou_identical() {
        let r = corners_of(10.0, 10.0, 20.0, 10.0, 0.3);
        assert!((rotated_iou(&r, &r) - 1.0).abs() < 1e-5);
    }

    /// 轴对齐部分重叠：a 占 [-1,1]，b 占 [0,2]，交集 1x2=2，并集 6 → 1/3。
    #[test]
    fn rotated_iou_axis_overlap() {
        let a = corners_of(0.0, 0.0, 2.0, 2.0, 0.0);
        let b = corners_of(1.0, 0.0, 2.0, 2.0, 0.0);
        assert!((rotated_iou(&a, &b) - 1.0 / 3.0).abs() < 1e-4);
    }

    /// 同心 45° 交叉：正八边形交集，解析值 IoU = √2/2 ≈ 0.7071。
    #[test]
    fn rotated_iou_cross() {
        let a = corners_of(0.0, 0.0, 2.0, 2.0, 0.0);
        let b = corners_of(0.0, 0.0, 2.0, 2.0, std::f32::consts::FRAC_PI_4);
        let iou = rotated_iou(&a, &b);
        assert!((iou - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-4, "45° 交叉 IoU={iou}");
    }

    /// 平移重叠：交集 1x29=29，并集 2×261−29=493 → 0.0588（多边形法精确值）。
    #[test]
    fn rotated_iou_shift() {
        let a = corners_of(100.0, 100.0, 9.0, 29.0, 0.0);
        let b = corners_of(108.0, 100.0, 9.0, 29.0, 0.0);
        let iou = rotated_iou(&a, &b);
        assert!((iou - 29.0 / 493.0).abs() < 1e-4, "平移 8px 的 9x29 框 IoU={iou}");
    }

    /// 完全分离 → 0。
    #[test]
    fn rotated_iou_disjoint() {
        let a = corners_of(0.0, 0.0, 2.0, 2.0, 0.0);
        let b = corners_of(100.0, 100.0, 2.0, 2.0, 0.7);
        assert!(rotated_iou(&a, &b) < 1e-6);
    }
}
