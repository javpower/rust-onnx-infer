//! 语义分割引擎（SegFormer-B0，ADE20K 150 类城市场景）。
//!
//! 输出逐像素类别图（含类别直方图与调色板叠加可视化）；SegFormer 输出为
//! 输入 1/4 分辨率的 logits，内部双线性上采样到输入尺寸后 argmax。

use crate::core::base::BaseOnnxEngine;
use crate::core::device_type::DeviceType;
use crate::core::engine::OnnxInferenceEngine;
use crate::error::{Result, VisionError};
use crate::imaging::Image;

/// ADE20K 150 类（SegFormer-B0 ADE 官方 id2label，index 即类别 id）。
pub const ADE20K_CLASSES: [&str; 150] = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane", "grass",
    "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant", "curtain",
    "chair", "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "rug",
    "field", "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub",
    "railing", "cushion", "base", "box", "column", "signboard", "chest of drawers", "counter",
    "sand", "sink", "skyscraper", "fireplace", "refrigerator", "grandstand", "path", "stairs",
    "runway", "case", "pool table", "pillow", "screen door", "stairway", "river", "bridge",
    "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill", "bench", "countertop",
    "stove", "palm", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine",
    "hovel", "bus", "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight",
    "booth", "television receiver", "airplane", "dirt track", "apparel", "pole", "land",
    "bannister", "escalator", "ottoman", "bottle", "buffet", "poster", "stage", "van", "ship",
    "fountain", "conveyer belt", "canopy", "washer", "plaything", "swimming pool", "stool",
    "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball", "food",
    "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher",
    "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray", "ashcan",
    "fan", "pier", "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator",
    "glass", "clock", "flag",
];

/// 语义分割结果：逐像素类别图。
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticMap {
    /// 类别 id 图（行主序，尺寸 = 推理输入尺寸）
    pub classes: Vec<u8>,
    /// 图宽
    pub width: usize,
    /// 图高
    pub height: usize,
    /// 类别总数（模型输出通道数）
    pub num_classes: usize,
}

impl SemanticMap {
    /// 类别 id → 名称（越界返回数字字符串）。
    pub fn class_name(&self, id: usize) -> String {
        ADE20K_CLASSES
            .get(id)
            .map(|s| s.to_string())
            .unwrap_or_else(|| id.to_string())
    }

    /// 类别直方图（(类 id, 像素数) 降序，不含未知类）。
    pub fn histogram(&self) -> Vec<(usize, usize)> {
        let mut hist = vec![0usize; self.num_classes];
        for &c in &self.classes {
            hist[c as usize] += 1;
        }
        let mut out: Vec<(usize, usize)> =
            hist.into_iter().enumerate().filter(|(_, n)| *n > 0).collect();
        out.sort_by(|a, b| b.1.cmp(&a.1));
        out
    }

    /// 确定性调色板（黄金角 hue 旋转，可视化用；非 ADE 官方色板）。
    pub fn palette_color(id: u8) -> [u8; 3] {
        let hue = (id as f32 * 137.508) % 360.0; // 黄金角，相邻类区分度最大
        let (r, g, b) = hsv_to_rgb(hue, 0.75, 0.95);
        [r, g, b]
    }

    /// 类别图转调色板 RGB 图。
    pub fn to_palette_image(&self) -> Image {
        let mut img = Image::new(self.width, self.height, 3);
        for (i, &c) in self.classes.iter().enumerate() {
            let color = Self::palette_color(c);
            img.data_mut()[i * 3..i * 3 + 3].copy_from_slice(&color);
        }
        img
    }
}

/// HSV → RGB（h ∈ [0,360)）。
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match (h / 60.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    (
        ((r + m) * 255.0).round() as u8,
        ((g + m) * 255.0).round() as u8,
        ((b + m) * 255.0).round() as u8,
    )
}

/// 语义分割引擎。
pub struct SemanticSegmentationEngine {
    /// 组合基类
    pub base: BaseOnnxEngine,
}

impl SemanticSegmentationEngine {
    /// 创建语义分割引擎（输入尺寸从模型读取，动态回退 512x512）。
    pub fn new(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        Self::with_input_size(model_path, device_type, -1, -1)
    }

    /// 创建语义分割引擎（指定输入尺寸；<=0 时从模型读取，动态模型回退默认）。
    pub fn with_input_size(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        input_height: i32,
        input_width: i32,
    ) -> Result<Self> {
        let mut base =
            BaseOnnxEngine::with_input_size(model_path, device_type, input_height, input_width)?;
        // ADE20K 预处理：RGB + ImageNet mean/std（SegFormer 官方 preprocessor）
        base.set_normalization([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
        if base.input_width() <= 0 || base.input_height() <= 0 {
            tracing::info!("动态输入模型，使用默认 512x512");
            base.input_width = 512;
            base.input_height = 512;
        }
        tracing::info!(
            "Semantic Segmentation Engine initialized: input={}x{}, device={}",
            base.input_width(),
            base.input_height(),
            device_type.name()
        );
        Ok(SemanticSegmentationEngine { base })
    }

    /// 单图推理，返回逐像素类别图（尺寸 = 推理输入尺寸）。
    pub fn predict_impl(&self, image: &Image) -> Result<SemanticMap> {
        let input = self.base.preprocess(image)?;
        let tensor = self.base.create_input_tensor(input)?;
        let output = self.base.run_inference(tensor)?;
        let logits = output.as_f32()?;

        // 输出 [1, C, h, w]（h/w 为输入的 1/4）
        if output.shape.len() != 4 {
            return Err(VisionError::inference(format!(
                "语义分割输出应为 4D logits，实际 shape={:?}",
                output.shape
            )));
        }
        let num_classes = output.shape[1] as usize;
        let h = output.shape[2] as usize;
        let w = output.shape[3] as usize;
        if logits.len() != num_classes * h * w {
            return Err(VisionError::inference(format!(
                "logits 元素数 {} != {}x{}x{}",
                logits.len(),
                num_classes,
                h,
                w
            )));
        }

        // 低分辨率 argmax → 上采样到输入尺寸（对类别 id 最近邻，避免插值出无意义类别）
        let (in_w, in_h) = (self.base.input_width() as usize, self.base.input_height() as usize);
        let argmax_lr = (0..h * w)
            .map(|i| {
                let mut best = 0usize;
                let mut best_v = f32::NEG_INFINITY;
                for c in 0..num_classes {
                    let v = logits[c * h * w + i];
                    if v > best_v {
                        best_v = v;
                        best = c;
                    }
                }
                best as u8
            })
            .collect::<Vec<u8>>();

        let mut classes = vec![0u8; in_w * in_h];
        for y in 0..in_h {
            let sy = (y * h / in_h).min(h - 1);
            for x in 0..in_w {
                let sx = (x * w / in_w).min(w - 1);
                classes[y * in_w + x] = argmax_lr[sy * w + sx];
            }
        }

        Ok(SemanticMap {
            classes,
            width: in_w,
            height: in_h,
            num_classes,
        })
    }

    /// 类别直方图（(类 id, 像素数) 降序）。
    pub fn class_histogram(&self, image: &Image) -> Result<Vec<(usize, usize)>> {
        Ok(self.predict_impl(image)?.histogram())
    }

    /// 调色板叠加可视化（alpha 混合原图）。
    pub fn overlay(&self, image: &Image, alpha: f32) -> Result<Image> {
        let map = self.predict_impl(image)?;
        let palette = map.to_palette_image();
        let overlay_img = crate::imaging::resize(
            &palette,
            image.width(),
            image.height(),
            crate::imaging::Interpolation::Nearest,
        )?;
        let mut out = image.clone();
        let a = alpha.clamp(0.0, 1.0) as f32;
        for (dst, src) in out.data_mut().chunks_exact_mut(3).zip(overlay_img.data().chunks_exact(3)) {
            for k in 0..3 {
                dst[k] = (dst[k] as f32 * (1.0 - a) + src[k] as f32 * a).round().clamp(0.0, 255.0) as u8;
            }
        }
        Ok(out)
    }
}

#[async_trait::async_trait]
impl OnnxInferenceEngine for SemanticSegmentationEngine {
    type Output = SemanticMap;

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

    fn set_confidence_threshold(&mut self, _threshold: f32) {}

    fn confidence_threshold(&self) -> f32 {
        0.0
    }
}
