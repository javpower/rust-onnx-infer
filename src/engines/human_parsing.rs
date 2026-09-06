//! 人体解析引擎（SegFormer-B2 人衣 18 类分区）。
//!
//! 将人物拆成帽子/头发/上衣/裤装/鞋/四肢等 18 类部件区域，用于"是否戴
//! 安全帽/手套""工装合规"等检查的上游。结构与语义分割引擎同源。

use crate::core::base::BaseOnnxEngine;
use crate::core::device_type::DeviceType;
use crate::core::engine::OnnxInferenceEngine;
use crate::error::{Result, VisionError};
use crate::imaging::Image;

/// ATR 人衣 18 类（mattmdjaga/segformer_b2_clothes 官方 id2label）。
pub const CLOTHES_CLASSES: [&str; 18] = [
    "Background",
    "Hat",
    "Hair",
    "Sunglasses",
    "Upper-clothes",
    "Skirt",
    "Pants",
    "Dress",
    "Belt",
    "Left-shoe",
    "Right-shoe",
    "Face",
    "Left-leg",
    "Right-leg",
    "Left-arm",
    "Right-arm",
    "Bag",
    "Scarf",
];

/// 人体解析结果：逐像素部件类别图。
#[derive(Debug, Clone, PartialEq)]
pub struct ParsingMap {
    /// 类别 id 图（行主序，尺寸 = 推理输入尺寸）
    pub classes: Vec<u8>,
    /// 图宽
    pub width: usize,
    /// 图高
    pub height: usize,
}

impl ParsingMap {
    /// 类别 id → 名称。
    pub fn class_name(&self, id: usize) -> &'static str {
        CLOTHES_CLASSES.get(id).copied().unwrap_or("Unknown")
    }

    /// 各部件像素占比（(类 id, 占比 0~1) 降序，不含 Background）。
    pub fn part_ratios(&self) -> Vec<(usize, f32)> {
        let total = (self.width * self.height) as f32;
        let mut hist = vec![0usize; 18];
        for &c in &self.classes {
            hist[c as usize] += 1;
        }
        let mut out: Vec<(usize, f32)> = hist
            .into_iter()
            .enumerate()
            .skip(1)
            .map(|(id, n)| (id, n as f32 / total))
            .filter(|(_, r)| *r > 0.0)
            .collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }
}

/// 人体解析引擎。
pub struct HumanParsingEngine {
    /// 组合基类
    pub base: BaseOnnxEngine,
}

impl HumanParsingEngine {
    /// 创建人体解析引擎（输入尺寸从模型读取，动态回退 512x512）。
    pub fn new(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        Self::with_input_size(model_path, device_type, -1, -1)
    }

    /// 创建人体解析引擎（指定输入尺寸）。
    pub fn with_input_size(
        model_path: impl AsRef<std::path::Path>,
        device_type: DeviceType,
        input_height: i32,
        input_width: i32,
    ) -> Result<Self> {
        let mut base =
            BaseOnnxEngine::with_input_size(model_path, device_type, input_height, input_width)?;
        base.set_normalization([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
        if base.input_width() <= 0 || base.input_height() <= 0 {
            base.input_width = 512;
            base.input_height = 512;
        }
        tracing::info!(
            "Human Parsing Engine initialized: input={}x{}, device={}",
            base.input_width(),
            base.input_height(),
            device_type.name()
        );
        Ok(HumanParsingEngine { base })
    }

    /// 单图推理，返回逐像素部件类别图（尺寸 = 推理输入尺寸）。
    pub fn predict_impl(&self, image: &Image) -> Result<ParsingMap> {
        let input = self.base.preprocess(image)?;
        let tensor = self.base.create_input_tensor(input)?;
        let output = self.base.run_inference(tensor)?;
        let logits = output.as_f32()?;

        if output.shape.len() != 4 {
            return Err(VisionError::inference(format!(
                "人体解析输出应为 4D logits，实际 shape={:?}",
                output.shape
            )));
        }
        let num_classes = output.shape[1] as usize;
        let h = output.shape[2] as usize;
        let w = output.shape[3] as usize;

        // 低分辨率 argmax → 最近邻上采样
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

        let (in_w, in_h) = (self.base.input_width() as usize, self.base.input_height() as usize);
        let mut classes = vec![0u8; in_w * in_h];
        for y in 0..in_h {
            let sy = (y * h / in_h).min(h - 1);
            for x in 0..in_w {
                let sx = (x * w / in_w).min(w - 1);
                classes[y * in_w + x] = argmax_lr[sy * w + sx];
            }
        }

        Ok(ParsingMap {
            classes,
            width: in_w,
            height: in_h,
        })
    }

    /// 各部件像素占比。
    pub fn part_ratios(&self, image: &Image) -> Result<Vec<(usize, f32)>> {
        Ok(self.predict_impl(image)?.part_ratios())
    }
}

#[async_trait::async_trait]
impl OnnxInferenceEngine for HumanParsingEngine {
    type Output = ParsingMap;

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
