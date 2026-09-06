//! OSNet x1.0 行人重识别（Person ReID）引擎（torchreid 标准，512 维 embedding）。
//!
//! 模型 `testmodels/osnet_x1_0.onnx`（9.46 MB，单文件）：
//! Qualcomm AI Hub 官方导出（v0.61.0），权重为 torchreid 的
//! `osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip`
//! （Market-1501 训练、OSNet 论文标准 256x128 输入）。
//! 实测签名（`cargo run --release --example model_probe -- --run=256x128 testmodels/osnet_x1_0.onnx`）：
//!
//! - 输入 `image` `[1,3,256,128]` Float32（H=256、W=128，**静态**）
//! - 输出 `embeddings` `[1,512]` Float32（图内 ReduceL2+Div 已做 L2 归一化）
//!
//! # 预处理决定（已查证，任务书中"无 mean/std"的说法与源码不符）
//!
//! - torchreid 测试时变换（`deep-person-reid/torchreid/data/transforms.py` 的
//!   `build_transforms`，逐行核实）：`Resize((h,w)) + ToTensor() + Normalize(
//!   mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`——即 **/255 后做 ImageNet
//!   mean/std 归一化，RGB 通道序**（norm_mean/norm_std 为 None 时同样回退 ImageNet 值）。
//! - 本 ONNX 文件在导出时（`normalize_image_torchvision`）已把该归一化折叠进计算图：
//!   首两层为 `Sub(image, [0.485,0.456,0.406]) → Div(_, [0.229,0.224,0.225])`（逐通道常量，
//!   实测读取 initializer 确认），因此**图外输入 = RGB、[0,1]**（仅 /255，不再做 mean/std）。
//! - Resize 为拉伸到 128x256（宽x高），对应 torchvision `Resize((256,128))` 默认双线性，
//!   本库取 [`Interpolation::Linear`]。
//! - 输出 embedding 图内已 L2，[`ReidEngine::embed`] 内再次 L2 为幂等兜底（零向量除外）。
//!
//! # API
//!
//! - [`ReidEngine::embed`]：按行人框（检测框）裁剪并提取 512 维 L2 归一化特征；
//! - [`ReidEngine::cosine_similarity`]：两个 embedding 的余弦相似度；
//! - [`ReidGallery`]：简易特征图库（`add` / `search`，线性扫描余弦，按 id 聚合取最优）。

use crate::core::base::BaseOnnxEngine;
use crate::core::device_type::DeviceType;
use crate::error::{Result, VisionError};
use crate::imaging::{resize, Image, Interpolation, Rect};

/// OSNet 标准输入高（H=256，torchreid 默认 256x128 的高）。
const INPUT_HEIGHT: i32 = 256;
/// OSNet 标准输入宽（W=128）。
const INPUT_WIDTH: i32 = 128;
/// 输出 embedding 维度（OSNet x1.0 为 512 维）。
const FEATURE_DIM: usize = 512;

/// 行人重识别（特征提取）引擎。
///
/// 典型流水线：[`crate::engines::detection`]（或行人检测）产出 person 框 →
/// [`ReidEngine::embed`] 提特征 → [`ReidGallery`] 检索 / 余弦比对。
pub struct ReidEngine {
    /// 组合基类
    pub base: BaseOnnxEngine,
}

impl ReidEngine {
    /// 创建 OSNet x1.0 行人重识别引擎（torchreid 预处理约定，归一化折叠在图内）。
    ///
    /// 输入固定 256x128（HxW，模型为静态维度）。
    pub fn new(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        let mut base =
            BaseOnnxEngine::with_input_size(model_path, device_type, INPUT_HEIGHT, INPUT_WIDTH)?;
        // 归一化已折叠进模型计算图（图首 Sub/Div = ImageNet mean/std，RGB 序），
        // 图外仅需 /255：即基类默认 (x/255 - 0) / 1，此处显式写出以固定约定
        base.set_normalization([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        Ok(ReidEngine { base })
    }

    // ============ 核心 API ============

    /// 按行人框提取 512 维 L2 归一化特征（主 API）。
    ///
    /// `person_box`：行人检测框（图像坐标，可由 [`crate::engines::detection`] 输出）；
    /// 允许越界/负坐标，超出图像的部分被裁掉，与图像无交集时报错。
    /// 返回长度 512 的 L2 归一化 embedding（模长为 1）。
    pub fn embed(&self, image: &Image, person_box: Rect) -> Result<Vec<f32>> {
        if image.is_empty() {
            return Err(VisionError::image("cannot embed empty image"));
        }
        let (x, y, w, h) = clamp_roi(image, &person_box)?;
        let crop = image.crop(x, y, w, h)?;
        self.embed_crop(&crop)
    }

    /// 计算两个 embedding 的余弦相似度（关联函数，含维度校验，维度不一致时 panic）。
    ///
    /// 输入应为 [`Self::embed`] 返回的 L2 归一化向量；非归一化向量结果同样正确
    /// （内部会除以各自模长），范围 [-1, 1]。
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        assert!(
            a.len() == b.len(),
            "cosine_similarity: embedding 维度不一致 {} vs {}",
            a.len(),
            b.len()
        );
        let mut dot = 0f32;
        let mut norm_a = 0f32;
        let mut norm_b = 0f32;
        for (&va, &vb) in a.iter().zip(b.iter()) {
            dot += va * vb;
            norm_a += va * va;
            norm_b += vb * vb;
        }
        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom <= f32::EPSILON {
            return 0.0;
        }
        (dot / denom).clamp(-1.0, 1.0)
    }

    // ==================== 内部实现 ====================

    /// 整图单框特征提取（trait `predict` 语义：把整图视作一个行人框）；
    /// 正式使用请优先 [`Self::embed`]（行人框裁剪）。
    pub fn predict_impl(&self, image: &Image) -> Result<Vec<f32>> {
        self.embed_crop(image)
    }

    /// 预处理 + 推理 + L2 归一化（输入为行人 crop，任意尺寸）。
    fn embed_crop(&self, crop: &Image) -> Result<Vec<f32>> {
        // 1. 拉伸 resize 到 128x256（宽x高；torchvision Resize 默认双线性）
        let resized = resize(
            crop,
            INPUT_WIDTH as usize,
            INPUT_HEIGHT as usize,
            Interpolation::Linear,
        )?;
        // 2. BGR→RGB + /255 + CHW（基类默认 mean=0/std=1；ImageNet 归一化在图内）
        let input_data = self.base.preprocess(&resized)?;

        // 3. 推理
        let input_tensor = self.base.create_input_tensor(input_data)?;
        let output = self.base.run_inference(input_tensor)?;

        // 4. 维度校验 + L2 归一化（图内已 L2，此处幂等兜底）
        let embedding = output.as_f32()?.to_vec();
        if embedding.len() != FEATURE_DIM {
            return Err(VisionError::inference(format!(
                "ReID embedding 维度不符: 期望 {FEATURE_DIM}，实际 {}（模型输出 shape={:?}）",
                embedding.len(),
                output.shape
            )));
        }
        Ok(l2_normalize(embedding))
    }
}

/// 把行人框裁剪到图像范围内（负坐标与越界部分丢弃），返回 `(x, y, w, h)`。
///
/// 与图像无交集或面积为 0 时报错（空 crop 无法提取有效特征）。
fn clamp_roi(image: &Image, person_box: &Rect) -> Result<(usize, usize, usize, usize)> {
    let (iw, ih) = (image.width() as i32, image.height() as i32);
    let x0 = person_box.x.clamp(0, iw);
    let y0 = person_box.y.clamp(0, ih);
    let x1 = person_box.right().clamp(0, iw);
    let y1 = person_box.bottom().clamp(0, ih);
    if x1 <= x0 || y1 <= y0 {
        return Err(VisionError::invalid_argument(format!(
            "行人框与图像无交集: rect=({},{},{},{}) image={iw}x{ih}",
            person_box.x, person_box.y, person_box.width, person_box.height
        )));
    }
    Ok((x0 as usize, y0 as usize, (x1 - x0) as usize, (y1 - y0) as usize))
}

/// 简易 ReID 特征图库（线性扫描余弦匹配，无索引结构）。
///
/// 适合几百条以内的规模；更大规模需外接 ANN 索引（TODO：如 hnsw/faiss 类后端）。
/// 同一 `id` 可多次 `add`（同一人的不同 crop），检索时按 id 聚合取最大相似度。
#[derive(Debug, Clone, Default)]
pub struct ReidGallery {
    entries: Vec<GalleryEntry>,
}

/// 图库条目（身份 id + L2 归一化特征）。
#[derive(Debug, Clone)]
struct GalleryEntry {
    id: u64,
    embedding: Vec<f32>,
}

impl ReidGallery {
    /// 创建空图库。
    pub fn new() -> Self {
        Self::default()
    }

    /// 添加一条特征（同一 id 可多次添加，检索时按 id 聚合取最优相似度）。
    pub fn add(&mut self, id: u64, embedding: Vec<f32>) {
        self.entries.push(GalleryEntry { id, embedding });
    }

    /// 图库条目数（同一 id 的多条特征分别计数）。
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 图库是否为空。
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 检索：线性扫描余弦相似度，返回相似度降序的前 `top_k` 个 `(id, 相似度)`。
    ///
    /// 同一 id 的多条特征聚合为一条（取最大相似度），因此结果中 id 不重复；
    /// `top_k` 为 0 时返回空表。
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(u64, f32)> {
        // 逐条扫描，按 id 聚合最优相似度
        let mut best: Vec<(u64, f32)> = Vec::with_capacity(self.entries.len());
        for entry in &self.entries {
            let sim = ReidEngine::cosine_similarity(query, &entry.embedding);
            match best.iter_mut().find(|(id, _)| *id == entry.id) {
                Some((_, s)) => {
                    if sim > *s {
                        *s = sim;
                    }
                }
                None => best.push((entry.id, sim)),
            }
        }
        // 相似度降序，取前 top_k
        best.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        best.truncate(top_k);
        best
    }
}

/// L2 归一化（零向量保持为零）。
fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    v
}

crate::impl_engine_forward!(ReidEngine, base, Vec<f32>,
    /// 单图推理：整图视作单个行人框提取特征（拉伸 resize 到 128x256）；
    /// 行人框裁剪提取请使用 [`ReidEngine::embed`]。返回 L2 归一化 512 维 embedding。
    fn predict(&self, image: &Image) -> Result<Vec<f32>> {
        self.predict_impl(image)
    }
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imaging::Rect;

    // ==================== 纯逻辑单测（无需模型） ====================

    #[test]
    fn cosine_similarity_basic() {
        // L2 归一化向量的点积即余弦
        let a = [1.0, 0.0, 0.0];
        assert_eq!(ReidEngine::cosine_similarity(&a, &a), 1.0);
        assert_eq!(ReidEngine::cosine_similarity(&a, &[0.0, 1.0, 0.0]), 0.0);
        assert_eq!(ReidEngine::cosine_similarity(&a, &[-1.0, 0.0, 0.0]), -1.0);
        // 非归一化向量同样正确
        assert!((ReidEngine::cosine_similarity(&[2.0, 0.0], &[3.0, 0.0]) - 1.0).abs() < 1e-6);
        // 零向量 → 0
        assert_eq!(ReidEngine::cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }

    #[test]
    fn l2_normalize_makes_unit_vector() {
        let v = l2_normalize(vec![3.0, 4.0]);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
        // 幂等：已归一化向量再归一化不变
        let v2 = l2_normalize(v.clone());
        assert!(v.iter().zip(v2.iter()).all(|(a, b)| (a - b).abs() < 1e-6));
        // 零向量保持为零
        assert_eq!(l2_normalize(vec![0.0; 4]), vec![0.0; 4]);
    }

    #[test]
    fn gallery_search_orders_by_similarity() {
        let mut gallery = ReidGallery::new();
        assert!(gallery.is_empty());
        gallery.add(1, vec![1.0, 0.0, 0.0]);
        gallery.add(2, vec![0.0, 1.0, 0.0]);
        gallery.add(3, vec![0.7, 0.7, 0.0]);
        gallery.add(1, vec![0.9, 0.1, 0.0]); // 同 id 第二条特征
        assert_eq!(gallery.len(), 4);

        let query = vec![1.0, 0.0, 0.0];
        let top = gallery.search(&query, 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 1, "id=1 最相似，应排第一");
        assert!((top[0].1 - 1.0).abs() < 1e-6);
        assert_eq!(top[1].0, 3, "id=3 次之（cos≈0.707）");
        // id 不重复（同 id 多条聚合取最优）
        let all = gallery.search(&query, 10);
        assert_eq!(all.len(), 3);
        assert!(all.windows(2).all(|w| w[0].1 >= w[1].1), "应按相似度降序");
    }

    #[test]
    fn clamp_roi_crops_out_of_bounds() {
        let img = Image::new(100, 200, 3);
        // 完全在图内
        assert_eq!(clamp_roi(&img, &Rect::new(10, 20, 30, 40)).unwrap(), (10, 20, 30, 40));
        // 负坐标 + 越界 → 收缩到交集（左裁到 0，上下越界部分裁掉）
        assert_eq!(clamp_roi(&img, &Rect::new(-10, -5, 50, 300)).unwrap(), (0, 0, 40, 200));
        // 与图像无交集 → 报错
        assert!(clamp_roi(&img, &Rect::new(200, 0, 10, 10)).is_err());
        assert!(clamp_roi(&img, &Rect::new(0, 0, 0, 10)).is_err());
    }

    // ==================== 真实模型自验（需 testmodels/，默认忽略） ====================

    /// 自验（bus.jpg）：
    /// 1. 同一人两个不同 crop（中心行人完整框 vs 收紧框）余弦相似度 > 0.5；
    /// 2. 同图"非行人区域"（巴士车身）与行人的相似度显著更低；
    /// 3. 图库检索：两人注册后以行人 A 查询，top-1 应命中 A。
    ///
    /// 运行：`cargo test --lib reid -- --ignored --nocapture`
    #[test]
    fn osnet_same_person_similar_different_region_less() {
        if !["testmodels/osnet_x1_0.onnx", "models/reid/osnet_x1_0.onnx", "testmodels/bus.jpg", "models/test_images/bus.jpg"].iter().any(|p| std::path::Path::new(p).exists()) {
            eprintln!("[SKIP] 模型/测试资产缺失（models/ 或 testmodels/）");
            return;
        }

        let dir = std::env::var("TESTMODELS_DIR").unwrap_or_else(|_| "testmodels".into());
        let image = Image::load(format!("{dir}/bus.jpg")).unwrap();
        let eng = ReidEngine::new(format!("{dir}/osnet_x1_0.onnx"), DeviceType::Cpu).unwrap();
        let (iw, ih) = (image.width(), image.height());

        // bus.jpg 分数坐标 crop（相对宽高），保证不同分辨率图片同样适用
        let frac = |fx: f32, fy: f32, fw: f32, fh: f32| -> Rect {
            Rect::new(
                (fx * iw as f32) as i32,
                (fy * ih as f32) as i32,
                (fw * iw as f32) as i32,
                (fh * ih as f32) as i32,
            )
        };
        let person_a = frac(0.36, 0.20, 0.20, 0.60); // 中心行人（完整框）
        let person_a_tight = frac(0.395, 0.27, 0.13, 0.46); // 同一行人（收紧框）
        let person_b = frac(0.0, 0.45, 0.17, 0.55); // 左下角行人
        let bus_body = frac(0.62, 0.05, 0.34, 0.50); // 非行人区域（巴士车身）

        // 把 crop 存到临时目录，便于肉眼核对框的位置
        let tmp = std::env::temp_dir();
        for (name, rect) in [
            ("reid_person_a", person_a),
            ("reid_person_a_tight", person_a_tight),
            ("reid_person_b", person_b),
            ("reid_bus_body", bus_body),
        ] {
            let (x, y, w, h) = clamp_roi(&image, &rect).unwrap();
            image
                .crop(x, y, w, h)
                .unwrap()
                .save(tmp.join(format!("{name}.png")))
                .unwrap();
        }

        let emb_a = eng.embed(&image, person_a).unwrap();
        let emb_a2 = eng.embed(&image, person_a_tight).unwrap();
        let emb_b = eng.embed(&image, person_b).unwrap();
        let emb_bus = eng.embed(&image, bus_body).unwrap();
        assert_eq!(emb_a.len(), 512, "OSNet x1.0 输出应为 512 维");
        // L2 归一化检查
        let norm: f32 = emb_a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "embedding 模长应为 1，实际 {norm}");

        let sim_same = ReidEngine::cosine_similarity(&emb_a, &emb_a2);
        let sim_a_b = ReidEngine::cosine_similarity(&emb_a, &emb_b);
        let sim_a_bus = ReidEngine::cosine_similarity(&emb_a, &emb_bus);
        let sim_b_bus = ReidEngine::cosine_similarity(&emb_b, &emb_bus);
        eprintln!("OSNet x1.0 余弦相似度（bus.jpg）：");
        eprintln!("  同一人（完整框 vs 收紧框）        : {sim_same:.4}");
        eprintln!("  行人A vs 行人B（不同人）          : {sim_a_b:.4}");
        eprintln!("  行人A vs 巴士车身（非行人区域）   : {sim_a_bus:.4}");
        eprintln!("  行人B vs 巴士车身（非行人区域）   : {sim_b_bus:.4}");

        assert!(sim_same > 0.5, "同一人两个 crop 相似度应 > 0.5，实际 {sim_same:.4}");
        assert!(
            sim_same > sim_a_bus && sim_same > sim_b_bus,
            "同人相似度应高于与非行人区域的相似度（same={sim_same:.4}, bus_a={sim_a_bus:.4}, bus_b={sim_b_bus:.4}）"
        );

        // 图库：注册两人（各两条特征），行人 A 特征查询 → top-1 = A，且 A 与 B 的分数有区分
        let mut gallery = ReidGallery::new();
        gallery.add(101, emb_a.clone());
        gallery.add(101, emb_a2);
        gallery.add(202, emb_b);
        let hits = gallery.search(&emb_a, 2);
        eprintln!(
            "图库检索（查询=行人A）：{:?}",
            hits.iter().map(|(id, s)| (*id, format!("{s:.4}"))).collect::<Vec<_>>()
        );
        assert_eq!(hits[0].0, 101, "top-1 应命中行人 A 的 id");
        assert!(hits[0].1 > hits[1].1, "top-1 相似度应高于第二名");
    }
}
