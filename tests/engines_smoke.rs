//! 全引擎真实模型冒烟测试。
//!
//! 模型文件默认取 `/Volumes/macEx/AI/vision-commons/models`，可用环境变量覆盖：
//! ```text
//! MODEL_DIR=/path/to/models cargo test --test engines_smoke -- --ignored --test-threads=1 --nocapture
//! ```
//! 所有用例标记 `#[ignore]`：默认 `cargo test` 不依赖模型文件。

use std::path::PathBuf;
use std::sync::Arc;

use rust_onnx_infer::core::factory::create_yoloe_engine;
use rust_onnx_infer::core::{AsyncBatchOptimizer, DeviceType};
use rust_onnx_infer::engines::birefnet::BiRefNetEngine;
use rust_onnx_infer::engines::classification::ClassificationEngine;
use rust_onnx_infer::engines::dedode_g::DedodeGEngine;
use rust_onnx_infer::engines::detection::DetectionEngine;
use rust_onnx_infer::engines::grounded_sam::GroundedSamEngine;
use rust_onnx_infer::engines::grounding_dino::GroundingDinoEngine;
use rust_onnx_infer::engines::image_enhance::{ImageEnhanceEngine, ImageEnhanceType};
use rust_onnx_infer::engines::lightglue::LightGlueEngine;
use rust_onnx_infer::engines::lomar::LoMaREngine;
use rust_onnx_infer::engines::real_esrgan::RealEsrganEngine;
use rust_onnx_infer::engines::roma_v2::RomaV2Engine;
use rust_onnx_infer::engines::sam::SamEngine;
use rust_onnx_infer::engines::sam2::Sam2Engine;
use rust_onnx_infer::engines::segmentation::SegmentationEngine;
use rust_onnx_infer::core::OnnxInferenceEngine;
use rust_onnx_infer::imaging::FloatMask;
use rust_onnx_infer::sahi::SahiConfig;
use rust_onnx_infer::{Image, Result};

fn model_dir() -> PathBuf {
    std::env::var("MODEL_DIR")
        .unwrap_or_else(|_| "models".to_string())
        .into()
}

/// 在 models/ 树中按文件名查找（兼容按引擎子目录布局）。
fn find_model_any(name: &str) -> PathBuf {
    let direct = model_dir().join(name);
    if direct.exists() {
        return direct;
    }
    if let Ok(entries) = walkdir_models(&model_dir()) {
        for p in entries {
            if p.file_name().map(|n| n == name).unwrap_or(false) {
                return p;
            }
        }
    }
    direct
}

fn walkdir_models(root: &std::path::Path) -> std::io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d)? {
            let e = e?;
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                out.push(p);
            }
        }
    }
    Ok(out)
}

fn test_image() -> Result<Image> {
    // 杯子.jpg：已在分类/检测示例中验证过的测试图
    Image::load(find_model_any("test_images/cup.jpg"))
}

fn feature_pair() -> (PathBuf, PathBuf) {
    (find_model_any("box.png"), find_model_any("box_in_scene.png"))
}

fn cpu() -> DeviceType {
    DeviceType::Cpu
}

macro_rules! smoke {
    ($name:ident, $desc:expr, $body:block) => {
        #[test]
        fn $name() -> rust_onnx_infer::Result<()> {
            let strict = std::env::var("MODEL_TESTS").ok().as_deref() == Some("1");
            let result = (|| -> rust_onnx_infer::Result<()> {
                eprintln!("--- {} ---", $desc);
                $body;
                eprintln!("[PASS] {}", $desc);
                Ok(())
            })();
            match result {
                Ok(()) => Ok(()),
                Err(e) => {
                    let missing_asset = matches!(
                        &e,
                        rust_onnx_infer::VisionError::ModelNotFound(_)
                            | rust_onnx_infer::VisionError::ImageDecode(_)
                            | rust_onnx_infer::VisionError::Io(_)
                    );
                    if missing_asset && !strict {
                        eprintln!("[SKIP] {} —— 模型/资产缺失: {e}", $desc);
                        Ok(())
                    } else {
                        Err(e)
                    }
                }
            }
        }
    };
}

// ==================== 分类 ====================

smoke!(classification_convnext, "分类引擎 convnext_tiny（for_auto）", {
    let engine = ClassificationEngine::for_auto(find_model_any("convnext_tiny.onnx"), cpu())?;
    let image = test_image()?;
    let r = engine.predict(&image)?;
    match &r {
        Some(r) => {
            let n = r.all_scores.as_ref().map(|s| s.len()).unwrap_or(0);
            assert!(n > 0, "all_scores 为空");
            eprintln!("top1={} conf={:.4} scores={}", r.class_name, r.confidence, n);
        }
        None => eprintln!("低于阈值，返回 None（合法）"),
    }
});

// ==================== 检测（4 种布局） ====================

smoke!(detection_traditional_yolov8, "检测引擎 yolov8n（传统 NMS 布局）", {
    let engine = DetectionEngine::for_yolo(find_model_any("yolov8n.onnx"), cpu(), 0.25, 0.45)?;
    let dets = engine.predict(&test_image()?)?;
    eprintln!("检测到 {} 个目标", dets.len());
    for d in dets.iter().take(3) {
        eprintln!("  {d}");
    }
    assert!(!dets.is_empty(), "yolov8n 在杯子上应有检出");
});

smoke!(detection_end2end_yolo26, "检测引擎 yolo26n（End2End 布局）", {
    let engine = DetectionEngine::for_yolo(find_model_any("yolo26n.onnx"), cpu(), 0.25, 0.45)?;
    let dets = engine.predict(&test_image()?)?;
    eprintln!("检测到 {} 个目标", dets.len());
    assert!(!dets.is_empty(), "yolo26n 在杯子上应有检出");
});

smoke!(detection_rtdetr, "检测引擎 rtdetr-l（Transformer）", {
    let engine = DetectionEngine::for_rtdetr(find_model_any("rtdetr-l.onnx"), cpu(), 0.5, 0.45)?;
    let dets = engine.predict(&test_image()?)?;
    eprintln!("检测到 {} 个目标", dets.len());
});

smoke!(detection_rfdetr, "检测引擎 rfdetr_base（Roboflow）", {
    let engine = DetectionEngine::for_rfdetr(find_model_any("rfdetr_base.onnx"), cpu(), 0.5)?;
    let dets = engine.predict(&test_image()?)?;
    eprintln!("检测到 {} 个目标", dets.len());
});

// ==================== 分割（3 种布局） ====================

smoke!(segmentation_traditional_yolov8seg, "分割引擎 yolov8n-seg（传统布局 + 掩码解码）", {
    let engine = SegmentationEngine::for_yolo(find_model_any("yolov8n-seg.onnx"), cpu(), 0.25, 0.45)?;
    let segs = engine.predict(&test_image()?)?;
    eprintln!("分割出 {} 个实例", segs.len());
    for s in segs.iter().take(2) {
        eprintln!("  {s} area={:.0}", s.mask_area());
    }
    assert!(!segs.is_empty(), "yolov8n-seg 应有检出");
    assert!(segs[0].mask.is_some(), "传统布局应产出掩码");
});

smoke!(segmentation_end2end_yolo26seg, "分割引擎 yolo26n-seg（End2End 布局）", {
    let engine = SegmentationEngine::for_yolo(find_model_any("yolo26n-seg.onnx"), cpu(), 0.25, 0.45)?;
    let segs = engine.predict(&test_image()?)?;
    eprintln!("分割出 {} 个实例", segs.len());
    assert!(!segs.is_empty(), "yolo26n-seg 应有检出");
});

smoke!(segmentation_rfdetr_seg, "分割引擎 rfdetr_seg_small（RF-DETR-Seg）", {
    let engine = SegmentationEngine::for_rfdetr_seg(find_model_any("rfdetr_seg_small.onnx"), cpu(), 0.5)?;
    let segs = engine.predict(&test_image()?)?;
    eprintln!("分割出 {} 个实例", segs.len());
});

// ==================== YOLOE（烘焙 + 运行时视觉提示） ====================

smoke!(yoloe_baked_text, "YOLOE 烘焙文本提示（yoloe_11s_text，person/bus 提示配 bus 图）", {
    let engine = create_yoloe_engine(find_model_any("yoloe_11s_text.onnx"), cpu(), 0.25)?;
    // 烘焙类别为 person/bus，用内容匹配的 bus.jpg（传统布局 [1,4+nc+32,8400]）
    let image = Image::load("testmodels/bus.jpg")?;
    let segs = engine.predict(&image)?;
    eprintln!("分割出 {} 个实例", segs.len());
    for s in segs.iter().take(3) {
        eprintln!("  {s}");
    }
    assert!(!segs.is_empty(), "bus.jpg 上烘焙提示应有检出");
});

smoke!(yoloe26_text_baked, "YOLOE-26 烘焙文本提示（End2End 逐类分数布局，person/bus）", {
    let engine = create_yoloe_engine(find_model_any("yoloe_26n_text.onnx"), cpu(), 0.25)?;
    let image = Image::load("testmodels/bus.jpg")?;
    let segs = engine.predict(&image)?;
    eprintln!("YOLOE-26 分割出 {} 个实例", segs.len());
    for s in segs.iter().take(6) {
        eprintln!("  {s}");
    }
    assert!(!segs.is_empty(), "bus.jpg 上 person/bus 类应有检出");
    assert!(
        segs.iter().any(|s| s.mask.is_some()),
        "YOLOE-26 End2End 应产出实例掩码"
    );
});

smoke!(yoloe_runtime_visual_prompts, "YOLOE 运行时视觉提示（encoder + pe detector）", {
    let engine = rust_onnx_infer::engines::yolo_e_runtime::YoloERuntimeEngine::new(
        find_model_any("yoloe_rt_encoder.onnx"),
        find_model_any("yoloe_rt_pe_detector.onnx"),
        cpu(),
    )?;
    let image = test_image()?;
    // 用检测已验证的杯子区域作为视觉提示框
    engine.set_visual_prompts(
        &image,
        vec![("cup".to_string(), vec![vec![145.0f32, 169.0, 473.0, 450.0]])],
    )?;
    let segs = engine.predict_without_sahi(&image)?;
    eprintln!("分割出 {} 个实例", segs.len());
    for s in segs.iter().take(3) {
        eprintln!("  {s}");
    }
});

// ==================== SAM / SAM2 ====================

smoke!(sam_point_and_box, "SAM 引擎 mobile_sam（点 + 框提示）", {
    let engine = SamEngine::new(find_model_any("mobile_sam.onnx"), cpu())?;
    let image = test_image()?;
    let by_box = engine.predict_box(&image, 145.0, 169.0, 473.0, 450.0)?;
    eprintln!("框提示 → {} 个掩码", by_box.len());
    assert!(!by_box.is_empty(), "SAM 框提示应有输出");
    let m = by_box[0].binary_mask()?;
    assert!(m.data().iter().any(|&v| v > 0), "掩码应有前景");
    let by_point = engine.predict_point(&image, 300.0, 300.0, 1)?;
    eprintln!("点提示 → {} 个掩码", by_point.len());
    assert!(!by_point.is_empty(), "SAM 点提示应有输出");
});

smoke!(sam2_box_prompt, "SAM2 引擎 sam2.1-tiny fused（框提示）", {
    let engine = Sam2Engine::new(find_model_any("sam2_fused.onnx"), cpu())?;
    let segs = engine.predict_box(&test_image()?, 145.0, 169.0, 473.0, 450.0)?;
    eprintln!("框提示 → {} 个掩码", segs.len());
    assert!(!segs.is_empty(), "SAM2 框提示应有输出");
    let m = segs[0].binary_mask()?;
    assert!(m.data().iter().any(|&v| v > 0), "掩码应有前景");
});

// ==================== 开放词表 ====================

smoke!(grounding_dino_text, "GroundingDINO 文本提示（tiny + tokenizer）", {
    let dir = find_model_any("grounding_dino_tiny.onnx").parent().map(|p| p.to_path_buf()).unwrap_or_else(model_dir);
    let engine = GroundingDinoEngine::new(
        dir.join("grounding_dino_tiny.onnx"),
        dir.join("tokenizer.json"),
        cpu(),
    )?;
    let dets = engine.predict_text(&test_image()?, "cup")?;
    eprintln!("'cup' → {} 个框", dets.len());
    for d in dets.iter().take(3) {
        eprintln!("  {d}");
    }
    assert!(!dets.is_empty(), "GroundingDINO 应检出 cup");
});

smoke!(grounded_sam_pipeline, "Grounded-SAM 流水线（DINO + SAM2）", {
    let dir = find_model_any("grounding_dino_tiny.onnx").parent().map(|p| p.to_path_buf()).unwrap_or_else(model_dir);
    let engine = GroundedSamEngine::new(
        dir.join("grounding_dino_tiny.onnx"),
        dir.join("tokenizer.json"),
        find_model_any("sam2_fused.onnx"),
        cpu(),
    )?;
    let segs = engine.predict_text(&test_image()?, "cup")?;
    eprintln!("'cup' → {} 个实例掩码", segs.len());
    assert!(!segs.is_empty(), "Grounded-SAM 应有输出");
    assert!(segs[0].mask.is_some(), "应有掩码");
});

// ==================== 抠图 / 超分 / 增强 ====================

smoke!(birefnet_matting, "BiRefNet 抠图（soft alpha + 二值掩码）", {
    let engine = BiRefNetEngine::new(find_model_any("birefnet_onnx_community.onnx"), cpu())?;
    let image = test_image()?;
    let result = engine.predict_impl(&image)?;
    eprintln!(
        "alpha={}x{}, 原图={}x{}, 耗时={}ms",
        result.alpha.width(),
        result.alpha.height(),
        result.original_width,
        result.original_height,
        result.elapsed_ms
    );
    assert_eq!(result.alpha.width() as i32, result.original_width);
    let mask = result.binary_mask(0.5);
    assert!(mask.data().iter().any(|&v| v > 0), "二值掩码应有前景");
    let _ = engine.predict_binary_mask(&image)?;
});

smoke!(real_esrgan_x2, "Real-ESRGAN 2x 超分（重叠分块）", {
    let engine = RealEsrganEngine::new(find_model_any("realesrgan_x2.onnx"), 2, cpu())?;
    let crop = test_image()?.crop(0, 0, 128, 128)?;
    let out = engine.predict_impl(&crop)?;
    eprintln!("输出 {}x{}（输入 128x128）", out.width(), out.height());
    assert_eq!(out.width(), 256, "2x 超分尺寸应为 256");
});

smoke!(image_enhance_denoise_lowlight_dehaze, "图像增强三件套（去噪/低光/去雾）", {
    let crop = test_image()?.crop(0, 0, 256, 256)?;
    let denoise = ImageEnhanceEngine::new(find_model_any("dncnn_color_blind.onnx"), ImageEnhanceType::Denoise, cpu())?;
    let out = denoise.predict_impl(&crop)?;
    assert_eq!((out.width(), out.height()), (256, 256), "去噪输出应同尺寸");
    eprintln!("去噪 OK");

    let low = ImageEnhanceEngine::new(find_model_any("zerodce.onnx"), ImageEnhanceType::LowLight, cpu())?;
    let out = low.predict_impl(&crop)?;
    assert_eq!((out.width(), out.height()), (256, 256), "低光输出应同尺寸");
    eprintln!("低光 OK");

    let dehaze = ImageEnhanceEngine::new(find_model_any("dehazeformer_s_outdoor.onnx"), ImageEnhanceType::Dehaze, cpu())?;
    let out = dehaze.predict_impl(&crop)?;
    assert_eq!((out.width(), out.height()), (256, 256), "去雾输出应同尺寸");
    eprintln!("去雾 OK");

    let pipeline = rust_onnx_infer::engines::image_enhance_pipeline::ImageEnhancePipeline::builder()
        .denoise(denoise)?
        .build()?;
    let out = pipeline.process(&crop)?;
    assert_eq!(out.width(), 256);
    eprintln!("流水线 OK");
});

// ==================== 特征匹配 ====================

smoke!(lightglue_match, "LightGlue 匹配（superpoint_lightglue pipeline）", {
    let (a, b) = feature_pair();
    let engine = LightGlueEngine::with_input_size(
        model_dir().join("lightglue/superpoint_lightglue_pipeline.onnx").to_string_lossy().to_string(),
        cpu(),
        0.3,
        640,
        640,
    )?;
    let result = engine.match_images(&Image::load(a)?, &Image::load(b)?)?;
    eprintln!(
        "kpts0={} kpts1={} matches={}",
        result.keypoints0.len(),
        result.keypoints1.len(),
        result.match_count()
    );
    assert!(result.match_count() > 10, "box 对应有大量匹配");
});

smoke!(dedode_detect, "DeDoDe-G 检测器 + 描述器", {
    let engine = DedodeGEngine::new(
        find_model_any("dedode_g_detector.onnx"),
        find_model_any("dedode_g_descriptor.onnx"),
        cpu(),
    )?;
    let d = engine.detect(&Image::load(feature_pair().0)?)?;
    eprintln!("关键点 {} 个", d.keypoints.len());
    assert!(d.keypoints.len() >= 100, "关键点数量应充足");
    assert_eq!(d.descriptors.len(), d.keypoints.len(), "描述子与关键点一一对应");
});

smoke!(lomar_match, "LoMa-R 匹配（LoMa-R 主干 + 共享 DeDoDe 检测器）", {
    let detector = Arc::new(DedodeGEngine::new(
        find_model_any("dedode_g_detector.onnx"),
        find_model_any("dedode_g_descriptor.onnx"),
        cpu(),
    )?);
    let engine = LoMaREngine::new(find_model_any("loma_R.onnx"), Arc::clone(&detector), cpu())?;
    let (a, b) = feature_pair();
    let result = engine.match_images(&Image::load(a)?, &Image::load(b)?)?;
    eprintln!("matches={} filter={}", result.match_count(), result.filter_threshold);
    assert!(result.match_count() > 5, "box 对应有匹配");
});

smoke!(romav2_match, "RoMaV2 密集匹配（warp/overlap + 稀疏采样）", {
    let engine = RomaV2Engine::new(find_model_any("romav2_base.onnx"), cpu())?;
    let (a, b) = feature_pair();
    let result = engine.match_images(&Image::load(a)?, &Image::load(b)?)?;
    eprintln!(
        "dense={}x{} sampled={} dense_conf(0.3)={}",
        result.dense_width,
        result.dense_height,
        result.match_count(),
        result.dense_count(0.3)
    );
    assert!(result.match_count() > 0, "RoMaV2 应有采样匹配");
});

// ==================== SAHI / 批量优化器 / 工具 ====================

smoke!(sahi_sliced_detection, "SAHI 切片推理（检测引擎 + GREEDYNMM 合并）", {
    let mut engine = DetectionEngine::for_yolo(find_model_any("yolov8n.onnx"), cpu(), 0.25, 0.45)?;
    engine.base.set_sahi_config(Some(SahiConfig::of(512, 512, 0.2, 0.2)));
    let dets = engine.predict(&test_image()?)?;
    eprintln!("SAHI 检出 {} 个目标", dets.len());
    engine.base.disable_sahi();
    let dets2 = engine.predict(&test_image()?)?;
    eprintln!("关闭 SAHI 后检出 {} 个目标", dets2.len());
});

smoke!(async_batch_optimizer, "异步批处理优化器（聚合 + 超时触发）", {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let engine = Arc::new(DetectionEngine::for_yolo(
            find_model_any("yolov8n.onnx"),
            cpu(),
            0.25,
            0.45,
        )?);
        let optimizer = AsyncBatchOptimizer::new(engine);
        let img = test_image()?;
        let rx1 = optimizer.submit_async(img.clone()).await?;
        let rx2 = optimizer.submit_async(img).await?;
        let r1 = rx1.await.unwrap()?;
        let r2 = rx2.await.unwrap()?;
        eprintln!("批量结果：{} / {} 个目标", r1.len(), r2.len());
        Ok::<_, rust_onnx_infer::VisionError>(())
    })?;
});

smoke!(matting_utils_composite, "MattingUtils（alpha 合成 + 棋盘格）", {
    use rust_onnx_infer::util::matting_utils;
    let bgr = test_image()?;
    let mut alpha = FloatMask::new(bgr.width(), bgr.height());
    // 半平面渐变 alpha，验证合成数学
    for y in 0..alpha.height() {
        for x in 0..alpha.width() {
            alpha.set(x, y, (x as f32 / alpha.width() as f32).min(1.0));
        }
    }
    let cutout = matting_utils::cutout_bgra(&bgr, &alpha)?;
    assert_eq!(cutout.channels(), 4, "cutout 应为 BGRA");
    let checker = matting_utils::composite_on_checkerboard(&bgr, &alpha, 16)?;
    assert_eq!(checker.channels(), 3);
    let gray = matting_utils::alpha_to_gray_u8(&alpha);
    assert_eq!(gray.width(), bgr.width());
    eprintln!("cutout/checkerboard/gray 全部 OK");
});
