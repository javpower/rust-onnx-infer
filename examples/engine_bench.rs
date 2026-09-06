//! 逐引擎推理速度基准（CPU / 单线程调用）。
//!
//! 运行：`cargo run --release --example engine_bench`
//! 每项测量：模型加载耗时、首帧（含预热路径）、稳态 N 次平均。

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rust_onnx_infer::core::OnnxInferenceEngine;
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
use rust_onnx_infer::engines::yolo_e_runtime::YoloERuntimeEngine;
use rust_onnx_infer::sahi::SahiConfig;
use rust_onnx_infer::{DeviceType, Image};

fn model_dir() -> PathBuf {
    std::env::var("MODEL_DIR")
        .unwrap_or_else(|_| "/Volumes/macEx/AI/vision-commons/models".to_string())
        .into()
}

fn cup() -> Image {
    Image::load("/Users/xiongguochao/Desktop/杯子.jpg").expect("test image")
}

/// 计时：首帧 + 稳态均值（warm 次调用取平均）
fn steady<F: FnMut()>(mut f: F, warm: usize) -> (f64, f64) {
    let t = Instant::now();
    f();
    let cold = t.elapsed().as_secs_f64() * 1000.0;
    let t = Instant::now();
    for _ in 0..warm {
        f();
    }
    let avg = t.elapsed().as_secs_f64() * 1000.0 / warm as f64;
    (cold, avg)
}

fn print_row(name: &str, load: Duration, cold: f64, avg: f64, note: &str) {
    println!(
        "{:<34} 加载 {:>7.0}ms | 首帧 {:>8.1}ms | 稳态 {:>8.1}ms | {}",
        name,
        load.as_secs_f64() * 1000.0,
        cold,
        avg,
        note
    );
}

fn main() {
    let dir = model_dir();
    let image = cup();
    let warm3 = 3;

    println!("设备 = CPU（无 GPU EP），测试图 558x550");
    println!(
        "{:<34} {:>10} | {:>10} | {:>10} | 备注",
        "引擎", "加载", "首帧", "稳态均值"
    );
    println!("{}", "-".repeat(110));

    // ---------- 分类 ----------
    let t = Instant::now();
    let engine = ClassificationEngine::for_auto(dir.join("convnext_tiny.onnx"), DeviceType::Cpu).unwrap();
    let load = t.elapsed();
    let (cold, avg) = steady(|| {
        let _ = engine.predict(&image).unwrap();
    }, warm3);
    print_row("分类 convnext_tiny", load, cold, avg, "");

    // ---------- 检测 ----------
    macro_rules! bench_detect {
        ($name:expr, $model:expr, $ctor:expr) => {
            let t = Instant::now();
            let engine = $ctor;
            let load = t.elapsed();
            let (cold, avg) = steady(|| {
                let _ = engine.predict(&image).unwrap();
            }, warm3);
            print_row($name, load, cold, avg, "");
        };
    }
    bench_detect!(
        "检测 yolov8n（传统+NMS）",
        dir.join("yolov8n.onnx"),
        DetectionEngine::for_yolo(dir.join("yolov8n.onnx"), DeviceType::Cpu, 0.25, 0.45).unwrap()
    );
    bench_detect!(
        "检测 yolo26n（End2End）",
        dir.join("yolo26n.onnx"),
        DetectionEngine::for_yolo(dir.join("yolo26n.onnx"), DeviceType::Cpu, 0.25, 0.45).unwrap()
    );
    bench_detect!(
        "检测 rtdetr-l（Transformer）",
        dir.join("rtdetr-l.onnx"),
        DetectionEngine::for_rtdetr(dir.join("rtdetr-l.onnx"), DeviceType::Cpu, 0.5, 0.45).unwrap()
    );
    bench_detect!(
        "检测 rfdetr_base",
        dir.join("rfdetr_base.onnx"),
        DetectionEngine::for_rfdetr(dir.join("rfdetr_base.onnx"), DeviceType::Cpu, 0.5).unwrap()
    );

    // ---------- 分割 ----------
    bench_detect!(
        "分割 yolov8n-seg（传统+掩码）",
        dir.join("yolov8n-seg.onnx"),
        SegmentationEngine::for_yolo(dir.join("yolov8n-seg.onnx"), DeviceType::Cpu, 0.25, 0.45).unwrap()
    );
    bench_detect!(
        "分割 yolo26n-seg（End2End）",
        dir.join("yolo26n-seg.onnx"),
        SegmentationEngine::for_yolo(dir.join("yolo26n-seg.onnx"), DeviceType::Cpu, 0.25, 0.45).unwrap()
    );
    bench_detect!(
        "分割 rfdetr_seg_small",
        dir.join("rfdetr_seg_small.onnx"),
        SegmentationEngine::for_rfdetr_seg(dir.join("rfdetr_seg_small.onnx"), DeviceType::Cpu, 0.5).unwrap()
    );

    // ---------- YOLOE ----------
    let t = Instant::now();
    let engine = create_yoloe(dir.join("yoloe_11s_text.onnx"));
    let load = t.elapsed();
    let (cold, avg) = steady(|| {
        let _ = engine.predict(&image).unwrap();
    }, warm3);
    print_row("YOLOE 11s（烘焙文本提示）", load, cold, avg, "");

    {
        let t = Instant::now();
        let engine = YoloERuntimeEngine::new(
            dir.join("yoloe_rt_encoder.onnx"),
            dir.join("yoloe_rt_pe_detector.onnx"),
            DeviceType::Cpu,
        )
        .unwrap();
        let load = t.elapsed();
        engine
            .set_visual_prompts(
                &image,
                vec![("cup".to_string(), vec![vec![145.0f32, 169.0, 473.0, 450.0]])],
            )
            .unwrap();
        let (cold, avg) = steady(|| {
            let _ = engine.predict_without_sahi(&image).unwrap();
        }, warm3);
        print_row("YOLOE-Runtime（视觉提示，不含编码）", load, cold, avg, "提示编码另计一次");
    }

    // ---------- SAM / SAM2 ----------
    let t = Instant::now();
    let sam = SamEngine::new(dir.join("mobile_sam.onnx"), DeviceType::Cpu).unwrap();
    let load = t.elapsed();
    let (cold, avg) = steady(|| {
        let _ = sam.predict_box(&image, 145.0, 169.0, 473.0, 450.0).unwrap();
    }, warm3);
    print_row("SAM mobile（框提示，含编码）", load, cold, avg, "");

    let t = Instant::now();
    let sam2 = Sam2Engine::new(dir.join("sam2_1_hiera_tiny/sam2_fused.onnx"), DeviceType::Cpu).unwrap();
    let load = t.elapsed();
    let (cold, avg) = steady(|| {
        let _ = sam2.predict_box(&image, 145.0, 169.0, 473.0, 450.0).unwrap();
    }, warm3);
    print_row("SAM2.1-tiny fused（框提示）", load, cold, avg, "每次重跑 encoder");

    // ---------- 开放词表 ----------
    let t = Instant::now();
    let dino = GroundingDinoEngine::new(
        dir.join("grounding_dino_tiny/grounding_dino_tiny.onnx"),
        dir.join("grounding_dino_tiny/tokenizer.json"),
        DeviceType::Cpu,
    )
    .unwrap();
    let load = t.elapsed();
    let (cold, avg) = steady(|| {
        let _ = dino.predict_text(&image, "cup").unwrap();
    }, warm3);
    print_row("GroundingDINO-tiny（文本提示）", load, cold, avg, "含分词");

    {
        let t = Instant::now();
        let gs = GroundedSamEngine::new(
            dir.join("grounding_dino_tiny/grounding_dino_tiny.onnx"),
            dir.join("grounding_dino_tiny/tokenizer.json"),
            dir.join("sam2_1_hiera_tiny/sam2_fused.onnx"),
            DeviceType::Cpu,
        )
        .unwrap();
        let load = t.elapsed();
        let (cold, avg) = steady(|| {
            let _ = gs.predict_text(&image, "cup").unwrap();
        }, warm3);
        print_row("Grounded-SAM（DINO+SAM2 流水线）", load, cold, avg, "");
    }

    // ---------- 抠图 / 超分 / 增强 ----------
    let t = Instant::now();
    let birefnet = BiRefNetEngine::new(dir.join("birefnet_onnx_community.onnx"), DeviceType::Cpu).unwrap();
    let load = t.elapsed();
    let (cold, avg) = steady(|| {
        let _ = birefnet.predict_impl(&image).unwrap();
    }, 2);
    print_row("BiRefNet 抠图（1024 输入）", load, cold, avg, "");

    let crop128 = image.crop(0, 0, 128, 128).unwrap();
    let t = Instant::now();
    let esrgan = RealEsrganEngine::new(dir.join("realesrgan_x2.onnx"), 2, DeviceType::Cpu).unwrap();
    let load = t.elapsed();
    let (cold, avg) = steady(|| {
        let _ = esrgan.predict_impl(&crop128).unwrap();
    }, warm3);
    print_row("Real-ESRGAN x2（128→256）", load, cold, avg, "");

    let crop256 = image.crop(0, 0, 256, 256).unwrap();
    macro_rules! bench_enhance {
        ($name:expr, $model:expr, $ty:expr) => {
            let t = Instant::now();
            let engine = ImageEnhanceEngine::new(dir.join($model), $ty, DeviceType::Cpu).unwrap();
            let load = t.elapsed();
            let (cold, avg) = steady(|| {
                let _ = engine.predict_impl(&crop256).unwrap();
            }, warm3);
            print_row($name, load, cold, avg, "");
        };
    }
    bench_enhance!("增强 去噪 DnCNN（256）", "dncnn_color_blind.onnx", ImageEnhanceType::Denoise);
    bench_enhance!("增强 低光 Zero-DCE（256）", "zerodce.onnx", ImageEnhanceType::LowLight);
    bench_enhance!("增强 去雾 DehazeFormer（256）", "dehazeformer_s_outdoor.onnx", ImageEnhanceType::Dehaze);

    // ---------- 特征匹配 ----------
    let pair_a = dir.join("loma_assets/box.png");
    let pair_b = dir.join("loma_assets/box_in_scene.png");
    let img_a = Image::load(&pair_a).unwrap();
    let img_b = Image::load(&pair_b).unwrap();

    let t = Instant::now();
    let lightglue = LightGlueEngine::with_input_size(
        "/Volumes/macEx/Downloads/superpoint_lightglue_pipeline.onnx",
        DeviceType::Cpu,
        0.3,
        640,
        640,
    )
    .unwrap();
    let load = t.elapsed();
    let (cold, avg) = steady(|| {
        let _ = lightglue.match_images(&img_a, &img_b).unwrap();
    }, warm3);
    print_row("LightGlue（双图 640）", load, cold, avg, "含双图特征提取");

    let t = Instant::now();
    let dedode = Arc::new(DedodeGEngine::new(
        dir.join("dedode_g_detector.onnx"),
        dir.join("dedode_g_descriptor.onnx"),
        DeviceType::Cpu,
    )
    .unwrap());
    let load = t.elapsed();
    let (cold, avg) = steady(|| {
        let _ = dedode.detect(&img_a).unwrap();
    }, warm3);
    print_row("DeDoDe-G 检测+描述（单图 784）", load, cold, avg, "");

    {
        let t = Instant::now();
        let lomar = LoMaREngine::new(dir.join("loma_R.onnx"), dedode.clone(), DeviceType::Cpu).unwrap();
        let load = t.elapsed();
        let (cold, avg) = steady(|| {
            let _ = lomar.match_images(&img_a, &img_b).unwrap();
        }, 2);
        print_row("LoMa-R 匹配（双图，含检测）", load, cold, avg, "检测器可复用缓存");
    }

    let t = Instant::now();
    let roma = RomaV2Engine::new(dir.join("romav2_base.onnx"), DeviceType::Cpu).unwrap();
    let load = t.elapsed();
    let (cold, avg) = steady(|| {
        let _ = roma.match_images(&img_a, &img_b).unwrap();
    }, 2);
    print_row("RoMaV2 密集匹配（双图 640）", load, cold, avg, "");

    // ---------- SAHI ----------
    {
        let t = Instant::now();
        let mut engine =
            DetectionEngine::for_yolo(dir.join("yolov8n.onnx"), DeviceType::Cpu, 0.25, 0.45).unwrap();
        engine
            .base
            .set_sahi_config(Some(SahiConfig::of(512, 512, 0.2, 0.2)));
        let load = t.elapsed();
        let (cold, avg) = steady(|| {
            let _ = engine.predict(&image).unwrap();
        }, 2);
        print_row("SAHI 切片检测（yolov8n + 512 切片）", load, cold, avg, "含整图标准预测");
    }

    let _ = create_yoloe; // silence unused if branch
}

fn create_yoloe(p: PathBuf) -> rust_onnx_infer::engines::segmentation::SegmentationEngine {
    rust_onnx_infer::core::factory::create_yoloe_engine(p, DeviceType::Cpu, 0.25).unwrap()
}
