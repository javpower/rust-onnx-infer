//! 优化实验基准：线程数 / CoreML EP / LoMa-R 模板缓存。
//!
//! 运行：`cargo run --release --example optimize_bench`

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rust_onnx_infer::core::runtime_config::OnnxRuntimeConfig;
use rust_onnx_infer::engines::birefnet::BiRefNetEngine;
use rust_onnx_infer::engines::dedode_g::DedodeGEngine;
use rust_onnx_infer::engines::detection::DetectionEngine;
use rust_onnx_infer::engines::grounding_dino::GroundingDinoEngine;
use rust_onnx_infer::engines::lomar::LoMaREngine;
use rust_onnx_infer::core::OnnxInferenceEngine;
use rust_onnx_infer::{DeviceType, Image};

fn model_dir() -> PathBuf {
    std::env::var("MODEL_DIR")
        .unwrap_or_else(|_| "/Volumes/macEx/AI/vision-commons/models".to_string())
        .into()
}

fn cup() -> Image {
    Image::load("/Users/xiongguochao/Desktop/杯子.jpg").expect("test image")
}

fn steady<F: FnMut()>(mut f: F, warm: usize) -> f64 {
    f(); // 首帧（预热/EP 编译），不计入稳态
    let t = Instant::now();
    for _ in 0..warm {
        f();
    }
    t.elapsed().as_secs_f64() * 1000.0 / warm as f64
}

fn row(name: &str, avg: f64) {
    println!("{:<52} 稳态 {:>9.1}ms", name, avg);
}

fn cfg(intra: usize) -> OnnxRuntimeConfig {
    OnnxRuntimeConfig::builder().intra_op_threads(intra).build()
}

fn main() {
    let dir = model_dir();
    let image = cup();

    println!("机器：8 核（4 性能核），对比 intra-op 线程与 CoreML EP");
    println!("{}", "-".repeat(84));

    // ==================== 实验 1：intra-op 线程数 ====================
    println!("\n[实验 1] intra-op 线程数（4 = 默认，8 = 全部逻辑核）");
    for model in ["yolov8n.onnx", "rtdetr-l.onnx"] {
        for intra in [4usize, 8] {
            let engine = DetectionEngine::for_yolo_with_config(
                dir.join(model),
                DeviceType::Cpu,
                0.25,
                0.45,
                cfg(intra),
            )
            .unwrap();
            let avg = steady(|| {
                let _ = engine.predict(&image).unwrap();
            }, 3);
            row(&format!("{model} intra={intra}"), avg);
        }
    }
    {
        for intra in [4usize, 8] {
            let engine =
                BiRefNetEngine::with_config(dir.join("birefnet_onnx_community.onnx"), DeviceType::Cpu, 1024, cfg(intra))
                    .unwrap();
            let avg = steady(|| {
                let _ = engine.predict_impl(&image).unwrap();
            }, 2);
            row(&format!("birefnet_onnx_community intra={intra}"), avg);
        }
    }

    // ==================== 实验 2：CoreML EP ====================
    println!("\n[实验 2] CoreML EP vs CPU（稳态；首帧含 EP 图编译另计）");
    {
        for (label, device) in [("CPU", DeviceType::Cpu), ("CoreML", DeviceType::Coreml)] {
            let engine =
                DetectionEngine::for_yolo(dir.join("yolov8n.onnx"), device, 0.25, 0.45).unwrap();
            let avg = steady(|| {
                let _ = engine.predict(&image).unwrap();
            }, 3);
            row(&format!("yolov8n @ {label}"), avg);
        }
        for (label, device) in [("CPU", DeviceType::Cpu), ("CoreML", DeviceType::Coreml)] {
            let engine =
                DetectionEngine::for_rtdetr(dir.join("rtdetr-l.onnx"), device, 0.5, 0.45).unwrap();
            let avg = steady(|| {
                let _ = engine.predict(&image).unwrap();
            }, 3);
            row(&format!("rtdetr-l @ {label}"), avg);
        }
        // birefnet @ CoreML 实测挂起（变形注意力算子不被 CoreML EP 支持），跳过
        if std::env::var("BENCH_BIREFNET_COREML").is_ok() {
            let engine =
                BiRefNetEngine::new(dir.join("birefnet_onnx_community.onnx"), DeviceType::Coreml).unwrap();
            let avg = steady(|| {
                let _ = engine.predict_impl(&image).unwrap();
            }, 2);
            row("birefnet @ CoreML", avg);
        }
        for (label, device) in [("CPU", DeviceType::Cpu), ("CoreML", DeviceType::Coreml)] {
            let engine = GroundingDinoEngine::with_config(
                dir.join("grounding_dino_tiny/grounding_dino_tiny.onnx"),
                dir.join("grounding_dino_tiny/tokenizer.json"),
                device,
                OnnxRuntimeConfig::defaults(),
            )
            .unwrap();
            let avg = steady(|| {
                let _ = engine.predict_text(&image, "cup").unwrap();
            }, 2);
            row(&format!("grounding_dino_tiny @ {label}"), avg);
        }
    }

    // ==================== 实验 3：LoMa-R 模板缓存 ====================
    println!("\n[实验 3] LoMa-R 模板缓存（同一模板图反复匹配场景图）");
    {
        let detector = Arc::new(
            DedodeGEngine::new(
                dir.join("dedode_g_detector.onnx"),
                dir.join("dedode_g_descriptor.onnx"),
                DeviceType::Cpu,
            )
            .unwrap(),
        );
        let lomar =
            LoMaREngine::new(dir.join("loma_R.onnx"), detector.clone(), DeviceType::Cpu).unwrap();
        let img_a = Image::load(dir.join("loma_assets/box.png")).unwrap();
        let img_b = Image::load(dir.join("loma_assets/box_in_scene.png")).unwrap();

        let t = Instant::now();
        let det_a = detector.detect(&img_a).unwrap();
        println!(
            "模板图检测+描述（一次，可复用）：{:.1}ms",
            t.elapsed().as_secs_f64() * 1000.0
        );

        let cold = steady(|| {
            let _ = lomar.match_images(&img_a, &img_b).unwrap();
        }, 1);
        row("每次全量 match_images（重复检测模板）", cold);

        let cached = steady(|| {
            let _ = lomar.match_images_cached(&det_a, &img_b).unwrap();
        }, 3);
        row("match_images_cached（复用模板特征）", cached);
    }

    let _ = Duration::default();
}
