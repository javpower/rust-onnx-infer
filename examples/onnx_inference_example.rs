//! ONNX 推理示例。
//!
//! 运行方式（模型与图像路径按需修改，或通过命令行参数覆盖第一个示例的路径）：
//! ```text
//! cargo run --example onnx_inference_example --features init-tracing
//! ```
//!
//! 对应原版的 Visualizer / OpenCVUtils，本示例内置了基于 imageproc 的
//! 简易可视化（画框 + 连线），仅用于示例演示。

use rust_onnx_infer::engines::classification::ClassificationEngine;
use rust_onnx_infer::core::OnnxInferenceEngine;
use rust_onnx_infer::imaging::Image;
use rust_onnx_infer::model::{ClassificationResult, Detection, LightGlueMatchResult, Segmentation};
use rust_onnx_infer::{core::builder, DeviceType, ModelType};

fn main() -> rust_onnx_infer::Result<()> {
    // 分类示例
    classification_example()?;
    // 检测示例
    detection_example()?;
    // 分割示例
    segmentation_example()?;
    // 使用工厂创建引擎
    factory_example()?;
    // 批量推理示例
    batch_inference_example()?;
    // LightGlue 特征匹配示例
    light_glue_example()?;
    Ok(())
}

fn arg_or(idx: usize, default: &str) -> String {
    std::env::args().nth(idx).unwrap_or_else(|| default.to_string())
}

/// 分类示例
fn classification_example() -> rust_onnx_infer::Result<()> {
    println!("=== Classification Example ===");

    let image_path = arg_or(1, "/Volumes/macEx/Downloads/整体/整体EKS2新/Image_20260106165021764.jpg");
    let model_path = arg_or(
        2,
        "/Users/xiongguochao/00-python-project/训练工具/runs/classify/train/weights/best.onnx",
    );

    // 加载图像
    let image = Image::load(&image_path)?;

    // 创建 YOLO 分类引擎
    let engine = ClassificationEngine::for_yolo_with_input_size(&model_path, DeviceType::Cpu, 224, 224)?;

    // 设置置信度阈值
    let mut engine = engine;
    engine.set_confidence_threshold(0.5);
    // 单图推理
    let result = engine.predict(&image)?;
    println!("Top-1: {}", result.as_ref().map(|r| r.to_string()).unwrap_or_else(|| "None".to_string()));

    // Top-K 推理
    let top5 = engine.predict_top_k(&image, 5)?;
    println!("Top-5:");
    for r in &top5 {
        println!("  {r}");
    }

    // 可视化
    if let Some(result) = &result {
        let visualized = draw_classification(&image, result)?;
        visualized.save("classification_result.jpg")?;
    }
    Ok(())
}

/// 检测示例
fn detection_example() -> rust_onnx_infer::Result<()> {
    println!("\n=== Detection Example ===");

    let image = Image::load(arg_or(3, "/Users/xiongguochao/Desktop/杯子.jpg"))?;
    let model_path = arg_or(4, "/Volumes/macEx/Downloads/yolo26l.onnx");

    // 创建检测引擎
    let engine = rust_onnx_infer::core::factory::create_detection_engine(
        &model_path,
        DeviceType::Cpu,
        0.5,
        0.45,
    )?;

    // 推理
    let detections = engine.predict(&image)?;
    println!("Detected {} objects:", detections.len());
    for det in &detections {
        println!("  {det}");
    }

    // 可视化
    let visualized = draw_detections(&image, &detections)?;
    visualized.save("detection_result.jpg")?;
    Ok(())
}

/// 分割示例
fn segmentation_example() -> rust_onnx_infer::Result<()> {
    println!("\n=== Segmentation Example ===");

    let image = Image::load(arg_or(
        5,
        "/Volumes/macEx/分割/yolo_dataset/images/val/Image_20251231150417721.bmp",
    ))?;
    let model_path = arg_or(6, "/Volumes/macEx/Downloads/runs/yolo26_5090_dual/weights/best.onnx");

    // 创建分割引擎
    let engine = rust_onnx_infer::core::factory::create_segmentation_engine(
        &model_path,
        DeviceType::Cpu,
        0.5,
        0.45,
    )?;

    let segmentations = engine.predict(&image)?;
    println!("Segmented {} objects", segmentations.len());

    // 可视化
    let visualized = draw_segmentations(&image, &segmentations)?;
    visualized.save("segmentation_result.jpg")?;
    Ok(())
}

/// 使用工厂（Builder 模式）创建引擎
fn factory_example() -> rust_onnx_infer::Result<()> {
    println!("\n=== Factory Example ===");

    let image = Image::load("image.jpg")?;

    // 使用 Builder 模式创建引擎
    let engine = builder()
        .model_path("yolov8n.onnx")
        .device_type(DeviceType::Auto)
        .model_type(ModelType::Detection)
        .conf_threshold(0.5)
        .nms_threshold(0.45)
        .build()?;

    match engine {
        rust_onnx_infer::core::factory::DynEngine::Detection(e) => {
            let detections = e.predict(&image)?;
            println!("Detected {} objects", detections.len());
        }
        _ => unreachable!("builder configured for DETECTION"),
    }
    Ok(())
}

/// 批量推理示例
fn batch_inference_example() -> rust_onnx_infer::Result<()> {
    println!("\n=== Batch Inference Example ===");

    // 加载多张图像
    let images = vec![
        Image::load("image1.jpg")?,
        Image::load("image2.jpg")?,
        Image::load("image3.jpg")?,
    ];

    let engine = rust_onnx_infer::core::factory::create_detection_engine(
        "yolov8n.onnx",
        DeviceType::Cuda,
        0.5,
        0.45,
    )?;

    // 批量推理
    let batch_results = engine.predict_batch(&images)?;
    for (i, results) in batch_results.iter().enumerate() {
        println!("Image {}: {} detections", i + 1, results.len());
    }
    Ok(())
}

/// LightGlue 特征匹配示例
fn light_glue_example() -> rust_onnx_infer::Result<()> {
    println!("\n=== LightGlue Feature Matching Example ===");

    let img0 = Image::load("/Users/xiongguochao/Desktop/nolabel/kb/template.png")?;
    let img1 = Image::load("/Users/xiongguochao/Desktop/nolabel/kb/KB_20260402_162001_camera0.jpg")?;

    let engine = rust_onnx_infer::core::factory::create_light_glue_engine_with_input_size(
        "/Volumes/macEx/Downloads/superpoint_lightglue_pipeline.onnx",
        DeviceType::Cpu,
        0.3,
        640,
        640,
    )?;

    // 单对匹配
    let result = engine.match_images(&img0, &img1)?;
    println!("Keypoints0: {}", result.keypoints0.len());
    println!("Keypoints1: {}", result.keypoints1.len());
    println!("Matches: {}", result.match_count());

    for m in &result.matches {
        println!(
            "  match: ({:.1},{:.1}) <-> ({:.1},{:.1})  score={:.3}",
            m.kp0.x, m.kp0.y, m.kp1.x, m.kp1.y, m.score
        );
    }

    // 生成可视化结果图（两张图并排 + 匹配连线）
    let visualized = draw_light_glue_match(&img0, &img1, &result)?;
    visualized.save("lightglue_match_result.jpg")?;
    println!("Match result saved to lightglue_match_result.jpg");

    // 批量匹配
    let pairs = vec![img0.clone(), img1.clone(), img0.clone(), img1.clone()];
    let batch = engine.match_batch(&pairs)?;
    println!("Batch size: {}", batch.len());
    for (i, r) in batch.iter().enumerate() {
        println!("  pair {i}: {} matches", r.match_count());
    }
    Ok(())
}

// ==================== 简易可视化工具 ====================

fn bgr(color: (u8, u8, u8)) -> image::Rgb<u8> {
    image::Rgb([color.2, color.1, color.0])
}

fn to_rgb(image: &Image) -> image::RgbImage {
    match image.to_dynamic().expect("image convert") {
        image::DynamicImage::ImageRgb8(rgb) => rgb,
        other => other.to_rgb8(),
    }
}

fn from_rgb(rgb: image::RgbImage) -> Image {
    Image::from_rgb(rgb.width() as usize, rgb.height() as usize, rgb.as_raw())
        .expect("rgb buffer size valid")
}

/// 3x5 点阵数字（简化示例字体）。
const DIGITS: [[u8; 5]; 10] = [
    [0b111, 0b101, 0b101, 0b101, 0b111], // 0
    [0b010, 0b110, 0b010, 0b010, 0b111], // 1
    [0b111, 0b001, 0b111, 0b100, 0b111], // 2
    [0b111, 0b001, 0b111, 0b001, 0b111], // 3
    [0b101, 0b101, 0b111, 0b001, 0b001], // 4
    [0b111, 0b100, 0b111, 0b001, 0b111], // 5
    [0b111, 0b100, 0b111, 0b101, 0b111], // 6
    [0b111, 0b001, 0b010, 0b010, 0b010], // 7
    [0b111, 0b101, 0b111, 0b101, 0b111], // 8
    [0b111, 0b101, 0b111, 0b001, 0b111], // 9
];

/// 在指定位置绘制 class_id 数字（3x5 点阵，放大 3 倍）。
fn draw_class_id(rgb: &mut image::RgbImage, x: i32, y: i32, class_id: i32, color: image::Rgb<u8>) {
    let digits = class_id.abs().to_string();
    let mut cx = x.max(0);
    for ch in digits.chars() {
        let glyph = DIGITS[ch.to_digit(10).unwrap_or(0) as usize];
        for (gy, row) in glyph.iter().enumerate() {
            for gx in 0..3 {
                if row & (0b100 >> gx) != 0 {
                    for sy in 0..3 {
                        for sx in 0..3 {
                            let px = cx + gx * 3 + sx;
                            let py = y + gy as i32 * 3 + sy;
                            if px >= 0 && py >= 0 && (px as u32) < rgb.width() && (py as u32) < rgb.height() {
                                rgb.put_pixel(px as u32, py as u32, color);
                            }
                        }
                    }
                }
            }
        }
        cx += 12;
    }
}

/// 分类结果可视化：左上角绘制 Top-1 类别与置信度。
fn draw_classification(image: &Image, result: &ClassificationResult) -> rust_onnx_infer::Result<Image> {
    let mut rgb = to_rgb(image);
    draw_class_id(&mut rgb, 8, 8, result.class_id as i32, bgr((0, 255, 0)));
    Ok(from_rgb(rgb))
}

/// 检测结果可视化：绘制检测框与类别名。
fn draw_detections(image: &Image, detections: &[Detection]) -> rust_onnx_infer::Result<Image> {
    let mut rgb = to_rgb(image);
    for det in detections {
        let (x1, y1, x2, y2) = (
            det.x1().round() as i32,
            det.y1().round() as i32,
            det.x2().round() as i32,
            det.y2().round() as i32,
        );
        imageproc::drawing::draw_hollow_rect_mut(&mut rgb, imageproc::rect::Rect::at(x1, y1).of_size((x2 - x1).max(1) as u32, (y2 - y1).max(1) as u32), bgr((0, 255, 0)));
        draw_class_id(&mut rgb, x1.max(0), (y1 - 18).max(0), det.class_id, bgr((0, 255, 0)));
    }
    Ok(from_rgb(rgb))
}

/// 分割结果可视化：半透明掩码叠加 + 检测框。
fn draw_segmentations(image: &Image, segmentations: &[Segmentation]) -> rust_onnx_infer::Result<Image> {
    let mut rgb = to_rgb(image);
    for seg in segmentations {
        if let Ok(mask) = seg.binary_mask() {
            let (mw, mh) = (mask.width(), mask.height());
            let (iw, ih) = (rgb.width() as usize, rgb.height() as usize);
            for y in 0..ih.min(mh) {
                for x in 0..iw.min(mw) {
                    if mask.data()[y * mw + x] > 0 {
                        let px = rgb.get_pixel_mut(x as u32, y as u32);
                        // 绿色 50% 叠加
                        px.0[1] = px.0[1].saturating_mul(5) / 10 + 128;
                    }
                }
            }
        }
        let d = &seg.detection;
        let (x1, y1, x2, y2) = (d.x1().round() as i32, d.y1().round() as i32, d.x2().round() as i32, d.y2().round() as i32);
        imageproc::drawing::draw_hollow_rect_mut(&mut rgb, imageproc::rect::Rect::at(x1, y1).of_size((x2 - x1).max(1) as u32, (y2 - y1).max(1) as u32), bgr((255, 0, 0)));
    }
    Ok(from_rgb(rgb))
}

/// LightGlue 匹配可视化：两图并排 + 匹配连线（坐标需按 scale/pad 还原到原图）。
fn draw_light_glue_match(
    img0: &Image,
    img1: &Image,
    result: &LightGlueMatchResult,
) -> rust_onnx_infer::Result<Image> {
    let rgb0 = to_rgb(img0);
    let rgb1 = to_rgb(img1);
    let h = rgb0.height().max(rgb1.height());
    let mut canvas = image::RgbImage::from_pixel(rgb0.width() + rgb1.width(), h, image::Rgb([0, 0, 0]));
    image::imageops::overlay(&mut canvas, &rgb0, 0, 0);
    image::imageops::overlay(&mut canvas, &rgb1, rgb0.width() as i64, 0);

    // 坐标还原：x_orig = (x - pad_left) / scale
    let restore = |kp: rust_onnx_infer::model::LightGlueKeyPoint, scale: f32, pad_left: i32, pad_top: i32| -> (f32, f32) {
        (
            (kp.x - pad_left as f32) / scale.max(f32::EPSILON),
            (kp.y - pad_top as f32) / scale.max(f32::EPSILON),
        )
    };
    for m in &result.matches {
        let (x0, y0) = restore(m.kp0, result.scale0, result.pad_left0, result.pad_top0);
        let (x1, y1) = restore(
            m.kp1,
            result.scale1,
            result.pad_left1 + rgb0.width() as i32,
            result.pad_top1,
        );
        imageproc::drawing::draw_line_segment_mut(
            &mut canvas,
            (x0, y0),
            (x1, y1),
            bgr((0, 200, 255)),
        );
    }
    Ok(from_rgb(canvas))
}
