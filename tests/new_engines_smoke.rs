//! 5 个新引擎（姿态 / 人脸检测 / 人脸识别 / 深度估计 / 风格迁移）真实模型冒烟测试。
//!
//! 模型与测试图默认取 `testmodels/`，可用环境变量覆盖：
//! ```text
//! TESTMODELS_DIR=/path/to/testmodels cargo test --test new_engines_smoke -- --ignored --test-threads=1 --nocapture
//! ```
//! 所有用例标记 `#[ignore]`：默认 `cargo test` 不依赖模型文件。

use std::path::{Path, PathBuf};

use rust_onnx_infer::core::OnnxInferenceEngine;
use rust_onnx_infer::engines::depth::DepthEstimationEngine;
use rust_onnx_infer::engines::face_detection::{FaceDetectionEngine, FaceDetectResult};
use rust_onnx_infer::engines::face_recognition::FaceRecognitionEngine;
use rust_onnx_infer::engines::pose::PoseEngine;
use rust_onnx_infer::engines::style_transfer::StyleTransferEngine;
use rust_onnx_infer::model::Keypoint;
use rust_onnx_infer::{DeviceType, Image, Result, VisionError};

// ==================== 公共辅助 ====================

/// 模型 / 测试图目录（环境变量 `TESTMODELS_DIR`，默认 `testmodels/`）。
fn testmodels_dir() -> PathBuf {
    std::env::var("TESTMODELS_DIR")
        .unwrap_or_else(|_| "testmodels/".to_string())
        .into()
}

fn cpu() -> DeviceType {
    DeviceType::Cpu
}

/// 桌面杯子图：深度 / 风格迁移的通用测试图（无人物、无人脸）。
fn cup_image() -> Result<Image> {
    Image::load("/Users/xiongguochao/Desktop/杯子.jpg")
}

/// 在候选文件名中找第一个存在的文件。
///
/// 候选既按 `TESTMODELS_DIR` 下的相对名查找，也接受候选里的绝对路径。
fn find_file(candidates: &[&str]) -> Result<PathBuf> {
    // 先按 models/ 树搜索候选
    for c in candidates {
        let p = find_any(c);
        if p.exists() {
            return Ok(p);
        }
    }
    let dir = testmodels_dir();
    for c in candidates {
        let p = dir.join(c);
        if p.is_file() {
            return Ok(p);
        }
    }
    for c in candidates {
        let p = Path::new(c);
        if p.is_file() {
            return Ok(p.to_path_buf());
        }
    }
    Err(VisionError::invalid_argument(format!(
        "未找到候选文件中的任何一个: {candidates:?}（TESTMODELS_DIR={}）",
        dir.display()
    )))
}

/// 姿态模型候选名（YOLO-Pose 家族）。
const POSE_MODELS: [&str; 3] = ["yolov8n-pose.onnx", "yolov8s-pose.onnx", "yolo11n-pose.onnx"];

/// 人脸检测模型候选名（YuNet 家族，OpenCV Zoo 命名优先）。
const YUNET_MODELS: [&str; 3] = [
    "face_detection_yunet_2023mar.onnx",
    "face_detection_yunet.onnx",
    "yunet.onnx",
];

/// 人脸识别模型候选名（SFace / MobileFaceNet）。
const FACE_REC_MODELS: [&str; 3] = [
    "face_recognition_sface_2021dec.onnx",
    "face_recognition_sface.onnx",
    "sface.onnx",
];

/// 深度估计模型候选名（Depth Anything / MiDaS）。
const DEPTH_MODELS: [&str; 4] = [
    "depth_anything_v2_small.onnx",
    "depth_anything_v2_vits.onnx",
    "depth_anything.onnx",
    "midas_v21_small.onnx",
];

/// 风格迁移模型候选名（Model Zoo cycleGAN 导出，带 `style_` 前缀）。
const STYLE_MODELS: [&str; 5] = [
    "style_candy-9.onnx",
    "style_mosaic-9.onnx",
    "style_transfer.onnx",
    "candy-9.onnx",
    "mosaic-9.onnx",
];

/// 含人测试图候选名（姿态估计用；bus.jpg 为 Ultralytics 经典姿态测试图）。
const PERSON_IMAGES: [&str; 4] = ["bus.jpg", "person.jpg", "people.jpg", "zidane.jpg"];

/// 含人脸测试图候选名（人脸检测 / 识别用；zidane.jpg 人脸大而清晰）。
const FACE_IMAGES: [&str; 4] = ["zidane.jpg", "face.jpg", "faces.jpg", "bus.jpg"];

/// 杯子图候选名（负样本 / 通用图；最后回退桌面原图）。
const CUP_IMAGES: [&str; 3] = ["cup.jpg", "杯子.jpg", "/Users/xiongguochao/Desktop/杯子.jpg"];

/// 从人脸检测结果取框 (x1, y1, x2, y2)。
fn face_rect(face: &FaceDetectResult) -> (f64, f64, f64, f64) {
    let b = face.detection.bbox;
    (b.x1, b.y1, b.x2, b.y2)
}

/// 把原图坐标系的 5 关键点平移到 crop 局部坐标系。
fn shifted_landmarks(face: &FaceDetectResult, dx: f32, dy: f32) -> Vec<Keypoint> {
    face.landmarks
        .iter()
        .map(|k| Keypoint::new(k.x - dx, k.y - dy, k.score))
        .collect()
}

/// 按边界裁剪（坐标 clamp 到图内，至少 1x1），返回裁剪图与实际裁剪原点 (x, y)。
fn crop_clamped(img: &Image, x1: f64, y1: f64, x2: f64, y2: f64) -> Result<(Image, usize, usize)> {
    let w = img.width() as f64;
    let h = img.height() as f64;
    let cx1 = x1.max(0.0).min(w - 1.0);
    let cy1 = y1.max(0.0).min(h - 1.0);
    let cx2 = x2.max(cx1 + 1.0).min(w);
    let cy2 = y2.max(cy1 + 1.0).min(h);
    let cropped = img.crop(
        cx1 as usize,
        cy1 as usize,
        (cx2 - cx1) as usize,
        (cy2 - cy1) as usize,
    )?;
    Ok((cropped, cx1 as usize, cy1 as usize))
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

// ==================== 姿态估计 ====================

smoke!(pose_yolo_person_keypoints, "姿态估计 yolov8n-pose（COCO 17 关键点）", {
    let engine = PoseEngine::for_yolo(find_file(&POSE_MODELS)?, cpu(), 0.25, 0.45)?;
    let image = Image::load(find_file(&PERSON_IMAGES)?)?;
    let results = engine.predict(&image)?;
    eprintln!(
        "人像图 {}x{}，检出 {} 个行人实例",
        image.width(),
        image.height(),
        results.len()
    );
    assert!(!results.is_empty(), "人像图应检出姿态实例");
    for r in &results {
        // COCO 17 点必须全量输出（score 低的点保留原值而非丢弃）
        assert_eq!(r.keypoints.len(), 17, "COCO 姿态关键点应为 17 个");
        eprintln!(
            "  {} conf={:.3} nose=({:.0},{:.0},{:.2})",
            r.detection.class_name,
            r.confidence(),
            r.keypoints[0].x,
            r.keypoints[0].y,
            r.keypoints[0].score
        );
    }
    // 至少一个实例存在可信关键点
    assert!(
        results
            .iter()
            .any(|r| r.keypoints.iter().any(|k| k.score > 0.3)),
        "应有置信度 > 0.3 的关键点"
    );
});

smoke!(pose_yolo26_end2end, "姿态估计 yolo26n-pose（End2End 布局）", {
    let engine = PoseEngine::for_yolo(
        std::path::Path::new(&std::env::var("MODEL_DIR").unwrap_or_else(|_| "/Volumes/macEx/AI/vision-commons/models".to_string())).join("yolo26n-pose.onnx"),
        cpu(),
        0.25,
        0.45,
    )?;
    let image = Image::load(find_file(&PERSON_IMAGES)?)?;
    let results = engine.predict(&image)?;
    eprintln!("yolo26n-pose（End2End）检出 {} 个行人实例", results.len());
    assert!(!results.is_empty(), "yolo26n-pose 应检出姿态实例");
    for r in &results {
        assert_eq!(r.keypoints.len(), 17, "COCO 姿态关键点应为 17 个");
        eprintln!(
            "  {} conf={:.3} nose=({:.0},{:.0},{:.2})",
            r.detection.class_name,
            r.confidence(),
            r.keypoints[0].x,
            r.keypoints[0].y,
            r.keypoints[0].score
        );
    }
    assert!(
        results
            .iter()
            .any(|r| r.keypoints.iter().any(|k| k.score > 0.3)),
        "应有置信度 > 0.3 的关键点"
    );
});

// ==================== 人脸检测 ====================

smoke!(face_detection_yunet, "人脸检测 YuNet（5 点 landmark）", {
    let engine = FaceDetectionEngine::new(find_file(&YUNET_MODELS)?, cpu())?;
    let image = Image::load(find_file(&FACE_IMAGES)?)?;
    let faces = engine.predict(&image)?;
    eprintln!(
        "人脸图 {}x{}，检出 {} 张人脸",
        image.width(),
        image.height(),
        faces.len()
    );
    assert!(!faces.is_empty(), "人脸图应检出人脸");
    for f in &faces {
        // YuNet 每张脸固定输出 5 个 landmark：左右眼、鼻尖、左右嘴角
        assert_eq!(f.landmarks.len(), 5, "YuNet 应输出 5 个 landmark");
        eprintln!(
            "  face conf={:.3} 框=({:.0},{:.0},{:.0},{:.0})",
            f.confidence(),
            f.detection.bbox.x1,
            f.detection.bbox.y1,
            f.detection.bbox.x2,
            f.detection.bbox.y2
        );
    }
});

// ==================== 人脸识别 ====================

smoke!(face_recognition_same_vs_other, "人脸识别 SFace（同人 cos>0.35 / 非人脸 cos<0.7）", {
    let image = Image::load(find_file(&FACE_IMAGES)?)?;

    // 1. YuNet 定位人脸，取 5 关键点
    let det = FaceDetectionEngine::new(find_file(&YUNET_MODELS)?, cpu())?;
    let faces = det.predict(&image)?;
    assert!(!faces.is_empty(), "同人样本需要先检出人脸");
    let (fx1, fy1, fx2, fy2) = face_rect(&faces[0]);

    // 2. SFace 引擎（输出 128 维 L2 归一化 embedding）：同人取两个略有差异的
    //    crop（原框 / 外扩 15%），关键点平移到各自局部坐标系后走 5 点对齐提取
    let rec = FaceRecognitionEngine::new_sface(find_file(&FACE_REC_MODELS)?, cpu())?;
    let dw = (fx2 - fx1) * 0.15;
    let dh = (fy2 - fy1) * 0.15;
    let (crop1, ox1, oy1) = crop_clamped(&image, fx1, fy1, fx2, fy2)?;
    let (crop2, ox2, oy2) = crop_clamped(&image, fx1 - dw, fy1 - dh, fx2 + dw, fy2 + dh)?;
    let e1 = rec.extract(&crop1, &shifted_landmarks(&faces[0], ox1 as f32, oy1 as f32))?;
    let e2 = rec.extract(&crop2, &shifted_landmarks(&faces[0], ox2 as f32, oy2 as f32))?;
    let same = FaceRecognitionEngine::cosine_similarity(&e1, &e2);
    eprintln!("同人两个 crop 余弦相似度 = {:.4}（dim={}）", same, e1.len());
    assert!(same > 0.35, "同人相似度应 > 0.35，实际 {same:.4}");

    // 3. 异人（宽松断言）：非人脸负样本（杯子图局部 crop）
    let cup = Image::load(find_file(&CUP_IMAGES)?)?;
    let (neg, _, _) = crop_clamped(&cup, 10.0, 10.0, 138.0, 138.0)?;
    let e3 = rec.predict_impl(&neg)?;
    let diff = FaceRecognitionEngine::cosine_similarity(&e1, &e3);
    eprintln!("同人 vs 非人脸 crop 余弦相似度 = {:.4}", diff);
    assert!(diff < 0.7, "非人脸相似度应 < 0.7，实际 {diff:.4}");
});

// ==================== 深度估计 ====================

smoke!(depth_estimation_range, "深度估计 DepthAnything（原图分辨率 FloatMask，值域 [0,1]）", {
    let engine = DepthEstimationEngine::new(find_file(&DEPTH_MODELS)?, cpu())?;
    let image = cup_image()?;
    let depth = engine.predict_depth(&image)?;
    let (mut mn, mut mx) = (f32::MAX, f32::MIN);
    for &v in depth.data() {
        mn = mn.min(v);
        mx = mx.max(v);
    }
    eprintln!(
        "输入 {}x{}，模型输入 {}x{}，depth 输出 {}x{}，值域 [{:.4}, {:.4}]",
        image.width(),
        image.height(),
        engine.base.input_width(),
        engine.base.input_height(),
        depth.width(),
        depth.height(),
        mn,
        mx
    );
    assert!(!depth.is_empty(), "深度图不应为空");
    // 引擎约定：模型输出 min-max 归一化后双线性还原到原图分辨率
    assert_eq!(
        (depth.width(), depth.height()),
        (image.width(), image.height()),
        "深度图尺寸应还原到原图分辨率"
    );
    assert!(
        mn >= 0.0 && mx <= 1.0,
        "深度值应归一化到 [0,1]，实际 [{mn:.4}, {mx:.4}]"
    );
});

// ==================== 风格迁移 ====================

smoke!(style_transfer_same_size, "风格迁移 candy（输出与输入同尺寸）", {
    let engine = StyleTransferEngine::new(find_file(&STYLE_MODELS)?, cpu())?;
    let image = cup_image()?;
    let out = engine.predict(&image)?;
    eprintln!(
        "输入 {}x{}x{} → 输出 {}x{}x{}",
        image.width(),
        image.height(),
        image.channels(),
        out.width(),
        out.height(),
        out.channels()
    );
    assert_eq!(out.width(), image.width(), "风格迁移输出宽应与输入一致");
    assert_eq!(out.height(), image.height(), "风格迁移输出高应与输入一致");
    assert_eq!(out.channels(), 3, "风格迁移输出应为 3 通道");
});

// ==================== 工厂入口 ====================

smoke!(factory_new_engine_entries, "工厂入口（5 个新引擎构造 + 单次推理）", {
    use rust_onnx_infer::core::factory::{
        create_depth_engine, create_face_detection_engine, create_face_recognition_engine,
        create_pose_engine, create_style_transfer_engine,
    };
    let image = cup_image()?;

    // 姿态：Box<dyn Output = Vec<PoseResult>>
    let pose = create_pose_engine(find_file(&POSE_MODELS)?, cpu(), 0.25)?;
    let n = pose.predict(&Image::load(find_file(&PERSON_IMAGES)?)?)?.len();
    eprintln!("create_pose_engine → {n} 个实例");

    // 人脸检测：Box<dyn Output = Vec<FaceDetectResult>>
    let fd = create_face_detection_engine(find_file(&YUNET_MODELS)?, cpu())?;
    let faces = fd.predict(&Image::load(find_file(&FACE_IMAGES)?)?)?;
    eprintln!("create_face_detection_engine → {} 张人脸", faces.len());
    assert_eq!(faces[0].landmarks.len(), 5, "工厂构造的引擎应输出 5 landmark");

    // 人脸识别：具体类型（SFace 约定）
    let fr = create_face_recognition_engine(find_file(&FACE_REC_MODELS)?, cpu())?;
    let face_img = Image::load(find_file(&FACE_IMAGES)?)?;
    let emb = if faces.is_empty() {
        Vec::new()
    } else {
        let (x1, y1, x2, y2) = face_rect(&faces[0]);
        let (crop, _, _) = crop_clamped(&face_img, x1, y1, x2, y2)?;
        fr.predict_impl(&crop)?
    };
    eprintln!("create_face_recognition_engine → dim={}", emb.len());

    // 深度：Box<dyn Output = FloatMask>
    let dep = create_depth_engine(find_file(&DEPTH_MODELS)?, cpu())?;
    let depth = dep.predict(&image)?;
    eprintln!("create_depth_engine → {}x{}", depth.width(), depth.height());
    assert_eq!(depth.width(), image.width(), "工厂构造的深度引擎应回原图分辨率");

    // 风格迁移：Box<dyn Output = Image>
    let st = create_style_transfer_engine(find_file(&STYLE_MODELS)?, cpu())?;
    let out = st.predict(&image)?;
    eprintln!("create_style_transfer_engine → {}x{}", out.width(), out.height());
    assert_eq!((out.width(), out.height()), (image.width(), image.height()));
});


// ==================== 实时姿态 / 人脸属性 / OCR ====================

fn tm_dir() -> std::path::PathBuf {
    std::env::var("TESTMODELS_DIR")
        .unwrap_or_else(|_| "testmodels".to_string())
        .into()
}

/// 在 testmodels/ 与 models/ 两棵树中查找文件（models/ 为按引擎组织的正式目录）。
fn find_any(name: &str) -> std::path::PathBuf {
    let direct = tm_dir().join(name);
    if direct.exists() {
        return direct;
    }
    for root in [std::path::PathBuf::from("models")] {
        if let Ok(entries) = walkdir_models(&root) {
            for p in entries {
                if p.file_name().map(|n| n == name).unwrap_or(false) {
                    return p;
                }
            }
        }
    }
    direct
}

fn walkdir_models(root: &std::path::Path) -> std::io::Result<Vec<std::path::PathBuf>> {
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

smoke!(rtmo_realtime_pose, "RTMO 实时姿态（一阶段 SimCC/End2End）", {
    let engine = rust_onnx_infer::engines::pose_rt::RealtimePoseEngine::new(
        find_any("rtmo_s.onnx"),
        cpu(),
    )?;
    let image = Image::load(find_file(&PERSON_IMAGES)?)?;
    let results = engine.predict(&image)?;
    eprintln!("RTMO 检出 {} 个行人", results.len());
    assert!(!results.is_empty(), "RTMO 应检出姿态实例");
    for r in &results {
        eprintln!("  {} conf={:.3} kpts={}", r.detection.class_name, r.confidence(), r.keypoints.len());
    }
    assert!(
        results.iter().any(|r| r.keypoints.iter().any(|k| k.score > 0.3)),
        "应有置信度 > 0.3 的关键点"
    );
});

smoke!(face_attribute_age_gender_expression, "人脸属性（年龄性别 + 表情，配合 YuNet）", {
    let det = rust_onnx_infer::engines::face_detection::FaceDetectionEngine::new(
        find_file(&YUNET_MODELS)?,
        cpu(),
    )?;
    let image = Image::load(find_any("zidane.jpg"))?;
    let faces = det.predict(&image)?;
    eprintln!("YuNet 检出 {} 张脸", faces.len());
    assert!(!faces.is_empty(), "zidane.jpg 应检出人脸");

    let rect = rust_onnx_infer::imaging::Rect::new(
        faces[0].detection.x1().round() as i32,
        faces[0].detection.y1().round() as i32,
        (faces[0].detection.x2() - faces[0].detection.x1()).round() as i32,
        (faces[0].detection.y2() - faces[0].detection.y1()).round() as i32,
    );

    let ag = rust_onnx_infer::engines::face_attribute::AgeGenderEngine::new(
        find_any("age_gender.onnx"),
        cpu(),
    )?;
    let attr = ag.predict_face(&image, rect)?;
    eprintln!("年龄 {:.1}，男性置信 {:.3}", attr.age, attr.gender_score);
    assert!((18.0..=75.0).contains(&attr.age), "官方年龄模型口径 [18,75]");

    let expr = rust_onnx_infer::engines::face_attribute::ExpressionEngine::new(
        find_any("emotion_ferplus.onnx"),
        cpu(),
    )?;
    let e = expr.predict_expr(&image, rect)?;
    eprintln!("表情 = {} ({:.3})", e.label, e.scores[e.label_id]);
    assert!(e.scores.len() == 8, "FER+ 应为 8 类");
});

smoke!(obb_both_layouts, "OBB 旋转框检测（v8 传统 + yolo26 End2End）", {
    for (name, path) in [
        ("yolov8n-obb 传统", find_any("yolov8n-obb.onnx")),
        ("yolo26n-obb End2End", find_any("yolo26n-obb.onnx")),
    ] {
        let engine = rust_onnx_infer::engines::obb_detection::ObbDetectionEngine::new(path, cpu())?;
        let image = Image::load(find_file(&PERSON_IMAGES)?)?;
        let results = engine.predict(&image)?;
        eprintln!("{name}: 检出 {} 个旋转框", results.len());
        for r in results.iter().take(3) {
            let (x1, y1, x2, y2) = r.aabb();
            eprintln!(
                "  {} conf={:.3} angle={:.3}rad aabb=({:.0},{:.0},{:.0},{:.0})",
                r.class_name, r.confidence, r.angle_rad, x1, y1, x2, y2
            );
        }
        assert!(results.len() <= 300, "结果数合理上限");
    }
});

smoke!(portrait_matting_and_deblur, "人像分割 + 去模糊（代理实现，首次真机验证）", {
    let engine = rust_onnx_infer::engines::portrait_matting::PortraitMattingEngine::new(
        find_any("pp_humanseg_2023mar.onnx"),
        cpu(),
    )?;
    let image = Image::load(find_file(&PERSON_IMAGES)?)?;
    let alpha = engine.predict_alpha(&image)?;
    let mean: f64 = alpha.data().iter().map(|&v| v as f64).sum::<f64>() / alpha.len() as f64;
    eprintln!("人像 alpha: {}x{}, 均值 {:.4}", alpha.width(), alpha.height(), mean);
    assert!(!alpha.is_empty());
    let cutout = engine.predict_cutout(&image)?;
    assert_eq!(cutout.channels(), 4, "cutout 应为 BGRA");

    let deblur = rust_onnx_infer::engines::deblur::DeblurEngine::new(
        find_any("nafnet_deblur_2025may.onnx"),
        cpu(),
    )?;
    // NafNet 对输入对齐有要求（Pad 报错说明 256 不满足），裁剪预留 32 倍数
    let small = image.crop(0, 0, 512, 512)?;
    let out = deblur.predict_impl(&small)?;
    eprintln!("去模糊输出: {}x{}（输入 512x512）", out.width(), out.height());
    assert_eq!((out.width(), out.height()), (512, 512));
});

smoke!(quality_and_qr, "图像质量评估（纯算法）+ 二维码检测（WeChat QR）", {
    let image = Image::load(find_file(&PERSON_IMAGES)?)?;
    let assessor = rust_onnx_infer::engines::image_quality::QualityAssessor::new();
    let report = assessor.assess(&image)?;
    eprintln!("质量: blur={:.1} bright={:.1} contrast={:.1} overall={:.3}",
        report.blur_score, report.brightness, report.contrast, report.overall);
    assert!(report.overall >= 0.0 && report.overall <= 1.0);

    let qr = rust_onnx_infer::engines::qr_detector::QrDetector::new(
        find_any("wechat_qr_detect.onnx"),
        cpu(),
    )?;
    let qr_image = Image::load(find_any("qr_test.png"))?;
    let dets = qr.detect(&qr_image)?;
    eprintln!("二维码检出 {} 个", dets.len());
    assert!(!dets.is_empty(), "二维码测试图应检出码区");
});

smoke!(hand_landmark_21, "手部 21 关键点（palm 检测 → RTMPose-hand 串联）", {
    let det = rust_onnx_infer::engines::hand_keypoint::HandDetectionEngine::new(
        find_any("hand_landmark_full.onnx"),
        cpu(),
    )?;
    let lm = rust_onnx_infer::engines::hand_keypoint::HandLandmarkEngine::new(
        find_any("rtmpose_m_hand.onnx"),
        cpu(),
    )?;
    // 手部图：生成含手绘图案有限——用人体图（hands 可能不可见时降级断言）
    let image = Image::load(find_file(&PERSON_IMAGES)?)?;
    let hands = det.detect(&image)?;
    eprintln!("检测到 {} 只手", hands.len());
    if let Some(first) = hands.first() {
        let rect = rust_onnx_infer::imaging::Rect::new(first.x, first.y, first.width, first.height);
        let result = lm.extract(&image, rect)?;
        eprintln!("21 点: n={}", result.points.len());
        assert_eq!(result.points.len(), 21);
    }
});

smoke!(wholebody_133_keypoints, "WholeBody 133 关键点（RTMPose-m：body17+foot6+face68+hand42）", {
    let engine = rust_onnx_infer::engines::wholebody::WholeBodyPoseEngine::new(
        find_any("rtmpose_m_wholebody.onnx"),
        cpu(),
    )?;
    let image = Image::load(find_file(&PERSON_IMAGES)?)?;
    // 整图当作单人区域（bus.jpg 4 人会拉低分数但可验证管线）
    let result = engine.predict(&image)?;
    eprintln!("133 点输出: n={}", result.keypoints.len());
    let high = result.keypoints.iter().filter(|k| k.score > 0.3).count();
    eprintln!("score>0.3 的点数: {high}");
    assert_eq!(result.keypoints.len(), 133, "WholeBody 应输出 133 点");
    assert!(high >= 17, "body 17 点应有可信检出（实际 {high}）");
    // 分段访问
    assert_eq!(result.keypoints[0..17].len(), 17);
    assert_eq!(result.keypoints.len(), 133);
    let lh = &result.keypoints[rust_onnx_infer::engines::wholebody::segments::LEFT_HAND];
    assert_eq!(lh.len(), 21);
});

smoke!(dart_open_vocab, "DART 开放词汇检测分割（SAM3 骨干，单类 bottle）", {
    let models_dir = std::env::var("DART_MODELS_DIR")
        .unwrap_or_else(|_| "/Volumes/macEx/AI/vision-commons/models/dart".to_string());
    let engine = rust_onnx_infer::engines::dart::DartEngine::new(&models_dir, cpu())?;
    engine.set_classes(&["bottle"])?;
    let image = Image::load(find_file(&PERSON_IMAGES)?)?;
    let results = engine.predict(&image)?;
    eprintln!("DART 检出 {} 个 bottle 实例", results.len());
    for r in results.iter().take(3) {
        eprintln!("  {r}");
    }
    // 管线完整性断言（跨域图允许 0 检出）
    assert!(results.len() <= 8, "单类检出数合理上限");
});

smoke!(action_recognition_stgcn, "骨架动作识别（ST-GCN，COCO17 帧序列）", {
    let engine = rust_onnx_infer::engines::action_recognition::ActionRecognitionEngine::new(
        tm_dir().join("action_stgcn.onnx"),
        cpu(),
    )?;
    // 合成站立序列（30 帧，关键点固定站姿）+ 跌倒尾帧
    use rust_onnx_infer::model::Keypoint;
    let standing: Vec<Keypoint> = [
        (320.0, 100.0), (310.0, 90.0), (330.0, 90.0), (305.0, 110.0), (335.0, 110.0),
        (280.0, 180.0), (360.0, 180.0), (270.0, 280.0), (370.0, 280.0), (265.0, 370.0),
        (375.0, 370.0), (300.0, 420.0), (340.0, 420.0), (295.0, 560.0), (345.0, 560.0),
        (290.0, 700.0), (350.0, 700.0),
    ]
    .iter()
    .map(|&(x, y)| Keypoint::new(x, y, 0.9))
    .collect();
    let frames: Vec<Vec<Keypoint>> = (0..24).map(|_| standing.clone()).collect();
    let r1 = engine.classify(&frames)?;
    eprintln!("站立序列: {} conf={:.3}", r1.label, r1.scores[r1.label_id]);
    // 跌倒尾帧：躯干放平
    let mut fallen = standing.clone();
    for k in fallen.iter_mut() {
        let nx = 320.0 + (k.y - 320.0) * 0.2;
        let ny = 320.0 + (k.x - 320.0) * 0.2;
        *k = Keypoint::new(nx, ny, k.score);
    }
    let mut frames2 = frames.clone();
    frames2.extend((0..6).map(|_| fallen.clone()));
    let r2 = engine.classify(&frames2)?;
    eprintln!("带跌倒尾帧: {} conf={:.3}", r2.label, r2.scores[r2.label_id]);
});

smoke!(gesture_rules, "手势分类（纯几何规则，合成 21 点）", {
    use rust_onnx_infer::engines::gesture::GestureClassifier;
    use rust_onnx_infer::model::Keypoint;
    let clf = GestureClassifier::new();
    // 合成张开手掌（腕 0,0；五指伸直向上）
    let mut palm = vec![Keypoint::new(0.0, 0.0, 1.0)];
    for (bx, by) in [(-0.3, -0.2), (-0.1, -0.25), (0.05, -0.22), (0.18, -0.18), (0.3, -0.15)] {
        palm.push(Keypoint::new(bx, by, 1.0));              // MCP
        palm.push(Keypoint::new(bx * 1.15, by - 0.12, 1.0)); // PIP
        palm.push(Keypoint::new(bx * 1.2, by - 0.22, 1.0));  // DIP
        palm.push(Keypoint::new(bx * 1.25, by - 0.3, 1.0));  // 指尖
    }
    let r = clf.classify(&palm)?;
    eprintln!("张开手掌 → {} ({:.2})", r.label, r.score);
    // 合成握拳（指尖贴回 MCP）
    let mut fist = vec![Keypoint::new(0.0, 0.0, 1.0)];
    for (bx, by) in [(-0.3, -0.2), (-0.1, -0.25), (0.05, -0.22), (0.18, -0.18), (0.3, -0.15)] {
        fist.push(Keypoint::new(bx, by, 1.0));
        fist.push(Keypoint::new(bx, by - 0.02, 1.0));
        fist.push(Keypoint::new(bx, by - 0.03, 1.0));
        fist.push(Keypoint::new(bx, by - 0.03, 1.0));
    }
    let r2 = clf.classify(&fist)?;
    eprintln!("握拳 → {} ({:.2})", r2.label, r2.score);
    assert_ne!(r.label, r2.label, "手掌与拳头应分类不同");
});

smoke!(face_liveness_cdcn, "人脸活体检测（CDCN 分割式，spoof 均分）", {
    let det = rust_onnx_infer::engines::face_detection::FaceDetectionEngine::new(
        find_file(&YUNET_MODELS)?,
        cpu(),
    )?;
    let engine = rust_onnx_infer::engines::face_liveness::FaceLivenessEngine::new(
        find_any("face_liveness.onnx"),
        cpu(),
    )?;
    let image = Image::load(find_any("zidane.jpg"))?;
    let faces = det.predict(&image)?;
    eprintln!("检出 {} 张脸", faces.len());
    assert!(!faces.is_empty());
    let mut low_spoof = 0;
    for f in &faces {
        let d = &f.detection;
        let rect = rust_onnx_infer::imaging::Rect::new(
            d.x1().round() as i32,
            d.y1().round() as i32,
            (d.x2() - d.x1()).round() as i32,
            (d.y2() - d.y1()).round() as i32,
        );
        let r = engine.predict_face(&image, rect)?;
        eprintln!("  {r}");
        if r.is_real {
            low_spoof += 1;
        }
    }
    eprintln!("低 spoof 判定 {}/{} 张", low_spoof, faces.len());
});

smoke!(fiqa_quality_onnx, "图像质量 ONNX（FIQA EdgeNeXt 深度 IQA，高分=好）", {
    let engine = rust_onnx_infer::engines::image_quality::ImageQualityEngine::new_fiqa(
        find_any("FIQA_EdgeNeXt_XXS_1x3x352x352.onnx"),
        cpu(),
    )?;
    let image = Image::load(find_file(&PERSON_IMAGES)?)?;
    let q = engine.quality(&image)?;
    eprintln!("FIQA 质量分: {:.3}", q);
    assert!((0.0..=1.0).contains(&q), "质量分应在 [0,1]");
});

smoke!(license_plate_det_rec, "车牌检测+识别（mnet 检测 + LPRNet 识别，透视矫正）", {
    let engine = rust_onnx_infer::engines::license_plate::LicensePlateEngine::new(
        tm_dir().join("mnet_plate.onnx"),
        tm_dir().join("Final_LPRNet_model.onnx"),
        cpu(),
    )?;
    let image = Image::load(tm_dir().join("plate_test.jpg"))?;
    let results = engine.recognize(&image)?;
    eprintln!("车牌检出 {} 个", results.len());
    for r in &results {
        eprintln!("  plate_type={:?} text={:?} conf={:.3}", r.plate_type, r.text, r.score);
    }
    assert!(!results.is_empty(), "车牌测试图应检出车牌");
    assert!(!results[0].text.is_empty(), "识别文本非空");
});

smoke!(qr_decode_content, "二维码检测+内容解码（WeChat QR + rqrr）", {
    let qr = rust_onnx_infer::engines::qr_detector::QrDetector::new(
        find_any("wechat_qr_detect.onnx"),
        cpu(),
    )?;
    let image = Image::load(find_any("qr_decode_test.png"))?;
    let results = qr.decode(&image)?;
    eprintln!("解码结果:");
    for (d, text) in &results {
        eprintln!("  框=({},{},{}x{}) 文本={:?}", d.x, d.y, d.w, d.h, text);
    }
    assert!(!results.is_empty(), "应检出二维码");
    let text = results.iter().find_map(|(_, t)| t.clone()).expect("应解码出内容");
    assert!(text.contains("rust-onnx-infer"), "解码内容应包含生成文本，实际: {text}");
});

smoke!(semantic_segmentation_ade, "语义分割（SegFormer-B0 ADE20k 150 类）", {
    let engine = rust_onnx_infer::engines::semantic_segmentation::SemanticSegmentationEngine::new(
        find_any("segformer_b0_ade.onnx"),
        cpu(),
    )?;
    let image = Image::load(find_file(&PERSON_IMAGES)?)?;
    let map = engine.predict(&image)?;
    let hist = map.histogram();
    eprintln!("语义分割 {}x{}，top5:", map.width, map.height);
    for (id, n) in hist.iter().take(5) {
        eprintln!("  {} {}: {} px", id, map.class_name(*id), n);
    }
    assert!(!hist.is_empty());
    let overlay = engine.overlay(&image, 0.5)?;
    assert_eq!(overlay.channels(), 3);
});

smoke!(human_parsing_clothes, "人体解析（SegFormer-B2 人衣 18 类）", {
    let engine = rust_onnx_infer::engines::human_parsing::HumanParsingEngine::with_input_size(
        find_any("segformer_b2_clothes.onnx"),
        cpu(),
        512,
        512,
    )?;
    let full = Image::load(find_file(&PERSON_IMAGES)?)?;
    // B2 clothes 模型期望紧凑人像：裁剪出人物区域（bus.jpg 左侧行人）
    let image = full.crop(40, 380, 320, 520)?;
    let map = engine.predict(&image)?;
    eprintln!("人体解析 {}x{}，部件占比:", map.width, map.height);
    for (id, r) in map.part_ratios().iter().take(6) {
        eprintln!("  {} {}: {:.1}%", id, map.class_name(*id), r * 100.0);
    }
    assert!(!map.part_ratios().is_empty(), "人像 crop 应解析出部件");
});

smoke!(obb_sahi_sliced, "OBB 切片推理（SAHI + aabb 近似合并）", {
    let mut engine = rust_onnx_infer::engines::obb_detection::ObbDetectionEngine::new(
        find_any("yolov8n-obb.onnx"),
        cpu(),
    )?;
    engine
        .base
        .set_sahi_config(Some(rust_onnx_infer::sahi::SahiConfig::of(512, 512, 0.2, 0.2)));
    let image = Image::load(find_file(&PERSON_IMAGES)?)?;
    let dets = engine.predict(&image)?;
    eprintln!("OBB SAHI 检出 {} 个（合并后 angle 近似 0）", dets.len());
    engine.base.disable_sahi();
    let dets2 = engine.predict(&image)?;
    eprintln!("关闭 SAHI 后 {} 个", dets2.len());
});

smoke!(face_landmark_106, "人脸 106 关键点（insightface 2d106det + YuNet 串联）", {
    let det = rust_onnx_infer::engines::face_detection::FaceDetectionEngine::new(
        find_file(&YUNET_MODELS)?,
        cpu(),
    )?;
    let engine = rust_onnx_infer::engines::face_landmark106::FaceLandmark106Engine::new(
        find_any("face_landmark106.onnx"),
        cpu(),
    )?;
    let image = Image::load(find_any("zidane.jpg"))?;
    let faces = det.predict(&image)?;
    eprintln!("YuNet 检出 {} 张脸", faces.len());
    assert!(!faces.is_empty());
    let d = &faces[0].detection;
    let rect = rust_onnx_infer::imaging::Rect::new(
        d.x1().round() as i32,
        d.y1().round() as i32,
        (d.x2() - d.x1()).round() as i32,
        (d.y2() - d.y1()).round() as i32,
    );
    let lm = engine.extract(&image, rect)?;
    eprintln!("106 点: n={} 首点=({:.1},{:.1})", lm.points.len(), lm.points[0].x, lm.points[0].y);
    assert_eq!(lm.points.len(), 106, "应输出 106 个关键点");
    // 所有点都应落在图内（外扩裁剪还原正确）
    assert!(
        lm.points.iter().all(|p| p.x >= 0.0 && p.y >= 0.0),
        "关键点应为非负坐标"
    );
});

smoke!(hand_detection, "手部检测（MediaPipe palm detector）", {
    let engine = rust_onnx_infer::engines::hand_keypoint::HandDetectionEngine::new(
        find_any("hand_landmark_full.onnx"),
        cpu(),
    )?;
    let image = Image::load(find_file(&PERSON_IMAGES)?)?;
    let dets = engine.detect(&image)?;
    eprintln!("手部检出 {} 个（跨域自然图允许 0）", dets.len());
    for d in dets.iter().take(3) {
        eprintln!("  hand score={:.3} ({},{},{}x{})", d.score, d.x, d.y, d.width, d.height);
    }
    assert!(dets.len() <= 4, "手部检测上限 4");
});

smoke!(ocr_pipeline_v5, "OCR 流水线（PP-OCRv5 mobile det + rec，中英混合）", {
    let pipeline = rust_onnx_infer::engines::ocr::OcrPipeline::new(
        find_any("ppocrv5_det.onnx"),
        find_any("ppocrv5_rec.onnx"),
        find_any("ppocrv5_dict.txt"),
        cpu(),
    )?;
    let image = Image::load(find_any("ocr_test.png"))?;
    let results = pipeline.predict(&image)?;
    eprintln!("v5 识别出 {} 行", results.len());
    let all_text: String = results
        .iter()
        .map(|(_, l)| l.text.clone())
        .collect::<Vec<_>>()
        .join(" ");
    eprintln!("文本: {all_text}");
    assert!(all_text.contains("OCR"), "v5 应识别出 OCR 字样");
    assert!(all_text.contains("2026"), "v5 应识别出 2026");
    assert!(results.iter().all(|(_, l)| l.score > 0.7), "v5 识别置信度应 > 0.7");
});

smoke!(yoloe_runtime_text_prompts, "YOLOE 运行时文本提示（文本编码器 + pe 检测器）", {
    let engine = rust_onnx_infer::engines::yolo_e_runtime::YoloERuntimeEngine::new(
        find_any("yoloe_rt_encoder.onnx"),
        find_any("yoloe_rt_pe_detector.onnx"),
        cpu(),
    )?;
    let mut engine = engine;
    engine
        .attach_text_encoder(
            find_any("yoloe_text_encoder.onnx"),
            find_any("yoloe_tpe_head.onnx"),
            find_any("clip_tokenizer.json"),
        )
        .unwrap();
    engine.set_text_prompts(&["person", "bus"]).unwrap();
    let image = Image::load(find_file(&PERSON_IMAGES)?)?;
    let results = engine.predict_without_sahi(&image)?;
    eprintln!("文本提示 [person, bus] 检出 {} 个实例", results.len());
    for r in results.iter().take(6) {
        eprintln!("  {r}");
    }
    assert!(results.len() >= 4, "bus.jpg 应检出 ≥4 个实例（4 人 + bus）");
    assert!(
        results.iter().any(|r| r.class_name() == "person")
            && results.iter().any(|r| r.class_name() == "bus"),
        "应同时包含 person 与 bus"
    );
});

smoke!(ocr_pipeline_det_rec, "OCR 流水线（PP-OCRv4 det + rec + CTC）", {
    let pipeline = rust_onnx_infer::engines::ocr::OcrPipeline::new(
        find_any("ch_PP-OCRv4_det_infer.onnx"),
        find_any("ch_PP-OCRv4_rec_infer.onnx"),
        find_any("ppocr_keys_v1.txt"),
        cpu(),
    )?;
    let image = Image::load(find_any("ocr_test.png"))?;
    let results = pipeline.predict(&image)?;
    eprintln!("识别出 {} 行", results.len());
    let all_text: String = results.iter().map(|(_, l)| l.text.clone()).collect::<Vec<_>>().join(" ");
    eprintln!("文本: {all_text}");
    assert!(all_text.contains("OCR"), "应识别出 OCR 字样");
    assert!(results.iter().all(|(_, l)| l.score > 0.7), "识别置信度应 > 0.7");
});