<div align="center">

# rust-onnx-infer

**One crate. 40+ vision engines. Zero native dependencies.**

Production-grade ONNX Runtime inference in pure Rust — from YOLO detection and SAM segmentation
to Grounding DINO, BiRefNet matting, ArcFace face recognition and RoMaV2 matching.
Every engine behind one unified sync/async API, compiled into a single self-contained binary.

[![crates.io](https://img.shields.io/crates/v/rust-onnx-infer?color=orange&logo=rust)](https://crates.io/crates/rust-onnx-infer)
[![docs.rs](https://img.shields.io/docsrs/rust-onnx-infer?logo=docsdotrs)](https://docs.rs/rust-onnx-infer)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT_OR_Apache--2.0-blue.svg)](#license)
[![Rust](https://img.shields.io/badge/rust-1.88%2B-dea584?logo=rust)](https://www.rust-lang.org)

[English](README.md) | [简体中文](README.cn.md)

</div>

---

## Highlights

- **Statically linked ONNX Runtime** — built on `ort` with prebuilt static libraries, so the
  artifact is a fully self-contained binary. No `LD_LIBRARY_PATH`, no environment setup, no
  runtime downloads.
- **Pure-Rust image stack** — `image` + `fast_image_resize` + built-in color-space,
  morphology and thresholding primitives. **No OpenCV**, no C++ toolchain, nothing to cross-compile.
- **One unified API for everything** — sync `predict` / `predict_batch` and tokio-native
  `predict_async` / `predict_batch_async`, identical across all 40+ engines.
- **Zero-config class labels** — category names parsed automatically from ONNX metadata
  (YOLO `names` / `labels` / `categories`, or comma-separated lists).
- **Built-in SAHI** — sliced inference for small objects in large images, with GREEDYNMM /
  NMM / NMS merging and IoU / IOS metrics.
- **GPU acceleration with graceful fallback** — CoreML on Apple Silicon (ANE/NPU), CUDA via
  feature flag; Qualcomm QNN, DirectML, OpenVINO, Android NNAPI and Huawei CANN all available
  through `ort` features. Unavailable providers fall back to CPU automatically.

## Installation

```bash
cargo add rust-onnx-infer
```

or add it to your `Cargo.toml`:

```toml
[dependencies]
rust-onnx-infer = "0.1"
```

## Quick start

```rust
use rust_onnx_infer::{core::factory::create_detection_engine, DeviceType, Image};

let image = Image::load("input.jpg")?;
let engine = create_detection_engine("yolo11n.onnx", DeviceType::Cpu, 0.5, 0.45)?;
let detections = engine.predict(&image)?;
for d in &detections {
    println!("{d}");
}
```

Async is a one-liner away (tokio):

```rust
let out = engine.predict_async(&img).await?;
```

Run the full example:

```bash
cargo run --example onnx_inference_example
```

## Supported models

| Category | Details |
|---|---|
| **Image classification** | YOLO-CLS (pixel/255), ImageNet-style (ResNet / MobileNet / EfficientNet / ViT, mean/std normalization), custom normalization |
| **Object detection** | YOLOv5 / v8 / v9 / v10 / v11 / v26 (auto-detects End2End vs. legacy layout), RT-DETR, DETR, RF-DETR |
| **Instance segmentation** | YOLO-Seg (legacy + End2End), RF-DETR-Seg, YOLOE (runtime visual prompts / runtime text prompts / baked prompts) |
| **Interactive segmentation** | SAM, SAM2 (single-file merged ONNX) |
| **Open vocabulary** | Grounding DINO (BERT tokenizer), Grounded-SAM (DINO + SAM2 pipeline), DART v2 (SAM3 backbone, text / visual-concept prompts, mask output) |
| **Saliency matting** | BiRefNet (soft alpha output) |
| **Super-resolution & enhancement** | Real-ESRGAN (2x/4x overlapping tiles), denoising (DnCNN), low-light enhancement (Zero-DCE), dehazing (DehazeFormer), multi-step enhancement pipeline |
| **Feature matching** | LightGlue (incl. batched), DeDoDe-G, LoMa-R, RoMaV2 (dense correspondence field) |
| **Pose estimation** | YOLOv8/11/26-Pose (COCO 17 keypoints, legacy + End2End), RTMO real-time pose (one-stage SimCC) |
| **Face** | YuNet detection (5 landmarks), SFace / ArcFace recognition (512/128-d), age & gender, 8-class expression, 106 fine-grained landmarks, liveness detection (CDCN) |
| **OCR** | PaddleOCR v4/v5 det+rec pipeline (DBNet detection + SVTR recognition + CTC decoding, Chinese & English) |
| **Depth estimation** | Depth Anything V2 / MiDaS (relative depth map, output at original resolution) |
| **Style transfer** | fast-neural-style (candy / mosaic etc., arbitrary size, adaptive to dynamic / static shapes) |
| **Person re-identification** | OSNet-x1.0 (512-d embedding + gallery retrieval) |
| **Table recognition** | SLANet-plus (structure tokens + cell boxes, HTML output, text backfilled via OCR) |
| **Rotated box detection (OBB)** | YOLOv8/11-OBB (legacy) + YOLO26-OBB (End2End), rotated-IoU NMS, DOTA 15 classes |
| **Semantic segmentation** | SegFormer-B0 (ADE20K 150 classes), class histogram + palette overlay |
| **Human parsing** | SegFormer-B2 person & clothing, 18 classes (hat / hair / upper / lower / shoes + limb part ratios) |
| **Portrait segmentation** | PP-HumanSeg / MODNet (real-time alpha matting, BGRA output) |
| **Deblurring** | NafNet (512-aligned input, same-size output) |
| **Image quality** | QualityAssessor pure algorithms (sharpness / brightness / contrast / noise) + FIQA deep IQA (352 input) |
| **QR code detection + decoding** | WeChat QR Detector (SSD prior) + rqrr content decoding |
| **Action recognition** | ST-GCN skeleton-based action classification (COCO17 frame sequences, pairs with the pose engine) |
| **Pose rules** | Model-free geometric rules (lying / standing / arms raised) |
| **Gesture classification** | Pure geometry on 21 hand keypoints (fist / palm / victory / thumbs-up etc.) |
| **License plate** | mnet detection (4-corner perspective correction) + LPRNet recognition (China single-line / green plates) |
| **Fine face landmarks** | insightface 2d106det (106 points, chained after YuNet) |
| **Hand** | MediaPipe Palm detection (up to 4 hands) + RTMPose-hand 21 keypoints |
| **Whole-body keypoints** | RTMPose-m WholeBody 133 points (body17 + foot6 + face68 + hand42) |

## Module layout

| Module | Contents |
|---|---|
| `core` | Engine traits / base engine / session factory / runtime config / engine factory / async batch optimizer |
| `engines` | Per-model engine implementations |
| `sahi` | Sliced inference: Slicer / Postprocess / Adapters / SlicedPredictor |
| `util` | MattingUtils (compositing matting results) |
| `model` | Result types: Detection / Segmentation / ClassificationResult / MatchResult / MattingResult |
| `imaging` | Image primitives: Image / resize / cvtColor / threshold / morphologyEx / boundingRect / FloatMask |

## Building

- On the first build, `ort` automatically downloads prebuilt ONNX Runtime **static libraries**
  (network required). After that, builds repeat fully offline.
- The repo's `.cargo/config.toml` clears `PKG_CONFIG_PATH` / `PKG_CONFIG_LIBDIR` so that
  `ort-sys` cannot pick up a system onnxruntime shared library via pkg-config — guaranteeing
  static linking and self-contained artifacts.
- Acceleration backends (Execution Providers): CPU by default; **CoreML** enabled on macOS
  (Apple Silicon can use the ANE/NPU); **CUDA** enabled via feature flag. Other NPU backends
  (Qualcomm QNN, Android NNAPI, DirectML, OpenVINO, Huawei CANN) have corresponding `ort`
  features — once enabled, the `DeviceType` enum and factory entry points are ready, and
  unavailable providers fall back to CPU automatically.
- All post-processing (NMS / CTC decoding / coordinate restoration) runs in pure Rust on the
  CPU; execution providers only accelerate the model forward pass.
- The artifact is self-contained: at runtime it only depends on OS-bundled frameworks, with no
  OpenCV / ONNX Runtime dynamic-library dependency (verify with
  `otool -L target/release/examples/onnx_inference_example`).

## Model directory (`models/`)

Models for every engine are organized in **per-engine subdirectories** (`testmodels/` holds
loose test assets only):

| Subdirectory | Contents |
|---|---|
| `classification/` | ConvNeXt-tiny |
| `detection/` | yolov8n / yolo26n / rtdetr-l / rfdetr_base |
| `segmentation/` | yolov8n-seg / yolo26n-seg / rfdetr_seg_small |
| `yoloe/` | 11s / 26n text-baked, rt encoder + detector, text encoder (text_encoder / tpe_head / clip_tokenizer) |
| `sam/` `sam2/` | mobile_sam, sam2_fused |
| `grounding_dino/` | dinor50 tiny + BERT tokenizer |
| `pose/` `pose_rt/` | yolov8n-pose, yolo26n-pose, rtmo_s |
| `face_detection/` `face_recognition/` `face_attribute/` `face_liveness/` `face_landmark106/` | yunet, sface, age_gender + emotion, CDCN liveness, 2d106det |
| `hand/` `wholebody/` | palm detection + rtmpose-hand 21 points, rtmpose-wholebody 133 points |
| `obb/` | yolov8n-obb / yolo26n-obb |
| `semantic_segmentation/` `human_parsing/` `portrait_matting/` | segformer b0/b2, pp_humanseg |
| `ocr/` | PP-OCRv4/v5 det + rec + dictionaries |
| `table_recognition/` `reid/` | slanet-plus, osnet_x1_0 |
| `deblur/` `depth/` `style_transfer/` `image_quality/` `qr_detector/` | nafnet, depth_anything_v2, candy / mosaic, FIQA, wechat_qr |
| `birefnet/` `real_esrgan/` `image_enhance/` `lightglue/` `dedode/` `lomar/` `roma_v2/` | corresponding models |
| `dart/` | Full DART v2 set (1.8 GB backbone, SAM License) |
| `test_images/` | Shared test images |

## Testing

Tests need no special flags: `cargo test` runs the full suite, and cases whose models or test
assets are missing SKIP automatically (the missing list is printed to the console). Set
`MODEL_TESTS=1` to make missing assets a hard failure (for CI).

Model paths are searched under `models/` and `testmodels/` by default, overridable via
`MODEL_DIR=models/<dir>`, `TESTMODELS_DIR=models/<dir>` and `DART_MODELS_DIR=models/dart`.

## Known limitations

- The official MediaPipe model for 21-point hand landmarks ships only in tflite format, so the
  library uses RTMPose-hand instead (fully supported).
- SAHI merging for OBB approximates with axis-aligned boxes (the upstream `sahi` library does
  not support rotated boxes); disable SAHI when precise angles matter.
- The DART engine uses ~3.5 GB of memory and takes ~45 s per class on CPU (a GPU EP is
  recommended).

## License

Dual-licensed under either the [MIT License](LICENSE-MIT) or the
[Apache License 2.0](LICENSE-APACHE), at your option.
