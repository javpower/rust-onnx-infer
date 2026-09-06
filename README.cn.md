<div align="center">

# rust-onnx-infer

**一个 crate，40+ 视觉引擎，零原生依赖。**

基于 ONNX Runtime 的生产级纯 Rust 推理库 —— 从 YOLO 检测、SAM 分割到 Grounding DINO
开放词表、BiRefNet 抠图、ArcFace 人脸识别、RoMaV2 特征匹配，40+ 引擎共享同一套
同步/异步 API，编译产物为单个自包含可执行文件。

[![crates.io](https://img.shields.io/crates/v/rust-onnx-infer?color=orange&logo=rust)](https://crates.io/crates/rust-onnx-infer)
[![docs.rs](https://img.shields.io/docsrs/rust-onnx-infer?logo=docsdotrs)](https://docs.rs/rust-onnx-infer)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT_OR_Apache--2.0-blue.svg)](#许可证)
[![Rust](https://img.shields.io/badge/rust-1.88%2B-dea584?logo=rust)](https://www.rust-lang.org)

[English](README.md) | 简体中文

</div>

---

## 亮点

- **静态链接 ONNX Runtime** —— 基于 `ort` + 预编译静态库，产物为完全自包含的可执行文件，
  无需 `LD_LIBRARY_PATH`、无需环境配置、无运行时下载。
- **纯 Rust 图像处理栈** —— `image` + `fast_image_resize` + 内置颜色空间/形态学/阈值等原语，
  **无 OpenCV**、无 C++ 工具链、无跨编译负担。
- **一套 API 打通所有引擎** —— 同步 `predict` / `predict_batch` + tokio 原生
  `predict_async` / `predict_batch_async`，40+ 引擎接口完全一致。
- **类别标签零配置** —— 自动从 ONNX 元数据解析类别名（YOLO `names` / `labels` /
  `categories`，或逗号分隔格式）。
- **内置 SAHI 切片推理** —— 大图小目标场景提升召回，支持 GREEDYNMM / NMM / NMS 合并
  与 IoU / IOS 度量。
- **GPU 加速 + 优雅回退** —— macOS 启用 CoreML（Apple Silicon 可走 ANE/NPU），CUDA 按
  feature 启用；Qualcomm QNN、DirectML、OpenVINO、Android NNAPI、华为 CANN 均可通过
  `ort` feature 开启，不可用时自动回退 CPU。

## 安装

```bash
cargo add rust-onnx-infer
```

或在 `Cargo.toml` 中添加：

```toml
[dependencies]
rust-onnx-infer = "0.1"
```

## 快速上手

```rust
use rust_onnx_infer::{core::factory::create_detection_engine, DeviceType, Image};

let image = Image::load("input.jpg")?;
let engine = create_detection_engine("yolo11n.onnx", DeviceType::Cpu, 0.5, 0.45)?;
let detections = engine.predict(&image)?;
for d in &detections {
    println!("{d}");
}
```

异步只需一行（tokio）：

```rust
let out = engine.predict_async(&img).await?;
```

运行完整示例：

```bash
cargo run --example onnx_inference_example
```

## 支持的模型

| 类别 | 说明 |
|---|---|
| **图像分类** | YOLO-CLS（pixel/255）、ImageNet 系（ResNet/MobileNet/EfficientNet/ViT，mean/std 归一化）、自定义归一化 |
| **目标检测** | YOLOv5/v8/v9/v10/v11/v26（自动识别 End2End 与传统布局）、RT-DETR、DETR、RF-DETR |
| **实例分割** | YOLO-Seg（传统 + End2End）、RF-DETR-Seg、YOLOE（运行时视觉提示 / 运行时文本提示 / 烘焙提示） |
| **交互式分割** | SAM、SAM2（单文件合并 ONNX） |
| **开放词表** | Grounding DINO（BERT 分词）、Grounded-SAM（DINO + SAM2 流水线）、DART v2（SAM3 骨干，文本/视觉概念提示，掩码输出） |
| **显著性抠图** | BiRefNet（soft alpha 输出） |
| **超分 / 增强** | Real-ESRGAN（2x/4x 重叠分块）、去噪（DnCNN）、低光增强（Zero-DCE）、去雾（DehazeFormer）、多步增强流水线 |
| **特征匹配** | LightGlue（含批量）、DeDoDe-G、LoMa-R、RoMaV2（密集对应场） |
| **姿态估计** | YOLOv8/11/26-Pose（COCO 17 点，传统+End2End）、RTMO 实时姿态（一阶段 SimCC） |
| **人脸** | YuNet 检测（5 地标）、SFace/ArcFace 识别（512/128 维）、年龄性别、8 类表情、106 精细关键点、活体检测（CDCN 分割式） |
| **文字识别** | PaddleOCR v4/v5 det+rec 流水线（DBNet 检测 + SVTR 识别 + CTC 解码，中英文） |
| **深度估计** | Depth Anything V2 / MiDaS（相对深度图，原图分辨率输出） |
| **风格迁移** | fast-neural-style（candy/mosaic 等，任意尺寸，动/静态自适应） |
| **行人重识别** | OSNet-x1.0（512 维 embedding + 图库检索） |
| **表格识别** | SLANet-plus（结构 token + 单元框，HTML 输出，配合 OCR 回填文本） |
| **旋转框检测** | YOLOv8/11-OBB（传统）+ YOLO26-OBB（End2End），旋转 IoU NMS，DOTA 15 类 |
| **语义分割** | SegFormer-B0（ADE20K 150 类），类别直方图 + 调色板叠加 |
| **人体解析** | SegFormer-B2 人衣 18 类（帽/发/上衣/裤/鞋/四肢部件占比） |
| **人像分割** | PP-HumanSeg / MODNet（实时 alpha 抠人，BGRA 输出） |
| **去模糊** | NafNet（512 对齐输入，输出同尺寸） |
| **图像质量** | QualityAssessor 纯算法（清晰度/亮度/对比度/噪声）+ FIQA 深度 IQA 挂载（352 输入） |
| **二维码检测+解码** | WeChat QR Detector（SSD 先验）+ rqrr 内容解码 |
| **动作识别** | ST-GCN 骨架动作分类（COCO17 帧序列输入，配合姿态引擎） |
| **姿态规则** | 零模型几何规则（躺卧/站立/举手判别） |
| **手势分类** | 手部 21 点纯几何规则（握拳/手掌/胜利/点赞等） |
| **车牌** | mnet 检测（四角透视矫正）+ LPRNet 识别（中国单行/绿牌） |
| **人脸精细关键点** | insightface 2d106det（106 点，YuNet 串联） |
| **手部** | MediaPipe Palm 检测（≤4 手）+ RTMPose-hand 21 关键点 |
| **全身关键点** | RTMPose-m WholeBody 133 点（body17+foot6+face68+hand42） |

## 模块结构

| 模块 | 内容 |
|---|---|
| `core` | 引擎接口 / 基础引擎 / 会话工厂 / 运行配置 / 引擎工厂 / 异步批处理优化器 |
| `engines` | 各模型引擎实现 |
| `sahi` | 切片推理：Slicer / Postprocess / Adapters / SlicedPredictor |
| `util` | MattingUtils（抠图结果合成） |
| `model` | Detection / Segmentation / ClassificationResult / MatchResult / MattingResult 等结果类型 |
| `imaging` | Image / resize / cvtColor / threshold / morphologyEx / boundingRect / FloatMask 等图像原语 |

## 构建说明

- 首次构建时 `ort` 会自动下载 ONNX Runtime 预编译**静态库**（需网络），之后离线可重复构建。
- 项目内 `.cargo/config.toml` 清空了 `PKG_CONFIG_PATH` / `PKG_CONFIG_LIBDIR`：防止 ort-sys
  通过 pkg-config 找到系统的 onnxruntime 动态库，确保静态链接、产物自包含。
- 加速后端（Execution Provider）：默认 CPU；macOS 启用 **CoreML**（Apple Silicon 可走
  ANE/NPU）；**CUDA**（NVIDIA GPU）按 feature 启用。其他 NPU 后端（Qualcomm QNN、
  Android NNAPI、DirectML、OpenVINO、华为 CANN）在 `ort` crate 中均有对应 feature，
  启用后 `DeviceType` 枚举与工厂入口即已就绪，不可用时自动回退 CPU。
- 所有模型的 NMS/CTC 解码/坐标还原等后处理在 CPU 纯 Rust 执行，EP 只加速模型前向。
- 产物为自包含二进制，运行时仅依赖 macOS 系统自带框架，无 OpenCV / ONNX Runtime
  动态库依赖（可用 `otool -L target/release/examples/onnx_inference_example` 验证）。

## 模型目录（`models/`）

所有引擎所需模型按**引擎名子目录**组织（`testmodels/` 仅为测试散件）：

| 子目录 | 内容 |
|---|---|
| `classification/` | ConvNeXt-tiny |
| `detection/` | yolov8n / yolo26n / rtdetr-l / rfdetr_base |
| `segmentation/` | yolov8n-seg / yolo26n-seg / rfdetr_seg_small |
| `yoloe/` | 11s/26n 文本烘焙版、rt 编码器+检测器、文本编码器（text_encoder/tpe_head/clip_tokenizer） |
| `sam/` `sam2/` | mobile_sam、sam2_fused |
| `grounding_dino/` | dinor50 tiny + BERT tokenizer |
| `pose/` `pose_rt/` | yolov8n-pose、yolo26n-pose、rtmo_s |
| `face_detection/` `face_recognition/` `face_attribute/` `face_liveness/` `face_landmark106/` | yunet、sface、age_gender+emotion、CDCN 活体、2d106det |
| `hand/` `wholebody/` | palm 检测 + rtmpose-hand 21 点、rtmpose-wholebody 133 点 |
| `obb/` | yolov8n-obb / yolo26n-obb |
| `semantic_segmentation/` `human_parsing/` `portrait_matting/` | segformer b0/b2、pp_humanseg |
| `ocr/` | PP-OCRv4/v5 det+rec+字典 |
| `table_recognition/` `reid/` | slanet-plus、osnet_x1_0 |
| `deblur/` `depth/` `style_transfer/` `image_quality/` `qr_detector/` | nafnet、depth_anything_v2、candy/mosaic、FIQA、wechat_qr |
| `birefnet/` `real_esrgan/` `image_enhance/` `lightglue/` `dedode/` `lomar/` `roma_v2/` | 对应模型 |
| `dart/` | DART v2 全套（骨干 1.8GB，SAM License） |
| `test_images/` | 各引擎共用测试图 |

## 测试

测试无需任何标志：`cargo test` 自动运行全部用例，模型/测试资产缺失的用例自动 SKIP
（控制台打印缺失清单）；设 `MODEL_TESTS=1` 可强制缺失即失败（CI 用）。

模型路径默认搜索 `models/` 与 `testmodels/` 两棵树，也可用 `MODEL_DIR=models/<dir>`、
`TESTMODELS_DIR=models/<dir>`、`DART_MODELS_DIR=models/dart` 指定。

## 已知限制

- 手部 21 点定位的 MediaPipe 官方模型仅 tflite 格式，库内使用 RTMPose-hand 替代（已完整支持）
- OBB 的 SAHI 合并以轴对齐近似（官方 sahi 不支持旋转框），精确角度场景请关闭 SAHI
- DART 引擎内存约 3.5GB，CPU 单类推理约 45s（推荐 GPU EP）

## 许可证

基于 [MIT License](LICENSE-MIT) 或 [Apache License 2.0](LICENSE-APACHE) 双协议开源，
使用者可任选其一。
