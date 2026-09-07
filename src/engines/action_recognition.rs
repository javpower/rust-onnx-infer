//! 骨架动作识别引擎（ST-GCN 系图卷积模型，骨架关键点序列 -> 动作类别）。
//!
//! **模型来源**（PINTO Model Zoo #248，[kenziyuliu/MS-G3D](https://github.com/kenziyuliu/MS-G3D)
//! 的 Kinetics-Skeleton 预训练权重，PyTorch 1.10 导出，PINTO0309 转换分发）：
//!
//! ```text
//! https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/248_MS-G3D/resources.tar.gz
//! 内含 msg3d_kinetics_joint_30x18x2.onnx（本引擎使用，另下载为 testmodels/action_stgcn.onnx）
//! ```
//!
//! **实测签名**（`cargo run --release --example model_probe -- testmodels/action_stgcn.onnx`）：
//!
//! ```text
//! 输入  name='0'    shape=[1, 3, 30, 18, 2]  type=Float32
//! 输出  name='1893' shape=[1, 400]           type=Float32
//! ```
//!
//! **输入五维布局 `[N=1, C=3, T=30, V=18, M=2]`**（行主序，ST-GCN 系标准 CNTVM 布局）：
//!
//! | 维度 | 含义 | 取值 |
//! |---|---|---|
//! | N | 批量 | 1（固定导出） |
//! | C | 通道：`[x, y, score]` | x/y 归一化坐标，score ∈ [0,1] |
//! | T | 时间帧数 | 30（固定导出） |
//! | V | 关键点数 | 18（OpenPose COCO-18，非 COCO-17） |
//! | M | 人数 | 2（单人场景第 2 人全零） |
//!
//! **关键点顺序映射**：模型使用 ST-GCN Kinetics-Skeleton 约定的 OpenPose COCO-18
//! （在 COCO-17 基础上于索引 1 处插入 `neck=双肩中点`，肩中点分数取两侧较小值；
//! 其余 17 点一一对应，眼睛/耳朵注意左右互换：COCO 顺序为 L 在前，OpenPose 为 R 在前）：
//! `op[0]=nose, op[1]=neck(合成), op[2..7]=R/L 肩肘腕, op[8..13]=R/L 髋膝踝,
//! op[14..17]=R/L 眼耳`。映射表见 [`COCO17_TO_OPENPOSE18`]。
//!
//! **坐标归一化**：训练数据（ST-GCN `kinetics_gendata.py` 流程）使用画面归一化坐标：
//! `x/W - 0.5`、`-(y/H - 0.5)`（y 翻转为数学方向）、score 无效处坐标置 0。
//! [`ActionRecognitionEngine::classify`] 只接收关键点像素序列、不含画面尺寸，
//! 因此改用**序列包围盒归一化**近似：以全部有效点的包围盒中心为原点、包围盒
//! 长边映射到 0.6（典型监控画面中人体占画面高度的比例），再同样翻转 y 轴、
//! 无效点置 0。该近似与训练分布存在尺度偏差，接入真实场景时建议按实际画面
//! 尺寸归一化（见 `classify` 注释内的说明）。
//!
//! **帧数补齐规则**（对齐 ST-GCN 官方 feeder）：帧数不足 T=30 时**尾部补全零帧**
//! （x=y=score=0，与训练 auto_pading 一致）；超过 30 帧时取**中间 30 帧窗口**
//! （确定性推理路径）。
//!
//! **输出**：400 类 Kinetics-400 logits，softmax 后即各类概率。标签表
//! [`KINETICS400_LABELS`] 取自 ST-GCN 官方 `resource/kinetics_skeleton/label_name.txt`
//! （与训练数据集 label_index 同序，按类名排序）。**注意：Kinetics-400 没有
//! "falling（跌倒）" 类**，本模型用于通用动作识别；跌倒报警请配合
//! [`super::pose_rules`] 的几何规则（躯干角度 / 躺卧判定）使用。
//!
//! **与 trait 的关系**：[`crate::core::engine::OnnxInferenceEngine`] 面向
//! "单图输入" 引擎（`predict(&Image)`），而本引擎消费**关键点序列**（由库内
//! 姿态引擎逐帧输出，如 [`super::pose::PoseEngine`] / [`super::pose_rt`]），
//! 主入口是 [`ActionRecognitionEngine::classify`]。为满足统一接口，
//! `impl_engine_forward!` 的 `Output = ActionPrediction`，但 `predict(&Image)`
//! 显式返回 `Unsupported` 错误（不做无意义的图像透传）。

use crate::core::base::{BaseOnnxEngine, TensorOutput};
use crate::core::device_type::DeviceType;
use crate::error::{Result, VisionError};
use crate::model::Keypoint;

/// 单次动作识别结果。
#[derive(Debug, Clone, PartialEq)]
pub struct ActionPrediction {
    /// 最可能动作的标签名（Kinetics-400 标签表）。
    pub label: String,
    /// 最可能动作的类别索引（0..400，对应 [`KINETICS400_LABELS`] 下标）。
    pub label_id: usize,
    /// 全部类别的 softmax 概率（长度 = 模型类别数，通常 400）。
    pub scores: Vec<f32>,
}

/// Kinetics-400 标签表（ST-GCN kinetics_skeleton/label_name.txt 顺序，
/// 与训练数据 label_index 一致；模型输出 logits 的第 i 维对应第 i 个标签）。
pub const KINETICS400_LABELS: [&str; 400] = [
    "abseiling", "air drumming", "answering questions", "applauding", "applying cream",
    "archery", "arm wrestling", "arranging flowers", "assembling computer", "auctioning",
    "baby waking up", "baking cookies", "balloon blowing", "bandaging", "barbequing",
    "bartending", "beatboxing", "bee keeping", "belly dancing", "bench pressing",
    "bending back", "bending metal", "biking through snow", "blasting sand", "blowing glass",
    "blowing leaves", "blowing nose", "blowing out candles", "bobsledding", "bookbinding",
    "bouncing on trampoline", "bowling", "braiding hair", "breading or breadcrumbing",
    "breakdancing", "brush painting", "brushing hair", "brushing teeth", "building cabinet",
    "building shed", "bungee jumping", "busking", "canoeing or kayaking", "capoeira",
    "carrying baby", "cartwheeling", "carving pumpkin", "catching fish",
    "catching or throwing baseball", "catching or throwing frisbee",
    "catching or throwing softball", "celebrating", "changing oil", "changing wheel",
    "checking tires", "cheerleading", "chopping wood", "clapping", "clay pottery making",
    "clean and jerk", "cleaning floor", "cleaning gutters", "cleaning pool", "cleaning shoes",
    "cleaning toilet", "cleaning windows", "climbing a rope", "climbing ladder",
    "climbing tree", "contact juggling", "cooking chicken", "cooking egg",
    "cooking on campfire", "cooking sausages", "counting money", "country line dancing",
    "cracking neck", "crawling baby", "crossing river", "crying", "curling hair",
    "cutting nails", "cutting pineapple", "cutting watermelon", "dancing ballet",
    "dancing charleston", "dancing gangnam style", "dancing macarena", "deadlifting",
    "decorating the christmas tree", "digging", "dining", "disc golfing", "diving cliff",
    "dodgeball", "doing aerobics", "doing laundry", "doing nails", "drawing",
    "dribbling basketball", "drinking", "drinking beer", "drinking shots", "driving car",
    "driving tractor", "drop kicking", "drumming fingers", "dunking basketball", "dying hair",
    "eating burger", "eating cake", "eating carrots", "eating chips", "eating doughnuts",
    "eating hotdog", "eating ice cream", "eating spaghetti", "eating watermelon",
    "egg hunting", "exercising arm", "exercising with an exercise ball", "extinguishing fire",
    "faceplanting", "feeding birds", "feeding fish", "feeding goats", "filling eyebrows",
    "finger snapping", "fixing hair", "flipping pancake", "flying kite", "folding clothes",
    "folding napkins", "folding paper", "front raises", "frying vegetables",
    "garbage collecting", "gargling", "getting a haircut", "getting a tattoo",
    "giving or receiving award", "golf chipping", "golf driving", "golf putting",
    "grinding meat", "grooming dog", "grooming horse", "gymnastics tumbling", "hammer throw",
    "headbanging", "headbutting", "high jump", "high kick", "hitting baseball", "hockey stop",
    "holding snake", "hopscotch", "hoverboarding", "hugging", "hula hooping", "hurdling",
    "hurling (sport)", "ice climbing", "ice fishing", "ice skating", "ironing",
    "javelin throw", "jetskiing", "jogging", "juggling balls", "juggling fire",
    "juggling soccer ball", "jumping into pool", "jumpstyle dancing", "kicking field goal",
    "kicking soccer ball", "kissing", "kitesurfing", "knitting", "krumping", "laughing",
    "laying bricks", "long jump", "lunge", "making a cake", "making a sandwich", "making bed",
    "making jewelry", "making pizza", "making snowman", "making sushi", "making tea",
    "marching", "massaging back", "massaging feet", "massaging legs",
    "massaging person's head", "milking cow", "mopping floor", "motorcycling",
    "moving furniture", "mowing lawn", "news anchoring", "opening bottle", "opening present",
    "paragliding", "parasailing", "parkour", "passing American football (in game)",
    "passing American football (not in game)", "peeling apples", "peeling potatoes",
    "petting animal (not cat)", "petting cat", "picking fruit", "planting trees", "plastering",
    "playing accordion", "playing badminton", "playing bagpipes", "playing basketball",
    "playing bass guitar", "playing cards", "playing cello", "playing chess",
    "playing clarinet", "playing controller", "playing cricket", "playing cymbals",
    "playing didgeridoo", "playing drums", "playing flute", "playing guitar",
    "playing harmonica", "playing harp", "playing ice hockey", "playing keyboard",
    "playing kickball", "playing monopoly", "playing organ", "playing paintball",
    "playing piano", "playing poker", "playing recorder", "playing saxophone",
    "playing squash or racquetball", "playing tennis", "playing trombone", "playing trumpet",
    "playing ukulele", "playing violin", "playing volleyball", "playing xylophone",
    "pole vault", "presenting weather forecast", "pull ups", "pumping fist", "pumping gas",
    "punching bag", "punching person (boxing)", "push up", "pushing car", "pushing cart",
    "pushing wheelchair", "reading book", "reading newspaper", "recording music",
    "riding a bike", "riding camel", "riding elephant", "riding mechanical bull",
    "riding mountain bike", "riding mule", "riding or walking with horse", "riding scooter",
    "riding unicycle", "ripping paper", "robot dancing", "rock climbing",
    "rock scissors paper", "roller skating", "running on treadmill", "sailing",
    "salsa dancing", "sanding floor", "scrambling eggs", "scuba diving", "setting table",
    "shaking hands", "shaking head", "sharpening knives", "sharpening pencil", "shaving head",
    "shaving legs", "shearing sheep", "shining shoes", "shooting basketball",
    "shooting goal (soccer)", "shot put", "shoveling snow", "shredding paper",
    "shuffling cards", "side kick", "sign language interpreting", "singing", "situp",
    "skateboarding", "ski jumping", "skiing (not slalom or crosscountry)",
    "skiing crosscountry", "skiing slalom", "skipping rope", "skydiving", "slacklining",
    "slapping", "sled dog racing", "smoking", "smoking hookah", "snatch weight lifting",
    "sneezing", "sniffing", "snorkeling", "snowboarding", "snowkiting", "snowmobiling",
    "somersaulting", "spinning poi", "spray painting", "spraying", "springboard diving",
    "squat", "sticking tongue out", "stomping grapes", "stretching arm", "stretching leg",
    "strumming guitar", "surfing crowd", "surfing water", "sweeping floor",
    "swimming backstroke", "swimming breast stroke", "swimming butterfly stroke",
    "swing dancing", "swinging legs", "swinging on something", "sword fighting", "tai chi",
    "taking a shower", "tango dancing", "tap dancing", "tapping guitar", "tapping pen",
    "tasting beer", "tasting food", "testifying", "texting", "throwing axe", "throwing ball",
    "throwing discus", "tickling", "tobogganing", "tossing coin", "tossing salad",
    "training dog", "trapezing", "trimming or shaving beard", "trimming trees", "triple jump",
    "tying bow tie", "tying knot (not on a tie)", "tying tie", "unboxing", "unloading truck",
    "using computer", "using remote controller (not gaming)", "using segway", "vault",
    "waiting in line", "walking the dog", "washing dishes", "washing feet", "washing hair",
    "washing hands", "water skiing", "water sliding", "watering plants", "waxing back",
    "waxing chest", "waxing eyebrows", "waxing legs", "weaving basket", "welding", "whistling",
    "windsurfing", "wrapping present", "wrestling", "writing", "yawning", "yoga", "zumba",
];

/// OpenPose COCO-18 第 v 点对应的 COCO-17 索引；`None` 为 neck（双肩中点，合成）。
///
/// OpenPose 顺序：`0 nose, 1 neck, 2 R-shoulder, 3 R-elbow, 4 R-wrist,
/// 5 L-shoulder, 6 L-elbow, 7 L-wrist, 8 R-hip, 9 R-knee, 10 R-ankle,
/// 11 L-hip, 12 L-knee, 13 L-ankle, 14 R-eye, 15 L-eye, 16 R-ear, 17 L-ear`。
/// 注意左右语义按解剖学（画面视角相反）：COCO-17 的 left_xxx 映射到 OpenPose 的
/// L-xxx（索引 5/6/7/11/12/13/15/17）。
const COCO17_TO_OPENPOSE18: [Option<usize>; 18] = [
    Some(0),  // op0  nose
    None,     // op1  neck = mid(coco5, coco6)
    Some(6),  // op2  right_shoulder
    Some(8),  // op3  right_elbow
    Some(10), // op4  right_wrist
    Some(5),  // op5  left_shoulder
    Some(7),  // op6  left_elbow
    Some(9),  // op7  left_wrist
    Some(12), // op8  right_hip
    Some(14), // op9  right_knee
    Some(16), // op10 right_ankle
    Some(11), // op11 left_hip
    Some(13), // op12 left_knee
    Some(15), // op13 left_ankle
    Some(2),  // op14 right_eye
    Some(1),  // op15 left_eye
    Some(4),  // op16 right_ear
    Some(3),  // op17 left_ear
];

/// 归一化后人体包围盒长边对应的坐标跨度（近似训练分布中
/// 人体占画面尺寸的比例；见模块文档"坐标归一化"）。
const NORMALIZED_SPAN: f32 = 0.6;

/// 骨架动作识别引擎（ST-GCN 系 MS-G3D，Kinetics-400）。
pub struct ActionRecognitionEngine {
    /// 组合基类（会话管理 / 推理 / 标签）
    pub base: BaseOnnxEngine,

    /// 输入通道数 C（=3：x, y, score）
    num_channels: usize,
    /// 输入帧数 T（本模型固定 30）
    num_frames: usize,
    /// 关键点数 V（=18，OpenPose COCO-18）
    num_joints: usize,
    /// 人数 M（=2，单人场景第 2 人全零）
    num_persons: usize,
    /// 类别数（本模型 400）
    num_classes: usize,
}

impl ActionRecognitionEngine {
    /// 创建骨架动作识别引擎（模型路径 + 设备）。
    ///
    /// 输入/输出维度从模型读取（动态维度回退 `3/30/18/2` 与 400），
    /// 标签固定使用内置 [`KINETICS400_LABELS`]（该 ONNX 元数据不含标签）。
    pub fn new(model_path: impl AsRef<std::path::Path>, device_type: DeviceType) -> Result<Self> {
        let base = BaseOnnxEngine::with_input_size(model_path, device_type, -1, -1)?;

        // 从会话读取 5D 输入签名（BaseOnnxEngine 的 NCHW 解析对 5D 输入无意义，
        // 这里自行解析；缺省回退 3/30/18/2）
        let (num_channels, num_frames, num_joints, num_persons) = {
            let session = base.session.lock().unwrap();
            match session.inputs().first() {
                Some(input) => match input.dtype() {
                    ort::value::ValueType::Tensor { shape, .. } if shape.len() == 5 => (
                        shrink_to_usize(shape[1], 3),
                        shrink_to_usize(shape[2], 30),
                        shrink_to_usize(shape[3], 18),
                        shrink_to_usize(shape[4], 2),
                    ),
                    _ => (3, 30, 18, 2),
                },
                None => (3, 30, 18, 2),
            }
        };

        // 类别数取自输出 shape[1]（[1, 400]），异常时回退标签表长度
        let num_classes = {
            let session = base.session.lock().unwrap();
            session
                .outputs()
                .first()
                .and_then(|o| match o.dtype() {
                    ort::value::ValueType::Tensor { shape, .. } if shape.len() == 2 && shape[1] > 0 => {
                        Some(shape[1] as usize)
                    }
                    _ => None,
                })
                .unwrap_or(KINETICS400_LABELS.len())
        };

        // 模型元数据无标签：写入内置标签表，使 trait labels()/get_label_name 可用
        let mut engine = ActionRecognitionEngine {
            base,
            num_channels,
            num_frames,
            num_joints,
            num_persons,
            num_classes,
        };
        let labels = KINETICS400_LABELS
            .iter()
            .take(engine.num_classes)
            .map(|s| s.to_string())
            .collect();
        engine.base.set_labels(labels);

        tracing::info!(
            "ActionRecognitionEngine initialized: layout=[1,{},T={},V={},M={}], classes={}, device={}",
            engine.num_channels,
            engine.num_frames,
            engine.num_joints,
            engine.num_persons,
            engine.num_classes,
            device_type.name()
        );
        Ok(engine)
    }

    /// 识别一段骨架关键点序列的动作。
    ///
    /// - `frames`：T 帧关键点序列，每帧为库内姿态引擎输出的 COCO 17 点
    ///   （像素坐标 + 置信度）；帧数超出模型窗口取中间帧，不足尾部补零帧
    /// - 返回：softmax 概率 + argmax 标签（[`ActionPrediction`]）
    pub fn classify(&self, frames: &[Vec<Keypoint>]) -> Result<ActionPrediction> {
        if frames.is_empty() {
            return Err(VisionError::invalid_argument(
                "classify requires at least one frame of keypoints",
            ));
        }

        // 1. 整理为 [1, C, T, V, M] 输入数据（映射 / 归一化 / 补帧）
        let data = build_input_data(frames, self.num_channels, self.num_frames, self.num_joints, self.num_persons);

        // 2. 推理
        let shape: Vec<i64> = vec![
            1,
            self.num_channels as i64,
            self.num_frames as i64,
            self.num_joints as i64,
            self.num_persons as i64,
        ];
        let input_tensor = ort::value::Tensor::from_array((shape, data))?;
        let outputs = self.base.run_multi_output(input_tensor)?;
        let output = Self::find_logits_output(&outputs)?;

        // 3. softmax + argmax
        let logits = output.as_f32()?;
        if logits.len() < self.num_classes {
            return Err(VisionError::inference(format!(
                "action model output has {} elements, expected {}",
                logits.len(),
                self.num_classes
            )));
        }
        let scores = softmax(&logits[..self.num_classes]);
        let label_id = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let label = KINETICS400_LABELS
            .get(label_id)
            .map(|s| s.to_string())
            .unwrap_or_else(|| label_id.to_string());

        tracing::debug!(
            "Action classified: label='{}' (id={}, prob={:.4}), frames={}",
            label,
            label_id,
            scores[label_id],
            frames.len()
        );
        Ok(ActionPrediction {
            label,
            label_id,
            scores,
        })
    }

    /// 从多个输出中定位 logits 张量（取元素数最多者，跳过元数据输出）。
    fn find_logits_output(outputs: &[TensorOutput]) -> Result<&TensorOutput> {
        if outputs.is_empty() {
            return Err(VisionError::inference("action model returned no outputs"));
        }
        let mut best = &outputs[0];
        for o in &outputs[1..] {
            if o.element_count() > best.element_count() {
                best = o;
            }
        }
        Ok(best)
    }

    /// 输入布局维度 (C, T, V, M)，供上层确认窗口长度 / 关键点约定。
    pub fn input_layout(&self) -> (usize, usize, usize, usize) {
        (
            self.num_channels,
            self.num_frames,
            self.num_joints,
            self.num_persons,
        )
    }

    /// 类别数（Kinetics-400 为 400）。
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }
}

// ==================== 预处理 ====================

/// 把关键点帧序列整理为模型输入数据（`[C, T, V, M]` 行主序，不含 batch 维）。
///
/// 步骤（详见模块文档）：
/// 1. 帧数 > T 时取中间窗口；< T 时尾部补零帧
/// 2. 逐帧把 COCO-17 映射为 OpenPose COCO-18（neck = 双肩中点，分数取较小侧）
/// 3. 全序列有效点包围盒归一化：中心移到原点、长边映射 [`NORMALIZED_SPAN`]，
///    y 轴翻转为数学方向（训练流程 `y = -(y - 0.5)`）；score <= 0 的点坐标置 0
fn build_input_data(
    frames: &[Vec<Keypoint>],
    channels: usize,
    t_frames: usize,
    v_joints: usize,
    m_persons: usize,
) -> Vec<f32> {
    let mut data = vec![0f32; channels * t_frames * v_joints * m_persons];

    // 1. 确定时间窗口（超长取中间 T 帧，不足尾部补零）
    let start = frames.len().saturating_sub(t_frames) / 2;
    let window: &[Vec<Keypoint>] = if frames.len() > t_frames {
        &frames[start..start + t_frames]
    } else {
        frames
    };

    // 2. 包围盒统计（只统计有效点，作用于整个窗口）
    let (mut xmin, mut ymin) = (f32::MAX, f32::MAX);
    let (mut xmax, mut ymax) = (f32::MIN, f32::MIN);
    let mut has_valid = false;
    for frame in window {
        for kp in frame.iter() {
            if kp.score > 0.0 {
                has_valid = true;
                xmin = xmin.min(kp.x);
                ymin = ymin.min(kp.y);
                xmax = xmax.max(kp.x);
                ymax = ymax.max(kp.y);
            }
        }
    }
    // 全无效序列保持全零输入（data 初始即 0）
    let (cx, cy, scale) = if has_valid {
        let cx = (xmin + xmax) / 2.0;
        let cy = (ymin + ymax) / 2.0;
        let long_side = (xmax - xmin).max(ymax - ymin).max(1e-6);
        (cx, cy, NORMALIZED_SPAN / long_side)
    } else {
        return data;
    };

    // 3. 逐帧映射 + 归一化，写入 person 0（person 1 保持全零）
    let stride_v_m = v_joints * m_persons;
    for (t, frame) in window.iter().enumerate() {
        let base_t = t * stride_v_m;

        // 3.1 neck：双肩中点（分数取较小侧；任一侧无效则 neck 无效）
        let (ls, rs) = (frame.get(5), frame.get(6));
        let neck = match (ls, rs) {
            (Some(a), Some(b)) if a.score > 0.0 && b.score > 0.0 => Some(Keypoint::new(
                (a.x + b.x) / 2.0,
                (a.y + b.y) / 2.0,
                a.score.min(b.score),
            )),
            _ => None,
        };

        // 3.2 写入 18 个 OpenPose 点
        for (v, coco_idx) in COCO17_TO_OPENPOSE18.iter().enumerate() {
            let (x, y, score) = match coco_idx {
                Some(idx) => match frame.get(*idx) {
                    Some(kp) if kp.score > 0.0 => (kp.x, kp.y, kp.score),
                    _ => (0.0, 0.0, 0.0),
                },
                None => match neck {
                    Some(kp) => (kp.x, kp.y, kp.score),
                    None => (0.0, 0.0, 0.0),
                },
            };
            // score==0 的点坐标保持 0（对齐训练预处理），有效点归一化 + y 翻转
            if score > 0.0 {
                // C 通道布局：C=0 -> x，C=1 -> y（翻转），C=2 -> score
                let layer = t_frames * v_joints * m_persons;
                data[base_t + v * m_persons] = (x - cx) * scale;
                data[layer + base_t + v * m_persons] = -(y - cy) * scale;
                data[2 * layer + base_t + v * m_persons] = score;
            }
        }
    }

    data
}

/// 数值稳定的 softmax（logits -> 概率分布）。
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::MIN, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

/// 维度收缩：静态正维度取原值，动态/非法回退默认值。
fn shrink_to_usize(dim: i64, default: usize) -> usize {
    if dim > 0 {
        dim as usize
    } else {
        default
    }
}

// ==================== 统一 trait 转发 ====================

crate::impl_engine_forward!(ActionRecognitionEngine, base, ActionPrediction,
    /// 单图推理（本引擎消费关键点序列而非图像，`classify` 为主入口；
    /// 为满足统一接口在此显式返回不支持）。
    fn predict(&self, _image: &crate::imaging::Image) -> Result<ActionPrediction> {
        Err(VisionError::Unsupported(
            "ActionRecognitionEngine consumes keypoint sequences via classify(), not images".into(),
        ))
    }
);

// ==================== 单元测试 ====================

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造一帧 COCO-17 关键点（只填给定索引，其余无效）。
    fn frame(points: &[(usize, f32, f32)], score: f32) -> Vec<Keypoint> {
        let mut kps: Vec<Keypoint> = (0..17).map(|_| Keypoint::new(0.0, 0.0, 0.0)).collect();
        for &(idx, x, y) in points {
            kps[idx] = Keypoint::new(x, y, score);
        }
        kps
    }

    /// 合成站立帧（近似真人比例，轻微摆动），坐标为像素。
    fn standing_frame(t: usize) -> Vec<Keypoint> {
        let sway = 3.0 * (t as f32 * 0.3).sin();
        frame(
            &[
                (0, 100.0 + sway, 70.0),   // nose
                (1, 96.0 + sway, 72.0),    // left_eye
                (2, 104.0 + sway, 72.0),   // right_eye
                (3, 92.0 + sway, 74.0),    // left_ear
                (4, 108.0 + sway, 74.0),   // right_ear
                (5, 90.0 + sway, 120.0),   // left_shoulder
                (6, 110.0 + sway, 120.0),  // right_shoulder
                (7, 70.0 + sway, 190.0),   // left_elbow
                (8, 130.0 + sway, 190.0),  // right_elbow
                (9, 65.0 + sway, 255.0),   // left_wrist
                (10, 135.0 + sway, 255.0), // right_wrist
                (11, 95.0 + sway, 300.0),  // left_hip
                (12, 105.0 + sway, 300.0), // right_hip
                (13, 90.0 + sway, 380.0),  // left_knee
                (14, 110.0 + sway, 380.0), // right_knee
                (15, 85.0 + sway, 455.0),  // left_ankle
                (16, 115.0 + sway, 455.0), // right_ankle
            ],
            0.95,
        )
    }

    /// 合成跌倒序列：前 12 帧站立，随后 y 骤降 + 躯干放平（躺地）。
    fn falling_sequence(t_frames: usize) -> Vec<Vec<Keypoint>> {
        let stand = standing_frame(0);
        // 躺地帧：躯干水平（肩 x≈90..110、髋 x≈275..285，同一 y 高度）
        let lying = frame(
            &[
                (0, 60.0, 290.0),
                (1, 56.0, 292.0),
                (2, 64.0, 292.0),
                (3, 52.0, 294.0),
                (4, 68.0, 294.0),
                (5, 90.0, 295.0),
                (6, 110.0, 305.0),
                (7, 70.0, 310.0),
                (8, 128.0, 300.0),
                (9, 55.0, 315.0),
                (10, 145.0, 305.0),
                (11, 275.0, 295.0),
                (12, 285.0, 305.0),
                (13, 340.0, 300.0),
                (14, 350.0, 310.0),
                (15, 400.0, 300.0),
                (16, 410.0, 310.0),
            ],
            0.95,
        );
        let mut frames = Vec::with_capacity(t_frames);
        for t in 0..t_frames {
            if t < 12 {
                frames.push(standing_frame(t));
            } else {
                // 12~18 帧：站立 -> 躺地线性插值（模拟跌倒过程），之后保持躺地
                let p = (((t - 11) as f32) / 6.0).min(1.0);
                let mut interpolated = Vec::with_capacity(17);
                for k in 0..17 {
                    let a = stand[k];
                    let b = lying[k];
                    interpolated.push(Keypoint::new(
                        a.x + (b.x - a.x) * p,
                        a.y + (b.y - a.y) * p,
                        a.score.max(b.score) * p.max(0.3),
                    ));
                }
                frames.push(interpolated);
            }
        }
        frames
    }

    #[test]
    fn test_coco17_to_openpose18_mapping() {
        let f = standing_frame(0);
        let data = build_input_data(&[f.clone(), f], 3, 30, 18, 2);
        let stride = 30 * 18 * 2; // T * V * M（C 层之间的偏移）

        // op[0] = nose（归一化后无需检查绝对值，只检查 score 通道 = 0.95）
        assert!((data[2 * stride] - 0.95).abs() < 1e-6, "op0 score");
        // op[1] = neck：x = 双肩中点，score = min(两侧)
        let neck_score = data[2 * stride + 2];
        assert!((neck_score - 0.95).abs() < 1e-6, "neck score");
        // neck 与左右肩 x 的归一化值之差对称（neck 是中点）
        let neck_x = data[2];
        let ls_x = data[5 * 2];
        let rs_x = data[2 * 2];
        assert!((neck_x - (ls_x + rs_x) / 2.0).abs() < 1e-5, "neck is midpoint");
        // op[7] = left_wrist（coco 9），op[4] = right_wrist（coco 10）：左右不互换
        // 站立帧左手腕 x=65 < 右手腕 x=135（画面左小右大），翻转不影响 x
        assert!(data[7 * 2] < data[4 * 2], "op7=LWrist < op4=RWrist in x");
        // 无效点：未提供的索引全零（person 1 整体全零）
        assert_eq!(data[1], 0.0, "person1 zeroed");
    }

    #[test]
    fn test_frame_padding_and_center_crop() {
        // 不足 30 帧：尾部补零帧
        let frames = vec![standing_frame(0); 5];
        let data = build_input_data(&frames, 3, 30, 18, 2);
        let frame_stride = 18 * 2; // V * M
        let t29 = 29 * frame_stride;
        assert_eq!(&data[t29..t29 + frame_stride], &[0f32; 36], "tail frames zero-padded");
        // 首帧有数据：用 score 通道验证（nose x 恰为包围盒中心，x 值为 0 属正常）
        assert!(data[2 * 30 * 18 * 2] != 0.0, "first frame has data");

        // 超过 30 帧：取中间窗口（40 帧 -> 从第 5 帧开始）
        let frames: Vec<Vec<Keypoint>> = (0..40).map(standing_frame).collect();
        let data = build_input_data(&frames, 3, 30, 18, 2);
        // 归一化基准应来自中间窗口 frames[5..35]（而非全部 40 帧），手工复算：
        let window = &frames[5..35];
        let (mut xmin, mut ymin) = (f32::MAX, f32::MAX);
        let (mut xmax, mut ymax) = (f32::MIN, f32::MIN);
        for f in window {
            for kp in f {
                if kp.score > 0.0 {
                    xmin = xmin.min(kp.x);
                    ymin = ymin.min(kp.y);
                    xmax = xmax.max(kp.x);
                    ymax = ymax.max(kp.y);
                }
            }
        }
        let cx = (xmin + xmax) / 2.0;
        let scale = NORMALIZED_SPAN / (xmax - xmin).max(ymax - ymin);
        // 窗口首帧 = 原第 5 帧，nose x = 100 + 3*sin(0.3*5)
        let expected_nose_x = (100.0 + 3.0 * (5.0f32 * 0.3).sin() - cx) * scale;
        assert!((data[0] - expected_nose_x).abs() < 1e-4, "center-crop window alignment");
    }

    #[test]
    fn test_normalization_range_and_y_flip() {
        let frames = vec![standing_frame(3), standing_frame(7)];
        let data = build_input_data(&frames, 3, 30, 18, 2);
        let frame_stride = 18 * 2;
        for t in 0..2 {
            for v in 0..18 {
                // x 通道（C=0）有效点必在 [-0.3, 0.3]（长边映射 0.6 -> 半径 0.3）
                let x = data[t * frame_stride + v * 2];
                if x != 0.0 {
                    assert!(x.abs() <= 0.3 + 1e-4, "x={x}");
                }
            }
        }
        // y 翻转验证：站立时 nose(0,70) 在包围盒垂直中心上方（像素 y 小），
        // 归一化后（翻转）应为正值。C=1 层偏移 = T*V*M = 30*18*2 = 1080
        let layer = 30 * 18 * 2;
        let nose_y = data[layer]; // t=0, v=0(nose), m=0
        assert!(nose_y > 0.0, "nose should be positive after y-flip, got {nose_y}");
        // 髋/踝在中心下方，翻转后为负（v=13 left_ankle，OpenPose 顺序下肢为 8~13）
        let ankle_y = data[layer + 13 * 2];
        assert!(ankle_y < 0.0, "ankle should be negative after y-flip, got {ankle_y}");
    }

    #[test]
    fn test_invalid_frames_produce_zero_input() {
        let empty_frame: Vec<Keypoint> = (0..17).map(|_| Keypoint::new(0.0, 0.0, 0.0)).collect();
        let frames = vec![empty_frame; 3];
        let data = build_input_data(&frames, 3, 30, 18, 2);
        assert!(data.iter().all(|&v| v == 0.0), "all-invalid frames -> all-zero input");
    }

    #[test]
    fn test_softmax_stable() {
        let scores = softmax(&[1000.0, 1001.0, 999.0]);
        assert!((scores.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(scores[1] > scores[0] && scores[0] > scores[2]);
    }

    /// 端到端自验：合成站立序列 vs 模拟跌倒序列（y 骤降 + 躯干放平）各推理一次。
    ///
    /// 需要模型文件 `testmodels/action_stgcn.onnx`，缺失时跳过（打印提示）。
    /// 注意：合成骨架不等于真实跌倒视频，此处仅验证流水线与输出分布，
    /// Kinetics-400 也没有 fall 类别，不能作为跌倒检测精度结论。
    #[test]
    fn test_classify_standing_vs_fall_smoke() {
        let model_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testmodels/action_stgcn.onnx");
        if !model_path.exists() {
            eprintln!("skip: {} not found", model_path.display());
            return;
        }

        let engine = ActionRecognitionEngine::new(&model_path, DeviceType::Cpu).unwrap();
        assert_eq!(engine.input_layout(), (3, 30, 18, 2));
        assert_eq!(engine.num_classes(), 400);

        // 站立序列
        let standing: Vec<Vec<Keypoint>> = (0..30).map(standing_frame).collect();
        let pred_stand = engine.classify(&standing).unwrap();
        let top5_stand = top_k(&pred_stand, 5);
        println!("站立序列 -> label='{}' (id={}, p={:.4})", pred_stand.label, pred_stand.label_id, pred_stand.scores[pred_stand.label_id]);
        println!("  top5: {:?}", top5_stand);

        // 跌倒序列
        let falling = falling_sequence(30);
        let pred_fall = engine.classify(&falling).unwrap();
        let top5_fall = top_k(&pred_fall, 5);
        println!("跌倒序列 -> label='{}' (id={}, p={:.4})", pred_fall.label, pred_fall.label_id, pred_fall.scores[pred_fall.label_id]);
        println!("  top5: {:?}", top5_fall);

        // 结构断言：softmax 概率归一、label 合法
        for pred in [&pred_stand, &pred_fall] {
            assert_eq!(pred.scores.len(), 400);
            assert!((pred.scores.iter().sum::<f32>() - 1.0).abs() < 1e-3, "softmax sums to 1");
            assert!(pred.label_id < 400);
            assert_eq!(pred.label, KINETICS400_LABELS[pred.label_id]);
        }
    }

    /// 取概率 top-k（测试辅助）。
    fn top_k(pred: &ActionPrediction, k: usize) -> Vec<(String, f32)> {
        let mut idx: Vec<usize> = (0..pred.scores.len()).collect();
        idx.sort_by(|a, b| pred.scores[*b].partial_cmp(&pred.scores[*a]).unwrap());
        idx.into_iter()
            .take(k)
            .map(|i| (KINETICS400_LABELS[i].to_string(), pred.scores[i]))
            .collect()
    }
}
