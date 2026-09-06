//! 姿态几何规则引擎（零模型，纯算法）。
//!
//! 输入库内姿态引擎（YOLO-Pose / RTMO 等）逐帧输出的 COCO 17 关键点
//! （[`crate::model::Keypoint`]，顺序见 [`crate::model::keypoint::coco`]），
//! 用纯几何规则给出躺卧 / 举手 / 躯干朝向等判断，可作为跌倒检测的第一道
//! 快速过滤，或与 [`super::action_recognition`] 的骨架动作识别互补。
//!
//! **几何约定**：图像坐标系 y 轴向下（y 越大越靠画面下方），因此
//! "A 在 B 上方" 等价于 `A.y < B.y`。所有角度均为度（degree）。
//!
//! **关键点索引**（COCO 17）：
//! `5=左肩, 6=右肩, 9=左腕, 10=右腕, 11=左髋, 12=右髋`。
//!
//! **有效性策略**：参与计算的关键点必须存在且 `score >= min_valid_score`
//! （默认 0.3）；数据不足时判定结果一律保守为 false（不报警），
//! [`BodyOrientation`] 则回退 [`BodyOrientation::Upright`]。
//!
//! 本模块不实现 [`crate::core::engine::OnnxInferenceEngine`] trait（无模型推理）。

use crate::model::Keypoint;

/// 躯干朝向分类。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyOrientation {
    /// 直立：躯干与竖直方向夹角 <= 竖直阈值（默认 30°）
    Upright,
    /// 倾斜：介于直立与躺卧之间
    Tilted,
    /// 躺卧：躯干与竖直方向夹角 >= 躺卧阈值（默认 60°），接近水平
    Lying,
}

/// 姿态几何规则引擎（纯算法，无模型）。
///
/// 阈值均可调；默认值适合常规监控俯视/平视相机下的站立-跌倒场景。
#[derive(Debug, Clone)]
pub struct PoseRules {
    /// 躺卧判定：躯干与竖直方向夹角阈值（度），默认 60°。
    /// 夹角 = 躯干向量（肩中点 -> 髋中点）与竖直向下方向 (0, 1) 的夹角。
    pub lying_angle_threshold_deg: f32,
    /// 躺卧判定的第二条件：躯干与水平方向的夹角上限（度），默认 30°。
    /// 即躯干"接近水平"（与竖直夹角 > 60° 时该条件恒成立，二者互为冗余校验，
    /// 显式保留以对应任务定义并允许独立放宽）。
    pub horizontal_angle_threshold_deg: f32,
    /// 直立判定：躯干与竖直方向夹角阈值（度），默认 30°。
    pub upright_angle_threshold_deg: f32,
    /// 举手判定：手腕需高于同侧肩膀的余量（像素，y 差值），默认 0.0。
    /// 设为正值可抑制手腕刚好贴着肩膀时的误报。
    pub hands_up_margin_px: f32,
    /// 关键点有效分数阈值，低于该值的关键点不参与计算，默认 0.3。
    pub min_valid_score: f32,
}

impl Default for PoseRules {
    fn default() -> Self {
        PoseRules {
            lying_angle_threshold_deg: 60.0,
            horizontal_angle_threshold_deg: 30.0,
            upright_angle_threshold_deg: 30.0,
            hands_up_margin_px: 0.0,
            min_valid_score: 0.3,
        }
    }
}

impl PoseRules {
    /// 创建规则引擎（默认阈值：躺卧 60° / 直立 30° / 分数 0.3）。
    pub fn new() -> Self {
        Self::default()
    }

    /// 修改躺卧角度阈值（度）。
    pub fn set_lying_angle_threshold(&mut self, deg: f32) {
        self.lying_angle_threshold_deg = deg;
    }

    /// 修改直立角度阈值（度）。
    pub fn set_upright_angle_threshold(&mut self, deg: f32) {
        self.upright_angle_threshold_deg = deg;
    }

    /// 修改举手余量（像素）。
    pub fn set_hands_up_margin(&mut self, px: f32) {
        self.hands_up_margin_px = px;
    }

    /// 修改关键点有效分数阈值。
    pub fn set_min_valid_score(&mut self, score: f32) {
        self.min_valid_score = score;
    }

    /// 是否躺卧：躯干（肩中点 -> 髋中点）与竖直方向夹角 > 60°，
    /// 且躯干方向接近水平（与水平方向夹角 <= 30°）。
    ///
    /// 两个条件由同一躯干向量计算，正常情况下前者蕴含后者（互为冗余校验）；
    /// 任一必需关键点（肩 5/6、髋 11/12，每侧至少一个）无效时保守返回 false。
    pub fn is_lying(&self, keypoints: &[Keypoint]) -> bool {
        let Some(torso) = self.torso_vector(keypoints) else {
            return false;
        };
        let angle_vertical = torso.angle_to_vertical_deg();
        let angle_horizontal = torso.angle_to_horizontal_deg();
        angle_vertical > self.lying_angle_threshold_deg
            && angle_horizontal <= self.horizontal_angle_threshold_deg
    }

    /// 是否举手：任一侧手腕（9=左腕 / 10=右腕）高于同侧肩膀
    /// （5=左肩 / 6=右肩）至少 `hands_up_margin_px` 像素。
    pub fn is_hands_up(&self, keypoints: &[Keypoint]) -> bool {
        let pairs = [(9usize, 5usize), (10usize, 6usize)]; // (手腕, 同侧肩)
        pairs.iter().any(|&(wrist_idx, shoulder_idx)| {
            match (
                self.valid_kp(keypoints, wrist_idx),
                self.valid_kp(keypoints, shoulder_idx),
            ) {
                (Some(w), Some(s)) => w.y < s.y - self.hands_up_margin_px,
                _ => false,
            }
        })
    }

    /// 躯干朝向分类：与竖直夹角 <= 30° 判 [`BodyOrientation::Upright`]，
    /// >= 60° 判 [`BodyOrientation::Lying`]，其余为 Tilted；关键点无效时保守回退 Upright。
    pub fn body_orientation(&self, keypoints: &[Keypoint]) -> BodyOrientation {
        let Some(torso) = self.torso_vector(keypoints) else {
            return BodyOrientation::Upright;
        };
        let angle = torso.angle_to_vertical_deg();
        if angle <= self.upright_angle_threshold_deg {
            BodyOrientation::Upright
        } else if angle >= self.lying_angle_threshold_deg {
            BodyOrientation::Lying
        } else {
            BodyOrientation::Tilted
        }
    }

    /// 躯干与竖直方向的夹角（度）；关键点无效时返回 None。
    ///
    /// 对外暴露便于上层做自定义阈值的时间平滑 / 报警滞回。
    pub fn torso_angle_to_vertical_deg(&self, keypoints: &[Keypoint]) -> Option<f32> {
        self.torso_vector(keypoints).map(|t| t.angle_to_vertical_deg())
    }

    /// 躯干向量：肩部中点 (5,6) -> 髋部中点 (11,12)，单位化后返回。
    ///
    /// 肩/髋各自优先取有效单侧点，双侧均有效时取中点（遮挡单侧仍可计算）。
    /// 任一端无法确定时返回 None。
    fn torso_vector(&self, keypoints: &[Keypoint]) -> Option<Vec2> {
        let (sx, sy) = self.midpoint_of_pair(keypoints, 5, 6)?;
        let (hx, hy) = self.midpoint_of_pair(keypoints, 11, 12)?;
        let v = Vec2::new(hx - sx, hy - sy);
        let len = v.norm();
        if len < 1e-6 {
            return None; // 肩髋重合（异常姿态/误检），无法确定方向
        }
        Some(Vec2::new(v.x / len, v.y / len))
    }

    /// 一对关键点的中点：双侧有效取中点；仅单侧有效取该侧；
    /// 双侧均无效返回 None。
    fn midpoint_of_pair(&self, keypoints: &[Keypoint], a: usize, b: usize) -> Option<(f32, f32)> {
        match (self.valid_kp(keypoints, a), self.valid_kp(keypoints, b)) {
            (Some(p), Some(q)) => Some(((p.x + q.x) / 2.0, (p.y + q.y) / 2.0)),
            (Some(p), None) => Some((p.x, p.y)),
            (None, Some(q)) => Some((q.x, q.y)),
            (None, None) => None,
        }
    }

    /// 取第 `idx` 个关键点（存在且 score 达阈值时返回）。
    fn valid_kp(&self, keypoints: &[Keypoint], idx: usize) -> Option<Keypoint> {
        keypoints
            .get(idx)
            .copied()
            .filter(|kp| kp.score >= self.min_valid_score)
    }
}

impl Default for BodyOrientation {
    fn default() -> Self {
        BodyOrientation::Upright
    }
}

/// 二维向量（仅本模块内部几何计算用）。
#[derive(Debug, Clone, Copy, PartialEq)]
struct Vec2 {
    x: f32,
    y: f32,
}

impl Vec2 {
    fn new(x: f32, y: f32) -> Self {
        Vec2 { x, y }
    }

    /// 模长。
    fn norm(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    /// 与竖直向下方向 (0, 1) 的夹角（度，[0, 180]）。
    ///
    /// 站立时躯干大致沿 +y（肩在上、髋在下），夹角接近 0；
    /// 水平躺卧时躯干沿 ±x，夹角接近 90。
    fn angle_to_vertical_deg(&self) -> f32 {
        // cosθ = v·(0,1) / |v|（|v| 已单位化）
        let cos = self.y.clamp(-1.0, 1.0);
        cos.acos().to_degrees()
    }

    /// 与水平方向（±x 轴）的最小夹角（度，[0, 90]）。
    fn angle_to_horizontal_deg(&self) -> f32 {
        let cos_abs = self.x.abs().clamp(0.0, 1.0);
        cos_abs.acos().to_degrees()
    }
}

// ==================== 单元测试 ====================

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造一组 COCO 17 关键点，只填 `points` 中给出的索引（其余置无效分数）。
    fn make_keypoints(points: &[(usize, f32, f32)], score: f32) -> Vec<Keypoint> {
        let mut kps: Vec<Keypoint> = (0..17).map(|_| Keypoint::new(0.0, 0.0, 0.0)).collect();
        for &(idx, x, y) in points {
            kps[idx] = Keypoint::new(x, y, score);
        }
        kps
    }

    /// 站立关键点：头在上，肩(y=120)在髋(y=300)正上方，躯干竖直。
    fn standing_keypoints() -> Vec<Keypoint> {
        make_keypoints(
            &[
                (5, 90.0, 120.0),   // 左肩
                (6, 110.0, 120.0),  // 右肩
                (9, 70.0, 260.0),   // 左腕（垂手，低于肩）
                (10, 130.0, 260.0), // 右腕
                (11, 95.0, 300.0),  // 左髋
                (12, 105.0, 300.0), // 右髋
            ],
            0.9,
        )
    }

    /// 躺卧关键点：躯干水平（肩中点 (200,300)，髋中点 (350,300)）。
    fn lying_keypoints() -> Vec<Keypoint> {
        make_keypoints(
            &[
                (5, 190.0, 295.0),  // 左肩
                (6, 210.0, 305.0),  // 右肩
                (9, 150.0, 310.0),  // 左腕（与躯干同高）
                (10, 155.0, 290.0), // 右腕
                (11, 340.0, 295.0), // 左髋
                (12, 360.0, 305.0), // 右髋
            ],
            0.9,
        )
    }

    #[test]
    fn test_standing_upright_and_not_lying() {
        let rules = PoseRules::new();
        let kps = standing_keypoints();
        // 躯干竖直向下：与竖直夹角 0°
        assert!((rules.torso_angle_to_vertical_deg(&kps).unwrap() - 0.0).abs() < 1e-3);
        assert_eq!(rules.body_orientation(&kps), BodyOrientation::Upright);
        assert!(!rules.is_lying(&kps));
        // 垂手不应判举手
        assert!(!rules.is_hands_up(&kps));
    }

    #[test]
    fn test_lying_detected() {
        let rules = PoseRules::new();
        let kps = lying_keypoints();
        // 躯干水平：与竖直夹角 90°
        let angle = rules.torso_angle_to_vertical_deg(&kps).unwrap();
        assert!((angle - 90.0).abs() < 1e-3, "angle = {angle}");
        assert_eq!(rules.body_orientation(&kps), BodyOrientation::Lying);
        assert!(rules.is_lying(&kps));
    }

    #[test]
    fn test_tilted_between_thresholds() {
        let rules = PoseRules::new();
        // 躯干 45° 斜向：肩中点 (100,100)，髋中点 (200,200)
        let kps = make_keypoints(
            &[
                (5, 90.0, 90.0),
                (6, 110.0, 110.0),
                (11, 190.0, 190.0),
                (12, 210.0, 210.0),
            ],
            0.9,
        );
        let angle = rules.torso_angle_to_vertical_deg(&kps).unwrap();
        assert!((angle - 45.0).abs() < 1e-3, "angle = {angle}");
        assert_eq!(rules.body_orientation(&kps), BodyOrientation::Tilted);
        assert!(!rules.is_lying(&kps)); // 45° < 60° 阈值
    }

    #[test]
    fn test_hands_up_left_and_right() {
        let rules = PoseRules::new();
        // 左腕举过头顶（y=80 < 左肩 y=120），右腕垂下
        let kps = make_keypoints(
            &[
                (5, 90.0, 120.0),
                (6, 110.0, 120.0),
                (9, 95.0, 80.0),    // 左腕高于左肩
                (10, 130.0, 260.0), // 右腕低于右肩
            ],
            0.9,
        );
        assert!(rules.is_hands_up(&kps));

        // 双手均低于肩
        let kps_both_down = make_keypoints(
            &[
                (5, 90.0, 120.0),
                (6, 110.0, 120.0),
                (9, 70.0, 260.0),
                (10, 130.0, 260.0),
            ],
            0.9,
        );
        assert!(!rules.is_hands_up(&kps_both_down));

        // 右腕举手（另一侧）
        let kps_right = make_keypoints(
            &[
                (5, 90.0, 120.0),
                (6, 110.0, 120.0),
                (9, 70.0, 260.0),
                (10, 105.0, 60.0), // 右腕高于右肩
            ],
            0.9,
        );
        assert!(rules.is_hands_up(&kps_right));
    }

    #[test]
    fn test_hands_up_margin_rejects_borderline() {
        let mut rules = PoseRules::new();
        rules.set_hands_up_margin(40.0); // 需要 y 至少低 40px
        let kps = make_keypoints(
            &[(5, 90.0, 120.0), (6, 110.0, 120.0), (9, 95.0, 110.0)], // 仅高 10px
            0.9,
        );
        assert!(!rules.is_hands_up(&kps));
    }

    #[test]
    fn test_invalid_keypoints_conservative() {
        let rules = PoseRules::new();
        // 全部关键点 score=0（无效）
        let empty = vec![Keypoint::new(0.0, 0.0, 0.0); 17];
        assert!(!rules.is_lying(&empty));
        assert!(!rules.is_hands_up(&empty));
        assert_eq!(rules.body_orientation(&empty), BodyOrientation::Upright);
        // 空切片同样安全
        assert!(!rules.is_lying(&[]));
        assert!(!rules.is_hands_up(&[]));
    }

    #[test]
    fn test_single_side_occlusion_still_works() {
        let rules = PoseRules::new();
        // 仅左侧关键点有效：躯干仍竖直（单侧肩 -> 单侧髋，x 对齐 90）
        let kps = make_keypoints(&[(5, 90.0, 120.0), (11, 90.0, 300.0)], 0.9);
        let angle = rules.torso_angle_to_vertical_deg(&kps).unwrap();
        assert!((angle - 0.0).abs() < 1.0, "angle = {angle}");
        assert_eq!(rules.body_orientation(&kps), BodyOrientation::Upright);
    }

    #[test]
    fn test_lying_threshold_adjustable() {
        let mut rules = PoseRules::new();
        rules.set_lying_angle_threshold(80.0); // 收紧到 80°
        // 60° 倾斜躯干不再判躺卧
        let kps = make_keypoints(
            &[
                (5, 90.0, 90.0),
                (6, 110.0, 110.0),
                (11, 150.0, 150.0),
                (12, 170.0, 170.0),
            ],
            0.9,
        );
        assert!(!rules.is_lying(&kps));
        // 60° 恰在默认躺卧阈值上、收紧后归为 Tilted
        assert_eq!(rules.body_orientation(&kps), BodyOrientation::Tilted);
    }
}
