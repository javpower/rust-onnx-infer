//! 结果模型层。

mod bounding_box;
mod classification;
mod detection;
mod keypoint;
mod lightglue_match;
mod lomar_match;
mod matting;
mod point2d;
mod romav2_match;
mod segmentation;

pub use bounding_box::BoundingBox;
pub use classification::ClassificationResult;
pub use detection::Detection;
pub use keypoint::{coco, face_landmarks, Keypoint, PoseResult};
pub use lightglue_match::{FeatureMatch as LightGlueFeatureMatch, LightGlueBatchResult, LightGlueMatchResult, MatchKeyPoint as LightGlueKeyPoint};
pub use lomar_match::{FeatureMatch as LoMaRFeatureMatch, LoMaRMatchResult, MatchKeyPoint as LoMaRKeyPoint};
pub use matting::MattingResult;
pub use point2d::Point2D;
pub use romav2_match::{DenseMatch, RomaV2MatchResult};
pub use segmentation::Segmentation;
