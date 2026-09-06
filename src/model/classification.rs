//! 图像分类结果。

/// 图像分类结果。
#[derive(Debug, Clone, PartialEq)]
pub struct ClassificationResult {
    /// 类别名称
    pub class_name: String,
    /// 类别 ID
    pub class_id: usize,
    /// 置信度
    pub confidence: f64,
    /// 所有类别的分数（可选）
    pub all_scores: Option<Vec<f32>>,
}

impl ClassificationResult {
    pub fn new(class_name: impl Into<String>, class_id: usize, confidence: f64) -> Self {
        ClassificationResult {
            class_name: class_name.into(),
            class_id,
            confidence,
            all_scores: None,
        }
    }

    /// 从分数数组创建 Top-K 结果。
    pub fn from_scores(scores: &[f32], labels: Option<&[String]>, top_k: usize) -> Vec<Self> {
        let mut idx: Vec<usize> = (0..scores.len()).collect();
        idx.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));
        idx.into_iter()
            .take(top_k)
            .map(|i| {
                let mut r = ClassificationResult::new(
                    labels
                        .and_then(|l| l.get(i))
                        .cloned()
                        .unwrap_or_else(|| i.to_string()),
                    i,
                    scores[i] as f64,
                );
                r.all_scores = Some(scores.to_vec());
                r
            })
            .collect()
    }

    /// 获取 Top-1 结果。
    pub fn top1(scores: &[f32], labels: Option<&[String]>) -> Self {
        let mut max_idx = 0;
        let mut max_score = scores.first().copied().unwrap_or(0.0);
        for (i, &s) in scores.iter().enumerate().skip(1) {
            if s > max_score {
                max_score = s;
                max_idx = i;
            }
        }
        let mut r = ClassificationResult::new(
            labels
                .and_then(|l| l.get(max_idx))
                .cloned()
                .unwrap_or_else(|| max_idx.to_string()),
            max_idx,
            max_score as f64,
        );
        r.all_scores = Some(scores.to_vec());
        r
    }
}

impl std::fmt::Display for ClassificationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Classification[{}(id={}), conf={:.4}]",
            self.class_name, self.class_id, self.confidence
        )
    }
}
