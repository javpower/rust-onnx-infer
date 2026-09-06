//! SAHI 切片坐标计算 —— 对齐 sahi==0.12.6 `sahi.slicing.get_slice_bboxes`
//! （对应 SahiSlicer`）。
//!
//! 算法为纯整数运算，与 Python 官方实现结果完全一致：
//! - 重叠像素取 **int 截断**：`overlap = (ratio * slice_size) as i32`（Python 的 `int()`
//!   对正数即截断，非四舍五入）
//! - 从左上角开始按 `slice_size - overlap` 步进滑动窗口
//! - 边界回退吸附：当窗口超出图像时，把右/下边界钳制到图像边缘并反向回退起点，
//!   保证最后一片恰好贴边（边缘小目标不会被切到图外）
//! - 未指定切片尺寸且 `auto_slice_resolution=true` 时，按图像分辨率自动选择
//!   切片划分（low/medium/high/ultra-high 四档，与官方 `get_auto_slice_params` 一致）
//!
//! 切片坐标格式为 `[x_min, y_min, x_max, y_max]`（左闭右开，与 Python ndarray 切片一致）。

use crate::error::{Result, VisionError};
use crate::imaging::Image;

/// 切片坐标 `[x_min, y_min, x_max, y_max]`（左闭右开，与 Python ndarray 切片一致）。
pub type SliceBbox = [i32; 4];

/// 切片结果：切片图像 + 在原图中的偏移（对应上游内部类 `SahiSlice{image, offsetX, offsetY}`）。
#[derive(Debug, Clone)]
pub struct SahiSlice {
    /// 切片图像（`Image::crop` 产出）
    pub image: Image,
    /// 切片在原图中的 X 偏移（x_min）
    pub offset_x: i32,
    /// 切片在原图中的 Y 偏移（y_min）
    pub offset_y: i32,
}

/// 计算切片坐标列表（与 Python 官方 `get_slice_bboxes` 结果逐项一致）。
///
/// # 参数
/// - `image_height` / `image_width`：原图高/宽
/// - `slice_height` / `slice_width`：切片尺寸；`None` 或 <=0 时走 auto（需 `auto_slice_resolution=true`）
/// - `auto_slice_resolution`：未指定切片尺寸时按分辨率自动计算
/// - `overlap_height_ratio` / `overlap_width_ratio`：重叠比例（如 0.2 表示相邻切片重叠 20% 像素）
///
/// # 返回
/// 每片坐标 `[x_min, y_min, x_max, y_max]`，顺序与官方一致（先行后列）。
pub fn get_slice_bboxes(
    image_height: i32,
    image_width: i32,
    slice_height: Option<i32>,
    slice_width: Option<i32>,
    auto_slice_resolution: bool,
    overlap_height_ratio: f64,
    overlap_width_ratio: f64,
) -> Result<Vec<SliceBbox>> {
    if image_height <= 0 || image_width <= 0 {
        return Err(VisionError::invalid_argument(format!(
            "image size must be positive, got {image_width}x{image_height}"
        )));
    }
    let mut slice_bboxes: Vec<SliceBbox> = Vec::new();

    let slice_h: i32;
    let slice_w: i32;
    let y_overlap: i32;
    let x_overlap: i32;
    // Python: `if slice_height and slice_width` —— 0 值等同于未指定
    if slice_height.is_some_and(|h| h > 0) && slice_width.is_some_and(|w| w > 0) {
        let sh = slice_height.unwrap();
        let sw = slice_width.unwrap();
        if overlap_height_ratio >= 1.0 {
            return Err(VisionError::invalid_argument(
                "Overlap ratio must be less than 1.0",
            ));
        }
        if overlap_width_ratio >= 1.0 {
            return Err(VisionError::invalid_argument(
                "Overlap ratio must be less than 1.0",
            ));
        }
        slice_h = sh;
        slice_w = sw;
        // Python int() 对正数为截断（非四舍五入）
        y_overlap = (overlap_height_ratio * sh as f64) as i32;
        x_overlap = (overlap_width_ratio * sw as f64) as i32;
    } else if auto_slice_resolution {
        // get_auto_slice_params 返回 (x_overlap, y_overlap, slice_width, slice_height)
        let p = get_auto_slice_params(image_height, image_width);
        x_overlap = p[0];
        y_overlap = p[1];
        slice_w = p[2];
        slice_h = p[3];
    } else {
        return Err(VisionError::invalid_argument(
            "Compute type is not auto and slice width and height are not provided.",
        ));
    }
    if slice_h <= 0 || slice_w <= 0 {
        return Err(VisionError::invalid_argument(format!(
            "slice size must be positive, got {slice_w}x{slice_h}"
        )));
    }

    // 与 Python 循环逐行对应：y_max/y_min 跨行保持，x_min/x_max 每行重置
    let mut y_max = 0i32;
    let mut y_min = 0i32;
    while y_max < image_height {
        let mut x_min = 0i32;
        let mut x_max = 0i32;
        y_max = y_min + slice_h;
        while x_max < image_width {
            x_max = x_min + slice_w;
            if y_max > image_height || x_max > image_width {
                // 边界回退吸附：末尾切片贴住图像右/下边缘
                x_max = x_max.min(image_width);
                y_max = y_max.min(image_height);
                x_min = (x_max - slice_w).max(0);
                y_min = (y_max - slice_h).max(0);
                slice_bboxes.push([x_min, y_min, x_max, y_max]);
            } else {
                slice_bboxes.push([x_min, y_min, x_max, y_max]);
            }
            x_min = x_max - x_overlap;
        }
        y_min = y_max - y_overlap;
    }
    Ok(slice_bboxes)
}

/// 按切片坐标批量裁剪切片图像（[`SahiSlice`] 视图，`Image::crop` 拷贝产出）。
pub fn slice_image(image: &Image, slice_bboxes: &[SliceBbox]) -> Result<Vec<SahiSlice>> {
    slice_bboxes
        .iter()
        .map(
            |&[x_min, y_min, x_max, y_max]| -> Result<SahiSlice> {
                Ok(SahiSlice {
                    image: image.crop(
                        x_min.max(0) as usize,
                        y_min.max(0) as usize,
                        (x_max - x_min).max(0) as usize,
                        (y_max - y_min).max(0) as usize,
                    )?,
                    offset_x: x_min,
                    offset_y: y_min,
                })
            },
        )
        .collect()
}

/// 未指定切片尺寸时按图像分辨率自动选择参数（官方 `get_auto_slice_params`）。
///
/// 返回 `[x_overlap, y_overlap, slice_width, slice_height]`。
pub fn get_auto_slice_params(height: i32, width: i32) -> [i32; 4] {
    let resolution = height * width;
    let factor = calc_resolution_factor(resolution);
    if factor <= 18 {
        get_resolution_selector("low", height, width)
    } else if factor < 21 {
        get_resolution_selector("medium", height, width)
    } else if factor < 24 {
        get_resolution_selector("high", height, width)
    } else {
        get_resolution_selector("ultra-high", height, width)
    }
}

/// 2^expo >= resolution 的最小 expo 减一（官方 `calc_resolution_factor`）。
pub fn calc_resolution_factor(resolution: i32) -> i32 {
    let mut expo = 0i32;
    while (1i64 << expo) < resolution as i64 {
        expo += 1;
    }
    expo - 1
}

/// 竖图/横图/方图判断（官方 `calc_aspect_ratio_orientation`）。
pub fn calc_aspect_ratio_orientation(width: i32, height: i32) -> &'static str {
    if width < height {
        "vertical"
    } else if width > height {
        "horizontal"
    } else {
        "square"
    }
}

/// 官方 `calc_slice_and_overlap_params`（含 `calc_ratio_and_slice`）：
/// 返回 `[x_overlap, y_overlap, slice_width, slice_height]`。
///
/// 官方语义注意点：slide 划分的是 row/col（vertical 时 slice_col = slide*2），
/// 而 slice_height 用 split_col 整除、slice_width 用 split_row 整除。
fn calc_slice_and_overlap_params(resolution: &str, height: i32, width: i32, orientation: &str) -> [i32; 4] {
    let slide: i32;
    let overlap_ratio: f64;
    match resolution {
        "medium" => {
            slide = 1;
            overlap_ratio = 0.8;
        }
        "high" => {
            slide = 2;
            overlap_ratio = 0.4;
        }
        "ultra-high" => {
            slide = 4;
            overlap_ratio = 0.4;
        }
        // low（官方 default 分支）
        _ => {
            let slice_height0 = height;
            let slice_width0 = width;
            let x_overlap0 = (slice_width0 as f64 * 1.0) as i32;
            let y_overlap0 = (slice_height0 as f64 * 1.0) as i32;
            return [x_overlap0, y_overlap0, slice_width0, slice_height0];
        }
    }
    // calc_ratio_and_slice: vertical → row=slide, col=slide*2；horizontal → row=slide*2, col=slide
    let split_row = if orientation == "vertical" {
        slide
    } else if orientation == "horizontal" {
        slide * 2
    } else {
        slide
    };
    let split_col = if orientation == "vertical" {
        slide * 2
    } else {
        slide
    };
    let slice_height = height / split_col;
    let slice_width = width / split_row;
    let x_overlap = (slice_width as f64 * overlap_ratio) as i32;
    let y_overlap = (slice_height as f64 * overlap_ratio) as i32;
    [x_overlap, y_overlap, slice_width, slice_height]
}

/// 官方 `get_resolution_selector`：返回 `[x_overlap, y_overlap, slice_width, slice_height]`。
fn get_resolution_selector(res: &str, height: i32, width: i32) -> [i32; 4] {
    let orientation = calc_aspect_ratio_orientation(width, height);
    calc_slice_and_overlap_params(res, height, width, orientation)
}
