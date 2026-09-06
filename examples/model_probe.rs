//! ONNX 模型结构探针（独立工具，不依赖库内引擎）。
//!
//! 用 ort 加载一个或多个 .onnx 文件，打印每个输入/输出的
//! name、shape（动态维度标 -1）与元素类型，用于确认模型 I/O 约定。
//!
//! 可选 `--run=HxW`：对声明中含动态维度的模型做一次零张量试推理，
//! 打印运行期真实输出 shape（用于实测动态维度是否真的可变）。
//!
//! 用法：
//!
//! ```text
//! cargo run --release --example model_probe -- <model1.onnx> [model2.onnx ...]
//! cargo run --release --example model_probe -- --run=320x320 <model.onnx>
//! ```

use std::path::Path;

use ort::session::Session;
use ort::value::{ValueType};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("用法: cargo run --release --example model_probe -- [--run=HxW] <model1.onnx> [model2.onnx ...]");
        std::process::exit(2);
    }

    // 解析 --run=HxW / --force=HxW 与模型路径列表
    let mut run_size: Option<(i64, i64)> = None;
    let mut force_size: Option<(i64, i64)> = None;
    let mut models = Vec::new();
    for arg in &args {
        if let Some(spec) = arg.strip_prefix("--run=") {
            run_size = Some(parse_hxw(spec)?);
        } else if let Some(spec) = arg.strip_prefix("--force=") {
            force_size = Some(parse_hxw(spec)?);
        } else {
            models.push(arg.clone());
        }
    }

    for arg in &models {
        let path = Path::new(arg);
        println!("=== {} ===", path.display());
        if !path.exists() {
            anyhow::bail!("模型文件不存在: {}", path.display());
        }

        let size = std::fs::metadata(path)?.len();
        println!("文件大小: {:.2} MB", size as f64 / (1024.0 * 1024.0));

        // 与 src/core/session_factory.rs 一致地创建会话（这里用默认 CPU/线程配置即可）
        let mut session = Session::builder()?.commit_from_file(path)?;

        println!("-- 输入 ({}):", session.inputs.len());
        for (i, input) in session.inputs.iter().enumerate() {
            print_value_info(i, &input.name, &input.input_type);
        }
        println!("-- 输出 ({}):", session.outputs.len());
        for (i, output) in session.outputs.iter().enumerate() {
            print_value_info(i, &output.name, &output.output_type);
        }

        // 可选：零张量试推理，实测输出 shape（单模型失败不中断整体探测）
        if let Some((h, w)) = run_size {
            if let Err(e) = run_zero_test(&mut session, h, w, false) {
                println!("    [试推理失败] {e}");
            }
        }
        if let Some((h, w)) = force_size {
            if let Err(e) = run_zero_test(&mut session, h, w, true) {
                println!("    [试推理失败] {e}");
            }
        }
        println!();
    }

    Ok(())
}

/// 解析 `HxW` 尺寸参数。
fn parse_hxw(spec: &str) -> anyhow::Result<(i64, i64)> {
    let (h, w) = spec
        .split_once(['x', 'X', '*'])
        .ok_or_else(|| anyhow::anyhow!("参数格式应为 HxW，收到: {spec}"))?;
    Ok((h.parse()?, w.parse()?))
}

/// 打印单个输入/输出的元信息（张量 shape 动态维度标 -1）。
fn print_value_info(idx: usize, name: &str, value_type: &ValueType) {
    match value_type {
        ValueType::Tensor { ty, shape, .. } => {
            let dims: Vec<String> = shape
                .iter()
                .map(|&d| if d < 0 { "-1".to_string() } else { d.to_string() })
                .collect();
            println!("  [{}] name='{}' shape=[{}] type={:?}", idx, name, dims.join(","), ty);
        }
        other => {
            println!("  [{}] name='{}' type={:?}（非张量）", idx, name, other);
        }
    }
}

/// 零张量试推理，打印运行期真实输出 shape。
///
/// - `force = false`：仅替换声明中的动态（-1）维度——第 2/3 维填 H/W，其余填 1；
/// - `force = true`：无视声明，把第 2/3 维强制填成 H/W（用于验证"静态"声明
///   的模型是否实际接受其他尺寸，即图内解码是否真为动态）。
///
/// 仅支持全 f32 张量输入的模型（本仓库探针目标模型均为 f32）。
fn run_zero_test(session: &mut Session, h: i64, w: i64, force: bool) -> anyhow::Result<()> {
    println!(
        "-- 试推理 ({}, H={h}, W={w}):",
        if force { "强制尺寸" } else { "动态维度填充" }
    );

    // 先克隆输入/输出名，避免与 run 的可变借用冲突
    let input_names: Vec<String> = session.inputs.iter().map(|i| i.name.clone()).collect();
    let output_names: Vec<String> = session.outputs.iter().map(|o| o.name.clone()).collect();

    let mut inputs = Vec::new();
    for (input, name) in session.inputs.iter().zip(&input_names) {
        let ValueType::Tensor { shape, .. } = &input.input_type else {
            anyhow::bail!("输入 '{}' 不是张量，跳过试推理", name);
        };
        // 尺寸维度：force 时第 2/3 维无条件填 H/W；否则仅替换 -1（其余 -1 维填 1）
        let dims: Vec<i64> = shape
            .iter()
            .enumerate()
            .map(|(axis, &d)| {
                if force && (axis == 2 || axis == 3) {
                    if axis == 2 { h } else { w }
                } else {
                    match d {
                        d if d >= 0 => d,
                        2 => h,
                        3 => w,
                        _ => 1,
                    }
                }
            })
            .collect();
        let count: usize = dims.iter().product::<i64>().try_into()?;
        let tensor = ort::value::Tensor::from_array((dims, vec![0f32; count]))?;
        inputs.push((name.as_str(), tensor));
    }

    let outputs = session.run(inputs)?;
    for name in &output_names {
        let value = outputs
            .get(name.as_str())
            .ok_or_else(|| anyhow::anyhow!("缺少输出 '{name}'"))?;
        match value.dtype() {
            ValueType::Tensor { ty, shape, .. } => {
                let dims: Vec<String> = shape
                    .iter()
                    .map(|&d| if d < 0 { "-1".to_string() } else { d.to_string() })
                    .collect();
                println!("    '{}' -> shape=[{}] type={:?}", name, dims.join(","), ty);
            }
            other => println!("    '{}' -> type={:?}（非张量）", name, other),
        }
    }
    Ok(())
}
