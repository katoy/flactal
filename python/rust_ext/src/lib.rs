//! マンデルブロ集合計算の高速Rust実装
//!
//! PyO3を使用してPythonから呼び出し可能な拡張モジュールとして提供

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// 1点のマンデルブロ計算
///
/// # Arguments
/// * `cx` - 複素数の実部
/// * `cy` - 複素数の虚部
/// * `max_iter` - 最大反復回数
///
/// # Returns
/// 発散するまでの反復回数
#[inline]
fn mandelbrot_point(cx: f64, cy: f64, max_iter: u32) -> f64 {
    let mut zx = 0.0;
    let mut zy = 0.0;

    for i in 0..max_iter {
        let zx2 = zx * zx;
        let zy2 = zy * zy;

        if zx2 + zy2 > 4.0 {
            return i as f64;
        }

        zy = 2.0 * zx * zy + cy;
        zx = zx2 - zy2 + cx;
    }

    max_iter as f64
}

/// マンデルブロ集合をベクトル化して高速に計算する
///
/// rayonによる並列計算で高速化
///
/// # Arguments
/// * `xmin` - x軸の最小値
/// * `xmax` - x軸の最大値
/// * `ymin` - y軸の最小値
/// * `ymax` - y軸の最大値
/// * `width` - 画像幅 (ピクセル)
/// * `height` - 画像高さ (ピクセル)
/// * `max_iter` - 最大反復回数
///
/// # Returns
/// 反復回数を格納した2次元配列 (height x width)
#[pyfunction]
fn mandelbrot_set_vectorized(
    py: Python<'_>,
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    width: usize,
    height: usize,
    max_iter: u32,
) -> Py<PyArray2<f64>> {
    // 結果配列を作成
    let mut result = vec![0.0f64; width * height];

    // x, y の刻み幅
    let x_step = (xmax - xmin) / (width as f64);
    let y_step = (ymax - ymin) / (height as f64);

    // 並列計算 (行単位で並列化)
    result
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(row, row_data)| {
            let cy = ymin + (row as f64) * y_step;
            for (col, pixel) in row_data.iter_mut().enumerate() {
                let cx = xmin + (col as f64) * x_step;
                *pixel = mandelbrot_point(cx, cy, max_iter);
            }
        });

    // NumPy配列に変換して返す
    let array = Array2::from_shape_vec((height, width), result).unwrap();
    array.into_pyarray(py).into()
}

/// Python モジュール定義
#[pymodule]
fn mandelbrot_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mandelbrot_set_vectorized, m)?)?;
    Ok(())
}
