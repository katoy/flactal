//! マンデルブロ集合計算関数

use num_complex::Complex;
use rug::{ops::Pow, Float};

/// マンデルブロ集合の反復回数を計算（f64高速版）
pub fn mandelbrot_iter_fast(c: Complex<f64>, max_iter: u32) -> u32 {
    let mut z = Complex::new(0.0, 0.0);

    for i in 0..max_iter {
        if z.norm_sqr() > 4.0 {
            return i;
        }
        z = z * z + c;
    }
    max_iter
}

/// マンデルブロ集合の反復回数を計算（高精度版）
pub fn mandelbrot_iter_hp(c_real: &Float, c_imag: &Float, max_iter: u32, precision: u32) -> u32 {
    let mut z_real = Float::with_val(precision, 0.0);
    let mut z_imag = Float::with_val(precision, 0.0);

    for i in 0..max_iter {
        let zr2 = Float::with_val(precision, z_real.clone().pow(2));
        let zi2 = Float::with_val(precision, z_imag.clone().pow(2));
        let mut norm_sqr = Float::with_val(precision, &zr2);
        norm_sqr += &zi2;

        if norm_sqr > 4.0 {
            return i;
        }

        let mut new_real = Float::with_val(precision, &zr2);
        new_real -= &zi2;
        new_real += c_real;

        let mut new_imag = Float::with_val(precision, &z_real * &z_imag);
        new_imag *= 2.0;
        new_imag += c_imag;

        z_real = new_real;
        z_imag = new_imag;
    }
    max_iter
}
