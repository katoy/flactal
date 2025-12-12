//! マンデルブロ集合計算関数

use num_complex::Complex;
use rug::{Assign, Float};

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

    // 作業用変数を事前に確保（アロケーション削減）
    let mut zr2 = Float::with_val(precision, 0.0);
    let mut zi2 = Float::with_val(precision, 0.0);
    let mut norm_sqr = Float::with_val(precision, 0.0);
    let mut next_r = Float::with_val(precision, 0.0);
    let mut next_i = Float::with_val(precision, 0.0);

    for i in 0..max_iter {
        // zr2 = z_real^2
        zr2.assign(&z_real);
        zr2.square_mut();

        // zi2 = z_imag^2
        zi2.assign(&z_imag);
        zi2.square_mut();

        // norm_sqr = zr2 + zi2
        norm_sqr.assign(&zr2);
        norm_sqr += &zi2;

        if norm_sqr > 4.0 {
            return i;
        }

        // next_r = zr2 - zi2 + c_real
        next_r.assign(&zr2);
        next_r -= &zi2;
        next_r += c_real;

        // next_i = 2 * z_real * z_imag + c_imag
        next_i.assign(&z_real);
        next_i *= &z_imag;
        next_i *= 2.0;
        next_i += c_imag;

        // update z
        z_real.assign(&next_r);
        z_imag.assign(&next_i);
    }
    max_iter
}
