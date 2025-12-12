//! マンデルバルブ (Mandelbulb) 3Dフラクタルビューア - カラフル版
//! RayonによるCPU並列レンダリング + レイマーチング法
//!
//! 操作方法:
//!   - W/A/S/D: カメラ移動 (前後左右)
//!   - Space/LShift: カメラ移動 (上昇/下降)
//!   - 矢印キー: カメラ回転
//!   - 1-9: パワー変更 (形状が変化)
//!   - R: リセット
//!   - Esc/Q: 終了

use glam::{Mat3, Vec3};
use minifb::{Key, Window, WindowOptions};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

// ==========================================
// 定数設定
// ==========================================
const WIDTH: usize = 640;
const HEIGHT: usize = 480;
const MAX_STEPS: usize = 150; // レイマーチングの最大ステップ数
const MAX_ITER: usize = 12; // フラクタル計算の反復回数（増加で複雑に）
const BAILOUT: f32 = 2.0;
const EPSILON: f32 = 0.0005; // より精密な衝突判定

// ==========================================
// HSVからRGBへの変換
// ==========================================
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let h = h.fract();
    let h = if h < 0.0 { h + 1.0 } else { h };

    let i = (h * 6.0).floor() as i32;
    let f = h * 6.0 - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);

    match i % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

// ==========================================
// マンデルバルブ距離関数 + 反復回数を返す
// ==========================================
fn map_with_iter(pos: Vec3, power: f32) -> (f32, usize, f32) {
    let mut z = pos;
    let mut dr = 1.0;
    let mut r = 0.0;
    let mut trap = f32::MAX; // オービットトラップ

    let mut i = 0;
    for iter in 0..MAX_ITER {
        r = z.length();
        if r > BAILOUT {
            i = iter;
            break;
        }
        i = iter;

        // オービットトラップ（原点からの最小距離を記録）
        trap = trap.min(z.length());

        dr = r.powf(power - 1.0) * power * dr + 1.0;

        let theta = z.z.atan2((z.x * z.x + z.y * z.y).sqrt());
        let phi = z.y.atan2(z.x);

        let zr = r.powf(power);
        let theta = theta * power;
        let phi = phi * power;

        z = Vec3::new(
            zr * theta.cos() * phi.cos(),
            zr * theta.cos() * phi.sin(),
            zr * theta.sin(),
        );

        z += pos;
    }

    let dist = 0.5 * r.ln() * r / dr;
    (dist, i, trap)
}

fn map(pos: Vec3, power: f32) -> f32 {
    map_with_iter(pos, power).0
}

// ==========================================
// 法線計算
// ==========================================
fn calc_normal(p: Vec3, power: f32) -> Vec3 {
    let e = Vec3::new(EPSILON, 0.0, 0.0);
    let n = Vec3::new(
        map(p + e, power) - map(p - e, power),
        map(p + Vec3::new(0.0, EPSILON, 0.0), power) - map(p - Vec3::new(0.0, EPSILON, 0.0), power),
        map(p + Vec3::new(0.0, 0.0, EPSILON), power) - map(p - Vec3::new(0.0, 0.0, EPSILON), power),
    );
    n.normalize()
}

// ==========================================
// カラフルなレンダリング
// ==========================================
fn ray_march(ro: Vec3, rd: Vec3, power: f32, time: f32) -> u32 {
    let mut t = 0.0;
    let mut hit = false;
    let mut steps = 0;
    let mut total_iter = 0;
    let mut min_trap = f32::MAX;

    for i in 0..MAX_STEPS {
        let p = ro + rd * t;
        let (d, iter, trap) = map_with_iter(p, power);
        total_iter = iter;
        min_trap = min_trap.min(trap);

        if d < EPSILON {
            hit = true;
            steps = i;
            break;
        }

        t += d * 0.8; // スローダウンでより精密に
        if t > 6.0 {
            break;
        }
    }

    if hit {
        let p = ro + rd * t;
        let normal = calc_normal(p, power);

        // 複数光源
        let light1 = Vec3::new(0.577, 0.577, -0.577);
        let light2 = Vec3::new(-0.5, 0.8, 0.3).normalize();

        let diff1 = normal.dot(light1).max(0.0);
        let diff2 = normal.dot(light2).max(0.0) * 0.5;

        // スペキュラー（ハイライト）
        let view_dir = -rd;
        let reflect_dir = (normal * (2.0 * normal.dot(light1))) - light1;
        let spec = view_dir.dot(reflect_dir).max(0.0).powf(32.0);

        // AO
        let ao = 1.0 - (steps as f32 / MAX_STEPS as f32).powf(0.4);

        // カラフルな色計算
        // 1. 反復回数に基づく虹色
        let hue1 = (total_iter as f32 / MAX_ITER as f32) + time * 0.1;

        // 2. 法線方向に基づく色相変化
        let hue2 = (normal.x + normal.y * 0.5 + 1.0) * 0.5;

        // 3. オービットトラップに基づく色
        let hue3 = min_trap * 2.0;

        // 4. 位置に基づく色
        let hue4 = (p.x + p.y + p.z) * 0.3;

        // 色を合成
        let final_hue = (hue1 * 0.4 + hue2 * 0.2 + hue3 * 0.2 + hue4 * 0.2).fract();
        let saturation = 0.8 + (1.0 - ao) * 0.2;
        let value = (diff1 + diff2 + 0.15) * ao;

        let (r_base, g_base, b_base) = hsv_to_rgb(final_hue, saturation, value.min(1.0));

        // スペキュラーハイライト追加
        let r = ((r_base + spec * 0.5).min(1.0) * 255.0) as u32;
        let g = ((g_base + spec * 0.5).min(1.0) * 255.0) as u32;
        let b = ((b_base + spec * 0.5).min(1.0) * 255.0) as u32;

        (r << 16) | (g << 8) | b
    } else {
        // グラデーション背景
        let gradient = (rd.y + 1.0) * 0.5;
        let bg_hue = 0.6 + time * 0.02; // 青〜紫系
        let (r, g, b) = hsv_to_rgb(bg_hue, 0.5, gradient * 0.15 + 0.02);
        let r = (r * 255.0) as u32;
        let g = (g * 255.0) as u32;
        let b = (b * 255.0) as u32;
        (r << 16) | (g << 8) | b
    }
}

// ==========================================
// カメラ
// ==========================================
struct Camera {
    pos: Vec3,
    rot_x: f32,
    rot_y: f32,
}

impl Camera {
    fn new() -> Self {
        Self {
            pos: Vec3::new(0.0, 0.0, -2.5),
            rot_x: 0.0,
            rot_y: 0.0,
        }
    }

    fn get_ray_dir(&self, uv: (f32, f32)) -> Vec3 {
        let dir = Vec3::new(uv.0, uv.1, 1.0).normalize();
        let rot = Mat3::from_rotation_y(self.rot_y) * Mat3::from_rotation_x(self.rot_x);
        rot * dir
    }

    fn forward(&self) -> Vec3 {
        let rot = Mat3::from_rotation_y(self.rot_y) * Mat3::from_rotation_x(self.rot_x);
        rot * Vec3::new(0.0, 0.0, 1.0)
    }

    fn right(&self) -> Vec3 {
        let rot = Mat3::from_rotation_y(self.rot_y);
        rot * Vec3::new(1.0, 0.0, 0.0)
    }
}

fn main() {
    let mut window = Window::new(
        "Mandelbulb 3D Explorer - Colorful Edition",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    window.set_target_fps(60);

    let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];
    let mut camera = Camera::new();
    let power = AtomicU32::new(2); // デフォルトパワー2（キー1）

    println!("=== Mandelbulb 3D Explorer - Colorful Edition ===");
    println!("  Move: W/A/S/D + Space/Shift");
    println!("  Look: Arrow Keys");
    println!("  Power: 1-9 keys (changes shape complexity)");
    println!("  Reset: R");

    while window.is_open() && !window.is_key_down(Key::Escape) && !window.is_key_down(Key::Q) {
        let frame_start = Instant::now();
        let time = 0.0; // アニメーション停止

        // --- 入力処理 ---
        let move_speed = 0.05;
        let rot_speed = 0.05;

        // スクリーンショット撮影
        if window.is_key_pressed(Key::P, minifb::KeyRepeat::No) {
            let mut img_buf: Vec<u8> = Vec::with_capacity(WIDTH * HEIGHT * 3);
            for pixel in &buffer {
                let r = ((pixel >> 16) & 0xFF) as u8;
                let g = ((pixel >> 8) & 0xFF) as u8;
                let b = (pixel & 0xFF) as u8;
                img_buf.push(r);
                img_buf.push(g);
                img_buf.push(b);
            }

            // assets ディレクトリが存在しない場合は作成
            let _ = std::fs::create_dir_all("assets");

            match image::save_buffer_with_format(
                "assets/cpu_screenshot.png",
                &img_buf,
                WIDTH as u32,
                HEIGHT as u32,
                image::ColorType::Rgb8,
                image::ImageFormat::Png,
            ) {
                Ok(_) => println!("Screenshot saved to assets/cpu_screenshot.png"),
                Err(e) => eprintln!("Failed to save screenshot: {}", e),
            }
        }

        if window.is_key_down(Key::W) {
            camera.pos += camera.forward() * move_speed;
        }
        if window.is_key_down(Key::S) {
            camera.pos -= camera.forward() * move_speed;
        }
        if window.is_key_down(Key::A) {
            camera.pos -= camera.right() * move_speed;
        }
        if window.is_key_down(Key::D) {
            camera.pos += camera.right() * move_speed;
        }
        if window.is_key_down(Key::Space) {
            camera.pos += Vec3::new(0.0, move_speed, 0.0);
        }
        if window.is_key_down(Key::LeftShift) {
            camera.pos -= Vec3::new(0.0, move_speed, 0.0);
        }

        if window.is_key_down(Key::Left) {
            camera.rot_y -= rot_speed;
        }
        if window.is_key_down(Key::Right) {
            camera.rot_y += rot_speed;
        }
        if window.is_key_down(Key::Up) {
            camera.rot_x -= rot_speed;
        }
        if window.is_key_down(Key::Down) {
            camera.rot_x += rot_speed;
        }

        // パワー変更
        if window.is_key_pressed(Key::Key1, minifb::KeyRepeat::No) {
            power.store(2, Ordering::Relaxed);
        }
        if window.is_key_pressed(Key::Key2, minifb::KeyRepeat::No) {
            power.store(3, Ordering::Relaxed);
        }
        if window.is_key_pressed(Key::Key3, minifb::KeyRepeat::No) {
            power.store(4, Ordering::Relaxed);
        }
        if window.is_key_pressed(Key::Key4, minifb::KeyRepeat::No) {
            power.store(5, Ordering::Relaxed);
        }
        if window.is_key_pressed(Key::Key5, minifb::KeyRepeat::No) {
            power.store(6, Ordering::Relaxed);
        }
        if window.is_key_pressed(Key::Key6, minifb::KeyRepeat::No) {
            power.store(7, Ordering::Relaxed);
        }
        if window.is_key_pressed(Key::Key7, minifb::KeyRepeat::No) {
            power.store(8, Ordering::Relaxed);
        }
        if window.is_key_pressed(Key::Key8, minifb::KeyRepeat::No) {
            power.store(9, Ordering::Relaxed);
        }
        if window.is_key_pressed(Key::Key9, minifb::KeyRepeat::No) {
            power.store(12, Ordering::Relaxed);
        }

        if window.is_key_pressed(Key::R, minifb::KeyRepeat::No) {
            camera = Camera::new();
            power.store(2, Ordering::Relaxed);
        }

        let current_power = power.load(Ordering::Relaxed) as f32;

        // --- 並列レンダリング ---
        buffer
            .par_chunks_mut(WIDTH)
            .enumerate()
            .for_each(|(y, row)| {
                let v = -((y as f32 / HEIGHT as f32) * 2.0 - 1.0);

                for (x, pixel) in row.iter_mut().enumerate() {
                    let u = (x as f32 / WIDTH as f32) * 2.0 - 1.0;
                    let aspect = WIDTH as f32 / HEIGHT as f32;
                    let u = u * aspect;

                    let ray_dir = camera.get_ray_dir((u, v));
                    *pixel = ray_march(camera.pos, ray_dir, current_power, time);
                }
            });

        window.update_with_buffer(&buffer, WIDTH, HEIGHT).unwrap();

        let elapsed = frame_start.elapsed();
        window.set_title(&format!(
            "Mandelbulb 3D (Power={}) - {:.1} ms ({:.1} fps)",
            current_power as i32,
            elapsed.as_secs_f32() * 1000.0,
            1.0 / elapsed.as_secs_f32().max(0.001)
        ));
    }
}
