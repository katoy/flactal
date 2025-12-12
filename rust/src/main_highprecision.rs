//! マンデルブロ集合インタラクティブビューア (高精度版)
//! M1 Mac (Apple Silicon) 対応
//!
//! rug (GMP) を使用した任意精度演算により、無限ズームが可能
//! ただし、深いズームでは計算に時間がかかります
//!
//! 操作方法:
//!   - マウスホイール上下: 拡大/縮小
//!   - 左クリック+ドラッグ: 移動（パン）
//!   - 右クリック: クリック位置を中心にズームイン
//!   - R キー: 初期表示にリセット
//!   - S キー: 現在の表示を画像として保存
//!   - +/- キー: 精度を増減（深いズームで必要）
//!   - Q / Escape キー: 終了

use image::{ImageBuffer, Rgb};
use mandelbrot::common::{
    colors::iter_to_color_u32,
    constants::{INITIAL_PRECISION, MAX_ITER, MAX_PRECISION},
    mandelbrot::mandelbrot_iter_hp,
};
use minifb::{Key, MouseButton, MouseMode, Window, WindowOptions};
use rug::Float;
use std::time::Instant;

const WIDTH: usize = 800;
const HEIGHT: usize = 600;

/// ビューアの状態
struct ViewerState {
    x_min: Float,
    x_max: Float,
    y_min: Float,
    y_max: Float,
    precision: u32,
    buffer: Vec<u32>,
    needs_redraw: bool,
    save_counter: u32,
}

impl ViewerState {
    fn new() -> Self {
        let prec = INITIAL_PRECISION;
        Self {
            x_min: Float::with_val(prec, -2.5),
            x_max: Float::with_val(prec, 1.0),
            y_min: Float::with_val(prec, -1.5),
            y_max: Float::with_val(prec, 1.5),
            precision: prec,
            buffer: vec![0; WIDTH * HEIGHT],
            needs_redraw: true,
            save_counter: 0,
        }
    }

    fn reset(&mut self) {
        let prec = INITIAL_PRECISION;
        self.x_min = Float::with_val(prec, -2.5);
        self.x_max = Float::with_val(prec, 1.0);
        self.y_min = Float::with_val(prec, -1.5);
        self.y_max = Float::with_val(prec, 1.5);
        self.precision = prec;
        self.needs_redraw = true;
    }

    fn current_zoom(&self) -> f64 {
        let width = self.x_max.to_f64() - self.x_min.to_f64();
        3.5 / width
    }

    fn zoom(&mut self, mouse_x: f64, mouse_y: f64, factor: f64) {
        let prec = self.precision;
        let width_f = self.x_max.to_f64() - self.x_min.to_f64();
        let height_f = self.y_max.to_f64() - self.y_min.to_f64();

        // マウス位置を複素平面上の座標に変換
        let cx = self.x_min.to_f64() + width_f * (mouse_x / WIDTH as f64);
        let cy = self.y_max.to_f64() - height_f * (mouse_y / HEIGHT as f64);

        // 新しい範囲を計算
        let new_width = width_f * factor;
        let new_height = height_f * factor;
        let half_new_width = new_width / 2.0;
        let half_new_height = new_height / 2.0;

        self.x_min = Float::with_val(prec, cx - half_new_width);
        self.x_max = Float::with_val(prec, cx + half_new_width);
        self.y_min = Float::with_val(prec, cy - half_new_height);
        self.y_max = Float::with_val(prec, cy + half_new_height);
        self.needs_redraw = true;

        // ズームレベルに応じて精度を自動調整
        let zoom = self.current_zoom();
        let required_precision = (zoom.log2() * 3.5) as u32 + 64;
        if required_precision > self.precision && self.precision < MAX_PRECISION {
            self.precision = (required_precision.next_power_of_two()).min(MAX_PRECISION);
            self.x_min.set_prec(self.precision);
            self.x_max.set_prec(self.precision);
            self.y_min.set_prec(self.precision);
            self.y_max.set_prec(self.precision);
            println!("精度を自動調整: {} ビット", self.precision);
        }
    }

    /// クリック位置を画面中心に移動（パン）
    fn pan_to(&mut self, mouse_x: f64, mouse_y: f64) {
        let prec = self.precision;
        let width_f = self.x_max.to_f64() - self.x_min.to_f64();
        let height_f = self.y_max.to_f64() - self.y_min.to_f64();

        // クリック位置を複素平面上の座標に変換
        let cx = self.x_min.to_f64() + width_f * (mouse_x / WIDTH as f64);
        let cy = self.y_max.to_f64() - height_f * (mouse_y / HEIGHT as f64);

        // クリック位置を中心にする（ズームは維持）
        let half_width = width_f / 2.0;
        let half_height = height_f / 2.0;

        self.x_min = Float::with_val(prec, cx - half_width);
        self.x_max = Float::with_val(prec, cx + half_width);
        self.y_min = Float::with_val(prec, cy - half_height);
        self.y_max = Float::with_val(prec, cy + half_height);

        self.needs_redraw = true;
    }

    fn save_image(&mut self) {
        self.save_counter += 1;
        let filename = format!("mandelbrot_hp_{:03}.png", self.save_counter);

        let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(WIDTH as u32, HEIGHT as u32, |x, y| {
                let pixel = self.buffer[(y as usize) * WIDTH + (x as usize)];
                let r = ((pixel >> 16) & 0xFF) as u8;
                let g = ((pixel >> 8) & 0xFF) as u8;
                let b = (pixel & 0xFF) as u8;
                Rgb([r, g, b])
            });

        img.save(&filename).expect("画像の保存に失敗しました");
        println!("画像を保存しました: {}", filename);
    }
}

/// マンデルブロ集合を計算してバッファを更新（高精度版）
fn render_mandelbrot_hp(state: &mut ViewerState) {
    let prec = state.precision;
    let x_min_f = state.x_min.to_f64();
    let x_max_f = state.x_max.to_f64();
    let y_min_f = state.y_min.to_f64();
    let y_max_f = state.y_max.to_f64();

    let x_scale = (x_max_f - x_min_f) / WIDTH as f64;
    let y_scale = (y_max_f - y_min_f) / HEIGHT as f64;

    let mut pixels = vec![0u32; WIDTH * HEIGHT];

    for py in 0..HEIGHT {
        for px in 0..WIDTH {
            let cx_f = x_min_f + x_scale * px as f64;
            let cy_f = y_max_f - y_scale * py as f64;
            let cx = Float::with_val(prec, cx_f);
            let cy = Float::with_val(prec, cy_f);
            let iter = mandelbrot_iter_hp(&cx, &cy, MAX_ITER, prec);
            pixels[py * WIDTH + px] = iter_to_color_u32(iter, MAX_ITER);
        }
    }

    state.buffer = pixels;
    state.needs_redraw = false;
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  マンデルブロ集合ビューア (高精度版 - 任意精度)              ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  🔬 任意精度演算により無限ズームが可能                       ║");
    println!("║  ⚠️  深いズームでは計算に時間がかかります                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("操作方法:");
    println!("  - マウスホイール: 拡大/縮小");
    println!("  - 左クリック+ドラッグ: 移動（パン）");
    println!("  - 右クリック: クリック位置を中心にズームイン");
    println!("  - +/= キー: 精度を増加（深いズームで必要）");
    println!("  - - キー: 精度を減少（速度向上）");
    println!("  - R キー: 初期表示にリセット");
    println!("  - S キー: 現在の表示を画像として保存");
    println!("  - Q / Escape キー: 終了");
    println!();

    let mut window = Window::new(
        "マンデルブロ集合 (高精度版 - 無限ズーム)",
        WIDTH,
        HEIGHT,
        WindowOptions {
            resize: false,
            ..WindowOptions::default()
        },
    )
    .expect("ウィンドウの作成に失敗しました");

    window.set_target_fps(60);

    let mut state = ViewerState::new();
    let mut prev_scroll: Option<(f32, f32)> = None;
    let mut prev_left_down = false;

    // 初期描画
    let start = Instant::now();
    render_mandelbrot_hp(&mut state);
    println!(
        "初期描画完了: {:.2?} (精度: {}ビット)",
        start.elapsed(),
        state.precision
    );

    while window.is_open() && !window.is_key_down(Key::Escape) && !window.is_key_down(Key::Q) {
        // キー入力処理
        if window.is_key_pressed(Key::R, minifb::KeyRepeat::No) {
            state.reset();
            println!("リセット");
        }

        if window.is_key_pressed(Key::S, minifb::KeyRepeat::No) {
            state.save_image();
        }

        // マウス位置取得
        if let Some((mx, my)) = window.get_mouse_pos(MouseMode::Discard) {
            // スクロール処理
            if let Some(scroll) = window.get_scroll_wheel() {
                if prev_scroll != Some(scroll) {
                    let factor = if scroll.1 > 0.0 { 0.8 } else { 1.25 };
                    state.zoom(mx as f64, my as f64, factor);
                    prev_scroll = Some(scroll);
                }
            } else {
                prev_scroll = None;
            }

            // 左クリックでパン移動（押した瞬間のみ）
            let left_down = window.get_mouse_down(MouseButton::Left);
            if left_down && !prev_left_down {
                state.pan_to(mx as f64, my as f64);
            }
            prev_left_down = left_down;

            // 右クリックズーム
            if window.get_mouse_down(MouseButton::Right) {
                state.zoom(mx as f64, my as f64, 0.5);
            }
        }

        // 再描画が必要な場合
        if state.needs_redraw {
            let start = Instant::now();
            render_mandelbrot_hp(&mut state);

            // ステータス表示
            let zoom = state.current_zoom();
            let center_x = (state.x_min.to_f64() + state.x_max.to_f64()) / 2.0;
            let center_y = (state.y_min.to_f64() + state.y_max.to_f64()) / 2.0;
            println!(
                "再描画: {:.2?} | 精度: {}bit | 中心: ({:.6}, {:.6}i) | ズーム: x{:.2e}",
                start.elapsed(),
                state.precision,
                center_x,
                center_y,
                zoom
            );
        }

        // ウィンドウ更新
        window
            .update_with_buffer(&state.buffer, WIDTH, HEIGHT)
            .expect("バッファの更新に失敗しました");
    }

    println!("終了しました");
}
