//! マンデルブロ集合インタラクティブビューア (ハイブリッド版)
//! M1 Mac (Apple Silicon) 対応
//!
//! ズームレベルに応じて自動的に計算精度を切り替え:
//!   - 浅いズーム（〜10^13倍）: f64 + Rayon並列処理（超高速）
//!   - 深いズーム（10^13倍〜）: rug任意精度（無限ズーム）
//!
//! 操作方法:
//!   - マウスホイール上下: 拡大/縮小
//!   - 左クリック+ドラッグ: 移動（パン）
//!   - 右クリック: クリック位置を中心にズームイン
//!   - R キー: 初期表示にリセット
//!   - S キー: 現在の表示を画像として保存
//!   - Q / Escape キー: 終了

use image::{ImageBuffer, Rgb};
use mandelbrot::common::{
    colors::iter_to_color_u32,
    constants::*,
    font::draw_text,
    mandelbrot::{mandelbrot_iter_fast, mandelbrot_iter_hp},
};
use minifb::{Key, MouseButton, MouseMode, Window, WindowOptions};
use num_complex::Complex;
use rayon::prelude::*;
use rug::Float;
use std::time::Instant;

/// 計算モード
#[derive(Clone, Copy, PartialEq)]
enum ComputeMode {
    Fast,
    HighPrecision,
}

impl std::fmt::Display for ComputeMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComputeMode::Fast => write!(f, "🚀 高速 (f64)"),
            ComputeMode::HighPrecision => write!(f, "🔬 高精度 (任意精度)"),
        }
    }
}

/// ビューアの状態
struct ViewerState {
    x_min: Float,
    x_max: Float,
    y_min: Float,
    y_max: Float,
    precision: u32,
    compute_mode: ComputeMode,
    buffer: Vec<u32>,            // ウィンドウ全体のバッファ
    mandelbrot_buffer: Vec<u32>, // マンデルブロ部分のみ
    needs_redraw: bool,
    save_counter: u32,
}

impl ViewerState {
    fn new() -> Self {
        let prec = INITIAL_PRECISION;
        let mut state = Self {
            x_min: Float::with_val(prec, -2.5),
            x_max: Float::with_val(prec, 1.0),
            y_min: Float::with_val(prec, -1.5),
            y_max: Float::with_val(prec, 1.5),
            precision: prec,
            compute_mode: ComputeMode::Fast,
            buffer: vec![0; WINDOW_WIDTH * WINDOW_HEIGHT],
            mandelbrot_buffer: vec![0; MANDELBROT_WIDTH * MANDELBROT_HEIGHT],
            needs_redraw: true,
            save_counter: 0,
        };
        state.draw_colorbar();
        state
    }

    fn reset(&mut self) {
        let prec = INITIAL_PRECISION;
        self.x_min = Float::with_val(prec, -2.5);
        self.x_max = Float::with_val(prec, 1.0);
        self.y_min = Float::with_val(prec, -1.5);
        self.y_max = Float::with_val(prec, 1.5);
        self.precision = prec;
        self.compute_mode = ComputeMode::Fast;
        self.needs_redraw = true;
    }

    fn current_zoom(&self) -> f64 {
        let width = self.x_max.to_f64() - self.x_min.to_f64();
        3.5 / width
    }

    fn update_compute_mode(&mut self) {
        let zoom = self.current_zoom();
        let old_mode = self.compute_mode;

        if zoom > PRECISION_THRESHOLD {
            self.compute_mode = ComputeMode::HighPrecision;
            let required_precision = (zoom.log2() * 3.5) as u32 + 64;
            if required_precision > self.precision && self.precision < MAX_PRECISION {
                self.precision = (required_precision.next_power_of_two()).min(MAX_PRECISION);
                self.x_min.set_prec(self.precision);
                self.x_max.set_prec(self.precision);
                self.y_min.set_prec(self.precision);
                self.y_max.set_prec(self.precision);
            }
        } else {
            self.compute_mode = ComputeMode::Fast;
        }

        if old_mode != self.compute_mode {
            println!("モード切替: {} → {}", old_mode, self.compute_mode);
        }
    }

    /// 画面上のピクセル座標を複素平面上の座標に変換
    fn pixel_to_complex(&self, x: f64, y: f64) -> (f64, f64) {
        let width_f = self.x_max.to_f64() - self.x_min.to_f64();
        let height_f = self.y_max.to_f64() - self.y_min.to_f64();

        let cx = self.x_min.to_f64() + width_f * (x / MANDELBROT_WIDTH as f64);
        let cy = self.y_max.to_f64() - height_f * (y / MANDELBROT_HEIGHT as f64);
        (cx, cy)
    }

    /// 指定された中心座標と現在のズーム倍率で範囲を更新
    fn update_bounds(&mut self, center_x: f64, center_y: f64, width_scale: f64) {
        let prec = self.precision;
        let width_f = self.x_max.to_f64() - self.x_min.to_f64();
        let height_f = self.y_max.to_f64() - self.y_min.to_f64();

        let new_width = width_f * width_scale;
        let new_height = height_f * width_scale;
        let half_new_width = new_width / 2.0;
        let half_new_height = new_height / 2.0;

        self.x_min = Float::with_val(prec, center_x - half_new_width);
        self.x_max = Float::with_val(prec, center_x + half_new_width);
        self.y_min = Float::with_val(prec, center_y - half_new_height);
        self.y_max = Float::with_val(prec, center_y + half_new_height);

        self.update_compute_mode();
        self.needs_redraw = true;
    }

    fn zoom(&mut self, mouse_x: f64, mouse_y: f64, factor: f64) {
        // カラーバー領域では無視
        if mouse_x >= MANDELBROT_WIDTH as f64 {
            return;
        }

        let (cx, cy) = self.pixel_to_complex(mouse_x, mouse_y);
        self.update_bounds(cx, cy, factor);
    }

    /// クリック位置を画面中心に移動（パン）
    fn pan_to(&mut self, mouse_x: f64, mouse_y: f64) {
        // カラーバー領域では無視
        if mouse_x >= MANDELBROT_WIDTH as f64 {
            return;
        }

        let (cx, cy) = self.pixel_to_complex(mouse_x, mouse_y);
        self.update_bounds(cx, cy, 1.0); // 倍率は1.0（変えない）
    }

    /// カラーバーを描画
    fn draw_colorbar(&mut self) {
        let bar_x_start = MANDELBROT_WIDTH + COLORBAR_MARGIN;
        let bar_x_end = bar_x_start + COLORBAR_BAR_WIDTH;
        let bar_y_start = 40;
        let bar_y_end = MANDELBROT_HEIGHT - 40;
        let bar_height = bar_y_end - bar_y_start;

        // 背景をグレーに
        for y in 0..WINDOW_HEIGHT {
            for x in MANDELBROT_WIDTH..WINDOW_WIDTH {
                self.buffer[y * WINDOW_WIDTH + x] = 0x404040;
            }
        }

        // カラーバー本体を描画
        for y in bar_y_start..bar_y_end {
            let t = 1.0 - (y - bar_y_start) as f64 / bar_height as f64;
            let iter = (t * MAX_ITER as f64) as u32;
            let color = iter_to_color_u32(iter, MAX_ITER);

            for x in bar_x_start..bar_x_end {
                self.buffer[y * WINDOW_WIDTH + x] = color;
            }
        }

        // 枠線
        let border_color = 0xFFFFFF;
        for x in bar_x_start..bar_x_end {
            self.buffer[(bar_y_start - 1) * WINDOW_WIDTH + x] = border_color;
            self.buffer[bar_y_end * WINDOW_WIDTH + x] = border_color;
        }
        for y in (bar_y_start - 1)..=bar_y_end {
            self.buffer[y * WINDOW_WIDTH + bar_x_start - 1] = border_color;
            self.buffer[y * WINDOW_WIDTH + bar_x_end] = border_color;
        }

        // 目盛りとラベルを描画
        let tick_values = [0, 64, 128, 192, 256];
        for &value in &tick_values {
            let t = value as f64 / MAX_ITER as f64;
            let y = bar_y_end - (t * bar_height as f64) as usize;

            // 目盛り線
            for x in bar_x_end..(bar_x_end + 5) {
                if y < WINDOW_HEIGHT {
                    self.buffer[y * WINDOW_WIDTH + x] = 0xFFFFFF;
                }
            }

            // 数値ラベルを描画
            let label = format!("{}", value);
            let label_x = bar_x_end + 7;
            let label_y = y.saturating_sub(3);
            draw_text(
                &mut self.buffer,
                WINDOW_WIDTH,
                WINDOW_HEIGHT,
                label_x,
                label_y,
                &label,
                0xCCCCCC,
            );
        }
    }

    /// マンデルブロ画像とカラーバーを合成
    fn compose_buffer(&mut self) {
        for y in 0..MANDELBROT_HEIGHT {
            for x in 0..MANDELBROT_WIDTH {
                self.buffer[y * WINDOW_WIDTH + x] =
                    self.mandelbrot_buffer[y * MANDELBROT_WIDTH + x];
            }
        }
    }

    fn save_image(&mut self) {
        self.save_counter += 1;
        let filename = format!("mandelbrot_{:03}.png", self.save_counter);

        let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(WINDOW_WIDTH as u32, WINDOW_HEIGHT as u32, |x, y| {
                let pixel = self.buffer[(y as usize) * WINDOW_WIDTH + (x as usize)];
                let r = ((pixel >> 16) & 0xFF) as u8;
                let g = ((pixel >> 8) & 0xFF) as u8;
                let b = (pixel & 0xFF) as u8;
                Rgb([r, g, b])
            });

        img.save(&filename).expect("画像の保存に失敗しました");
        println!("画像を保存しました: {}", filename);
    }
}

// ===== f64高速版の計算 =====

fn render_fast(state: &mut ViewerState) {
    let x_min = state.x_min.to_f64();
    let x_max = state.x_max.to_f64();
    let y_min = state.y_min.to_f64();
    let y_max = state.y_max.to_f64();

    let x_scale = (x_max - x_min) / MANDELBROT_WIDTH as f64;
    let y_scale = (y_max - y_min) / MANDELBROT_HEIGHT as f64;

    let pixels: Vec<u32> = (0..MANDELBROT_HEIGHT)
        .into_par_iter()
        .flat_map(|y| {
            (0..MANDELBROT_WIDTH)
                .map(|x| {
                    let cx = x_min + x as f64 * x_scale;
                    let cy = y_max - y as f64 * y_scale;
                    let c = Complex::new(cx, cy);
                    let iter = mandelbrot_iter_fast(c, MAX_ITER);
                    iter_to_color_u32(iter, MAX_ITER)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    state.mandelbrot_buffer = pixels;
}

// ===== 高精度版の計算 =====

fn render_high_precision(state: &mut ViewerState) {
    let prec = state.precision;
    let x_min_f = state.x_min.to_f64();
    let x_max_f = state.x_max.to_f64();
    let y_min_f = state.y_min.to_f64();
    let y_max_f = state.y_max.to_f64();

    // 低解像度で計算
    let x_scale = (x_max_f - x_min_f) / HP_RENDER_WIDTH as f64;
    let y_scale = (y_max_f - y_min_f) / HP_RENDER_HEIGHT as f64;

    let mut low_res_pixels = vec![0u32; HP_RENDER_WIDTH * HP_RENDER_HEIGHT];

    // 背景を初期化
    let offset_x = (MANDELBROT_WIDTH - HP_RENDER_WIDTH) / 2;
    let offset_y = (MANDELBROT_HEIGHT - HP_RENDER_HEIGHT) / 2;
    state.mandelbrot_buffer = vec![0x202020u32; MANDELBROT_WIDTH * MANDELBROT_HEIGHT];

    // プログレスバー更新頻度調整: 全体の1%ごとに更新 (ただし最低1回)
    let update_interval = std::cmp::max(1, HP_RENDER_HEIGHT / 100);

    for py in 0..HP_RENDER_HEIGHT {
        // 計算
        for px in 0..HP_RENDER_WIDTH {
            let cx_f = x_min_f + x_scale * px as f64;
            let cy_f = y_max_f - y_scale * py as f64;
            let cx = Float::with_val(prec, cx_f);
            let cy = Float::with_val(prec, cy_f);
            let iter = mandelbrot_iter_hp(&cx, &cy, MAX_ITER, prec);
            low_res_pixels[py * HP_RENDER_WIDTH + px] = iter_to_color_u32(iter, MAX_ITER);

            // 現在の行を即座に描画
            let dest_x = offset_x + px;
            let dest_y = offset_y + py;
            state.mandelbrot_buffer[dest_y * MANDELBROT_WIDTH + dest_x] =
                low_res_pixels[py * HP_RENDER_WIDTH + px];
        }

        // コンソールにプログレスバーを表示 (間引いて更新)
        if py % update_interval == 0 || py == HP_RENDER_HEIGHT - 1 {
            let progress = (py + 1) as f64 / HP_RENDER_HEIGHT as f64;
            let bar_width = 30;
            let filled = (progress * bar_width as f64) as usize;
            let empty = bar_width - filled;
            print!(
                "\r🔬 計算中: [{}{}] {:>3}%",
                "█".repeat(filled),
                "░".repeat(empty),
                ((py + 1) * 100 / HP_RENDER_HEIGHT)
            );
            use std::io::Write;
            std::io::stdout().flush().ok();
        }
    }
    println!(" 完了!");
}

fn render_mandelbrot(state: &mut ViewerState) {
    match state.compute_mode {
        ComputeMode::Fast => render_fast(state),
        ComputeMode::HighPrecision => render_high_precision(state),
    }
    state.compose_buffer();
    state.needs_redraw = false;
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  マンデルブロ集合ビューア (ハイブリッド版)                   ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  🚀 浅いズーム: f64 + 並列処理（超高速）                     ║");
    println!("║  🔬 深いズーム: 任意精度（自動切替、無限ズーム可能）         ║");
    println!("║  切替閾値: 10^13倍                                           ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("操作方法:");
    println!("  - マウスホイール: 拡大/縮小");
    println!("  - 左クリック+ドラッグ: 移動（パン）");
    println!("  - 右クリック: クリック位置を中心にズームイン");
    println!("  - R キー: 初期表示にリセット");
    println!("  - S キー: 現在の表示を画像として保存");
    println!("  - Q / Escape キー: 終了");
    println!();

    let mut window = Window::new(
        "マンデルブロ集合 (ハイブリッド版 - 自動精度切替)",
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        WindowOptions {
            resize: false,
            ..WindowOptions::default()
        },
    )
    .expect("ウィンドウの作成に失敗しました");

    window.set_target_fps(60);

    let mut state = ViewerState::new();
    let mut prev_scroll: Option<(f32, f32)> = None;

    // 初期描画
    let start = Instant::now();
    render_mandelbrot(&mut state);
    println!(
        "初期描画完了: {:.2?} [{}]",
        start.elapsed(),
        state.compute_mode
    );

    let mut prev_left_down = false;

    while window.is_open() && !window.is_key_down(Key::Escape) && !window.is_key_down(Key::Q) {
        if window.is_key_pressed(Key::R, minifb::KeyRepeat::No) {
            state.reset();
            println!("リセット");
        }

        if window.is_key_pressed(Key::S, minifb::KeyRepeat::No) {
            state.save_image();
        }

        if let Some((mx, my)) = window.get_mouse_pos(MouseMode::Discard) {
            if let Some(scroll) = window.get_scroll_wheel() {
                if prev_scroll != Some(scroll) {
                    let factor = if scroll.1 > 0.0 { ZOOM_FACTOR_IN } else { ZOOM_FACTOR_OUT };
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

            if window.get_mouse_down(MouseButton::Right) {
                state.zoom(mx as f64, my as f64, ZOOM_FACTOR_IN);
            }
        }

        if state.needs_redraw {
            let start = Instant::now();
            render_mandelbrot(&mut state);

            let zoom = state.current_zoom();
            let center_x = (state.x_min.to_f64() + state.x_max.to_f64()) / 2.0;
            let center_y = (state.y_min.to_f64() + state.y_max.to_f64()) / 2.0;

            let mode_info = match state.compute_mode {
                ComputeMode::Fast => "🚀".to_string(),
                ComputeMode::HighPrecision => format!("🔬 {}bit", state.precision),
            };

            // ウィンドウタイトルを更新してモードを表示（テキストのみ）
            let title_mode = match state.compute_mode {
                ComputeMode::Fast => "CPU".to_string(),
                ComputeMode::HighPrecision => format!("HP {}bit", state.precision),
            };
            let title = format!("マンデルブロ集合 [{}] x{:.2e}", title_mode, zoom);
            window.set_title(&title);

            println!(
                "再描画: {:.2?} {} | 中心: ({:.6}, {:.6}i) | ズーム: x{:.2e}",
                start.elapsed(),
                mode_info,
                center_x,
                center_y,
                zoom
            );
        }

        window
            .update_with_buffer(&state.buffer, WINDOW_WIDTH, WINDOW_HEIGHT)
            .expect("バッファの更新に失敗しました");
    }

    println!("終了しました");
}