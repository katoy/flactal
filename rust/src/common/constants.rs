//! 共通定数

/// マンデルブロ描画領域の幅
pub const MANDELBROT_WIDTH: usize = 800;
/// マンデルブロ描画領域の高さ
pub const MANDELBROT_HEIGHT: usize = 600;

/// 高精度モード時の低解像度設定
pub const HP_RENDER_WIDTH: usize = 200;
pub const HP_RENDER_HEIGHT: usize = 150;

/// カラーバーの設定
pub const COLORBAR_WIDTH: usize = 60;
pub const COLORBAR_MARGIN: usize = 20;
pub const COLORBAR_BAR_WIDTH: usize = 20;

/// 全体のウィンドウサイズ
pub const WINDOW_WIDTH: usize = MANDELBROT_WIDTH + COLORBAR_WIDTH;
pub const WINDOW_HEIGHT: usize = MANDELBROT_HEIGHT;

/// 最大反復回数
pub const MAX_ITER: u32 = 256;

/// 初期精度（ビット）
pub const INITIAL_PRECISION: u32 = 128;

/// 最大精度（ビット）
pub const MAX_PRECISION: u32 = 4096;

/// 高精度計算モードへの切り替え閾値（ズーム倍率）
pub const PRECISION_THRESHOLD: f64 = 1e13;

/// マウスホイールによるズームアウト倍率
pub const ZOOM_FACTOR_OUT: f64 = 1.25;

/// マウスホイールによるズームイン倍率（右クリックも同様）
pub const ZOOM_FACTOR_IN: f64 = 0.8;
