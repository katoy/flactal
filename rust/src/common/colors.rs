//! カラーマップと色変換関数

/// Python版と同じカラーマップ
pub const COLORS: [(f64, f64, f64); 10] = [
    (0.0, 0.0, 0.2), // 深い青
    (0.1, 0.2, 0.5), // 青
    (0.2, 0.5, 0.8), // 水色
    (0.5, 0.8, 0.9), // 薄い水色
    (1.0, 1.0, 0.8), // クリーム
    (1.0, 0.8, 0.3), // 黄色
    (1.0, 0.5, 0.1), // オレンジ
    (0.8, 0.2, 0.1), // 赤
    (0.5, 0.0, 0.2), // 暗い赤
    (0.0, 0.0, 0.0), // 黒
];

/// 反復回数から色を計算（u32形式: 0xRRGGBB）
pub fn iter_to_color_u32(iter: u32, max_iter: u32) -> u32 {
    if iter >= max_iter {
        return 0x000000;
    }

    let t = iter as f64 / max_iter as f64;
    let scaled = t * (COLORS.len() - 1) as f64;
    let idx = (scaled as usize).min(COLORS.len() - 2);
    let frac = scaled - idx as f64;

    let (r1, g1, b1) = COLORS[idx];
    let (r2, g2, b2) = COLORS[idx + 1];

    let r = ((r1 + (r2 - r1) * frac) * 255.0) as u8;
    let g = ((g1 + (g2 - g1) * frac) * 255.0) as u8;
    let b = ((b1 + (b2 - b1) * frac) * 255.0) as u8;

    ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}
