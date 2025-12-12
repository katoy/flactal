//! ビットマップフォントと描画関数

/// 5x7 ビットマップフォント（0-9）
pub const FONT_5X7: [[u8; 7]; 10] = [
    [
        0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
    ], // 0
    [
        0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
    ], // 1
    [
        0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111,
    ], // 2
    [
        0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110,
    ], // 3
    [
        0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
    ], // 4
    [
        0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
    ], // 5
    [
        0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
    ], // 6
    [
        0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
    ], // 7
    [
        0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
    ], // 8
    [
        0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100,
    ], // 9
];

/// 1文字を描画
pub fn draw_char(
    buffer: &mut [u32],
    buffer_width: usize,
    buffer_height: usize,
    x: usize,
    y: usize,
    c: char,
    color: u32,
) {
    if let Some(digit) = c.to_digit(10) {
        let glyph = &FONT_5X7[digit as usize];
        for (row, &bits) in glyph.iter().enumerate() {
            for col in 0..5 {
                if (bits >> (4 - col)) & 1 == 1 {
                    let px = x + col;
                    let py = y + row;
                    if px < buffer_width && py < buffer_height {
                        buffer[py * buffer_width + px] = color;
                    }
                }
            }
        }
    }
}

/// 文字列を描画
pub fn draw_text(
    buffer: &mut [u32],
    buffer_width: usize,
    buffer_height: usize,
    x: usize,
    y: usize,
    text: &str,
    color: u32,
) {
    let mut cursor_x = x;
    for c in text.chars() {
        draw_char(buffer, buffer_width, buffer_height, cursor_x, y, c, color);
        cursor_x += 6; // 文字幅5 + 間隔1
    }
}
