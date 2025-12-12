// マンデルブロ集合計算シェーダー (WGSL)
// 各ピクセルの反復回数をGPUで並列計算する

struct Params {
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
    width: u32,
    height: u32,
    max_iter: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    // ピクセル座標を複素数座標に変換
    let x_scale = (params.x_max - params.x_min) / f32(params.width);
    let y_scale = (params.y_max - params.y_min) / f32(params.height);
    
    let c_real = params.x_min + f32(x) * x_scale;
    let c_imag = params.y_max - f32(y) * y_scale;
    
    // マンデルブロ反復計算
    var z_real: f32 = 0.0;
    var z_imag: f32 = 0.0;
    var iter: u32 = 0u;
    
    for (var i: u32 = 0u; i < params.max_iter; i = i + 1u) {
        let zr2 = z_real * z_real;
        let zi2 = z_imag * z_imag;
        
        if (zr2 + zi2 > 4.0) {
            break;
        }
        
        z_imag = 2.0 * z_real * z_imag + c_imag;
        z_real = zr2 - zi2 + c_real;
        iter = i + 1u;
    }
    
    // 結果を出力バッファに書き込み
    let idx = y * params.width + x;
    output[idx] = iter;
}
