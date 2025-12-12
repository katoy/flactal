// 頂点シェーダー - フルスクリーンクワッド
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    
    // フルスクリーンの三角形（2つで画面全体をカバー）
    let x = f32(vertex_index & 1u) * 4.0 - 1.0;
    let y = f32((vertex_index >> 1u) & 1u) * 4.0 - 1.0;
    
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    
    return out;
}

// Uniform バッファ
struct Params {
    camera_pos_power: vec4<f32>, // xyz: pos, w: power
    rotation: vec2<f32>,         // x: rot_x, y: rot_y
    time: f32,
    aspect: f32,
}

@group(0) @binding(0) var<uniform> params: Params;

const MAX_STEPS: u32 = 100u;
const MAX_ITER: u32 = 10u;
const BAILOUT: f32 = 2.0;
const EPSILON: f32 = 0.001;

// HSVからRGBへの変換
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    var hue = fract(h);
    if (hue < 0.0) { hue = hue + 1.0; }
    
    let i = floor(hue * 6.0);
    let f = hue * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);

    let ii = i32(i) % 6;
    if (ii == 0) { return vec3<f32>(v, t, p); }
    if (ii == 1) { return vec3<f32>(q, v, p); }
    if (ii == 2) { return vec3<f32>(p, v, t); }
    if (ii == 3) { return vec3<f32>(p, q, v); }
    if (ii == 4) { return vec3<f32>(t, p, v); }
    return vec3<f32>(v, p, q);
}

// マンデルバルブ距離関数
fn map_with_iter(pos: vec3<f32>, power: f32) -> vec3<f32> {
    var z = pos;
    var dr = 1.0;
    var r = 0.0;
    var trap = 1e10;
    var iterations = 0u;

    for (var iter = 0u; iter < MAX_ITER; iter = iter + 1u) {
        r = length(z);
        if (r > BAILOUT) {
            iterations = iter;
            break;
        }
        iterations = iter;

        trap = min(trap, length(z));
        dr = pow(r, power - 1.0) * power * dr + 1.0;

        let theta = atan2(z.z, sqrt(z.x * z.x + z.y * z.y));
        let phi = atan2(z.y, z.x);

        let zr = pow(r, power);
        let new_theta = theta * power;
        let new_phi = phi * power;

        z = vec3<f32>(
            zr * cos(new_theta) * cos(new_phi),
            zr * cos(new_theta) * sin(new_phi),
            zr * sin(new_theta)
        );

        z = z + pos;
    }

    let dist = 0.5 * log(r) * r / dr;
    return vec3<f32>(dist, f32(iterations), trap);
}

fn map(pos: vec3<f32>, power: f32) -> f32 {
    return map_with_iter(pos, power).x;
}

// 法線計算
fn calc_normal(p: vec3<f32>, power: f32) -> vec3<f32> {
    let e = EPSILON;
    let n = vec3<f32>(
        map(p + vec3<f32>(e, 0.0, 0.0), power) - map(p - vec3<f32>(e, 0.0, 0.0), power),
        map(p + vec3<f32>(0.0, e, 0.0), power) - map(p - vec3<f32>(0.0, e, 0.0), power),
        map(p + vec3<f32>(0.0, 0.0, e), power) - map(p - vec3<f32>(0.0, 0.0, e), power)
    );
    return normalize(n);
}

// ベクトル回転
fn rotate_x(v: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec3<f32>(v.x, v.y * c - v.z * s, v.y * s + v.z * c);
}

fn rotate_y(v: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec3<f32>(v.x * c - v.z * s, v.y, v.x * s + v.z * c);
}

// フラグメントシェーダー
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let u = (in.uv.x * 2.0 - 1.0) * params.aspect;
    let v = -(in.uv.y * 2.0 - 1.0);
    
    var dir = normalize(vec3<f32>(u, v, 1.0));
    dir = rotate_x(dir, params.rotation.x);
    dir = rotate_y(dir, params.rotation.y);
    
    let camera_pos = params.camera_pos_power.xyz;
    let power = params.camera_pos_power.w;
    
    // レイマーチング
    var t = 0.0;
    var hit = false;
    var steps = 0u;
    var total_iter = 0u;
    var min_trap = 1e10;
    
    for (var i = 0u; i < MAX_STEPS; i = i + 1u) {
        let p = camera_pos + dir * t;
        let result = map_with_iter(p, power);
        let d = result.x;
        total_iter = u32(result.y);
        min_trap = min(min_trap, result.z);
        
        if (d < EPSILON) {
            hit = true;
            steps = i;
            break;
        }
        
        t = t + d * 0.8;
        if (t > 6.0) {
            break;
        }
    }
    
    if (hit) {
        let p = camera_pos + dir * t;
        let normal = calc_normal(p, power);
        
        let light1 = normalize(vec3<f32>(0.577, 0.577, -0.577));
        let light2 = normalize(vec3<f32>(-0.5, 0.8, 0.3));
        
        let diff1 = max(dot(normal, light1), 0.0);
        let diff2 = max(dot(normal, light2), 0.0) * 0.5;
        
        let view_dir = -dir;
        let reflect_dir = 2.0 * dot(normal, light1) * normal - light1;
        let spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0);
        
        let ao = 1.0 - pow(f32(steps) / f32(MAX_STEPS), 0.4);
        
        let hue1 = f32(total_iter) / f32(MAX_ITER) + params.time * 0.1;
        let hue2 = (normal.x + normal.y * 0.5 + 1.0) * 0.5;
        let hue3 = min_trap * 2.0;
        let hue4 = (p.x + p.y + p.z) * 0.3;
        
        let final_hue = fract(hue1 * 0.4 + hue2 * 0.2 + hue3 * 0.2 + hue4 * 0.2);
        let saturation = 0.8 + (1.0 - ao) * 0.2;
        let value = min((diff1 + diff2 + 0.15) * ao, 1.0);
        
        var rgb = hsv_to_rgb(final_hue, saturation, value);
        rgb = rgb + vec3<f32>(spec * 0.5);
        rgb = min(rgb, vec3<f32>(1.0));
        
        return vec4<f32>(rgb, 1.0);
    } else {
        let gradient = (dir.y + 1.0) * 0.5;
        let bg_hue = 0.6 + params.time * 0.02;
        let rgb = hsv_to_rgb(bg_hue, 0.5, gradient * 0.15 + 0.02);
        return vec4<f32>(rgb, 1.0);
    }
}
