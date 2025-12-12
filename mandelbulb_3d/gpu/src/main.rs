//! マンデルバルブ (Mandelbulb) 3Dフラクタルビューア - GPU版
//! wgpu + WGSLフラグメントシェーダーによるGPUレンダリング
//!
//! 操作方法:
//!   - W/A/S/D: カメラ移動 (前後左右)
//!   - Space/LShift: カメラ移動 (上昇/下降)
//!   - 矢印キー: カメラ回転
//!   - 1-9: パワー変更 (形状が変化)
//!   - R: リセット
//!   - Esc/Q: 終了

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Vec2, Vec3, Vec4};
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::{
    dpi::PhysicalSize,
    event::{ElementState, Event, KeyEvent, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::WindowBuilder,
};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    camera_pos_power: Vec4, // xyz: camera_pos, w: power
    rotation: Vec2,         // x: rot_x, y: rot_y
    time: f32,
    aspect: f32,
}

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

    fn forward(&self) -> Vec3 {
        let rot = Mat3::from_rotation_y(self.rot_y) * Mat3::from_rotation_x(self.rot_x);
        rot * Vec3::new(0.0, 0.0, 1.0)
    }

    fn right(&self) -> Vec3 {
        let rot = Mat3::from_rotation_y(self.rot_y);
        rot * Vec3::new(1.0, 0.0, 0.0)
    }

    fn move_forward(&mut self, amount: f32) {
        self.pos += self.forward() * amount;
    }

    fn move_right(&mut self, amount: f32) {
        self.pos += self.right() * amount;
    }
}

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Mandelbulb 3D GPU Explorer")
            .with_inner_size(PhysicalSize::new(WIDTH, HEIGHT))
            .with_resizable(false)
            .build(&event_loop)
            .unwrap(),
    );

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let surface = instance.create_surface(window.clone()).unwrap();

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    }))
    .expect("Failed to find GPU adapter");

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        },
        None,
    ))
    .expect("Failed to create device");

    let surface_caps = surface.get_capabilities(&adapter);
    let surface_format = surface_caps.formats[0];

    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        format: surface_format,
        width: WIDTH,
        height: HEIGHT,
        present_mode: wgpu::PresentMode::AutoVsync,
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &config);

    // シェーダー読み込み
    let shader_source = include_str!("../shaders/mandelbulb.wgsl");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // パラメータバッファ
    let mut camera = Camera::new();
    let mut power = 2.0f32;
    // let start_time = Instant::now(); // 不要

    let params = Params {
        camera_pos_power: Vec4::new(camera.pos.x, camera.pos.y, camera.pos.z, power),
        rotation: Vec2::new(camera.rot_x, camera.rot_y),
        time: 0.0, // アニメーション停止
        aspect: WIDTH as f32 / HEIGHT as f32,
    };

    let param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Params Buffer"),
        contents: bytemuck::cast_slice(&[params]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // バインドグループレイアウト
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: param_buffer.as_entire_binding(),
        }],
    });

    // レンダーパイプライン
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    // キー状態
    let mut keys_pressed = std::collections::HashSet::new();

    println!("=== Mandelbulb 3D GPU Explorer ===");
    println!("  Move: W/A/S/D + Space/Shift");
    println!("  Look: Arrow Keys");
    println!("  Power: 1-9 keys");
    println!("  Screenshot: P");
    println!("  Reset: R");

    let _ = event_loop.run(move |event, elwt| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => elwt.exit(),
            WindowEvent::Focused(false) => {
                keys_pressed.clear();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state,
                        ..
                    },
                ..
            } => match state {
                ElementState::Pressed => {
                    keys_pressed.insert(key);

                    match key {
                        KeyCode::Escape | KeyCode::KeyQ => elwt.exit(),
                        KeyCode::KeyR => {
                            camera = Camera::new();
                            power = 2.0;
                        }
                        KeyCode::Digit1 => power = 2.0,
                        KeyCode::Digit2 => power = 3.0,
                        KeyCode::Digit3 => power = 4.0,
                        KeyCode::Digit4 => power = 5.0,
                        KeyCode::Digit5 => power = 6.0,
                        KeyCode::Digit6 => power = 7.0,
                        KeyCode::Digit7 => power = 8.0,
                        KeyCode::Digit8 => power = 9.0,
                        KeyCode::Digit9 => power = 12.0,
                        _ => {}
                    }
                }
                ElementState::Released => {
                    keys_pressed.remove(&key);
                }
            },
            WindowEvent::RedrawRequested => {
                let frame_start = Instant::now();

                // 入力処理
                let move_speed = 0.05;
                let rot_speed = 0.05;

                if keys_pressed.contains(&KeyCode::KeyW) {
                    camera.move_forward(move_speed);
                }
                if keys_pressed.contains(&KeyCode::KeyS) {
                    camera.move_forward(-move_speed);
                }
                if keys_pressed.contains(&KeyCode::KeyA) {
                    camera.move_right(-move_speed);
                }
                if keys_pressed.contains(&KeyCode::KeyD) {
                    camera.move_right(move_speed);
                }
                if keys_pressed.contains(&KeyCode::Space) {
                    camera.pos.y += move_speed;
                }
                if keys_pressed.contains(&KeyCode::ShiftLeft) {
                    camera.pos.y -= move_speed;
                }
                if keys_pressed.contains(&KeyCode::ArrowLeft) {
                    camera.rot_y -= rot_speed;
                }
                if keys_pressed.contains(&KeyCode::ArrowRight) {
                    camera.rot_y += rot_speed;
                }
                if keys_pressed.contains(&KeyCode::ArrowUp) {
                    camera.rot_x -= rot_speed;
                }
                if keys_pressed.contains(&KeyCode::ArrowDown) {
                    camera.rot_x += rot_speed;
                }

                // パラメータ更新
                let params = Params {
                    camera_pos_power: Vec4::new(camera.pos.x, camera.pos.y, camera.pos.z, power),
                    rotation: Vec2::new(camera.rot_x, camera.rot_y),
                    time: 0.0,
                    aspect: WIDTH as f32 / HEIGHT as f32,
                };
                queue.write_buffer(&param_buffer, 0, bytemuck::cast_slice(&[params]));

                // レンダリング
                let output = match surface.get_current_texture() {
                    Ok(t) => t,
                    Err(_) => {
                        surface.configure(&device, &config);
                        return;
                    }
                };
                let view = output
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });

                {
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Render Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    render_pass.set_pipeline(&render_pipeline);
                    render_pass.set_bind_group(0, &bind_group, &[]);
                    render_pass.draw(0..3, 0..1);
                }

                if keys_pressed.contains(&KeyCode::KeyP) {
                    let u32_size = std::mem::size_of::<u32>() as u32;
                    let texture_width = config.width;
                    let texture_height = config.height;
                    let bytes_per_row = u32_size * texture_width;
                    let padded_bytes_per_row = (bytes_per_row + 255) & !255;

                    let buffer_size = (padded_bytes_per_row * texture_height) as u64;

                    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("Screenshot Buffer"),
                        size: buffer_size,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    });

                    encoder.copy_texture_to_buffer(
                        wgpu::ImageCopyTexture {
                            texture: &output.texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::ImageCopyBuffer {
                            buffer: &buffer,
                            layout: wgpu::ImageDataLayout {
                                offset: 0,
                                bytes_per_row: Some(padded_bytes_per_row),
                                rows_per_image: Some(texture_height),
                            },
                        },
                        wgpu::Extent3d {
                            width: texture_width,
                            height: texture_height,
                            depth_or_array_layers: 1,
                        },
                    );

                    queue.submit(std::iter::once(encoder.finish()));

                    let slice = buffer.slice(..);
                    slice.map_async(wgpu::MapMode::Read, move |_| {});
                    device.poll(wgpu::Maintain::Wait);

                    let data = slice.get_mapped_range();

                    let mut img_buf =
                        Vec::with_capacity((texture_width * texture_height * 4) as usize);
                    for chunk in data.chunks(padded_bytes_per_row as usize) {
                        img_buf.extend_from_slice(&chunk[..bytes_per_row as usize]);
                    }

                    for pixel in img_buf.chunks_exact_mut(4) {
                        pixel.swap(0, 2);
                    }

                    let _ = std::fs::create_dir_all("../assets");

                    match image::save_buffer_with_format(
                        "../assets/gpu_screenshot.png",
                        &img_buf,
                        texture_width,
                        texture_height,
                        image::ColorType::Rgba8,
                        image::ImageFormat::Png,
                    ) {
                        Ok(_) => println!("Screenshot saved to assets/gpu_screenshot.png"),
                        Err(e) => eprintln!("Failed to save screenshot: {}", e),
                    }

                    buffer.unmap();
                } else {
                    queue.submit(std::iter::once(encoder.finish()));
                }

                output.present();

                let elapsed = frame_start.elapsed();
                window.set_title(&format!(
                    "Mandelbulb 3D GPU (Power={}) - {:.1} ms ({:.1} fps)",
                    power as i32,
                    elapsed.as_secs_f32() * 1000.0,
                    1.0 / elapsed.as_secs_f32().max(0.001)
                ));

                window.request_redraw();
            }
            _ => {}
        },
        Event::AboutToWait => {
            window.request_redraw();
        }
        _ => {}
    });
}
