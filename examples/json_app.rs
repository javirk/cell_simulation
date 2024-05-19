use std::{time::{SystemTime, UNIX_EPOCH}, collections::{HashMap, VecDeque}};
use winit::{
    event::{Event, WindowEvent, KeyboardInput, VirtualKeyCode, ElementState, MouseScrollDelta},
    event_loop::{ControlFlow, EventLoop, },
    window::Window,
    dpi::LogicalSize,
};
use std::time::Instant;
use imgui::*;
use simulation::{Simulation, Setup, UniformBuffer, RenderParams, Render3D};
use imgui_wgpu::{Renderer, RendererConfig};

use crate::statistics::StatisticContainer;


fn setup_imgui(window: &Window) -> (imgui::Context, imgui_winit_support::WinitPlatform) {
    let hidpi_factor = window.scale_factor();
    let mut imgui = imgui::Context::create();
    let mut platform = imgui_winit_support::WinitPlatform::init(&mut imgui);
    platform.attach_window(
        imgui.io_mut(),
        window,
        imgui_winit_support::HiDpiMode::Default,
    );
    imgui.set_ini_filename(None);

    let font_size = (13.0 * hidpi_factor) as f32;
    imgui.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;

    imgui.fonts().add_font(&[FontSource::DefaultFontData {
        config: Some(imgui::FontConfig {
            oversample_h: 1,
            pixel_snap_h: true,
            size_pixels: font_size,
            ..Default::default()
        }),
    }]);
    (imgui, platform)
}

struct CellSimulation {
    simulation: Simulation,
    renderer: Render3D,
    uniform_buffer: UniformBuffer,
    all_stats: HashMap<String, StatisticContainer>
}

// #[derive(Debug)]
// struct StatisticContainer {
//     x: VecDeque<u32>,
//     y: VecDeque<f32>,  // Must be f32 for the UI
// }

// impl StatisticContainer {

//     fn new(capacity: usize) -> Self {
//         StatisticContainer { x: VecDeque::with_capacity(capacity), y: VecDeque::with_capacity(capacity) }
//     }

//     fn add(&mut self, x: u32, y: f32) {
//         while self.x.capacity() == self.x.len() {
//             self.x.pop_front();
//             self.y.pop_front();
//         }
//         self.x.push_back(x);
//         self.y.push_back(y);
//     }

//     #[allow(dead_code)]
//     fn mean(&self) -> f32 {
//         let mut sum = 0.;
//         for y in &self.y {
//             sum += y;
//         }
//         sum / self.y.len() as f32
//     }

//     fn last(&self) -> f32 {
//         match self.y.back() {
//             Some(y) => *y,
//             None => 0.
//         }
//     }
// }

// fn make_all_stats(metrics_log: Vec<&str>) -> HashMap<String, StatisticContainer> {
//     let mut all_stats = HashMap::new();
//     for metric in metrics_log {
//         all_stats.insert(metric.to_string(), StatisticContainer::new(100));
//     }
//     all_stats
// }


fn setup_system(state: &Setup, device: &wgpu::Device) -> CellSimulation {
    let uniform_buffer = UniformBuffer::new(device);

    let render_params = RenderParams::from_file("saved_models/easy.json", device).unwrap();
    let (simulation, texture) = Simulation::from_file("saved_models/easy.json", device).unwrap();

    let renderer = Render3D::new(&texture, &simulation.lattice_params, &render_params, &state.config(), device);

    let stats_container = make_all_stats(vec!["A", "B", "C"]);
    
    CellSimulation {
        simulation,
        renderer,
        uniform_buffer: uniform_buffer,
        all_stats: stats_container
    }
}

fn step_system(
    simulation: &mut CellSimulation,
    mut mouse_slice: i32,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    command_encoder: &mut wgpu::CommandEncoder
) -> i32 {
    mouse_slice = mouse_slice.max(0).min(simulation.simulation.lattice_params.raw.res[2] as i32 - 1);
    let frame_num = simulation.uniform_buffer.data.frame_num;
    //let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    simulation.simulation.step(frame_num, command_encoder, device, 50);
    
    simulation.uniform_buffer.data.frame_num += 1;
    simulation.uniform_buffer.data.itime = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros() as u32;
    simulation.uniform_buffer.data.slice = mouse_slice as u32;

    queue.write_buffer(&simulation.uniform_buffer.buffer, 0, bytemuck::cast_slice(&[simulation.uniform_buffer.data]));

    return mouse_slice;
}

fn toggle_axis(simulation: &mut CellSimulation) {
    let axis = simulation.uniform_buffer.data.slice_axis;
    simulation.uniform_buffer.data.slice_axis = (axis + 1) % 3;
}

fn toggle_view(simulation: &mut CellSimulation) {
    let view = simulation.uniform_buffer.data.rendering_view;
    simulation.uniform_buffer.data.rendering_view = (view + 1) % 3;
}


pub async fn run() {
    env_logger::init();

    let event_loop = EventLoop::new();
    let window = Window::new(&event_loop).unwrap();
    window.set_inner_size(LogicalSize {
        width: 1280.0,
        height: 720.0,
    });
    window.set_title(&format!("Example"));

    let mut state = Setup::new(window).await;

    let mut last_frame_inst = Instant::now();
    let (mut frame_count, mut accum_time, mut fps, mut time) = (0, 0., 0., 0.);
    //let hidpi_factor = window.scale_factor();

    let (mut imgui, platform) = setup_imgui(state.window());

    let renderer_config = RendererConfig {
        texture_format: state.config().format,
        ..Default::default()
    };
    let mut renderer = Renderer::new(&mut imgui, &state.device, &state.queue, renderer_config);

    // Setup the simulation.
    let mut simulation = setup_system(&state, &state.device);
    
    let mut slice_wheel: i32 = 0;
    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window().id() => if !state.input(event) {
                if !simulation.renderer.input(event) {
                    match event {
                        WindowEvent::CloseRequested {..} => *control_flow = ControlFlow::Exit,
                        WindowEvent::KeyboardInput {
                            input: KeyboardInput {
                                state,
                                virtual_keycode: Some(keycode),
                                ..
                            },
                            ..
                        } => {
                            match keycode {
                                VirtualKeyCode::X => {
                                    if *state == ElementState::Released {
                                        toggle_axis(&mut simulation);
                                    }
                                }
                                VirtualKeyCode::V => {
                                    if *state == ElementState::Released {
                                        toggle_view(&mut simulation);
                                    }
                                }
                                VirtualKeyCode::Escape => {
                                    *control_flow = ControlFlow::Exit;
                                }
                                _ => {}
                            }
                        }
                        WindowEvent::Resized(physical_size) => {
                            state.resize(*physical_size);
                        }
                        WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                            state.resize(**new_inner_size);
                        }
                        WindowEvent::MouseWheel { delta, .. } => {
                            match delta {
                                MouseScrollDelta::LineDelta(_, y) => {
                                    slice_wheel += *y as i32;
                                },
                                MouseScrollDelta::PixelDelta(_) => {}
                            }
                        }
                        _ => {}
                    }
                    state.window().request_redraw(); // Redraw after any input
                }
            }
            Event::RedrawRequested(window_id) if window_id == state.window().id() => {
                // Main part
                let frame = match state.surface().get_current_texture() {
                    Ok(frame) => frame,
                    Err(e) => {
                        eprintln!("dropped frame: {:?}", e);
                        return;
                    }
                };
                platform
                    .prepare_frame(imgui.io_mut(), &state.window())
                    .expect("Failed to prepare frame");

                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                
                let mut encoder = state.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                _ = simulation.renderer.render(&mut encoder, &view);
                

                let slice_name = match simulation.uniform_buffer.data.slice_axis {
                    0 => "X",
                    1 => "Y",
                    2 => "Z",
                    _ => "X",
                };

                let viewing_mode = match simulation.uniform_buffer.data.rendering_view {
                    0 => "Particles",
                    1 => "Regions",
                    2 => "Reservoirs",
                    _ => "Volume",
                };

                let ui = imgui.frame();
                {
                    let display_size = ui.io().display_size;
                    let window = ui.window("Information");
                    window
                        .size([200.0, display_size[1]], Condition::Always)
                        .position([0.0, 0.0], Condition::Always)
                        .build(|| {
                            ui.text(format!("FPS: {:.1}", fps));
                            ui.text(format!("Time: {:.3}", time));
                            ui.text(format!("Slice: {}", slice_wheel));
                            ui.text(format!("Slice axis: {}", slice_name));
                            ui.text(format!("Viewing mode: {}", viewing_mode));
                            // TODO: Add a way to choose the species
                            for (name, stat) in simulation.all_stats.iter() {
                                ui.text(format!("{}: {}", name, stat.last()));
                                ui.plot_lines(name, &stat.y.as_slices().0)
                                    .graph_size([200.0, 80.0])
                                    .build();
                            }
                        });
                }

                //let mut encoder: wgpu::CommandEncoder = state.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load, // Do not clear
                            // load: wgpu::LoadOp::Clear(clear_color),
                            store: true,
                        },
                    })],
                    depth_stencil_attachment: None,
                });

                simulation.renderer.update(&state.queue);
                
                renderer
                    .render(imgui.render(), &state.queue, &state.device, &mut rpass)
                    .expect("Rendering failed");
                    
                drop(rpass);

                state.queue.submit(Some(encoder.finish()));

                frame.present();
            }
            Event::RedrawEventsCleared => {
                {
                    accum_time += last_frame_inst.elapsed().as_secs_f32();
                    last_frame_inst = Instant::now();
                    frame_count += 1;
                    if frame_count == 100 {
                        fps = frame_count as f32 / accum_time;
                        accum_time = 0.0;
                        frame_count = 0;
                    }
                    time += simulation.simulation.lattice_params.raw.tau;
                }

                let mut command_encoder = state.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                
                slice_wheel = step_system(&mut simulation, slice_wheel, &state.device, &state.queue, &mut command_encoder);
                {
                    while let Some(sample) = simulation.simulation.stats.pop_front() {
                        simulation.all_stats.entry(sample.name).and_modify(|k| k.add(sample.iteration_count, sample.value as f32));
                    }
                }

                state.queue.submit(Some(command_encoder.finish()));
                
                
                // RedrawRequested will only trigger once, unless we manually
                // request it.
                if frame_count % 2 == 0 {
                    state.window().request_redraw();
                }                
            }
            _ => {}
        }
    });
}

use std::env;

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    env::set_var("RUST_LOG", "simulation=debug");
    pollster::block_on(run());
}