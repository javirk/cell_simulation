use std::{time::{SystemTime, UNIX_EPOCH}, collections::{HashMap, VecDeque}};
use winit::{
    event::{Event, WindowEvent, KeyboardInput, VirtualKeyCode, ElementState, MouseScrollDelta},
    event_loop::{ControlFlow, EventLoop, },
    window::Window,
    dpi::LogicalSize,
};
use std::time::Instant;
use imgui::*;
use simulation::{Simulation, Setup, Uniform, UniformBuffer, RenderParams, LatticeParams, Texture, Render3D};
use simulation::RegionType;
use imgui_wgpu::{Renderer, RendererConfig};


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

#[derive(Debug)]
struct StatisticContainer {
    x: VecDeque<u32>,
    y: VecDeque<f32>,  // Must be f32 for the UI
}

impl StatisticContainer {

    fn new(capacity: usize) -> Self {
        StatisticContainer { x: VecDeque::with_capacity(capacity), y: VecDeque::with_capacity(capacity) }
    }

    fn add(&mut self, x: u32, y: f32) {
        while self.x.capacity() == self.x.len() {
            self.x.pop_front();
            self.y.pop_front();
        }
        self.x.push_back(x);
        self.y.push_back(y);
    }

    #[allow(dead_code)]
    fn mean(&self) -> f32 {
        let mut sum = 0.;
        for y in &self.y {
            sum += y;
        }
        sum / self.y.len() as f32
    }

    fn last(&self) -> f32 {
        match self.y.back() {
            Some(y) => *y,
            None => 0.
        }
    }
}

fn make_all_stats(metrics_log: Vec<&str>) -> HashMap<String, StatisticContainer> {
    let mut all_stats = HashMap::new();
    for metric in metrics_log {
        all_stats.insert(metric.to_string(), StatisticContainer::new(100));
    }
    all_stats
}


fn setup_system(state: &Setup, device: &wgpu::Device) -> CellSimulation {
    let uniform = Uniform {
        itime: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u32,
        frame_num: 0,
        slice: 0,
        slice_axis: 2,
        rendering_view: 0,
    };
    let uniform_buffer = UniformBuffer::new(uniform, device);

    let render_param_data: Vec<usize> = vec![
        720, // height
        1280, // width
    ];

    let lattice_resolution = [64, 64, 128];
    let dimensions: [f32; 3] = [0.8, 0.8, 2.];
    let tau = 3E-3;
    let lambda = 31.25E-9;
    
    let render_params = RenderParams::new(device, &render_param_data);
    let simulation_params = LatticeParams::new(dimensions, lattice_resolution, tau, lambda);
    let texture = Texture::new(&lattice_resolution, wgpu::TextureFormat::R32Float, false, &device);
    
    let mut simulation = Simulation::new(simulation_params);

    // simulation.add_region(RegionType::Sphere { name: "one".to_string(), center: [0.5,0.5,0.25], radius: 0.125 }, 8.15E-14/6.);
    // simulation.add_region(RegionType::Cube { name: "one".to_string(), p0: [0., 0., 0.], pf: [1., 1., 1.] }, 8.15E-14/6.);
    // simulation.add_region(RegionType::SphericalShell { shell_name: "one".to_string(), interior_name: "two".to_string(), center: [0.5,0.5,0.25], internal_radius: 0.125, external_radius: 0.25 }, 8.15E-14/6.);

    // simulation.add_region(RegionType::Cylinder { name: "one".to_string(), p0: [0.5, 0.5, 0.1], pf: [0.5, 0.5, 0.4], radius: 0.125 }, 8.15E-14/6.);
    // simulation.add_region(RegionType::CylindricalShell { 
    //     shell_name: "one".to_string(), interior_name: "two".to_string(), p0: [0.5, 0., 0.1], pf: [0.5, 0.5, 0.4], internal_radius: 0.125, external_radius: 0.25 
    // }, 8.15E-14/6.);
    // simulation.add_region(RegionType::SemiSphere { name: "one".to_string(), center: [0.5,0.5,0.5], radius: 0.5, direction: [0., 0., 1.] }, 8.15E-14/6.);
    simulation.add_region(RegionType::Capsid { 
        shell_name: "membrane".to_string(), interior_name: "interior".to_string(), center: [0.4, 0.4, 1.], dir: [0., 0., 1.], internal_radius: 0.37, external_radius: 0.4, total_length: 2. 
    }, 8.15E-14/6.);
    let base_region = RegionType::Sphere { name: "one".to_string(), center: [0.5,0.5,0.25], radius: 0.1 };
    // simulation.add_sparse_region("sparse", base_region, "interior", 10000, 0.);

    simulation.prepare_regions();

    // simulation.add_particle_count("A", "interior", 1000, true, false);
    // simulation.add_particle_count("B", "interior", 1000, false, false);
    // simulation.add_particle_count("C", "interior", 0, false, false);
    // simulation.add_particle_count("D", "membrane", 5000, false, false);
    // simulation.fill_region("E", "sparse", false);
    simulation.add_particle_concentration("Iex", "membrane", 0.10, false, true);

    // simulation.add_reaction(vec!["A", "B"], vec!["C"], 5.82);
    // simulation.add_reaction(vec!["C"], vec!["A", "B"], 0.351);

    simulation.prepare_for_gpu(&uniform_buffer, &texture, device);

    let renderer = Render3D::new(&uniform_buffer, &texture, &simulation.lattice_params, &render_params, &state.config(), device);

    let stats_container = make_all_stats(vec!["A"]);
    
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
    view: &wgpu::TextureView,
    device: &wgpu::Device,
    queue: &wgpu::Queue
) -> i32 {
    mouse_slice = mouse_slice.max(0).min(simulation.simulation.lattice_params.raw.res[2] as i32 - 1);
    let frame_num = simulation.uniform_buffer.data.frame_num;
    let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    simulation.simulation.step(frame_num, &mut command_encoder, device);
    
    _ = simulation.renderer.render(&mut command_encoder, &view);

    simulation.uniform_buffer.data.frame_num += 1;
    simulation.uniform_buffer.data.itime = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros() as u32;
    simulation.uniform_buffer.data.slice = mouse_slice as u32;

    queue.write_buffer(&simulation.uniform_buffer.buffer, 0, bytemuck::cast_slice(&[simulation.uniform_buffer.data]));
    queue.submit(Some(command_encoder.finish()));

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
                }
            }
            Event::RedrawRequested(window_id) if window_id == state.window().id() => {
                // Main part
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
                
                slice_wheel = step_system(&mut simulation, slice_wheel, &view, &state.device, &state.queue);
                {
                    while let Some(sample) = simulation.simulation.stats.pop_front() {
                        simulation.all_stats.entry(sample.name).and_modify(|k| k.add(sample.iteration_count, sample.value as f32));
                    }
                }

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
                    
                    // ui.show_metrics_window(&mut true);
                }

                let mut encoder: wgpu::CommandEncoder = state.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
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
                // RedrawRequested will only trigger once, unless we manually
                // request it.
                state.window().request_redraw();
            }
            _ => {}
        }
    });
}

use std::env;

fn main() {
    //env::set_var("RUST_BACKTRACE", "1");
    pollster::block_on(run());
}