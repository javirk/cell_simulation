use std::{time::{SystemTime, UNIX_EPOCH}, env, collections::{HashMap, VecDeque}};
use winit::{
    event::{self, Event, WindowEvent, KeyboardInput, VirtualKeyCode, ElementState, MouseScrollDelta},
    event_loop::{ControlFlow, EventLoop, },
    window::Window,
    dpi::LogicalSize,
};
use std::time::Instant;
use imgui::*;
use simulation::{Simulation, Setup, Render, Uniform, UniformBuffer, RenderParams, LatticeParams, Texture};
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
    renderer: Render,
    uniform_buffer: UniformBuffer,
    all_stats: HashMap<String, StatisticContaner>
}

#[derive(Debug)]
struct StatisticContaner {
    x: VecDeque<u32>,
    y: VecDeque<f32>,  // Must be f32 for the UI

}

impl StatisticContaner {

    fn new(capacity: usize) -> Self {
        StatisticContaner { x: VecDeque::with_capacity(capacity), y: VecDeque::with_capacity(capacity) }
    }

    fn add(&mut self, x: u32, y: f32) {
        while self.x.capacity() == self.x.len() {
            self.x.pop_front();
            self.y.pop_front();
        }
        self.x.push_back(x);
        self.y.push_back(y);
    }

    fn mean(&self) -> f32 {
        let mut sum = 0.;
        for y in &self.y {
            sum += y;
        }
        sum / self.y.len() as f32
    }
}

fn make_all_stats(metrics_log: Vec<&str>) -> HashMap<String, StatisticContaner> {
    let mut all_stats = HashMap::new();
    for metric in metrics_log {
        all_stats.insert(metric.to_string(), StatisticContaner::new(100));
    }
    all_stats
}


fn setup_system(state: &Setup, device: &wgpu::Device) -> CellSimulation {
    let uniform = Uniform {
        itime: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u32,
        frame_num: 0
    };
    let uniform_buffer = UniformBuffer::new(uniform, device);

    let render_param_data: Vec<usize> = vec![
        720, // height
        1280, // width
    ];

    let lattice_resolution = [64, 64, 64];
    
    let render_params = RenderParams::new(device, &render_param_data);
    let simulation_params = LatticeParams::new(vec![1., 1., 1.,], lattice_resolution);
    let texture = Texture::new(&lattice_resolution, false, &device);
    
    let mut simulation = Simulation::new(simulation_params);
    let renderer = Render::new(&uniform_buffer, &texture, &simulation.lattice_params.lattice_params, &render_params, &state.config, device);

    simulation.add_region("one", vec![0.,0.,0.], vec![1.,1.,1.], 8.5E-14);
    // simulation.add_region("two", vec![0.2,0.2,0.2], vec![0.8,0.8,0.8], 6.3);
    simulation.add_particle("p1", "one", 500, true);
    simulation.add_particle("p2", "one", 500, false);
    simulation.add_particle("p3", "one", 0, true);

    //simulation.add_reaction(vec!["p1"], vec!["p2"], 0.);
    simulation.add_reaction(vec!["p1", "p2"], vec!["p3"], 6.);

    simulation.prepare_for_gpu(&uniform_buffer, &texture, device);

    let stats_container = make_all_stats(vec!["p1", "p3"]);
    
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
    let frame_num = simulation.uniform_buffer.data.frame_num;
    let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    simulation.simulation.step(frame_num, &mut command_encoder, device);
    mouse_slice = simulation.renderer.render(mouse_slice, &mut command_encoder, &view);

    simulation.uniform_buffer.data.frame_num += 1;
    simulation.uniform_buffer.data.itime = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u32;

    queue.write_buffer(&simulation.uniform_buffer.buffer, 0, bytemuck::cast_slice(&[simulation.uniform_buffer.data]));
    queue.submit(Some(command_encoder.finish()));

    return mouse_slice;
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
    let (mut frame_count, mut accum_time, mut fps) = (0, 0., 0.);
    //let hidpi_factor = window.scale_factor();

    let (mut imgui, platform) = setup_imgui(state.window());

    let renderer_config = RendererConfig {
        texture_format: state.config.format,
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
                match event {
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => *control_flow = ControlFlow::Exit,
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
                }

                let frame = match state.surface.get_current_texture() {
                    Ok(frame) => frame,
                    Err(e) => {
                        eprintln!("dropped frame: {:?}", e);
                        return;
                    }
                };
                platform
                    .prepare_frame(imgui.io_mut(), &state.window)
                    .expect("Failed to prepare frame");

                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                
                slice_wheel = step_system(&mut simulation, slice_wheel, &view, &state.device, &state.queue);
                {
                    while let Some(sample) = simulation.simulation.stats.pop_front() {
                        //simulation.all_stats[&sample.name].add(sample.iteration_count, sample.value);
                        simulation.all_stats.entry(sample.name).and_modify(|k| k.add(sample.iteration_count, sample.value as f32));
                        //my_map.entry("a").and_modify(|k| *k += 10);
                    }
                }

                let ui = imgui.frame();
                {   
                    let display_size = ui.io().display_size;
                    let window = ui.window("Information");
                    window
                        .size([200.0, display_size[1]], Condition::Always)
                        .position([0.0, 0.0], Condition::Always)
                        .build(|| {
                            ui.text(format!("FPS: {:.1}", fps));
                            ui.text(format!("Slice: {}", slice_wheel));
                            // TODO: Add concentration plot and a way to choose the species
                            for (name, stat) in simulation.all_stats.iter() {
                                ui.text(format!("{}: {}", name, stat.mean()));
                                ui.plot_lines(name, &stat.y.as_slices().0)
                                    .graph_size([200.0, 80.0])
                                    .build();
                            }
                        });
                    
                    //ui.show_metrics_window(&mut true);
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

fn main() {
    pollster::block_on(run());
}