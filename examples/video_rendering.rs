use std::{time::{SystemTime, UNIX_EPOCH}};
use winit::{
    event::{Event, WindowEvent, KeyboardInput, VirtualKeyCode, ElementState, MouseScrollDelta},
    event_loop::{ControlFlow, EventLoop, },
    window::Window,
    dpi::LogicalSize,
};
use simulation::{Simulation, Setup, Uniform, UniformBuffer, RenderParams, LatticeParams, Texture, Render3D};
use simulation::RegionType;



struct CellSimulation {
    simulation: Simulation,
    renderer: Render3D,
    uniform_buffer: UniformBuffer,
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
        state.size.unwrap().height as usize, 
        state.size.unwrap().width as usize,
    ];

    let lattice_resolution = [64, 64, 128];
    let dimensions: [f32; 3] = [0.8, 0.8, 2.];
    let tau = 3E-3;
    let lambda = 31.25E-9;
    
    let render_params = RenderParams::new(device, &render_param_data);
    let simulation_params = LatticeParams::new(dimensions, lattice_resolution, tau, lambda);
    let texture = Texture::new(&lattice_resolution, wgpu::TextureFormat::R32Float, false, &device);
    
    let mut simulation = Simulation::new(simulation_params);

    simulation.add_region(RegionType::Capsid { 
        shell_name: "membrane".to_string(), interior_name: "interior".to_string(), center: [0.4, 0.4, 1.], dir: [0., 0., 1.], internal_radius: 0.37, external_radius: 0.4, total_length: 2. 
    }, 8.15E-14/6.);
    // let base_region = RegionType::Sphere { name: "one".to_string(), center: [0.5,0.5,0.25], radius: 0.1 };
    // simulation.add_sparse_region("sparse", base_region, "interior", 10000, 0.);

    simulation.prepare_regions();

    simulation.add_particle_count("A", "interior", 5000, true, false);
    simulation.add_particle_count("B", "interior", 5000, false, false);
    simulation.add_particle_count("C", "interior", 0, false, false);
    simulation.add_particle_count("D", "membrane", 8000, false, false);
    // simulation.fill_region("E", "sparse", false);

    simulation.add_reaction(vec!["A", "B"], vec!["C"], 5.82);
    simulation.add_reaction(vec!["C"], vec!["A", "B"], 0.351);

    simulation.prepare_for_gpu(&uniform_buffer, &texture, device);

    let renderer = Render3D::new(&uniform_buffer, &texture, &simulation.lattice_params, &render_params, &state.config(), device);
    
    CellSimulation {
        simulation,
        renderer,
        uniform_buffer: uniform_buffer,
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
    println!("{:?}", state.size.unwrap().width);
    //let hidpi_factor = window.scale_factor();

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
                let frame = match state.surface().get_current_texture() {
                    Ok(frame) => frame,
                    Err(e) => {
                        eprintln!("dropped frame: {:?}", e);
                        return;
                    }
                };

                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                
                slice_wheel = step_system(&mut simulation, slice_wheel, &view, &state.device, &state.queue);

                let mut encoder: wgpu::CommandEncoder = state.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                let rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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