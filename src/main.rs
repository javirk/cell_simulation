use std::time::{Instant, SystemTime, UNIX_EPOCH};

use lattice::Lattice;
use lattice_params::{Params, LatticeParams};
use simulation::Simulation;
use types::Particle;
use wgpu::util::DeviceExt;

use crate::{render_params::RenderParams, render::Renderer, texture::Texture, uniforms::{Uniform, UniformBuffer}};

const MAX_PARTICLES_SITE: usize = 15;

mod simulation;
mod framework;
mod lattice;
mod lattice_params;
mod types;
mod rdme;
mod render;
mod texture;
mod render_params;
mod preprocessor;
mod uniforms;



struct CellSimulation {
    simulation: Simulation,
    renderer: Renderer,
    uniform_buffer: UniformBuffer,
}

impl framework::Framework for CellSimulation {
    fn required_limits() -> wgpu::Limits {
        wgpu::Limits::downlevel_defaults()
    }

    fn required_downlevel_capabilities() -> wgpu::DownlevelCapabilities {
        wgpu::DownlevelCapabilities {
            flags: wgpu::DownlevelFlags::COMPUTE_SHADERS,
            ..Default::default()
        }
    }

    fn update(&mut self, _event: winit::event::WindowEvent) {
        //empty
    }

    fn resize(
        &mut self,
        _sc_desc: &wgpu::SurfaceConfiguration,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
        //empty
    }

    fn init(
        config: &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self {
        // Create texture
        // Init life, setting the initial state
        // Init renderer
        let uniform = Uniform {
            itime: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u32,
            frame_num: 0
        };
        let uniform_buffer = UniformBuffer::new(uniform, device);
        
        let particles = vec![0, 1, 1, 1];
        
        let render_param_data: Vec<usize> = vec![
            768, // height
            1024, // width
        ];
        let texture = Texture::new(&device, &render_param_data, wgpu::TextureFormat::R32Float);

        let render_params = RenderParams::new(device, &render_param_data);
        let simulation_params = LatticeParams::new(vec![1., 1., 1.,], vec![3, 1, 1], device);

        let mut lattices = Vec::<Lattice>::new();
        for i in 0..2 {
            lattices.push(Lattice::new(&simulation_params.lattice_params, device))
        }

        lattices[0].init_random_particles(&particles);
        println!("{}", lattices[0]);
        let lattice_data = lattices[0].lattice.clone();

        lattices[0].rewrite_buffer(queue);
        lattices[1].rewrite_buffer_data(queue, &lattice_data);

        let simulation = Simulation::new(&uniform_buffer, &lattices, &simulation_params, device);
        let renderer = Renderer::new(&uniform_buffer, &texture, &render_params, config, device);

        CellSimulation {
            simulation,
            renderer,
            uniform_buffer: uniform_buffer
        }
    }

    fn render(
        &mut self,
        view: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _spawner: &framework::Spawner,
    ) {
        // Create command encoder
        // Run life step
        // Run render step
        // Submit queue
        let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        self.simulation.step(self.uniform_buffer.data.frame_num, &mut command_encoder);
        self.renderer.render(&mut command_encoder, &view);

        queue.submit(Some(command_encoder.finish()));
        self.uniform_buffer.data.frame_num += 1;
        self.uniform_buffer.data.itime = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u32;

        queue.write_buffer(&self.uniform_buffer.buffer, 0, bytemuck::cast_slice(&[self.uniform_buffer.data]));

        // self.simulation.update_frame_num(self.frame_num, queue);
        // self.renderer.update_uniforms(&self.uniforms, queue);
        
    }
}

/// run example
fn main() {
    framework::run::<CellSimulation>("Cell Simulation");
}