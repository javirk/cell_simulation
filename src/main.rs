use lattice::Lattice;
use lattice_params::{Params, LatticeParams};
use simulation::Simulation;
use types::Particle;
use wgpu::util::DeviceExt;

use crate::{render_params::RenderParams, render::Renderer, texture::Texture};

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



struct CellSimulation {
    simulation: Simulation,
    renderer: Renderer,
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

        let simulation = Simulation::new(&lattices, &simulation_params, device);
        let renderer = Renderer::new(&texture, &render_params, config, device);

        CellSimulation {
            simulation,
            renderer
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

        self.simulation.step(&mut command_encoder);
        self.renderer.render(&mut command_encoder, &view);

        queue.submit(Some(command_encoder.finish()));
    }
}

/// run example
fn main() {
    framework::run::<CellSimulation>("Cell Simulation");
}