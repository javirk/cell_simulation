use lattice::Lattice;
use lattice_params::{Params, LatticeParams};
use types::Particle;
use wgpu::util::DeviceExt;

const MAX_PARTICLES_SITE: usize = 16;

mod simulation;
mod framework;
mod lattice;
mod lattice_params;
mod types;
mod rdme;

use crate::{
    simulation::Cell,
};




struct CellSimulation {
    cell: Cell,  // It's actually not a cell, but a simulation
    //renderer: Renderer,
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
        let particles = vec![3, 4, 78, 1];

        let params = LatticeParams::new(vec![1., 1., 1.,], vec![32, 32, 32], device);

        let mut lattices = Vec::<Lattice>::new();
        for i in 0..2 {
            lattices.push(Lattice::new(&params.lattice_params, device))
        }

        let lattice_data: Vec<Particle> = lattices[0].init_random_particles(&particles);

        queue.write_buffer(&lattices[0].lattice_buff, 0, &lattice_data);
        queue.write_buffer(&lattices[1].lattice_buff, 0, &lattice_data);


        CellSimulation {
            cell: simulation,
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

        self.cell.step(&mut command_encoder);

        queue.submit(Some(command_encoder.finish()));
    }
}

/// run example
fn main() {
    framework::run::<CellSimulation>("Cell Simulation");
}