use std::time::{SystemTime, UNIX_EPOCH};
use std::env;
use lattice::Lattice;
use lattice_params::LatticeParams;
use simulation::Simulation;

use crate::{render_params::RenderParams, render::Renderer, texture::Texture, uniforms::{Uniform, UniformBuffer}};

const MAX_PARTICLES_SITE: usize = 8;

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
    lattices: Vec<Lattice>,
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
        
        let particles = vec![0, 1];
        
        let render_param_data: Vec<usize> = vec![
            768, // height
            1024, // width
        ];
        let texture = Texture::new(&device, &render_param_data);

        let render_params = RenderParams::new(device, &render_param_data);
        let simulation_params = LatticeParams::new(vec![1., 1., 1.,], vec![2, 1, 1], device);

        let mut lattices = Vec::<Lattice>::new();
        for i in 0..2 {
            lattices.push(Lattice::new(&simulation_params.lattice_params, device))
        }

        lattices[0].init_random_particles(&particles);
        println!("{}", lattices[0]);
        println!("{:?}", lattices[0].occupancy);
        let lattice_data = lattices[0].lattice.clone();
        let occupancy_data = lattices[0].occupancy.clone();

        lattices[0].rewrite_buffers(queue);
        lattices[1].rewrite_buffer_data(queue, &lattice_data, &occupancy_data);

        let simulation = Simulation::new(&uniform_buffer, &lattices, &simulation_params, &texture, device);
        let renderer = Renderer::new(&uniform_buffer, &texture, &render_params, config, device);

        CellSimulation {
            simulation,
            lattices,
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
        let frame_num = self.uniform_buffer.data.frame_num;
        let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        self.simulation.step(self.uniform_buffer.data.frame_num, &mut command_encoder);
        self.renderer.render(&mut command_encoder, &view);
        
        // Copy source to destination (lattice and occupancy)
        // This drops performance by 2x, but I don't know any other way to do it
        command_encoder.copy_buffer_to_buffer(
            &self.lattices[frame_num as usize % 2].lattice_buff, 
            0, 
            &self.lattices[(frame_num as usize + 1) % 2].lattice_buff, 
            0,
            self.lattices[frame_num as usize % 2].lattice_buff_size as u64
        );

        command_encoder.copy_buffer_to_buffer(
            &self.lattices[frame_num as usize % 2].occupancy_buff, 
            0, 
            &self.lattices[(frame_num as usize + 1) % 2].occupancy_buff, 
            0,
            self.lattices[frame_num as usize % 2].occupancy_buff_size as u64
        );

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
    //env::set_var("RUST_BACKTRACE", "1");
    framework::run::<CellSimulation>("Cell Simulation");
}