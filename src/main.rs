use std::time::{SystemTime, UNIX_EPOCH};
use lattice_params::LatticeParams;
use simulation::Simulation;

const MAX_PARTICLES_SITE: usize = 4;
const WORKGROUP_SIZE: (u32, u32, u32) = (1, 1, 1);

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
mod cme;

use crate::{
    render_params::RenderParams,
    render::Renderer,
    texture::Texture,
    uniforms::{Uniform, UniformBuffer},
};


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
        device: &wgpu::Device,
    ) -> Self {
        // Create texture
        // Init life, setting the initial state
        // Init renderer
        let uniform = Uniform {
            itime: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u32,
            frame_num: 0
        };
        let uniform_buffer = UniformBuffer::new(uniform, device);
        
        // This same data is hard coded in the framework file!
        let render_param_data: Vec<usize> = vec![
            720, // height
            1280, // width
        ];
        
        let render_params = RenderParams::new(device, &render_param_data);
        let simulation_params = LatticeParams::new(vec![1., 1., 1.,], vec![2, 2, 1], device);
        let texture = Texture::new(&simulation_params.lattice_params, &device);
        
        let mut simulation = Simulation::new(simulation_params, device);
        let renderer = Renderer::new(&uniform_buffer, &texture, &simulation.lattice_params.lattice_params, &render_params, config, device);

        simulation.add_region("one", vec![0.,0.,0.], vec![1.,1.,1.], 8.15E-14);
        // simulation.add_region("two", vec![0.2,0.2,0.2], vec![0.8,0.8,0.8], 6.3);
        simulation.add_particle("p1", "one", 2);
        simulation.add_particle("p2", "one", 2);

        //simulation.add_reaction(vec!["p1"], vec!["p2"], 1.);

        simulation.prepare_for_gpu(&uniform_buffer, &texture, device);
        
        CellSimulation {
            simulation,
            renderer,
            uniform_buffer: uniform_buffer,
        }
    }

    fn render(
        &mut self,
        mut mouse_slice: i32,
        view: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _spawner: &framework::Spawner,
    ) -> i32 {
        // Create command encoder
        // Run life step
        // Run render step
        // Submit queue
        let frame_num = self.uniform_buffer.data.frame_num;
        let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        self.simulation.step(self.uniform_buffer.data.frame_num, &mut command_encoder);
        mouse_slice = self.renderer.render(mouse_slice, &mut command_encoder, &view);
        
        // Copy source to destination (lattice and occupancy)
        // This drops performance by 2x, but I don't know any other way to do it
        command_encoder.copy_buffer_to_buffer(
            &self.simulation.lattices[(frame_num as usize + 1) % 2].lattice_buff.as_ref().expect(""), 
            0, 
            &self.simulation.lattices[frame_num as usize % 2].lattice_buff.as_ref().expect(""), 
            0,
            self.simulation.lattices[frame_num as usize % 2].lattice_buff_size as u64
        );

        command_encoder.copy_buffer_to_buffer(
            &self.simulation.lattices[(frame_num as usize + 1) % 2].occupancy_buff.as_ref().expect(""), 
            0, 
            &self.simulation.lattices[frame_num as usize % 2].occupancy_buff.as_ref().expect(""), 
            0,
            self.simulation.lattices[frame_num as usize % 2].occupancy_buff_size as u64
        );

        //queue.submit(Some(command_encoder.finish()));
        self.uniform_buffer.data.frame_num += 1;
        self.uniform_buffer.data.itime = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u32;

        queue.write_buffer(&self.uniform_buffer.buffer, 0, bytemuck::cast_slice(&[self.uniform_buffer.data]));
        queue.submit(Some(command_encoder.finish()));

        return mouse_slice;
    }
}

/// run example
fn main() {
    // env::set_var("RUST_BACKTRACE", "1");
    framework::run::<CellSimulation>("Cell Simulation");
}