use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use std::mem;

use crate::MAX_PARTICLES_SITE;

// ---------------------------------------------------------------------------
// Structures that are shared between Rust and the compute/fragment shaders.

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Params {
    // x, y, z are measurements. Making this a vector would be more elegant. TODO
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub x_res : u32,
    pub y_res: u32,
    pub z_res: u32,
    max_particles_site: u32,
    pub n_regions: u32,
    lambda: f32, // I hope this only depends on the lattice constants.
    tau: f32,
}

// ---------------------------------------------------------------------------

pub struct LatticeParams {
    pub lattice_params: Params,
    param_buf: Option<wgpu::Buffer>,
}

impl LatticeParams {
    pub fn new(
        dimensions: Vec<f32>, resolution: Vec<usize>,
    ) -> Self {
        let lattice_params = Params {
            x: dimensions[0],
            y: dimensions[1],
            z: dimensions[2],
            x_res: resolution[0] as u32,
            y_res: resolution[1] as u32,
            z_res: resolution[2] as u32,
            max_particles_site: MAX_PARTICLES_SITE as u32,
            n_regions: 1,
            lambda: 31.25E-9,
            tau: 2E-3,
        };

        LatticeParams {
            lattice_params,
            param_buf: None,
        }
    }

    pub fn create_buffer(&mut self, device: &wgpu::Device) {
        self.param_buf = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("parameters buffer"),
            contents: bytemuck::bytes_of(&self.lattice_params),
            usage: wgpu::BufferUsages::UNIFORM,
        }));
    }

    pub fn dimensions(&self) -> usize {
        (self.lattice_params.x_res * self.lattice_params.y_res * self.lattice_params.z_res) as usize
    }

    pub fn binding_resource(&self) -> wgpu::BindingResource {
        self.param_buf.as_ref().expect("Buffer not created yet").as_entire_binding()
    }

    pub fn binding_type(&self) -> wgpu::BindingType {
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(mem::size_of::<Params>() as _),
        }
    }
}

impl Params {
    pub fn dimensions(&self) -> usize {
        (self.x_res * self.y_res * self.z_res) as usize
    }
}