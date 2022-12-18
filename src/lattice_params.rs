use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use std::mem;

use crate::MAX_PARTICLES_SITE;

// ---------------------------------------------------------------------------
// Structures that are shared between Rust and the compute/fragment shaders.

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Params {
    // x, y, z are measurements
    x: f32,
    y: f32,
    z: f32,
    pub x_res : u32,
    pub y_res: u32,
    pub z_res: u32,
    max_particles_site: u32,
    lambda: f32,
    D: f32,
    tau: f32
}

// ---------------------------------------------------------------------------

pub struct LatticeParams {
    pub lattice_params: Params,
    param_buf: wgpu::Buffer,
}

impl LatticeParams {
    pub fn new(
        dimensions: Vec<f32>, resolution: Vec<usize>,
        device: &wgpu::Device,
    ) -> Self {
        let lattice_params = Params {
            x: dimensions[0],
            y: dimensions[1],
            z: dimensions[2],
            x_res: resolution[0] as u32,
            y_res: resolution[1] as u32,
            z_res: resolution[2] as u32,
            max_particles_site: MAX_PARTICLES_SITE as u32,
            lambda: 1.,
            D: 1.,
            tau: 0.1
        };

        let param_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("parameters buffer"),
            contents: bytemuck::bytes_of(&lattice_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        LatticeParams {
            lattice_params,
            param_buf,
        }
    }

    pub fn size(&self) -> usize {
        mem::size_of::<Params>()
    }

    pub fn dimensions(&self) -> usize {
        (self.lattice_params.x_res * self.lattice_params.y_res * self.lattice_params.z_res) as usize
    }

    pub fn binding_resource(&self) -> wgpu::BindingResource {
        self.param_buf.as_entire_binding()
    }

    pub fn binding_type(&self) -> wgpu::BindingType {
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(mem::size_of::<LatticeParams>() as _),
        }
    }
}

impl Params {
    pub fn dimensions(&self) -> usize {
        (self.x_res * self.y_res * self.z_res) as usize
    }
}