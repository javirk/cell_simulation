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
    pub dims: [f32; 3],
    pub res: [u32; 3],
    max_particles_site: u32,
    pub n_regions: u32,
    lambda: f32, // I hope this only depends on the lattice constants.
    pub tau: f32,
}

// ---------------------------------------------------------------------------

pub struct LatticeParams {
    pub raw: Params,
    param_buf: Option<wgpu::Buffer>,
}

impl LatticeParams {
    pub fn new(
        dimensions: [f32; 3], resolution: [u32; 3], tau: f32, lambda: f32,
    ) -> Self {
        let lattice_params = Params {
            dims: dimensions,
            res: resolution,
            max_particles_site: MAX_PARTICLES_SITE as u32,
            n_regions: 1,
            lambda: lambda,
            tau: tau
        };

        LatticeParams {
            raw: lattice_params,
            param_buf: None,
        }
    }

    pub fn create_buffer(&mut self, device: &wgpu::Device) {
        self.param_buf = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("parameters buffer"),
            contents: bytemuck::bytes_of(&self.raw),
            usage: wgpu::BufferUsages::UNIFORM,
        }));
    }

    pub fn dimensions(&self) -> usize {
        (self.raw.res[0] * self.raw.res[1] * self.raw.res[2]) as usize
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
        (self.res[0] * self.res[1] * self.res[2]) as usize
    }
}