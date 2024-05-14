use std::fs::File;
use anyhow::Ok;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use std::io::{BufReader, prelude::*};
use serde_json::Value;
use std::error::Error;

use crate::{lattice::Lattice, MAX_PARTICLES_SITE, utils::json_value_to_array};

// ---------------------------------------------------------------------------
// Structures that are shared between Rust and the compute/fragment shaders.

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Params {
    // The ordering here is very important. Vectors at the end of the struct because of padding issues
    max_particles_site: u32,
    pub n_regions: u32,
    lambda: f32,
    pub tau: f32,
    pub dims: [f32; 3],
    _padding: u32,
    pub res: [u32; 3],
    _padding2: u32,
}

impl Params {
    pub fn get_voxel_size(&self) -> [f32; 3] {
        [
            self.dims[0] / self.res[0] as f32, 
            self.dims[1] / self.res[1] as f32,
            self.dims[2] / self.res[2] as f32
        ]
    }

    pub fn get_res_usize(&self) -> [usize; 3] {
        [
            self.res[0] as usize,
            self.res[1] as usize,
            self.res[2] as usize
        ]
    }

    pub fn get_res_f32(&self) -> [f32; 3] {
        [
            self.res[0] as f32,
            self.res[1] as f32,
            self.res[2] as f32
        ]
    }
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
            _padding: 0,
            res: resolution,
            _padding2: 0,
            max_particles_site: MAX_PARTICLES_SITE as u32,
            n_regions: 1,
            lambda: lambda,
            tau: tau
        };

        println!("Lattice parameters: {}", MAX_PARTICLES_SITE as u32);

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
            //min_binding_size: wgpu::BufferSize::new(mem::size_of::<Params>() as _),
            min_binding_size: None,
        }
    }

    pub fn add_region(&mut self) {
        self.raw.n_regions += 1;
    }

    pub fn remove_region(&mut self) {
        self.raw.n_regions -= 1;
    }

    pub fn res(&self) -> [u32; 3] {
        self.raw.res
    }

    pub fn dims(&self) -> [f32; 3] {
        self.raw.dims
    }

    pub fn get_res_usize(&self) -> [usize; 3] {
        self.raw.get_res_usize()
    }

    pub fn get_res_f32(&self) -> [f32; 3] {
        self.raw.get_res_f32()
    }

    pub fn get_voxel_size(&self) -> [f32; 3] {
        self.raw.get_voxel_size()
    }
}

impl Params {
    pub fn dimensions(&self) -> usize {
        (self.res[0] * self.res[1] * self.res[2]) as usize
    }
}

impl LatticeParams {
    pub fn from_json_value(parameters: &Value) -> Self {
        json_value_to_array::<[f32; 3]>(&parameters["dimensions"]);
        // let mut lattice_params: Params = Params {
        //     dims: json_value_to_array(&parameters["dimensions"]),
        //     _padding: 0,
        //     res: json_value_to_array(&parameters["lattice_resolution"]),
        //     _padding2: 0,
        //     max_particles_site: MAX_PARTICLES_SITE as u32,
        //     n_regions: 1,
        //     lambda: parameters["lambda"].as_f64().unwrap() as f32,
        //     tau: parameters["tau"].as_f64().unwrap() as f32
        // };

        let lattice_params = Params {
            dims: [0.8, 0.8, 2.0],
            _padding: 0,
            res: [32, 32, 64],
            _padding2: 0,
            max_particles_site: MAX_PARTICLES_SITE as u32,
            n_regions: 1,
            lambda: 0.3,
            tau: 0.3
        };
        
        LatticeParams {
            raw: lattice_params,
            param_buf: None,
        }
    }

}