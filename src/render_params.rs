use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use std::mem;
use serde_json::Value;
use std::fs::File;
use std::io::{prelude::*};


// ---------------------------------------------------------------------------
// Structures that are shared between Rust and the compute/fragment shaders.

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Params {
    pub width : u32,
    pub height : u32,
}

// ---------------------------------------------------------------------------

pub struct RenderParams {
    param_buf : wgpu::Buffer,
    pub params: Params
}

impl RenderParams {
    pub fn new(
        device: &wgpu::Device,
        dimensions: &Vec<usize>,
    ) -> Self {
        let params = Params {
            width: dimensions[1] as u32,
            height: dimensions[0] as u32,
        };
        let param_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("parameters buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        RenderParams {
            param_buf,
            params
        }
    }

    pub fn binding_resource(&self) -> wgpu::BindingResource {
        self.param_buf.as_entire_binding()
    }

    pub fn binding_type(&self) -> wgpu::BindingType {
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(mem::size_of::<Params>() as _),
        }
    }

    pub fn from_file<P: AsRef<std::path::Path>>(
        path: P,
        device: &wgpu::Device,
    ) -> Result<Self, String> {
        let mut file = File::open(path).unwrap();
        let mut buff = String::new();
        file.read_to_string(&mut buff).unwrap();
     
        let data: Value = serde_json::from_str(&buff).unwrap();

        let dimensions = vec![
            data["rendering"]["height"].as_u64().unwrap() as usize,
            data["rendering"]["width"].as_u64().unwrap() as usize,
        ];
        Ok(RenderParams::new(device, &dimensions))

    }
}