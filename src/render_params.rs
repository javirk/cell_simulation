use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use std::mem;


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
}