use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use std::mem;
use std::time::{SystemTime, UNIX_EPOCH};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Uniform {
    pub itime: u32,
    pub frame_num: u32,
    pub slice: u32,
    pub slice_axis: u32,
    pub rendering_view: u32,
}

pub struct UniformBuffer {
    pub data: Uniform,
    pub buffer: wgpu::Buffer,
}

impl UniformBuffer {
    pub fn new(device: &wgpu::Device,) -> Self {
        let uniform = Uniform {
            itime: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u32,
            frame_num: 0,
            slice: 0,
            slice_axis: 2,
            rendering_view: 0,
        };
        UniformBuffer::new_from_uniform(uniform, device)
    }

    pub fn new_from_uniform(uniform: Uniform, device: &wgpu::Device) -> Self {
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniform buffer"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        UniformBuffer {
            data: uniform,
            buffer: uniform_buf,
        }
    }

    pub fn binding_resource(&self) -> wgpu::BindingResource {
        self.buffer.as_entire_binding()
    }

    pub fn binding_type(&self) -> wgpu::BindingType {
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(mem::size_of::<Uniform>() as _),
        }
    }
}