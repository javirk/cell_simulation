use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use std::mem;

// ---------------------------------------------------------------------------
// Structures that are shared between Rust and the compute/fragment shaders.

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Params {
    num_species: u32,
    num_reactions: u32
}

// ---------------------------------------------------------------------------

pub struct ReactionParams {
    pub raw_params: Params,
    param_buf: Option<wgpu::Buffer>,
}

impl ReactionParams {
    pub fn new(
        num_species: u32, num_reactions: u32, device: &wgpu::Device
    ) -> Self {
        let reaction_params = Params {
            num_species,
            num_reactions
        };

        let param_buf = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("reaction parameters buffer"),
            contents: bytemuck::bytes_of(&reaction_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        }));

        ReactionParams {
            raw_params: reaction_params,
            param_buf: param_buf,
        }
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