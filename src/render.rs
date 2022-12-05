use bytemuck::{Pod, Zeroable};
use std::{borrow::Cow, mem};
use wgpu::util::DeviceExt;

use crate::{
    texture::Texture, 
    render_params::RenderParams, 
    preprocessor::ShaderBuilder,
    uniforms::UniformBuffer
};

pub struct Renderer {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: usize,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    _pos: [f32; 4],
    _tex_coord: [f32; 2],
}

impl Renderer {
    fn vertex(pos: [i8; 2], tc: [i8; 2]) -> Vertex {
        Vertex {
            _pos: [pos[0] as f32, pos[1] as f32, 1.0, 1.0],
            _tex_coord: [tc[0] as f32, tc[1] as f32],
        }
    }

    fn create_vertices() -> (Vec<Vertex>, Vec<u16>) {
        let vertex_data = [
            Renderer::vertex([-1, -1], [0, 0]),
            Renderer::vertex([ 1, -1], [1, 0]),
            Renderer::vertex([ 1,  1], [1, 1]),
            Renderer::vertex([-1,  1], [0, 1]),
        ];

        let index_data: &[u16] = &[
            0, 1, 2, 2, 3, 0,
        ];

        (vertex_data.to_vec(), index_data.to_vec())
    }

    pub fn new(
        uniform_buffer: &UniformBuffer,
        texture: &Texture,
        render_params: &RenderParams,
        config: &wgpu::SurfaceConfiguration,
        device: &wgpu::Device,
    ) -> Self {

        //let uniform_buffer = UniformBuffer::new(uniform.clone(), device);

        // Load and compile the shaders.
        let binding = ShaderBuilder::new("render.wgsl").unwrap();
        let shader_builder = binding.build();

        let shader = device.create_shader_module(shader_builder);

        // Create the vertex and index buffers.
        let vertex_size = mem::size_of::<Vertex>();
        let (vertex_data, index_data) = Renderer::create_vertices();
        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex buffers"),
            contents: bytemuck::cast_slice(&vertex_data),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index buffers"),
            contents: bytemuck::cast_slice(&index_data),
            usage: wgpu::BufferUsages::INDEX,
        });


        // Bind the texture and params using a bind group.
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: texture.binding_type(wgpu::StorageTextureAccess::ReadOnly),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: render_params.binding_type(),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: uniform_buffer.binding_type(),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });
        // TODO: Add many groups. Frame shouldn't be with texture.
        let bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&texture.texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: render_params.binding_resource(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniform_buffer.binding_resource(),
                    },
                ],
                label: Some("render_bind_group"),
            }
        );


        // Create the render pipeline.
        let vertex_buffers = [wgpu::VertexBufferLayout {
            array_stride: vertex_size as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    // Pos
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    // Texture
                    format: wgpu::VertexFormat::Float32x2,
                    offset: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 1,
                },
            ],
        }];
         
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &vertex_buffers,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None
        });

        // Done.
        Renderer {
            vertex_buf: vertex_buf,
            index_buf: index_buf,
            index_count: index_data.len(),
            pipeline: pipeline,
            bind_group: bind_group,
        }
    }

    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {

        let color_attachments = [Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: true,
            }
        })];

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &color_attachments,
            depth_stencil_attachment: None,
        });
        rpass.push_debug_group("Prepare data for draw.");
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint16);
        rpass.set_vertex_buffer(0, self.vertex_buf.slice(..));
        rpass.pop_debug_group();
        rpass.insert_debug_marker("Draw!");
        rpass.draw_indexed(0..self.index_count as u32, 0, 0..1);
    }
}