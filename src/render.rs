use bytemuck::{Pod, Zeroable};
use std::{mem};
use wgpu::util::DeviceExt;
use cgmath::{prelude::*, vec1};
use rand;

use crate::{
    texture::Texture, 
    render_params::RenderParams, 
    preprocessor::ShaderBuilder,
    uniforms::UniformBuffer, lattice_params::Params, lattice
};

pub struct Renderer {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: usize,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    instances: Vec<Instance>,
    instance_buf: wgpu::Buffer,
    lattice_dimensions: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    _pos: [f32; 3],
    _tex_coord: [f32; 3],
}

struct Instance {
    position: cgmath::Vector3<f32>,
    scaling: cgmath::Vector3<f32>,
    tex_coords: cgmath::Vector3<f32>,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
    tex_coords: [f32; 3]
}

impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout { 
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3
                }
            ]
        }
    }
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        InstanceRaw {
            model: (cgmath::Matrix4::from_translation(self.position) * cgmath::Matrix4::from_nonuniform_scale(self.scaling[0], self.scaling[1], self.scaling[2])).into(),
            tex_coords: self.tex_coords.into()
        }
    }
}

impl InstanceRaw {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    // Vertex shader uses locations 0, and 1 already
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // A mat4 takes up 4 vertex slots as it is technically 4 vec4s. We need to define a slot
                // for each vec4. We'll have to reassemble the mat4 in
                // the shader.
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // Color is a vec3, so we only need to define one more slot
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

const VERTICES: &[Vertex] = &[
    Vertex { _pos: [-1., -1., 0.0], _tex_coord: [0.0, 0.0, 0.0] },
    Vertex { _pos: [1., -1., 0.0], _tex_coord: [0.0, 0.0, 0.0] },
    Vertex { _pos: [1., 1., 0.0], _tex_coord: [0.0, 0.0, 0.0] },
    Vertex { _pos: [-1., 1., 0.0], _tex_coord: [0.0, 0.0, 0.0] },
];

const INDICES: &[u16] = &[
    0, 1, 2, 2, 3, 0
];



impl Renderer {

    pub fn new(
        uniform_buffer: &UniformBuffer,
        texture: &Texture,
        lattice_params: &Params,
        render_params: &RenderParams,
        config: &wgpu::SurfaceConfiguration,
        device: &wgpu::Device,
    ) -> Self {
        // Load and compile the shaders.
        let binding = ShaderBuilder::new("render.wgsl").unwrap();
        let shader_builder = binding.build();

        let shader = device.create_shader_module(shader_builder);

        // Create the vertex and index buffers.
        let vertex_buf = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex buffer"),
                contents: bytemuck::cast_slice(VERTICES),
                usage: wgpu:: BufferUsages::VERTEX,
            }
        );

        let index_buf = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Index buffer"),
                contents: bytemuck::cast_slice(INDICES),
                usage: wgpu::BufferUsages::INDEX
            }
        );

        // This creates the instances. All the math is for the translation vector. TODO: Add margins.
        let lattice_res_f32 = [lattice_params.x_res.clone() as f32, lattice_params.y_res.clone() as f32, lattice_params.z_res.clone() as f32];
        let (width, height) = (render_params.params.width as f32, render_params.params.height as f32);
        let mut l = 0.;
        if (width / (lattice_params.x * lattice_res_f32[0])) <= (height / (lattice_params.y * lattice_res_f32[1])) {
            // X is the limiting dimension  
            l = width / lattice_res_f32[0];
        } else {
            // Y is the limiting dimension
            l = height / lattice_res_f32[1];
        }
        
        let scaling_x = l / width;
        let scaling_y = l / height;
        let left_margin = 200. / width;  // Because of the GUI. Left margin is 200 pixels.
        let instances = (0..lattice_params.z_res).flat_map(|z| {
            (0..lattice_params.x_res).flat_map(move |x| {
                (0..lattice_params.y_res).map(move |y|{
                    let position = cgmath::Vector3 { 
                        x: lattice_res_f32[0] * scaling_x * (1. / lattice_res_f32[0]) * (2. * (x as f32) - lattice_res_f32[0] + 1.) + left_margin,
                        y: lattice_res_f32[1] * scaling_y * (1. / lattice_res_f32[1]) * (2. * (y as f32) - lattice_res_f32[1] + 1.),
                        z: z as f32 
                    };
                    let scaling = cgmath::Vector3 { x: scaling_x, y: scaling_y, z: 1. };
                    Instance {
                        position,
                        tex_coords: cgmath::Vector3 { 
                            x: x as f32,
                            y: y as f32,
                            z: z as f32
                        },
                        scaling
                    }
                })
            })
        }).collect::<Vec<_>>();

        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buf = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&instance_data),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        println!("Instance count: {}", instances.len());


        // Bind the texture and params using a bind group.
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                buffers: &[Vertex::desc(),  InstanceRaw::desc()],
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
        println!("Instances: {}", instances.len());

        // Done.
        Renderer {
            vertex_buf: vertex_buf,
            index_buf: index_buf,
            index_count: INDICES.len(),
            pipeline: pipeline,
            bind_group: bind_group,
            instance_buf,
            instances,
            lattice_dimensions: lattice_res_f32
        }
    }

    pub fn render(
        &mut self,
        mut slice: i32,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) -> i32 {

        slice = slice.max(0).min(self.lattice_dimensions[2] as i32 - 1);

        let instance_step = self.lattice_dimensions[0] as u32 * self.lattice_dimensions[1] as u32;
        let instance_idx = instance_step * slice as u32;

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
        rpass.push_debug_group("Prepare data for drawing.");
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint16);
        rpass.set_vertex_buffer(0, self.vertex_buf.slice(..));
        rpass.set_vertex_buffer(1, self.instance_buf.slice(..));
        rpass.pop_debug_group();
        rpass.insert_debug_marker("Draw!");
        rpass.draw_indexed(0..self.index_count as u32, 0, instance_idx..instance_idx + instance_step);
        return slice;
    }
}