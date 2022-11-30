use std::{borrow::Cow, mem};
use wgpu::util::DeviceExt;
use rand::Rng;

use crate::{life_params::LifeParams, texture::Texture};

const WORKGROUP_SIZE: (u32, u32) = (8, 8);

pub struct Life {
    // Data for the compute shader.
    compute_pipeline: wgpu::ComputePipeline,
    bind_groups: Vec<wgpu::BindGroup>,
    cell_buffers: Vec<wgpu::Buffer>,
    frame_num: usize,
}

impl Life {
    pub fn new(
        texture: &Texture,
        params: &Vec<usize>,
        params_gpu: &LifeParams,
        device: &wgpu::Device,
    ) -> Self {
        // Make buffers
        // Bind group layout -> bind groups
        // Compute pipeline layout -> Then computepipeline

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("life.wgsl"))),
        });

        // Parameters
        let dimensions = params[0] * params[1] * 3;  // 3 for RGB
        let cell_buff_size = dimensions * mem::size_of::<f32>();

        // Bind group layout
        let life_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { 
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new((params.len() * mem::size_of::<u32>()) as _,)
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { 
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(cell_buff_size as _),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { 
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(cell_buff_size as _),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: texture.binding_type(wgpu::StorageTextureAccess::ReadWrite),
                        count: None,
                    },
                ],
                label: None
            }
        );

        let compute_pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("compute"),
                bind_group_layouts: &[&life_bind_group_layout],
                push_constant_ranges: &[],
            }
        );

        let compute_pipeline = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Compute pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &compute_shader,
                entry_point: "life",
            }
        );

        // Initial data
        let mut initial_cell_data = vec![0.0f32; dimensions as usize];
        let mut rng = rand::thread_rng();
        for i in 0..dimensions {
            initial_cell_data[i] = rng.gen_range(0..2) as f32;
        }

        // Buffers
        let mut cell_buffers = Vec::<wgpu::Buffer>::new();
        for i in 0..2 {
            cell_buffers.push(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Cell Buffer {}", i)),
                    contents: bytemuck::cast_slice(&initial_cell_data),
                    usage: wgpu::BufferUsages::VERTEX
                        | wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST,
                }),
            );
        }

        // Bind groups
        let mut cell_bind_groups = Vec::<wgpu::BindGroup>::new();
        for i in 0..2 {
            cell_bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &life_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_gpu.binding_resource(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: cell_buffers[i].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: cell_buffers[(i + 1) % 2].as_entire_binding(), // bind to opposite buffer
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: texture.binding_resource(),
                    },
                ],
                label: Some("Bind_group"),
            }));
        }

        Life { 
            compute_pipeline: compute_pipeline,
            bind_groups: cell_bind_groups,
            cell_buffers: cell_buffers,
            frame_num: 0,
        }
    }

    pub fn step(
        &mut self,
        command_encoder: &mut wgpu::CommandEncoder,
        params: &Vec<usize>,
    ) {
        // command encoder as input
        // Compute pass
        // Set pipeline, bind group
        // Dispatch
        let xdim = params[1] as u32 + WORKGROUP_SIZE.0 - 1;
        let xgroups = xdim / WORKGROUP_SIZE.0;
        let ydim = params[0] as u32 + WORKGROUP_SIZE.1 - 1;
        let ygroups = ydim / WORKGROUP_SIZE.1;

        let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&self.compute_pipeline);
        cpass.set_bind_group(0, &self.bind_groups[self.frame_num % 2], &[]);
        cpass.dispatch_workgroups(xgroups, ygroups, 3);

        self.frame_num += 1;

    }
}