use std::{borrow::Cow, mem};
use wgpu::util::DeviceExt;
use rand::Rng;

use crate::rdme::RDME;

const WORKGROUP_SIZE: (u32, u32) = (8, 8);

pub struct Simulation {
    // Data for the compute shader.
    compute_pipeline: wgpu::ComputePipeline,  // Maybe RDME and CME here
    bind_groups: Vec<wgpu::BindGroup>,
    bind_group_layout: wgpu::BindGroupLayout,
    frame_num: usize,
}

impl Simulation {
    pub fn new(
        lattice: &Vec<Lattice>,
        lattice_params: &LatticeParams,
        device: &wgpu::Device,
    ) -> Self {
        // Bind group layout
        let bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { 
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(lattice_params.size() as _,)
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { 
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(lattice[0].buff_size as _),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { 
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(lattice[1].buff_size as _),
                        },
                        count: None,
                    },
                ],
                label: None
            }
        );

        // Bind groups
        let mut bind_groups = Vec::<wgpu::BindGroup>::new();
        for i in 0..2 {
            bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: lattice_params.binding_resource(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: lattice[i].lattice_buff.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: lattice[(i + 1) % 2].lattice_buff.as_entire_binding(), // bind to opposite buffer
                    },
                ],
                label: Some("Simulation bind group"),
            }));
        }

        let rdme = RDME::new(&bind_group_layout, &device);
        
    }

    pub fn step(
        &mut self,
        command_encoder: &mut wgpu::CommandEncoder,
    ) {
        // command encoder as input
        // Compute pass
        // Set pipeline, bind group
        // Dispatch

        // let xdim = params[1] as u32 + WORKGROUP_SIZE.0 - 1;
        // let xgroups = xdim / WORKGROUP_SIZE.0;
        // let ydim = params[0] as u32 + WORKGROUP_SIZE.1 - 1;
        // let ygroups = ydim / WORKGROUP_SIZE.1;

        // let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        // cpass.set_pipeline(&self.compute_pipeline);
        // cpass.set_bind_group(0, &self.bind_groups[self.frame_num % 2], &[]);
        // cpass.dispatch_workgroups(xgroups, ygroups, 3);

        // self.frame_num += 1;

    }
}