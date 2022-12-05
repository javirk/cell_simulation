
use crate::{
    rdme::RDME, 
    lattice_params::{LatticeParams, Params}, 
    lattice::Lattice,
    uniforms::UniformBuffer
};

const WORKGROUP_SIZE: (u32, u32) = (8, 8);

pub struct Simulation {
    // Data for the compute shader.
    rdme: RDME,
    bind_groups: Vec<wgpu::BindGroup>,
    bind_group_layout: wgpu::BindGroupLayout,
    lattice_params: Params,
}

impl Simulation {
    pub fn new(
        uniform_buffer: &UniformBuffer,
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

        Simulation {
            rdme,
            bind_groups,
            bind_group_layout,
            lattice_params: lattice_params.lattice_params,
        }
        
    }

    pub fn step(
        &mut self,
        frame_num: u32,
        command_encoder: &mut wgpu::CommandEncoder,
    ) {
        self.rdme.step(&self.bind_groups[frame_num as usize % 2], command_encoder, &self.lattice_params);

        // cme step

    }
}