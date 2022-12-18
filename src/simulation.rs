use wgpu::util::DeviceExt;
use crate::{
    rdme::RDME, 
    texture::Texture, 
    lattice_params::{LatticeParams, Params}, 
    lattice::Lattice,
    uniforms::UniformBuffer
};

const WORKGROUP_SIZE: (u32, u32) = (8, 8);

pub struct Simulation {
    // Data for the compute shader.
    rdme: RDME,
    bind_groups: Vec<wgpu::BindGroup>,
    data_bind_group_layout: wgpu::BindGroupLayout,
    lattice_bind_group_layout: wgpu::BindGroupLayout,
    lattice_params: Params,
}

impl Simulation {
    pub fn new(
        uniform_buffer: &UniformBuffer,
        lattice: &Vec<Lattice>,
        lattice_params: &LatticeParams,
        texture: &Texture,
        device: &wgpu::Device,
    ) -> Self {
        let lock: Vec<u32> = vec![0; lattice[0].occupancy.len()];
        let lock_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Lock buffer"),
                contents: bytemuck::cast_slice(&lock),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }
        );
        let lock_buff_size = lock.len() * std::mem::size_of::<u32>();
        println!("Lock buffer size: {} bytes", lock_buff_size);

        // Bind group layout
        let data_bind_group_layout = device.create_bind_group_layout(
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
                        ty: uniform_buffer.binding_type(),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { 
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(lock_buff_size as _,)
                        },
                        count: None,
                    },
                ],
                label: Some("Data bind group layout")
            }
        );

        let lattice_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { 
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(lattice[0].lattice_buff_size as _),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { 
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(lattice[1].lattice_buff_size as _),
                        },
                        count: None,
                    },
                    // Occupancy now
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { 
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(lattice[0].occupancy_buff_size as _),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { 
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(lattice[1].occupancy_buff_size as _),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: texture.binding_type(wgpu::StorageTextureAccess::ReadWrite),
                        count: None,
                    },
                ],
                label: None
            }
        );

        // Bind groups
        let mut bind_groups = Vec::<wgpu::BindGroup>::new();
        bind_groups.push(
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &data_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: lattice_params.binding_resource(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: uniform_buffer.binding_resource(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: lock_buffer.as_entire_binding(),
                    },
                ],
                label: Some("Data bind group"),
            })
        );

        for i in 0..2 {
            bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &lattice_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: lattice[i].lattice_buff.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: lattice[(i + 1) % 2].lattice_buff.as_entire_binding(), // bind to opposite buffer
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: lattice[i].occupancy_buff.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: lattice[(i + 1) % 2].occupancy_buff.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: texture.binding_resource(),
                    },
                ],
                label: Some("Lattice bind group"),
            }));
        }

        let rdme = RDME::new(&data_bind_group_layout, &lattice_bind_group_layout, &device);

        Simulation {
            rdme,
            bind_groups,
            data_bind_group_layout,
            lattice_bind_group_layout,
            lattice_params: lattice_params.lattice_params,
        }
        
    }

    pub fn step(
        &mut self,
        frame_num: u32,
        command_encoder: &mut wgpu::CommandEncoder,
    ) {
        //println!("Step {}", frame_num);
        self.rdme.step(&self.bind_groups[0], &self.bind_groups[1 + (frame_num as usize % 2)], command_encoder, &self.lattice_params);

        // cme step

    }
}