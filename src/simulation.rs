use wgpu::{util::DeviceExt};
use crate::{
    rdme::RDME, 
    texture::Texture, 
    lattice_params::LatticeParams, 
    lattice::Lattice,
    uniforms::UniformBuffer, types::{Region, Particle},
    WORKGROUP_SIZE,
    preprocessor::ShaderBuilder
};

pub struct Simulation {
    // Data for the compute shader.
    rdme: Option<RDME>,
    bind_groups: Vec<wgpu::BindGroup>,
    pub lattices: Vec<Lattice>,
    pub lattice_params: LatticeParams,
    regions: Regions,
    locks: Lock,
    diffusion_matrix: Matrix,
    texture_compute_pipeline: Option<wgpu::ComputePipeline>
}

struct Matrix {
    matrix: Vec<f32>,
    buffer: Option<wgpu::Buffer>,
    buf_size: Option<wgpu::BufferSize>
}

struct Regions {
    regions: Vec<Region>,
    positions: Vec<Vec<Vec<f32>>>,  // Starting corner, ending corner
    names: Vec<String>,
    buffer: Option<wgpu::Buffer>,
    buf_size: Option<wgpu::BufferSize>
}

struct Lock {
    buffer: wgpu::Buffer,
    buf_size: Option<wgpu::BufferSize>,
}

impl Simulation {
    pub fn new(
        lattice_params: LatticeParams,
        device: &wgpu::Device,
    ) -> Self {
        // A simulation has a lattice first
        let mut lattices = Vec::<Lattice>::new();
        for _ in 0..2 {
            lattices.push(Lattice::new(&lattice_params.lattice_params, device))
        }

        //lattices[0].init_random_particles(&particles);

        let regions = Regions {
            regions: vec![0 as Region; lattice_params.dimensions()],
            positions: vec![vec![vec![0.; 3], vec![1.; 3]]; 1],
            names: vec![String::from("default")],
            buffer: None,
            buf_size: None
        };

        // The diffusion matrix has 3 dimensions: region x region x particle type
        let diffusion_matrix = Matrix {
            matrix: vec![8.15E-14; 1],
            buffer: None,
            buf_size: None
        };

        let lock = Simulation::prepare_locks(&lattices[0], device);

        //let bind_group_layouts = Simulation::build_bind_group_layouts(&lattice_params, &lattices, uniform_buffer, &lock, texture, device);
        
        let bind_groups = Vec::<wgpu::BindGroup>::new();

        Simulation {
            rdme: None,
            bind_groups,
            lattices,
            lattice_params,
            regions,
            locks: lock,
            diffusion_matrix,
            texture_compute_pipeline: None
        }
        
    }

    pub fn step(
        &mut self,
        frame_num: u32,
        command_encoder: &mut wgpu::CommandEncoder,
    ) {
        self.rdme.as_ref()
            .expect("RDME must be initialized first")
            .step(&self.bind_groups[0], &self.bind_groups[1 + (frame_num as usize % 2)], command_encoder, &self.lattice_params.lattice_params);

        // cme step

        // Fill the texture
        self.texture_pass(frame_num, command_encoder);

    }

    pub fn prepare_for_gpu(&mut self, 
        uniform_buffer: &UniformBuffer,
        texture: &Texture,
        device: &wgpu::Device,
    ) {
        
        self.lattices[1].lattice = self.lattices[0].lattice.clone();
        self.lattices[1].occupancy = self.lattices[0].occupancy.clone();
        self.lattices[0].start_buffers(device);
        self.lattices[1].start_buffers(device);
        
        // I'll have to write the buffers here: diffusion matrix and regions
        // Diffusion matrix
        let diffusion_matrix_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Diffusion matrix buffer"),
            contents: bytemuck::cast_slice(&self.diffusion_matrix.matrix),
            usage: wgpu::BufferUsages::STORAGE,
        });
        self.diffusion_matrix.buffer = Some(diffusion_matrix_buffer);
        self.diffusion_matrix.buf_size = wgpu::BufferSize::new((self.diffusion_matrix.matrix.len() * std::mem::size_of::<f32>()) as _,);

        // Regions
        let regions_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Regions buffer"),
            contents: bytemuck::cast_slice(&self.regions.regions),
            usage: wgpu::BufferUsages::STORAGE,
        });
        self.regions.buffer = Some(regions_buffer);
        self.regions.buf_size = wgpu::BufferSize::new((self.regions.regions.len() * std::mem::size_of::<Region>()) as _,);
        self.lattice_params.create_buffer(device);

        // Bind group layouts should also be built here. Problem is, they are needed for RDME before.
        let bind_group_layouts = self.build_bind_group_layouts(uniform_buffer, texture, device);

        // RDME
        let rdme = RDME::new(&bind_group_layouts, &device);
        self.rdme = Some(rdme);

        // Texture compute pipeline
        let texture_compute_pipeline = self.build_texture_compute_pipeline(&bind_group_layouts, device);
        self.texture_compute_pipeline = Some(texture_compute_pipeline);

        // Bind groups
        let bind_groups = self.build_bind_groups(
            &bind_group_layouts,
            uniform_buffer,
            texture,
            device
        );

        self.bind_groups = bind_groups;
        
    }


}

// Region and particle methods
impl Simulation {
    // I'm not sure if this should be in here or in the lattice part. Diffusion matrix should be outside, though.
    pub fn add_region(&mut self, name: &str, starting_pos: Vec<f32>, ending_pos: Vec<f32>, default_diffusion_rate: f32) {
        // Add a new region. It will be a rectangle for now. Defined by the starting and ending positions.
        assert!(starting_pos[0] <= ending_pos[0] && starting_pos[1] <= ending_pos[1] && starting_pos[2] <= ending_pos[2]);
        let default_transition_rate: f32 = 8.15E-14;
        let new_region_idx = self.regions.names.len() as Region;
        self.regions.names.push(String::from(name));
        self.regions.positions.push(vec![starting_pos.clone(), ending_pos.clone()]);

        let res = (self.lattice_params.lattice_params.x_res as usize, self.lattice_params.lattice_params.y_res as usize, self.lattice_params.lattice_params.z_res as usize);
        let start = (
            (starting_pos[0] * res.0 as f32) as usize,
            (starting_pos[1] * res.1 as f32) as usize,
            (starting_pos[2] * res.2 as f32) as usize
        );
        let end = (
            (ending_pos[0] * res.0 as f32) as usize,
            (ending_pos[1] * res.1 as f32) as usize,
            (ending_pos[2] * res.2 as f32) as usize
        );

        for x in start.0..end.0 {
            for y in start.1..end.1 {
                for z in start.2..end.2 {
                    self.regions.regions[x + y * res.0 + z * res.0 * res.1] = new_region_idx;
                }
            }
        }

        // Add the region to the diffusion matrix. It's a 3D matrix. We start by reading the length up to now:
        let num_regions = new_region_idx as usize;  // Regions
        let num_particles = self.lattices[0].particle_names.len();  // Particles
        let num_new_regions = num_regions + 1;

        let mut new_diffusion = vec![default_transition_rate; num_new_regions * num_new_regions * num_particles];

        Simulation::matrix_to_matrix(&self.diffusion_matrix.matrix, &mut new_diffusion, num_regions, num_new_regions, num_particles);

        // Modify the last element of the added matrix. It's the diffusion rate of the particles in the region
        for i_part in 0..num_particles {
            new_diffusion[(num_new_regions * num_new_regions - 1) * (i_part + 1)] = default_diffusion_rate;
        }

        println!("Region {} added. New diffusion matrix: {:?}", name, new_diffusion);

        self.diffusion_matrix.matrix = new_diffusion;
        self.lattice_params.lattice_params.n_regions += 1;
    }

    pub fn add_particle(&mut self, name: &str, to_region: &str, count: u32) {
        // Add particles to the simulation. It will be added to the region specified by to_region
        let region_idx = self.regions.names.iter().position(|x| x == to_region).unwrap() as usize;
        let particle_idx = self.lattices[0].particle_names.len() as Particle;

        self.lattices[0].particle_names.push(String::from(name));
        let starting_region = &self.regions.positions[region_idx][0];
        let ending_region = &self.regions.positions[region_idx][1];

        self.lattices[0].init_random_particles(particle_idx, count, &starting_region, &ending_region);

        // Update the diffusion matrix now
        let num_regions = self.regions.names.len();  // Regions

        let mut slice = self.diffusion_matrix.matrix[..num_regions*num_regions].to_vec();
        self.diffusion_matrix.matrix.append(&mut slice); 
        println!("Particle {} added. New diffusion matrix: {:?}", name, self.diffusion_matrix.matrix);

    }
    #[allow(dead_code)]
    pub fn set_diffusion_rate() {
        // TODO
    }
    #[allow(dead_code)]
    pub fn set_diffusion_rate_particle() {
        // TODO
    }

    fn matrix_to_matrix(mtx: &Vec<f32>, new_mtx: &mut Vec<f32>, prev_rowcol: usize, new_rowcol: usize, third_dim: usize) {
        for i in 0..prev_rowcol {
            for j in 0..prev_rowcol {
                for k in 0..third_dim {
                    new_mtx[i + j * new_rowcol + k * new_rowcol * new_rowcol] = mtx[i + j * prev_rowcol + k * prev_rowcol * prev_rowcol];
                }
            }
        }
    }
}


// GPU functions
impl Simulation {
    #[allow(dead_code)]
    pub fn copy_buffers_data(&mut self, queue: &wgpu::Queue) {
        let lattice_data = self.lattices[0].lattice.clone();
        let occupancy_data = self.lattices[0].occupancy.clone();

        // self.lattices[0].rewrite_buffers(queue);
        // self.lattices[1].rewrite_buffer_data(queue, &lattice_data, &occupancy_data);

    }

    fn prepare_locks(lattice: &Lattice, device: &wgpu::Device) -> Lock {
        let lock: Vec<u32> = vec![0; lattice.occupancy.len()];
        let lock_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Lock buffer"),
                contents: bytemuck::cast_slice(&lock),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }
        );
        let lock_buff_size = wgpu::BufferSize::new((lock.len() * std::mem::size_of::<u32>()) as _,);
        Lock {
            buffer: lock_buffer,
            buf_size: lock_buff_size
        }
    }

    fn build_bind_group_layouts(
        &self,
        uniform_buffer: &UniformBuffer,
        texture: &Texture,
        device: &wgpu::Device
    ) -> Vec<wgpu::BindGroupLayout> {

        let data_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { 
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: self.lattice_params.buffer_size()
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
                            min_binding_size: self.locks.buf_size
                        },
                        count: None,
                    },
                    // Region data
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { 
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: self.regions.buf_size
                        },
                        count: None,
                    },
                    // Diffusion matrix
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { 
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: self.diffusion_matrix.buf_size
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
                            min_binding_size: wgpu::BufferSize::new(self.lattices[0].lattice_buff_size as _),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { 
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(self.lattices[1].lattice_buff_size as _),
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
                            min_binding_size: wgpu::BufferSize::new(self.lattices[0].occupancy_buff_size as _),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { 
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(self.lattices[1].occupancy_buff_size as _),
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
        vec![data_bind_group_layout, lattice_bind_group_layout]
    }

    fn build_bind_groups(
        &self,
        bind_group_layouts: &Vec<wgpu::BindGroupLayout>,
        uniform_buffer: &UniformBuffer,
        texture: &Texture,
        device: &wgpu::Device
    ) -> Vec::<wgpu::BindGroup> {
        let data_bind_group_layout = &bind_group_layouts[0];
        let lattice_bind_group_layout = &bind_group_layouts[1];

        let mut bind_groups = Vec::<wgpu::BindGroup>::new();
        bind_groups.push(
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: data_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.lattice_params.binding_resource(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: uniform_buffer.binding_resource(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.locks.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.regions.buffer.as_ref().expect("").as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.diffusion_matrix.buffer.as_ref().expect("").as_entire_binding(),
                    },
                ],
                label: Some("Data bind group"),
            })
        );

        for i in 0..2 {
            bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: lattice_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.lattices[i].lattice_buff.as_ref().expect("").as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.lattices[(i + 1) % 2].lattice_buff.as_ref().expect("").as_entire_binding(), // bind to opposite buffer
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.lattices[i].occupancy_buff.as_ref().expect("").as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.lattices[(i + 1) % 2].occupancy_buff.as_ref().expect("").as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: texture.binding_resource(),
                    },
                ],
                label: Some("Lattice bind group"),
            }));
        }
        return bind_groups;

    }

    fn build_texture_compute_pipeline(&self, bind_group_layouts: &Vec<wgpu::BindGroupLayout>, device: &wgpu::Device) -> wgpu::ComputePipeline {
        let data_bind_group_layout = &bind_group_layouts[0];
        let lattice_bind_group_layout = &bind_group_layouts[1];

        let binding = ShaderBuilder::new("final_stage.wgsl").unwrap();
        let shader_builder = binding.build();
        let compute_shader = device.create_shader_module(shader_builder);

        let compute_pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("Texture compute pipeline layout"),
                bind_group_layouts: &[data_bind_group_layout, lattice_bind_group_layout],
                push_constant_ranges: &[],
            }
        );
        let compute_pipeline = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Texture compute pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &compute_shader,
                entry_point: "main",
            }
        );
        compute_pipeline
    }

    fn texture_pass(&self, frame_num: u32, command_encoder: &mut wgpu::CommandEncoder) {
        let xdim = self.lattice_params.lattice_params.x_res as u32 + WORKGROUP_SIZE.0 - 1;
        let xgroups = xdim / WORKGROUP_SIZE.0;
        let ydim = self.lattice_params.lattice_params.y_res as u32 + WORKGROUP_SIZE.1 - 1;
        let ygroups = ydim / WORKGROUP_SIZE.1;
        let zdim = self.lattice_params.lattice_params.z_res as u32 + WORKGROUP_SIZE.2 - 1;
        let zgroups = zdim / WORKGROUP_SIZE.2;

        // Main compute pass
        {
            let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.push_debug_group("Prepare texture");
            cpass.set_pipeline(&self.texture_compute_pipeline.as_ref().expect(""));
            cpass.set_bind_group(0, &self.bind_groups[0], &[]);
            cpass.set_bind_group(1, &self.bind_groups[1 + (frame_num as usize % 2)], &[]);
            cpass.insert_debug_marker("Dispatching texture pass");
            cpass.dispatch_workgroups(xgroups, ygroups, zgroups);
            cpass.pop_debug_group();
        }        

    }
}
