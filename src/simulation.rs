use std::default;
use std::fs::File;
use std::collections::{VecDeque, HashMap};
use cgmath::num_traits::Pow;
use std::io::{BufReader, prelude::*};
use std::path::Path;
use serde_json::{Result, Value};

use log::{debug, info};
use tensor_wgpu::{Tensor2, Tensor3, Tensor1};
use wgpu::{util::DeviceExt};
use ndarray::{prelude::*, StrideShape};
use crate::MAX_PARTICLES_SITE;
use crate::{
    rdme::RDME, 
    texture::Texture, 
    lattice_params::LatticeParams, 
    lattice::Lattice,
    uniforms::UniformBuffer, types::{Region, Particle},
    preprocessor::ShaderBuilder,
    cme::CME,
    reactions_params::ReactionParams,
    statistics::{StatisticsGroup, SolverStatisticSample},
    region::{RegionType, Regions, Sphere},
    utils::{json_to_array, split_whitespace}
};


pub struct Simulation {
    rdme: Option<RDME>,
    cme: Option<CME>,
    bind_groups: Vec<wgpu::BindGroup>,
    pub lattices: Vec<Lattice>,
    pub lattice_params: LatticeParams,

    pub stats: VecDeque<SolverStatisticSample<i32>>,
    statistics_groups: Option<StatisticsGroup>,
    
    regions: Regions,
    diffusion_matrix: Tensor3<f32>,
    stoichiometry_matrix: Tensor2<i32>,
    reactions_idx: Tensor2<u32>,
    reaction_rates: Tensor2<f32>,
    reaction_params: ReactionParams,
    texture_compute_pipeline: Option<wgpu::ComputePipeline>,
}



impl Simulation {
    pub fn new(
        lattice_params: LatticeParams,
    ) -> Self {
        // A simulation has a lattice first
        let mut lattices = Vec::<Lattice>::new();
        for _ in 0..2 {
            lattices.push(Lattice::new(&lattice_params.raw))
        }
        
        let shape_regions = lattice_params.get_res_usize().f();
        let initial_volume = lattice_params.res().iter().product::<u32>();

        let regions = Regions {
            regions: Tensor3::<Region>::zeros(shape_regions),
            types: vec![RegionType::Cube { name: "background".to_string(), p0: [0.; 3], pf: lattice_params.dims() }],
            volumes: vec![initial_volume],
            index_buffer: None,
        };

        let reaction_params = ReactionParams::new(0, 0);

        let mut diffusion_matrix = Tensor3::<f32>::zeros((1, 1, 1).f());
        diffusion_matrix[[0, 0, 0]] = 8.15E-14 / 6.;
        let stoichiometry_matrix = Tensor2::<i32>::zeros((1, 1).f());
        let reactions_idx = Tensor2::<u32>::zeros((1, 3).f());
        let reaction_rates = Tensor2::<f32>::zeros((1, 1).f());

        let bind_groups = Vec::<wgpu::BindGroup>::new();

        Simulation {
            cme: None,
            rdme: None,
            bind_groups,
            lattices,
            lattice_params,
            regions,
            diffusion_matrix,
            stoichiometry_matrix,
            reactions_idx,
            reaction_rates,
            reaction_params,
            texture_compute_pipeline: None,
            statistics_groups: None,
            stats: VecDeque::<SolverStatisticSample<i32>>::new()
        }
        
    }

    pub fn step(
        &mut self,
        frame_num: u32,
        command_encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        write_freq: u32
    ) {
        self.rdme.as_ref()
            .expect("RDME must be initialized first")
            .step(
                &self.bind_groups[0],
                &self.bind_groups[1 + (frame_num as usize % 2)],
                &self.bind_groups[3],
                &self.statistics_groups.as_ref().expect("").bind_group,
                command_encoder, 
                &self.lattice_params.raw
            );

        // cme step
        self.cme.as_ref()
            .expect("CME must be initialized first")
            .step(
                &self.bind_groups[0],
                &self.bind_groups[1 + (frame_num as usize % 2)],
                &self.bind_groups[3],
                &self.statistics_groups.as_ref().expect("").bind_group,
                &self.bind_groups[4],
                command_encoder,
                &self.lattice_params.raw
            );

        // Fill the texture
        self.texture_pass(frame_num, command_encoder);

        // Copy source to destination (lattice and occupancy)
        command_encoder.copy_buffer_to_buffer(
            self.lattices[(frame_num as usize + 1) % 2].lattice.buffer(), 0, 
            self.lattices[frame_num as usize % 2].lattice.buffer(), 0,
            self.lattices[frame_num as usize % 2].lattice.buffer_size() as u64
        );

        command_encoder.copy_buffer_to_buffer(
            self.lattices[(frame_num as usize + 1) % 2].occupancy.buffer(), 0, 
            self.lattices[frame_num as usize % 2].occupancy.buffer(), 0,
            self.lattices[frame_num as usize % 2].occupancy.buffer_size() as u64
        );

        if frame_num % write_freq == 0 && frame_num > 0 {
            pollster::block_on(self.start_error_buffer_readbacks(device, frame_num));
        }

    }

    pub fn prepare_for_gpu(&mut self, 
        uniform_buffer: &UniformBuffer,
        texture: &Texture,
        device: &wgpu::Device,
    ) {
        let usage = wgpu::BufferUsages::STORAGE;

        self.lattices[1].lattice = self.lattices[0].lattice.clone();
        self.lattices[1].occupancy = self.lattices[0].occupancy.clone();
        self.lattices[0].start_buffers(device);
        self.lattices[1].start_buffers(device);
        
        // Diffusion matrix
        self.diffusion_matrix.create_buffer(device, usage, Some("Diffusion matrix buffer"));
        debug!("Diffusion matrix {}", self.diffusion_matrix);

        // Regions
        self.regions.regions.create_buffer(device, usage, Some("Regions Buffer"));

        self.lattice_params.create_buffer(device);
        self.reaction_params.create_buffer(device);
        
        // Stoichiometry matrix
        self.stoichiometry_matrix.create_buffer(device, usage, Some("Stoichiometry matrix buffer"));

        // Reactions idx
        self.reactions_idx.create_buffer(device, usage, Some("Reactions idx buffer"));

        // Reaction rates
        self.reaction_rates.create_buffer(device, usage, Some("Reaction rates buffer"));

        let bind_group_layouts = self.build_bind_group_layouts(uniform_buffer, texture, device);

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

        self.statistics_groups = Some(self.create_statistics(device));

        // RDME
        let rdme = RDME::new(&bind_group_layouts, &self.statistics_groups.as_ref().expect(""), &device);
        self.rdme = Some(rdme);

        // CME
        let cme = CME::new(&bind_group_layouts, &self.statistics_groups.as_ref().expect(""), &device);
        self.cme = Some(cme);
    }

    pub fn from_file<P: AsRef<Path>>(path: P, uniform_buffer: &UniformBuffer, device: &wgpu::Device) -> Result<(Self, Texture)> {
        let mut file = File::open(path).unwrap();
        let mut buff = String::new();
        file.read_to_string(&mut buff).unwrap();
     
        let data: Value = serde_json::from_str(&buff).unwrap();
        let simulation_params = LatticeParams::from_json_value(&data["parameters"]);

        let mut simulation = Simulation::new(simulation_params);
        simulation.json_regions(&data["regions"]);
        simulation.prepare_regions();
        simulation.json_particles(&data["particles"]);
        simulation.json_reactions(&data["reactions"]);

        // Create texture for GPU
        let lattice_resolution_usize = simulation.lattice_params.get_res_usize();
        let lattice_resolution: [u32; 3] = [lattice_resolution_usize[0] as u32, lattice_resolution_usize[1] as u32, lattice_resolution_usize[2] as u32];
        let texture = Texture::new(&lattice_resolution, wgpu::TextureFormat::R32Float, false, &device);
        
        simulation.prepare_for_gpu(&uniform_buffer, &texture, device);

        Ok((simulation, texture))
    }

}

// Read from json
impl Simulation {
    fn json_regions(&mut self, regions: &Value) {
        // regions is a list of regions. Loop over them and add
        assert!(regions.is_array());
        for region in regions.as_array().unwrap() {
            let region_type = region["type"].as_str().unwrap();
            match region_type {
                "cube" => {
                    let name = region["name"].as_str().unwrap();
                    let p0: [f32; 3] = json_to_array(&region["p0"]).unwrap();
                    let pf: [f32; 3] = json_to_array(&region["pf"]).unwrap();
                    let diffusion_rate = region["base_diffusion_rate"].as_f64().unwrap() as f32;
                    self.add_region(RegionType::Cube {
                        name: name.to_string(), 
                        p0, 
                        pf
                    }, diffusion_rate);
                },
                "capsid" => {
                    let shell_name = region["shell_name"].as_str().unwrap();
                    let interior_name = region["interior_name"].as_str().unwrap();
                    let center: [f32; 3] = json_to_array(&region["center"]).unwrap();
                    let dir: [f32; 3] = json_to_array(&region["dir"]).unwrap();
                    let internal_radius = region["internal_radius"].as_f64().unwrap() as f32;
                    let external_radius = region["external_radius"].as_f64().unwrap() as f32;
                    let total_length = region["total_length"].as_f64().unwrap() as f32;
                    let diffusion_rate = region["base_diffusion_rate"].as_f64().unwrap() as f32;
                    // let transition_rate = region["transition_rate"].as_f64().unwrap() as f32;
                    self.add_region(RegionType::Capsid { 
                        shell_name: shell_name.to_string(), 
                        interior_name: interior_name.to_string(), 
                        center, 
                        dir, 
                        internal_radius,
                        external_radius, 
                        total_length 
                    }, diffusion_rate);
                },
                "sparse" => {
                    panic!("Sparse regions not implemented yet");
                },
                _ => panic!("Region type not implemented yet")
            }
        }
    }
    
    fn json_particles(&mut self, particles: &Value) {
        // particles is a list of particles. Loop over them and add
        assert!(particles.is_array());
        for particle in particles.as_array().unwrap() {
            let name = particle["name"].as_str().unwrap();
            let to_region = particle["to_region"].as_str().unwrap();
            let logging = particle["logging"].as_bool().unwrap();
            let is_reservoir = particle["is_reservoir"].as_bool().unwrap();

            if particle.get("count") != None {
                let count = particle["count"].as_u64().unwrap() as u32;
                self.add_particle_count(name, to_region, count, logging, is_reservoir);
            } else if particle.get("concentration") != None {
                let concentration = particle["concentration"].as_f64().unwrap() as f32;
                self.add_particle_concentration(name, to_region, concentration, logging, is_reservoir);

            } else {
                panic!("Particle must have count or concentration");
            }

            // Sometimes diffusion rate is given. Other times it is not.
            if particle.get("diffusion_rate") != None {
                let diffusion_rate = particle["diffusion_rate"].as_object().unwrap();
                for (region, value) in diffusion_rate.iter() {
                    let rate = value.as_f64().unwrap() as f32;
                    self.set_diffusion_rate_particle(name, region, rate);
                }
            }

        }
    }

    fn json_reactions(&mut self, reactions: &Value) {
        // reactions are objects "reaction" : f32
        if reactions.is_null() {
            println!("No reactions found");
            return;
        }
        for (reaction_str, value) in reactions.as_object().unwrap() {
            let reaction = reaction_str.as_str();
            let mut line_vec = split_whitespace(reaction);
            // line_vec = line_vec[1..].to_vec();
            debug!("Line vec: {:?}", line_vec);
    
            // Reference: reactants: Vec<&str>, products: Vec<&str>, k: f32
            let mut reactants: Vec<&str> = Vec::new();
            let mut products: Vec<&str> = Vec::new();
            let mut k: f32 = value.as_f64().unwrap() as f32;
    
            let mut before_arrow: bool = true;
    
            for value in line_vec {
                match value.parse::<f32>() {
                    Ok(new_k) => {
                        k = k*new_k;
                        continue;
                    },
                    Err(_) => ()
                };
                match value {
                    "->" => before_arrow = false,
                    "+" => (),
                    _ => {
                        if before_arrow {
                            reactants.push(value);
                        } else {
                            products.push(value);
                        }
                    }
                }
            }
            self.add_reaction(reactants, products, k);
        }
    }
}

// Statistics
impl Simulation {
    fn create_statistics(&self, device: &wgpu::Device) -> StatisticsGroup {
        let lattice_data = &self.lattices[0].concentrations.data;
        let concentration = lattice_data.sum_axis(Axis(0)).sum_axis(Axis(0)).sum_axis(Axis(0)).to_vec();
        // I have to do this because I can't use the lattice data directly. I have to copy it to a new vector. TODO: Add functionality to tensor-wgpu

        let mut concentration_tensor = Tensor1::<i32>::from_data(
            concentration,
            StrideShape::from((self.reaction_params.raw_params.num_species as usize + 1,))
        );
        concentration_tensor.create_buffer(
            device,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::MAP_READ,
            Some("Concentration buffer")
        );

        // Particles to log:
        let mut hash_concentration = HashMap::new();
        for particle in self.lattices[0].logging_particles.iter() {
            hash_concentration.insert(
                self.lattices[0].particle_names[*particle as usize].clone(), 
                [
                    0, // Idx of the bind group (concentration = 0) 
                    *particle // Padding of the storage buffer. It's 1D, so it equals the idx of the particle
                ]
            );
        }
        StatisticsGroup::new(vec![concentration_tensor], hash_concentration, device)
    }


    pub async fn start_error_buffer_readbacks(&mut self, device: &wgpu::Device, frame_num: u32) {
        for (name, data) in self.statistics_groups.as_ref().expect("msg").logging_stats.iter() {
            let padding = data[1] as usize;
            let buffer = self.statistics_groups.as_ref().expect("msg").stats[data[0] as usize].buffer.as_ref().expect("msg");
            let buffer_slice = buffer.slice(..);
            // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

            device.poll(wgpu::Maintain::Wait);  // Do I need this if I'm using pollster to call this function?
            receiver.receive().await.unwrap().unwrap();
        
            let data_arr = buffer_slice.get_mapped_range();
            // data_arr has as many elements as the buffer size = len * size_of::<i32>. So 4 spaces per number. How do I get the final number?
            let val = i32::from_ne_bytes(data_arr[padding * 4..padding*4 + 4].try_into().unwrap());

            self.stats.push_back(
                SolverStatisticSample {
                    name: name.clone(),
                    value: val,
                    iteration_count: frame_num
                }
            );

            drop(data_arr);
            buffer.unmap();
        }
    }
}

// Region and particle methods
impl Simulation {

    pub fn prepare_regions(&mut self) {
        self.regions.prepare_regions();
    }
    
    pub fn add_region(&mut self, reg: RegionType, diffusion_rate: f32) {
        //self.regions.names.push(String::from(name));
        let transition_rate: f32 = 0.; //8.15E-14 / 6.;
        match reg {
            RegionType::Cube { name, p0, pf} => self.add_region_cube(&name, p0, pf, diffusion_rate, transition_rate),
            RegionType::Sphere { name, center, radius } => self.add_region_sphere(&name, center, radius, diffusion_rate, transition_rate),

            RegionType::Cylinder { name, p0, pf, radius} => self.add_region_cylinder(&name, p0, pf, radius, diffusion_rate, transition_rate),

            RegionType::SemiSphere { name, center, radius, direction } => self.add_region_semisphere(&name, center, radius, direction, diffusion_rate, transition_rate),

            RegionType::SphericalShell { shell_name, interior_name, center, internal_radius, external_radius } => {
                // A spherical shell is composed of two spheres: one interior and one exterior
                self.add_region_sphere(&shell_name, center, external_radius, diffusion_rate, transition_rate);
                self.add_region_sphere(&interior_name, center, internal_radius, diffusion_rate, transition_rate);
            },
            RegionType::CylindricalShell { shell_name, interior_name, p0, pf, internal_radius, external_radius } => {
                // A cylindrical shell is composed of two cylinders: one interior and one exterior
                self.add_region_cylinder(&shell_name, p0, pf, external_radius, diffusion_rate, transition_rate);
                self.add_region_cylinder(&interior_name, p0, pf, internal_radius, diffusion_rate, transition_rate);
            },
            RegionType::Capsid { shell_name, interior_name, center, dir, internal_radius, external_radius, total_length } => {
                // A cylinder and two semispheres
                //assert!(total_length <= 1.);
                let cylinder_length = total_length - 2. * external_radius;
                let p0 = [
                    center[0] - dir[0] * cylinder_length / 2.,
                    center[1] - dir[1] * cylinder_length / 2.,
                    center[2] - dir[2] * cylinder_length / 2.
                ];
                let pf = [
                    center[0] + dir[0] * cylinder_length / 2.,
                    center[1] + dir[1] * cylinder_length / 2.,
                    center[2] + dir[2] * cylinder_length / 2.
                ];
                debug!("p0: {:?}, pf: {:?}", p0, pf);
                let neg_dir = [-dir[0], -dir[1], -dir[2]];
                self.add_region_cylinder(&shell_name, p0, pf, external_radius, diffusion_rate, transition_rate);
                self.add_region_cylinder(&interior_name, p0, pf, internal_radius, diffusion_rate, transition_rate);
                self.add_region_semisphere("cap1", p0, external_radius, neg_dir, diffusion_rate, transition_rate);
                self.add_region_semisphere("cap2", pf, external_radius, dir, diffusion_rate, transition_rate);
                self.add_region_semisphere("inside1", p0, internal_radius, neg_dir, diffusion_rate, transition_rate);
                self.add_region_semisphere("inside2", pf, internal_radius, dir, diffusion_rate, transition_rate);
                // Join cap1 and cap2 to shell name
                // Join inside1 and inside2 to interior name
                self.join_regions("cap1", &shell_name);
                self.join_regions("cap2", &shell_name);
                self.join_regions("inside1", &interior_name);
                self.join_regions("inside2", &interior_name);
            }
            _ => panic!("Region type not implemented yet")
        }
        info!("Added region {:?}", self.regions.types.last().unwrap());
    }

    pub fn add_sparse_region(&mut self, name: &str, base_region: RegionType, to_region: &str, max_volume: u32, diffusion_rate: f32) {
        let to_region_idx = self.find_region_index(to_region).unwrap();
        assert!(self.regions.volumes[to_region_idx] > max_volume);
        debug!("Volume of {} is {}", to_region, self.regions.volumes[to_region_idx]);

        let transition_rate: f32 = 0.; //8.15E-14 / 6.;
        let voxel_size = self.lattice_params.get_voxel_size();
        let voxel_size_squared = voxel_size.iter().map(|x| (*x).pow(2)).collect::<Vec<f32>>();

        let radius = match base_region {
            RegionType::Sphere { name: _, center: _, radius } => {
                radius
            },
            _ => panic!("Only spheres can be added as sparse regions")
        };
        let radius_squared = radius * radius;
        let radius_voxels = voxel_size.iter().map(|&x| (radius / x) as usize).collect::<Vec<usize>>();

        let new_region_idx = self.regions.types.len() as Region;

        let mut curr_volume = 0u32;
        let mut added = false;
        while curr_volume < max_volume {            
            // Generate a random point making sure that the sphere fits.
            let point = self.regions.generate_boundary_aware(to_region_idx, voxel_size, &radius_voxels);

            // Add the region iterating over the tensor
            for i in point[0] - radius_voxels[0]..point[0] + radius_voxels[0] {
                for j in point[1] - radius_voxels[1]..point[1] + radius_voxels[1] {
                    for k in point[2] - radius_voxels[2]..point[2] + radius_voxels[2] {
                        let dist: f32 = (i.abs_diff(point[0]) as f32).pow(2.) * voxel_size_squared[0] + 
                                        (j.abs_diff(point[1]) as f32).pow(2.) * voxel_size_squared[1] +
                                        (k.abs_diff(point[2]) as f32).pow(2.) * voxel_size_squared[2];
                        if dist < radius_squared {
                            self.regions.check_regions_collision([i, j, k]);
                            self.regions.set_value_position(new_region_idx, [i, j, k]);
                            curr_volume += 1;
                        }
                    }
                }
            }
            added = true;
        }

        if added {
            let base_region = Sphere { name: name.to_string(), center: [0.; 3], radius: radius };
            self.regions.types.push(RegionType::Sparse { name: name.to_string(), base_region: base_region });
            self.update_matrices_region(diffusion_rate, transition_rate);
            self.regions.volumes.push(curr_volume);
        } else {
            panic!("Could not add sparse region");
        }
    }

    fn update_matrices_region(&mut self, diffusion_rate: f32, transition_rate: f32) {
        // Add the region to the diffusion matrix. 
        self.diffusion_matrix.enlarge_dimension(0, transition_rate);
        self.diffusion_matrix.enlarge_dimension(1, transition_rate);
        let num_regions = self.regions.types.len() - 1;
        let num_particles = self.lattices[0].particle_names.len();
        for i_part in 0..num_particles {
            self.diffusion_matrix[[num_regions, num_regions, i_part]] = diffusion_rate;
        }
        // println!("Region added. New diffusion matrix: {}", self.diffusion_matrix);
        self.lattice_params.add_region(); // TODO: This must be a method of lattice params
    }

    fn add_region_cylinder(&mut self, name: &str, p0: [f32; 3], pf: [f32; 3], radius: f32, diffusion_rate: f32, transition_rate: f32) {
        self.regions.types.push(RegionType::Cylinder { name: name.to_string(), p0, pf, radius });
        
        let new_region_idx = (self.regions.types.len() - 1) as Region;
        let res = self.lattice_params.get_res_f32();
        let voxel_size = self.lattice_params.get_voxel_size();
        let radius_lattice = [
            (radius / voxel_size[0]),
            (radius / voxel_size[1]),
            (radius / voxel_size[2])
        ];

        // Calculate center of the two planes
        let ip0 = [p0[0] / voxel_size[0], p0[1] / voxel_size[1], p0[2] / voxel_size[2]];
        let ipf = [pf[0] / voxel_size[0], pf[1] / voxel_size[1], pf[2] / voxel_size[2]];
        // println!("ip0: {:?}, ipf: {:?}", ip0, ipf);

        let v = [ipf[0] - ip0[0], ipf[1] - ip0[1], ipf[2] - ip0[2]];
        let v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        let r_const: f32 = radius * radius * v2;

        let voxel_size_squared = voxel_size.iter().map(|x| (*x).pow(2)).collect::<Vec<f32>>();
        let mut volume = 0u32;

        // We only need to check inside the box. Probably there's a better way with iters.
        // No need to do fancy stuff because ip0_i < ipf_i for all i 
        let mut x = (ip0[0] - radius_lattice[0]).max(0.);
        let mut y = (ip0[1] - radius_lattice[1]).max(0.);
        let mut z = (ip0[2] - radius_lattice[2]).max(0.);

        let xf = (ipf[0] + radius_lattice[0]).min(res[0]);
        let yf = (ipf[1] + radius_lattice[1]).min(res[1]);
        let zf = (ipf[2] + radius_lattice[2]).min(res[2]);

        while x < xf {
            while y < yf {
                while z < zf {
                    // Lies betwen the planes:
                    let w = [x - ipf[0], y - ipf[1], z - ipf[2]];
                    let proj = w[0] * v[0] + w[1] * v[1] + w[2] * v[2];
                    if proj > 0. {
                        z += 1.;
                        continue;
                    }

                    let w = [x - ip0[0], y - ip0[1], z - ip0[2]];
                    let proj = w[0] * v[0] + w[1] * v[1] + w[2] * v[2];
                    if proj < 0. {
                        z += 1.;
                        continue;
                    }

                    // Lies inside the cylinder:
                    let cp = [w[1] * v[2] - w[2] * v[1], w[2] * v[0] - w[0] * v[2], w[0] * v[1] - w[1] * v[0]];
                    let cp_norm = cp[0] * cp[0] * voxel_size_squared[0] + 
                                       cp[1] * cp[1] * voxel_size_squared[1] + 
                                       cp[2] * cp[2] * voxel_size_squared[2];

                    if cp_norm <= r_const {
                        let pos = [x as usize, y as usize, z as usize];
                        self.regions.check_regions_collision(pos);
                        self.regions.set_value_position(new_region_idx, pos);
                        volume += 1;
                    }
                    z += 1.;
                }
                z = (ip0[2] - radius_lattice[2]).max(0.);
                y += 1.;
            }
            y = (ip0[1] - radius_lattice[1]).max(0.);
            x += 1.;
        }

        self.update_matrices_region(diffusion_rate, transition_rate);
        self.regions.volumes.push(volume);
    }

    fn add_region_cube(&mut self, name: &str, starting_pos: [f32; 3], ending_pos: [f32; 3], diffusion_rate: f32, transition_rate: f32) {
        self.regions.types.push(RegionType::Cube { name: name.to_string(), p0: starting_pos, pf: ending_pos });
        
        assert!(starting_pos[0] <= ending_pos[0] && starting_pos[1] <= ending_pos[1] && starting_pos[2] <= ending_pos[2]);
        let new_region_idx = (self.regions.types.len() - 1) as Region;
        let voxel_size = self.lattice_params.get_voxel_size();
        
        let start = (
            (starting_pos[0] / voxel_size[0]) as usize,  // Probably dividing here by the dimensions
            (starting_pos[1] / voxel_size[1]) as usize,
            (starting_pos[2] / voxel_size[2]) as usize
        );
        let end = (
            (ending_pos[0] / voxel_size[0]) as usize,
            (ending_pos[1] / voxel_size[1]) as usize,
            (ending_pos[2] / voxel_size[2]) as usize
        );

        let volume = (end.0 - start.0) * (end.1 - start.1) * (end.2 - start.2);

        for x in start.0..end.0 {
            for y in start.1..end.1 {
                for z in start.2..end.2 {
                    let pos = [x as usize, y as usize, z as usize];
                    self.regions.check_regions_collision(pos);
                    self.regions.set_value_position(new_region_idx, [x, y, z]);
                }
            }
        }
        self.update_matrices_region(diffusion_rate, transition_rate);
        self.regions.volumes.push(volume as u32);
    }

    fn add_region_sphere(&mut self, name: &str, center: [f32; 3], radius: f32, diffusion_rate: f32, transition_rate: f32) {
        self.regions.types.push(RegionType::Sphere { name: name.to_string(), center: center, radius: radius });
        let res = self.lattice_params.get_res_f32();
        let new_region_idx = (self.regions.types.len() - 1) as Region;
        let voxel_size = self.lattice_params.get_voxel_size();

        let center = (
            center[0] / voxel_size[0],
            center[1] / voxel_size[1],
            center[2] / voxel_size[2]
        );
        let radius_squared = radius.powf(2.);
        //let iradius = (radius * res.0 as f32) as i32; //TODO: Use real measurements
        
        let voxel_size_squared = voxel_size.iter().map(|x| (*x).pow(2)).collect::<Vec<f32>>();
        let mut volume = 0u32;
        let (mut x, mut y, mut z) = (0., 0., 0.);
        while x < res[0] {
            while y < res[1] {
                while z < res[2] {
                    let dist: f32 = (x - center.0).pow(2) * voxel_size_squared[0] + 
                                    (y - center.1).pow(2) * voxel_size_squared[1] +
                                    (z - center.2).pow(2) * voxel_size_squared[2];
                    if dist < radius_squared {
                        let pos = [x as usize, y as usize, z as usize];
                        self.regions.check_regions_collision(pos);
                        self.regions.set_value_position(new_region_idx, [x as usize, y as usize, z as usize]);
                        volume += 1;
                    }
                    z += 1.;
                }
                z = 0.;
                y += 1.;
            }
            y = 0.;
            x += 1.;
        }
        self.update_matrices_region(diffusion_rate, transition_rate);
        self.regions.volumes.push(volume);
    }

    fn add_region_semisphere(&mut self, name: &str, center: [f32; 3], radius: f32, direction: [f32; 3], diffusion_rate: f32, transition_rate: f32) {
        self.regions.types.push(RegionType::SemiSphere { name: name.to_string(), center, radius, direction });
        let res = [self.lattice_params.raw.res[0] as f32, self.lattice_params.raw.res[1] as f32, self.lattice_params.raw.res[2] as f32];
        let voxel_size = self.lattice_params.get_voxel_size();

        let center = (
            center[0] / voxel_size[0],
            center[1] / voxel_size[1],
            center[2] / voxel_size[2]
        );
        let radius_squared = radius.powf(2.);
        let voxel_size_squared = voxel_size.iter().map(|x| (*x).pow(2)).collect::<Vec<f32>>();
        
        let new_region_idx = (self.regions.types.len() - 1) as Region;
        let (mut x, mut y, mut z) = (0., 0., 0.);
        let mut volume = 0u32;
        while x < res[0] {
            while y < res[1] {
                while z < res[2] {
                    if ((x - center.0) as f32) * direction[0] + ((y - center.1) as f32) * direction[1] + ((z - center.2) as f32) * direction[2] < 0. {
                        z += 1.;
                        continue;
                    }
                    let dist: f32 = (x - center.0).pow(2) * voxel_size_squared[0] + 
                                    (y - center.1).pow(2) * voxel_size_squared[1] +
                                    (z - center.2).pow(2) * voxel_size_squared[2];
                    if dist < radius_squared {
                        let pos = [x as usize, y as usize, z as usize];
                        self.regions.check_regions_collision(pos);
                        self.regions.set_value_position(new_region_idx, pos);
                        volume += 1;
                    }
                    z += 1.;
                }
                z = 0.;
                y += 1.;
            }
            y = 0.;
            x += 1.;
        }
        self.update_matrices_region(diffusion_rate, transition_rate);
        self.regions.volumes.push(volume);
    }

    fn find_region_index(&mut self, name: &str) -> Option<usize> {
        self.regions.types.iter().position(|region| {
            match region {
                RegionType::Cube { name: region_name, .. } => { region_name == name },
                RegionType::Sphere { name: region_name, .. } => { region_name == name },
                RegionType::Cylinder { name: region_name,.. } => { region_name == name },
                RegionType::SemiSphere { name: region_name,.. } => { region_name == name },
                RegionType::Sparse { name: region_name, .. } => { region_name == name },
                _ => { false }
            }
        })
    }

    fn join_regions(&mut self, region_delete: &str, to_region: &str) {
        let region_delete_idx = self.find_region_index(region_delete).expect("Region not found") as Region;
        let to_region_idx = self.find_region_index(to_region).expect("Region not found") as Region;

        let res = [
            self.lattice_params.raw.res[0] as usize, 
            self.lattice_params.raw.res[1] as usize,
            self.lattice_params.raw.res[2] as usize
        ];

        for x in 0..res[0] {
            for y in 0..res[1] {
                for z in 0..res[2] {
                    if self.regions.cell([x, y, z]) == region_delete_idx {
                        self.regions.set_value_position(to_region_idx, [x, y, z]);
                    }
                    if self.regions.cell([x, y, z]) > region_delete_idx {
                        self.regions.substract_value_position(1, [x, y, z]);
                    }
                }
            }
        }

        // Update the regions matrix: region_keep can be removed
        self.diffusion_matrix.remove_element_at(0, region_delete_idx as usize);
        self.diffusion_matrix.remove_element_at(1, region_delete_idx as usize);
        self.lattice_params.remove_region();

        // Update the volume
        self.regions.volumes[to_region_idx as usize] += self.regions.volumes[region_delete_idx as usize];

        self.regions.remove_region(region_delete_idx as usize);
    }

    pub fn add_particle_count(&mut self, name: &str, to_region: &str, count: u32, logging: bool, is_reservoir: bool) {
        // Add particles to the simulation. It will be added to the region specified by to_region
        let region_idx = self.find_region_index(to_region).expect("Region not found");
        let particle_idx = self.lattices[0].particle_names.len() as Particle;

        self.lattices[0].particle_names.push(String::from(name));

        let regions_idx_buffer = &self.regions.index_buffer.as_ref().unwrap()[&(region_idx as u32)];

        self.lattices[0].init_random_particles_region(particle_idx, count, regions_idx_buffer, is_reservoir);

        // Update the diffusion matrix
        self.diffusion_matrix.copy_dimension(2);

        // Add one column to stoichiometry matrix.
        self.stoichiometry_matrix.enlarge_dimension(1, 0);
        self.reaction_params.raw_params.num_species += 1;
        if logging {
            self.lattices[0].logging_particles.push(particle_idx);
        }
        info!("{} particles of type {} added to region {}", count, name, to_region);
        debug!("Concentrations after adding particles: {:?}", self.lattices[0].concentrations.shape());
    }

    pub fn add_particle_concentration(&mut self, name: &str, to_region: &str, concentration: f32, logging: bool, is_reservoir: bool) {
        // Take the volume of the region and calculate the number of particles. Then call add_particle_count
        let region_idx = self.find_region_index(to_region).expect("Region not found");
        let volume = self.regions.volumes[region_idx] as f32;
        let count = (concentration * volume) as u32;
        self.add_particle_count(name, to_region, count, logging, is_reservoir);
        info!("Concentration {} of particles of type {} added to region {}", concentration, name, to_region);
    }

    pub fn fill_region(&mut self, name: &str, to_region: &str, logging: bool) {
        let region_idx = self.find_region_index(to_region).expect("Region not found");
        let particle_idx = self.lattices[0].particle_names.len() as Particle;
        self.lattices[0].particle_names.push(String::from(name));
        let regions_idx_buffer = &self.regions.index_buffer.as_ref().unwrap()[&(region_idx as u32)];
        self.lattices[0].fill_region_particles(particle_idx, regions_idx_buffer);
        // Update the diffusion matrix
        self.diffusion_matrix.copy_dimension(2);

        // Add one column to stoichiometry matrix.
        self.stoichiometry_matrix.enlarge_dimension(1, 0);
        self.reaction_params.raw_params.num_species += 1;
        if logging {
            self.lattices[0].logging_particles.push(particle_idx);
        }
    }

    pub fn particle_random_walk(&mut self, name: &str, to_region: &str, total_length: f32, block_length: f32, radius: f32, logging: bool) {
        // Add this particle with a random walk within the region
        let region_idx = self.find_region_index(to_region).expect("Region not found");
        let particle_idx = self.lattices[0].particle_names.len() as Particle;

        self.lattices[0].particle_names.push(String::from(name));

        // let regions_idx_buffer = &self.regions.index_buffer.as_ref().unwrap()[&(region_idx as u32)];
        let _ = self.lattices[0].init_random_walk_particles(particle_idx, total_length, block_length, radius, region_idx, &self.regions);

        // Update the diffusion matrix
        self.diffusion_matrix.copy_dimension(2);

        // Add one column to stoichiometry matrix.
        self.stoichiometry_matrix.enlarge_dimension(1, 0);
        self.reaction_params.raw_params.num_species += 1;
        if logging {
            self.lattices[0].logging_particles.push(particle_idx);
        }

        info!("Particle {} added to region {} as random walk", name, to_region);

    }

    #[allow(dead_code)]
    pub fn set_diffusion_rate(&mut self, region: &str, diffusion_rate: f32) {
        // Write diffusion rate for all the particles in a region
        let region_idx = self.find_region_index(region).expect("Region not found");    
        let num_particles = self.lattices[0].particle_names.len();
        for i_part in 0..num_particles {
            self.diffusion_matrix[[region_idx, region_idx, i_part]] = diffusion_rate;
        }
    }

    #[allow(dead_code)]
    pub fn set_diffusion_rate_particle(&mut self, particle: &str, region: &str, diffusion_rate: f32) {
        // Write diffusion rate for a given particle. Note: Diffusion != Transition (Region x Region x Particle)
        let region_idx = self.find_region_index(region).expect("Region not found");    
        let particle_idx = self.lattices[0].particle_names.iter().position(|x| x == particle).unwrap() as usize;
        self.diffusion_matrix[[region_idx, region_idx, particle_idx]] = diffusion_rate;
    }

    #[allow(dead_code)]
    pub fn set_transition_rate(&mut self, from_region: &str, to_region: &str, transition_rate: f32) {
        // Write transition rate for all the particles in a region  
        let from_region_idx = self.find_region_index(from_region).expect("Region not found");    
        let to_region_idx = self.find_region_index(to_region).expect("Region not found");    
        let num_particles = self.lattices[0].particle_names.len();
        for i_part in 0..num_particles {
            self.diffusion_matrix[[from_region_idx, to_region_idx, i_part]] = transition_rate;
        }
    }

    #[allow(dead_code)]
    pub fn set_transition_rate_particle(&mut self, particle: &str, from_region: &str, to_region: &str, transition_rate: f32) {
        // Write transition rate for a given particle. Note: Diffusion != Transition (Region x Region x Particle)
        let from_region_idx = self.find_region_index(from_region).expect("Region not found");    
        let to_region_idx = self.find_region_index(to_region).expect("Region not found");    
        let particle_idx = self.lattices[0].particle_names.iter().position(|x| x == particle).unwrap() as usize;
        self.diffusion_matrix[[from_region_idx, to_region_idx, particle_idx]] = transition_rate;
    }

    pub fn add_reaction(&mut self, reactants: Vec<&str>, products: Vec<&str>, k: f32) {
        info!("Adding reaction: {} -> {} with rate {}", reactants.iter().map(|x| *x).collect::<Vec<&str>>().join(" + "), products.iter().map(|x| *x).collect::<Vec<&str>>().join(" + "), k);
        // Add a reaction to the simulation. It is independent of the region since particles are defined per region.
        // Maybe the following can be made two map iters
        let mut reactants_idx = Vec::<usize>::new();
        for reactant in reactants {
            match self.lattices[0].find_particle(reactant) {
                Some(idx) => reactants_idx.push(idx),
                None => panic!("Reactant {} not found", reactant)
            };
        }
        let mut products_idx = Vec::<usize>::new();
        for product in products {
            match self.lattices[0].find_particle(product) {
                Some(idx) => products_idx.push(idx),
                None => panic!("Product {} not found", product)
            };
        }
        debug!("Reactants: {:?}, products: {:?}", reactants_idx, products_idx);

        // Update stoichiometry matrix
        self.stoichiometry_matrix.enlarge_dimension(0, 0);
        let num_rows = self.stoichiometry_matrix.shape()[0];

        for reactant_idx in &reactants_idx {
            self.stoichiometry_matrix[[num_rows - 1, *reactant_idx]] -= 1;
        }
        for product_idx in products_idx {
            self.stoichiometry_matrix[[num_rows - 1, product_idx]] += 1;
        }
        debug!("New stoichiometry matrix: {} with {} rows and {} columns", self.stoichiometry_matrix, self.stoichiometry_matrix.shape()[0], self.stoichiometry_matrix.shape()[1]);

        // Update index matrix
        reactants_idx.extend(std::iter::repeat(0 as usize).take(3 - reactants_idx.len()));
        let reactants_idx_u32 = reactants_idx.iter().map(|x| *x as u32).collect::<Vec<u32>>();
        self.reactions_idx.concatenate_vector(&reactants_idx_u32, 0);
        debug!("Reactants idx : {:?}", reactants_idx_u32);
        debug!("New reactions index matrix: {}", self.reactions_idx);

        // Update reaction rates vector
        self.reaction_rates.concatenate_vector(&vec![k], 0);
        debug!("New reaction rates vector: {}", self.reaction_rates);
        self.reaction_params.raw_params.num_reactions += 1;
    }
}


// GPU functions
impl Simulation {
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
                        ty: self.lattice_params.binding_type(),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: self.reaction_params.binding_type(),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: uniform_buffer.binding_type(),
                        count: None,
                    },
                    // Region data
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { 
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(self.regions.regions.buffer_size() as _),
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
                            min_binding_size: wgpu::BufferSize::new(self.diffusion_matrix.buffer_size() as _)
                        },
                        count: None,
                    },
                ],
                label: Some("Data bind group layout")
            }
        );

        let boundaries_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    // Reservoirs
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { 
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(self.lattices[0].reservoir.buffer_size() as _),
                        },
                        count: None,
                    },
                ],
                label: Some("Boundaries bind group layout")
            }
        );

        // Lattice and RDME bind group layouts
        let lattice_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { 
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(self.lattices[0].lattice.buffer_size() as _),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { 
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(self.lattices[1].lattice.buffer_size() as _),
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
                            min_binding_size: wgpu::BufferSize::new(self.lattices[0].occupancy.buffer_size() as _),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { 
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(self.lattices[1].occupancy.buffer_size() as _),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: texture.binding_type(wgpu::StorageTextureAccess::ReadWrite),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { 
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(self.lattices[0].lock_cells.buffer_size() as _),
                        },
                        count: None,
                    },
                ],
                label: None
            }
        );

        let reactions_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { 
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(self.lattices[0].concentrations.buffer_size() as _),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { 
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(self.stoichiometry_matrix.buffer_size() as _)
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { 
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(self.reactions_idx.buffer_size() as _)
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { 
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(self.reaction_rates.buffer_size() as _)
                        },
                        count: None,
                    },
                ],
                label: None
            }
        );
        vec![data_bind_group_layout, lattice_bind_group_layout, reactions_bind_group_layout, boundaries_bind_group_layout]
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
        let reactions_bind_group_layout = &bind_group_layouts[2];
        let boundaries_bind_group_layout = &bind_group_layouts[3];

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
                        resource: self.reaction_params.binding_resource(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniform_buffer.binding_resource(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.regions.regions.binding_resource(),
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
                        resource: self.lattices[i].lattice.binding_resource(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.lattices[(i + 1) % 2].lattice.binding_resource(), // bind to opposite buffer
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.lattices[i].occupancy.binding_resource(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.lattices[(i + 1) % 2].occupancy.binding_resource(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: texture.binding_resource(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.lattices[0].lock_cells.binding_resource(),
                    },
                ],
                label: Some("Lattice bind group"),
            }));
        }

        bind_groups.push(
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: reactions_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.lattices[0].concentrations.binding_resource(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.stoichiometry_matrix.binding_resource(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.reactions_idx.binding_resource(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.reaction_rates.binding_resource(),
                    },
                ],
                label: Some("Reactions bind group"),
            })
        );

        bind_groups.push(
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: boundaries_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.lattices[0].reservoir.binding_resource(),
                    },
                ],
                label: Some("Boundaries bind group"),
            })
        );

        return bind_groups;

    }

    fn build_texture_compute_pipeline(&self, bind_group_layouts: &Vec<wgpu::BindGroupLayout>, device: &wgpu::Device) -> wgpu::ComputePipeline {
        let data_bind_group_layout = &bind_group_layouts[0];
        let lattice_bind_group_layout = &bind_group_layouts[1];
        let boundaries_bind_group_layout = &bind_group_layouts[3];

        let binding = ShaderBuilder::new("final_stage.wgsl").unwrap();
        let shader_builder = binding.build();
        let compute_shader = device.create_shader_module(shader_builder);

        let compute_pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("Texture compute pipeline layout"),
                bind_group_layouts: &[data_bind_group_layout, lattice_bind_group_layout, boundaries_bind_group_layout],
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
        const WORKGROUP_SIZE: (u32, u32, u32) = (1, 1, 1);
        let xdim = self.lattice_params.raw.res[0] as u32 + WORKGROUP_SIZE.0 - 1;
        let xgroups = xdim / WORKGROUP_SIZE.0;
        let ydim = self.lattice_params.raw.res[1] as u32 + WORKGROUP_SIZE.1 - 1;
        let ygroups = ydim / WORKGROUP_SIZE.1;
        let zdim = self.lattice_params.raw.res[2] as u32 + WORKGROUP_SIZE.2 - 1;
        let zgroups = zdim / WORKGROUP_SIZE.2;

        // Main compute pass
        {
            let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.push_debug_group("Prepare texture");
            cpass.set_pipeline(&self.texture_compute_pipeline.as_ref().expect(""));
            cpass.set_bind_group(0, &self.bind_groups[0], &[]);
            cpass.set_bind_group(1, &self.bind_groups[1 + (frame_num as usize % 2)], &[]);
            cpass.set_bind_group(2, &self.bind_groups[4], &[]);
            //cpass.set_bind_group(2, &self.statistics.as_ref().expect("").bind_group, &[]);
            cpass.insert_debug_marker("Dispatching texture pass");
            cpass.dispatch_workgroups(xgroups, ygroups, zgroups);
            cpass.pop_debug_group();
        }        
    }
}
