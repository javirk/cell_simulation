use core::fmt;
use std::collections::VecDeque;
use rand::Rng;
use tensor_wgpu::{Tensor4, Tensor3};
use ndarray::prelude::*;
use log::debug;

use crate::{MAX_PARTICLES_SITE};
use crate::lattice_params::Params;
use crate::region::Regions;
use crate::types::Particle;
use crate::utils::random_direction_sphere;

pub struct Lattice {
    pub lattice: Tensor4<Particle>,
    pub occupancy: Tensor3<u32>,
    pub concentrations: Tensor4<i32>,
    pub reservoir: Tensor3<Particle>,
    pub lattice_params: Params,
    pub particle_names: Vec<String>,
    pub logging_particles: Vec<Particle>,
    pub lock_cells: Tensor3<u32>,
}

impl Lattice {
    pub fn new(
        params: &Params,
    ) -> Self {
        let shape_3d = (params.res[0] as usize, params.res[1] as usize, params.res[2] as usize).f();
        let shape_lattice = (params.res[0] as usize, params.res[1] as usize, params.res[2] as usize, MAX_PARTICLES_SITE as usize).f();
        let shape_concentrations = (params.res[0] as usize, params.res[1] as usize, params.res[2] as usize, 1).f();

        let particle_names: Vec<String> = vec![String::from("void")];

        let lattice: Tensor4<Particle> = Tensor4::<Particle>::zeros(shape_lattice);
        let occupancy: Tensor3<u32> = Tensor3::<u32>::zeros(shape_3d);
        let concentrations: Tensor4<i32> = Tensor4::<i32>::zeros(shape_concentrations);
        let reservoir = Tensor3::<Particle>::zeros(shape_3d);
        let lock_cells = Tensor3::<u32>::zeros(shape_3d);

        Lattice {
            lattice,
            occupancy,
            concentrations,
            reservoir,
            lattice_params: *params,
            particle_names,
            logging_particles: Vec::new(),
            lock_cells,
        }
    }

    pub fn start_buffers(&mut self, device: &wgpu::Device) {
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        self.lattice.create_buffer(device, usage, Some("Lattice Buffer"));
        self.occupancy.create_buffer(device, usage, Some("Occupancy Buffer"));
        self.concentrations.create_buffer(device, usage, Some("Concentrations Buffer"));
        self.reservoir.create_buffer(device, usage, Some("Reservoir Buffer"));
        self.lock_cells.create_buffer(device, usage, Some("Lock Cells Buffer"));
    }


    pub fn init_random_particles_region(&mut self, particle: Particle, num_particles: u32, regions_idx_buffer: &Vec<u32>, is_reservoir: bool) {
        self.concentrations.enlarge_dimension(3, 0);
        let mut rng = rand::thread_rng();
        for _ in 0..num_particles {
            let mut i_ret = 0;
            while i_ret < 100 {
                let position_idx = rng.gen_range(0..regions_idx_buffer.len());
                let position = regions_idx_buffer[position_idx];
                // Transform the position idx to the 3D coordinates
                let site = self.idx_to_site_usize(position);
                // There must be a better way to do this
                if is_reservoir {
                    match self.add_reservoir_site(site, particle) {
                        Ok(_) => break,
                        Err(_) => {
                            i_ret += 1;
                            continue;
                        },
                    }
                } else {
                    match self.add_particle_site(site, particle) {
                        Ok(_) => break,
                        Err(_) => {
                            i_ret += 1;
                            continue;
                        },
                    }
                }

            }
            if i_ret == 100 {
                panic!("Could not add particle to region");
            }
        }
    }

    pub fn fill_region_particles(&mut self, particle: Particle, regions_idx_buffer: &Vec<u32>) {
        self.concentrations.enlarge_dimension(3, 0);
        for position_idx in 0..regions_idx_buffer.len() {
            let position = regions_idx_buffer[position_idx];

            // Transform the position idx to the 3D coordinates
            let site = self.idx_to_site_usize(position);
            match self.add_particle_site(site, particle) {
                Ok(_) => continue,
                Err(_) => panic!("Could not add particle to region"),
            }
        }
    }

    pub fn init_random_walk_particles(&mut self, particle: Particle, total_length: f32, block_length: f32, radius: f32, region_idx: usize, regions: &Regions) -> Result<(), String> {
        // The particles are added as a cylinder, in blocks of length block_length. Each block has a random direction with at most 75 degrees to the previous
        // If the outside of the region is reached, we step back a few steps (step_backwards) and try again
        self.concentrations.enlarge_dimension(3, 0);
        let step_backwards: i32 = 25;
        let mut curr_length = 0.;

        // First block is random. Both position and direction, but the distance to the edge of the region in the direction must be greater than steps_backwards * block_length
        let voxel_size = self.lattice_params.get_voxel_size();
        let safety_radius = block_length * (step_backwards as f32) * 0.4;  // 0.4 is a safety factor
        let safety_radius_voxels = voxel_size.iter().map(|&x| (safety_radius / x) as usize).collect::<Vec<usize>>();
        let site_us = regions.generate_boundary_aware(region_idx, voxel_size, &safety_radius_voxels);
        let mut site = [site_us[0] as u32, site_us[1] as u32, site_us[2] as u32];
        let mut rng = rand::thread_rng();
        let mut direction = random_direction_sphere(&mut rng);

        // VecDeque to store the last steps_backwards sites
        let mut queue_sites: VecDeque<[u32; 3]> = VecDeque::new();
        queue_sites.push_back(site);
        let mut queue_particles: VecDeque<Vec<u32>> = VecDeque::new();

        while curr_length < total_length {
            // Add a block of particles
            site = match self.add_particles_cylinder(site, direction, block_length, radius, particle, regions) {
                Err(e) => {  // Error, added_particles
                    // Step back step_backwards steps. Extremely unsafe because it could go out of bounds, but in theory we took care of that before at regions.generate_boundary_aware
                    let steps = (queue_sites.len() as i32 - 1).min(step_backwards);
                    //println!("{}: Stepping back {} steps", e.0, steps);
                    //println!("Current length: {}", queue_sites.len());
                    for _ in 0..steps {
                        queue_sites.pop_back();
                        let particles_remove = queue_particles.pop_back().unwrap();
                        for particle_idx in particles_remove {
                            let site_remove = self.idx_to_site_usize(particle_idx);
                            self.remove_particle_site(site_remove, particle)?;
                        }
                        curr_length -= block_length;
                    }
                    assert_eq!(queue_sites.len(), 1);
                    queue_sites[0]
                },
                Ok(s) => {  // Final site, added particles
                    let step_back_u = step_backwards as usize;
                    curr_length += block_length;
                    queue_sites.push_back(s.0);
                    if queue_sites.len() > step_back_u + 1 {
                        queue_sites.pop_front();
                    }

                    let added_particles = s.1;
                    queue_particles.push_back(added_particles);
                    if queue_particles.len() > step_back_u + 1 {
                        queue_particles.pop_front();
                    }
                    debug!("Added particles. Current length: {}", curr_length);
                    s.0
                }
            };
            direction = random_direction_sphere(&mut rng);
        }
        Ok(())
    }

    /// Adds a cylinder of particles of length length and radius radius, with the center at initial_site and direction direction
    /// Returns the final site (final site of the cylinder) and the list of added particles as idx. One should use self.idx_to_site later to get the 3D coordinates
    fn add_particles_cylinder(&mut self, initial_site: [u32; 3], direction: [f32; 3], length: f32, radius: f32, particle: Particle, regions: &Regions) -> Result<([u32; 3], Vec<u32>), (String, Vec<u32>)> {
        // length and radius are in real units, so I have to convert them first. initial_site is in lattice units. 
        // TODO: Put all this outside so it's not calculated many times
        let initial_region = regions.get_value_position([initial_site[0] as usize, initial_site[1] as usize, initial_site[2] as usize]);
        
        let voxel_size = self.lattice_params.get_voxel_size();
        let res = self.lattice_params.get_res_f32();
        let length_lattice = [
            length / voxel_size[0],
            length / voxel_size[1],
            length / voxel_size[2]
        ];
        let radius_lattice = [
            (radius / voxel_size[0]),
            (radius / voxel_size[1]),
            (radius / voxel_size[2])
        ];

        let ip0 = initial_site.iter().map(|x| *x as f32).collect::<Vec<f32>>();
        let ipf = [
            ip0[0] + direction[0] * length_lattice[0],
            ip0[1] + direction[1] * length_lattice[1],
            ip0[2] + direction[2] * length_lattice[2]
        ];
        // let sign = direction.iter().map(|x| x.signum()).collect::<Vec<f32>>();
        
        let v = [ipf[0] - ip0[0], ipf[1] - ip0[1], ipf[2] - ip0[2]];
        let v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        let r_const: f32 = radius * radius * v2;

        let voxel_size_squared = voxel_size.iter().map(|x| (*x).powf(2.)).collect::<Vec<f32>>();

        // We only need to check inside the box.
        // In this case, we don't have ip0_i < ipf_i. So we need to find the corners of the box
        let mut x = ((ip0[0] - radius_lattice[0]).min(ipf[0] + radius_lattice[0])).max(0.);
        let mut y = ((ip0[1] - radius_lattice[1]).min(ipf[1] + radius_lattice[1])).max(0.);
        let mut z = ((ip0[2] - radius_lattice[2]).min(ipf[2] + radius_lattice[2])).max(0.);

        let xf = ((ipf[0] + radius_lattice[0]).max(ip0[0] - radius_lattice[0])).min(res[0]);
        let yf = ((ipf[1] + radius_lattice[1]).max(ip0[1] - radius_lattice[1])).min(res[1]);
        let zf = ((ipf[2] + radius_lattice[2]).max(ip0[2] - radius_lattice[2])).min(res[2]);

        let mut added_particles: Vec<u32> = Vec::new();
        
        // Now for sure x < xf, y < yf, z < zf
        while x < xf {
            while y < yf {
                while z < zf {
                    // The point should fall inside the region. If not, panic
                    let site = [x as usize, y as usize, z as usize];
                    if regions.get_value_position(site) != initial_region {
                        return Err((format!("Error: Site {:?} is not inside the region.", site), added_particles));
                    }

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
                        let site = [x as usize, y as usize, z as usize];
                        match self.add_particle_site(site, particle) {
                            Ok(_) => {
                                added_particles.push(self.site_to_idx_usize(site));
                                z += 1.;
                                continue
                            },
                            Err(_e) => {
                                // Return an error
                                return Err((format!("Error: Could not add particle to site {:?}.", site), added_particles));
                            }
                        }
                    }
                    z += 1.;
                }
                z = ((ip0[2] - radius_lattice[2]).min(ipf[2] + radius_lattice[2])).max(0.);
                y += 1.;
            }
            y = ((ip0[1] - radius_lattice[1]).min(ipf[1] + radius_lattice[1])).max(0.);
            x += 1.;
        }

        Ok(([ipf[0] as u32, ipf[1] as u32, ipf[2] as u32], added_particles))

    }

    fn add_particle_site(&mut self, site: [usize; 3], particle: Particle) -> Result<String, String> {
        if self.occupancy[[site[0], site[1], site[2]]] == MAX_PARTICLES_SITE as u32 {
            return Err(String::from("Lattice site is full"));
        }
        let lattice_index = (site[0], site[1], site[2], self.occupancy[[site[0], site[1], site[2]]] as usize);
        let concentration_index = (site[0], site[1], site[2], particle as usize);
        self.lattice[lattice_index] = particle;
        self.occupancy[[site[0], site[1], site[2]]] += 1;
        self.concentrations[concentration_index] += 1;
        Ok(String::from("Particle added"))
    }

    fn add_reservoir_site(&mut self, site: [usize; 3], particle: Particle) -> Result<String, String> {
        if self.reservoir[[site[0], site[1], site[2]]] > 0u32 {
            return Err(String::from("Particle already added to site"));
        }
        self.reservoir[[site[0], site[1], site[2]]] = particle;
        self.concentrations[[site[0], site[1], site[2], particle as usize]] = 1;

        Ok(String::from("Particle added to reservoir site"))
    }

    fn remove_particle_site(&mut self, site: [usize; 3], particle: Particle) -> Result<String, String> {
        if self.occupancy[[site[0], site[1], site[2]]] == 0 {
            return Err(String::from("Lattice site is empty"));
        }
        let lattice_index = (site[0], site[1], site[2], self.occupancy[[site[0], site[1], site[2]]] as usize - 1);
        let concentration_index = (site[0], site[1], site[2], particle as usize);
        self.lattice[lattice_index] = 0u32;
        self.occupancy[[site[0], site[1], site[2]]] -= 1;
        self.concentrations[concentration_index] -= 1;
        Ok(String::from("Particle removed"))
    }

    #[allow(dead_code)]
    pub fn find_particle(&self, name: &str) -> Option<usize> {
        for i in 1..self.particle_names.len() as usize {  // Particle 0 is void
            if &self.particle_names[i] == name {
                return Some(i);
            }
        }
        None
    }

    pub fn site_to_idx(&self, site: [u32; 3]) -> u32 {
        site[0] * self.lattice_params.res[1] * self.lattice_params.res[2] + site[1] * self.lattice_params.res[2] + site[2]
    }

    pub fn site_to_idx_usize(&self, site: [usize; 3]) -> u32 {
        let site_u32: [u32; 3] = site.iter().map(|&x| x as u32).collect::<Vec<u32>>().try_into().unwrap();
        self.site_to_idx(site_u32)
    }

    fn idx_to_site(&self, position: u32) -> [u32; 3] {
        let k = position % self.lattice_params.res[2];
        let j = ((position - k) / self.lattice_params.res[2]) % self.lattice_params.res[1];
        let i = ((position - k) / self.lattice_params.res[2] - j) / self.lattice_params.res[1];
        [i, j, k]
    }

    fn idx_to_site_usize(&self, position: u32) -> [usize; 3] {
        let pos_3d = self.idx_to_site(position);
        pos_3d.iter().map(|&x| x as usize).collect::<Vec<usize>>().try_into().unwrap()
    }
}

impl fmt::Display for Lattice {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.lattice.fmt(f)
    }
}