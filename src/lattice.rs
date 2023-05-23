use core::fmt;
use rand::Rng;
use tensor_wgpu::{Tensor4, Tensor3};
use ndarray::prelude::*;

use crate::MAX_PARTICLES_SITE;
use crate::lattice_params::Params;
use crate::types::Particle;


pub struct Lattice {
    pub lattice: Tensor4<Particle>,
    pub occupancy: Tensor3<u32>,
    pub concentrations: Tensor4<i32>,
    pub reservoir: Tensor3<Particle>,
    lattice_params: Params,
    pub particle_names: Vec<String>,
    pub logging_particles: Vec<Particle>,
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

        Lattice {
            lattice,
            occupancy,
            concentrations,
            reservoir,
            lattice_params: *params,
            particle_names,
            logging_particles: Vec::new(),
        }
    }

    pub fn start_buffers(&mut self, device: &wgpu::Device) {
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        self.lattice.create_buffer(device, usage, Some("Lattice Buffer"));
        self.occupancy.create_buffer(device, usage, Some("Occupancy Buffer"));
        self.concentrations.create_buffer(device, usage, Some("Concentrations Buffer"));
        self.reservoir.create_buffer(device, usage, Some("Reservoir Buffer"));
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
                let k = position % self.lattice_params.res[2];
                let j = ((position - k) / self.lattice_params.res[2]) % self.lattice_params.res[1];
                let i = ((position - k) / self.lattice_params.res[2] - j) / self.lattice_params.res[1];
                let site = [i as usize, j as usize, k as usize];
                // There must be a better way to do this
                if is_reservoir {
                    match self.add_particle_site(site, particle) {
                        Ok(_) => break,
                        Err(_) => {
                            i_ret += 1;
                            continue;
                        },
                    }
                } else {
                    match self.add_reservoir_site(site, particle) {
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
            let k = position % self.lattice_params.res[2];
            let j = ((position - k) / self.lattice_params.res[2]) % self.lattice_params.res[1];
            let i = ((position - k) / self.lattice_params.res[2] - j) / self.lattice_params.res[1];

            let site = [i as usize, j as usize, k as usize];
            match self.add_particle_site(site, particle) {
                Ok(_) => continue,
                Err(_) => panic!("Could not add particle to region"),
            }
        }
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

    #[allow(dead_code)]
    pub fn find_particle(&self, name: &str) -> Option<usize> {
        for i in 1..self.particle_names.len() as usize {  // Particle 0 is void
            if &self.particle_names[i] == name {
                return Some(i);
            }
        }
        None
    }
}

impl fmt::Display for Lattice {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.lattice.fmt(f)
    }
}