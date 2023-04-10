use core::fmt;
use rand::{Rng, distributions::Uniform};
use rand_distr::Distribution;
use tensor_wgpu::{Tensor4, Tensor3};
use ndarray::prelude::*;

use crate::{MAX_PARTICLES_SITE, types::Region};
use crate::lattice_params::Params;
use crate::types::Particle;


pub struct Lattice {
    pub lattice: Tensor4<Particle>,
    pub occupancy: Tensor3<u32>,
    pub concentrations: Tensor4<i32>,
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

        Lattice {
            lattice,
            occupancy,
            concentrations,
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
    }

    pub fn init_random_particles_region(&mut self, particle: Particle, num_particles: u32, regions: &Tensor3<Region>, region_idx: u32) {
        self.concentrations.enlarge_dimension(3, 0);
        let mut rng = rand::thread_rng();
        for _ in 0..num_particles {
            loop {
                let site = [
                    rng.gen_range(0..self.lattice_params.res[0] as usize),
                    rng.gen_range(0..self.lattice_params.res[1] as usize),
                    rng.gen_range(0..self.lattice_params.res[2] as usize),
                ];
                if regions[site] == region_idx {
                    match self.add_particle_site(site, particle) {
                        Ok(_) => break,
                        Err(_) => continue,
                    }
                }
            }
        }
    }

    pub fn init_random_particles_cube(&mut self, particle: Particle, num_particles: u32, starting_region: &[f32; 3], ending_region: &[f32; 3]) {     
        // The following fills a cube with the particles
        // TODO: I should add an assert to make sure num_particles is smaller than the volume of the region (particles fit in the region)
        self.concentrations.enlarge_dimension(3, 0);
        let res_f32 = self.lattice_params.res.iter().map(|x| *x as f32).collect::<Vec<f32>>();

        let mut rng = rand::thread_rng();
        for _ in 0..num_particles {
            let mut arr = [0f32; 3];
            loop { // Potentially infinite loop if the lattice is full. The previous assert should take care of this
                rng.fill(&mut arr[..]);

                // Convert the random numbers to the region
                for i in 0..3 {
                    arr[i] = starting_region[i] + (ending_region[i] - starting_region[i]) * arr[i];
                }
                let site = [
                    (arr[0] * res_f32[0]) as usize,
                    (arr[1] * res_f32[1]) as usize,
                    (arr[2] * res_f32[2]) as usize,
                ];
                match self.add_particle_site(site, particle) {
                    Ok(_) => break,
                    Err(_) => continue,
                }
            }
        }
    }

    pub fn init_random_particles_sphere(&mut self, particle: Particle, num_particles: u32, center: &[f32; 3], radius: f32) {
        // TODO: This function is extremely disgusting
        self.concentrations.enlarge_dimension(3, 0);
        let res_f32 = self.lattice_params.res.iter().map(|x| *x as f32).collect::<Vec<f32>>();


        let radius_lattice = [
            (radius * res_f32[0]) as i32,
            (radius * res_f32[1]) as i32,
            (radius * res_f32[2]) as i32,
        ];

        let squared_radius = radius_lattice.iter().map(|x| x.pow(2) as f32).collect::<Vec<f32>>();

        let center_lattice = (
            (center[0] * res_f32[0]) as i32,
            (center[1] * res_f32[1]) as i32,
            (center[2] * res_f32[2]) as i32,
        );

        let between = (
            Uniform::from((center_lattice.0 - radius_lattice[0]).max(0)..(center_lattice.0 + radius_lattice[0]).min(self.lattice_params.res[0] as i32)),
            Uniform::from((center_lattice.1 - radius_lattice[1]).max(0)..(center_lattice.1 + radius_lattice[1]).min(self.lattice_params.res[1] as i32)),
            Uniform::from((center_lattice.2 - radius_lattice[2]).max(0)..(center_lattice.2 + radius_lattice[2]).min(self.lattice_params.res[2] as i32)),
        );

        let mut rng = rand::thread_rng();
        for _ in 0..num_particles {
            loop {
                let u: i32 = between.0.sample(&mut rng);
                let v: i32 = between.1.sample(&mut rng);
                let w: i32 = between.2.sample(&mut rng);

                if ((u - center_lattice.0).pow(2) as f32) / squared_radius[0] + ((v - center_lattice.1).pow(2) as f32) / squared_radius[1] + ((w - center_lattice.2).pow(2) as f32) / squared_radius[2] <= 1. {
                    let site = [u as usize, v as usize, w as usize];
                    match self.add_particle_site(site, particle) {
                        Ok(_) => continue,
                        Err(_) => break,
                    }
                }
                break;
            }
        }
    }


    #[allow(dead_code)]
    pub fn init_random_particles_sphere_fast(&mut self, particle: Particle, num_particles: u32, center: &[f32; 3], radius: f32) {
        // Fast and also wrong
        // The following fills a sphere with the particles
        self.concentrations.enlarge_dimension(3, 0);

        let mut rng = rand::thread_rng();
        for _ in 0..num_particles {
            loop { // Add the assert to avoid this infinite loop
                let u: f32 = rng.gen();
                let v: f32 = rng.gen();
                let w: f32 = rng.gen();

                let r = radius * u.powf(1.0/3.0);
                let theta = 2.0 * std::f32::consts::PI * v;
                let phi = (2.0 * w - 1.0).acos();

                let sin_phi = phi.sin();
                let x = r * sin_phi * theta.cos();
                let y = r * sin_phi * theta.sin();
                let z = r * phi.cos();

                let arr = [x + center[0], y + center[1], z + center[2]];
                let site = [
                    (arr[0] * self.lattice_params.res[0] as f32) as usize,
                    (arr[1] * self.lattice_params.res[1] as f32) as usize,
                    (arr[2] * self.lattice_params.res[2] as f32) as usize,
                ];

                match self.add_particle_site(site, particle) {
                    Ok(_) => break,
                    Err(_) => continue,
                }
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