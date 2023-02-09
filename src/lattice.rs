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
    lattice_params: Params,
    pub particle_names: Vec<String>,
    pub logging_particles: Vec<Particle>,
}

impl Lattice {
    pub fn new(
        params: &Params,
    ) -> Self {
        let shape_3d = (params.x_res as usize, params.y_res as usize, params.z_res as usize).f();
        let shape_lattice = (params.x_res as usize, params.y_res as usize, params.z_res as usize, MAX_PARTICLES_SITE as usize).f();
        let shape_concentrations = (params.x_res as usize, params.y_res as usize, params.z_res as usize, 1).f();

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

    pub fn init_random_particles(&mut self, particle: Particle, num_particles: u32, starting_region: &Vec<f32>, ending_region: &Vec<f32>) {       
        // The following fills a cube with the particles
        // TODO: I should add an assert to make sure num_particles is smaller than the volume of the region (particles fit in the region)

        self.concentrations.enlarge_dimension(3, 0);

        let mut rng = rand::thread_rng();
        for _ in 0..num_particles {
            let mut arr = [0f32; 3];
            loop { // Potentially infinite loop if the lattice is full. The previous assert should take care of this
                rng.fill(&mut arr[..]);

                // Convert the random numbers to the region
                for i in 0..3 {
                    arr[i] = starting_region[i] + (ending_region[i] - starting_region[i]) * arr[i];
                }
                match self.add_particle_site(arr[0], arr[1], arr[2], particle) {
                    Ok(_) => break,
                    Err(_) => continue,
                }
            }
        }
    }


    fn add_particle_site(&mut self, x: f32, y: f32, z: f32, particle: Particle) -> Result<String, String> {
        let res = (self.lattice_params.x_res as usize, self.lattice_params.y_res as usize, self.lattice_params.z_res as usize);

        let x_lattice = (x * res.0 as f32) as usize;  // "as usize" already floors the number, so ok.
        let y_lattice = (y * res.1 as f32) as usize;
        let z_lattice = (z * res.2 as f32) as usize;

        // Starting point of the cell
        if self.occupancy[[x_lattice, y_lattice, z_lattice]] == MAX_PARTICLES_SITE as u32 {
            return Err(String::from("Lattice site is full"));
        }
        let lattice_index = (x_lattice, y_lattice, z_lattice, self.occupancy[[x_lattice, y_lattice, z_lattice]] as usize);
        let concentration_index = (x_lattice, y_lattice, z_lattice, particle as usize);
        self.lattice[lattice_index] = particle;
        self.occupancy[[x_lattice, y_lattice, z_lattice]] += 1;
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