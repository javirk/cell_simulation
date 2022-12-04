use wgpu::util::DeviceExt;
use core::fmt;
use std::mem;
use rand::Rng;

use crate::MAX_PARTICLES_SITE;
use crate::lattice_params::Params;
use crate::types::Particle;

pub struct Lattice {
    pub lattice_buff: wgpu::Buffer,
    pub buff_size: usize,
    lattice_params: Params,
    pub lattice: Vec<Particle>
}

impl Lattice {
    pub fn new(
        params: &Params,
        device: &wgpu::Device,
    ) -> Self {
        let dimensions = params.dimensions();
        let lattice_buff_size = dimensions * (MAX_PARTICLES_SITE + 1) * mem::size_of::<Particle>();

        let lattice: Vec<Particle> = vec![0 as Particle; dimensions * (MAX_PARTICLES_SITE + 1)];  // For the occupancy

        let lattice_buff = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Lattice Buffer"),
            contents: bytemuck::cast_slice(&lattice),
            usage: wgpu::BufferUsages::VERTEX  // What is vertex? Why not uniform?
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
        });

        Lattice { 
            lattice_buff,
            buff_size: lattice_buff_size,
            lattice_params: *params,
            lattice
        }
    }

    pub fn init_random_particles(&mut self, num_particles: &Vec<usize>) { 
        let num_species: Particle = num_particles.len() as Particle;
        let total_particles: usize = num_particles.iter().sum();
        let dimensions = self.lattice_params.dimensions();

        assert!(total_particles <= dimensions * MAX_PARTICLES_SITE);
        
        // The following fills a cube with the particles
        let mut continuous_lattice = Vec::<(Particle, f32, f32, f32)>::new();
        let mut rng = rand::thread_rng();
        for specie in 0..num_species {
            for particle in 0..num_particles[specie as usize] {
                let mut arr = [0f32; 3];
                rng.fill(&mut arr[..]);

                continuous_lattice.push((specie, arr[0], arr[1], arr[2]))
            }
        }

        // Discretize the lattice
        let mut lattice: Vec<Particle> = vec![0 as Particle; dimensions * (MAX_PARTICLES_SITE + 1)];
        for (particle, x, y, z) in continuous_lattice {
            self.add_particle_site(&mut lattice, x, y, z, particle)
        }
        self.lattice = lattice;
    }

    pub fn rewrite_buffer(&mut self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.lattice_buff, 0, &self.lattice);
    }

    pub fn rewrite_buffer_data(&mut self, queue: &wgpu::Queue, data: &Vec<Particle>) {
        queue.write_buffer(&self.lattice_buff, 0, data);
    }

    fn add_particle_site(&self, lattice: &mut Vec<Particle>, x: f32, y: f32, z: f32, particle: Particle) {
        // Find index of site
        // Count particles in this site
        // If full -> throw error
        // Add at the proper index 
        let res = (self.lattice_params.x_res as usize, self.lattice_params.y_res as usize, self.lattice_params.z_res as usize);

        let x_lattice = (x * res.0 as f32) as usize;  // "as usize" already floors the number, so ok.
        let y_lattice = (y * res.1 as f32) as usize;
        let z_lattice = (z * res.2 as f32) as usize;

        // Starting point of the cell
        let (last_element, index) = self.get_last_element_site_coords(lattice, res, x_lattice, y_lattice, z_lattice);
        lattice[last_element] = particle;
        lattice[index + MAX_PARTICLES_SITE] += 1;

    }

    fn get_last_element_site_coords(&self, lattice: &Vec<Particle>, resolution: (usize, usize, usize), x: usize, y: usize, z: usize) -> (usize, usize) {
        let index: usize = (x + y * resolution.0 + z * resolution.0 * resolution.1) * (MAX_PARTICLES_SITE + 1);
        let last_element = index + lattice[index + MAX_PARTICLES_SITE] as usize;
        assert!(lattice[index + MAX_PARTICLES_SITE] < MAX_PARTICLES_SITE as u8, "Lattice is full at site {}", index);
        (last_element, index)
    }

    fn get_last_element_site_idx(&self, lattice: &Vec<Particle>, index: usize) -> usize {
        let last_element = index + lattice[index + MAX_PARTICLES_SITE + 1] as usize;
        assert!(lattice[index + MAX_PARTICLES_SITE + 1] < MAX_PARTICLES_SITE as u8, "Lattice is full at site {}", index);
        last_element
    }

    pub fn binding_resource(&self) -> wgpu::BindingResource {
        self.lattice_buff.as_entire_binding()
    }
}

impl fmt::Display for Lattice {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // TODO: Add 3D structure in some way.
        let mut comma_separated = String::new();
        let mut i = 0;
        for num in &self.lattice[0..self.lattice.len()] {
            if (i + 1) % (MAX_PARTICLES_SITE + 1) != 0 {
                comma_separated.push_str(&num.to_string());
                comma_separated.push_str(", ");
            } else {
                comma_separated.push_str("| ");
                comma_separated.push_str(&num.to_string());
                comma_separated.push_str(" | ");
            }
            i += 1;
        }
        write!(f, "{}", comma_separated)
    }
}