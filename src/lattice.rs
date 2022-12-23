use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use core::fmt;
use std::mem;
use rand::Rng;

use crate::MAX_PARTICLES_SITE;
use crate::lattice_params::Params;
use crate::types::Particle;


pub struct Lattice {
    pub lattice_buff: wgpu::Buffer,
    pub lattice_buff_size: usize,
    pub occupancy_buff: wgpu::Buffer,
    pub occupancy_buff_size: usize,
    lattice_params: Params,
    pub lattice: Vec<Particle>,
    pub occupancy: Vec<u32>,
}

impl Lattice {
    pub fn new(
        params: &Params,
        device: &wgpu::Device,
    ) -> Self {
        let dimensions = params.dimensions();
        let lattice_buff_size = dimensions * MAX_PARTICLES_SITE * mem::size_of::<Particle>();
        let occupancy_buff_size = dimensions * mem::size_of::<u32>();
        println!("Lattice buffer size: {} bytes", lattice_buff_size);

        let lattice: Vec<Particle> = vec![0 as Particle; dimensions * MAX_PARTICLES_SITE];
        let occupancy: Vec<u32> = vec![0 as u32; dimensions];

        let lattice_buff = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Lattice Buffer"),
            contents: bytemuck::cast_slice(&lattice),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let occupancy_buff = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Occupancy Buffer"),
            contents: bytemuck::cast_slice(&occupancy),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        Lattice { 
            lattice_buff,
            lattice_buff_size,
            occupancy_buff,
            occupancy_buff_size,
            lattice_params: *params,
            lattice,
            occupancy,
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
                let arr = [0.1; 3];

                continuous_lattice.push((specie, arr[0], arr[1], arr[2]))
            }
        }

        // Discretize the lattice
        let mut lattice: Vec<Particle> = vec![0 as Particle; dimensions * MAX_PARTICLES_SITE];
        let mut occupancy: Vec<u32> = vec![0 as u32; dimensions];
        for (particle, x, y, z) in continuous_lattice {
            self.add_particle_site(&mut lattice, &mut occupancy, x, y, z, particle)
        }
        self.lattice = lattice;
        self.occupancy = occupancy;
    }

    pub fn rewrite_buffers(&mut self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.lattice_buff, 0, bytemuck::cast_slice(&self.lattice));
        queue.write_buffer(&self.occupancy_buff, 0, bytemuck::cast_slice(&self.occupancy));
    }

    pub fn rewrite_buffer_data(&mut self, queue: &wgpu::Queue, lattice_data: &Vec<Particle>, occupancy_data: &Vec<u32>) {
        queue.write_buffer(&self.lattice_buff, 0, bytemuck::cast_slice(lattice_data));
        queue.write_buffer(&self.occupancy_buff, 0, bytemuck::cast_slice(occupancy_data));
    }

    fn add_particle_site(&self, lattice: &mut Vec<Particle>, occupancy: &mut Vec<u32>, x: f32, y: f32, z: f32, particle: Particle) {
        // Find index of site
        // Count particles in this site
        // If full -> throw error
        // Add at the proper index 
        let res = (self.lattice_params.x_res as usize, self.lattice_params.y_res as usize, self.lattice_params.z_res as usize);

        let x_lattice = (x * res.0 as f32) as usize;  // "as usize" already floors the number, so ok.
        let y_lattice = (y * res.1 as f32) as usize;
        let z_lattice = (z * res.2 as f32) as usize;

        // Starting point of the cell
        let (occ_index, lattice_index) = self.get_last_element_site_coords(res, x_lattice, y_lattice, z_lattice);
        lattice[lattice_index + occupancy[occ_index] as usize] = particle;
        occupancy[occ_index] += 1;
    }

    fn get_last_element_site_coords(&self, resolution: (usize, usize, usize), x: usize, y: usize, z: usize) -> (usize, usize) {
        let occ_index = x + y * resolution.0 + z * resolution.0 * resolution.1;
        let index = occ_index * MAX_PARTICLES_SITE;
        (occ_index, index)
    }

    fn get_last_element_site_idx(&self, lattice: &Vec<Particle>, index: usize) -> usize {
        let last_element = index + lattice[index + MAX_PARTICLES_SITE + 1] as usize;
        assert!(lattice[index + MAX_PARTICLES_SITE + 1] < MAX_PARTICLES_SITE as u32, "Lattice is full at site {}", index);
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
            if (i + 1) % MAX_PARTICLES_SITE != 0 {
                comma_separated.push_str(&num.to_string());
                comma_separated.push_str(", ");
            } else {
                comma_separated.push_str(&num.to_string());
                comma_separated.push_str(" | ");
            }
            i += 1;
        }
        write!(f, "{}", comma_separated)
    }
}