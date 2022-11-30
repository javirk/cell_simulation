use wgpu::util::DeviceExt;
use std::mem;
use rand::Rng;

use crate::MAX_PARTICLES_SITE;
use crate::lattice_params::Params;
use crate::types::Particle;


pub struct Lattice {
    pub lattice_buff: wgpu::Buffer,
    pub buff_size: usize,
    lattice_params: Params
}

impl Lattice {
    pub fn new(
        params: &Params,
        device: &wgpu::Device,
    ) -> Self {
        let dimensions = params.dimensions();
        let lattice_buff_size = dimensions * MAX_PARTICLES_SITE * mem::size_of::<Particle>();

        let lattice: Vec<Particle> = vec![0 as Particle; dimensions * MAX_PARTICLES_SITE];

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
            lattice_params: *params
        }
    }

    pub fn init_random_particles(&self, num_particles: &Vec<usize>) -> Vec<Particle> { 
        let num_species: Particle = num_particles.len() as Particle;
        let total_particles: usize = num_particles.iter().sum();
        let dimensions = self.lattice_params.dimensions();

        assert!(total_particles <= dimensions * MAX_PARTICLES_SITE);
        
        // The following fills a cube with the particles
        let continuous_lattice: Vec<(Particle, f32, f32, f32)>;
        let mut rng = rand::thread_rng();
        for specie in 0..num_species {
            for particle in 0..num_particles[specie as usize] {
                let mut arr = [0f32; 3];
                rng.fill(&mut arr[..]);

                continuous_lattice.push((specie, arr[0], arr[1], arr[2]))
            }
        }

        // Discretize the lattice
        let lattice: Vec<Particle> = vec![0 as Particle; dimensions * MAX_PARTICLES_SITE];
        for (particle, x, y, z) in continuous_lattice {
            self.add_particle_site(&lattice, x, y, z, particle)
        }
        lattice

    }

    fn add_particle_site(&self, lattice: &Vec<Particle>, x: f32, y: f32, z: f32, particle: Particle) {
        // Find index of site
        // Count particles in this site
        // If full -> throw error
        // Add at the proper index 
        let res = (self.lattice_params.x_res as usize, self.lattice_params.y_res as usize, self.lattice_params.z_res as usize);

        let x_lattice = (x * res.0 as f32) as usize;  // "as usize" already floors the number, so ok.
        let y_lattice = (y * res.1 as f32) as usize;
        let z_lattice = (z * res.2 as f32) as usize;

        // Starting point of the cell
        let index = x_lattice + y_lattice * res.0 + z_lattice * res.0 * res.1;
        for i in index..index + MAX_PARTICLES_SITE {
            if lattice[i] == 0 {
                lattice[i] = particle;
                break;
            }
        }
        panic!("Lattice is full at site {}", index)
    }

    pub fn binding_resource(&self) -> wgpu::BindingResource {
        self.lattice_buff.as_entire_binding()
    }
}