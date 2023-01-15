use wgpu::util::DeviceExt;
use core::fmt;
use std::mem;
use rand::Rng;
use tensor_wgpu::Tensor;

use crate::MAX_PARTICLES_SITE;
use crate::lattice_params::Params;
use crate::types::Particle;


pub struct Lattice {
    pub lattice_buff: Option<wgpu::Buffer>,
    pub lattice_buff_size: usize,
    pub occupancy_buff: Option<wgpu::Buffer>,
    pub occupancy_buff_size: usize,
    pub concentrations_buff: Option<wgpu::Buffer>,
    pub concentrations_buff_size: usize,
    lattice_params: Params,
    pub lattice: Vec<Particle>,
    pub particle_names: Vec<String>,
    pub occupancy: Vec<u32>,
    pub concentrations: Vec<u32>,
}

impl Lattice {
    pub fn new(
        params: &Params,
    ) -> Self {
        let dimensions = params.dimensions();
        let lattice_buff_size = dimensions * MAX_PARTICLES_SITE * mem::size_of::<Particle>();
        let occupancy_buff_size = dimensions * mem::size_of::<u32>();
        let concentrations_buff_size = dimensions * mem::size_of::<u32>();  // This changes overtime!

        let lattice: Vec<Particle> = vec![0 as Particle; dimensions * MAX_PARTICLES_SITE];
        let particle_names: Vec<String> = vec![String::from("void")];
        let occupancy: Vec<u32> = vec![0 as u32; dimensions];
        let concentrations: Vec<u32> = vec![0 as Particle; dimensions];

        Lattice { 
            lattice_buff: None,
            lattice_buff_size,
            occupancy_buff: None,
            occupancy_buff_size,
            concentrations_buff: None,
            concentrations_buff_size,
            lattice_params: *params,
            lattice,
            particle_names,
            occupancy,
            concentrations
        }
    }

    pub fn start_buffers(&mut self, device: &wgpu::Device) {
        let lattice_buff = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Lattice Buffer"),
            contents: bytemuck::cast_slice(&self.lattice),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let occupancy_buff = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Occupancy Buffer"),
            contents: bytemuck::cast_slice(&self.occupancy),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let concentrations_buff = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Concentrations Buffer"),
            contents: bytemuck::cast_slice(&self.concentrations),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        self.lattice_buff = Some(lattice_buff);
        self.occupancy_buff = Some(occupancy_buff);
        self.concentrations_buff = Some(concentrations_buff);
    }

    pub fn init_random_particles(&mut self, particle: Particle, num_particles: u32, starting_region: &Vec<f32>, ending_region: &Vec<f32>) {       
        // The following fills a cube with the particles
        // TODO: I should add an assert to make sure num_particles is smaller than the volume of the region (particles fit in the region)

        let mut rng = rand::thread_rng();

        //let mut lattice: Vec<Particle> = vec![0 as Particle; dimensions * MAX_PARTICLES_SITE];
        //let mut occupancy: Vec<u32> = vec![0 as u32; dimensions];

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
        // TODO: Update concentrations

        //self.lattice = lattice;
        //self.occupancy = occupancy;
        println!("Occupancy: {:?}", self.occupancy);
    }


    fn add_particle_site(&mut self, x: f32, y: f32, z: f32, particle: Particle) -> Result<String, String> {
        let res = (self.lattice_params.x_res as usize, self.lattice_params.y_res as usize, self.lattice_params.z_res as usize);

        let x_lattice = (x * res.0 as f32) as usize;  // "as usize" already floors the number, so ok.
        let y_lattice = (y * res.1 as f32) as usize;
        let z_lattice = (z * res.2 as f32) as usize;

        // Starting point of the cell
        let (occ_index, lattice_index) = self.get_last_element_site_coords(res, x_lattice, y_lattice, z_lattice);
        if self.occupancy[occ_index] == MAX_PARTICLES_SITE as u32 {
            return Err(String::from("Lattice site is full"));
        }
        self.lattice[lattice_index + self.occupancy[occ_index] as usize] = particle;
        self.occupancy[occ_index] += 1;
        Ok(String::from("Particle added"))
    }

    fn add_particle_concentration(&mut self, x: f32, y: f32, z: f32, particle: Particle) {
        let current_num_particles = self.concentrations.len() / self.lattice_params.dimensions();
        if particle > current_num_particles as u32 {
            // Not ok, I must take the previous number into account
            self.concentrations.resize(self.lattice_params.dimensions() * (particle + 1) as usize, 0);
        }
    }

    fn get_last_element_site_coords(&self, resolution: (usize, usize, usize), x: usize, y: usize, z: usize) -> (usize, usize) {
        let occ_index = x + y * resolution.0 + z * resolution.0 * resolution.1;
        let index = occ_index * MAX_PARTICLES_SITE;
        (occ_index, index)
    }

    pub fn lattice_binding_resource(&self) -> wgpu::BindingResource {
        self.lattice_buff.as_ref().expect("").as_entire_binding()
    }

    pub fn occupancy_binding_resource(&self) -> wgpu::BindingResource {
        self.occupancy_buff.as_ref().expect("").as_entire_binding()
    }

    pub fn concentrations_binding_resource(&self) -> wgpu::BindingResource {
        self.concentrations_buff.as_ref().expect("").as_entire_binding()
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