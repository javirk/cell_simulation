use std::collections::HashMap;

use tensor_wgpu::Tensor3;
use rand::Rng;
use ndarray::{prelude::*, StrideShape};

use crate::{types::Region, debug_println};

#[derive(Debug)]
pub enum RegionType {
    // Cube, Sphere and Cylinder are primitives, the rest are compositions
    Cube { name: String, p0: [f32; 3], pf: [f32; 3] },
    Sphere { name: String, center: [f32; 3], radius: f32 },
    SemiSphere { name: String, center: [f32; 3], radius: f32, direction: [f32; 3] },
    Cylinder { name: String, p0: [f32; 3], pf: [f32; 3], radius: f32 },
    SphericalShell { shell_name: String, interior_name: String, center: [f32; 3], internal_radius: f32, external_radius: f32 },
    CylindricalShell { shell_name: String, interior_name: String, p0: [f32; 3], pf: [f32; 3], internal_radius: f32, external_radius: f32 },
    Capsid { shell_name: String, interior_name: String, center: [f32; 3], dir: [f32; 3], internal_radius: f32, external_radius: f32, total_length: f32 },
    Sparse { name: String, base_region: Sphere },
}

pub struct Cube {
    pub name: String,
    pub p0: [f32; 3],
    pub pf: [f32; 3],
}

#[derive(Debug)]
pub struct Sphere {
    pub name: String,
    pub center: [f32; 3],
    pub radius: f32,
}

#[allow(dead_code)]
pub struct SemiSphere {
    pub name: String,
    pub center: [f32; 3],
    pub radius: f32,
    pub direction: [f32; 3],
}

#[allow(dead_code)]
pub struct Cylinder {
    pub name: String,
    pub p0: [f32; 3],
    pub pf: [f32; 3],
    pub radius: f32,
}

#[allow(dead_code)]
pub struct SphericalShell {
    pub shell_name: String,
    pub interior_name: String,
    pub center: [f32; 3],
    pub internal_radius: f32,
    pub external_radius: f32,
}

#[allow(dead_code)]
pub struct CylindricalShell {
    pub shell_name: String,
    pub interior_name: String,
    pub p0: [f32; 3],
    pub pf: [f32; 3],
    pub internal_radius: f32,
    pub external_radius: f32,
}

#[allow(dead_code)]
pub struct Capsid {
    pub shell_name: String,
    pub interior_name: String,
    pub center: [f32; 3],
    pub dir: [f32; 3],
    pub internal_radius: f32,
    pub external_radius: f32,
    pub total_length: f32,
}

#[allow(dead_code)]
pub struct Sparse {
    pub name: String,
    pub base_region: Sphere,
}

pub struct Regions {
    pub regions: Tensor3<Region>,
    pub types: Vec<RegionType>,
    pub volumes: Vec<u32>,
    pub index_buffer: Option<HashMap<Region, Vec<u32>>>,
}

impl Regions {
    pub fn get_value_position(&self, position: [usize; 3]) -> Region {
        self.regions[[position[0], position[1], position[2]]]
    }

    pub fn set_value_position(&mut self, value: Region, position: [usize; 3]) {  // TODO: Make usize a T
        self.regions[[position[0], position[1], position[2]]] = value;  
    }

    pub fn substract_value_position(&mut self, value: Region, position: [usize; 3]) {  // TODO: Make usize a T
        self.regions[[position[0], position[1], position[2]]] -= value;  
    }

    pub fn cell(&self, position: [usize; 3]) -> Region {
        self.regions[[position[0], position[1], position[2]]]
    }

    pub fn remove_region(&mut self, idx: usize) {
        self.types.remove(idx);
        self.volumes.remove(idx);
    }

    #[allow(dead_code)]
    pub fn get_region(&self, idx: usize) -> &RegionType {
        &self.types[idx]
    }

    pub fn check_regions_collision(&mut self, position: [usize; 3]) {
        // If the site already has a region, substract one to the volume of that region
        let old_region = self.get_value_position(position);
        if old_region != 0 {
            self.volumes[old_region as usize] -= 1;
        }
    }

    pub fn prepare_regions(&mut self) {
        let shape = self.regions.shape();
        let mut index_buffer: HashMap<u32, Vec<u32>> = HashMap::new();
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let value_1d = k + j * shape[2] + i * shape[1] * shape[2];
                    let region = self.regions[[i, j, k]];

                    match index_buffer.get_mut(&region) {
                        Some(region_list) => region_list.push(value_1d as u32),
                        None => {
                            index_buffer.insert(region, vec![value_1d as u32]);
                        }
                    }
                }
            }
        }
        self.index_buffer = Some(index_buffer);
    }

    pub fn generate_boundary_aware(&self, region_idx: usize, voxel_size: [f32; 3], radius_voxels: &Vec<usize>) -> [usize; 3] {
        // Generate a point inside the region. It is the center of a sphere. With this function, we make sure that the whole sphere fits inside the region.
        // Steps:
        // 1. Generate a random point inside the region.
        // 2. Check if the point or the surroundings belong to the region. If not, return to 1.
        // 3. Make sure that the whole base_region fits inside the to_region. If not, return to 1.
        let mut retries = 0u32;
        let mut point: [usize; 3] = [0, 0, 0];
        let mut found = false;
        while retries < 10 && !found {
            point = self.types[region_idx].generate_lattice(voxel_size);
            debug_println!("generate_boundary_aware: Point: {:?}, radius: {:?}\n", point, radius_voxels);
            if self.get_value_position(point) as usize != region_idx {
                retries += 1;
                continue;
            }
    
            // Make sure it fits
            // First side
            if point[0] < radius_voxels[0] || point[1] < radius_voxels[1] || point[2] < radius_voxels[2] {
                retries += 1;
                continue;
            }
            // Second side
            if point[0] + radius_voxels[0] >= self.regions.shape()[0] || point[1] + radius_voxels[1] >= self.regions.shape()[1] || point[2] + radius_voxels[2] >= self.regions.shape()[2] {
                retries += 1;
                continue;
            }
            let data = self.regions.data.slice(
                s![point[0] - radius_voxels[0]..point[0] + radius_voxels[0], 
                point[1] - radius_voxels[1]..point[1] + radius_voxels[1], 
                point[2] - radius_voxels[2]..point[2] + radius_voxels[2]]
            );  // This is a very ugly and hacky way of slicing the tensor, but it's probably the fastest
    
            if data.sum() != (region_idx * data.len()) as u32 {
                // The whole base region doesn't fit inside the to_region
                retries += 1;
                continue;
            }
            found = true;
        }
        if !found {
            panic!("Couldn't find a suitable point for the region");
        }
        point
    }

}

pub trait Random {
    fn generate(&self) -> [f32; 3];
    fn generate_lattice(&self, voxel_size: [f32; 3]) -> [usize; 3];
}

impl Random for RegionType {
    fn generate(&self) -> [f32; 3] {
        use RegionType::*;
        match *&self {
            Cube { name: _, p0, pf } => {
                let mut rng = rand::thread_rng();
                let x = rng.gen_range(p0[0]..pf[0]); // Does this work? They are both &f32
                let y = rng.gen_range(p0[1]..pf[1]);
                let z = rng.gen_range(p0[2]..pf[2]);
                [x, y, z]
            },
            Sphere { name: _, center, radius } => {
                let mut rng = rand::thread_rng();
                let theta = rng.gen_range(0.0..2.0 * std::f32::consts::PI);
                let phi = rng.gen_range(0.0..std::f32::consts::PI);
                let r = rng.gen_range(0.0..*radius);
                let x = r * theta.cos() * phi.sin() + center[0];
                let y = r * theta.sin() * phi.sin() + center[1];
                let z = r * phi.cos() + center[2];
                [x, y, z]
            },
            SemiSphere { name: _, center, radius, direction: _ } => {
                let mut rng = rand::thread_rng();
                let theta = rng.gen_range(0.0..2.0 * std::f32::consts::PI);
                let phi = rng.gen_range(0.0..std::f32::consts::PI);
                let r = rng.gen_range(0.0..*radius);
                let x = r * theta.cos() * phi.sin() + center[0];
                let y = r * theta.sin() * phi.sin() + center[1];
                let z = r * phi.cos() + center[2];
                [x, y, z]
            },
            Cylinder { name: _, p0, pf, radius } => {
                //println!("Cylinder: {:?}", self);
                let mut rng = rand::thread_rng();
                let theta = rng.gen_range(0.0..2.0 * std::f32::consts::PI);
                let r = rng.gen_range(0.0..*radius);
                let x = r * theta.cos() + p0[0];
                let y = r * theta.sin() + p0[1];
                let z = rng.gen_range(p0[2]..pf[2]);
                [x, y, z]
            },
            _ => [0.0, 0.0, 0.0],
        }
    }

    fn generate_lattice(&self, voxel_size: [f32; 3]) -> [usize; 3] {
        let point = self.generate();
        [(point[0] / voxel_size[0]) as usize, (point[1] / voxel_size[1]) as usize, (point[2] / voxel_size[2]) as usize]
    }
}