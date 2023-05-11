use tensor_wgpu::Tensor3;
use rand::{Rng, distributions::Uniform};

use crate::types::Region;

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

pub struct SemiSphere {
    pub name: String,
    pub center: [f32; 3],
    pub radius: f32,
    pub direction: [f32; 3],
}

pub struct Cylinder {
    pub name: String,
    pub p0: [f32; 3],
    pub pf: [f32; 3],
    pub radius: f32,
}

pub struct SphericalShell {
    pub shell_name: String,
    pub interior_name: String,
    pub center: [f32; 3],
    pub internal_radius: f32,
    pub external_radius: f32,
}

pub struct CylindricalShell {
    pub shell_name: String,
    pub interior_name: String,
    pub p0: [f32; 3],
    pub pf: [f32; 3],
    pub internal_radius: f32,
    pub external_radius: f32,
}

pub struct Capsid {
    pub shell_name: String,
    pub interior_name: String,
    pub center: [f32; 3],
    pub dir: [f32; 3],
    pub internal_radius: f32,
    pub external_radius: f32,
    pub total_length: f32,
}

pub struct Sparse {
    pub name: String,
    pub base_region: Sphere,
}

pub struct Regions {
    pub regions: Tensor3<Region>,
    pub types: Vec<RegionType>,
    pub volumes: Vec<u32>,
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

    pub fn get_region(&self, idx: usize) -> &RegionType {
        &self.types[idx]
    }
}

pub trait Random {
    fn generate(&self) -> [f32; 3];
    fn generate_usize(&self) -> [usize; 3];
}

impl Random for RegionType {
    fn generate(&self) -> [f32; 3] {
        use RegionType::*;
        match *&self {
            Cube { name, p0, pf } => {
                let mut rng = rand::thread_rng();
                let x = rng.gen_range(p0[0]..pf[0]); // Does this work? They are both &f32
                let y = rng.gen_range(p0[1]..pf[1]);
                let z = rng.gen_range(p0[2]..pf[2]);
                [x, y, z]
            },
            Sphere { name, center, radius } => {
                let mut rng = rand::thread_rng();
                let theta = rng.gen_range(0.0..2.0 * std::f32::consts::PI);
                let phi = rng.gen_range(0.0..std::f32::consts::PI);
                let r = rng.gen_range(0.0..*radius);
                let x = r * theta.cos() * phi.sin() + center[0];
                let y = r * theta.sin() * phi.sin() + center[1];
                let z = r * phi.cos() + center[2];
                [x, y, z]
            },
            SemiSphere { name, center, radius, direction } => {
                let mut rng = rand::thread_rng();
                let theta = rng.gen_range(0.0..2.0 * std::f32::consts::PI);
                let phi = rng.gen_range(0.0..std::f32::consts::PI);
                let r = rng.gen_range(0.0..*radius);
                let x = r * theta.cos() * phi.sin() + center[0];
                let y = r * theta.sin() * phi.sin() + center[1];
                let z = r * phi.cos() + center[2];
                [x, y, z]
            },
            Cylinder { name, p0, pf, radius } => {
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

    fn generate_usize(&self) -> [usize; 3] {
        let point = self.generate();
        [point[0] as usize, point[1] as usize, point[2] as usize]
    }
}