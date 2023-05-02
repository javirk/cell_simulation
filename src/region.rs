use tensor_wgpu::Tensor3;
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
}

pub struct Cube {
    pub name: String,
    pub p0: [f32; 3],
    pub pf: [f32; 3],
}

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

pub struct Regions {
    pub regions: Tensor3<Region>,
    pub types: Vec<RegionType>,
}

impl Regions {
    pub fn set_value_position(&mut self, value: usize, position: [usize; 3]) {  // TODO: Make usize a T
        self.regions[[position[0], position[1], position[2]]] = value as Region;  
    }

    pub fn substract_value_position(&mut self, value: usize, position: [usize; 3]) {  // TODO: Make usize a T
        self.regions[[position[0], position[1], position[2]]] -= value as Region;  
    }

    pub fn cell(&self, position: [usize; 3]) -> Region {
        self.regions[[position[0], position[1], position[2]]]
    }
}