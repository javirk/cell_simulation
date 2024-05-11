use std::io::Error;
pub use lattice_params::LatticeParams;
pub use simulation::Simulation;
pub use setup::Setup;
pub use uniforms::{Uniform, UniformBuffer};
pub use render_params::RenderParams;
pub use texture::Texture;
pub use renderer_3d::Render3D;
pub use renderer_2d::Render2D;
pub use region::{Cube, Sphere, RegionType};

type Result<T> = std::result::Result<T, Error>;

const MAX_PARTICLES_SITE: usize = 8;  // Never larger than 16
const RDME_WORKGROUP_SIZE: (u32, u32, u32) = (1, 1, 1);
const CME_WORKGROUP_SIZE: (u32, u32, u32) = (2, 1, 1);
const MAX_REACTIONS: usize = 100;

mod simulation;
mod lattice;
mod lattice_params;
mod types;
mod rdme;
mod texture;
mod render_params;
mod preprocessor;
mod uniforms;
mod cme;
mod reactions_params;
mod setup;
mod statistics;
mod utils;
mod renderer_3d;
mod renderer_2d;
mod region;
mod macros;