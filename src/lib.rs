pub use lattice_params::LatticeParams;
pub use simulation::Simulation;
pub use render::Render;
pub use setup::Setup;
pub use uniforms::{Uniform, UniformBuffer};
pub use render_params::RenderParams;
pub use texture::Texture;

const MAX_PARTICLES_SITE: usize = 16;
const WORKGROUP_SIZE: (u32, u32, u32) = (1, 1, 1);
const MAX_REACTIONS: usize = 100;

mod simulation;
mod lattice;
mod lattice_params;
mod types;
mod rdme;
mod render;
mod texture;
mod render_params;
mod preprocessor;
mod uniforms;
mod cme;
mod reactions_params;
mod setup;
mod statistics;