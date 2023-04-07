let MAX_REACTIONS: u32 = 100u;

// struct LatticeParams {
//     x: f32,
//     y: f32,
//     z: f32,
//     x_res : u32,
//     y_res: u32,
//     z_res: u32,
//     max_particles_site: u32,
//     n_regions: u32,
//     lambda: f32,
//     tau: f32,
// };

struct LatticeParams {
    max_particles_site: u32,
    n_regions: u32,
    lambda: f32,
    tau: f32,
    dims: vec3<f32>,
    res: vec3<u32>,
};

struct ReactionParams {
    num_species: u32,
    num_reactions: u32
}

struct Uniforms {
    itime: u32,
    frame_num: u32,
    slice: u32,
    slice_axis: u32,
}

struct Locks {
	locks: array<atomic<u32>>,
};

fn get_index_lattice(id_volume: vec3<u32>, params: LatticeParams) -> i32 {
    return i32((id_volume.x + id_volume.y * params.res.x + id_volume.z * params.res.x * params.res.y) * params.max_particles_site);
}

fn get_index_occupancy(id_volume: vec3<u32>, params: LatticeParams) -> i32 {
    return i32(id_volume.x + id_volume.y * params.res.x + id_volume.z * params.res.x * params.res.y);
}

