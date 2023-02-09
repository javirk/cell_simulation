//!include random.wgsl
//!include functions.wgsl

struct Uniforms {
    itime: u32,
    frame_num: u32,
}

struct Lattice {
    lattice: array<atomic<u32>>,
};


@group(0) @binding(0) var<uniform> params: LatticeParams;
@group(0) @binding(1) var<uniform> reaction_params: ReactionParams;
@group(0) @binding(2) var<uniform> unif: Uniforms;
@group(0) @binding(3) var<storage> regions: array<u32>;
@group(0) @binding(4) var<storage> diffusion_matrix: array<f32>;

@group(1) @binding(0) var<storage> latticeSrc: Lattice;
@group(1) @binding(1) var<storage, read_write> latticeDest: Lattice;
@group(1) @binding(2) var<storage> occupancySrc: array<u32>;
@group(1) @binding(3) var<storage, read_write> occupancyDest: array<atomic<u32>>;

@group(2) @binding(0) var <storage, read_write> concentrations: array<atomic<i32>>;


fn writeLatticeSite(idx_lattice: i32, value: u32) {
    var i = 0;
    let max_particles = i32(params.max_particles_site);
    var success = false;

    loop {
        var resValue = atomicCompareExchangeWeak(&(latticeDest.lattice[idx_lattice + i]), 0u, value);
        success = resValue.exchanged;
        if (success || (i >= max_particles)) {
            break;
        }
        i += 1;
    }
}


fn move_particle_site(particle: u32, volume_src: vec3<u32>, volume_dest: vec3<u32>, idx_particle: u32) {
    // particle: number of particle (species)
    // idx particle: index of the particle in the source lattice
    
    let idx_occupancy_src: i32 = get_index_occupancy(volume_src, params);
    let idx_occupancy_dest: i32 = get_index_occupancy(volume_dest, params);
    let idx_lattice_dest: i32 = get_index_lattice(volume_dest, params);
    let idx_concentration_src: i32 = idx_occupancy_src * i32(reaction_params.num_species + 1u) + i32(particle);
    let idx_concentration_dest: i32 = idx_occupancy_dest * i32(reaction_params.num_species + 1u) + i32(particle);
   
    // First, move particle away. If it doesn't fit, move it back
    let num_particles_dest = atomicAdd(&occupancyDest[idx_occupancy_dest], 1u);
    if num_particles_dest <= params.max_particles_site {
        writeLatticeSite(idx_lattice_dest, particle);
        atomicSub(&occupancyDest[idx_occupancy_src], 1u);
        atomicStore(&latticeDest.lattice[idx_particle], 0u);
        atomicAdd(&concentrations[idx_concentration_dest], 1);
        atomicSub(&concentrations[idx_concentration_src], 1);
    } else {
        // Particle doesn't fit
        atomicSub(&occupancyDest[idx_occupancy_dest], 1u);
    }
}


fn move_particle(particle: u32, i_movement: i32, id_volume: vec3<u32>, initial_index: u32) -> f32 {
    var volume_dest: vec3<u32> = id_volume;
    switch i_movement {
        case 0: {
            // - x
            if (id_volume.x == 0u) { return 0.; }
            volume_dest.x -= 1u;
        }
        case 1: {
            // + x
            if (id_volume.x == (params.x_res - 1u)) { return 0.; }
            volume_dest.x += 1u;
        }
        case 2: {
            // - y
            if (id_volume.y == 0u) { return 0.; }
            volume_dest.y -= 1u;
        }
        case 3: {
            // + y
            if (id_volume.y == (params.y_res - 1u)) { return 0.; }
            volume_dest.y += 1u;
        }
        case 4: {
            // - z
            if (id_volume.z == 0u) { return 0.; }
            volume_dest.z -= 1u;
        }
        case 5: {
            // + z
            if (id_volume.z == (params.z_res - 1u)) { return 0.; }
            volume_dest.z += 1u;
        }
        default: {
            return 1.;
        }
    }
    move_particle_site(particle, id_volume, volume_dest, initial_index);
    return 1.;
}

fn probability_value(src_region: u32, dest_region: u32, particle: u32) -> f32 {
    let val: f32 = diffusion_matrix[src_region + dest_region * params.n_regions + params.n_regions * params.n_regions * particle];
    return val * params.tau / (params.lambda * params.lambda);
}

fn create_probability_vector(volume_id: vec3<u32>, particle: u32, cumulative_probability: ptr<function, array<f32, 7>>) {
    // Fix: probabilities should be equal in all directions
    var probability_vector: array<f32, 7>;
    let src_region = regions[get_index_occupancy(volume_id, params)];
    // -x
    if (volume_id.x > 0u) {
        let dest_region = regions[get_index_occupancy(vec3<u32>(volume_id.x - 1u, volume_id.y, volume_id.z), params)];
        (*cumulative_probability)[0] = probability_value(src_region, dest_region, particle);
    } else {
        (*cumulative_probability)[0] = 0.;
    }
    
    // +x
    if (volume_id.x < (params.x_res - 1u)) {
        let dest_region = regions[get_index_occupancy(vec3<u32>(volume_id.x + 1u, volume_id.y, volume_id.z), params)];
        (*cumulative_probability)[1] = (*cumulative_probability)[0] + probability_value(src_region, dest_region, particle);
    } else {
        (*cumulative_probability)[1] = (*cumulative_probability)[0];
    }

    // -y
    if (volume_id.y > 0u) {
        let dest_region = regions[get_index_occupancy(vec3<u32>(volume_id.x, volume_id.y - 1u, volume_id.z), params)];
        (*cumulative_probability)[2] = (*cumulative_probability)[1] + probability_value(src_region, dest_region, particle);
    } else {
        (*cumulative_probability)[2] = (*cumulative_probability)[1];
    }

    // +y
    if (volume_id.y < (params.y_res - 1u)) {
        let dest_region = regions[get_index_occupancy(vec3<u32>(volume_id.x, volume_id.y + 1u, volume_id.z), params)];
        (*cumulative_probability)[3] = (*cumulative_probability)[2] + probability_value(src_region, dest_region, particle);
    } else {
        (*cumulative_probability)[3] = (*cumulative_probability)[2];
    }

    // -z
    if (volume_id.z > 0u) {
        let dest_region = regions[get_index_occupancy(vec3<u32>(volume_id.x, volume_id.y, volume_id.z - 1u), params)];
        (*cumulative_probability)[4] = (*cumulative_probability)[3] + probability_value(src_region, dest_region, particle);
    } else {
        (*cumulative_probability)[4] = (*cumulative_probability)[3];
    }

    // +z
    if (volume_id.z < (params.z_res - 1u)) {
        let dest_region = regions[get_index_occupancy(vec3<u32>(volume_id.x, volume_id.y, volume_id.z + 1u), params)];
        (*cumulative_probability)[5] = (*cumulative_probability)[4] + probability_value(src_region, dest_region, particle);
    } else {
        (*cumulative_probability)[5] = (*cumulative_probability)[4];
    }
    
    (*cumulative_probability)[6] = 1.;
}


@compute @workgroup_size(1, 1, 1)
fn rdme(@builtin(global_invocation_id) global_id: vec3<u32>) {
    //textureStore(texture, vec3<i32>(0, 0, 0), vec4<f32>(0., 0., 0., 0.));
    let X: u32 = global_id.x;
    let Y: u32 = global_id.y;
    let Z: u32 = global_id.z;
    let W: u32 = params.x_res;
    let H: u32 = params.y_res;
    let D: u32 = params.z_res;
    let particles_site: u32 = params.max_particles_site;

    let idx_lattice = u32(get_index_lattice(global_id, params));
    let idx_occupancy = get_index_occupancy(global_id, params);

    let occupancy: u32 = occupancySrc[idx_occupancy];
    var state: u32;
    var rand_number: f32;

    for (var i_part: u32 = idx_lattice; i_part < idx_lattice + occupancy; i_part += 1u) {
        if latticeSrc.lattice[i_part] == 0u {
            continue;
        }
        // For each particle, we have to find D. It also depends on the neighbouring regions
        var p: array<f32, 7>;
        create_probability_vector(global_id, latticeSrc.lattice[i_part], &p);

        state = Hash_Wang(unif.itime + X + Y + Z + u32(i_part));
        rand_number = UniformFloat(state);

        var i: i32 = 0;
        while (rand_number > p[i]) {
            i += 1;
        }
                        
        let val = move_particle(latticeSrc.lattice[i_part], i, global_id, i_part);
    }
}