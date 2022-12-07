//!include random.wgsl

struct LatticeParams {
    x: f32,
    y: f32,
    z: f32,
    x_res : u32,
    y_res: u32,
    z_res: u32,
    max_particles_site: u32,
    lambda: f32,
    D: f32,
    tau: f32
};

struct Uniforms {
    frame_num: u32,
    itime: u32
}

struct Lattice {
    lattice: array<atomic<u32>>,
};



@group(0) @binding(0) var<uniform> params: LatticeParams;
@group(0) @binding(1) var<storage> latticeSrc: Lattice;
@group(0) @binding(2) var<storage, write> latticeDst: Lattice;

fn occupancy_src(idx: u32) -> u32 {
    return latticeSrc[idx + params.max_particles_site];
}

fn occupancy_dest(idx: u32) -> u32 {
    return latticeDest[idx + params.max_particles_site];
}

fn get_index_lattice(id_volume: vec3<u32>) -> u32 {
    return (id_volume.x + id_volume.y * params.x_res + id_volume.z * params.x_res * params.y_res) * (params.max_particles_site + 1);
}

fn move_particle_site(particle: u32, volume_src: vec3<u32>, volume_dest: vec3<u32>) {
    var occ_dest: u32;
    var idx_dest: u32;
    idx_dest = get_index_lattice(volume_dest);
    occ_dest = occupancy_dest(idx_dest);
    if (occ_dest >= params.max_particles_site) { break; }
    // Add particle
    atomicExchange(latticeDest[idx_dest + occ_dest], particle);
    atomicAdd(latticeDest[idx_dest + params.max_particles_site], 1);
    
    // Remove particle: I need the index of the particle in the source volume
    idx_src = get_index_lattice(volume_src);
    occ_src = occupancy_src(idx_src);
    atomicExchange(latticeSrc[idx_])
}


fn move_particle(particle: u32, i_movement: u8, id_volume: vec3<u32>, initial_index: u32) {
    var occ_dest: u32;
    var volume_dest: vec3<u32> = id_volume;
    switch i_movement {
        case 0: {
            // - x
            if (id_volume.x == 0) { break; }
            volume_dest.x -= 1;
            idx_dest = get_index_lattice(volume_dest);
            occ_dest = occupancy_dest(idx_dest);
            if (occ_dest >= max_particles_site) { break; }
            // Add particle
            atomicExchange(latticeDest[idx_dest + occ_dest], particle);
            atomicAdd(latticeDest[idx_dest + params.max_particles_site], 1);
        }
        case 1: {
            // + x
            if (id_volume.x == params.W) { break; }
            volume_dest.x += 1;
            idx_dest = get_index_lattice(volume_dest);
            occ_dest = occupancy_dest(idx_dest);
            if (occ_dest >= max_particles_site) { break; }
            atomicExchange(latticeDest[idx_dest + occ_dest], particle);
            atomicAdd(latticeDest[idx_dest + params.max_particles_site], 1);
        }

    }
}



@compute @workgroup_size(4, 4, 4)
fn rdme(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let X: u32 = global_id.x;
    let Y: u32 = global_id.y;
    let Z: u32 = global_id.z;
    let W: u32 = params.x_res;
    let H: u32 = params.y_res;
    let D: u32 = params.z_res;
    let particles_site: u32 = params.max_particles_site;
    let p: f32 = params.D * params.tau / (params.lambda * params.lambda);

    let initial_index: u32 = (X + Y * W + Z * H * W) * (particles_site + 1);
    let occupancy: u32 = latticeSrc[initial_index + particles_site]

    var state: u32;
    var rand_number: f32;

    for (var i_part: u32 = initial_index; i_part < initial_index + occupancy; i_part += 1) {
        state = Hash_Wang(unif.itime + X + Y + Z + i_part);
        rand_number = UniformFloat(state);
        var i: u8 = 0;
        while (rand_number < p * (i + 1) && i < 7) {
            i += 1;
        }


        if (i != 6) { // == 6: particle stays
            // Particle stays
        } 
    }

    // iter latticeSrc particles
    // Compute probability to stay or move
    // Where to copy? Not trivial -> First 0 and another compute pass to sort?

    // var key: u32 = H * X + Y;  // For example
    // var state: u32 = Hash_Wang(key);
    // var random_float: f32 = UniformFloat(state);

}