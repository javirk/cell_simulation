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
    lattice: array<u32>,
};

struct Locks {
    locks: array<atomic<u32>>,
}



@group(0) @binding(0) var<uniform> params: LatticeParams;
@group(0) @binding(1) var<uniform> unif: Uniforms;
@group(0) @binding(2) var<storage, read_write> locks: Locks;

@group(1) @binding(0) var<storage> latticeSrc: Lattice;
@group(1) @binding(1) var<storage, read_write> latticeDest: Lattice;
@group(1) @binding(2) var<storage> occupancySrc: array<u32>;
@group(1) @binding(3) var<storage, read_write> occupancyDest: array<u32>;
@group(1) @binding(4) var texture: texture_storage_2d<r32float, read_write>;


fn lock(location: i32) -> bool {
    let lock_ptr = &locks.locks[location];
    let original_lock_value = atomicLoad(lock_ptr);
    if (original_lock_value > 0u) {
        return false;
    }
    return atomicAdd(lock_ptr, 1u) == original_lock_value;
}

fn unlock(location: i32) {
    atomicStore(&locks.locks[location], 0u);
}


fn get_index_lattice(id_volume: vec3<u32>) -> i32 {
    return i32((id_volume.x + id_volume.y * params.x_res + id_volume.z * params.x_res * params.y_res) * params.max_particles_site);
}

fn get_index_occupancy(id_volume: vec3<u32>) -> i32 {
    return i32(id_volume.x + id_volume.y * params.x_res + id_volume.z * params.x_res * params.y_res);
}

fn move_particle_site(particle: u32, volume_src: vec3<u32>, volume_dest: vec3<u32>, idx_particle: u32) {
    // particle: number of particle (species)
    // idx particle: index of the particle in the source lattice
    
    let idx_occupancy_src: i32 = get_index_occupancy(volume_src);
    var idx_occupancy_dest: i32 = get_index_occupancy(volume_dest);
   
    // First, move particle away. Lock the destination cube for that
    while (lock(idx_occupancy_dest)) {
        let idx_destination: i32 = get_index_lattice(volume_dest) + i32(occupancyDest[idx_occupancy_dest]);
        
        // Make sure that the destination lattice is not full
        if (occupancyDest[idx_occupancy_dest] >= params.max_particles_site) { 
            unlock(idx_occupancy_dest);
            return;
        }

        latticeDest.lattice[idx_destination] = particle;
        occupancyDest[idx_occupancy_dest] += 1u;

        latticeDest.lattice[idx_particle] = 0u;
        occupancyDest[idx_occupancy_src] -= 1u;
    }
    unlock(idx_occupancy_dest);

    // Second, remove particle from source lattice (convert to 0)
    while(lock(idx_occupancy_src)) {
        latticeDest.lattice[idx_particle] = 0u;
        occupancyDest[idx_occupancy_src] -= 1u;
    }
    unlock(idx_occupancy_src);
}


fn move_particle(particle: u32, i_movement: i32, id_volume: vec3<u32>, initial_index: u32) -> f32 {
    var occ_dest: u32;
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
        default: {

        }
    }
    move_particle_site(particle, id_volume, volume_dest, initial_index);
    return 1.;
}



@compute @workgroup_size(2, 1, 1)
fn rdme(@builtin(global_invocation_id) global_id: vec3<u32>) {
    textureStore(texture, vec2<i32>(0,0), vec4<f32>(0., 0.0, 0.0, 1.0));
    let X: u32 = global_id.x;
    let Y: u32 = global_id.y;
    let Z: u32 = global_id.z;
    let W: u32 = params.x_res;
    let H: u32 = params.y_res;
    let D: u32 = params.z_res;
    let particles_site: u32 = params.max_particles_site;
    let p: f32 = params.D * params.tau / (params.lambda * params.lambda);

    let idx_lattice = get_index_lattice(global_id);
    let idx_occupancy = get_index_occupancy(global_id);

    let occupancy: u32 = occupancySrc[idx_occupancy];
    var state: u32;
    var rand_number: f32;

    for (var i_part: u32 = idx_lattice; i_part < idx_lattice + occupancy; i_part += 1u) {
        state = Hash_Wang(unif.itime + X + Y + Z + i_part);
        rand_number = UniformFloat(state);
        var i: i32 = 0;
        while ((rand_number < p * f32(i + 1)) && (i < 7)) {
            i += 1;
        }
                        
        let val = move_particle(latticeSrc.lattice[i_part], i, global_id, idx_lattice);
        textureStore(texture, vec2<i32>(0,0), vec4<f32>(val, 0.0, 0.0, 1.0));
    }
}