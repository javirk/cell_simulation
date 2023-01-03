//!include functions.wgsl

@group(0) @binding(0) var<uniform> params: LatticeParams;
@group(1) @binding(1) var<storage, read_write> latticeDest: array<u32>;
@group(1) @binding(3) var<storage, read_write> occupancyDest: array<u32>;
@group(1) @binding(4) var texture: texture_storage_3d<r32float, read_write>;


@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let X: u32 = global_id.x;
    let Y: u32 = global_id.y;
    let Z: u32 = global_id.z;
    let idx_lattice = u32(get_index_lattice(global_id, params));
    let idx_occupancy = get_index_occupancy(global_id, params);
    let occupancy: u32 = occupancyDest[idx_occupancy];

    // Find the majority element in the volume
    // var maxcount: u32 = 0u;
    // var element_max_freq: u32 = 0u;
    // for (var i = 0u; i < occupancy; i += 1u) {
    //     var count = 0u;
    //     for (var j = 0u; j < occupancy; j += 1u) {
    //         if (latticeDest[idx_lattice + i] == latticeDest[idx_lattice + j]) {
    //             count += 1u;
    //         }
    //     }
    //     if (count > maxcount) {
    //         maxcount = count;
    //         element_max_freq = latticeDest[idx_lattice + i];
    //     }
    // }
    // textureStore(texture, vec3<i32>(i32(X), i32(Y), i32(Z)), vec4<f32>(f32(element_max_freq), 0.0, 0.0, 0.0));
    var color: u32 = 0u;
    var debug_occ: u32 = 0u;
    for (var i = 0u; i < params.max_particles_site; i += 1u) {
        if (latticeDest[idx_lattice + i] == 1u) {
            color = 1u;
            debug_occ += 1u;
            // break;
        }
    }
    if debug_occ != occupancy {
        color = 1u;
    } else {
        color = 0u;
    }

    textureStore(texture, vec3<i32>(i32(X), i32(Y), i32(Z)), vec4<f32>(f32(color), 0.0, 0.0, 0.0));

    // Reorder the lattice elements. All 0 at the end. O(N) complexity, N = max_particles_site
    var j = 0u;
    for (var i = 0u; i < params.max_particles_site; i += 1u) {
        if (latticeDest[idx_lattice + i] != 0u) {
            var temp = latticeDest[idx_lattice + i];
            latticeDest[idx_lattice + i] = latticeDest[idx_lattice + j];
            latticeDest[idx_lattice + j] = temp;
            j += 1u;
        }
    }
}