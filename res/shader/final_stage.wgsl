//!include functions.wgsl
//!include random.wgsl


@group(0) @binding(0) var<uniform> params: LatticeParams;
@group(0) @binding(2) var<uniform> unif: Uniforms;
@group(0) @binding(3) var<storage> regions: array<u32>;
@group(1) @binding(1) var<storage, read_write> latticeDest: array<u32>;
@group(1) @binding(3) var<storage, read_write> occupancyDest: array<u32>;
@group(1) @binding(4) var texture: texture_storage_3d<r32float, read_write>;

@group(2) @binding(4) var<storage> reservoirs: array<u32>;


@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let X: u32 = global_id.x;
    let Y: u32 = global_id.y;
    let Z: u32 = global_id.z;
    let idx_lattice = u32(get_index_lattice(global_id, params));
    let idx_occupancy = get_index_occupancy(global_id, params);
    let occupancy: u32 = occupancyDest[idx_occupancy];

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

    if (params.res[unif.slice_axis] - global_id[unif.slice_axis]) < unif.slice {
        textureStore(texture, vec3<i32>(i32(X), i32(Y), i32(Z)), vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    switch unif.rendering_view {
        case 0u, default: {
            // MAIN RENDERING VIEW
            // Find the majority element in the volume and save it to the texture for visualization. It depends on the mouse position for slicing.
            var maxcount: u32 = 0u;
            var element_max_freq: u32 = 0u;
            for (var i = 0u; i < occupancy; i += 1u) {
                var count = 0u;
                for (var j = 0u; j < occupancy; j += 1u) {
                    if (latticeDest[idx_lattice + i] == latticeDest[idx_lattice + j]) {
                        count += 1u;
                    }
                }
                if (count > maxcount) {
                    maxcount = count;
                    element_max_freq = latticeDest[idx_lattice + i];
                }
            }
            textureStore(texture, vec3<i32>(i32(X), i32(Y), i32(Z)), vec4<f32>(f32(element_max_freq) / 255., 0.0, 0.0, 0.0));
        }
        case 1u: {
            // Render regions
            textureStore(texture, vec3<i32>(i32(X), i32(Y), i32(Z)), vec4<f32>(f32(regions[idx_occupancy]) / 255., 0.0, 0.0, 0.0));
        }
        case 2u: {
            // Render reservoirs
            textureStore(texture, vec3<i32>(i32(X), i32(Y), i32(Z)), vec4<f32>(f32(reservoirs[idx_occupancy]) / 255., 0.0, 0.0, 0.0));
        }
    }
}