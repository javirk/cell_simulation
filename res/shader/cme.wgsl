//!include random.wgsl
//!include functions.wgsl

struct Uniforms {
    itime: u32,
    frame_num: u32,
}

struct Lattice {
    lattice: array<u32>,
};


@group(0) @binding(0) var<uniform> params: LatticeParams;
@group(0) @binding(1) var<uniform> unif: Uniforms;

@group(1) @binding(1) var<storage, read_write> latticeDest: Lattice;
@group(1) @binding(4) var texture: texture_storage_3d<r32float, read_write>;

@group(2) @binding(0) var <storage, read> stoichiometry: array<f32>;
@group(2) @binding(1) var <storage, read> reactions_idx: array<i32>;
@group(2) @binding(2) var <storage, read> reaction_rates: array<u32>;




@compute @workgroup_size(1, 1, 1)
fn cme(@builtin(global_invocation_id) global_id: vec3<u32>) {

}