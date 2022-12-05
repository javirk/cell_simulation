//!include random.wgsl

struct LatticeParams {
    x: f32,
    y: f32,
    z: f32,
    x_res : u32,
    y_res: u32,
    z_res: u32
};

struct Uniforms {
    frame_num: u32,
}

struct Lattice {
    lattice: array<atomic<u32>>,
};

@group(0) @binding(0) var<uniform> params: LatticeParams;
@group(0) @binding(1) var<storage> latticeSrc: Lattice;
@group(0) @binding(2) var<storage, write> latticeDst: Lattice;


@compute @workgroup_size(4, 4, 4)
fn rdme(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let X: u32 = global_id.x;
    let Y: u32 = global_id.y;
    let Z: u32 = global_id.z;
    let W: u32 = params.x_res;
    let H: u32 = params.y_res;
    let D: u32 = params.z_res;

    var index: u32 = X + Y * W + Z * H * W;

    // iter latticeSrc particles
    // Compute probability to stay or move
    // Where to copy? Not trivial -> First 0 and another compute pass to sort?

    // var key: u32 = H * X + Y;  // For example
    // var state: u32 = Hash_Wang(key);
    // var random_float: f32 = UniformFloat(state);

}