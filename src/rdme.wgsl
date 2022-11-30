struct LatticeParams {
    x: f32,
    y: f32,
    z: f32,
    x_res : u32,
    y_res: u32,
    z_res: u32
};

struct Lattice {
    lattice: array<u8>,
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

    // iter latticeSrc particles
    // Compute probability to stay or move
    // Where to copy? Not trivial -> First 0 and another compute pass to sort?





    let thresh: f32 = 0.001;
    let v0: f32 = 0.04;
    let v1: f32 = 0.3;

    if (X > W || Y > H) {
        return;
    }

    var count: i32 = 0;
    for (var y: i32 = i32(Y - 1u); y <= i32(Y + 1u); y = y + 1) {
        for (var x: i32 = i32(X - 1u); x <= i32(X + 1u); x = x + 1) {
            let yw: u32 = u32(y + i32(H)) % H;
            let xw: u32 = u32(x + i32(W)) % W;
            if (cellSrc.cells[yw*W + xw + H*W*Z] > thresh) {
                count = count + 1;
            } 
        }
    }

    var loadCoord: vec3<i32> = vec3<i32>(
        i32(X),
        i32(Y),
        i32(Z),
    );

    let pix: u32 = Y * W + X + H*W*Z;
    let ov: f32 = cellSrc.cells[pix];
    let oc: f32 = textureLoad(texture, loadCoord).x;
    let was_alive: bool = ov > thresh;
    var nv: f32;
    var nc: f32;


    if (was_alive && (count == 3 || count == 4)) {
        nv = 1.;
        nc = add_color(oc, v1);
    } else {
        if (!was_alive && count == 3) {
            nv = 1.;
            nc = add_color(oc, v1);
        } else {
            nv = 0.;
            nc = susbtract_color(oc, v0);
        }
    }

    cellDst.cells[pix] = nv;

    textureStore(texture, vec3<i32>(i32(X), i32(Y), i32(Z)), vec4<f32>(nc, 0.0, 0.0, 1.0));
}