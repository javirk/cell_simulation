struct LifeParams {
    width : u32,
    height : u32,
};

struct Cells {
    cells : array<f32>,
};

@group(0) @binding(0) var<uniform> params: LifeParams;
@group(0) @binding(1) var<storage> cellSrc: Cells;
@group(0) @binding(2) var<storage, read_write> cellDst: Cells;
@group(0) @binding(3) var texture: texture_storage_3d<r32float, read_write>;

fn add_color(oc: f32, v: f32) -> f32 {
    var nc: f32;
    if (oc + v < 1.) {
        nc = oc + v;
    } else {
        nc = 1.;
    }
    return nc;
}

fn susbtract_color(oc: f32, v: f32) -> f32 {
    var nc: f32;
    if (oc - v > 0.) {
        nc = oc - v;
    } else {
        nc = 0.;
    }
    return nc;
}

@compute @workgroup_size(8, 8, 3)
fn life(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let X: u32 = global_id.x;
    let Y: u32 = global_id.y;
    let Z: u32 = global_id.z;
    let W: u32 = params.width;
    let H: u32 = params.height;
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

