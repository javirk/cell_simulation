//!include random.wgsl

struct InstanceInput {
    @location(2) model_matrix_0: vec4<f32>,
    @location(3) model_matrix_1: vec4<f32>,
    @location(4) model_matrix_2: vec4<f32>,
    @location(5) model_matrix_3: vec4<f32>,
    @location(6) tex_coord: vec3<f32>,
};

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coord: vec3<f32>
};
 
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coord: vec3<f32>,
};

struct LatticeParams {
    width : u32,
    height : u32,
};

struct Uniforms {
    frame_num: u32,
    itime: u32
}

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput
) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );
    var out: VertexOutput;
    out.tex_coord = instance.tex_coord;
    out.position = model_matrix * vec4<f32>(model.position, 1.0);
    out.position[2] = 0.0;
    return out;
}

fn particle_to_color(val: u32) -> vec3<f32> {
    switch (val) {
        case 1u: { return vec3<f32>(219., 95., 87.) / 255.; }
        case 2u: { return vec3<f32>(219., 194., 87.) / 255.; }
        case 6u: { return vec3<f32>(145., 219., 87.) / 255.; }
        case 4u: { return vec3<f32>(87., 219., 128.) / 255.; }
        case 5u: { return vec3<f32>(87., 211., 219.) / 255.; }
        case 3u: { return vec3<f32>(87., 112., 219.) / 255.; }
        case 7u: { return vec3<f32>(219., 87., 178.) / 255.; }
        default: { return vec3<f32>(0., 0., 0.); }
    }
}

@group(0) @binding(0) var texture: texture_storage_3d<r32float, read>;
@group(0) @binding(1) var<uniform> params: LatticeParams;
@group(0) @binding(2) var<uniform> unif: Uniforms;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Load the texture
    var loadCoord: vec3<i32> = vec3<i32>(
        i32(in.tex_coord[0]),
        i32(in.tex_coord[1]),
        i32(in.tex_coord[2]),
    );
    var number = textureLoad(texture, loadCoord).x;
    var color = particle_to_color(u32(number));
    return vec4<f32>(color, 1.0);
    // TODO: Add borders with the region information.
}