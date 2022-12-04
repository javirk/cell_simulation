//!include random.wgsl

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coord: vec3<f32>,
};

struct LatticeParams {
    width : u32,
    height : u32,
};

@vertex
fn vs_main(
    @location(0) position: vec4<f32>,
    @location(1) tex_coord: vec3<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.position = position;
    out.tex_coord = tex_coord;
    return out;
}

@group(0) @binding(0) var texture: texture_storage_2d<r32float, read>;
@group(0) @binding(1) var<uniform> params : LatticeParams;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let state: u32 = Hash_Wang(params.width);
    let color: f32 = UniformFloat(state);
    return vec4<f32>(color, 0.0f, 0.0f, 1.0f);
}