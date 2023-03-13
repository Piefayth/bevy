struct VertexOutput {
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @builtin(position) position: vec4<f32>,
};

struct RoundUiMaterial {
    color: vec4<f32>,
};

@group(2) @binding(0)
var<uniform> material: RoundUiMaterial;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let r: f32 = dot(in.uv, in.uv);

    if (r > .95) {
        discard;
    }

    let normalized = (in.uv + vec2<f32>(1., 1.)) / 2.;
    return vec4<f32>(normalized.rg, 0., 1.0);
}

fn circle(uv: vec2<f32>, pos: vec2<f32>, rad: f32, color: vec3<f32>) -> vec4<f32>  {
	let d = length(pos - uv) - rad;
	let t = clamp(d, 0.0, 1.0);
	return vec4<f32>(color, 1.0 - t);
}