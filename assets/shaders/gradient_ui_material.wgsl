struct VertexOutput {
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @builtin(position) position: vec4<f32>,
};

struct GradientUiMaterial {
    color_one: vec4<f32>,
    color_two: vec4<f32>,
};

@group(2) @binding(0)
var<uniform> material: GradientUiMaterial;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    var color = mix(material.color_one, material.color_two, in.uv.x);
    return color;
}