#import bevy_render::view::View

const PI: f32 = 3.14159265358979323846;
const SAMPLES: i32 = #SHADOW_SAMPLES;

@group(0) @binding(0) var<uniform> view: View;
#ifndef FULL_REBUILD
@group(1) @binding(0) var damage_mask: texture_2d<f32>;
#endif

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) point: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) @interpolate(flat) size: vec2<f32>,
    @location(3) @interpolate(flat) radius: vec4<f32>,
    @location(4) @interpolate(flat) blur: f32,
    @location(5) @interpolate(flat) clip: vec4<f32>,
}

fn gaussian(x: f32, sigma: f32) -> f32 {
    return exp(-(x * x) / (2.0 * sigma * sigma)) / (sqrt(2.0 * PI) * sigma);
}

fn erf(p: vec2<f32>) -> vec2<f32> {
    let s = sign(p);
    let a = abs(p);
    var result = 1.0 + (0.278393 + (0.230389 + 0.078108 * (a * a)) * a) * a;
    result *= result;
    return s - s / (result * result);
}

fn select_corner(p: vec2<f32>, c: vec4<f32>) -> f32 {
    return mix(mix(c.x, c.y, step(0.0, p.x)), mix(c.w, c.z, step(0.0, p.x)), step(0.0, p.y));
}

fn horizontal_shadow(x: f32, y: f32, blur: f32, corner: f32, half_size: vec2<f32>) -> f32 {
    let d = min(half_size.y - corner - abs(y), 0.0);
    let c = half_size.x - corner + sqrt(max(0.0, corner * corner - d * d));
    let integral = 0.5 + 0.5 * erf((x + vec2(-c, c)) * (sqrt(0.5) / blur));
    return integral.y - integral.x;
}

fn rounded_shadow(point: vec2<f32>, blur: f32, corners: vec4<f32>, size: vec2<f32>) -> f32 {
    let half_size = size * 0.5;
    let low = point.y - half_size.y;
    let high = point.y + half_size.y;
    let start = clamp(-3.0 * blur, low, high);
    let end = clamp(3.0 * blur, low, high);
    let sample_step = (end - start) / f32(SAMPLES);
    var y = start + sample_step * 0.5;
    var value = 0.0;
    for (var i = 0; i < SAMPLES; i++) {
        value += horizontal_shadow(
            point.x,
            point.y - y,
            blur,
            select_corner(point, corners),
            half_size,
        ) * gaussian(y, blur) * sample_step;
        y += sample_step;
    }
    return value;
}

@vertex
fn vertex(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) transform: vec4<f32>,
    @location(1) translation: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) size: vec2<f32>,
    @location(4) radius: vec4<f32>,
    @location(5) blur: f32,
    @location(6) bounds: vec2<f32>,
    @location(7) clip: vec4<f32>,
) -> VertexOutput {
    let indices = array(0u, 2u, 3u, 0u, 1u, 2u);
    let corners = array(
        vec2(-0.5, -0.5),
        vec2(0.5, -0.5),
        vec2(0.5, 0.5),
        vec2(-0.5, 0.5),
    );
    let local = corners[indices[vertex_index]] * bounds;
    let position = vec2(
        transform.x * local.x + transform.z * local.y + translation.x,
        transform.y * local.x + transform.w * local.y + translation.y,
    );
    var out: VertexOutput;
    out.position = view.clip_from_world * vec4(position, 0.0, 1.0);
    out.point = local;
    out.color = color;
    out.size = size;
    out.radius = radius;
    out.blur = blur;
    out.clip = clip;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    if any(in.position.xy < in.clip.xy) || any(in.position.xy >= in.clip.zw) {
        discard;
    }
#ifndef FULL_REBUILD
    if textureLoad(damage_mask, vec2<i32>(in.position.xy), 0).r < 0.5 {
        discard;
    }
#endif
    let blur = max(in.blur, 0.01);
    let coverage = rounded_shadow(in.point, blur, in.radius, in.size);
    return vec4(in.color.rgb, in.color.a * coverage);
}
