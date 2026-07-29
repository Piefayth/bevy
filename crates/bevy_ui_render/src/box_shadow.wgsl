#import bevy_render::view::View;
#import bevy_render::globals::Globals;
#import bevy_ui::ui_node::{
    select_corner_radius
}

const PI: f32 = 3.14159265358979323846;
const SAMPLES: i32 = #SHADOW_SAMPLES;

@group(0) @binding(0) var<uniform> view: View;

const QUAD_CORNERS = array(
    vec2(-0.5, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5),
    vec2(-0.5, -0.5),
    vec2(0.5, -0.5),
    vec2(0.5, 0.5),
);
const QUAD_CORNER_INDICES = array(0u, 2u, 3u, 0u, 1u, 2u);

#ifdef UI_STORAGE_INSTANCE
struct UiGeometryInstance {
    transform_x: vec2<f32>,
    transform_y: vec2<f32>,
    translation: vec2<f32>,
    size: vec2<f32>,
    position_diff_first: vec4<f32>,
    position_diff_second: vec4<f32>,
    uv_first: vec4<f32>,
    uv_second: vec4<f32>,
};

struct BoxShadowStyleInstance {
    color: vec4<f32>,
    size: vec2<f32>,
    size_padding: vec2<f32>,
    radius_x: vec4<f32>,
    radius_y: vec4<f32>,
    blur: f32,
    blur_padding: u32,
    bounds: vec2<f32>,
};

@group(1) @binding(0) var<storage, read> geometry_instances: array<UiGeometryInstance>;
@group(1) @binding(1) var<storage, read> style_instances: array<BoxShadowStyleInstance>;
#endif

struct BoxShadowVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) point: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) @interpolate(flat) size: vec2<f32>,
    @location(3) @interpolate(flat) radius_x: vec4<f32>,
    @location(4) @interpolate(flat) radius_y: vec4<f32>,
    @location(5) @interpolate(flat) blur: f32,
}

fn gaussian(x: f32, sigma: f32) -> f32 {
    return exp(-(x * x) / (2. * sigma * sigma)) / (sqrt(2. * PI) * sigma);
}

// Approximates the Gauss error function: https://en.wikipedia.org/wiki/Error_function
fn erf(p: vec2<f32>) -> vec2<f32> {
    let s = sign(p);
    let a = abs(p);
    // fourth degree polynomial approximation for erf
    var result = 1.0 + (0.278393 + (0.230389 + 0.078108 * (a * a)) * a) * a;
    result = result * result;
    return s - s / (result * result);
}

fn horizontalRoundedBoxShadow(x: f32, y: f32, blur: f32, radius: vec2<f32>, half_size: vec2<f32>) -> f32 {
    var c = half_size.x;
    if 0.0 < min(radius.x, radius.y) {
        let d = min(half_size.y - radius.y - abs(y), 0.);
        c = half_size.x - radius.x + radius.x * sqrt(max(0., 1. - d * d / (radius.y * radius.y)));
    }
    let integral = 0.5 + 0.5 * erf((x + vec2(-c, c)) * (sqrt(0.5) / blur));
    return integral.y - integral.x;
}

fn roundedBoxShadow(
    lower: vec2<f32>,
    upper: vec2<f32>,
    point: vec2<f32>,
    blur: f32,
    corners_x: vec4<f32>,
    corners_y: vec4<f32>,
) -> f32 {
    let center = (lower + upper) * 0.5;
    let half_size = (upper - lower) * 0.5;
    let p = point - center;
    let low = p.y - half_size.y;
    let high = p.y + half_size.y;
    let start = clamp(-3. * blur, low, high);
    let end = clamp(3. * blur, low, high);
    let step = (end - start) / f32(SAMPLES);
    var y = start + step * 0.5;
    var value: f32 = 0.0;
    for (var i = 0; i < SAMPLES; i++) {
        let corner = select_corner_radius(p, corners_x, corners_y);
        value += horizontalRoundedBoxShadow(p.x, p.y - y, blur, corner, half_size) * gaussian(y, blur) * step;
        y += step;
    }
    return value;
}

fn unpack_corner(first: vec4<f32>, second: vec4<f32>, corner: u32) -> vec2<f32> {
    switch corner {
        case 0u: { return first.xy; }
        case 1u: { return first.zw; }
        case 2u: { return second.xy; }
        default: { return second.zw; }
    }
}

@vertex
fn vertex(
    @builtin(vertex_index) vertex_index: u32,
#ifdef UI_STORAGE_INSTANCE
    @location(0) instance_index: u32,
#else
    @location(0) transform_x: vec2<f32>,
    @location(1) transform_y: vec2<f32>,
    @location(2) translation: vec2<f32>,
    @location(3) geometry_size: vec2<f32>,
    @location(4) position_diff_01: vec4<f32>,
    @location(5) position_diff_23: vec4<f32>,
    @location(6) uv_01: vec4<f32>,
    @location(7) uv_23: vec4<f32>,
    @location(8) vertex_color: vec4<f32>,
    @location(9) size: vec2<f32>,
    @location(10) radius_x: vec4<f32>,
    @location(11) radius_y: vec4<f32>,
    @location(12) blur: f32,
    @location(13) bounds: vec2<f32>,
#endif
) -> BoxShadowVertexOutput {
#ifdef UI_STORAGE_INSTANCE
    let geometry = geometry_instances[instance_index];
    let style = style_instances[instance_index];
    let transform_x = geometry.transform_x;
    let transform_y = geometry.transform_y;
    let translation = geometry.translation;
    let geometry_size = geometry.size;
    let position_diff_01 = geometry.position_diff_first;
    let position_diff_23 = geometry.position_diff_second;
    let uv_01 = geometry.uv_first;
    let uv_23 = geometry.uv_second;
    let vertex_color = style.color;
    let size = style.size;
    let radius_x = style.radius_x;
    let radius_y = style.radius_y;
    let blur = style.blur;
    let bounds = style.bounds;
#endif
    let corner_index = QUAD_CORNER_INDICES[vertex_index];
    let local_position = QUAD_CORNERS[vertex_index] * geometry_size;
    let position_diff = unpack_corner(position_diff_01, position_diff_23, corner_index);
    let world_position =
        transform_x * local_position.x +
        transform_y * local_position.y +
        translation +
        position_diff;
    let uv = unpack_corner(uv_01, uv_23, corner_index);
    var out: BoxShadowVertexOutput;
    out.position = view.clip_from_world * vec4(world_position, 0.0, 1.0);
    out.point = (uv.xy - 0.5) * bounds;
    out.color = vertex_color;
    out.size = size;
    out.radius_x = radius_x;
    out.radius_y = radius_y;
    out.blur = blur;
    return out;
}

@fragment
fn fragment(
    in: BoxShadowVertexOutput,
) -> @location(0) vec4<f32> {
    let g = in.color.a * roundedBoxShadow(-0.5 * in.size, 0.5 * in.size, in.point, max(in.blur, 0.01), in.radius_x, in.radius_y);
    return vec4(in.color.rgb, g);
}
