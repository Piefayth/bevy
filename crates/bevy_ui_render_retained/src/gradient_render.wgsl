#import bevy_render::view::View
#import bevy_ui::ui_node::{draw_uinode_background, draw_uinode_border}
#import bevy_ui::gradient::{
    conic_distance,
    interpolate_gradient,
    linear_distance,
    radial_distance,
}

const RADIAL = 16u;
const CONIC = 128u;
const BORDER_LEFT = 256u;
const BORDER_TOP = 512u;
const BORDER_RIGHT = 1024u;
const BORDER_BOTTOM = 2048u;
const BORDER_ANY = BORDER_LEFT + BORDER_TOP + BORDER_RIGHT + BORDER_BOTTOM;

@group(0) @binding(0) var<uniform> view: View;
@group(1) @binding(0) var damage_mask: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) point: vec2<f32>,
    @location(1) @interpolate(flat) size: vec2<f32>,
    @location(2) @interpolate(flat) flags: u32,
    @location(3) @interpolate(flat) radius: vec4<f32>,
    @location(4) @interpolate(flat) border: vec4<f32>,
    @location(5) @interpolate(flat) g_start: vec2<f32>,
    @location(6) @interpolate(flat) dir: vec2<f32>,
    @location(7) @interpolate(flat) start_color: vec4<f32>,
    @location(8) @interpolate(flat) lengths_hint: vec3<f32>,
    @location(9) @interpolate(flat) end_color: vec4<f32>,
}

@vertex
fn vertex(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) transform: vec4<f32>,
    @location(1) translation: vec2<f32>,
    @location(2) size: vec2<f32>,
    @location(3) flags: u32,
    @location(4) radius: vec4<f32>,
    @location(5) border: vec4<f32>,
    @location(6) g_start: vec2<f32>,
    @location(7) dir: vec2<f32>,
    @location(8) start_color: vec4<f32>,
    @location(9) lengths_hint: vec3<f32>,
    @location(10) end_color: vec4<f32>,
) -> VertexOutput {
    let indices = array(0u, 2u, 3u, 0u, 1u, 2u);
    let corners = array(
        vec2(-0.5, -0.5),
        vec2(0.5, -0.5),
        vec2(0.5, 0.5),
        vec2(-0.5, 0.5),
    );
    let local = corners[indices[vertex_index]] * size;
    let position = vec2(
        transform.x * local.x + transform.z * local.y + translation.x,
        transform.y * local.x + transform.w * local.y + translation.y,
    );
    var out: VertexOutput;
    out.position = view.clip_from_world * vec4(position, 0.0, 1.0);
    out.point = local;
    out.size = size;
    out.flags = flags;
    out.radius = radius;
    out.border = border;
    out.g_start = g_start;
    out.dir = dir;
    out.start_color = start_color;
    out.lengths_hint = lengths_hint;
    out.end_color = end_color;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    if textureLoad(damage_mask, vec2<i32>(in.position.xy), 0).r < 0.5 {
        discard;
    }
    var distance: f32;
    if (in.flags & RADIAL) != 0u {
        distance = radial_distance(in.point, in.g_start, in.dir.x);
    } else if (in.flags & CONIC) != 0u {
        distance = conic_distance(in.dir.x, in.point, in.g_start);
    } else {
        distance = linear_distance(in.point, in.g_start, in.dir);
    }
    let color = interpolate_gradient(
        distance,
        in.start_color,
        in.lengths_hint.x,
        in.end_color,
        in.lengths_hint.y,
        in.lengths_hint.z,
        in.flags,
    );
    if (in.flags & BORDER_ANY) != 0u {
        return draw_uinode_border(color, in.point, in.size, in.radius, in.border, in.flags);
    }
    return draw_uinode_background(color, in.point, in.size, in.radius, in.border, in.flags);
}
