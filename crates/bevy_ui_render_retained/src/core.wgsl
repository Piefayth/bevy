#import bevy_render::view::View
#import bevy_ui::ui_node::{
    draw_uinode_background,
    draw_uinode_border,
}

const TEXTURED = 1u;
const BORDER_LEFT = 256u;
const BORDER_TOP = 512u;
const BORDER_RIGHT = 1024u;
const BORDER_BOTTOM = 2048u;
const BORDER_ANY = BORDER_LEFT + BORDER_TOP + BORDER_RIGHT + BORDER_BOTTOM;
const CLIPPED = 1u;
const GLYPH = 2u;

@group(0) @binding(0) var<uniform> view: View;
@group(1) @binding(0) var damage_mask: texture_2d<f32>;
@group(2) @binding(0) var sprite_texture: texture_2d<f32>;
@group(2) @binding(1) var sprite_sampler: sampler;

struct VertexOutput {
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) @interpolate(flat) size: vec2<f32>,
    @location(3) @interpolate(flat) flags: u32,
    @location(4) @interpolate(flat) radius: vec4<f32>,
    @location(5) @interpolate(flat) border: vec4<f32>,
    @location(6) point: vec2<f32>,
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vertex(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) transform: vec4<f32>,
    @location(1) translation: vec2<f32>,
    @location(2) clip: vec4<f32>,
    @location(3) uv_rect: vec4<f32>,
    @location(4) color: vec4<f32>,
    @location(5) radius: vec4<f32>,
    @location(6) border: vec4<f32>,
    @location(7) size_inverse_atlas: vec4<f32>,
    @location(8) metadata: vec2<u32>,
) -> VertexOutput {
    let corner_indices = array(0u, 2u, 3u, 0u, 1u, 2u);
    let corner = corner_indices[vertex_index];
    let unit_positions = array(
        vec2(-0.5, -0.5),
        vec2(0.5, -0.5),
        vec2(0.5, 0.5),
        vec2(-0.5, 0.5),
    );
    let local = unit_positions[corner] * size_inverse_atlas.xy;
    let position = vec2(
        transform.x * local.x + transform.z * local.y + translation.x,
        transform.y * local.x + transform.w * local.y + translation.y,
    );

    var position_delta = vec2(0.0);
    if (metadata.y & CLIPPED) != 0u {
        switch corner {
            case 0u: {
                position_delta = max(clip.xy - position, vec2(0.0));
            }
            case 1u: {
                position_delta = vec2(
                    min(clip.z - position.x, 0.0),
                    max(clip.y - position.y, 0.0),
                );
            }
            case 2u: {
                position_delta = min(clip.zw - position, vec2(0.0));
            }
            default: {
                position_delta = vec2(
                    max(clip.x - position.x, 0.0),
                    min(clip.w - position.y, 0.0),
                );
            }
        }
    }

    let corner_flags = array(0u, 2u, 6u, 4u);
    let uv_corners = array(
        uv_rect.xy,
        uv_rect.zy,
        uv_rect.zw,
        uv_rect.xw,
    );
    var out: VertexOutput;
    out.position = view.clip_from_world * vec4(position + position_delta, 0.0, 1.0);
    out.uv = uv_corners[corner] + position_delta * size_inverse_atlas.zw;
    out.color = color;
    out.size = size_inverse_atlas.xy;
    out.flags = metadata.x | corner_flags[corner];
    out.radius = radius;
    out.border = border;
    out.point = select(local + position_delta, vec2(0.0), (metadata.y & GLYPH) != 0u);
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    if textureLoad(damage_mask, vec2<i32>(in.position.xy), 0).r < 0.5 {
        discard;
    }
    let texture_color = textureSample(sprite_texture, sprite_sampler, in.uv);
    let color = select(in.color, in.color * texture_color, (in.flags & TEXTURED) != 0u);
    if (in.flags & BORDER_ANY) != 0u {
        return draw_uinode_border(color, in.point, in.size, in.radius, in.border, in.flags);
    }
    return draw_uinode_background(color, in.point, in.size, in.radius, in.border, in.flags);
}
