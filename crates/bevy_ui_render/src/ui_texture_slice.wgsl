#import bevy_render::view::View;
#import bevy_render::globals::Globals;

@group(0) @binding(0)
var<uniform> view: View;
@group(0) @binding(1)
var<uniform> globals: Globals;

@group(1) @binding(0) var sprite_texture: texture_2d<f32>;
@group(1) @binding(1) var sprite_sampler: sampler;

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

struct UiTextureSliceStyleInstance {
    color: vec4<f32>,
    texture_slices: vec4<f32>,
    target_slices: vec4<f32>,
    repeat: vec4<f32>,
    atlas_rect: vec4<f32>,
};

@group(2) @binding(0) var<storage, read> geometry_instances: array<UiGeometryInstance>;
@group(2) @binding(1) var<storage, read> style_instances: array<UiTextureSliceStyleInstance>;
#endif

struct UiVertexOutput {
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,

    // Defines the dividing line that are used to split the texture atlas rect into corner, side and center slices
    // The distances are normalized and from the top left corner of the texture atlas rect
    // x = distance of the left vertical dividing line
    // y = distance of the top horizontal dividing line
    // z = distance of the right vertical dividing line
    // w = distance of the bottom horizontal dividing line
    @location(2) @interpolate(flat) texture_slices: vec4<f32>,

    // Defines the dividing line that are used to split the render target into corner, side and center slices
    // The distances are normalized and from the top left corner of the render target
    // x = distance of left vertical dividing line
    // y = distance of top horizontal dividing line
    // z = distance of right vertical dividing line
    // w = distance of bottom horizontal dividing line
    @location(3) @interpolate(flat) target_slices: vec4<f32>,

    // The number of times the side or center texture slices should be repeated when mapping them to the border slices
    // x = number of times to repeat along the horizontal axis for the side textures
    // y = number of times to repeat along the vertical axis for the side textures
    // z = number of times to repeat along the horizontal axis for the center texture
    // w = number of times to repeat along the vertical axis for the center texture
    @location(4) @interpolate(flat) repeat: vec4<f32>,

    // normalized texture atlas rect coordinates
    // x, y = top, left corner of the atlas rect
    // z, w = bottom, right corner of the atlas rect
    @location(5) @interpolate(flat) atlas_rect: vec4<f32>,
    @builtin(position) position: vec4<f32>,
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
    @location(3) size: vec2<f32>,
    @location(4) position_diff_01: vec4<f32>,
    @location(5) position_diff_23: vec4<f32>,
    @location(6) uv_01: vec4<f32>,
    @location(7) uv_23: vec4<f32>,
    @location(8) vertex_color: vec4<f32>,
    @location(9) texture_slices: vec4<f32>,
    @location(10) target_slices: vec4<f32>,
    @location(11) repeat: vec4<f32>,
    @location(12) atlas_rect: vec4<f32>,
#endif
) -> UiVertexOutput {
#ifdef UI_STORAGE_INSTANCE
    let geometry = geometry_instances[instance_index];
    let style = style_instances[instance_index];
    let transform_x = geometry.transform_x;
    let transform_y = geometry.transform_y;
    let translation = geometry.translation;
    let size = geometry.size;
    let position_diff_01 = geometry.position_diff_first;
    let position_diff_23 = geometry.position_diff_second;
    let uv_01 = geometry.uv_first;
    let uv_23 = geometry.uv_second;
    let vertex_color = style.color;
    let texture_slices = style.texture_slices;
    let target_slices = style.target_slices;
    let repeat = style.repeat;
    let atlas_rect = style.atlas_rect;
#endif
    let corner_index = QUAD_CORNER_INDICES[vertex_index];
    let local_position = QUAD_CORNERS[vertex_index] * size;
    let position_diff = unpack_corner(position_diff_01, position_diff_23, corner_index);
    let world_position =
        transform_x * local_position.x +
        transform_y * local_position.y +
        translation +
        position_diff;
    let vertex_uv = unpack_corner(uv_01, uv_23, corner_index);
    var out: UiVertexOutput;
    out.uv = vertex_uv;
    out.color = vertex_color;
    out.position = view.clip_from_world * vec4<f32>(world_position, 0.0, 1.0);
    out.texture_slices = texture_slices;
    out.target_slices = target_slices;
    out.repeat = repeat;
    out.atlas_rect = atlas_rect;
    return out;
}

/// maps a point along the axis of the render target to slice coordinates
fn map_axis_with_repeat(
    // normalized distance along the axis
    p: f32,
    // target min dividing point
    il: f32,
    // target max dividing point
    ih: f32,
    // slice min dividing point
    tl: f32,
    // slice max dividing point
    th: f32,
    // number of times to repeat the slice for sides and the center
    r: f32,
) -> f32 {
    if p < il {
        // inside one of the two left (horizontal axis) or top (vertical axis) corners
        return (p / il) * tl;
    } else if ih < p {
        // inside one of the two (horizontal axis) or top (vertical axis) corners
        return th + ((p - ih) / (1 - ih)) * (1 - th);
    } else {
        // not inside a corner, repeat the texture
        return tl + fract((r * (p - il)) / (ih - il)) * (th - tl);
    }
}

fn map_uvs_to_slice(
    uv: vec2<f32>,
    target_slices: vec4<f32>,
    texture_slices: vec4<f32>,
    repeat: vec4<f32>,
) -> vec2<f32> {
    var r: vec2<f32>;
    if target_slices.x <= uv.x && uv.x <= target_slices.z && target_slices.y <= uv.y && uv.y <= target_slices.w {
        // use the center repeat values if the uv coords are inside the center slice of the target
        r = repeat.zw;
    } else {
        // use the side repeat values if the uv coords are outside the center slice
        r = repeat.xy;
    }

    // map horizontal axis
    let x = map_axis_with_repeat(uv.x, target_slices.x, target_slices.z, texture_slices.x, texture_slices.z, r.x);

    // map vertical axis
    let y = map_axis_with_repeat(uv.y, target_slices.y, target_slices.w, texture_slices.y, texture_slices.w, r.y);

    return vec2(x, y);
}

@fragment
fn fragment(in: UiVertexOutput) -> @location(0) vec4<f32> {
    // map the target uvs to slice coords
    let uv = map_uvs_to_slice(in.uv, in.target_slices, in.texture_slices, in.repeat);

    // map the slice coords to texture coords
    let atlas_uv = in.atlas_rect.xy + uv * (in.atlas_rect.zw - in.atlas_rect.xy);

    return in.color * textureSample(sprite_texture, sprite_sampler, atlas_uv);
}
