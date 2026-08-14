#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#ifdef SRGB_TO_LINEAR
#import bevy_render::color_operations::srgb_to_linear
#endif
#ifdef OKLAB_TO_LINEAR
#import bevy_render::color_operations::oklab_to_linear_rgb
#endif

@group(0) @binding(0) var retained_ui: texture_2d<f32>;

struct RectVertexInput {
    @builtin(vertex_index) vertex_index: u32,
    @location(0) rect: vec4<f32>,
}

@vertex
fn rect_vertex(in: RectVertexInput) -> @builtin(position) vec4<f32> {
    let positions = array(
        vec2(in.rect.x, in.rect.y),
        vec2(in.rect.z, in.rect.y),
        vec2(in.rect.z, in.rect.w),
        vec2(in.rect.x, in.rect.y),
        vec2(in.rect.z, in.rect.w),
        vec2(in.rect.x, in.rect.w),
    );
    return vec4(positions[in.vertex_index], 0.0, 1.0);
}

@fragment
fn wipe() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}

@fragment
fn mask() -> @location(0) f32 {
    return 1.0;
}

@fragment
fn copy_retained(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(retained_ui, vec2<i32>(position.xy), 0);
}

struct BoundaryVertexInput {
    @builtin(vertex_index) vertex_index: u32,
    @location(0) transform: vec4<f32>,
    @location(1) translation: vec2<f32>,
    @location(2) size: vec2<f32>,
    @location(3) uv_rect: vec4<f32>,
    @location(4) opacity: f32,
    @location(5) target_size: vec2<f32>,
}

struct BoundaryVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) opacity: f32,
}

@vertex
fn boundary_vertex(in: BoundaryVertexInput) -> BoundaryVertexOutput {
    let corners = array(0u, 2u, 3u, 0u, 1u, 2u);
    let unit_positions = array(
        vec2(-0.5, -0.5),
        vec2(0.5, -0.5),
        vec2(0.5, 0.5),
        vec2(-0.5, 0.5),
    );
    let uv_positions = array(
        in.uv_rect.xy,
        in.uv_rect.zy,
        in.uv_rect.zw,
        in.uv_rect.xw,
    );
    let corner = corners[in.vertex_index];
    let local = unit_positions[corner] * in.size;
    let world = vec2(
        in.transform.x * local.x + in.transform.z * local.y + in.translation.x,
        in.transform.y * local.x + in.transform.w * local.y + in.translation.y,
    );
    var out: BoundaryVertexOutput;
    out.position = vec4(
        world.x * 2.0 / in.target_size.x - 1.0,
        1.0 - world.y * 2.0 / in.target_size.y,
        0.0,
        1.0,
    );
    out.uv = uv_positions[corner];
    out.opacity = in.opacity;
    return out;
}

fn boundary_color(in: BoundaryVertexOutput) -> vec4<f32> {
    let size = vec2<f32>(textureDimensions(final_world));
    return textureLoad(final_world, vec2<i32>(in.uv * size), 0) * in.opacity;
}

@fragment
fn boundary_fragment(in: BoundaryVertexOutput) -> @location(0) vec4<f32> {
    return boundary_color(in);
}

@group(1) @binding(0) var composition_damage_mask: texture_2d<f32>;

@fragment
fn masked_boundary_fragment(in: BoundaryVertexOutput) -> @location(0) vec4<f32> {
    if textureLoad(composition_damage_mask, vec2<i32>(in.position.xy), 0).r < 0.5 {
        discard;
    }
    return boundary_color(in);
}

@group(0) @binding(1) var final_world: texture_2d<f32>;
@group(0) @binding(2) var final_sampler: sampler;

fn convert_output_color(input: vec4<f32>) -> vec4<f32> {
    var color = input;
#ifdef SRGB_TO_LINEAR
    color = vec4(srgb_to_linear(color.rgb), color.a);
#endif
#ifdef OKLAB_TO_LINEAR
    color = vec4(oklab_to_linear_rgb(color.rgb), color.a);
#endif
    return color;
}

@fragment
fn plain_blit(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    return convert_output_color(textureSample(final_world, final_sampler, in.uv));
}

@fragment
fn final_blit(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let ui_size = vec2<f32>(textureDimensions(retained_ui));
    let ui = textureLoad(retained_ui, vec2<i32>(in.uv * ui_size), 0);
    let world = textureSample(final_world, final_sampler, in.uv);
    return convert_output_color(ui + world * (1.0 - ui.a));
}
