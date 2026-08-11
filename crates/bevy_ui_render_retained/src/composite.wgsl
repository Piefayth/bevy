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
fn copy_retained(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(retained_ui, vec2<i32>(position.xy), 0);
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
