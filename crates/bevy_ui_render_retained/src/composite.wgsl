@group(0) @binding(0) var retained_ui: texture_2d<f32>;

@fragment
fn fragment(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(retained_ui));
    return textureLoad(retained_ui, vec2<i32>(uv * size), 0);
}

@fragment
fn wipe() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}
