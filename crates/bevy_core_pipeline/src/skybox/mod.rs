use bevy_app::{App, Plugin};
use bevy_asset::{embedded_asset, load_embedded_asset, AssetServer, Handle};
use bevy_camera::Exposure;
use bevy_ecs::{
    change_detection::{DetectChanges, Ref},
    lifecycle::RemovedComponents,
    prelude::{Component, Entity},
    query::{Changed, Has, With},
    resource::Resource,
    schedule::IntoScheduleConfigs,
    system::{Commands, Local, Query, Res, ResMut},
};
use bevy_light::Skybox;
use bevy_log::warn_once;
use bevy_math::Mat4;
use bevy_render::{
    render_asset::RenderAssets,
    render_resource::{
        binding_types::{sampler, texture_cube, uniform_buffer},
        *,
    },
    renderer::{RenderDevice, RenderQueue},
    sync_component::{SyncComponent, SyncComponentPlugin},
    sync_world::RenderEntity,
    texture::GpuImage,
    view::{ExtractedView, Msaa, ViewUniform, ViewUniforms},
    Extract, ExtractSchedule, GpuResourceAppExt, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_shader::Shader;
use bevy_transform::components::Transform;
use bevy_utils::default;

use crate::core_3d::CORE_3D_DEPTH_FORMAT;

pub struct SkyboxPlugin;

impl Plugin for SkyboxPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "skybox.wgsl");

        app.add_plugins(SyncComponentPlugin::<Skybox, Self>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_gpu_resource::<SpecializedRenderPipelines<SkyboxPipeline>>()
            .init_resource::<SkyboxUniformBuffer>()
            .add_systems(ExtractSchedule, extract_skybox)
            .add_systems(RenderStartup, init_skybox_pipeline)
            .add_systems(
                Render,
                (
                    prepare_skybox_pipelines.in_set(RenderSystems::Prepare),
                    prepare_skybox_uniforms.in_set(RenderSystems::PrepareResources),
                    prepare_skybox_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                ),
            );
    }
}

impl SyncComponent<SkyboxPlugin> for Skybox {
    type Target = (Self, SkyboxUniforms, SkyboxPipelineId, SkyboxBindGroup);
}

// This is needed because of the orphan rule not allowing implementing
// foreign trait ExtractComponent on foreign type Skybox
pub fn extract_skybox(
    mut commands: Commands,
    mut previous_len: Local<usize>,
    query: Extract<Query<(RenderEntity, &Skybox, Option<&Exposure>)>>,
    existing: Query<(&Skybox, &SkyboxUniforms)>,
) {
    let mut values = Vec::with_capacity(*previous_len);
    for (entity, skybox, exposure) in &query {
        let exposure = exposure
            .map(Exposure::exposure)
            .unwrap_or_else(|| Exposure::default().exposure());
        let uniforms = SkyboxUniforms {
            brightness: skybox.brightness * exposure,
            transform: Transform::from_rotation(skybox.rotation.inverse()).to_matrix(),
            #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
            _webgl2_padding_8b: 0,
            #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
            _webgl2_padding_12b: 0,
            #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
            _webgl2_padding_16b: 0,
        };
        // VENDORED CHANGE: a static sky re-inserted every frame kept its
        // uniforms perpetually Changed, which made the uniform upload and
        // the bind-group rebuild below run every frame for values that never
        // move. Skip the insert when nothing differs; the downstream systems
        // gate on change detection this makes meaningful.
        if let Ok((old_sky, old_uniforms)) = existing.get(entity)
            && old_sky.image == skybox.image
            && old_uniforms.brightness == uniforms.brightness
            && old_uniforms.transform == uniforms.transform
        {
            continue;
        }
        values.push((entity, (skybox.clone(), uniforms)));
    }
    *previous_len = values.len();
    commands.try_insert_batch(values);
}

/// The skybox uniforms' home, replacing the generic
/// `UniformComponentPlugin` whose prepare rewrote the buffer every frame
/// unconditionally.
#[derive(Resource, Default)]
pub struct SkyboxUniformBuffer(DynamicUniformBuffer<SkyboxUniforms>);

/// A view's offset into [`SkyboxUniformBuffer`] (the vendored stand-in for
/// `DynamicUniformIndex<SkyboxUniforms>`, whose fields are private).
#[derive(Component, Clone, Copy)]
pub struct SkyboxUniformIndex(u32);

/// Rebuild + upload the uniform buffer ONLY when a skybox changed or went
/// away. With the extraction skip above, a static sky costs nothing here.
fn prepare_skybox_uniforms(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut buffer: ResMut<SkyboxUniformBuffer>,
    all: Query<(Entity, &SkyboxUniforms)>,
    changed: Query<(), Changed<SkyboxUniforms>>,
    mut removed: RemovedComponents<SkyboxUniforms>,
) {
    let any_removed = removed.read().next().is_some();
    if changed.is_empty() && !any_removed && buffer.0.buffer().is_some() {
        return;
    }
    let count = all.iter().len();
    let Some(mut writer) = buffer.0.get_writer(count, &render_device, &render_queue) else {
        return;
    };
    let entities = all
        .iter()
        .map(|(entity, component)| (entity, SkyboxUniformIndex(writer.write(component))))
        .collect::<Vec<_>>();
    commands.try_insert_batch(entities);
}

// TODO: Replace with a push constant once WebGPU gets support for that
#[derive(Component, ShaderType, Clone)]
pub struct SkyboxUniforms {
    brightness: f32,
    transform: Mat4,
    #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
    _webgl2_padding_8b: u32,
    #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
    _webgl2_padding_12b: u32,
    #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
    _webgl2_padding_16b: u32,
}

#[derive(Resource)]
struct SkyboxPipeline {
    bind_group_layout: BindGroupLayoutDescriptor,
    shader: Handle<Shader>,
}

impl SkyboxPipeline {
    fn new(shader: Handle<Shader>) -> Self {
        Self {
            bind_group_layout: BindGroupLayoutDescriptor::new(
                "skybox_bind_group_layout",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::FRAGMENT,
                    (
                        texture_cube(TextureSampleType::Float { filterable: true }),
                        sampler(SamplerBindingType::Filtering),
                        uniform_buffer::<ViewUniform>(true)
                            .visibility(ShaderStages::VERTEX_FRAGMENT),
                        uniform_buffer::<SkyboxUniforms>(true),
                    ),
                ),
            ),
            shader,
        }
    }
}

fn init_skybox_pipeline(mut commands: Commands, asset_server: Res<AssetServer>) {
    let shader = load_embedded_asset!(asset_server.as_ref(), "skybox.wgsl");
    commands.insert_resource(SkyboxPipeline::new(shader));
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct SkyboxPipelineKey {
    target_format: TextureFormat,
    samples: u32,
    depth_format: TextureFormat,
}

impl SpecializedRenderPipeline for SkyboxPipeline {
    type Key = SkyboxPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("skybox_pipeline".into()),
            layout: vec![self.bind_group_layout.clone()],
            vertex: VertexState {
                shader: self.shader.clone(),
                ..default()
            },
            depth_stencil: Some(DepthStencilState {
                format: key.depth_format,
                depth_write_enabled: Some(false),
                depth_compare: Some(CompareFunction::GreaterEqual),
                stencil: StencilState {
                    front: StencilFaceState::IGNORE,
                    back: StencilFaceState::IGNORE,
                    read_mask: 0,
                    write_mask: 0,
                },
                bias: DepthBiasState {
                    constant: 0,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: MultisampleState {
                count: key.samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                targets: vec![Some(ColorTargetState {
                    format: key.target_format,
                    // BlendState::REPLACE is not needed here, and None will be potentially much faster in some cases.
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            ..default()
        }
    }
}

#[derive(Component)]
pub struct SkyboxPipelineId(pub CachedRenderPipelineId);

fn prepare_skybox_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<SkyboxPipeline>>,
    pipeline: Res<SkyboxPipeline>,
    cameras: Query<(Entity, &ExtractedView, &Msaa), With<Skybox>>,
) {
    for (entity, view, msaa) in &cameras {
        let pipeline_id = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            SkyboxPipelineKey {
                target_format: view.target_format,
                samples: msaa.samples(),
                depth_format: CORE_3D_DEPTH_FORMAT,
            },
        );

        commands
            .entity(entity)
            .insert(SkyboxPipelineId(pipeline_id));
    }
}

#[derive(Component)]
pub struct SkyboxBindGroup(pub (BindGroup, u32));

fn prepare_skybox_bind_groups(
    mut commands: Commands,
    pipeline: Res<SkyboxPipeline>,
    view_uniforms: Res<ViewUniforms>,
    skybox_uniforms: Res<SkyboxUniformBuffer>,
    images: Res<RenderAssets<GpuImage>>,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    mut cached_buffers: Local<Option<(BufferId, BufferId)>>,
    mut cached_views: Local<bevy_ecs::entity::EntityHashMap<(TextureViewId, SamplerId)>>,
    views: Query<(
        Entity,
        Ref<Skybox>,
        Ref<SkyboxUniformIndex>,
        Has<SkyboxBindGroup>,
    )>,
) {
    // VENDORED CHANGE: the bind group only depends on the two buffers, the
    // image, and the skybox itself — rebuild when one of those moves, not
    // every frame.
    let buffer_ids = (
        view_uniforms.uniforms.buffer().map(|b| b.id()),
        skybox_uniforms.0.buffer().map(|b| b.id()),
    );
    let (Some(view_buffer_id), Some(skybox_buffer_id)) = buffer_ids else {
        return;
    };
    let buffers_moved = *cached_buffers != Some((view_buffer_id, skybox_buffer_id));
    *cached_buffers = Some((view_buffer_id, skybox_buffer_id));

    for (entity, skybox, skybox_uniform_index, has_bind_group) in &views {
        // The bind group bakes in the two buffers, the image's texture view,
        // and the dynamic offset — a change to ANY of those invalidates it.
        // The index is rewritten whenever the uniform buffer rebuilds, which
        // covers another skybox's removal shifting this one's offset without
        // either buffer reallocating.
        // Bevy can recreate the sampler while reusing the texture view, so
        // the sampler's identity is part of the cache key too.
        let cached_view_id = cached_views.get(&entity).copied();
        let current_view_id = skybox
            .image
            .as_ref()
            .and_then(|handle| images.get(handle))
            .map(|image| (image.texture_view.id(), image.sampler.id()));
        if !buffers_moved
            && has_bind_group
            && !skybox.is_changed()
            && !skybox_uniform_index.is_changed()
            && cached_view_id == current_view_id
        {
            continue;
        }
        if let (Some(image_handle), Some(view_uniforms), Some(skybox_uniforms)) = (
            &skybox.image,
            view_uniforms.uniforms.binding(),
            skybox_uniforms.0.binding(),
        ) && let Some(image) = images.get(image_handle)
            && sanity_check_skybox_image_and_warn(entity, &skybox, image)
        {
            let bind_group = render_device.create_bind_group(
                "skybox_bind_group",
                &pipeline_cache.get_bind_group_layout(&pipeline.bind_group_layout),
                &BindGroupEntries::sequential((
                    &image.texture_view,
                    &image.sampler,
                    view_uniforms,
                    skybox_uniforms,
                )),
            );
            cached_views.insert(entity, (image.texture_view.id(), image.sampler.id()));

            commands
                .entity(entity)
                .insert(SkyboxBindGroup((bind_group, skybox_uniform_index.0)));
        } else {
            cached_views.remove(&entity);
            commands.entity(entity).remove::<SkyboxBindGroup>();
        }
    }
}

fn sanity_check_skybox_image_and_warn(entity: Entity, skybox: &Skybox, image: &GpuImage) -> bool {
    let texture_view_dimension: Option<TextureViewDimension> = image
        .texture_view_descriptor
        .as_ref()
        .and_then(|desc| desc.dimension);
    let dimension_ok = texture_view_dimension == Some(TextureViewDimension::Cube);
    if !dimension_ok {
        // The texture view is not a cubemap and will fail validation if rendered.
        // In this case, we ignore the skybox so as not to break rendering.
        //
        // There are other possible misconfigurations which will fail and which we do not
        // catch here, but this is a common mistake (passing an unaltered 2D image to `Skybox`).
        warn_once!(
            "skybox {entity}'s image {image:?} has texture view dimension \
                        {texture_view_dimension:?}, but it must be TextureViewDimension::Cube \
                        to render a skybox",
            image = skybox.image
        );
    }
    dimension_ok
}
