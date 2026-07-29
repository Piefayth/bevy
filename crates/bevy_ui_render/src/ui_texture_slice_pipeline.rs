use core::hash::Hash;

use crate::retained::{
    batch_retained_ui, remove_owner, vertex_storage_supported, RemovedUiNode, RetainedBatchItem,
    RetainedUiBatch, UiCameraPipelineState, UiInstanceArena,
};
use crate::*;
use bevy_asset::*;
use bevy_color::{ColorToComponents, LinearRgba};
use bevy_ecs::{
    entity::EntityHashMap,
    system::{
        lifetimeless::{Read, SRes},
        *,
    },
};
use bevy_image::prelude::*;
use bevy_math::{Affine2, FloatOrd, Rect, Vec2};
use bevy_mesh::VertexBufferLayout;
use bevy_platform::collections::HashMap;
use bevy_render::{
    impl_atomic_pod,
    render_asset::RenderAssets,
    render_phase::*,
    render_resource::{
        binding_types::{storage_buffer_read_only_sized, uniform_buffer},
        *,
    },
    renderer::{RenderDevice, RenderQueue},
    texture::GpuImage,
    view::*,
    Extract, ExtractSchedule, Render, RenderSystems,
};
use bevy_render::{sync_world::MainEntity, GpuResourceAppExt, RenderStartup};
use bevy_shader::Shader;
use bevy_sprite::{SliceScaleMode, SpriteImageMode, TextureSlicer};
use bevy_sprite_render::SpriteAssetEvents;
use bevy_ui::widget::NodeImageMode;
use bevy_ui::{ComputedStackIndex, VisualBox};
use bevy_utils::default;
use binding_types::{sampler, texture_2d};
use bytemuck::{Pod, Zeroable};

pub struct UiTextureSlicerPlugin;

impl Plugin for UiTextureSlicerPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "ui_texture_slice.wgsl");

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_command::<TransparentUi, DrawUiTextureSlices>()
                .init_resource::<ExtractedUiTextureSlices>()
                .init_gpu_resource::<UiTextureSliceMeta>()
                .init_gpu_resource::<UiTextureSliceImageBindGroups>()
                .init_gpu_resource::<SpecializedRenderPipelines<UiTextureSlicePipeline>>()
                .add_systems(RenderStartup, init_ui_texture_slice_pipeline)
                .add_systems(
                    ExtractSchedule,
                    extract_ui_texture_slices.in_set(RenderUiSystems::ExtractTextureSlice),
                )
                .add_systems(
                    Render,
                    (
                        queue_ui_slices.in_set(RenderSystems::Queue),
                        prepare_ui_slices.in_set(RenderSystems::PrepareBindGroups),
                    ),
                );
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Default, Pod, Zeroable)]
struct UiTextureSliceStyleInstance {
    pub color: [f32; 4],
    pub slices: [f32; 4],
    pub border: [f32; 4],
    pub repeat: [f32; 4],
    pub atlas: [f32; 4],
}

impl_atomic_pod!(UiTextureSliceStyleInstance, UiTextureSliceStyleInstanceBlob);

#[derive(Clone, Copy, Debug)]
struct UiTextureSliceBatchKey {
    pipeline: CachedRenderPipelineId,
    image: AssetId<Image>,
}

#[derive(Resource)]
pub struct UiTextureSliceMeta {
    geometry_instances: AtomicSparseBufferVec<UiGeometryInstance>,
    style_instances: AtomicSparseBufferVec<UiTextureSliceStyleInstance>,
    instance_indices: RawBufferVec<u32>,
    view_bind_group: Option<BindGroup>,
    instance_bind_group: Option<BindGroup>,
    instance_buffer_ids: Option<(BufferId, BufferId)>,
    batches: Vec<RetainedUiBatch<UiTextureSliceBatchKey>>,
    arena: UiInstanceArena,
    camera_states: EntityHashMap<UiCameraPipelineState>,
    use_storage_buffers: bool,
}

impl Default for UiTextureSliceMeta {
    fn default() -> Self {
        Self {
            geometry_instances: AtomicSparseBufferVec::new(
                BufferUsages::VERTEX | BufferUsages::STORAGE,
                0,
                "UI texture slice geometry instances".into(),
            ),
            style_instances: AtomicSparseBufferVec::new(
                BufferUsages::VERTEX | BufferUsages::STORAGE,
                0,
                "UI texture slice style instances".into(),
            ),
            instance_indices: RawBufferVec::new(BufferUsages::VERTEX),
            view_bind_group: None,
            instance_bind_group: None,
            instance_buffer_ids: None,
            batches: Vec::new(),
            arena: UiInstanceArena::default(),
            camera_states: EntityHashMap::default(),
            use_storage_buffers: false,
        }
    }
}

#[derive(Resource, Default)]
pub struct UiTextureSliceImageBindGroups {
    pub values: HashMap<AssetId<Image>, BindGroup>,
}

#[derive(Resource)]
pub struct UiTextureSlicePipeline {
    pub view_layout: BindGroupLayoutDescriptor,
    pub image_layout: BindGroupLayoutDescriptor,
    pub instance_layout: BindGroupLayoutDescriptor,
    pub shader: Handle<Shader>,
}

pub fn init_ui_texture_slice_pipeline(mut commands: Commands, asset_server: Res<AssetServer>) {
    let view_layout = BindGroupLayoutDescriptor::new(
        "ui_texture_slice_view_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::VERTEX_FRAGMENT,
            uniform_buffer::<ViewUniform>(true),
        ),
    );

    let image_layout = BindGroupLayoutDescriptor::new(
        "ui_texture_slice_image_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
            ),
        ),
    );
    let instance_layout = BindGroupLayoutDescriptor::new(
        "ui_texture_slice_instance_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX,
            (
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
            ),
        ),
    );

    commands.insert_resource(UiTextureSlicePipeline {
        view_layout,
        image_layout,
        instance_layout,
        shader: load_embedded_asset!(asset_server.as_ref(), "ui_texture_slice.wgsl"),
    });
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct UiTextureSlicePipelineKey {
    pub target_format: TextureFormat,
    pub storage_buffers: bool,
}

impl SpecializedRenderPipeline for UiTextureSlicePipeline {
    type Key = UiTextureSlicePipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let buffers = if key.storage_buffers {
            vec![VertexBufferLayout::from_vertex_formats(
                VertexStepMode::Instance,
                vec![VertexFormat::Uint32],
            )]
        } else {
            vec![
                ui_geometry_vertex_layout(),
                ui_texture_slice_style_vertex_layout(),
            ]
        };
        let mut shader_defs = Vec::new();
        let mut layout = vec![self.view_layout.clone(), self.image_layout.clone()];
        if key.storage_buffers {
            shader_defs.push("UI_STORAGE_INSTANCE".into());
            layout.push(self.instance_layout.clone());
        }

        RenderPipelineDescriptor {
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: shader_defs.clone(),
                buffers,
                ..default()
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs,
                targets: vec![Some(ColorTargetState {
                    format: key.target_format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            layout,
            label: Some("ui_texture_slice_pipeline".into()),
            ..default()
        }
    }
}

fn ui_texture_slice_style_vertex_layout() -> VertexBufferLayout {
    VertexBufferLayout::from_vertex_formats(
        VertexStepMode::Instance,
        vec![
            VertexFormat::Float32x4,
            VertexFormat::Float32x4,
            VertexFormat::Float32x4,
            VertexFormat::Float32x4,
            VertexFormat::Float32x4,
        ],
    )
    .offset_locations_by(8)
}

pub struct ExtractedUiTextureSlice {
    pub stack_index: u32,
    pub transform: Affine2,
    pub rect: Rect,
    pub atlas_rect: Option<Rect>,
    pub image: AssetId<Image>,
    pub clip: Option<Rect>,
    pub color: LinearRgba,
    pub image_scale_mode: SpriteImageMode,
    pub flip_x: bool,
    pub flip_y: bool,
    pub inverse_scale_factor: f32,
}

/// A render-world resource that stores all texture slices in the scene.
#[derive(Resource, Default)]
pub struct ExtractedUiTextureSlices {
    /// The list of texture slices grouped by their main-world entity, along with
    /// each group's target camera entity.
    ///
    /// This is a two-level data structure so that we can quickly remove all
    /// texture slices associated with a main-world entity when it changes.
    pub slices: MainEntityHashMap<(Entity, EntityIndexMap<ExtractedUiTextureSlice>)>,
    pub changed: MainEntityHashSet,
    removed: Vec<RemovedUiNode>,
}

pub fn extract_ui_texture_slices(
    mut commands: Commands,
    mut extracted_ui_slicers: ResMut<ExtractedUiTextureSlices>,
    texture_atlases: Extract<Res<Assets<TextureAtlasLayout>>>,
    slicers_query: Extract<
        Query<
            (
                Entity,
                &ComputedNode,
                &ComputedStackIndex,
                &UiGlobalTransform,
                &InheritedVisibility,
                Option<&CalculatedClip>,
                &ComputedUiTargetCamera,
                &ImageNode,
            ),
            Or<(
                Changed<ComputedNode>,
                Changed<ComputedStackIndex>,
                Changed<UiGlobalTransform>,
                Changed<InheritedVisibility>,
                Changed<CalculatedClip>,
                Changed<ComputedUiTargetCamera>,
                Changed<ImageNode>,
                // The `bevy_ui::widget::update_image_content_size_system` marks
                // `ImageNodeSize` as changed to indicate that the image metrics
                // and/or texture atlas layout changed, so we need to watch for
                // changes to that component, even though we don't read it.
                Changed<ImageNodeSize>,
            )>,
        >,
    >,
    camera_map: Extract<UiCameraMap>,
    (
        mut removed_computed_node_query,
        mut removed_computed_stack_index_query,
        mut removed_ui_global_transform_query,
        mut removed_inherited_visibility_query,
        mut removed_calculated_clip_query,
        mut removed_computed_ui_target_camera_query,
        mut removed_image_node_query,
    ): (
        Extract<RemovedComponents<ComputedNode>>,
        Extract<RemovedComponents<ComputedStackIndex>>,
        Extract<RemovedComponents<UiGlobalTransform>>,
        Extract<RemovedComponents<InheritedVisibility>>,
        Extract<RemovedComponents<CalculatedClip>>,
        Extract<RemovedComponents<ComputedUiTargetCamera>>,
        Extract<RemovedComponents<ImageNode>>,
    ),
    mut nodes_processed_this_frame: Local<MainEntityHashSet>,
) {
    nodes_processed_this_frame.clear();
    extracted_ui_slicers.changed.clear();
    extracted_ui_slicers.removed.clear();
    let mut camera_mapper = camera_map.get_mapper();

    for (entity, uinode, stack_index, transform, inherited_visibility, clip, camera, image) in
        &slicers_query
    {
        let main_entity = MainEntity::from(entity);
        extracted_ui_slicers.changed.insert(main_entity);

        // If there were any previous UI slices for this entity, despawn them.
        let extracted = &mut *extracted_ui_slicers;
        if let Some((_, old_slices)) =
            remove_owner(&mut extracted.slices, main_entity, &mut extracted.removed)
        {
            for render_entity in old_slices.keys() {
                commands.entity(*render_entity).despawn();
            }
        }

        let visual_box = match image.visual_box {
            VisualBox::ContentBox => uinode.content_box(),
            VisualBox::PaddingBox => uinode.padding_box(),
            VisualBox::BorderBox => uinode.border_box(),
        };

        // Skip invisible images
        if !inherited_visibility.get()
            || image.color.is_fully_transparent()
            || image.image.id() == TRANSPARENT_IMAGE_HANDLE.id()
            || visual_box.size().cmple(Vec2::ZERO).any()
        {
            continue;
        }

        let image_scale_mode = match image.image_mode.clone() {
            NodeImageMode::Sliced(texture_slicer) => SpriteImageMode::Sliced(texture_slicer),
            NodeImageMode::Tiled {
                tile_x,
                tile_y,
                stretch_value,
            } => SpriteImageMode::Tiled {
                tile_x,
                tile_y,
                stretch_value,
            },
            _ => continue,
        };

        let Some(extracted_camera_entity) = camera_mapper.map(camera) else {
            continue;
        };
        nodes_processed_this_frame.insert(main_entity);

        let atlas_rect = image
            .texture_atlas
            .as_ref()
            .and_then(|s| s.texture_rect(&texture_atlases))
            .map(|r| r.as_rect());

        let atlas_rect = match (atlas_rect, image.rect) {
            (None, None) => None,
            (None, Some(image_rect)) => Some(image_rect),
            (Some(atlas_rect), None) => Some(atlas_rect),
            (Some(atlas_rect), Some(mut image_rect)) => {
                image_rect.min += atlas_rect.min;
                image_rect.max += atlas_rect.min;
                Some(image_rect)
            }
        };

        extracted_ui_slicers
            .slices
            .entry(main_entity)
            .or_insert_with(|| (extracted_camera_entity, Default::default()))
            .1
            .insert(
                commands.spawn_empty().id(),
                ExtractedUiTextureSlice {
                    stack_index: stack_index.0,
                    transform: Affine2::from(*transform)
                        * Affine2::from_translation(visual_box.center()),
                    color: image.color.into(),
                    rect: Rect {
                        min: Vec2::ZERO,
                        max: visual_box.size(),
                    },
                    clip: clip.map(|clip| clip.clip),
                    image: image.image.id(),
                    image_scale_mode,
                    atlas_rect,
                    flip_x: image.flip_x,
                    flip_y: image.flip_y,
                    inverse_scale_factor: uinode.inverse_scale_factor,
                },
            );
    }

    // Only remove the render-world data if we didn't handle the node above.
    // It's possible that a relevant component was removed and added in the same
    // frame.
    for main_entity in removed_computed_node_query
        .read()
        .chain(removed_computed_stack_index_query.read())
        .chain(removed_ui_global_transform_query.read())
        .chain(removed_inherited_visibility_query.read())
        .chain(removed_calculated_clip_query.read())
        .chain(removed_computed_ui_target_camera_query.read())
        .chain(removed_image_node_query.read())
    {
        let main_entity = MainEntity::from(main_entity);
        if nodes_processed_this_frame.contains(&main_entity) {
            continue;
        }
        extracted_ui_slicers.changed.insert(main_entity);
        let extracted = &mut *extracted_ui_slicers;
        let Some((_, extracted_nodes)) =
            remove_owner(&mut extracted.slices, main_entity, &mut extracted.removed)
        else {
            continue;
        };
        for render_entity in extracted_nodes.keys() {
            commands.entity(*render_entity).despawn();
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "it's a system that needs a lot of them"
)]
pub fn queue_ui_slices(
    extracted_slices: Res<ExtractedUiTextureSlices>,
    pipeline: Res<UiTextureSlicePipeline>,
    mut meta: ResMut<UiTextureSliceMeta>,
    mut pipelines: ResMut<SpecializedRenderPipelines<UiTextureSlicePipeline>>,
    mut phases: ResMut<ViewSortedRenderPhases<TransparentUi>>,
    render_views: Query<(Entity, &UiCameraView), With<ExtractedView>>,
    camera_views: Query<&ExtractedView>,
    pipeline_cache: Res<PipelineCache>,
    draw_functions: Res<DrawFunctions<TransparentUi>>,
    render_device: Res<RenderDevice>,
) {
    let draw_function = draw_functions.read().id::<DrawUiTextureSlices>();
    let storage_buffers = vertex_storage_supported(&render_device, 2);
    let mut active_cameras = HashSet::new();
    let mut invalidated_cameras = HashSet::new();
    for (camera_entity, ui_camera_view) in &render_views {
        let Ok(view) = camera_views.get(ui_camera_view.0) else {
            continue;
        };
        let state = UiCameraPipelineState {
            retained_view_entity: view.retained_view_entity,
            pipeline: pipelines.specialize(
                &pipeline_cache,
                &pipeline,
                UiTextureSlicePipelineKey {
                    target_format: view.target_format,
                    storage_buffers,
                },
            ),
        };
        active_cameras.insert(camera_entity);
        if meta.camera_states.insert(camera_entity, state) != Some(state) {
            invalidated_cameras.insert(camera_entity);
        }
    }
    for removed in &extracted_slices.removed {
        let Some(camera_state) = meta.camera_states.get(&removed.camera_entity) else {
            continue;
        };
        if let Some(phase) = phases.get_mut(&camera_state.retained_view_entity) {
            phase.remove(removed.render_entity, removed.main_entity);
        }
    }
    let mut dirty = extracted_slices.changed.clone();
    if !invalidated_cameras.is_empty() {
        dirty.extend(extracted_slices.slices.iter().filter_map(
            |(main_entity, (camera_entity, _))| {
                invalidated_cameras
                    .contains(camera_entity)
                    .then_some(*main_entity)
            },
        ));
    }
    for main_entity in dirty {
        let Some((camera_entity, slices)) = extracted_slices.slices.get(&main_entity) else {
            continue;
        };
        let Some(camera_state) = meta.camera_states.get(camera_entity) else {
            continue;
        };
        let Some(phase) = phases.get_mut(&camera_state.retained_view_entity) else {
            continue;
        };
        for (render_entity, slice) in slices {
            phase.add_retained(TransparentUi {
                draw_function,
                pipeline: camera_state.pipeline,
                entity: (*render_entity, main_entity),
                sort_key: FloatOrd(slice.stack_index as f32 + stack_z_offsets::BACKGROUND_COLOR),
                batch_range: 0..0,
                extra_index: PhaseItemExtraIndex::None,
                indexed: false,
                batch_index: None,
            });
        }
    }
    meta.camera_states
        .retain(|camera, _| active_cameras.contains(camera));
}

enum SliceGeneration {
    Ready(UiGeometryInstance, UiTextureSliceStyleInstance),
    Culled,
    PendingImage,
}

fn generate_slice_instance(
    slice: &ExtractedUiTextureSlice,
    gpu_images: &RenderAssets<GpuImage>,
) -> SliceGeneration {
    let Some(gpu_image) = gpu_images.get(slice.image) else {
        return SliceGeneration::PendingImage;
    };
    let size = slice.rect.size();
    let (position_diff, culled) = clipping_offsets(slice.transform, Vec2::ZERO, size, slice.clip);
    if culled {
        return SliceGeneration::Culled;
    }
    let uvs = [
        slice.rect.min + position_diff[0],
        Vec2::new(slice.rect.max.x, slice.rect.min.y) + position_diff[1],
        slice.rect.max + position_diff[2],
        Vec2::new(slice.rect.min.x, slice.rect.max.y) + position_diff[3],
    ]
    .map(|position| position / slice.rect.max);
    let gpu_image_size = gpu_image.size_2d().as_vec2();
    let (image_size, mut atlas) = if let Some(atlas) = slice.atlas_rect {
        (
            atlas.size(),
            [
                atlas.min.x / gpu_image_size.x,
                atlas.min.y / gpu_image_size.y,
                atlas.max.x / gpu_image_size.x,
                atlas.max.y / gpu_image_size.y,
            ],
        )
    } else {
        (gpu_image_size, [0.0, 0.0, 1.0, 1.0])
    };
    if slice.flip_x {
        atlas.swap(0, 2);
    }
    if slice.flip_y {
        atlas.swap(1, 3);
    }
    let [slices, border, repeat] = compute_texture_slices(
        image_size,
        size * slice.inverse_scale_factor,
        &slice.image_scale_mode,
    );
    let (position_diff_01, position_diff_23) = pack_corners(position_diff);
    let (uv_01, uv_23) = pack_corners(uvs);
    SliceGeneration::Ready(
        UiGeometryInstance {
            transform_x: slice.transform.x_axis.into(),
            transform_y: slice.transform.y_axis.into(),
            translation: slice.transform.translation.into(),
            size: size.into(),
            position_diff_01,
            position_diff_23,
            uv_01,
            uv_23,
        },
        UiTextureSliceStyleInstance {
            color: slice.color.to_f32_array(),
            slices,
            border,
            repeat,
            atlas,
        },
    )
}

fn rebuild_slice_owner(
    main_entity: MainEntity,
    meta: &mut UiTextureSliceMeta,
    extracted_slices: &ExtractedUiTextureSlices,
    gpu_images: &RenderAssets<GpuImage>,
) {
    meta.arena.free_owner(main_entity);
    let Some((_, slices)) = extracted_slices.slices.get(&main_entity) else {
        return;
    };
    let mut owned = Vec::with_capacity(slices.len());
    let mut pending = false;
    for (render_entity, slice) in slices {
        match generate_slice_instance(slice, gpu_images) {
            SliceGeneration::Ready(geometry, style) => {
                let (start, capacity) = meta.arena.alloc(1);
                meta.geometry_instances.grow(start + capacity);
                meta.style_instances.grow(start + capacity);
                meta.geometry_instances.set(start, geometry);
                meta.style_instances.set(start, style);
                meta.arena.insert(*render_entity, start, 1, capacity);
            }
            SliceGeneration::Culled => meta.arena.insert_empty(*render_entity),
            SliceGeneration::PendingImage => {
                pending = true;
                meta.arena.insert_empty(*render_entity);
            }
        }
        owned.push(*render_entity);
    }
    if pending {
        meta.arena.pending_assets.insert(main_entity);
    }
    if !owned.is_empty() {
        meta.arena.owners.insert(main_entity, owned);
    }
}

pub fn prepare_ui_slices(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    mut meta: ResMut<UiTextureSliceMeta>,
    extracted_slices: Res<ExtractedUiTextureSlices>,
    view_uniforms: Res<ViewUniforms>,
    pipeline: Res<UiTextureSlicePipeline>,
    mut image_bind_groups: ResMut<UiTextureSliceImageBindGroups>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    mut phases: ResMut<ViewSortedRenderPhases<TransparentUi>>,
    events: Res<SpriteAssetEvents>,
    (
        mut sparse_buffer_update_jobs,
        mut sparse_buffer_update_bind_groups,
        sparse_buffer_update_pipelines,
    ): (
        ResMut<SparseBufferUpdateJobs>,
        ResMut<SparseBufferUpdateBindGroups>,
        Res<SparseBufferUpdatePipelines>,
    ),
) {
    let mut changed_images = HashSet::new();
    for event in &events.images {
        match event {
            AssetEvent::Unused { .. } => {}
            AssetEvent::Added { id }
            | AssetEvent::LoadedWithDependencies { id }
            | AssetEvent::Modified { id }
            | AssetEvent::Removed { id } => {
                changed_images.insert(*id);
                image_bind_groups.values.remove(id);
            }
        }
    }
    meta.use_storage_buffers = vertex_storage_supported(&render_device, 2);
    if meta.arena.needs_compaction() || !meta.arena.initialized {
        meta.arena.reset();
        meta.geometry_instances.clear();
        meta.style_instances.clear();
        for main_entity in extracted_slices.slices.keys().copied() {
            rebuild_slice_owner(main_entity, &mut meta, &extracted_slices, &gpu_images);
        }
    } else {
        let mut dirty = extracted_slices.changed.clone();
        dirty.extend(meta.arena.pending_assets.iter().copied());
        if !changed_images.is_empty() {
            dirty.extend(extracted_slices.slices.iter().filter_map(
                |(main_entity, (_, slices))| {
                    slices
                        .values()
                        .any(|slice| changed_images.contains(&slice.image))
                        .then_some(*main_entity)
                },
            ));
        }
        for main_entity in dirty {
            rebuild_slice_owner(main_entity, &mut meta, &extracted_slices, &gpu_images);
        }
    }
    meta.geometry_instances
        .write_buffers(&render_device, &render_queue);
    meta.style_instances
        .write_buffers(&render_device, &render_queue);
    meta.geometry_instances.prepare_to_populate_buffers(
        &render_device,
        &pipeline_cache,
        &mut sparse_buffer_update_jobs,
        &mut sparse_buffer_update_bind_groups,
        &sparse_buffer_update_pipelines,
    );
    meta.style_instances.prepare_to_populate_buffers(
        &render_device,
        &pipeline_cache,
        &mut sparse_buffer_update_jobs,
        &mut sparse_buffer_update_bind_groups,
        &sparse_buffer_update_pipelines,
    );
    let Some(view_binding) = view_uniforms.uniforms.binding() else {
        meta.batches.clear();
        return;
    };
    meta.view_bind_group = Some(render_device.create_bind_group(
        "ui_texture_slice_view_bind_group",
        &pipeline_cache.get_bind_group_layout(&pipeline.view_layout),
        &BindGroupEntries::single(view_binding),
    ));
    if meta.use_storage_buffers {
        if let (Some(geometry_buffer), Some(style_buffer)) = (
            meta.geometry_instances.buffer(),
            meta.style_instances.buffer(),
        ) {
            let ids = (geometry_buffer.id(), style_buffer.id());
            if meta.instance_buffer_ids != Some(ids) {
                meta.instance_bind_group = Some(render_device.create_bind_group(
                    "ui_texture_slice_instance_bind_group",
                    &pipeline_cache.get_bind_group_layout(&pipeline.instance_layout),
                    &BindGroupEntries::sequential((
                        geometry_buffer.as_entire_binding(),
                        style_buffer.as_entire_binding(),
                    )),
                ));
                meta.instance_buffer_ids = Some(ids);
            }
        } else {
            meta.instance_bind_group = None;
            meta.instance_buffer_ids = None;
        }
    }
    let UiTextureSliceMeta {
        arena,
        instance_indices,
        use_storage_buffers,
        ..
    } = &mut *meta;
    meta.batches = batch_retained_ui(
        &mut phases,
        instance_indices,
        *use_storage_buffers,
        |item| {
            let Some(slice) = extracted_slices
                .slices
                .get(&item.main_entity())
                .and_then(|(_, slices)| slices.get(&item.entity()))
            else {
                return RetainedBatchItem::NotOwned;
            };
            let Some(slot) = arena.slots.get(&item.entity()) else {
                return RetainedBatchItem::Culled;
            };
            if slot.instances.count == 0 || gpu_images.get(slice.image).is_none() {
                RetainedBatchItem::Culled
            } else {
                RetainedBatchItem::Drawable {
                    instances: slot.instances,
                    key: UiTextureSliceBatchKey {
                        pipeline: item.pipeline,
                        image: slice.image,
                    },
                }
            }
        },
        |left, right| left.pipeline == right.pipeline && left.image == right.image,
        |_, _| {},
    );
    meta.instance_indices
        .write_buffer(&render_device, &render_queue);
    for batch in &meta.batches {
        let image = gpu_images
            .get(batch.key.image)
            .expect("texture slice image was validated while batching");
        image_bind_groups
            .values
            .entry(batch.key.image)
            .or_insert_with(|| {
                render_device.create_bind_group(
                    "ui_texture_slice_image_layout",
                    &pipeline_cache.get_bind_group_layout(&pipeline.image_layout),
                    &BindGroupEntries::sequential((&image.texture_view, &image.sampler)),
                )
            });
    }
}

pub type DrawUiTextureSlices = (
    SetItemPipeline,
    SetSlicerViewBindGroup<0>,
    SetSlicerTextureBindGroup<1>,
    SetSlicerInstanceBindGroup<2>,
    DrawSlicer,
);

pub struct SetSlicerInstanceBindGroup<const I: usize>;
impl<const I: usize> RenderCommand<TransparentUi> for SetSlicerInstanceBindGroup<I> {
    type Param = SRes<UiTextureSliceMeta>;
    type ViewQuery = ();
    type ItemQuery = ();

    fn render<'w>(
        _item: &TransparentUi,
        _view: (),
        _entity: Option<()>,
        meta: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let meta = meta.into_inner();
        if meta.use_storage_buffers {
            let Some(bind_group) = meta.instance_bind_group.as_ref() else {
                return RenderCommandResult::Failure("missing texture slice instance bind group");
            };
            pass.set_bind_group(I, bind_group, &[]);
        }
        RenderCommandResult::Success
    }
}

pub struct SetSlicerViewBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetSlicerViewBindGroup<I> {
    type Param = SRes<UiTextureSliceMeta>;
    type ViewQuery = Read<ViewUniformOffset>;
    type ItemQuery = ();

    fn render<'w>(
        _item: &P,
        view_uniform: &'w ViewUniformOffset,
        _entity: Option<()>,
        ui_meta: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(view_bind_group) = ui_meta.into_inner().view_bind_group.as_ref() else {
            return RenderCommandResult::Failure("view_bind_group not available");
        };
        pass.set_bind_group(I, view_bind_group, &[view_uniform.offset]);
        RenderCommandResult::Success
    }
}
pub struct SetSlicerTextureBindGroup<const I: usize>;
impl<const I: usize> RenderCommand<TransparentUi> for SetSlicerTextureBindGroup<I> {
    type Param = (
        SRes<UiTextureSliceImageBindGroups>,
        SRes<UiTextureSliceMeta>,
    );
    type ViewQuery = ();
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        item: &TransparentUi,
        _view: (),
        _entity: Option<()>,
        (image_bind_groups, meta): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(batch_index) = item.batch_index else {
            return RenderCommandResult::Skip;
        };
        let Some(batch) = meta.into_inner().batches.get(batch_index as usize) else {
            return RenderCommandResult::Failure("texture slice batch index out of range");
        };
        let Some(bind_group) = image_bind_groups.into_inner().values.get(&batch.key.image) else {
            return RenderCommandResult::Failure("missing texture slice image bind group");
        };
        pass.set_bind_group(I, bind_group, &[]);
        RenderCommandResult::Success
    }
}
pub struct DrawSlicer;
impl RenderCommand<TransparentUi> for DrawSlicer {
    type Param = SRes<UiTextureSliceMeta>;
    type ViewQuery = ();
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        item: &TransparentUi,
        _view: (),
        _entity: Option<()>,
        ui_meta: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(batch_index) = item.batch_index else {
            return RenderCommandResult::Skip;
        };
        let meta = ui_meta.into_inner();
        let Some(batch) = meta.batches.get(batch_index as usize) else {
            return RenderCommandResult::Failure("texture slice batch index out of range");
        };
        if meta.use_storage_buffers {
            let Some(indices) = meta.instance_indices.buffer() else {
                return RenderCommandResult::Failure("missing texture slice instance indices");
            };
            pass.set_vertex_buffer(0, indices.slice(..));
        } else {
            let Some(geometry) = meta.geometry_instances.buffer() else {
                return RenderCommandResult::Failure("missing texture slice geometry instances");
            };
            let Some(style) = meta.style_instances.buffer() else {
                return RenderCommandResult::Failure("missing texture slice style instances");
            };
            pass.set_vertex_buffer(0, geometry.slice(..));
            pass.set_vertex_buffer(1, style.slice(..));
        }
        pass.draw(0..6, batch.range.clone());
        RenderCommandResult::Success
    }
}

fn compute_texture_slices(
    image_size: Vec2,
    target_size: Vec2,
    image_scale_mode: &SpriteImageMode,
) -> [[f32; 4]; 3] {
    match image_scale_mode {
        SpriteImageMode::Sliced(TextureSlicer {
            border: border_rect,
            center_scale_mode,
            sides_scale_mode,
            max_corner_scale,
        }) => {
            let min_coeff = (target_size / image_size)
                .min_element()
                .min(*max_corner_scale);

            // calculate the normalized extents of the nine-patched image slices
            let slices = [
                border_rect.min_inset.x / image_size.x,
                border_rect.min_inset.y / image_size.y,
                1. - border_rect.max_inset.x / image_size.x,
                1. - border_rect.max_inset.y / image_size.y,
            ];

            // calculate the normalized extents of the target slices
            let border = [
                (border_rect.min_inset.x / target_size.x) * min_coeff,
                (border_rect.min_inset.y / target_size.y) * min_coeff,
                1. - (border_rect.max_inset.x / target_size.x) * min_coeff,
                1. - (border_rect.max_inset.y / target_size.y) * min_coeff,
            ];

            let image_side_width = image_size.x * (slices[2] - slices[0]);
            let image_side_height = image_size.y * (slices[3] - slices[1]);
            let target_side_width = target_size.x * (border[2] - border[0]);
            let target_side_height = target_size.y * (border[3] - border[1]);

            // compute the number of times to repeat the side and center slices when tiling along each axis
            // if the returned value is `1.` the slice will be stretched to fill the axis.
            let repeat_side_x =
                compute_tiled_subaxis(image_side_width, target_side_width, sides_scale_mode);
            let repeat_side_y =
                compute_tiled_subaxis(image_side_height, target_side_height, sides_scale_mode);
            let repeat_center_x =
                compute_tiled_subaxis(image_side_width, target_side_width, center_scale_mode);
            let repeat_center_y =
                compute_tiled_subaxis(image_side_height, target_side_height, center_scale_mode);

            [
                slices,
                border,
                [
                    repeat_side_x,
                    repeat_side_y,
                    repeat_center_x,
                    repeat_center_y,
                ],
            ]
        }
        SpriteImageMode::Tiled {
            tile_x,
            tile_y,
            stretch_value,
        } => {
            let rx = compute_tiled_axis(*tile_x, image_size.x, target_size.x, *stretch_value);
            let ry = compute_tiled_axis(*tile_y, image_size.y, target_size.y, *stretch_value);
            [[0., 0., 1., 1.], [0., 0., 1., 1.], [1., 1., rx, ry]]
        }
        SpriteImageMode::Auto => {
            unreachable!("Slices can not be computed for SpriteImageMode::Stretch")
        }
        SpriteImageMode::Scale(_) => {
            unreachable!("Slices can not be computed for SpriteImageMode::Scale")
        }
    }
}

fn compute_tiled_axis(tile: bool, image_extent: f32, target_extent: f32, stretch: f32) -> f32 {
    if tile {
        let s = image_extent * stretch;
        target_extent / s
    } else {
        1.
    }
}

fn compute_tiled_subaxis(image_extent: f32, target_extent: f32, mode: &SliceScaleMode) -> f32 {
    match mode {
        SliceScaleMode::Stretch => 1.,
        SliceScaleMode::Tile { stretch_value } => {
            let s = image_extent * *stretch_value;
            target_extent / s
        }
    }
}

#[cfg(test)]
mod tests {
    use core::mem::{offset_of, size_of};

    use super::*;
    use crate::retained::{preprocess_wgsl_for_test, validate_wgsl_for_test};

    #[test]
    fn texture_slice_instance_layout_matches_shader_storage_layout() {
        let layout = ui_texture_slice_style_vertex_layout();
        assert_eq!(size_of::<UiTextureSliceStyleInstance>(), 80);
        assert_eq!(layout.array_stride, 80);
        assert_eq!(layout.attributes.len(), 5);
        assert_eq!(
            layout
                .attributes
                .iter()
                .map(|attribute| attribute.offset as usize)
                .collect::<Vec<_>>(),
            vec![
                offset_of!(UiTextureSliceStyleInstance, color),
                offset_of!(UiTextureSliceStyleInstance, slices),
                offset_of!(UiTextureSliceStyleInstance, border),
                offset_of!(UiTextureSliceStyleInstance, repeat),
                offset_of!(UiTextureSliceStyleInstance, atlas),
            ]
        );
    }

    #[test]
    fn tiled_slice_repeat_counts_preserve_independent_axes() {
        let [slices, border, repeat] = compute_texture_slices(
            Vec2::new(20.0, 10.0),
            Vec2::new(100.0, 25.0),
            &SpriteImageMode::Tiled {
                tile_x: true,
                tile_y: false,
                stretch_value: 0.5,
            },
        );
        assert_eq!(slices, [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(border, [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(repeat, [1.0, 1.0, 10.0, 1.0]);
    }

    #[test]
    fn texture_slice_shader_validates_for_direct_and_storage_instances() {
        let source = include_str!("ui_texture_slice.wgsl");
        let prefix = "
            struct View { clip_from_world: mat4x4<f32>, }
            struct Globals { time: f32, }
        ";
        for storage in [false, true] {
            let definitions = if storage {
                &["UI_STORAGE_INSTANCE"][..]
            } else {
                &[][..]
            };
            let processed = preprocess_wgsl_for_test(source, definitions, prefix);
            validate_wgsl_for_test(
                &format!("texture slice shader (storage={storage})"),
                &processed,
            );
        }
    }
}
