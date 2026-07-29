//! Box shadows rendering

use core::hash::Hash;

use bevy_app::prelude::*;
use bevy_asset::*;
use bevy_camera::visibility::InheritedVisibility;
use bevy_color::{Alpha, ColorToComponents, LinearRgba};
use bevy_ecs::entity::{EntityHashMap, EntityIndexMap};
use bevy_ecs::prelude::*;
use bevy_ecs::system::{
    lifetimeless::{Read, SRes},
    *,
};
use bevy_math::{vec2, Affine2, FloatOrd, Rect, Vec2};
use bevy_mesh::VertexBufferLayout;
use bevy_render::sync_world::{MainEntity, MainEntityHashMap, MainEntityHashSet};
use bevy_render::{
    impl_atomic_pod,
    render_phase::*,
    render_resource::{
        binding_types::{storage_buffer_read_only_sized, uniform_buffer},
        *,
    },
    renderer::{RenderDevice, RenderQueue},
    view::*,
    Extract, ExtractSchedule, Render, RenderSystems,
};
use bevy_render::{GpuResourceAppExt, RenderApp, RenderStartup};
use bevy_shader::{Shader, ShaderDefVal};
use bevy_ui::{
    BoxShadow, CalculatedClip, ComputedNode, ComputedStackIndex, ComputedUiRenderTargetInfo,
    ComputedUiTargetCamera, ResolvedBorderRadius, UiGlobalTransform, Val,
};
use bevy_utils::default;
use bytemuck::{Pod, Zeroable};

use crate::{
    retained::{
        batch_retained_ui, remove_owner, vertex_storage_supported, RemovedUiNode,
        RetainedBatchItem, RetainedUiBatch, UiCameraPipelineState, UiInstanceArena,
    },
    BoxShadowSamples, RenderUiSystems, TransparentUi, UiCameraMap, UiGeometryInstance,
};

use super::{clipping_offsets, pack_corners, stack_z_offsets, UiCameraView};

/// A plugin that enables the rendering of box shadows.
pub struct BoxShadowPlugin;

impl Plugin for BoxShadowPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "box_shadow.wgsl");

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_command::<TransparentUi, DrawBoxShadows>()
                .init_resource::<ExtractedBoxShadows>()
                .init_gpu_resource::<BoxShadowMeta>()
                .init_gpu_resource::<SpecializedRenderPipelines<BoxShadowPipeline>>()
                .add_systems(RenderStartup, init_box_shadow_pipeline)
                .add_systems(
                    ExtractSchedule,
                    extract_shadows.in_set(RenderUiSystems::ExtractBoxShadows),
                )
                .add_systems(
                    Render,
                    (
                        queue_shadows.in_set(RenderSystems::Queue),
                        prepare_shadows.in_set(RenderSystems::PrepareBindGroups),
                    ),
                );
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Default, Pod, Zeroable)]
struct BoxShadowStyleInstance {
    color: [f32; 4],
    size: [f32; 2],
    size_padding: [f32; 2],
    radius: [[f32; 4]; 2],
    blur: f32,
    blur_padding: u32,
    bounds: [f32; 2],
}

impl_atomic_pod!(BoxShadowStyleInstance, BoxShadowStyleInstanceBlob);

/// Contains the vertices and bind groups to be sent to the GPU
#[derive(Resource)]
pub struct BoxShadowMeta {
    geometry_instances: AtomicSparseBufferVec<UiGeometryInstance>,
    style_instances: AtomicSparseBufferVec<BoxShadowStyleInstance>,
    instance_indices: RawBufferVec<u32>,
    view_bind_group: Option<BindGroup>,
    instance_bind_group: Option<BindGroup>,
    instance_buffer_ids: Option<(BufferId, BufferId)>,
    batches: Vec<RetainedUiBatch<CachedRenderPipelineId>>,
    arena: UiInstanceArena,
    camera_states: EntityHashMap<UiCameraPipelineState>,
    use_storage_buffers: bool,
}

impl Default for BoxShadowMeta {
    fn default() -> Self {
        Self {
            geometry_instances: AtomicSparseBufferVec::new(
                BufferUsages::VERTEX | BufferUsages::STORAGE,
                0,
                "box shadow geometry instances".into(),
            ),
            style_instances: AtomicSparseBufferVec::new(
                BufferUsages::VERTEX | BufferUsages::STORAGE,
                0,
                "box shadow style instances".into(),
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

#[derive(Resource)]
pub struct BoxShadowPipeline {
    pub view_layout: BindGroupLayoutDescriptor,
    pub instance_layout: BindGroupLayoutDescriptor,
    pub shader: Handle<Shader>,
}

pub fn init_box_shadow_pipeline(mut commands: Commands, asset_server: Res<AssetServer>) {
    let view_layout = BindGroupLayoutDescriptor::new(
        "box_shadow_view_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::VERTEX_FRAGMENT,
            uniform_buffer::<ViewUniform>(true),
        ),
    );
    let instance_layout = BindGroupLayoutDescriptor::new(
        "box_shadow_instance_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX,
            (
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
            ),
        ),
    );

    commands.insert_resource(BoxShadowPipeline {
        view_layout,
        instance_layout,
        shader: load_embedded_asset!(asset_server.as_ref(), "box_shadow.wgsl"),
    });
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct BoxShadowPipelineKey {
    pub target_format: TextureFormat,
    /// Number of samples, a higher value results in better quality shadows.
    pub samples: u32,
    pub storage_buffers: bool,
}

impl SpecializedRenderPipeline for BoxShadowPipeline {
    type Key = BoxShadowPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let buffers = if key.storage_buffers {
            vec![VertexBufferLayout::from_vertex_formats(
                VertexStepMode::Instance,
                vec![VertexFormat::Uint32],
            )]
        } else {
            vec![
                crate::pipeline::ui_geometry_vertex_layout(),
                box_shadow_style_vertex_layout(),
            ]
        };
        let mut shader_defs = vec![ShaderDefVal::UInt("SHADOW_SAMPLES".into(), key.samples)];
        let mut layout = vec![self.view_layout.clone()];
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
            label: Some("box_shadow_pipeline".into()),
            ..default()
        }
    }
}

fn box_shadow_style_vertex_layout() -> VertexBufferLayout {
    VertexBufferLayout {
        array_stride: 80,
        step_mode: VertexStepMode::Instance,
        attributes: vec![
            VertexAttribute {
                format: VertexFormat::Float32x4,
                offset: 0,
                shader_location: 8,
            },
            VertexAttribute {
                format: VertexFormat::Float32x2,
                offset: 16,
                shader_location: 9,
            },
            VertexAttribute {
                format: VertexFormat::Float32x4,
                offset: 32,
                shader_location: 10,
            },
            VertexAttribute {
                format: VertexFormat::Float32x4,
                offset: 48,
                shader_location: 11,
            },
            VertexAttribute {
                format: VertexFormat::Float32,
                offset: 64,
                shader_location: 12,
            },
            VertexAttribute {
                format: VertexFormat::Float32x2,
                offset: 72,
                shader_location: 13,
            },
        ],
    }
}

/// Description of a shadow to be sorted and queued for rendering
pub struct ExtractedBoxShadow {
    pub stack_index: u32,
    pub transform: Affine2,
    pub bounds: Vec2,
    pub clip: Option<Rect>,
    pub color: LinearRgba,
    pub radius: ResolvedBorderRadius,
    pub blur_radius: f32,
    pub size: Vec2,
}

/// List of extracted shadows to be sorted and queued for rendering
#[derive(Resource, Default)]
pub struct ExtractedBoxShadows {
    /// The list of box shadows grouped by their main-world entity, along with
    /// each group's target camera entity.
    ///
    /// This is a two-level data structure so that we can quickly remove all box
    /// shadows associated with a main-world entity when it changes.
    pub box_shadows: MainEntityHashMap<(Entity, EntityIndexMap<ExtractedBoxShadow>)>,
    pub changed: MainEntityHashSet,
    removed: Vec<RemovedUiNode>,
}

pub fn extract_shadows(
    mut commands: Commands,
    mut extracted_box_shadows: ResMut<ExtractedBoxShadows>,
    box_shadow_query: Extract<
        Query<
            (
                Entity,
                &ComputedNode,
                &ComputedStackIndex,
                &UiGlobalTransform,
                &InheritedVisibility,
                &BoxShadow,
                Option<&CalculatedClip>,
                &ComputedUiTargetCamera,
                &ComputedUiRenderTargetInfo,
            ),
            Or<(
                Changed<ComputedNode>,
                Changed<ComputedStackIndex>,
                Changed<UiGlobalTransform>,
                Changed<InheritedVisibility>,
                Changed<BoxShadow>,
                Changed<CalculatedClip>,
                Changed<ComputedUiTargetCamera>,
                Changed<ComputedUiRenderTargetInfo>,
            )>,
        >,
    >,
    camera_map: Extract<UiCameraMap>,
    (
        mut removed_computed_node_query,
        mut removed_computed_stack_index_query,
        mut removed_ui_global_transform_query,
        mut removed_inherited_visibility_query,
        mut removed_box_shadow_query,
        mut removed_calculated_clip_query,
        mut removed_computed_ui_target_camera_query,
        mut removed_computed_ui_render_target_info_query,
    ): (
        Extract<RemovedComponents<ComputedNode>>,
        Extract<RemovedComponents<ComputedStackIndex>>,
        Extract<RemovedComponents<UiGlobalTransform>>,
        Extract<RemovedComponents<InheritedVisibility>>,
        Extract<RemovedComponents<BoxShadow>>,
        Extract<RemovedComponents<CalculatedClip>>,
        Extract<RemovedComponents<ComputedUiTargetCamera>>,
        Extract<RemovedComponents<ComputedUiRenderTargetInfo>>,
    ),
    mut nodes_processed_this_frame: Local<MainEntityHashSet>,
) {
    nodes_processed_this_frame.clear();
    extracted_box_shadows.changed.clear();
    extracted_box_shadows.removed.clear();

    let mut mapping = camera_map.get_mapper();

    for (entity, uinode, stack_index, transform, visibility, box_shadow, clip, camera, target) in
        &box_shadow_query
    {
        let main_entity = MainEntity::from(entity);
        extracted_box_shadows.changed.insert(main_entity);

        // If there were any previous box shadows for this entity, despawn them.
        let extracted = &mut *extracted_box_shadows;
        if let Some((_, old_shadows)) = remove_owner(
            &mut extracted.box_shadows,
            main_entity,
            &mut extracted.removed,
        ) {
            for render_entity in old_shadows.keys() {
                commands.entity(*render_entity).despawn();
            }
        }

        // Skip if no visible shadows
        if !visibility.get() || box_shadow.is_empty() || uinode.is_empty() {
            continue;
        }

        let Some(extracted_camera_entity) = mapping.map(camera) else {
            continue;
        };
        let ui_physical_viewport_size = target.physical_size().as_vec2();
        let scale_factor = target.scale_factor();

        for drop_shadow in box_shadow.iter() {
            if drop_shadow.color.is_fully_transparent() {
                continue;
            }

            let resolve_val = |val, base, scale_factor| match val {
                Val::Auto => 0.,
                Val::Px(px) => px * scale_factor,
                Val::Percent(percent) => percent / 100. * base,
                Val::Vw(percent) => percent / 100. * ui_physical_viewport_size.x,
                Val::Vh(percent) => percent / 100. * ui_physical_viewport_size.y,
                Val::VMin(percent) => percent / 100. * ui_physical_viewport_size.min_element(),
                Val::VMax(percent) => percent / 100. * ui_physical_viewport_size.max_element(),
            };

            let spread_x = resolve_val(drop_shadow.spread_radius, uinode.size().x, scale_factor);
            let spread_ratio = (spread_x + uinode.size().x) / uinode.size().x;

            let spread = vec2(spread_x, uinode.size().y * spread_ratio - uinode.size().y);

            let blur_radius = resolve_val(drop_shadow.blur_radius, uinode.size().x, scale_factor);
            let offset = vec2(
                resolve_val(drop_shadow.x_offset, uinode.size().x, scale_factor),
                resolve_val(drop_shadow.y_offset, uinode.size().y, scale_factor),
            );

            let shadow_size = uinode.size() + spread;
            if shadow_size.cmple(Vec2::ZERO).any() {
                continue;
            }

            nodes_processed_this_frame.insert(main_entity);

            let radius = ResolvedBorderRadius {
                top_left: uinode.border_radius.top_left * spread_ratio,
                top_right: uinode.border_radius.top_right * spread_ratio,
                bottom_left: uinode.border_radius.bottom_left * spread_ratio,
                bottom_right: uinode.border_radius.bottom_right * spread_ratio,
            };

            extracted_box_shadows
                .box_shadows
                .entry(main_entity)
                .or_insert_with(|| (extracted_camera_entity, Default::default()))
                .1
                .insert(
                    commands.spawn_empty().id(),
                    ExtractedBoxShadow {
                        stack_index: stack_index.0,
                        transform: Affine2::from(transform) * Affine2::from_translation(offset),
                        color: drop_shadow.color.into(),
                        bounds: shadow_size + 6. * blur_radius,
                        clip: clip.map(|clip| clip.clip),
                        radius,
                        blur_radius,
                        size: shadow_size,
                    },
                );
        }
    }

    // Only remove the render-world data if we didn't handle the node above.
    // It's possible that a relevant component was removed and added in the same
    // frame.
    for main_entity in removed_computed_node_query
        .read()
        .chain(removed_computed_stack_index_query.read())
        .chain(removed_ui_global_transform_query.read())
        .chain(removed_inherited_visibility_query.read())
        .chain(removed_box_shadow_query.read())
        .chain(removed_calculated_clip_query.read())
        .chain(removed_computed_ui_target_camera_query.read())
        .chain(removed_computed_ui_render_target_info_query.read())
    {
        let main_entity = MainEntity::from(main_entity);
        if nodes_processed_this_frame.contains(&main_entity) {
            continue;
        }
        extracted_box_shadows.changed.insert(main_entity);
        let extracted = &mut *extracted_box_shadows;
        let Some((_, extracted_nodes)) = remove_owner(
            &mut extracted.box_shadows,
            main_entity,
            &mut extracted.removed,
        ) else {
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
pub fn queue_shadows(
    extracted_box_shadows: Res<ExtractedBoxShadows>,
    box_shadow_pipeline: Res<BoxShadowPipeline>,
    mut box_shadow_meta: ResMut<BoxShadowMeta>,
    mut pipelines: ResMut<SpecializedRenderPipelines<BoxShadowPipeline>>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<TransparentUi>>,
    render_views: Query<(Entity, &UiCameraView, Option<&BoxShadowSamples>), With<ExtractedView>>,
    camera_views: Query<&ExtractedView>,
    pipeline_cache: Res<PipelineCache>,
    draw_functions: Res<DrawFunctions<TransparentUi>>,
    render_device: Res<RenderDevice>,
) {
    let draw_function = draw_functions.read().id::<DrawBoxShadows>();
    let storage_buffers = vertex_storage_supported(&render_device, 2);
    let mut active_cameras = bevy_platform::collections::HashSet::new();
    let mut invalidated_cameras = bevy_platform::collections::HashSet::new();

    for (camera_entity, default_camera_view, shadow_samples) in &render_views {
        let Ok(view) = camera_views.get(default_camera_view.0) else {
            continue;
        };
        let state = UiCameraPipelineState {
            retained_view_entity: view.retained_view_entity,
            pipeline: pipelines.specialize(
                &pipeline_cache,
                &box_shadow_pipeline,
                BoxShadowPipelineKey {
                    target_format: view.target_format,
                    samples: shadow_samples.copied().unwrap_or_default().0,
                    storage_buffers,
                },
            ),
        };
        active_cameras.insert(camera_entity);
        if box_shadow_meta.camera_states.insert(camera_entity, state) != Some(state) {
            invalidated_cameras.insert(camera_entity);
        }
    }

    for removed in &extracted_box_shadows.removed {
        let Some(camera_state) = box_shadow_meta.camera_states.get(&removed.camera_entity) else {
            continue;
        };
        if let Some(phase) = transparent_render_phases.get_mut(&camera_state.retained_view_entity) {
            phase.remove(removed.render_entity, removed.main_entity);
        }
    }

    let mut dirty = extracted_box_shadows.changed.clone();
    if !invalidated_cameras.is_empty() {
        dirty.extend(extracted_box_shadows.box_shadows.iter().filter_map(
            |(main_entity, (camera_entity, _))| {
                invalidated_cameras
                    .contains(camera_entity)
                    .then_some(*main_entity)
            },
        ));
    }

    for main_entity in dirty {
        let Some((camera_entity, extracted_sub_shadows)) =
            extracted_box_shadows.box_shadows.get(&main_entity)
        else {
            continue;
        };
        let Some(camera_state) = box_shadow_meta.camera_states.get(camera_entity) else {
            continue;
        };
        let Some(transparent_phase) =
            transparent_render_phases.get_mut(&camera_state.retained_view_entity)
        else {
            continue;
        };
        for (entity, extracted_shadow) in extracted_sub_shadows {
            transparent_phase.add_retained(TransparentUi {
                draw_function,
                pipeline: camera_state.pipeline,
                entity: (*entity, main_entity),
                sort_key: FloatOrd(
                    extracted_shadow.stack_index as f32 + stack_z_offsets::BOX_SHADOW,
                ),

                batch_range: 0..0,
                extra_index: PhaseItemExtraIndex::None,
                indexed: true,
                batch_index: None,
            });
        }
    }
    box_shadow_meta
        .camera_states
        .retain(|camera, _| active_cameras.contains(camera));
}

fn generate_shadow_instance(
    shadow: &ExtractedBoxShadow,
) -> Option<(UiGeometryInstance, BoxShadowStyleInstance)> {
    let (position_diff, culled) =
        clipping_offsets(shadow.transform, Vec2::ZERO, shadow.bounds, shadow.clip);
    if culled {
        return None;
    }
    let uvs = [
        position_diff[0],
        Vec2::new(shadow.bounds.x + position_diff[1].x, position_diff[1].y),
        shadow.bounds + position_diff[2],
        Vec2::new(position_diff[3].x, shadow.bounds.y + position_diff[3].y),
    ]
    .map(|position| position / shadow.bounds);
    let (position_diff_01, position_diff_23) = pack_corners(position_diff);
    let (uv_01, uv_23) = pack_corners(uvs);
    Some((
        UiGeometryInstance {
            transform_x: shadow.transform.x_axis.into(),
            transform_y: shadow.transform.y_axis.into(),
            translation: shadow.transform.translation.into(),
            size: shadow.bounds.into(),
            position_diff_01,
            position_diff_23,
            uv_01,
            uv_23,
        },
        BoxShadowStyleInstance {
            color: shadow.color.to_f32_array(),
            size: shadow.size.into(),
            size_padding: [0.0; 2],
            radius: shadow.radius.into(),
            blur: shadow.blur_radius,
            blur_padding: 0,
            bounds: shadow.bounds.into(),
        },
    ))
}

fn rebuild_shadow_owner(
    main_entity: MainEntity,
    meta: &mut BoxShadowMeta,
    extracted_shadows: &ExtractedBoxShadows,
) {
    meta.arena.free_owner(main_entity);
    let Some((_, shadows)) = extracted_shadows.box_shadows.get(&main_entity) else {
        return;
    };
    let mut owned = Vec::with_capacity(shadows.len());
    for (render_entity, shadow) in shadows {
        if let Some((geometry, style)) = generate_shadow_instance(shadow) {
            let (start, capacity) = meta.arena.alloc(1);
            meta.geometry_instances.grow(start + capacity);
            meta.style_instances.grow(start + capacity);
            meta.geometry_instances.set(start, geometry);
            meta.style_instances.set(start, style);
            meta.arena.insert(*render_entity, start, 1, capacity);
        } else {
            meta.arena.insert_empty(*render_entity);
        }
        owned.push(*render_entity);
    }
    if !owned.is_empty() {
        meta.arena.owners.insert(main_entity, owned);
    }
}

pub fn prepare_shadows(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    mut ui_meta: ResMut<BoxShadowMeta>,
    extracted_shadows: Res<ExtractedBoxShadows>,
    view_uniforms: Res<ViewUniforms>,
    box_shadow_pipeline: Res<BoxShadowPipeline>,
    mut phases: ResMut<ViewSortedRenderPhases<TransparentUi>>,
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
    ui_meta.use_storage_buffers = vertex_storage_supported(&render_device, 2);
    if ui_meta.arena.needs_compaction() || !ui_meta.arena.initialized {
        ui_meta.arena.reset();
        ui_meta.geometry_instances.clear();
        ui_meta.style_instances.clear();
        for main_entity in extracted_shadows.box_shadows.keys().copied() {
            rebuild_shadow_owner(main_entity, &mut ui_meta, &extracted_shadows);
        }
    } else {
        for main_entity in extracted_shadows.changed.iter().copied() {
            rebuild_shadow_owner(main_entity, &mut ui_meta, &extracted_shadows);
        }
    }

    ui_meta
        .geometry_instances
        .write_buffers(&render_device, &render_queue);
    ui_meta
        .style_instances
        .write_buffers(&render_device, &render_queue);
    ui_meta.geometry_instances.prepare_to_populate_buffers(
        &render_device,
        &pipeline_cache,
        &mut sparse_buffer_update_jobs,
        &mut sparse_buffer_update_bind_groups,
        &sparse_buffer_update_pipelines,
    );
    ui_meta.style_instances.prepare_to_populate_buffers(
        &render_device,
        &pipeline_cache,
        &mut sparse_buffer_update_jobs,
        &mut sparse_buffer_update_bind_groups,
        &sparse_buffer_update_pipelines,
    );

    let Some(view_binding) = view_uniforms.uniforms.binding() else {
        ui_meta.batches.clear();
        return;
    };
    ui_meta.view_bind_group = Some(render_device.create_bind_group(
        "box_shadow_view_bind_group",
        &pipeline_cache.get_bind_group_layout(&box_shadow_pipeline.view_layout),
        &BindGroupEntries::single(view_binding),
    ));

    if ui_meta.use_storage_buffers {
        if let (Some(geometry_buffer), Some(style_buffer)) = (
            ui_meta.geometry_instances.buffer(),
            ui_meta.style_instances.buffer(),
        ) {
            let ids = (geometry_buffer.id(), style_buffer.id());
            if ui_meta.instance_buffer_ids != Some(ids) {
                ui_meta.instance_bind_group = Some(render_device.create_bind_group(
                    "box_shadow_instance_bind_group",
                    &pipeline_cache.get_bind_group_layout(&box_shadow_pipeline.instance_layout),
                    &BindGroupEntries::sequential((
                        geometry_buffer.as_entire_binding(),
                        style_buffer.as_entire_binding(),
                    )),
                ));
                ui_meta.instance_buffer_ids = Some(ids);
            }
        } else {
            ui_meta.instance_bind_group = None;
            ui_meta.instance_buffer_ids = None;
        }
    }

    let BoxShadowMeta {
        arena,
        instance_indices,
        use_storage_buffers,
        ..
    } = &mut *ui_meta;
    let batches = batch_retained_ui(
        &mut phases,
        instance_indices,
        *use_storage_buffers,
        |item| {
            if extracted_shadows
                .box_shadows
                .get(&item.main_entity())
                .and_then(|(_, shadows)| shadows.get(&item.entity()))
                .is_none()
            {
                return RetainedBatchItem::NotOwned;
            }
            let Some(slot) = arena.slots.get(&item.entity()) else {
                return RetainedBatchItem::Culled;
            };
            if slot.instances.count == 0 {
                RetainedBatchItem::Culled
            } else {
                RetainedBatchItem::Drawable {
                    instances: slot.instances,
                    key: item.pipeline,
                }
            }
        },
        PartialEq::eq,
        |_, _| {},
    );
    ui_meta.batches = batches;
    ui_meta
        .instance_indices
        .write_buffer(&render_device, &render_queue);
}

pub type DrawBoxShadows = (
    SetItemPipeline,
    SetBoxShadowViewBindGroup<0>,
    SetBoxShadowInstanceBindGroup<1>,
    DrawBoxShadow,
);

pub struct SetBoxShadowInstanceBindGroup<const I: usize>;
impl<const I: usize> RenderCommand<TransparentUi> for SetBoxShadowInstanceBindGroup<I> {
    type Param = SRes<BoxShadowMeta>;
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
                return RenderCommandResult::Failure("missing box shadow instance bind group");
            };
            pass.set_bind_group(I, bind_group, &[]);
        }
        RenderCommandResult::Success
    }
}

pub struct SetBoxShadowViewBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetBoxShadowViewBindGroup<I> {
    type Param = SRes<BoxShadowMeta>;
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

pub struct DrawBoxShadow;
impl RenderCommand<TransparentUi> for DrawBoxShadow {
    type Param = SRes<BoxShadowMeta>;
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
        let ui_meta = ui_meta.into_inner();
        let Some(batch) = ui_meta.batches.get(batch_index as usize) else {
            return RenderCommandResult::Failure("box shadow batch index out of range");
        };
        if ui_meta.use_storage_buffers {
            let Some(indices) = ui_meta.instance_indices.buffer() else {
                return RenderCommandResult::Failure("missing box shadow instance indices");
            };
            pass.set_vertex_buffer(0, indices.slice(..));
        } else {
            let Some(geometry) = ui_meta.geometry_instances.buffer() else {
                return RenderCommandResult::Failure("missing box shadow geometry instances");
            };
            let Some(style) = ui_meta.style_instances.buffer() else {
                return RenderCommandResult::Failure("missing box shadow style instances");
            };
            pass.set_vertex_buffer(0, geometry.slice(..));
            pass.set_vertex_buffer(1, style.slice(..));
        }
        pass.draw(0..6, batch.range.clone());
        RenderCommandResult::Success
    }
}

#[cfg(test)]
mod tests {
    use core::mem::{offset_of, size_of};

    use super::*;
    use crate::retained::{preprocess_wgsl_for_test, validate_wgsl_for_test};

    #[test]
    fn shadow_instance_layout_matches_shader_storage_layout() {
        let layout = box_shadow_style_vertex_layout();
        assert_eq!(size_of::<BoxShadowStyleInstance>(), 80);
        assert_eq!(layout.array_stride, 80);
        assert_eq!(layout.attributes.len(), 6);
        assert_eq!(
            layout
                .attributes
                .iter()
                .map(|attribute| attribute.offset as usize)
                .collect::<Vec<_>>(),
            vec![
                offset_of!(BoxShadowStyleInstance, color),
                offset_of!(BoxShadowStyleInstance, size),
                offset_of!(BoxShadowStyleInstance, radius),
                offset_of!(BoxShadowStyleInstance, radius) + 16,
                offset_of!(BoxShadowStyleInstance, blur),
                offset_of!(BoxShadowStyleInstance, bounds),
            ]
        );
    }

    #[test]
    fn shadow_instance_preserves_clipping_uvs_and_elliptical_radii() {
        let shadow = ExtractedBoxShadow {
            stack_index: 0,
            transform: Affine2::from_translation(Vec2::new(50.0, 40.0)),
            bounds: Vec2::new(100.0, 80.0),
            clip: Some(Rect::new(10.0, 20.0, 90.0, 70.0)),
            color: LinearRgba::WHITE,
            radius: ResolvedBorderRadius {
                top_left: Vec2::new(1.0, 2.0),
                top_right: Vec2::new(3.0, 4.0),
                bottom_right: Vec2::new(5.0, 6.0),
                bottom_left: Vec2::new(7.0, 8.0),
            },
            blur_radius: 6.0,
            size: Vec2::new(90.0, 70.0),
        };
        let (geometry, style) = generate_shadow_instance(&shadow).unwrap();
        assert_eq!(geometry.position_diff_01, [10.0, 20.0, -10.0, 20.0]);
        assert_eq!(geometry.position_diff_23, [-10.0, -10.0, 10.0, -10.0]);
        assert_eq!(geometry.uv_01, [0.1, 0.25, 0.9, 0.25]);
        assert_eq!(geometry.uv_23, [0.9, 0.875, 0.1, 0.875]);
        assert_eq!(style.radius, [[1.0, 3.0, 5.0, 7.0], [2.0, 4.0, 6.0, 8.0]]);
        assert_eq!(style.blur, 6.0);
    }

    #[test]
    fn shadow_shader_validates_for_direct_and_storage_instances() {
        let source = include_str!("box_shadow.wgsl").replace("#SHADOW_SAMPLES", "8");
        let prefix = "
            struct View { clip_from_world: mat4x4<f32>, }
            fn select_corner_radius(
                point: vec2<f32>,
                radii_x: vec4<f32>,
                radii_y: vec4<f32>,
            ) -> vec2<f32> {
                return vec2(radii_x.x, radii_y.x);
            }
        ";
        for storage in [false, true] {
            let definitions = if storage {
                &["UI_STORAGE_INSTANCE"][..]
            } else {
                &[][..]
            };
            let processed = preprocess_wgsl_for_test(&source, definitions, prefix);
            validate_wgsl_for_test(
                &format!("box shadow shader (storage={storage})"),
                &processed,
            );
        }
    }
}
