use crate::retained::{
    batch_retained_ui, remove_owner, RemovedUiNode, RetainedBatchItem, RetainedUiBatch,
    UiInstanceArena,
};
use crate::ui_material::{MaterialNode, UiMaterial, UiMaterialKey};
use crate::*;
use bevy_asset::*;
use bevy_ecs::{
    entity::EntityHashMap,
    prelude::With,
    system::{
        lifetimeless::{Read, SRes},
        *,
    },
};
use bevy_math::{Affine2, FloatOrd, Rect, Vec2};
use bevy_mesh::VertexBufferLayout;
use bevy_render::{
    globals::{GlobalsBuffer, GlobalsUniform},
    impl_atomic_pod,
    render_asset::{PrepareAssetError, RenderAsset, RenderAssetPlugin, RenderAssets},
    render_phase::*,
    render_resource::{binding_types::uniform_buffer, *},
    renderer::{RenderDevice, RenderQueue},
    sync_world::MainEntity,
    view::*,
    Extract, ExtractSchedule, Render, RenderSystems,
};
use bevy_render::{GpuResourceAppExt, RenderApp, RenderStartup};
use bevy_shader::{load_shader_library, Shader, ShaderRef};
use bevy_sprite::BorderRect;
use bevy_ui::ComputedStackIndex;
use bevy_utils::default;
use bytemuck::{Pod, Zeroable};
use core::{hash::Hash, marker::PhantomData};

/// Adds the necessary ECS resources and render logic to enable rendering entities using the given
/// [`UiMaterial`] asset type (which includes [`UiMaterial`] types).
pub struct UiMaterialPlugin<M: UiMaterial>(PhantomData<M>);

impl<M: UiMaterial> Default for UiMaterialPlugin<M> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<M: UiMaterial> Plugin for UiMaterialPlugin<M>
where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "ui_vertex_output.wgsl");

        embedded_asset!(app, "ui_material.wgsl");

        app.init_asset::<M>()
            .register_type::<MaterialNode<M>>()
            .add_plugins(RenderAssetPlugin::<PreparedUiMaterial<M>, GpuImage>::default());

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_command::<TransparentUi, DrawUiMaterial<M>>()
                .init_resource::<ExtractedUiMaterialNodes<M>>()
                .init_gpu_resource::<UiMaterialMeta<M>>()
                .init_gpu_resource::<SpecializedRenderPipelines<UiMaterialPipeline<M>>>()
                .add_systems(RenderStartup, init_ui_material_pipeline::<M>)
                .add_systems(
                    ExtractSchedule,
                    extract_ui_material_nodes::<M>.in_set(RenderUiSystems::ExtractBackgrounds),
                )
                .add_systems(
                    Render,
                    (
                        queue_ui_material_nodes::<M>.in_set(RenderSystems::Queue),
                        prepare_uimaterial_nodes::<M>.in_set(RenderSystems::PrepareBindGroups),
                    ),
                );
        }
    }
}

#[derive(Resource)]
pub struct UiMaterialMeta<M: UiMaterial> {
    vertices: AtomicSparseBufferVec<UiMaterialVertex>,
    view_bind_group: Option<BindGroup>,
    batches: Vec<RetainedUiBatch<UiMaterialBatchKey<M>>>,
    arena: UiInstanceArena,
    camera_states: EntityHashMap<UiMaterialCameraState>,
    material_keys: HashMap<AssetId<M>, M::Data>,
    pending_queue: MainEntityHashSet,
    marker: PhantomData<M>,
}

impl<M: UiMaterial> Default for UiMaterialMeta<M> {
    fn default() -> Self {
        Self {
            vertices: AtomicSparseBufferVec::new(
                BufferUsages::VERTEX | BufferUsages::STORAGE,
                0,
                "retained UI material vertices".into(),
            ),
            view_bind_group: Default::default(),
            batches: Vec::new(),
            arena: UiInstanceArena::default(),
            camera_states: EntityHashMap::default(),
            material_keys: HashMap::default(),
            pending_queue: MainEntityHashSet::default(),
            marker: PhantomData,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Default, Pod, Zeroable)]
pub struct UiMaterialVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
    pub size: [f32; 2],
    pub border: [f32; 4],
    pub radius: [[f32; 4]; 2],
}

impl_atomic_pod!(UiMaterialVertex, UiMaterialVertexBlob);

#[derive(Debug)]
struct UiMaterialBatchKey<M: UiMaterial> {
    pipeline: CachedRenderPipelineId,
    material: AssetId<M>,
}

impl<M: UiMaterial> Clone for UiMaterialBatchKey<M> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<M: UiMaterial> Copy for UiMaterialBatchKey<M> {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct UiMaterialCameraState {
    retained_view_entity: RetainedViewEntity,
    target_format: TextureFormat,
}

/// Render pipeline data for a given [`UiMaterial`]
#[derive(Resource)]
pub struct UiMaterialPipeline<M: UiMaterial> {
    pub ui_layout: BindGroupLayoutDescriptor,
    pub view_layout: BindGroupLayoutDescriptor,
    pub vertex_shader: Handle<Shader>,
    pub fragment_shader: Handle<Shader>,
    marker: PhantomData<M>,
}

impl<M: UiMaterial> SpecializedRenderPipeline for UiMaterialPipeline<M>
where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    type Key = UiMaterialKey<M>;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let vertex_layout = VertexBufferLayout::from_vertex_formats(
            VertexStepMode::Vertex,
            vec![
                // position
                VertexFormat::Float32x3,
                // uv
                VertexFormat::Float32x2,
                // size
                VertexFormat::Float32x2,
                // border widths
                VertexFormat::Float32x4,
                // border radius x values (top left, top right, bottom right, bottom left)
                VertexFormat::Float32x4,
                // border radius y values (top left, top right, bottom right, bottom left)
                VertexFormat::Float32x4,
            ],
        );
        let shader_defs = Vec::new();

        let mut descriptor = RenderPipelineDescriptor {
            vertex: VertexState {
                shader: self.vertex_shader.clone(),
                shader_defs: shader_defs.clone(),
                buffers: vec![vertex_layout],
                ..default()
            },
            fragment: Some(FragmentState {
                shader: self.fragment_shader.clone(),
                shader_defs,
                targets: vec![Some(ColorTargetState {
                    format: key.target_format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            label: Some("ui_material_pipeline".into()),
            ..default()
        };

        descriptor.layout = vec![self.view_layout.clone(), self.ui_layout.clone()];

        M::specialize(&mut descriptor, key);

        descriptor
    }
}

pub fn init_ui_material_pipeline<M: UiMaterial>(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    render_device: Res<RenderDevice>,
) {
    let ui_layout = M::bind_group_layout_descriptor(&render_device);

    let view_layout = BindGroupLayoutDescriptor::new(
        "ui_view_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                uniform_buffer::<ViewUniform>(true),
                uniform_buffer::<GlobalsUniform>(false),
            ),
        ),
    );

    let load_default = || load_embedded_asset!(asset_server.as_ref(), "ui_material.wgsl");

    commands.insert_resource(UiMaterialPipeline::<M> {
        ui_layout,
        view_layout,
        vertex_shader: match M::vertex_shader() {
            ShaderRef::Default => load_default(),
            ShaderRef::Handle(handle) => handle,
            ShaderRef::Path(path) => asset_server.load(path),
        },
        fragment_shader: match M::fragment_shader() {
            ShaderRef::Default => load_default(),
            ShaderRef::Handle(handle) => handle,
            ShaderRef::Path(path) => asset_server.load(path),
        },
        marker: PhantomData,
    });
}

pub type DrawUiMaterial<M> = (
    SetItemPipeline,
    SetMatUiViewBindGroup<M, 0>,
    SetUiMaterialBindGroup<M, 1>,
    DrawUiMaterialNode<M>,
);

pub struct SetMatUiViewBindGroup<M: UiMaterial, const I: usize>(PhantomData<M>);
impl<P: PhaseItem, M: UiMaterial, const I: usize> RenderCommand<P> for SetMatUiViewBindGroup<M, I> {
    type Param = SRes<UiMaterialMeta<M>>;
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
            return RenderCommandResult::Failure("UI material view bind group not available");
        };
        pass.set_bind_group(I, view_bind_group, &[view_uniform.offset]);
        RenderCommandResult::Success
    }
}

pub struct SetUiMaterialBindGroup<M: UiMaterial, const I: usize>(PhantomData<M>);
impl<M: UiMaterial, const I: usize> RenderCommand<TransparentUi> for SetUiMaterialBindGroup<M, I> {
    type Param = (
        SRes<RenderAssets<PreparedUiMaterial<M>>>,
        SRes<UiMaterialMeta<M>>,
    );
    type ViewQuery = ();
    type ItemQuery = ();

    fn render<'w>(
        item: &TransparentUi,
        _view: (),
        _entity: Option<()>,
        (materials, meta): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(batch_index) = item.batch_index else {
            return RenderCommandResult::Skip;
        };
        let Some(batch) = meta.into_inner().batches.get(batch_index as usize) else {
            return RenderCommandResult::Failure("UI material batch index out of range");
        };
        let Some(material) = materials.into_inner().get(batch.key.material) else {
            return RenderCommandResult::Skip;
        };
        pass.set_bind_group(I, &material.bind_group, &[]);
        RenderCommandResult::Success
    }
}

pub struct DrawUiMaterialNode<M>(PhantomData<M>);
impl<M: UiMaterial> RenderCommand<TransparentUi> for DrawUiMaterialNode<M> {
    type Param = SRes<UiMaterialMeta<M>>;
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
            return RenderCommandResult::Failure("UI material batch index out of range");
        };
        let Some(vertices) = meta.vertices.buffer() else {
            return RenderCommandResult::Failure("missing retained UI material vertices");
        };
        pass.set_vertex_buffer(0, vertices.slice(..));
        pass.draw(batch.range.clone(), 0..1);
        RenderCommandResult::Success
    }
}

pub struct ExtractedUiMaterialNode<M: UiMaterial> {
    pub stack_index: u32,
    pub transform: Affine2,
    pub rect: Rect,
    pub border: BorderRect,
    pub border_radius: [[f32; 4]; 2],
    pub material: AssetId<M>,
    pub clip: Option<Rect>,
}

/// A render-world resource that stores all material nodes in the scene.
#[derive(Resource)]
pub struct ExtractedUiMaterialNodes<M: UiMaterial> {
    /// The list of material nodes grouped by their main-world entity, along with
    /// each group's target camera entity.
    ///
    /// This is a two-level data structure so that we can quickly remove all
    /// material nodes associated with a main-world entity when it changes.
    pub uinodes: MainEntityHashMap<(Entity, EntityIndexMap<ExtractedUiMaterialNode<M>>)>,
    pub changed: MainEntityHashSet,
    removed: Vec<RemovedUiNode>,
}

impl<M: UiMaterial> Default for ExtractedUiMaterialNodes<M> {
    fn default() -> Self {
        Self {
            uinodes: Default::default(),
            changed: Default::default(),
            removed: Default::default(),
        }
    }
}

pub fn extract_ui_material_nodes<M: UiMaterial>(
    mut commands: Commands,
    mut extracted_uinodes: ResMut<ExtractedUiMaterialNodes<M>>,
    materials: Extract<Res<Assets<M>>>,
    uinode_query: Extract<
        Query<
            (
                Entity,
                &ComputedNode,
                &ComputedStackIndex,
                &UiGlobalTransform,
                &MaterialNode<M>,
                &InheritedVisibility,
                Option<&CalculatedClip>,
                &ComputedUiTargetCamera,
            ),
            Or<(
                Changed<ComputedNode>,
                Changed<ComputedStackIndex>,
                Changed<UiGlobalTransform>,
                Changed<MaterialNode<M>>,
                Changed<InheritedVisibility>,
                Changed<CalculatedClip>,
                Changed<ComputedUiTargetCamera>,
            )>,
        >,
    >,
    camera_map: Extract<UiCameraMap>,
    (
        mut removed_computed_node_query,
        mut removed_computed_stack_index_query,
        mut removed_ui_global_transform_query,
        mut removed_material_node_query,
        mut removed_inherited_visibility_query,
        mut removed_calculated_clip_query,
        mut removed_computed_ui_target_camera_query,
    ): (
        Extract<RemovedComponents<ComputedNode>>,
        Extract<RemovedComponents<ComputedStackIndex>>,
        Extract<RemovedComponents<UiGlobalTransform>>,
        Extract<RemovedComponents<MaterialNode<M>>>,
        Extract<RemovedComponents<InheritedVisibility>>,
        Extract<RemovedComponents<CalculatedClip>>,
        Extract<RemovedComponents<ComputedUiTargetCamera>>,
    ),
    mut nodes_to_reextract_next_frame: Local<MainEntityHashSet>,
    mut nodes_processed_this_frame: Local<MainEntityHashSet>,
) {
    nodes_processed_this_frame.clear();
    extracted_uinodes.changed.clear();
    extracted_uinodes.removed.clear();
    let mut camera_mapper = camera_map.get_mapper();
    let nodes_to_reextract = mem::take(&mut *nodes_to_reextract_next_frame);

    for (
        entity,
        computed_node,
        stack_index,
        transform,
        handle,
        inherited_visibility,
        clip,
        camera,
    ) in uinode_query.iter().chain(
        nodes_to_reextract
            .into_iter()
            .filter_map(|main_entity| uinode_query.get(main_entity.entity()).ok()),
    ) {
        let main_entity = MainEntity::from(entity);
        extracted_uinodes.changed.insert(main_entity);

        // Make sure we don't process the same node more than once.
        // This is possible if the node was marked for reextraction on the
        // previous frame and was also otherwise changed on this frame.
        if nodes_processed_this_frame.contains(&main_entity) {
            continue;
        }
        // If there were any previous UI nodes for this entity, despawn them.
        let extracted = &mut *extracted_uinodes;
        if let Some((_, old_nodes)) =
            remove_owner(&mut extracted.uinodes, main_entity, &mut extracted.removed)
        {
            for render_entity in old_nodes.keys() {
                commands.entity(*render_entity).despawn();
            }
        }

        // skip invisible nodes
        if !inherited_visibility.get() || computed_node.is_empty() {
            continue;
        }

        // If the material hasn't finished loading, skip the entity, and
        // remember that we did so that we reextract the node next frame.
        if !materials.contains(handle) {
            nodes_to_reextract_next_frame.insert(main_entity);
            continue;
        }

        let Some(extracted_camera_entity) = camera_mapper.map(camera) else {
            continue;
        };
        nodes_processed_this_frame.insert(main_entity);

        extracted_uinodes
            .uinodes
            .entry(main_entity)
            .or_insert_with(|| (extracted_camera_entity, Default::default()))
            .1
            .insert(
                commands.spawn_empty().id(),
                ExtractedUiMaterialNode {
                    stack_index: stack_index.0,
                    transform: transform.into(),
                    material: handle.id(),
                    rect: Rect {
                        min: Vec2::ZERO,
                        max: computed_node.size(),
                    },
                    border: computed_node.border(),
                    border_radius: computed_node.border_radius().into(),
                    clip: clip.map(|clip| clip.clip),
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
        .chain(removed_material_node_query.read())
        .chain(removed_inherited_visibility_query.read())
        .chain(removed_calculated_clip_query.read())
        .chain(removed_computed_ui_target_camera_query.read())
    {
        let main_entity = MainEntity::from(main_entity);
        if nodes_processed_this_frame.contains(&main_entity) {
            continue;
        }
        extracted_uinodes.changed.insert(main_entity);
        let extracted = &mut *extracted_uinodes;
        let Some((_, extracted_nodes)) =
            remove_owner(&mut extracted.uinodes, main_entity, &mut extracted.removed)
        else {
            continue;
        };
        for render_entity in extracted_nodes.keys() {
            commands.entity(*render_entity).despawn();
        }
    }
}

fn generate_material_vertices(
    node: &ExtractedUiMaterialNode<impl UiMaterial>,
) -> Option<[UiMaterialVertex; 6]> {
    let size = node.rect.size();
    let (position_diff, culled) = clipping_offsets(node.transform, Vec2::ZERO, size, node.clip);
    if culled {
        return None;
    }
    let positions =
        QUAD_VERTEX_POSITIONS.map(|position| node.transform.transform_point2(position * size));
    let uvs = [
        node.rect.min + position_diff[0],
        Vec2::new(node.rect.max.x, node.rect.min.y) + position_diff[1],
        node.rect.max + position_diff[2],
        Vec2::new(node.rect.min.x, node.rect.max.y) + position_diff[3],
    ]
    .map(|position| position / node.rect.max);
    let border = [
        node.border.min_inset.x,
        node.border.min_inset.y,
        node.border.max_inset.x,
        node.border.max_inset.y,
    ];
    Some(QUAD_INDICES.map(|corner| {
        UiMaterialVertex {
            position: (positions[corner] + position_diff[corner])
                .extend(1.0)
                .into(),
            uv: uvs[corner].into(),
            size: size.into(),
            border,
            radius: node.border_radius,
        }
    }))
}

fn rebuild_material_owner<M: UiMaterial>(
    main_entity: MainEntity,
    meta: &mut UiMaterialMeta<M>,
    extracted: &ExtractedUiMaterialNodes<M>,
) {
    meta.arena.free_owner(main_entity);
    let Some((_, nodes)) = extracted.uinodes.get(&main_entity) else {
        return;
    };
    let mut owned = Vec::with_capacity(nodes.len());
    for (render_entity, node) in nodes {
        if let Some(vertices) = generate_material_vertices(node) {
            let count = vertices.len() as u32;
            let (start, capacity) = meta.arena.alloc_exact(count);
            meta.vertices.grow(start + capacity);
            for (offset, vertex) in vertices.into_iter().enumerate() {
                meta.vertices.set(start + offset as u32, vertex);
            }
            meta.arena.insert(*render_entity, start, count, capacity);
        } else {
            meta.arena.insert_empty(*render_entity);
        }
        owned.push(*render_entity);
    }
    if !owned.is_empty() {
        meta.arena.owners.insert(main_entity, owned);
    }
}

pub fn prepare_uimaterial_nodes<M: UiMaterial>(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    mut meta: ResMut<UiMaterialMeta<M>>,
    extracted: Res<ExtractedUiMaterialNodes<M>>,
    view_uniforms: Res<ViewUniforms>,
    globals_buffer: Res<GlobalsBuffer>,
    pipeline: Res<UiMaterialPipeline<M>>,
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
    if meta.arena.needs_compaction() || !meta.arena.initialized {
        meta.arena.reset();
        meta.vertices.clear();
        for main_entity in extracted.uinodes.keys().copied() {
            rebuild_material_owner(main_entity, &mut meta, &extracted);
        }
    } else {
        for main_entity in extracted.changed.iter().copied() {
            rebuild_material_owner(main_entity, &mut meta, &extracted);
        }
    }
    meta.vertices.write_buffers(&render_device, &render_queue);
    meta.vertices.prepare_to_populate_buffers(
        &render_device,
        &pipeline_cache,
        &mut sparse_buffer_update_jobs,
        &mut sparse_buffer_update_bind_groups,
        &sparse_buffer_update_pipelines,
    );
    let (Some(view_binding), Some(globals_binding)) = (
        view_uniforms.uniforms.binding(),
        globals_buffer.buffer.binding(),
    ) else {
        meta.batches.clear();
        return;
    };
    meta.view_bind_group = Some(render_device.create_bind_group(
        "ui_material_view_bind_group",
        &pipeline_cache.get_bind_group_layout(&pipeline.view_layout),
        &BindGroupEntries::sequential((view_binding, globals_binding)),
    ));
    let UiMaterialMeta { arena, .. } = &*meta;
    let mut unused_indices = RawBufferVec::new(BufferUsages::VERTEX);
    meta.batches = batch_retained_ui(
        &mut phases,
        &mut unused_indices,
        false,
        |item| {
            let Some(node) = extracted
                .uinodes
                .get(&item.main_entity())
                .and_then(|(_, nodes)| nodes.get(&item.entity()))
            else {
                return RetainedBatchItem::NotOwned;
            };
            let Some(slot) = arena.slots.get(&item.entity()) else {
                return RetainedBatchItem::Culled;
            };
            if slot.instances.count == 0 {
                RetainedBatchItem::Culled
            } else {
                RetainedBatchItem::Drawable {
                    instances: slot.instances,
                    key: UiMaterialBatchKey {
                        pipeline: item.pipeline,
                        material: node.material,
                    },
                }
            }
        },
        |left, right| left.pipeline == right.pipeline && left.material == right.material,
        |_, _| {},
    );
}

pub struct PreparedUiMaterial<T: UiMaterial> {
    pub bindings: BindingResources,
    pub bind_group: BindGroup,
    pub key: T::Data,
}

impl<M: UiMaterial> RenderAsset for PreparedUiMaterial<M> {
    type SourceAsset = M;

    type Param = (
        SRes<RenderDevice>,
        SRes<PipelineCache>,
        SRes<UiMaterialPipeline<M>>,
        M::Param,
    );

    fn prepare_asset(
        material: Self::SourceAsset,
        _: AssetId<Self::SourceAsset>,
        (render_device, pipeline_cache, pipeline, material_param): &mut SystemParamItem<
            Self::Param,
        >,
        _: Option<&Self>,
    ) -> Result<Self, PrepareAssetError<Self::SourceAsset>> {
        let bind_group_data = material.bind_group_data();
        match material.as_bind_group(
            &pipeline.ui_layout.clone(),
            render_device,
            pipeline_cache,
            material_param,
        ) {
            Ok(prepared) => Ok(PreparedUiMaterial {
                bindings: prepared.bindings,
                bind_group: prepared.bind_group,
                key: bind_group_data,
            }),
            Err(AsBindGroupError::RetryNextUpdate) => {
                Err(PrepareAssetError::RetryNextUpdate(material))
            }
            Err(other) => Err(PrepareAssetError::AsBindGroupError(other)),
        }
    }
}

pub fn queue_ui_material_nodes<M: UiMaterial>(
    extracted: Res<ExtractedUiMaterialNodes<M>>,
    draw_functions: Res<DrawFunctions<TransparentUi>>,
    pipeline: Res<UiMaterialPipeline<M>>,
    mut meta: ResMut<UiMaterialMeta<M>>,
    mut pipelines: ResMut<SpecializedRenderPipelines<UiMaterialPipeline<M>>>,
    pipeline_cache: Res<PipelineCache>,
    render_materials: Res<RenderAssets<PreparedUiMaterial<M>>>,
    mut phases: ResMut<ViewSortedRenderPhases<TransparentUi>>,
    render_views: Query<(Entity, &UiCameraView), With<ExtractedView>>,
    camera_views: Query<&ExtractedView>,
) where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    let draw_function = draw_functions.read().id::<DrawUiMaterial<M>>();
    let mut active_cameras = HashSet::new();
    let mut invalidated_cameras = HashSet::new();
    for (camera_entity, ui_camera_view) in &render_views {
        let Ok(view) = camera_views.get(ui_camera_view.0) else {
            continue;
        };
        let state = UiMaterialCameraState {
            retained_view_entity: view.retained_view_entity,
            target_format: view.target_format,
        };
        active_cameras.insert(camera_entity);
        if meta.camera_states.insert(camera_entity, state) != Some(state) {
            invalidated_cameras.insert(camera_entity);
        }
    }
    for removed in &extracted.removed {
        let Some(camera_state) = meta.camera_states.get(&removed.camera_entity) else {
            continue;
        };
        if let Some(phase) = phases.get_mut(&camera_state.retained_view_entity) {
            phase.remove(removed.render_entity, removed.main_entity);
        }
    }

    let mut current_material_keys = HashMap::default();
    let mut changed_materials = HashSet::new();
    for (_, nodes) in extracted.uinodes.values() {
        for node in nodes.values() {
            if current_material_keys.contains_key(&node.material) {
                continue;
            }
            match render_materials.get(node.material) {
                Some(material) => {
                    if meta.material_keys.get(&node.material) != Some(&material.key) {
                        changed_materials.insert(node.material);
                    }
                    current_material_keys.insert(node.material, material.key.clone());
                }
                None => {
                    if meta.material_keys.contains_key(&node.material) {
                        changed_materials.insert(node.material);
                    }
                }
            }
        }
    }
    meta.material_keys = current_material_keys;

    let mut dirty = extracted.changed.clone();
    dirty.extend(meta.pending_queue.iter().copied());
    if !invalidated_cameras.is_empty() {
        dirty.extend(
            extracted
                .uinodes
                .iter()
                .filter_map(|(main_entity, (camera_entity, _))| {
                    invalidated_cameras
                        .contains(camera_entity)
                        .then_some(*main_entity)
                }),
        );
    }
    if !changed_materials.is_empty() {
        dirty.extend(
            extracted
                .uinodes
                .iter()
                .filter_map(|(main_entity, (_, nodes))| {
                    nodes
                        .values()
                        .any(|node| changed_materials.contains(&node.material))
                        .then_some(*main_entity)
                }),
        );
    }

    for main_entity in dirty {
        let Some((camera_entity, nodes)) = extracted.uinodes.get(&main_entity) else {
            meta.pending_queue.remove(&main_entity);
            continue;
        };
        let Some(camera_state) = meta.camera_states.get(camera_entity) else {
            continue;
        };
        let Some(phase) = phases.get_mut(&camera_state.retained_view_entity) else {
            continue;
        };
        let mut pending = false;
        for (render_entity, node) in nodes {
            let Some(material) = render_materials.get(node.material) else {
                pending = true;
                phase.remove(*render_entity, main_entity);
                continue;
            };
            let item_pipeline = pipelines.specialize(
                &pipeline_cache,
                &pipeline,
                UiMaterialKey {
                    target_format: camera_state.target_format,
                    bind_group_data: material.key.clone(),
                },
            );
            phase.add_retained(TransparentUi {
                draw_function,
                pipeline: item_pipeline,
                entity: (*render_entity, main_entity),
                sort_key: FloatOrd(node.stack_index as f32 + M::stack_z_offset()),
                batch_range: 0..0,
                extra_index: PhaseItemExtraIndex::None,
                indexed: false,
                batch_index: None,
            });
        }
        if pending {
            meta.pending_queue.insert(main_entity);
        } else {
            meta.pending_queue.remove(&main_entity);
        }
    }
    meta.camera_states
        .retain(|camera, _| active_cameras.contains(camera));
}

#[cfg(test)]
mod tests {
    use core::mem::size_of;

    use super::*;
    use crate::retained::{preprocess_wgsl_for_test, validate_wgsl_for_test};
    use bevy_asset::uuid::Uuid;
    use bevy_reflect::TypePath;

    #[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
    struct TestMaterial {}

    impl UiMaterial for TestMaterial {}

    #[test]
    fn material_vertex_is_atomic_sparse_buffer_compatible() {
        assert_eq!(size_of::<UiMaterialVertex>(), 19 * size_of::<u32>());
        assert!(size_of::<UiMaterialVertex>() <= 32 * size_of::<u32>());
    }

    #[test]
    fn retained_material_vertices_preserve_legacy_triangle_order_and_clipping() {
        let node = ExtractedUiMaterialNode::<TestMaterial> {
            stack_index: 0,
            transform: Affine2::from_translation(Vec2::new(50.0, 40.0)),
            rect: Rect::from_center_size(Vec2::ZERO, Vec2::new(100.0, 80.0)),
            border: BorderRect::ZERO,
            border_radius: [[0.0; 4]; 2],
            material: AssetId::Uuid {
                uuid: Uuid::from_u128(42),
            },
            clip: Some(Rect::new(10.0, 20.0, 90.0, 70.0)),
        };
        let vertices = generate_material_vertices(&node).unwrap();
        let expected = [
            Vec2::new(10.0, 20.0),
            Vec2::new(90.0, 70.0),
            Vec2::new(10.0, 70.0),
            Vec2::new(10.0, 20.0),
            Vec2::new(90.0, 20.0),
            Vec2::new(90.0, 70.0),
        ];
        for (vertex, expected) in vertices.iter().zip(expected) {
            assert_eq!(Vec2::from_slice(&vertex.position[..2]), expected);
        }
    }

    #[test]
    fn public_material_vertex_output_retains_the_legacy_fragment_abi() {
        let output = preprocess_wgsl_for_test(include_str!("ui_vertex_output.wgsl"), &[], "");
        let custom_fragment = preprocess_wgsl_for_test(
            include_str!("../../../assets/shaders/custom_ui_material.wgsl"),
            &[],
            "",
        );
        assert!(output.contains("@location(2) border_radius: vec4<f32>"));
        assert!(output.contains("@location(3) @interpolate(flat) size: vec2<f32>"));
        validate_wgsl_for_test(
            "legacy custom UI material fragment ABI",
            &format!("{output}\n{custom_fragment}"),
        );
    }
}
