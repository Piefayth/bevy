//! Persistent preparation and instanced drawing for ordinary UI nodes.

use crate::scene::{PaintId, RetainedDraw, RetainedDrawItem, RetainedUiScene};
use bevy::{
    app::SubApp,
    asset::{load_embedded_asset, AssetEvent, AssetId, AssetServer, Handle},
    color::ColorToComponents,
    ecs::{
        entity::Entity,
        query::With,
        schedule::IntoScheduleConfigs,
        system::{lifetimeless::*, Commands, Res, ResMut, SystemParamItem},
    },
    image::Image,
    math::FloatOrd,
    mesh::{VertexBufferLayout, VertexFormat},
    platform::collections::HashMap,
    render::{
        render_asset::RenderAssets,
        render_phase::{
            AddRenderCommand, DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand,
            RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewSortedRenderPhases,
        },
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BlendState, BufferUsages,
            ColorTargetState, ColorWrites, FragmentState, PipelineCache, RawBufferVec,
            RenderPipelineDescriptor, SpecializedRenderPipeline, SpecializedRenderPipelines,
            VertexState, VertexStepMode,
        },
        renderer::{RenderDevice, RenderQueue},
        sync_world::MainEntity,
        texture::GpuImage,
        view::{ExtractedView, ViewUniformOffset, ViewUniforms},
        GpuResourceAppExt,
    },
    sprite_render::SpriteAssetEvents,
    ui_render::{
        init_ui_pipeline, shader_flags, TransparentUi, UiAntiAlias, UiCameraView, UiPipeline,
        UiPipelineKey,
    },
};
use bytemuck::{Pod, Zeroable};

pub(crate) enum RetainedCoreSource<'a> {
    Prepared(&'a [GpuUiInstance]),
    Deferred(PaintId),
}

#[derive(Clone)]
pub(crate) struct RetainedCoreRun {
    pub(crate) render_entity: Entity,
    pub(crate) main_entity: MainEntity,
    pub(crate) camera: Entity,
    pub(crate) z_order: f32,
    pub(crate) image: AssetId<Image>,
    entries: core::ops::Range<usize>,
    instances: core::ops::Range<usize>,
    items: u32,
}

#[derive(Clone, Copy)]
enum RetainedCoreEntry {
    Prepared([usize; 2]),
    Deferred {
        id: PaintId,
        border_flags: Option<u32>,
    },
}

#[derive(bevy::prelude::Resource, Default)]
pub(crate) struct RetainedCoreRuns {
    runs: Vec<RetainedCoreRun>,
    entries: Vec<RetainedCoreEntry>,
    instances: Vec<GpuUiInstance>,
    has_deferred: bool,
}

impl RetainedCoreRuns {
    pub(crate) fn clear(&mut self) {
        self.runs.clear();
        self.entries.clear();
        self.instances.clear();
        self.has_deferred = false;
    }

    pub(crate) fn image(&self, run: usize) -> AssetId<Image> {
        self.runs[run].image
    }

    #[expect(clippy::too_many_arguments, reason = "one run has one phase identity")]
    pub(crate) fn start(
        &mut self,
        render_entity: Entity,
        main_entity: MainEntity,
        camera: Entity,
        z_order: f32,
        image: AssetId<Image>,
        source: RetainedCoreSource<'_>,
        border_flags: Option<u32>,
    ) -> usize {
        let entries = self.entries.len()..self.entries.len();
        let instances = self.instances.len()..self.instances.len();
        self.runs.push(RetainedCoreRun {
            render_entity,
            main_entity,
            camera,
            z_order,
            image,
            entries,
            instances,
            items: 0,
        });
        let index = self.runs.len() - 1;
        self.push(index, source, border_flags);
        index
    }

    pub(crate) fn push(
        &mut self,
        run: usize,
        source: RetainedCoreSource<'_>,
        border_flags: Option<u32>,
    ) {
        self.runs[run].items = self.runs[run]
            .items
            .checked_add(1)
            .expect("retained UI run item count exceeds u32");
        match source {
            RetainedCoreSource::Prepared(instances) => {
                let start = self.instances.len();
                self.instances.extend(instances.iter().map(|&instance| {
                    let mut instance = instance;
                    if let Some(flags) = border_flags {
                        instance.set_border_flags(flags);
                    }
                    instance
                }));
                let end = self.instances.len();
                self.entries.push(RetainedCoreEntry::Prepared([start, end]));
                self.runs[run].entries.end += 1;
                self.runs[run].instances.end = end;
            }
            RetainedCoreSource::Deferred(id) => {
                self.entries
                    .push(RetainedCoreEntry::Deferred { id, border_flags });
                self.runs[run].entries.end += 1;
                self.has_deferred = true;
            }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuUiInstance {
    transform: [f32; 4],
    translation: [f32; 2],
    clip: [f32; 4],
    uv_rect: [f32; 4],
    color: [f32; 4],
    radius: [f32; 4],
    border: [f32; 4],
    size_inverse_atlas: [f32; 4],
    metadata: [u32; 2],
}

impl GpuUiInstance {
    fn set_border_flags(&mut self, flags: u32) {
        self.metadata[0] &= !shader_flags::BORDER_ALL;
        self.metadata[0] |= flags;
    }

    pub(crate) fn set_color(&mut self, color: bevy::color::LinearRgba) {
        self.color = color.to_f32_array();
    }
}

struct CoreBatch {
    range: core::ops::Range<u32>,
    image: AssetId<Image>,
    items: u32,
}

#[derive(bevy::prelude::Resource)]
pub(crate) struct RetainedCore {
    instances: RawBufferVec<GpuUiInstance>,
    batches: HashMap<Entity, CoreBatch>,
    image_bind_groups: HashMap<AssetId<Image>, BindGroup>,
    view_bind_group: Option<BindGroup>,
}

impl Default for RetainedCore {
    fn default() -> Self {
        Self {
            instances: RawBufferVec::new(BufferUsages::VERTEX),
            batches: HashMap::default(),
            image_bind_groups: HashMap::default(),
            view_bind_group: None,
        }
    }
}

impl RetainedCore {
    pub(crate) fn batch_range(&self, entity: Entity) -> Option<core::ops::Range<u32>> {
        self.batches.get(&entity).map(|batch| batch.range.clone())
    }

    pub(crate) fn item_count(&self, entity: Entity) -> Option<u32> {
        self.batches.get(&entity).map(|batch| batch.items)
    }
}

#[derive(bevy::prelude::Resource)]
pub(crate) struct RetainedCorePipeline {
    view_layout: BindGroupLayoutDescriptor,
    mask_layout: BindGroupLayoutDescriptor,
    image_layout: BindGroupLayoutDescriptor,
    shader: Handle<bevy::shader::Shader>,
}

impl SpecializedRenderPipeline for RetainedCorePipeline {
    type Key = UiPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let instance_layout = VertexBufferLayout::from_vertex_formats(
            VertexStepMode::Instance,
            vec![
                VertexFormat::Float32x4,
                VertexFormat::Float32x2,
                VertexFormat::Float32x4,
                VertexFormat::Float32x4,
                VertexFormat::Float32x4,
                VertexFormat::Float32x4,
                VertexFormat::Float32x4,
                VertexFormat::Float32x4,
                VertexFormat::Uint32x2,
            ],
        );
        let shader_defs: Vec<_> = key
            .anti_alias
            .then(|| "ANTI_ALIAS".into())
            .into_iter()
            .collect();
        RenderPipelineDescriptor {
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: shader_defs.clone(),
                buffers: vec![instance_layout],
                ..Default::default()
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs,
                targets: vec![Some(ColorTargetState {
                    format: key.target_format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
                ..Default::default()
            }),
            layout: vec![
                self.view_layout.clone(),
                self.mask_layout.clone(),
                self.image_layout.clone(),
            ],
            label: Some("retained_ui_core_pipeline".into()),
            ..Default::default()
        }
    }
}

pub(crate) fn init_retained_core_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    ui_pipeline: Res<UiPipeline>,
) {
    commands.insert_resource(RetainedCorePipeline {
        view_layout: ui_pipeline.view_layout.clone(),
        mask_layout: crate::mask::layout(),
        image_layout: ui_pipeline.image_layout.clone(),
        shader: load_embedded_asset!(asset_server.as_ref(), "core.wgsl"),
    });
}

pub(crate) fn queue_retained_core(
    runs: Res<RetainedCoreRuns>,
    pipeline: Res<RetainedCorePipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<RetainedCorePipeline>>,
    mut phases: ResMut<ViewSortedRenderPhases<TransparentUi>>,
    render_views: bevy::ecs::system::Query<
        (&UiCameraView, Option<&UiAntiAlias>),
        With<ExtractedView>,
    >,
    camera_views: bevy::ecs::system::Query<&ExtractedView>,
    pipeline_cache: Res<PipelineCache>,
    draw_functions: Res<DrawFunctions<TransparentUi>>,
) {
    let draw_function = draw_functions.read().id::<DrawRetainedCore>();
    for (index, run) in runs.runs.iter().enumerate() {
        let Ok((default_camera_view, anti_alias)) = render_views.get(run.camera) else {
            continue;
        };
        let Ok(view) = camera_views.get(default_camera_view.0) else {
            continue;
        };
        let Some(phase) = phases.get_mut(&view.retained_view_entity) else {
            continue;
        };
        let pipeline = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            UiPipelineKey {
                target_format: view.target_format,
                anti_alias: matches!(anti_alias, None | Some(UiAntiAlias::On)),
            },
        );
        phase.add_transient(TransparentUi {
            draw_function,
            pipeline,
            entity: (run.render_entity, run.main_entity),
            sort_key: FloatOrd(run.z_order),
            index,
            batch_range: 0..1,
            extra_index: PhaseItemExtraIndex::None,
            indexed: false,
        });
    }
}

pub(crate) fn prepare_retained_core(
    mut runs: ResMut<RetainedCoreRuns>,
    state: Res<RetainedUiScene>,
    mut core: ResMut<RetainedCore>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    events: Res<SpriteAssetEvents>,
    view_uniforms: Res<ViewUniforms>,
    pipeline: Res<RetainedCorePipeline>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    for event in &events.images {
        if let AssetEvent::Modified { id } | AssetEvent::Removed { id } = event {
            core.image_bind_groups.remove(id);
        }
    }

    core.batches.clear();
    if runs.has_deferred {
        core.instances.clear();
        let surfaces = state.lock();
        for run in &runs.runs {
            let start = u32::try_from(core.instances.len())
                .expect("retained UI instance count exceeds u32");
            for &entry in &runs.entries[run.entries.clone()] {
                match entry {
                    RetainedCoreEntry::Prepared([start, end]) => {
                        core.instances
                            .values_mut()
                            .extend_from_slice(&runs.instances[start..end]);
                    }
                    RetainedCoreEntry::Deferred { id, border_flags } => {
                        let Some(draw) = surfaces.core_draw(id) else {
                            continue;
                        };
                        prepare_instances(draw, border_flags, &gpu_images, &mut core.instances);
                    }
                }
            }
            let end = u32::try_from(core.instances.len())
                .expect("retained UI instance count exceeds u32");
            if start != end {
                core.batches.insert(
                    run.render_entity,
                    CoreBatch {
                        range: start..end,
                        image: run.image,
                        items: run.items,
                    },
                );
            }
        }
    } else {
        core.instances.clear();
        core::mem::swap(core.instances.values_mut(), &mut runs.instances);
        for run in &runs.runs {
            let start =
                u32::try_from(run.instances.start).expect("retained UI instance count exceeds u32");
            let end =
                u32::try_from(run.instances.end).expect("retained UI instance count exceeds u32");
            core.batches.insert(
                run.render_entity,
                CoreBatch {
                    range: start..end,
                    image: run.image,
                    items: run.items,
                },
            );
        }
    }
    for run in &runs.runs {
        if let Some(image) = gpu_images.get(run.image) {
            core.image_bind_groups.entry(run.image).or_insert_with(|| {
                render_device.create_bind_group(
                    "retained_ui_core_image_bind_group",
                    &pipeline_cache.get_bind_group_layout(&pipeline.image_layout),
                    &BindGroupEntries::sequential((&image.texture_view, &image.sampler)),
                )
            });
        }
    }

    if core.instances.is_empty() {
        core.view_bind_group = None;
        return;
    }
    core.instances.write_buffer(&render_device, &render_queue);
    core.view_bind_group = view_uniforms.uniforms.binding().map(|binding| {
        render_device.create_bind_group(
            "retained_ui_core_view_bind_group",
            &pipeline_cache.get_bind_group_layout(&pipeline.view_layout),
            &BindGroupEntries::single(binding),
        )
    });
}

pub(crate) fn prepare_persistent_instance(draw: &RetainedDraw) -> Option<GpuUiInstance> {
    let RetainedDrawItem::Node(node) = draw.item else {
        return None;
    };
    let atlas_extent = if draw.image == AssetId::default() {
        bevy::math::Vec2::ONE
    } else {
        let image_extent = node.image_extent?;
        node.atlas_scaling
            .map(|scaling| image_extent * scaling)
            .unwrap_or(node.rect.max)
    };
    Some(prepare_node_instance(draw, atlas_extent))
}

fn prepare_instance(
    draw: &RetainedDraw,
    gpu_images: &RenderAssets<GpuImage>,
) -> Option<GpuUiInstance> {
    let RetainedDrawItem::Node(node) = draw.item else {
        return None;
    };
    let image = gpu_images.get(draw.image)?;
    let atlas_extent = node
        .atlas_scaling
        .map(|scaling| image.size_2d().as_vec2() * scaling)
        .unwrap_or(node.rect.max);
    Some(prepare_node_instance(draw, atlas_extent))
}

fn prepare_instances(
    draw: &RetainedDraw,
    border_flags: Option<u32>,
    gpu_images: &RenderAssets<GpuImage>,
    instances: &mut RawBufferVec<GpuUiInstance>,
) {
    match &draw.item {
        RetainedDrawItem::Node(_) => {
            if let Some(mut instance) = prepare_instance(draw, gpu_images) {
                if let Some(flags) = border_flags {
                    instance.set_border_flags(flags);
                }
                instances.push(instance);
            }
        }
        RetainedDrawItem::Glyphs(glyphs) => {
            let Some(image) = gpu_images.get(draw.image) else {
                return;
            };
            for &glyph in glyphs.iter() {
                instances.push(prepare_glyph_instance(
                    draw,
                    glyph,
                    image.size_2d().as_vec2(),
                ));
            }
        }
        _ => unreachable!("persistent core runs contain only nodes and glyphs"),
    }
}

pub(crate) fn prepare_glyph_instance(
    draw: &RetainedDraw,
    glyph: crate::scene::RetainedGlyph,
    atlas_extent: bevy::math::Vec2,
) -> GpuUiInstance {
    let transform = draw.transform.to_cols_array();
    let translation = draw.transform.transform_point2(glyph.translation());
    let rect = glyph.rect();
    let clip = draw.clip.unwrap_or_default();
    GpuUiInstance {
        transform: [transform[0], transform[1], transform[2], transform[3]],
        translation: translation.to_array(),
        clip: [clip.min.x, clip.min.y, clip.max.x, clip.max.y],
        uv_rect: [
            rect.min.x / atlas_extent.x,
            rect.min.y / atlas_extent.y,
            rect.max.x / atlas_extent.x,
            rect.max.y / atlas_extent.y,
        ],
        color: glyph.color().to_f32_array(),
        radius: [0.0; 4],
        border: [0.0; 4],
        size_inverse_atlas: [
            rect.width(),
            rect.height(),
            atlas_extent.x.recip(),
            atlas_extent.y.recip(),
        ],
        metadata: [shader_flags::TEXTURED, u32::from(draw.clip.is_some()) | 2],
    }
}

fn prepare_node_instance(draw: &RetainedDraw, atlas_extent: bevy::math::Vec2) -> GpuUiInstance {
    let RetainedDrawItem::Node(node) = draw.item else {
        unreachable!("persistent core preparation only accepts ordinary nodes")
    };
    let textured = draw.image != AssetId::default();
    let mut uv_min = node.rect.min / atlas_extent;
    let mut uv_max = node.rect.max / atlas_extent;
    let mut inverse_atlas = atlas_extent.recip();
    if node.flip_x {
        core::mem::swap(&mut uv_min.x, &mut uv_max.x);
        inverse_atlas.x *= -1.0;
    }
    if node.flip_y {
        core::mem::swap(&mut uv_min.y, &mut uv_max.y);
        inverse_atlas.y *= -1.0;
    }
    let mut flags = if textured { shader_flags::TEXTURED } else { 0 };
    match node.node_type {
        bevy::ui_render::NodeType::Border(border_flags) => flags |= border_flags,
        bevy::ui_render::NodeType::Inverted => flags |= shader_flags::INVERT,
        bevy::ui_render::NodeType::Rect => {}
    }
    let transform = draw.transform.to_cols_array();
    let clip = draw.clip.unwrap_or_default();
    GpuUiInstance {
        transform: [transform[0], transform[1], transform[2], transform[3]],
        translation: [transform[4], transform[5]],
        clip: [clip.min.x, clip.min.y, clip.max.x, clip.max.y],
        uv_rect: [uv_min.x, uv_min.y, uv_max.x, uv_max.y],
        color: node.color.to_f32_array(),
        radius: node.border_radius.into(),
        border: [
            node.border.min_inset.x,
            node.border.min_inset.y,
            node.border.max_inset.x,
            node.border.max_inset.y,
        ],
        size_inverse_atlas: [
            node.rect.width(),
            node.rect.height(),
            inverse_atlas.x,
            inverse_atlas.y,
        ],
        metadata: [flags, u32::from(draw.clip.is_some())],
    }
}

pub(crate) type DrawRetainedCore = (
    SetItemPipeline,
    SetRetainedCoreViewBindGroup<0>,
    SetRetainedCoreImageBindGroup<2>,
    DrawRetainedCoreInstances,
);

pub(crate) struct SetRetainedCoreViewBindGroup<const I: usize>;

impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetRetainedCoreViewBindGroup<I> {
    type Param = SRes<RetainedCore>;
    type ViewQuery = Read<ViewUniformOffset>;
    type ItemQuery = ();

    fn render<'w>(
        _item: &P,
        view_uniform: &'w ViewUniformOffset,
        _entity: Option<()>,
        core: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(bind_group) = core.into_inner().view_bind_group.as_ref() else {
            return RenderCommandResult::Failure("retained UI view bind group is unavailable");
        };
        pass.set_bind_group(I, bind_group, &[view_uniform.offset]);
        RenderCommandResult::Success
    }
}

pub(crate) struct SetRetainedCoreImageBindGroup<const I: usize>;

impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetRetainedCoreImageBindGroup<I> {
    type Param = SRes<RetainedCore>;
    type ViewQuery = ();
    type ItemQuery = ();

    fn render<'w>(
        item: &P,
        _view: (),
        _entity: Option<()>,
        core: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let core = core.into_inner();
        let Some(batch) = core.batches.get(&item.entity()) else {
            return RenderCommandResult::Skip;
        };
        let Some(bind_group) = core.image_bind_groups.get(&batch.image) else {
            return RenderCommandResult::Failure("retained UI image bind group is unavailable");
        };
        pass.set_bind_group(I, bind_group, &[]);
        RenderCommandResult::Success
    }
}

pub(crate) struct DrawRetainedCoreInstances;

impl<P: PhaseItem> RenderCommand<P> for DrawRetainedCoreInstances {
    type Param = SRes<RetainedCore>;
    type ViewQuery = ();
    type ItemQuery = ();

    fn render<'w>(
        item: &P,
        _view: (),
        _entity: Option<()>,
        core: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let core = core.into_inner();
        let Some(batch) = core.batches.get(&item.entity()) else {
            return RenderCommandResult::Skip;
        };
        let Some(instances) = core.instances.buffer() else {
            return RenderCommandResult::Failure("retained UI instance buffer is unavailable");
        };
        pass.set_vertex_buffer(0, instances.slice(..));
        pass.draw(0..6, batch.range.clone());
        RenderCommandResult::Success
    }
}

pub(crate) fn register_retained_core(app: &mut SubApp) {
    app.init_resource::<RetainedCore>()
        .init_resource::<RetainedCoreRuns>()
        .init_gpu_resource::<SpecializedRenderPipelines<RetainedCorePipeline>>()
        .add_render_command::<TransparentUi, DrawRetainedCore>()
        .add_systems(
            bevy::render::RenderStartup,
            init_retained_core_pipeline.after(init_ui_pipeline),
        );
}
