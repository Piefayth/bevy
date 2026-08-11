//! Persistent UI layer and composition.

use crate::background::extract_retained_backgrounds;
use crate::border::extract_retained_borders;
use crate::image::{extract_retained_images, resolve_ready_images, RetainedImageDependencies};
use crate::scene::{
    cleanup_retained_ui, replay_retained_ui, RetainedItem, RetainedItems, RetainedUiPaintCounters,
    RetainedUiScene,
};
use crate::text::{extract_retained_text, resolve_ready_text_atlases, RetainedTextDependencies};
use crate::{damage::exact_union, PhysicalRect, RepairPlan};
use alloc::collections::VecDeque;
use bevy::{
    asset::{embedded_asset, load_embedded_asset, AssetServer},
    core_pipeline::{
        upscaling::upscaling, Core2d, Core2dSystems, Core3d, Core3dSystems, FullscreenShader,
    },
    ecs::{
        query::With,
        schedule::IntoScheduleConfigs,
        system::{Commands, Query, Res, ResMut},
    },
    log::error,
    math::{FloatOrd, UVec2},
    prelude::{App, Plugin, Resource, World},
    render::{
        camera::ExtractedCamera,
        render_asset::RenderAssets,
        render_phase::{DrawFunctions, PhaseItem, PhaseItemExtraIndex, ViewSortedRenderPhases},
        render_resource::{
            binding_types::texture_2d, BindGroup, BindGroupEntries, BindGroupLayoutDescriptor,
            BindGroupLayoutEntries, CachedRenderPipelineId, ColorTargetState, ColorWrites,
            Extent3d, FragmentState, LoadOp, Operations, Origin3d, PipelineCache,
            RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor,
            ShaderStages, SpecializedRenderPipeline, SpecializedRenderPipelines, StoreOp, Texture,
            TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
            TextureView, TextureViewDescriptor,
        },
        renderer::{RenderContext, RenderDevice, ViewQuery},
        texture::GpuImage,
        view::{ExtractedView, RetainedViewEntity, ViewTarget},
        ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
    },
    ui_render::{
        DrawUiItem, ExtractedUiNodes, RenderUiSystems, TransparentUi, UiAntiAlias, UiCameraView,
        UiItemBatch, UiPipeline, UiPipelineKey, UiViewTarget,
    },
};
use core::{
    ops::Range,
    sync::atomic::{AtomicU64, Ordering},
};
use std::{
    collections::{HashMap, HashSet},
    sync::{Mutex, PoisonError},
};

/// Adds a persistent UI layer while reusing Bevy's public UI phase and draw commands.
#[derive(Default)]
pub struct RetainedUiRenderPlugin;

impl Plugin for RetainedUiRenderPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "composite.wgsl");

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<LayerSurfaces>()
            .init_resource::<RetainedUiLayerCounters>()
            .init_resource::<RetainedUiScene>()
            .init_resource::<RetainedImageDependencies>()
            .init_resource::<RetainedTextDependencies>()
            .init_resource::<RetainedItems>()
            .init_resource::<RetainedUiPaintCounters>()
            .add_systems(
                ExtractSchedule,
                extract_retained_backgrounds.in_set(RenderUiSystems::ExtractBackgrounds),
            )
            .add_systems(
                ExtractSchedule,
                extract_retained_images.in_set(RenderUiSystems::ExtractImages),
            )
            .add_systems(
                ExtractSchedule,
                extract_retained_borders.in_set(RenderUiSystems::ExtractBorders),
            )
            .add_systems(
                ExtractSchedule,
                extract_retained_text.in_set(RenderUiSystems::ExtractText),
            )
            .add_systems(
                ExtractSchedule,
                replay_retained_ui.after(RenderUiSystems::ExtractDebug),
            )
            .add_systems(
                Render,
                (cleanup_retained_ui, cleanup_layer_surfaces)
                    .chain()
                    .in_set(RenderSystems::PrepareResources),
            )
            .add_systems(
                Render,
                (resolve_ready_images, resolve_ready_text_atlases)
                    .in_set(RenderSystems::PrepareResources),
            )
            .add_systems(Render, queue_retained_uinodes.in_set(RenderSystems::Queue))
            .add_systems(RenderStartup, init_composite_pipeline)
            .add_systems(
                Core2d,
                retained_ui_pass
                    .after(Core2dSystems::PostProcess)
                    .before(upscaling),
            )
            .add_systems(
                Core3d,
                retained_ui_pass
                    .after(Core3dSystems::PostProcess)
                    .before(upscaling),
            );
    }
}

fn queue_retained_uinodes(
    extracted_uinodes: Res<ExtractedUiNodes>,
    ui_pipeline: Res<UiPipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<UiPipeline>>,
    mut phases: ResMut<ViewSortedRenderPhases<TransparentUi>>,
    render_views: Query<(&UiCameraView, Option<&UiAntiAlias>), With<ExtractedView>>,
    camera_views: Query<&ExtractedView>,
    pipeline_cache: Res<PipelineCache>,
    draw_functions: Res<DrawFunctions<TransparentUi>>,
) {
    let draw_function = draw_functions.read().id::<DrawUiItem>();
    for (index, node) in extracted_uinodes.uinodes.iter().enumerate() {
        let Ok((default_camera_view, ui_anti_alias)) =
            render_views.get(node.extracted_camera_entity)
        else {
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
            &ui_pipeline,
            UiPipelineKey {
                target_format: view.target_format,
                anti_alias: matches!(ui_anti_alias, None | Some(UiAntiAlias::On)),
            },
        );
        phase.add_transient(TransparentUi {
            draw_function,
            pipeline,
            entity: (node.render_entity, node.main_entity),
            sort_key: FloatOrd(node.z_order),
            index,
            batch_range: 0..0,
            extra_index: PhaseItemExtraIndex::None,
            indexed: true,
        });
    }
}

/// Deterministic retained-layer work accumulated by the render world.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RetainedUiLayerWork {
    /// Persistent layer surfaces allocated or replaced.
    pub surfaces_created: u64,
    /// Atomic UI-layer repair transactions encoded.
    pub repairs: u64,
    /// Physical pixels cleared and repainted across exact repair regions.
    pub repair_pixels: u64,
    /// Individual paint items replayed across all repair regions.
    pub items_replayed: u64,
    /// Layer composites encoded over world output.
    pub composites: u64,
    /// Scissored texture draws issued by layer composites.
    pub composite_draws: u64,
    /// Physical pixels covered by exact layer-content bounds.
    pub composite_pixels: u64,
}

/// Atomic render-world counters for retained-layer acceptance tests and diagnostics.
#[derive(Resource, Default)]
pub struct RetainedUiLayerCounters {
    surfaces_created: AtomicU64,
    repairs: AtomicU64,
    repair_pixels: AtomicU64,
    items_replayed: AtomicU64,
    composites: AtomicU64,
    composite_draws: AtomicU64,
    composite_pixels: AtomicU64,
}

impl RetainedUiLayerCounters {
    /// Returns a consistent-enough monotonic snapshot for diagnostics and tests.
    pub fn snapshot(&self) -> RetainedUiLayerWork {
        RetainedUiLayerWork {
            surfaces_created: self.surfaces_created.load(Ordering::Relaxed),
            repairs: self.repairs.load(Ordering::Relaxed),
            repair_pixels: self.repair_pixels.load(Ordering::Relaxed),
            items_replayed: self.items_replayed.load(Ordering::Relaxed),
            composites: self.composites.load(Ordering::Relaxed),
            composite_draws: self.composite_draws.load(Ordering::Relaxed),
            composite_pixels: self.composite_pixels.load(Ordering::Relaxed),
        }
    }
}

#[derive(Resource)]
struct CompositePipeline {
    layout: BindGroupLayoutDescriptor,
    shader: bevy::asset::Handle<bevy::shader::Shader>,
    vertex: bevy::render::render_resource::VertexState,
    pipelines: Mutex<HashMap<TextureFormat, CachedRenderPipelineId>>,
    wipe_pipelines: Mutex<HashMap<TextureFormat, CachedRenderPipelineId>>,
}

impl SpecializedRenderPipeline for CompositePipeline {
    type Key = TextureFormat;

    fn specialize(&self, format: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("retained_ui_composite_pipeline".into()),
            layout: vec![self.layout.clone()],
            vertex: self.vertex.clone(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    format,
                    blend: Some(
                        bevy::render::render_resource::BlendState::PREMULTIPLIED_ALPHA_BLENDING,
                    ),
                    write_mask: ColorWrites::ALL,
                })],
                ..Default::default()
            }),
            ..Default::default()
        }
    }
}

impl CompositePipeline {
    fn wipe_pipeline(
        &self,
        format: TextureFormat,
        pipeline_cache: &PipelineCache,
    ) -> CachedRenderPipelineId {
        *self
            .wipe_pipelines
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .entry(format)
            .or_insert_with(|| {
                pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
                    label: Some("retained_ui_wipe_pipeline".into()),
                    layout: Vec::new(),
                    vertex: self.vertex.clone(),
                    fragment: Some(FragmentState {
                        shader: self.shader.clone(),
                        entry_point: Some("wipe".into()),
                        targets: vec![Some(ColorTargetState {
                            format,
                            blend: None,
                            write_mask: ColorWrites::ALL,
                        })],
                        ..Default::default()
                    }),
                    ..Default::default()
                })
            })
    }
}

fn init_composite_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    fullscreen_shader: Res<FullscreenShader>,
) {
    commands.insert_resource(CompositePipeline {
        layout: BindGroupLayoutDescriptor::new(
            "retained_ui_composite_layout",
            &BindGroupLayoutEntries::single(
                ShaderStages::FRAGMENT,
                texture_2d(TextureSampleType::Float { filterable: false }),
            ),
        ),
        shader: load_embedded_asset!(asset_server.as_ref(), "composite.wgsl"),
        vertex: fullscreen_shader.to_vertex_state(),
        pipelines: Mutex::new(HashMap::new()),
        wipe_pipelines: Mutex::new(HashMap::new()),
    });
}

struct LayerSlot {
    texture: Texture,
    view: TextureView,
    bind_group: BindGroup,
    initialized: bool,
    generation: u64,
}

struct CommittedDamage {
    generation: u64,
    regions: Vec<PhysicalRect>,
}

struct LayerSurface {
    size: UVec2,
    format: TextureFormat,
    slots: [LayerSlot; 2],
    active: usize,
    has_content: bool,
    composite_regions: Vec<PhysicalRect>,
    generation: u64,
    history: VecDeque<CommittedDamage>,
}

impl LayerSurface {
    fn new(
        render_device: &RenderDevice,
        pipeline_cache: &PipelineCache,
        pipeline: &CompositePipeline,
        size: UVec2,
        format: TextureFormat,
    ) -> Self {
        let layout = pipeline_cache.get_bind_group_layout(&pipeline.layout);
        let create_slot = || {
            let texture = render_device.create_texture(&TextureDescriptor {
                label: Some("retained_ui_layer"),
                size: Extent3d {
                    width: size.x,
                    height: size.y,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage: TextureUsages::RENDER_ATTACHMENT
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_SRC
                    | TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&TextureViewDescriptor::default());
            let bind_group = render_device.create_bind_group(
                "retained_ui_composite_bind_group",
                &layout,
                &BindGroupEntries::single(&view),
            );
            LayerSlot {
                texture,
                view,
                bind_group,
                initialized: false,
                generation: 0,
            }
        };

        Self {
            size,
            format,
            slots: [create_slot(), create_slot()],
            active: 0,
            has_content: false,
            composite_regions: Vec::new(),
            generation: 0,
            history: VecDeque::new(),
        }
    }

    fn matches(&self, size: UVec2, format: TextureFormat) -> bool {
        self.size == size && self.format == format
    }

    fn commit(
        &mut self,
        active: usize,
        regions: Vec<PhysicalRect>,
        composite_regions: Vec<PhysicalRect>,
    ) {
        self.generation += 1;
        self.active = active;
        self.has_content = !composite_regions.is_empty();
        self.composite_regions = composite_regions;
        self.slots[active].initialized = true;
        self.slots[active].generation = self.generation;
        self.history.push_back(CommittedDamage {
            generation: self.generation,
            regions,
        });
        let oldest_slot = self.slots[0].generation.min(self.slots[1].generation);
        while self
            .history
            .front()
            .is_some_and(|damage| damage.generation <= oldest_slot)
        {
            self.history.pop_front();
        }
    }
}

#[derive(Resource, Default)]
struct LayerSurfaces(Mutex<HashMap<RetainedViewEntity, LayerSurface>>);

fn cleanup_layer_surfaces(views: Query<&ExtractedView>, surfaces: Res<LayerSurfaces>) {
    let live: HashSet<_> = views.iter().map(|view| view.retained_view_entity).collect();
    surfaces
        .0
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .retain(|view, _| live.contains(view));
}

fn phase_is_ready(
    phase: &bevy::render::render_phase::SortedRenderPhase<TransparentUi>,
    pipeline_cache: &PipelineCache,
    world: &World,
) -> bool {
    if phase.items.is_empty() {
        return false;
    }
    let items = world
        .resource::<RetainedItems>()
        .0
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    let dependencies = world.resource::<RetainedImageDependencies>();
    let text_dependencies = world.resource::<RetainedTextDependencies>();
    let gpu_images = world.resource::<RenderAssets<GpuImage>>();
    for index in 0..phase.items.len() {
        let item = phase.items.get_index(index).unwrap().1;
        let Some(metadata) = items.get(&item.entity()) else {
            return false;
        };
        if world.get::<UiItemBatch>(item.entity()).is_none() {
            let unavailable_image = metadata.image
                != bevy::asset::AssetId::<bevy::image::Image>::default()
                && gpu_images.get(metadata.image).is_none()
                && !dependencies.is_pending(metadata.image)
                && !text_dependencies.is_pending(metadata.image);
            if unavailable_image {
                continue;
            }
            return false;
        }
        if pipeline_cache.get_render_pipeline(item.pipeline).is_none() {
            return false;
        }
    }
    true
}

fn target_rect(size: UVec2) -> PhysicalRect {
    PhysicalRect::from_min_max(0, 0, size.x as i32, size.y as i32)
        .expect("render targets have nonzero size")
}

fn clipped_damage(plan: Option<&RepairPlan>, size: UVec2) -> Vec<PhysicalRect> {
    let target = target_rect(size);
    match plan {
        Some(plan) => plan
            .regions()
            .iter()
            .filter_map(|region| region.intersection(target))
            .collect(),
        None => vec![target],
    }
}

fn clear_layer_slot(ctx: &mut RenderContext, slot: &LayerSlot) {
    ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("retained_ui_initialize_layer"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: &slot.view,
            depth_slice: None,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(Default::default()),
                store: StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
}

fn synchronize_pending_layer(surface: &LayerSurface, pending: usize, ctx: &mut RenderContext) {
    let pending_slot = &surface.slots[pending];
    if !pending_slot.initialized {
        clear_layer_slot(ctx, pending_slot);
    }
    if !surface.has_content {
        return;
    }

    for damage in surface
        .history
        .iter()
        .filter(|damage| damage.generation > pending_slot.generation)
    {
        for region in &damage.regions {
            let origin = Origin3d {
                x: region.min_x() as u32,
                y: region.min_y() as u32,
                z: 0,
            };
            let mut source = surface.slots[surface.active].texture.as_image_copy();
            source.origin = origin;
            let mut destination = pending_slot.texture.as_image_copy();
            destination.origin = origin;
            ctx.command_encoder().copy_texture_to_texture(
                source,
                destination,
                Extent3d {
                    width: (region.max_x() - region.min_x()) as u32,
                    height: (region.max_y() - region.min_y()) as u32,
                    depth_or_array_layers: 1,
                },
            );
        }
    }
}

fn replay_runs(
    phase: &bevy::render::render_phase::SortedRenderPhase<TransparentUi>,
    items: &HashMap<bevy::ecs::entity::Entity, RetainedItem>,
    region: PhysicalRect,
    world: &World,
) -> Vec<Range<usize>> {
    let mut runs = Vec::new();
    let mut start = None;
    for index in 0..phase.items.len() {
        let item = phase.items.get_index(index).unwrap().1;
        let intersects = world.get::<UiItemBatch>(item.entity()).is_some()
            && items.get(&item.entity()).is_none_or(|item| {
                item.coverage
                    .iter()
                    .any(|coverage| coverage.intersection(region).is_some())
            });
        match (start, intersects) {
            (None, true) => start = Some(index),
            (Some(run_start), false) => {
                runs.push(run_start..index);
                start = None;
            }
            _ => {}
        }
    }
    if let Some(run_start) = start {
        runs.push(run_start..phase.items.len());
    }
    runs
}

fn exact_composite_regions(
    phase: Option<&bevy::render::render_phase::SortedRenderPhase<TransparentUi>>,
    items: &HashMap<bevy::ecs::entity::Entity, RetainedItem>,
    size: UVec2,
    world: &World,
) -> Vec<PhysicalRect> {
    let Some(phase) = phase else {
        return Vec::new();
    };
    let target = target_rect(size);
    let mut occupied = Vec::with_capacity(phase.items.len());
    for index in 0..phase.items.len() {
        let item = phase.items.get_index(index).unwrap().1;
        if world.get::<UiItemBatch>(item.entity()).is_none() {
            continue;
        }
        let Some(item) = items.get(&item.entity()) else {
            return vec![target];
        };
        occupied.extend(
            item.coverage
                .iter()
                .filter_map(|coverage| coverage.intersection(target)),
        );
    }
    exact_union(occupied)
}

#[expect(
    clippy::too_many_arguments,
    reason = "render pass inputs are independent resources"
)]
fn retained_ui_pass(
    world: &World,
    view: ViewQuery<&UiCameraView>,
    ui_view_query: Query<(&ExtractedView, &UiViewTarget)>,
    ui_view_target_query: Query<(&ViewTarget, &ExtractedCamera)>,
    transparent_render_phases: Res<ViewSortedRenderPhases<TransparentUi>>,
    composite_pipeline: Option<Res<CompositePipeline>>,
    pipeline_cache: Res<PipelineCache>,
    surfaces: Res<LayerSurfaces>,
    counters: Res<RetainedUiLayerCounters>,
    mut ctx: RenderContext,
) {
    let Some(composite_pipeline) = composite_pipeline else {
        return;
    };
    let ui_view_entity = view.into_inner().0;
    let Ok((extracted_view, ui_view_target)) = ui_view_query.get(ui_view_entity) else {
        return;
    };
    let Ok((target, camera)) = ui_view_target_query.get(ui_view_target.0) else {
        return;
    };
    let Some(size) = camera.physical_viewport_size else {
        return;
    };

    let phase = transparent_render_phases.get(&extracted_view.retained_view_entity);
    let phase_has_items = phase.is_some_and(|phase| !phase.items.is_empty());
    let scene = world.resource::<RetainedUiScene>();
    let repair_plan = scene.repair_plan(ui_view_target.0);
    let damage_regions = clipped_damage(repair_plan.as_ref(), size);
    if let Some(plan) = repair_plan.as_ref()
        && damage_regions.is_empty()
    {
        scene.acknowledge(ui_view_target.0, plan);
    }
    let repair_requested = phase_has_items || (repair_plan.is_some() && !damage_regions.is_empty());
    let repair_ready = if phase_has_items {
        phase.is_some_and(|phase| phase_is_ready(phase, &pipeline_cache, world))
    } else {
        repair_plan.is_some()
    };

    let mut surfaces = surfaces.0.lock().unwrap_or_else(PoisonError::into_inner);
    let existing_matches = surfaces
        .get(&extracted_view.retained_view_entity)
        .is_some_and(|surface| surface.matches(size, extracted_view.target_format));
    if !existing_matches && !repair_requested {
        drop(surfaces);
        let target = target_rect(size);
        if scene.has_visible_records(ui_view_target.0, target) {
            scene.invalidate(ui_view_target.0, target);
        }
        return;
    }
    let surface = surfaces
        .entry(extracted_view.retained_view_entity)
        .or_insert_with(|| {
            counters.surfaces_created.fetch_add(1, Ordering::Relaxed);
            LayerSurface::new(
                ctx.render_device(),
                &pipeline_cache,
                &composite_pipeline,
                size,
                extracted_view.target_format,
            )
        });
    if !surface.matches(size, extracted_view.target_format) {
        *surface = LayerSurface::new(
            ctx.render_device(),
            &pipeline_cache,
            &composite_pipeline,
            size,
            extracted_view.target_format,
        );
        counters.surfaces_created.fetch_add(1, Ordering::Relaxed);
    }

    let wipe_pipeline = (repair_requested && repair_ready)
        .then(|| composite_pipeline.wipe_pipeline(extracted_view.target_format, &pipeline_cache))
        .and_then(|id| pipeline_cache.get_render_pipeline(id));
    if let Some(wipe_pipeline) = wipe_pipeline {
        let pending = 1 - surface.active;
        synchronize_pending_layer(surface, pending, &mut ctx);
        let items = world
            .resource::<RetainedItems>()
            .0
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let mut result = Ok(());
        let mut replayed = 0;
        let composite_regions = exact_composite_regions(phase, &items, size, world);
        for region in &damage_regions {
            let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some("retained_ui_repair"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &surface.slots[pending].view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Load,
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_scissor_rect(
                region.min_x() as u32,
                region.min_y() as u32,
                (region.max_x() - region.min_x()) as u32,
                (region.max_y() - region.min_y()) as u32,
            );
            pass.set_render_pipeline(wipe_pipeline);
            pass.draw(0..3, 0..1);

            if let Some(phase) = phase.filter(|phase| !phase.items.is_empty()) {
                for run in replay_runs(phase, &items, *region, world) {
                    replayed += run.len() as u64;
                    if let Err(err) = phase.render_range(&mut pass, world, ui_view_entity, run) {
                        result = Err(err);
                        break;
                    }
                }
            }
            if result.is_err() {
                break;
            }
        }
        drop(items);

        match result {
            Ok(()) => {
                let repaired_pixels = damage_regions.iter().map(PhysicalRect::area).sum::<u64>();
                surface.commit(pending, damage_regions, composite_regions);
                counters.repairs.fetch_add(1, Ordering::Relaxed);
                counters
                    .repair_pixels
                    .fetch_add(repaired_pixels, Ordering::Relaxed);
                counters
                    .items_replayed
                    .fetch_add(replayed, Ordering::Relaxed);
                if let Some(plan) = repair_plan.as_ref() {
                    world
                        .resource::<RetainedUiScene>()
                        .acknowledge(ui_view_target.0, plan);
                }
            }
            Err(err) => {
                surface.slots[pending].initialized = false;
                surface.slots[pending].generation = 0;
                error!("retained UI repair was deferred: {err:?}");
            }
        }
    }

    if !surface.has_content {
        return;
    }

    let pipeline_id = *composite_pipeline
        .pipelines
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .entry(extracted_view.target_format)
        .or_insert_with(|| {
            pipeline_cache
                .queue_render_pipeline(composite_pipeline.specialize(extracted_view.target_format))
        });
    let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline_id) else {
        return;
    };

    let attachment = target.get_unsampled_color_attachment();
    let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("retained_ui_composite"),
        color_attachments: &[Some(attachment)],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_render_pipeline(pipeline);
    pass.set_bind_group(0, &surface.slots[surface.active].bind_group, &[]);
    if let Some(viewport) = camera.viewport.as_ref() {
        pass.set_camera_viewport(viewport);
    }
    let viewport_origin = camera
        .viewport
        .as_ref()
        .map(|viewport| viewport.physical_position)
        .unwrap_or_default();
    for region in &surface.composite_regions {
        pass.set_scissor_rect(
            viewport_origin.x + region.min_x() as u32,
            viewport_origin.y + region.min_y() as u32,
            (region.max_x() - region.min_x()) as u32,
            (region.max_y() - region.min_y()) as u32,
        );
        pass.draw(0..3, 0..1);
    }
    counters.composites.fetch_add(1, Ordering::Relaxed);
    counters
        .composite_draws
        .fetch_add(surface.composite_regions.len() as u64, Ordering::Relaxed);
    counters.composite_pixels.fetch_add(
        surface
            .composite_regions
            .iter()
            .map(PhysicalRect::area)
            .sum(),
        Ordering::Relaxed,
    );
}
