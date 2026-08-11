//! Persistent UI layer and composition.

use crate::background::extract_retained_backgrounds;
use crate::border::extract_retained_borders;
use crate::gradient::{extract_retained_gradients, RetainedGradientDependencies};
use crate::image::{extract_retained_images, RetainedImageDependencies};
use crate::material::RetainedPendingMaterials;
use crate::sampled_image::{
    extract_sampled_image_changes, resolve_ready_sampled_images, RetainedSampledImages,
    RetainedUiImageWrites,
};
use crate::scene::{
    cleanup_retained_ui, replay_retained_ui, RetainedItem, RetainedItems, RetainedMaterialReplays,
    RetainedUiPaintCounters, RetainedUiScene,
};
use crate::shadow::{extract_retained_shadows, RetainedShadowDependencies};
use crate::text::{extract_retained_text, RetainedTextDependencies};
use crate::viewport::{extract_retained_viewports, RetainedViewportDependencies};
use crate::{PhysicalRect, RepairPlan};
use alloc::collections::VecDeque;
use bevy::{
    asset::{embedded_asset, load_embedded_asset, AssetServer},
    camera::{CameraOutputMode, ClearColor, ClearColorConfig, CompositingSpace},
    core_pipeline::{
        blit::{BlitPipeline, BlitPipelineKey},
        upscaling::upscaling,
        Core2d, Core2dSystems, Core3d, Core3dSystems, FullscreenShader,
    },
    ecs::{
        entity::Entity,
        query::With,
        schedule::{IntoScheduleConfigs, ScheduleCleanupPolicy},
        system::{Commands, Local, Query, Res, ResMut},
    },
    log::error,
    math::{FloatOrd, UVec2},
    prelude::{App, Plugin, Resource, World},
    render::{
        camera::ExtractedCamera,
        diagnostic::RecordDiagnostics,
        render_asset::RenderAssets,
        render_phase::{
            DrawFunctionId, DrawFunctions, PhaseItem, PhaseItemExtraIndex, ViewSortedRenderPhases,
        },
        render_resource::{
            binding_types::{sampler, texture_2d},
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
            BlendState, CachedRenderPipelineId, ColorTargetState, ColorWrites, Extent3d,
            FragmentState, LoadOp, Operations, Origin3d, PipelineCache, RenderPassColorAttachment,
            RenderPassDescriptor, RenderPipelineDescriptor, SamplerBindingType, ShaderStages,
            SpecializedRenderPipelines, StoreOp, Texture, TextureDescriptor, TextureDimension,
            TextureFormat, TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor,
            TextureViewId,
        },
        renderer::{RenderContext, RenderDevice, ViewQuery},
        texture::{FallbackImageZero, GpuImage},
        view::{ExtractedView, RetainedViewEntity, ViewTarget},
        ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
    },
    ui_render::{
        box_shadow::{BoxShadowInfrastructurePlugin, DrawBoxShadows, UiShadowsBatch},
        gradient::{DrawGradientFns, GradientBatch, GradientInfrastructurePlugin},
        ui_texture_slice_pipeline::{
            queue_ui_slice_items, DrawUiTextureSliceItem, UiTextureSlicerBatch,
            UiTextureSlicerInfrastructurePlugin,
        },
        DrawUiItem, ExtractedUiNodes, RenderUiSystems, TransparentUi, UiAntiAlias, UiCameraView,
        UiItemBatch, UiMaterialBatchRange, UiPipeline, UiPipelineKey, UiViewTarget,
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

/// Adds a persistent UI layer and folds it into Bevy's final blit.
#[derive(Default)]
pub struct RetainedUiRenderPlugin;

impl Plugin for RetainedUiRenderPlugin {
    fn build(&self, app: &mut App) {
        if app.is_plugin_added::<bevy::ui::UiPlugin>()
            && !app.is_plugin_added::<crate::RetainedUiMainWorldPlugin>()
        {
            app.add_plugins(crate::RetainedUiMainWorldPlugin);
        }
        app.add_plugins(BoxShadowInfrastructurePlugin);
        app.add_plugins(GradientInfrastructurePlugin);
        app.add_plugins(UiTextureSlicerInfrastructurePlugin);
        embedded_asset!(app, "composite.wgsl");

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<LayerSurfaces>()
            .init_resource::<RetainedUiLayerCounters>()
            .init_resource::<RetainedUiScene>()
            .init_resource::<RetainedGradientDependencies>()
            .init_resource::<RetainedImageDependencies>()
            .init_resource::<RetainedMaterialReplays>()
            .init_resource::<RetainedPendingMaterials>()
            .init_resource::<RetainedSampledImages>()
            .init_resource::<RetainedUiImageWrites>()
            .init_resource::<RetainedShadowDependencies>()
            .init_resource::<RetainedTextDependencies>()
            .init_resource::<RetainedViewportDependencies>()
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
                extract_retained_viewports
                    .in_set(RenderUiSystems::ExtractViewportNodes)
                    .before(replay_retained_ui),
            )
            .add_systems(
                ExtractSchedule,
                extract_retained_shadows
                    .in_set(RenderUiSystems::ExtractBoxShadows)
                    .before(replay_retained_ui),
            )
            .add_systems(
                ExtractSchedule,
                extract_retained_borders.in_set(RenderUiSystems::ExtractBorders),
            )
            .add_systems(
                ExtractSchedule,
                extract_retained_gradients
                    .in_set(RenderUiSystems::ExtractGradient)
                    .before(replay_retained_ui),
            )
            .add_systems(
                ExtractSchedule,
                extract_retained_text.in_set(RenderUiSystems::ExtractText),
            )
            .add_systems(
                ExtractSchedule,
                extract_sampled_image_changes
                    .before(extract_retained_images)
                    .before(extract_retained_text)
                    .before(extract_retained_viewports),
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
                resolve_ready_sampled_images.in_set(RenderSystems::PrepareResources),
            )
            .add_systems(Render, queue_retained_uinodes.in_set(RenderSystems::Queue))
            .add_systems(Render, queue_ui_slice_items.in_set(RenderSystems::Queue))
            .add_systems(
                RenderStartup,
                (clear_final_pipelines, init_retained_ui_pipelines).chain(),
            )
            .add_systems(
                Render,
                prepare_final_pipelines
                    .in_set(RenderSystems::Prepare)
                    .ambiguous_with_all(),
            );

        let removed_2d = render_app
            .remove_systems_in_set(Core2d, upscaling, ScheduleCleanupPolicy::RemoveSystemsOnly)
            .expect("Core2dPlugin must be added before RetainedUiRenderPlugin");
        assert_eq!(removed_2d, 1, "Core2d must contain one final writer");
        let removed_3d = render_app
            .remove_systems_in_set(Core3d, upscaling, ScheduleCleanupPolicy::RemoveSystemsOnly)
            .expect("Core3dPlugin must be added before RetainedUiRenderPlugin");
        assert_eq!(removed_3d, 1, "Core3d must contain one final writer");

        render_app
            .add_systems(Core2d, retained_ui_pass.after(Core2dSystems::PostProcess))
            .add_systems(Core3d, retained_ui_pass.after(Core3dSystems::PostProcess));
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
    /// Current texture payload bytes owned by persistent layer surfaces.
    pub surface_bytes: u64,
    /// Atomic UI-layer repair transactions encoded.
    pub repairs: u64,
    /// Physical pixels cleared and repainted across exact repair regions.
    pub repair_pixels: u64,
    /// Individual paint items replayed across all repair regions.
    pub items_replayed: u64,
    /// Prepared quads submitted across all replayed paint items.
    pub quads_replayed: u64,
    /// Final blits that sampled and composited a retained UI layer.
    pub composites: u64,
    /// Physical output pixels that sampled the retained UI layer.
    pub ui_sample_pixels: u64,
}

/// Atomic render-world counters for retained-layer acceptance tests and diagnostics.
#[derive(Resource, Default)]
pub struct RetainedUiLayerCounters {
    surfaces_created: AtomicU64,
    surface_bytes: AtomicU64,
    repairs: AtomicU64,
    repair_pixels: AtomicU64,
    items_replayed: AtomicU64,
    quads_replayed: AtomicU64,
    composites: AtomicU64,
    ui_sample_pixels: AtomicU64,
}

impl RetainedUiLayerCounters {
    /// Returns a consistent-enough monotonic snapshot for diagnostics and tests.
    pub fn snapshot(&self) -> RetainedUiLayerWork {
        RetainedUiLayerWork {
            surfaces_created: self.surfaces_created.load(Ordering::Relaxed),
            surface_bytes: self.surface_bytes.load(Ordering::Relaxed),
            repairs: self.repairs.load(Ordering::Relaxed),
            repair_pixels: self.repair_pixels.load(Ordering::Relaxed),
            items_replayed: self.items_replayed.load(Ordering::Relaxed),
            quads_replayed: self.quads_replayed.load(Ordering::Relaxed),
            composites: self.composites.load(Ordering::Relaxed),
            ui_sample_pixels: self.ui_sample_pixels.load(Ordering::Relaxed),
        }
    }
}

#[derive(Resource)]
struct RetainedUiPipelines {
    final_layout: BindGroupLayoutDescriptor,
    shader: bevy::asset::Handle<bevy::shader::Shader>,
    vertex: bevy::render::render_resource::VertexState,
    final_pipelines: Mutex<HashMap<(BlitPipelineKey, bool), CachedRenderPipelineId>>,
    wipe_pipelines: Mutex<HashMap<TextureFormat, CachedRenderPipelineId>>,
}

impl RetainedUiPipelines {
    fn final_pipeline(
        &self,
        key: BlitPipelineKey,
        fused: bool,
        pipeline_cache: &PipelineCache,
    ) -> CachedRenderPipelineId {
        *self
            .final_pipelines
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .entry((key, fused))
            .or_insert_with(|| {
                let mut shader_defs = Vec::new();
                match key.source_space {
                    Some(CompositingSpace::Srgb) => shader_defs.push("SRGB_TO_LINEAR".into()),
                    Some(CompositingSpace::Oklab) => shader_defs.push("OKLAB_TO_LINEAR".into()),
                    Some(CompositingSpace::Linear) | None => {}
                }
                pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
                    label: Some("retained_ui_final_blit_pipeline".into()),
                    layout: vec![self.final_layout.clone()],
                    vertex: self.vertex.clone(),
                    fragment: Some(FragmentState {
                        shader: self.shader.clone(),
                        shader_defs,
                        entry_point: Some(if fused { "final_blit" } else { "plain_blit" }.into()),
                        targets: vec![Some(ColorTargetState {
                            format: key.target_format,
                            blend: key.blend_state,
                            write_mask: ColorWrites::ALL,
                        })],
                    }),
                    ..Default::default()
                })
            })
    }

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

fn init_retained_ui_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    fullscreen_shader: Res<FullscreenShader>,
) {
    commands.insert_resource(RetainedUiPipelines {
        final_layout: BindGroupLayoutDescriptor::new(
            "retained_ui_final_blit_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    sampler(SamplerBindingType::NonFiltering),
                ),
            ),
        ),
        shader: load_embedded_asset!(asset_server.as_ref(), "composite.wgsl"),
        vertex: fullscreen_shader.to_vertex_state(),
        final_pipelines: Mutex::new(HashMap::new()),
        wipe_pipelines: Mutex::new(HashMap::new()),
    });
}

#[derive(bevy::prelude::Component)]
struct RetainedFinalPipeline {
    plain: CachedRenderPipelineId,
    fused: CachedRenderPipelineId,
    key: BlitPipelineKey,
}

fn clear_final_pipelines(
    mut commands: Commands,
    views: Query<Entity, With<RetainedFinalPipeline>>,
) {
    for entity in &views {
        commands.entity(entity).remove::<RetainedFinalPipeline>();
    }
}

fn prepare_final_pipelines(
    mut commands: Commands,
    mut pipeline_cache: ResMut<PipelineCache>,
    pipeline: Res<RetainedUiPipelines>,
    views: Query<(
        Entity,
        &ViewTarget,
        Option<&ExtractedCamera>,
        Option<&RetainedFinalPipeline>,
    )>,
) {
    for (entity, view_target, camera, prepared) in &views {
        let blend_state = camera.and_then(|camera| match camera.output_mode {
            CameraOutputMode::Skip => None,
            CameraOutputMode::Write { blend_state, .. } => blend_state.or_else(|| {
                (camera.sorted_camera_index_for_target > 0).then_some(BlendState::ALPHA_BLENDING)
            }),
        });
        let Some(target_format) = view_target.out_texture_view_format() else {
            continue;
        };
        let key = BlitPipelineKey {
            target_format,
            blend_state,
            samples: 1,
            source_space: view_target.compositing_space,
        };
        if prepared.is_some_and(|prepared| prepared.key == key) {
            continue;
        }
        let plain = pipeline.final_pipeline(key, false, &pipeline_cache);
        let fused = pipeline.final_pipeline(key, true, &pipeline_cache);
        pipeline_cache.block_on_render_pipeline(plain);
        pipeline_cache.block_on_render_pipeline(fused);
        commands
            .entity(entity)
            .insert(RetainedFinalPipeline { plain, fused, key });
    }
}

struct LayerSlot {
    texture: Texture,
    view: TextureView,
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
    generation: u64,
    history: VecDeque<CommittedDamage>,
}

impl LayerSurface {
    fn new(render_device: &RenderDevice, size: UVec2, format: TextureFormat) -> Self {
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
            LayerSlot {
                texture,
                view,
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
            generation: 0,
            history: VecDeque::new(),
        }
    }

    fn matches(&self, size: UVec2, format: TextureFormat) -> bool {
        self.size == size && self.format == format
    }

    fn payload_bytes(&self) -> u64 {
        u64::from(self.size.x)
            * u64::from(self.size.y)
            * u64::from(
                self.format
                    .block_copy_size(None)
                    .expect("render-attachment formats have a fixed texel size"),
            )
            * self.slots.len() as u64
    }

    fn commit(&mut self, active: usize, regions: Vec<PhysicalRect>, has_content: bool) {
        self.generation += 1;
        self.active = active;
        self.has_content = has_content;
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

fn cleanup_layer_surfaces(
    views: Query<&ExtractedView>,
    surfaces: Res<LayerSurfaces>,
    counters: Res<RetainedUiLayerCounters>,
) {
    let live: HashSet<_> = views.iter().map(|view| view.retained_view_entity).collect();
    let mut surfaces = surfaces.0.lock().unwrap_or_else(PoisonError::into_inner);
    let removed_bytes = surfaces
        .iter()
        .filter(|(view, _)| !live.contains(view))
        .map(|(_, surface)| surface.payload_bytes())
        .sum();
    surfaces.retain(|view, _| live.contains(view));
    counters
        .surface_bytes
        .fetch_sub(removed_bytes, Ordering::Relaxed);
}

fn phase_is_ready(
    phase: &bevy::render::render_phase::SortedRenderPhase<TransparentUi>,
    pipeline_cache: &PipelineCache,
    world: &World,
    draw_functions: RetainedDrawFunctionIds,
) -> bool {
    if phase.items.is_empty() {
        return false;
    }
    let items = world
        .resource::<RetainedItems>()
        .0
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    let sampled_images = world.resource::<RetainedSampledImages>();
    let gpu_images = world.resource::<RenderAssets<GpuImage>>();
    for index in 0..phase.items.len() {
        let item = phase.items.get_index(index).unwrap().1;
        if item_batch_range(world, item, draw_functions).is_none() {
            if let Some(metadata) = items.get(&item.entity()) {
                let mut unavailable_image = false;
                for image in &metadata.sampled_images {
                    if gpu_images.get(*image).is_some() {
                        continue;
                    }
                    if sampled_images.is_pending(*image) {
                        return false;
                    }
                    unavailable_image = true;
                }
                if unavailable_image {
                    continue;
                }
            }
            return false;
        }
        if pipeline_cache.get_render_pipeline(item.pipeline).is_none() {
            return false;
        }
    }
    true
}

#[derive(Clone, Copy)]
struct RetainedDrawFunctionIds {
    box_shadow: DrawFunctionId,
    gradient: DrawFunctionId,
    ui: DrawFunctionId,
    texture_slice: DrawFunctionId,
}

fn retained_draw_function_ids(world: &World) -> RetainedDrawFunctionIds {
    let draw_functions = world.resource::<DrawFunctions<TransparentUi>>().read();
    RetainedDrawFunctionIds {
        box_shadow: draw_functions.id::<DrawBoxShadows>(),
        gradient: draw_functions.id::<DrawGradientFns>(),
        ui: draw_functions.id::<DrawUiItem>(),
        texture_slice: draw_functions.id::<DrawUiTextureSliceItem>(),
    }
}

fn item_batch_range(
    world: &World,
    item: &TransparentUi,
    draw_functions: RetainedDrawFunctionIds,
) -> Option<Range<u32>> {
    if item.draw_function == draw_functions.box_shadow {
        world
            .get::<UiShadowsBatch>(item.entity())
            .map(|batch| batch.range.clone())
    } else if item.draw_function == draw_functions.gradient {
        world
            .get::<GradientBatch>(item.entity())
            .map(|batch| batch.range.clone())
    } else if item.draw_function == draw_functions.ui {
        world
            .get::<UiItemBatch>(item.entity())
            .map(|batch| batch.range.clone())
    } else if item.draw_function == draw_functions.texture_slice {
        world
            .get::<UiTextureSlicerBatch>(item.entity())
            .map(|batch| batch.range.clone())
    } else {
        world
            .get::<UiMaterialBatchRange>(item.entity())
            .map(|batch| batch.range.clone())
    }
}

fn prepared_quad_count(
    world: &World,
    item: &TransparentUi,
    draw_functions: RetainedDrawFunctionIds,
) -> u64 {
    let indices = item_batch_range(world, item, draw_functions)
        .map(|range| range.len())
        .unwrap_or_default();
    u64::try_from(indices / 6).expect("prepared UI quad count exceeds u64")
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

fn replay_plan(
    phase: &bevy::render::render_phase::SortedRenderPhase<TransparentUi>,
    items: &HashMap<Entity, RetainedItem>,
    region: PhysicalRect,
    world: &World,
    draw_functions: RetainedDrawFunctionIds,
) -> crate::ReplayPlan {
    crate::ReplayPlan::for_region(
        region,
        (0..phase.items.len()).map(|index| {
            let item = phase.items.get_index(index).unwrap().1;
            if item_batch_range(world, item, draw_functions).is_none() {
                crate::ReplayItem::Culled
            } else if let Some(item) = items.get(&item.entity()) {
                crate::ReplayItem::Bounded(&item.coverage)
            } else {
                crate::ReplayItem::Unbounded
            }
        }),
    )
}

fn phase_has_drawable_items(
    phase: Option<&bevy::render::render_phase::SortedRenderPhase<TransparentUi>>,
    world: &World,
    draw_functions: RetainedDrawFunctionIds,
) -> bool {
    let Some(phase) = phase else {
        return false;
    };
    (0..phase.items.len()).any(|index| {
        let item = phase.items.get_index(index).unwrap().1;
        item_batch_range(world, item, draw_functions).is_some_and(|range| !range.is_empty())
    })
}

#[derive(Default)]
struct FinalBindGroupCache {
    cached: Option<(TextureViewId, TextureViewId, BindGroup)>,
}

#[expect(
    clippy::too_many_arguments,
    reason = "the final writer preserves Bevy's independent camera output inputs"
)]
fn final_blit(
    target: &ViewTarget,
    camera: &ExtractedCamera,
    prepared: &RetainedFinalPipeline,
    ui_view: &TextureView,
    has_ui: bool,
    pipeline: &RetainedUiPipelines,
    pipeline_cache: &PipelineCache,
    blit_pipeline: &BlitPipeline,
    global_clear_color: &ClearColor,
    cache: &mut FinalBindGroupCache,
    ctx: &mut RenderContext,
) -> bool {
    let clear_config = match camera.output_mode {
        CameraOutputMode::Write { clear_color, .. } => clear_color,
        CameraOutputMode::Skip => return false,
    };
    let clear_color = match clear_config {
        ClearColorConfig::Default => Some(global_clear_color.0),
        ClearColorConfig::Custom(color) => Some(color),
        ClearColorConfig::None => None,
    };
    let main_view = target.main_texture_view();
    let bind_group = match &mut cache.cached {
        Some((main_id, ui_id, bind_group))
            if *main_id == main_view.id() && *ui_id == ui_view.id() =>
        {
            bind_group
        }
        cached => {
            let bind_group = ctx.render_device().create_bind_group(
                "retained_ui_final_blit_bind_group",
                &pipeline_cache.get_bind_group_layout(&pipeline.final_layout),
                &BindGroupEntries::sequential((ui_view, main_view, &blit_pipeline.sampler)),
            );
            let (_, _, bind_group) = cached.insert((main_view.id(), ui_view.id(), bind_group));
            bind_group
        }
    };
    let Some(attachment) = target.out_texture_color_attachment(clear_color.map(Into::into)) else {
        return false;
    };
    let pass_descriptor = RenderPassDescriptor {
        label: Some("retained_ui_final_blit"),
        color_attachments: &[Some(attachment)],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    };
    let pipeline_id = if has_ui {
        prepared.fused
    } else {
        prepared.plain
    };
    let Some(render_pipeline) = pipeline_cache.get_render_pipeline(pipeline_id) else {
        #[cfg(target_os = "macos")]
        ctx.command_encoder().begin_render_pass(&pass_descriptor);
        return false;
    };

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let time_span = diagnostics.time_span(ctx.command_encoder(), "retained_ui_final_blit");
    {
        let mut pass = ctx.command_encoder().begin_render_pass(&pass_descriptor);
        if let Some(viewport) = &camera.viewport {
            pass.set_scissor_rect(
                viewport.physical_position.x,
                viewport.physical_position.y,
                viewport.physical_size.x,
                viewport.physical_size.y,
            );
        }
        pass.set_pipeline(render_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
    time_span.end(ctx.command_encoder());
    true
}

#[expect(
    clippy::too_many_arguments,
    reason = "render pass inputs are independent resources"
)]
fn retained_ui_pass(
    world: &World,
    view: ViewQuery<(
        &UiCameraView,
        &ViewTarget,
        &ExtractedCamera,
        &RetainedFinalPipeline,
    )>,
    ui_view_query: Query<(&ExtractedView, &UiViewTarget)>,
    transparent_render_phases: Res<ViewSortedRenderPhases<TransparentUi>>,
    pipelines: Res<RetainedUiPipelines>,
    pipeline_cache: Res<PipelineCache>,
    blit_pipeline: Res<BlitPipeline>,
    fallback: Res<FallbackImageZero>,
    clear_color: Res<ClearColor>,
    surfaces: Res<LayerSurfaces>,
    counters: Res<RetainedUiLayerCounters>,
    mut final_cache: Local<FinalBindGroupCache>,
    mut ctx: RenderContext,
) {
    let (ui_camera_view, target, camera, final_pipeline) = view.into_inner();
    let ui_view_entity = ui_camera_view.0;
    let Ok((extracted_view, ui_view_target)) = ui_view_query.get(ui_view_entity) else {
        let _ = final_blit(
            target,
            camera,
            final_pipeline,
            &fallback.texture_view,
            false,
            &pipelines,
            &pipeline_cache,
            &blit_pipeline,
            &clear_color,
            &mut final_cache,
            &mut ctx,
        );
        return;
    };
    let Some(size) = camera.physical_viewport_size else {
        let _ = final_blit(
            target,
            camera,
            final_pipeline,
            &fallback.texture_view,
            false,
            &pipelines,
            &pipeline_cache,
            &blit_pipeline,
            &clear_color,
            &mut final_cache,
            &mut ctx,
        );
        return;
    };

    let phase = transparent_render_phases.get(&extracted_view.retained_view_entity);
    let draw_functions = retained_draw_function_ids(world);
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
    let material_pending = world
        .resource::<RetainedPendingMaterials>()
        .contains(ui_view_target.0);
    let repair_ready = !material_pending
        && if phase_has_items {
            phase.is_some_and(|phase| phase_is_ready(phase, &pipeline_cache, world, draw_functions))
        } else {
            repair_plan.is_some()
        };

    let mut surfaces = surfaces.0.lock().unwrap_or_else(PoisonError::into_inner);
    let existing_matches = surfaces
        .get(&extracted_view.retained_view_entity)
        .is_some_and(|surface| surface.matches(size, extracted_view.target_format));
    if !existing_matches && !repair_requested {
        drop(surfaces);
        let viewport = target_rect(size);
        if scene.has_visible_records(ui_view_target.0, viewport) {
            scene.invalidate(ui_view_target.0, viewport);
        }
        let _ = final_blit(
            target,
            camera,
            final_pipeline,
            &fallback.texture_view,
            false,
            &pipelines,
            &pipeline_cache,
            &blit_pipeline,
            &clear_color,
            &mut final_cache,
            &mut ctx,
        );
        return;
    }
    let surface = surfaces
        .entry(extracted_view.retained_view_entity)
        .or_insert_with(|| {
            counters.surfaces_created.fetch_add(1, Ordering::Relaxed);
            let surface =
                LayerSurface::new(ctx.render_device(), size, extracted_view.target_format);
            counters
                .surface_bytes
                .fetch_add(surface.payload_bytes(), Ordering::Relaxed);
            surface
        });
    if !surface.matches(size, extracted_view.target_format) {
        let previous_bytes = surface.payload_bytes();
        *surface = LayerSurface::new(ctx.render_device(), size, extracted_view.target_format);
        counters.surfaces_created.fetch_add(1, Ordering::Relaxed);
        let current_bytes = surface.payload_bytes();
        if current_bytes >= previous_bytes {
            counters
                .surface_bytes
                .fetch_add(current_bytes - previous_bytes, Ordering::Relaxed);
        } else {
            counters
                .surface_bytes
                .fetch_sub(previous_bytes - current_bytes, Ordering::Relaxed);
        }
    }

    let wipe_pipeline = (repair_requested && repair_ready)
        .then(|| pipelines.wipe_pipeline(extracted_view.target_format, &pipeline_cache))
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
        let mut replayed_quads = 0;
        let has_content = phase_has_drawable_items(phase, world, draw_functions);
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
                for run in replay_plan(phase, &items, *region, world, draw_functions)
                    .runs()
                    .iter()
                    .cloned()
                {
                    replayed += run.len() as u64;
                    for index in run.clone() {
                        let item = phase.items.get_index(index).unwrap().1;
                        replayed_quads += prepared_quad_count(world, item, draw_functions);
                    }
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
                surface.commit(pending, damage_regions, has_content);
                counters.repairs.fetch_add(1, Ordering::Relaxed);
                counters
                    .repair_pixels
                    .fetch_add(repaired_pixels, Ordering::Relaxed);
                counters
                    .items_replayed
                    .fetch_add(replayed, Ordering::Relaxed);
                counters
                    .quads_replayed
                    .fetch_add(replayed_quads, Ordering::Relaxed);
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

    let has_content = surface.has_content;
    let ui_view = if has_content {
        &surface.slots[surface.active].view
    } else {
        &fallback.texture_view
    };
    let final_blit_encoded = final_blit(
        target,
        camera,
        final_pipeline,
        ui_view,
        has_content,
        &pipelines,
        &pipeline_cache,
        &blit_pipeline,
        &clear_color,
        &mut final_cache,
        &mut ctx,
    );
    if has_content && final_blit_encoded {
        counters.composites.fetch_add(1, Ordering::Relaxed);
        counters
            .ui_sample_pixels
            .fetch_add(u64::from(size.x) * u64::from(size.y), Ordering::Relaxed);
    }
}
