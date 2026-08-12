//! Persistent UI layer and composition.

use crate::background::{extract_retained_backgrounds, PendingRetainedBackgrounds};
use crate::border::{extract_retained_borders, RetainedBorderDependencies};
use crate::boundary::{
    extract_boundaries, invalidate_volatile_paint_targets, prepare_boundary_views,
    propagate_boundary_damage, BoundaryViews,
};
use crate::core::{
    prepare_retained_core, queue_retained_core, register_retained_core, DrawRetainedCore,
    DrawRetainedCoreRebuild, RetainedCore,
};
use crate::gradient::{extract_retained_gradients, RetainedGradientDependencies};
use crate::gradient_render::{
    prepare as prepare_retained_gradients, queue as queue_retained_gradients,
    register as register_retained_gradients, DrawRetainedGradients, RetainedGradients,
};
use crate::image::{extract_retained_images, RetainedImageDependencies};
use crate::material::RetainedPendingMaterials;
use crate::sampled_image::{
    extract_sampled_image_changes, resolve_ready_sampled_images, RetainedSampledImages,
    RetainedUiImageWrites,
};
use crate::scene::{
    apply_retained_paints, cleanup_retained_ui, extract_boundary_ownership,
    extract_retained_placements, replay_retained_ui, PendingGradientPaints, PendingImagePaints,
    PendingShadowPaints, PendingViewportPaints, RetainedFullRebuilds, RetainedItems,
    RetainedMaterialReplays, RetainedRepairPlans, RetainedUiPaintCounters, RetainedUiScene,
};
use crate::shadow::{extract_retained_shadows, RetainedShadowDependencies};
use crate::shadow_render::{
    prepare as prepare_retained_shadows, queue as queue_retained_shadows,
    register as register_retained_shadows, DrawRetainedShadows, RetainedShadows,
};
use crate::text::{extract_retained_text, RetainedTextDependencies};
use crate::viewport::{extract_retained_viewports, RetainedViewportDependencies};
use crate::{PhysicalRect, RepairPlan};
use alloc::collections::VecDeque;
use bevy::{
    asset::{embedded_asset, load_embedded_asset, AssetServer},
    core_pipeline::{
        blit::BlitPipeline,
        upscaling::{upscaling, ViewOutputOverlays},
        Core2d, Core2dSystems, Core3d, Core3dSystems,
    },
    ecs::{
        entity::Entity,
        query::With,
        schedule::{IntoScheduleConfigs, ScheduleCleanupPolicy},
        system::{Commands, Local, Query, Res, ResMut, SystemParamItem},
    },
    log::error,
    math::{FloatOrd, UVec2, Vec2},
    mesh::{VertexBufferLayout, VertexFormat},
    prelude::{App, Plugin, Resource, World},
    render::{
        camera::ExtractedCamera,
        render_asset::RenderAssets,
        render_phase::{
            AddRenderCommand, DrawFunctionId, DrawFunctions, PhaseItem, PhaseItemExtraIndex,
            RenderCommand, RenderCommandResult, TrackedRenderPass, ViewSortedRenderPhases,
        },
        render_resource::{
            binding_types::{sampler, texture_2d},
            BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, BlendState,
            BufferUsages, CachedRenderPipelineId, ColorTargetState, ColorWrites, Extent3d,
            FragmentState, LoadOp, Operations, PipelineCache, RawBufferVec,
            RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor,
            SamplerBindingType, ShaderStages, StoreOp, Texture, TextureDescriptor,
            TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureView,
            TextureViewDescriptor, VertexAttribute, VertexState, VertexStepMode,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
        texture::GpuImage,
        view::{ExtractedView, RetainedViewEntity},
        ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
    },
    ui_render::{
        gradient::GradientInfrastructurePlugin,
        ui_texture_slice_pipeline::{
            queue_ui_slice_items, DrawUiTextureSliceItem, UiTextureSlicerBatch,
            UiTextureSlicerInfrastructurePlugin,
        },
        PrepareUiSystems, RenderUiSystems, TransparentUi, UiCameraView, UiMaterialBatchRange,
        UiViewTarget,
    },
};
use bytemuck::{Pod, Zeroable};
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
        app.add_plugins(UiTextureSlicerInfrastructurePlugin);
        app.add_plugins(GradientInfrastructurePlugin);
        embedded_asset!(app, "core.wgsl");
        embedded_asset!(app, "gradient_render.wgsl");
        embedded_asset!(app, "shadow.wgsl");
        embedded_asset!(app, "composite.wgsl");

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .world_mut()
            .resource_mut::<ViewOutputOverlays>()
            .enable();
        render_app.add_render_command::<TransparentUi, DrawRetainedBoundary>();
        register_retained_core(render_app);
        register_retained_gradients(render_app);
        register_retained_shadows(render_app);

        let removed_node_preparation = render_app
            .remove_systems_in_set(
                Render,
                PrepareUiSystems::Nodes,
                ScheduleCleanupPolicy::RemoveSystemsOnly,
            )
            .expect("UiRenderInfrastructurePlugin must be added before RetainedUiRenderPlugin");
        assert_eq!(
            removed_node_preparation, 1,
            "UI infrastructure must contain one transient node preparation system"
        );

        render_app
            .init_resource::<LayerSurfaces>()
            .init_resource::<BoundaryViews>()
            .init_resource::<RetainedUiLayerCounters>()
            .init_resource::<RetainedUiScene>()
            .init_resource::<RetainedGradientDependencies>()
            .init_resource::<RetainedBorderDependencies>()
            .init_resource::<RetainedImageDependencies>()
            .init_resource::<RetainedMaterialReplays>()
            .init_resource::<RetainedPendingMaterials>()
            .init_resource::<RetainedSampledImages>()
            .init_resource::<RetainedUiImageWrites>()
            .init_resource::<RetainedShadowDependencies>()
            .init_resource::<RetainedTextDependencies>()
            .init_resource::<RetainedViewportDependencies>()
            .init_resource::<RetainedItems>()
            .init_resource::<PendingRetainedBackgrounds>()
            .init_resource::<PendingGradientPaints>()
            .init_resource::<PendingImagePaints>()
            .init_resource::<PendingShadowPaints>()
            .init_resource::<PendingViewportPaints>()
            .init_resource::<RetainedRepairPlans>()
            .init_resource::<RetainedFullRebuilds>()
            .init_resource::<RetainedUiPaintCounters>()
            .init_resource::<LayerRects>()
            .init_resource::<BoundaryBatches>()
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
                    .before(extract_retained_images)
                    .before(extract_retained_text),
            )
            .add_systems(
                ExtractSchedule,
                extract_retained_shadows.in_set(RenderUiSystems::ExtractBoxShadows),
            )
            .add_systems(
                ExtractSchedule,
                extract_retained_borders.in_set(RenderUiSystems::ExtractBorders),
            )
            .add_systems(
                ExtractSchedule,
                extract_retained_gradients.in_set(RenderUiSystems::ExtractGradient),
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
                extract_boundaries
                    .after(RenderUiSystems::ExtractCameraViews)
                    .before(extract_boundary_ownership),
            )
            .add_systems(
                ExtractSchedule,
                extract_boundary_ownership
                    .before(extract_retained_placements)
                    .before(RenderUiSystems::ExtractBoxShadows)
                    .before(RenderUiSystems::ExtractBackgrounds)
                    .before(RenderUiSystems::ExtractBorders)
                    .before(extract_retained_viewports)
                    .before(extract_retained_gradients),
            )
            .add_systems(
                ExtractSchedule,
                extract_retained_placements
                    .before(RenderUiSystems::ExtractBoxShadows)
                    .before(RenderUiSystems::ExtractBackgrounds)
                    .before(RenderUiSystems::ExtractBorders)
                    .before(extract_retained_viewports)
                    .before(extract_retained_gradients),
            )
            .add_systems(
                ExtractSchedule,
                invalidate_volatile_paint_targets
                    .after(RenderUiSystems::ExtractDebug)
                    .after(RenderUiSystems::ExtractBackgrounds)
                    .after(extract_boundaries)
                    .before(propagate_boundary_damage)
                    .before(prepare_boundary_views),
            )
            .add_systems(
                ExtractSchedule,
                prepare_boundary_views
                    .after(RenderUiSystems::ExtractDebug)
                    .after(extract_retained_gradients)
                    .after(extract_retained_viewports)
                    .after(propagate_boundary_damage),
            )
            .add_systems(
                ExtractSchedule,
                propagate_boundary_damage
                    .after(RenderUiSystems::ExtractDebug)
                    .after(extract_retained_gradients)
                    .after(extract_retained_viewports),
            )
            .add_systems(
                ExtractSchedule,
                apply_retained_paints
                    .after(RenderUiSystems::ExtractBoxShadows)
                    .after(RenderUiSystems::ExtractBackgrounds)
                    .after(RenderUiSystems::ExtractImages)
                    .after(RenderUiSystems::ExtractBorders)
                    .after(RenderUiSystems::ExtractText)
                    .after(RenderUiSystems::ExtractDebug)
                    .after(extract_retained_gradients)
                    .after(extract_retained_viewports)
                    .before(propagate_boundary_damage)
                    .before(prepare_boundary_views)
                    .before(replay_retained_ui),
            )
            .add_systems(
                ExtractSchedule,
                replay_retained_ui
                    .after(prepare_boundary_views)
                    .after(propagate_boundary_damage),
            )
            .add_systems(
                Render,
                (
                    cleanup_retained_ui,
                    resolve_ready_sampled_images,
                    cleanup_layer_surfaces,
                    clear_layer_rects,
                )
                    .chain()
                    .in_set(RenderSystems::PrepareResources),
            )
            .add_systems(Render, queue_retained_core.in_set(RenderSystems::Queue))
            .add_systems(Render, queue_boundaries.in_set(RenderSystems::Queue))
            .add_systems(
                Render,
                queue_retained_gradients.in_set(RenderSystems::Queue),
            )
            .add_systems(Render, queue_retained_shadows.in_set(RenderSystems::Queue))
            .add_systems(Render, queue_ui_slice_items.in_set(RenderSystems::Queue))
            .add_systems(
                Render,
                prepare_retained_core.in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(
                Render,
                prepare_boundaries.in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(
                Render,
                prepare_retained_gradients.in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(
                Render,
                prepare_retained_shadows.in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(RenderStartup, init_retained_ui_pipelines);

        render_app
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
    mask_layout: BindGroupLayoutDescriptor,
    shader: bevy::asset::Handle<bevy::shader::Shader>,
    rect_vertex: VertexState,
    wipe_pipelines: Mutex<HashMap<TextureFormat, CachedRenderPipelineId>>,
    copy_pipelines: Mutex<HashMap<TextureFormat, CachedRenderPipelineId>>,
    mask_pipeline: Mutex<Option<CachedRenderPipelineId>>,
    boundary_pipelines: Mutex<HashMap<TextureFormat, CachedRenderPipelineId>>,
}

impl RetainedUiPipelines {
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
                    vertex: self.rect_vertex.clone(),
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

    fn copy_pipeline(
        &self,
        format: TextureFormat,
        pipeline_cache: &PipelineCache,
    ) -> CachedRenderPipelineId {
        *self
            .copy_pipelines
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .entry(format)
            .or_insert_with(|| {
                pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
                    label: Some("retained_ui_copy_pipeline".into()),
                    layout: vec![self.final_layout.clone()],
                    vertex: self.rect_vertex.clone(),
                    fragment: Some(FragmentState {
                        shader: self.shader.clone(),
                        entry_point: Some("copy_retained".into()),
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

    fn mask_pipeline(&self, pipeline_cache: &PipelineCache) -> CachedRenderPipelineId {
        *self
            .mask_pipeline
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get_or_insert_with(|| {
                pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
                    label: Some("retained_ui_damage_mask_pipeline".into()),
                    layout: Vec::new(),
                    vertex: self.rect_vertex.clone(),
                    fragment: Some(FragmentState {
                        shader: self.shader.clone(),
                        entry_point: Some("mask".into()),
                        targets: vec![Some(ColorTargetState {
                            format: TextureFormat::R8Unorm,
                            blend: None,
                            write_mask: ColorWrites::RED,
                        })],
                        ..Default::default()
                    }),
                    ..Default::default()
                })
            })
    }

    fn boundary_pipeline(
        &self,
        format: TextureFormat,
        pipeline_cache: &PipelineCache,
    ) -> CachedRenderPipelineId {
        *self
            .boundary_pipelines
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .entry(format)
            .or_insert_with(|| {
                pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
                    label: Some("retained_ui_boundary_pipeline".into()),
                    layout: vec![self.final_layout.clone()],
                    vertex: VertexState {
                        shader: self.shader.clone(),
                        entry_point: Some("boundary_vertex".into()),
                        buffers: vec![VertexBufferLayout::from_vertex_formats(
                            VertexStepMode::Instance,
                            vec![
                                VertexFormat::Float32x4,
                                VertexFormat::Float32x2,
                                VertexFormat::Float32x2,
                                VertexFormat::Float32x4,
                                VertexFormat::Float32,
                                VertexFormat::Float32x2,
                            ],
                        )],
                        ..Default::default()
                    },
                    fragment: Some(FragmentState {
                        shader: self.shader.clone(),
                        entry_point: Some("boundary_fragment".into()),
                        targets: vec![Some(ColorTargetState {
                            format,
                            blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                            write_mask: ColorWrites::ALL,
                        })],
                        ..Default::default()
                    }),
                    ..Default::default()
                })
            })
    }
}

fn init_retained_ui_pipelines(mut commands: Commands, asset_server: Res<AssetServer>) {
    let shader = load_embedded_asset!(asset_server.as_ref(), "composite.wgsl");
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
        mask_layout: crate::mask::layout(),
        shader: shader.clone(),
        rect_vertex: VertexState {
            shader,
            shader_defs: Vec::new(),
            entry_point: Some("rect_vertex".into()),
            buffers: vec![VertexBufferLayout {
                array_stride: size_of::<[f32; 4]>() as u64,
                step_mode: VertexStepMode::Instance,
                attributes: vec![VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 0,
                }],
            }],
        },
        wipe_pipelines: Mutex::new(HashMap::new()),
        copy_pipelines: Mutex::new(HashMap::new()),
        mask_pipeline: Mutex::new(None),
        boundary_pipelines: Mutex::new(HashMap::new()),
    });
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BoundaryInstance {
    transform: [f32; 4],
    translation: [f32; 2],
    size: [f32; 2],
    uv_rect: [f32; 4],
    opacity: f32,
    target_size: [f32; 2],
}

struct BoundaryBatch {
    range: Range<u32>,
    surface: Entity,
}

#[derive(Resource)]
struct BoundaryBatches {
    instances: RawBufferVec<BoundaryInstance>,
    batches: HashMap<Entity, BoundaryBatch>,
}

impl Default for BoundaryBatches {
    fn default() -> Self {
        let mut instances = RawBufferVec::new(BufferUsages::VERTEX);
        instances.set_label(Some("retained UI boundary instances"));
        Self {
            instances,
            batches: HashMap::new(),
        }
    }
}

fn queue_boundaries(
    items: Res<RetainedItems>,
    pipelines: Res<RetainedUiPipelines>,
    pipeline_cache: Res<PipelineCache>,
    draw_functions: Res<DrawFunctions<TransparentUi>>,
    mut phases: ResMut<ViewSortedRenderPhases<TransparentUi>>,
    render_views: Query<&UiCameraView, With<ExtractedView>>,
    camera_views: Query<&ExtractedView>,
) {
    let draw_function = draw_functions.read().id::<DrawRetainedBoundary>();
    let draws = items
        .boundaries
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    for draw in draws.values() {
        let Ok(parent_view) = render_views.get(draw.camera) else {
            continue;
        };
        let Ok(view) = camera_views.get(parent_view.0) else {
            continue;
        };
        let Some(phase) = phases.get_mut(&view.retained_view_entity) else {
            continue;
        };
        let pipeline = pipelines.boundary_pipeline(view.target_format, &pipeline_cache);
        phase.add_transient(TransparentUi {
            draw_function,
            pipeline,
            entity: (draw.render_entity, draw.main_entity),
            sort_key: FloatOrd(draw.z_order),
            index: 0,
            batch_range: 0..1,
            extra_index: PhaseItemExtraIndex::None,
            indexed: false,
        });
    }
}

struct DrawRetainedBoundary;

impl<P: PhaseItem> RenderCommand<P> for DrawRetainedBoundary {
    type Param = ();
    type ViewQuery = ();
    type ItemQuery = ();

    fn render<'w>(
        _item: &P,
        _view: (),
        _entity: Option<()>,
        _param: SystemParamItem<'w, '_, Self::Param>,
        _pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        RenderCommandResult::Failure("retained boundaries require their source surface binding")
    }
}

fn prepare_boundaries(
    items: Res<RetainedItems>,
    views: Res<BoundaryViews>,
    cameras: Query<&ExtractedCamera>,
    mut boundaries: ResMut<BoundaryBatches>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    boundaries.instances.clear();
    boundaries.batches.clear();
    let draws = items
        .boundaries
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    for draw in draws.values() {
        let target_origin = views.views.get(&draw.camera).map_or(Vec2::ZERO, |view| {
            Vec2::new(view.rect.min_x() as f32, view.rect.min_y() as f32)
        });
        let target_size = views
            .views
            .get(&draw.camera)
            .map(|view| view.size().as_vec2())
            .or_else(|| {
                cameras
                    .get(draw.camera)
                    .ok()?
                    .physical_viewport_size
                    .map(UVec2::as_vec2)
            })
            .unwrap_or(draw.size);
        let transform = draw.transform.to_cols_array();
        let start =
            u32::try_from(boundaries.instances.len()).expect("retained boundary count exceeds u32");
        boundaries.instances.push(BoundaryInstance {
            transform: [transform[0], transform[1], transform[2], transform[3]],
            translation: [
                transform[4] - target_origin.x,
                transform[5] - target_origin.y,
            ],
            size: draw.size.to_array(),
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            opacity: draw.opacity,
            target_size: target_size.to_array(),
        });
        boundaries.batches.insert(
            draw.render_entity,
            BoundaryBatch {
                range: start..start + 1,
                surface: draw.surface,
            },
        );
    }
    if !boundaries.instances.is_empty() {
        boundaries.instances.write_buffer(&device, &queue);
    }
}

#[derive(Resource)]
struct LayerRects(Mutex<RawBufferVec<[f32; 4]>>);

impl Default for LayerRects {
    fn default() -> Self {
        let mut rects = RawBufferVec::new(BufferUsages::VERTEX);
        rects.set_label(Some("retained UI layer rectangles"));
        Self(Mutex::new(rects))
    }
}

fn clear_layer_rects(layer_rects: Res<LayerRects>) {
    layer_rects
        .0
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clear();
}

struct LayerSlot {
    _texture: Texture,
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
    _mask_texture: Texture,
    mask_view: TextureView,
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
                _texture: texture,
                view,
                initialized: false,
                generation: 0,
            }
        };

        let mask_texture = render_device.create_texture(&TextureDescriptor {
            label: Some("retained_ui_damage_mask"),
            size: Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R8Unorm,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let mask_view = mask_texture.create_view(&TextureViewDescriptor::default());

        Self {
            size,
            format,
            slots: [create_slot(), create_slot()],
            _mask_texture: mask_texture,
            mask_view,
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
            + u64::from(self.size.x) * u64::from(self.size.y)
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
    views: Query<(Entity, &ExtractedView), With<UiViewTarget>>,
    mut boundary_views: ResMut<BoundaryViews>,
    surfaces: Res<LayerSurfaces>,
    counters: Res<RetainedUiLayerCounters>,
    mut live_camera_views: Local<HashSet<RetainedViewEntity>>,
) {
    let current_camera_views: HashSet<_> = views
        .iter()
        .filter(|(entity, _)| !boundary_views.views.contains_key(entity))
        .map(|(_, view)| view.retained_view_entity)
        .collect();
    let mut surfaces = surfaces.0.lock().unwrap_or_else(PoisonError::into_inner);
    let removed_bytes = live_camera_views
        .difference(&current_camera_views)
        .copied()
        .chain(boundary_views.take_retired())
        .filter_map(|view| surfaces.remove(&view))
        .map(|surface| surface.payload_bytes())
        .sum();
    *live_camera_views = current_camera_views;
    counters
        .surface_bytes
        .fetch_sub(removed_bytes, Ordering::Relaxed);
}

fn phase_is_repair_ready(
    phase: &bevy::render::render_phase::SortedRenderPhase<TransparentUi>,
    pipeline_cache: &PipelineCache,
    world: &World,
    draw_functions: RetainedDrawFunctionIds,
) -> bool {
    let items = world
        .resource::<RetainedItems>()
        .items
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    let sampled_images = world.resource::<RetainedSampledImages>().lock();
    let gpu_images = world.resource::<RenderAssets<GpuImage>>();
    let core = world.resource::<RetainedCore>();
    let gradients = world.resource::<RetainedGradients>();
    let shadows = world.resource::<RetainedShadows>();
    let batch_may_be_omitted = |entity| {
        let Some(metadata) = items.get(&entity) else {
            return false;
        };
        let mut unavailable = false;
        for &image in &metadata.sampled_images {
            if gpu_images.get(image).is_some() {
                continue;
            }
            if sampled_images.is_pending(image) {
                return false;
            }
            unavailable = true;
        }
        unavailable
    };

    (0..phase.items.len()).all(|index| {
        let item = phase.items.get_index(index).unwrap().1;
        let boundary = item.draw_function == draw_functions.boundary;
        let core_image = draw_functions
            .is_core(item.draw_function)
            .then(|| core.image(item.entity()))
            .flatten();
        let prepared = if boundary {
            world
                .resource::<BoundaryBatches>()
                .batches
                .contains_key(&item.entity())
        } else if draw_functions.is_core(item.draw_function) {
            core.direct_ready(item.entity())
                && core_image.is_none_or(|image| !sampled_images.is_pending(image))
        } else if item.draw_function == draw_functions.retained_gradient {
            gradients.direct_ready(item.entity())
        } else if item.draw_function == draw_functions.retained_shadow {
            shadows.direct_ready(item.entity())
        } else {
            item_batch_range(world, item, draw_functions).is_some()
        };
        let omitted = core_image.map_or_else(
            || batch_may_be_omitted(item.entity()),
            |image| gpu_images.get(image).is_none() && !sampled_images.is_pending(image),
        );
        (prepared || omitted) && pipeline_cache.get_render_pipeline(item.pipeline).is_some()
    })
}

#[derive(Clone, Copy)]
struct RetainedDrawFunctionIds {
    boundary: DrawFunctionId,
    core: DrawFunctionId,
    core_rebuild: DrawFunctionId,
    retained_gradient: DrawFunctionId,
    retained_shadow: DrawFunctionId,
    texture_slice: DrawFunctionId,
}

impl RetainedDrawFunctionIds {
    fn is_core(self, draw: DrawFunctionId) -> bool {
        draw == self.core || draw == self.core_rebuild
    }
}

fn retained_draw_function_ids(world: &World) -> RetainedDrawFunctionIds {
    let draw_functions = world.resource::<DrawFunctions<TransparentUi>>().read();
    RetainedDrawFunctionIds {
        boundary: draw_functions.id::<DrawRetainedBoundary>(),
        core: draw_functions.id::<DrawRetainedCore>(),
        core_rebuild: draw_functions.id::<DrawRetainedCoreRebuild>(),
        retained_gradient: draw_functions.id::<DrawRetainedGradients>(),
        retained_shadow: draw_functions.id::<DrawRetainedShadows>(),
        texture_slice: draw_functions.id::<DrawUiTextureSliceItem>(),
    }
}

fn item_batch_range(
    world: &World,
    item: &TransparentUi,
    draw_functions: RetainedDrawFunctionIds,
) -> Option<Range<u32>> {
    if let Some(batch) = world
        .resource::<BoundaryBatches>()
        .batches
        .get(&item.entity())
    {
        Some(batch.range.clone())
    } else if draw_functions.is_core(item.draw_function) {
        world.resource::<RetainedCore>().batch_range(item.entity())
    } else if item.draw_function == draw_functions.retained_shadow {
        world
            .resource::<RetainedShadows>()
            .batch_range(item.entity())
    } else if item.draw_function == draw_functions.retained_gradient {
        world
            .resource::<RetainedGradients>()
            .batch_range(item.entity())
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

fn prepared_counts(
    world: &World,
    item: &TransparentUi,
    draw_functions: RetainedDrawFunctionIds,
) -> (u64, u64) {
    if world
        .resource::<BoundaryBatches>()
        .batches
        .contains_key(&item.entity())
    {
        return (1, 1);
    }
    if draw_functions.is_core(item.draw_function) {
        return world
            .resource::<RetainedCore>()
            .counts(item.entity())
            .map(|(items, quads)| {
                (
                    u64::from(items),
                    u64::try_from(quads).expect("prepared UI quad count exceeds u64"),
                )
            })
            .unwrap_or_default();
    }
    if item.draw_function == draw_functions.retained_shadow {
        return world
            .resource::<RetainedShadows>()
            .counts(item.entity())
            .map(|(items, quads)| {
                (
                    u64::from(items),
                    u64::try_from(quads).expect("prepared shadow count exceeds u64"),
                )
            })
            .unwrap_or_default();
    }
    if item.draw_function == draw_functions.retained_gradient {
        return world
            .resource::<RetainedGradients>()
            .counts(item.entity())
            .map(|(items, quads)| {
                (
                    u64::from(items),
                    u64::try_from(quads).expect("prepared gradient count exceeds u64"),
                )
            })
            .unwrap_or_default();
    }
    let indices = item_batch_range(world, item, draw_functions)
        .map(|range| range.len())
        .unwrap_or_default();
    (
        u64::from(indices != 0),
        u64::try_from(indices / 6).expect("prepared UI quad count exceeds u64"),
    )
}

fn prepared_masked_counts(world: &World, camera: Entity) -> (u64, u64) {
    [
        world.resource::<RetainedCore>().prepared_work(camera),
        world.resource::<RetainedGradients>().prepared_work(camera),
        world.resource::<RetainedShadows>().prepared_work(camera),
    ]
    .into_iter()
    .fold((0, 0), |sum, work| (sum.0 + work.0, sum.1 + work.1))
}

fn target_rect(size: UVec2) -> PhysicalRect {
    PhysicalRect::from_min_max(0, 0, size.x as i32, size.y as i32)
        .expect("render targets have nonzero size")
}

fn clipped_damage_to(plan: &RepairPlan, target: PhysicalRect) -> Vec<PhysicalRect> {
    plan.regions()
        .iter()
        .filter_map(|region| region.intersection(target))
        .collect()
}

fn translate_rect(rect: PhysicalRect, x: i32, y: i32) -> PhysicalRect {
    PhysicalRect::from_min_max(
        rect.min_x() - x,
        rect.min_y() - y,
        rect.max_x() - x,
        rect.max_y() - y,
    )
    .expect("translating a nonempty damage rectangle keeps it nonempty")
}

fn boundary_depth(surface: Entity, views: &BoundaryViews) -> usize {
    let mut depth = 0;
    let mut current = surface;
    while let Some(view) = views.views.get(&current)
        && views.views.contains_key(&view.parent_surface)
    {
        depth += 1;
        current = view.parent_surface;
    }
    depth
}

fn coverage_intersects(coverage: &crate::PaintCoverage, region: PhysicalRect) -> bool {
    coverage
        .iter()
        .any(|coverage| coverage.intersection(region).is_some())
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

fn pending_sync_regions(
    surface: &LayerSurface,
    pending: usize,
    ctx: &mut RenderContext,
) -> Vec<PhysicalRect> {
    let pending_slot = &surface.slots[pending];
    if !pending_slot.initialized {
        clear_layer_slot(ctx, pending_slot);
    }
    if !surface.has_content {
        return Vec::new();
    }

    surface
        .history
        .iter()
        .filter(|damage| damage.generation > pending_slot.generation)
        .flat_map(|damage| damage.regions.iter().copied())
        .collect()
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

#[derive(Clone, Copy)]
struct SurfaceRequest {
    view_entity: Entity,
    retained_view_entity: RetainedViewEntity,
    paint_entity: Entity,
    bounds: PhysicalRect,
    size: UVec2,
    format: TextureFormat,
}

struct RepairResources<'a> {
    phases: &'a ViewSortedRenderPhases<TransparentUi>,
    pipelines: &'a RetainedUiPipelines,
    pipeline_cache: &'a PipelineCache,
    blit_pipeline: &'a BlitPipeline,
    counters: &'a RetainedUiLayerCounters,
    render_queue: &'a RenderQueue,
    layer_rects: &'a LayerRects,
    repair_plans: &'a RetainedRepairPlans,
    boundary_views: &'a BoundaryViews,
}

fn surface_for_request<'a>(
    request: SurfaceRequest,
    counters: &RetainedUiLayerCounters,
    surfaces: &'a mut HashMap<RetainedViewEntity, LayerSurface>,
    device: &RenderDevice,
) -> &'a mut LayerSurface {
    let surface = surfaces
        .entry(request.retained_view_entity)
        .or_insert_with(|| {
            counters.surfaces_created.fetch_add(1, Ordering::Relaxed);
            let surface = LayerSurface::new(device, request.size, request.format);
            counters
                .surface_bytes
                .fetch_add(surface.payload_bytes(), Ordering::Relaxed);
            surface
        });
    if surface.matches(request.size, request.format) {
        return surface;
    }

    let previous_bytes = surface.payload_bytes();
    *surface = LayerSurface::new(device, request.size, request.format);
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
    surface
}

#[expect(
    clippy::too_many_lines,
    reason = "one atomic repair keeps preflight, mask, replay, and commit visibly contiguous"
)]
fn repair_surface(
    world: &World,
    request: SurfaceRequest,
    resources: &RepairResources<'_>,
    surfaces: &mut HashMap<RetainedViewEntity, LayerSurface>,
    ctx: &mut RenderContext,
) {
    let phase = resources.phases.get(&request.retained_view_entity);
    let draw_functions = retained_draw_function_ids(world);
    let scene = world.resource::<RetainedUiScene>();
    let boundary_batches = world.resource::<BoundaryBatches>();
    let existing_matches = surfaces
        .get(&request.retained_view_entity)
        .is_some_and(|surface| surface.matches(request.size, request.format));
    let Some(repair_plan) = resources.repair_plans.0.get(&request.paint_entity).cloned() else {
        if !existing_matches && scene.has_visible_records(request.paint_entity, request.bounds) {
            scene.invalidate(request.paint_entity, request.bounds);
        }
        return;
    };
    let global_damage = clipped_damage_to(&repair_plan, request.bounds);
    if global_damage.is_empty() {
        scene.acknowledge(request.paint_entity, &repair_plan);
        if !existing_matches && scene.has_visible_records(request.paint_entity, request.bounds) {
            scene.invalidate(request.paint_entity, request.bounds);
        }
        return;
    }
    let local_damage: Vec<_> = global_damage
        .iter()
        .map(|region| translate_rect(*region, request.bounds.min_x(), request.bounds.min_y()))
        .collect();
    let material_pending = world
        .resource::<RetainedPendingMaterials>()
        .contains(request.paint_entity);
    let has_visible_records = scene.has_visible_records(request.paint_entity, request.bounds);
    let boundary_sources: HashMap<_, _> = if boundary_batches.batches.is_empty() {
        HashMap::default()
    } else {
        phase
            .into_iter()
            .flat_map(|phase| {
                (0..phase.items.len()).map(|index| phase.items.get_index(index).unwrap().1)
            })
            .filter_map(|item| {
                let batch = boundary_batches.batches.get(&item.entity())?;
                let view = resources.boundary_views.views.get(&batch.surface)?;
                let source = surfaces.get(&view.retained_view_entity)?;
                source.slots[source.active]
                    .initialized
                    .then(|| (item.entity(), source.slots[source.active].view.clone()))
            })
            .collect()
    };
    let boundaries_ready = boundary_batches.batches.is_empty()
        || phase.into_iter().all(|phase| {
            (0..phase.items.len()).all(|index| {
                let item = phase.items.get_index(index).unwrap().1;
                !boundary_batches.batches.contains_key(&item.entity())
                    || boundary_sources.contains_key(&item.entity())
            })
        });
    let repair_ready = boundaries_ready
        && !material_pending
        && if has_visible_records {
            phase.is_some_and(|phase| {
                !phase.items.is_empty()
                    && phase_is_repair_ready(phase, resources.pipeline_cache, world, draw_functions)
            })
        } else {
            true
        };
    let surface = surface_for_request(request, resources.counters, surfaces, ctx.render_device());

    let repair_pipelines = repair_ready
        .then(|| {
            (
                resources
                    .pipelines
                    .wipe_pipeline(request.format, resources.pipeline_cache),
                resources
                    .pipelines
                    .copy_pipeline(request.format, resources.pipeline_cache),
                resources.pipelines.mask_pipeline(resources.pipeline_cache),
            )
        })
        .and_then(|(wipe, copy, mask)| {
            Some((
                resources.pipeline_cache.get_render_pipeline(wipe)?,
                resources.pipeline_cache.get_render_pipeline(copy)?,
                resources.pipeline_cache.get_render_pipeline(mask)?,
            ))
        });
    let Some((wipe_pipeline, copy_pipeline, mask_pipeline)) = repair_pipelines else {
        return;
    };

    let pending = 1 - surface.active;
    let mut sync_regions = pending_sync_regions(surface, pending, ctx);
    if sync_regions == local_damage {
        sync_regions.clear();
    }
    let items = world
        .resource::<RetainedItems>()
        .items
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    let mut result = Ok(());
    let mut replayed = 0;
    let mut replayed_quads = 0;
    let has_content = phase_has_drawable_items(phase, world, draw_functions);
    let mut layer_rects = resources
        .layer_rects
        .0
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    let rect_start = u32::try_from(layer_rects.len()).expect("repair rectangle count exceeds u32");
    let width = request.size.x as f32;
    let height = request.size.y as f32;
    layer_rects.extend(sync_regions.iter().chain(&local_damage).map(|region| {
        [
            region.min_x() as f32 * 2.0 / width - 1.0,
            1.0 - region.min_y() as f32 * 2.0 / height,
            region.max_x() as f32 * 2.0 / width - 1.0,
            1.0 - region.max_y() as f32 * 2.0 / height,
        ]
    }));
    let rect_end = layer_rects.len();
    let upload_start = if rect_end > layer_rects.capacity() {
        layer_rects.reserve(rect_end.next_power_of_two(), ctx.render_device());
        0
    } else {
        rect_start as usize
    };
    layer_rects
        .write_buffer_range(resources.render_queue, upload_start..rect_end)
        .expect("the repair rectangle arena was reserved before upload");
    let rect_buffer = layer_rects
        .buffer()
        .expect("nonempty damage must allocate a rectangle buffer");
    let sync_instances =
        u32::try_from(sync_regions.len()).expect("sync rectangle count exceeds u32");
    let damage_instances =
        u32::try_from(local_damage.len()).expect("wipe rectangle count exceeds u32");
    let sync_range = rect_start..rect_start + sync_instances;
    let damage_range = sync_range.end..sync_range.end + damage_instances;
    let copy_bind_group = (!sync_regions.is_empty()).then(|| {
        let source = &surface.slots[surface.active].view;
        ctx.render_device().create_bind_group(
            "retained_ui_copy_bind_group",
            &resources
                .pipeline_cache
                .get_bind_group_layout(&resources.pipelines.final_layout),
            &BindGroupEntries::sequential((source, source, &resources.blit_pipeline.sampler)),
        )
    });
    let mask_bind_group = ctx.render_device().create_bind_group(
        "retained_ui_damage_mask_bind_group",
        &resources
            .pipeline_cache
            .get_bind_group_layout(&resources.pipelines.mask_layout),
        &BindGroupEntries::single(&surface.mask_view),
    );
    let boundary_bind_groups: HashMap<_, _> = boundary_sources
        .iter()
        .map(|(entity, source)| {
            let bind_group = ctx.render_device().create_bind_group(
                "retained_ui_boundary_bind_group",
                &resources
                    .pipeline_cache
                    .get_bind_group_layout(&resources.pipelines.final_layout),
                &BindGroupEntries::sequential((source, source, &resources.blit_pipeline.sampler)),
            );
            (*entity, bind_group)
        })
        .collect();
    {
        let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("retained_ui_damage_mask"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &surface.mask_view,
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
        pass.set_render_pipeline(mask_pipeline);
        pass.set_vertex_buffer(0, rect_buffer.slice(..));
        pass.draw(0..6, damage_range.clone());
    }
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
    pass.set_scissor_rect(0, 0, request.size.x, request.size.y);
    pass.set_vertex_buffer(0, rect_buffer.slice(..));
    if let Some(copy_bind_group) = &copy_bind_group {
        pass.set_render_pipeline(copy_pipeline);
        pass.set_bind_group(0, copy_bind_group, &[]);
        pass.draw(0..6, sync_range);
    }
    pass.set_render_pipeline(wipe_pipeline);
    pass.draw(0..6, damage_range);

    if let Some(phase) = phase.filter(|phase| !phase.items.is_empty()) {
        (replayed, replayed_quads) = prepared_masked_counts(world, request.paint_entity);
        let mut full_run_start = None;
        for index in 0..=phase.items.len() {
            let phase_item =
                (index < phase.items.len()).then(|| phase.items.get_index(index).unwrap().1);
            let prepared = phase_item
                .and_then(|item| item_batch_range(world, item, draw_functions))
                .is_some_and(|range| !range.is_empty());
            let boundary = phase_item
                .and_then(|item| boundary_batches.batches.get(&item.entity()))
                .is_some();
            let mask_clipped = prepared
                && !boundary
                && phase_item.is_some_and(|phase_item| {
                    phase_item.draw_function == draw_functions.core
                        || phase_item.draw_function == draw_functions.retained_shadow
                        || phase_item.draw_function == draw_functions.retained_gradient
                });

            if mask_clipped {
                full_run_start.get_or_insert(index);
                continue;
            }

            if let Some(start) = full_run_start.take() {
                let run = start..index;
                pass.set_scissor_rect(0, 0, request.size.x, request.size.y);
                pass.set_bind_group(1, &mask_bind_group, &[]);
                if let Err(err) = phase.render_range(&mut pass, world, request.view_entity, run) {
                    result = Err(err);
                    break;
                }
            }

            let Some(item) = phase_item.filter(|_| prepared) else {
                continue;
            };
            let coverage = items.get(&item.entity()).map(|item| &item.coverage);
            for (global, local) in global_damage
                .iter()
                .zip(&local_damage)
                .filter(|(global, _)| {
                    coverage.is_none_or(|coverage| coverage_intersects(coverage, **global))
                })
            {
                let _ = global;
                pass.set_scissor_rect(
                    local.min_x() as u32,
                    local.min_y() as u32,
                    (local.max_x() - local.min_x()) as u32,
                    (local.max_y() - local.min_y()) as u32,
                );
                replayed += 1;
                replayed_quads += prepared_counts(world, item, draw_functions).1;
                if let Some(batch) = boundary_batches.batches.get(&item.entity()) {
                    let pipeline = resources
                        .pipeline_cache
                        .get_render_pipeline(item.pipeline)
                        .expect("boundary pipeline was preflighted");
                    let buffer = boundary_batches
                        .instances
                        .buffer()
                        .expect("prepared boundary instances must have a GPU buffer");
                    pass.set_render_pipeline(pipeline);
                    pass.set_bind_group(0, &boundary_bind_groups[&item.entity()], &[]);
                    pass.set_vertex_buffer(0, buffer.slice(..));
                    pass.draw(0..6, batch.range.clone());
                } else if let Err(err) =
                    phase.render_range(&mut pass, world, request.view_entity, index..index + 1)
                {
                    result = Err(err);
                    break;
                }
            }
            if result.is_err() {
                break;
            }
        }
    }
    drop(pass);
    drop(items);

    match result {
        Ok(()) => {
            let repaired_pixels = local_damage.iter().map(PhysicalRect::area).sum::<u64>();
            surface.commit(pending, local_damage, has_content);
            resources.counters.repairs.fetch_add(1, Ordering::Relaxed);
            resources
                .counters
                .repair_pixels
                .fetch_add(repaired_pixels, Ordering::Relaxed);
            resources
                .counters
                .items_replayed
                .fetch_add(replayed, Ordering::Relaxed);
            resources
                .counters
                .quads_replayed
                .fetch_add(replayed_quads, Ordering::Relaxed);
            scene.acknowledge(request.paint_entity, &repair_plan);
        }
        Err(err) => {
            surface.slots[pending].initialized = false;
            surface.slots[pending].generation = 0;
            error!("retained UI repair was deferred: {err:?}");
        }
    }
}

fn rebuild_surface(
    world: &World,
    request: SurfaceRequest,
    resources: &RepairResources<'_>,
    surfaces: &mut HashMap<RetainedViewEntity, LayerSurface>,
    ctx: &mut RenderContext,
) -> bool {
    let Some(repair_plan) = resources.repair_plans.0.get(&request.paint_entity).cloned() else {
        return false;
    };
    if world
        .resource::<RetainedPendingMaterials>()
        .contains(request.paint_entity)
    {
        return false;
    }
    let Some(phase) = resources.phases.get(&request.retained_view_entity) else {
        return false;
    };
    let draw_functions = retained_draw_function_ids(world);
    let scene = world.resource::<RetainedUiScene>();
    if scene.has_visible_records(request.paint_entity, request.bounds)
        && (phase.items.is_empty()
            || !phase_is_repair_ready(phase, resources.pipeline_cache, world, draw_functions))
    {
        return false;
    }
    let surface = surface_for_request(request, resources.counters, surfaces, ctx.render_device());
    let pending = 1 - surface.active;
    let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("retained_ui_full_rebuild"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: &surface.slots[pending].view,
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
    let result = phase.render(&mut pass, world, request.view_entity);
    drop(pass);
    if let Err(err) = result {
        surface.slots[pending].initialized = false;
        surface.slots[pending].generation = 0;
        error!("retained UI full rebuild was deferred: {err:?}");
        return false;
    }

    let (mut items, mut quads) = prepared_masked_counts(world, request.paint_entity);
    for index in 0..phase.items.len() {
        let item = phase.items.get_index(index).unwrap().1;
        if draw_functions.is_core(item.draw_function)
            || item.draw_function == draw_functions.retained_shadow
            || item.draw_function == draw_functions.retained_gradient
        {
            continue;
        }
        let counts = prepared_counts(world, item, draw_functions);
        items += counts.0;
        quads += counts.1;
    }
    let full = PhysicalRect::from_min_max(0, 0, request.size.x as i32, request.size.y as i32)
        .expect("retained UI surfaces are nonempty");
    surface.commit(pending, vec![full], true);
    resources.counters.repairs.fetch_add(1, Ordering::Relaxed);
    resources
        .counters
        .repair_pixels
        .fetch_add(full.area(), Ordering::Relaxed);
    resources
        .counters
        .items_replayed
        .fetch_add(items, Ordering::Relaxed);
    resources
        .counters
        .quads_replayed
        .fetch_add(quads, Ordering::Relaxed);
    scene.acknowledge(request.paint_entity, &repair_plan);
    true
}

#[expect(
    clippy::too_many_arguments,
    reason = "render pass inputs are independent resources"
)]
fn retained_ui_pass(
    world: &World,
    view: ViewQuery<(Entity, &UiCameraView, &ExtractedCamera)>,
    ui_view_query: Query<(&ExtractedView, &UiViewTarget)>,
    transparent_render_phases: Res<ViewSortedRenderPhases<TransparentUi>>,
    pipelines: Res<RetainedUiPipelines>,
    pipeline_cache: Res<PipelineCache>,
    blit_pipeline: Res<BlitPipeline>,
    overlays: Res<ViewOutputOverlays>,
    surfaces: Res<LayerSurfaces>,
    counters: Res<RetainedUiLayerCounters>,
    render_queue: Res<RenderQueue>,
    layer_rects: Res<LayerRects>,
    repair_plans: Res<RetainedRepairPlans>,
    mut ctx: RenderContext,
) {
    let (main_view_entity, ui_camera_view, camera) = view.into_inner();
    overlays.set(main_view_entity, None);
    if matches!(camera.output_mode, bevy::camera::CameraOutputMode::Skip) {
        return;
    }
    let ui_view_entity = ui_camera_view.0;
    let Ok((extracted_view, ui_view_target)) = ui_view_query.get(ui_view_entity) else {
        return;
    };
    let Some(size) = camera.physical_viewport_size else {
        return;
    };

    let boundary_views = world.resource::<BoundaryViews>();
    let mut boundaries: Vec<_> = boundary_views
        .active()
        .filter(|boundary| boundary.source_camera == ui_view_target.0)
        .cloned()
        .collect();
    boundaries.sort_by_key(|boundary| {
        core::cmp::Reverse(boundary_depth(boundary.surface, boundary_views))
    });
    let repair_resources = RepairResources {
        phases: &transparent_render_phases,
        pipelines: &pipelines,
        pipeline_cache: &pipeline_cache,
        blit_pipeline: &blit_pipeline,
        counters: &counters,
        render_queue: &render_queue,
        layer_rects: &layer_rects,
        repair_plans: &repair_plans,
        boundary_views,
    };
    if !boundaries.is_empty() {
        let mut surfaces = surfaces.0.lock().unwrap_or_else(PoisonError::into_inner);
        for boundary in boundaries {
            repair_surface(
                world,
                SurfaceRequest {
                    view_entity: boundary.surface,
                    retained_view_entity: boundary.retained_view_entity,
                    paint_entity: boundary.surface,
                    bounds: boundary.rect,
                    size: boundary.size(),
                    format: boundary.format,
                },
                &repair_resources,
                &mut surfaces,
                &mut ctx,
            );
        }
    }

    let request = SurfaceRequest {
        view_entity: ui_view_entity,
        retained_view_entity: extracted_view.retained_view_entity,
        paint_entity: ui_view_target.0,
        bounds: target_rect(size),
        size,
        format: extracted_view.target_format,
    };
    let mut surfaces = surfaces.0.lock().unwrap_or_else(PoisonError::into_inner);
    let full_rebuild = world
        .resource::<RetainedFullRebuilds>()
        .0
        .contains(&request.paint_entity);
    if full_rebuild {
        rebuild_surface(world, request, &repair_resources, &mut surfaces, &mut ctx);
    } else {
        repair_surface(world, request, &repair_resources, &mut surfaces, &mut ctx);
    }
    let Some(surface) = surfaces.get(&extracted_view.retained_view_entity) else {
        return;
    };
    if surface.has_content {
        overlays.set(
            main_view_entity,
            Some(surface.slots[surface.active].view.clone()),
        );
        counters.composites.fetch_add(1, Ordering::Relaxed);
        counters
            .ui_sample_pixels
            .fetch_add(u64::from(size.x) * u64::from(size.y), Ordering::Relaxed);
    }
}
