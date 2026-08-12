//! Persistent UI layer and composition.

use crate::background::extract_retained_backgrounds;
use crate::border::extract_retained_borders;
use crate::core::{
    prepare_retained_core, queue_retained_core, register_retained_core, DrawRetainedCore,
    RetainedCore,
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
    cleanup_retained_ui, extract_retained_placements, replay_retained_ui, RetainedItems,
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
    math::UVec2,
    mesh::{VertexBufferLayout, VertexFormat},
    prelude::{App, Plugin, Resource, World},
    render::{
        camera::ExtractedCamera,
        diagnostic::RecordDiagnostics,
        render_asset::RenderAssets,
        render_phase::{DrawFunctionId, DrawFunctions, PhaseItem, ViewSortedRenderPhases},
        render_resource::{
            binding_types::{sampler, texture_2d},
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
            BlendState, BufferUsages, CachedRenderPipelineId, ColorTargetState, ColorWrites,
            Extent3d, FragmentState, LoadOp, Operations, PipelineCache, RawBufferVec,
            RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor,
            SamplerBindingType, ShaderStages, StoreOp, Texture, TextureDescriptor,
            TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureView,
            TextureViewDescriptor, TextureViewId, VertexAttribute, VertexState, VertexStepMode,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
        texture::{FallbackImageZero, GpuImage},
        view::{ExtractedView, RetainedViewEntity, ViewTarget},
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
            .init_resource::<RetainedRepairPlans>()
            .init_resource::<RetainedUiPaintCounters>()
            .init_resource::<LayerRects>()
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
                    .before(extract_retained_text)
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
                extract_retained_placements
                    .before(RenderUiSystems::ExtractBoxShadows)
                    .before(RenderUiSystems::ExtractBackgrounds)
                    .before(extract_retained_viewports)
                    .before(extract_retained_gradients),
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
            .add_systems(Render, queue_retained_core.in_set(RenderSystems::Queue))
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
                prepare_retained_gradients.in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(
                Render,
                prepare_retained_shadows.in_set(RenderSystems::PrepareBindGroups),
            )
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
    vertex: VertexState,
    rect_vertex: VertexState,
    final_pipelines: Mutex<HashMap<(BlitPipelineKey, bool), CachedRenderPipelineId>>,
    wipe_pipelines: Mutex<HashMap<TextureFormat, CachedRenderPipelineId>>,
    copy_pipelines: Mutex<HashMap<TextureFormat, CachedRenderPipelineId>>,
    mask_pipeline: Mutex<Option<CachedRenderPipelineId>>,
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
}

fn init_retained_ui_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    fullscreen_shader: Res<FullscreenShader>,
) {
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
        vertex: fullscreen_shader.to_vertex_state(),
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
        final_pipelines: Mutex::new(HashMap::new()),
        wipe_pipelines: Mutex::new(HashMap::new()),
        copy_pipelines: Mutex::new(HashMap::new()),
        mask_pipeline: Mutex::new(None),
    });
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
    let sampled_images = world.resource::<RetainedSampledImages>().lock();
    let gpu_images = world.resource::<RenderAssets<GpuImage>>();
    let image_unavailable = |entity| {
        let mut unavailable = false;
        if let Some(metadata) = items.get(&entity) {
            for image in &metadata.sampled_images {
                if gpu_images.get(*image).is_some() {
                    continue;
                }
                if sampled_images.is_pending(*image) {
                    return None;
                }
                unavailable = true;
            }
        }
        Some(unavailable)
    };
    for index in 0..phase.items.len() {
        let item = phase.items.get_index(index).unwrap().1;
        if item.draw_function == draw_functions.core {
            if item_batch_range(world, item, draw_functions).is_none() {
                let Some(unavailable_image) = image_unavailable(item.entity()) else {
                    return false;
                };
                if unavailable_image {
                    continue;
                }
                return false;
            }
            if pipeline_cache.get_render_pipeline(item.pipeline).is_none() {
                return false;
            }
            continue;
        }
        if item_batch_range(world, item, draw_functions).is_none() {
            let Some(unavailable_image) = image_unavailable(item.entity()) else {
                return false;
            };
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

#[derive(Clone, Copy)]
struct RetainedDrawFunctionIds {
    core: DrawFunctionId,
    retained_gradient: DrawFunctionId,
    retained_shadow: DrawFunctionId,
    texture_slice: DrawFunctionId,
}

fn retained_draw_function_ids(world: &World) -> RetainedDrawFunctionIds {
    let draw_functions = world.resource::<DrawFunctions<TransparentUi>>().read();
    RetainedDrawFunctionIds {
        core: draw_functions.id::<DrawRetainedCore>(),
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
    if item.draw_function == draw_functions.core {
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

fn prepared_quad_count(
    world: &World,
    item: &TransparentUi,
    draw_functions: RetainedDrawFunctionIds,
) -> u64 {
    if item.draw_function == draw_functions.core {
        return world
            .resource::<RetainedCore>()
            .batch_range(item.entity())
            .map(|range| u64::try_from(range.len()).expect("prepared UI quad count exceeds u64"))
            .unwrap_or_default();
    }
    if item.draw_function == draw_functions.retained_shadow {
        return world
            .resource::<RetainedShadows>()
            .batch_range(item.entity())
            .map(|range| u64::try_from(range.len()).expect("prepared shadow count exceeds u64"))
            .unwrap_or_default();
    }
    if item.draw_function == draw_functions.retained_gradient {
        return world
            .resource::<RetainedGradients>()
            .batch_range(item.entity())
            .map(|range| u64::try_from(range.len()).expect("prepared gradient count exceeds u64"))
            .unwrap_or_default();
    }
    let indices = item_batch_range(world, item, draw_functions)
        .map(|range| range.len())
        .unwrap_or_default();
    u64::try_from(indices / 6).expect("prepared UI quad count exceeds u64")
}

fn prepared_item_count(
    world: &World,
    item: &TransparentUi,
    draw_functions: RetainedDrawFunctionIds,
) -> u64 {
    let count = if item.draw_function == draw_functions.core {
        world.resource::<RetainedCore>().item_count(item.entity())
    } else if item.draw_function == draw_functions.retained_shadow {
        world
            .resource::<RetainedShadows>()
            .item_count(item.entity())
    } else if item.draw_function == draw_functions.retained_gradient {
        world
            .resource::<RetainedGradients>()
            .item_count(item.entity())
    } else {
        Some(1)
    };
    u64::from(count.unwrap_or_default())
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
    render_queue: Res<RenderQueue>,
    layer_rects: Res<LayerRects>,
    repair_plans: Res<RetainedRepairPlans>,
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
    let repair_plan = repair_plans.0.get(&ui_view_target.0).cloned();
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

    let repair_pipelines = (repair_requested && repair_ready)
        .then(|| {
            (
                pipelines.wipe_pipeline(extracted_view.target_format, &pipeline_cache),
                pipelines.copy_pipeline(extracted_view.target_format, &pipeline_cache),
                pipelines.mask_pipeline(&pipeline_cache),
            )
        })
        .and_then(|(wipe, copy, mask)| {
            Some((
                pipeline_cache.get_render_pipeline(wipe)?,
                pipeline_cache.get_render_pipeline(copy)?,
                pipeline_cache.get_render_pipeline(mask)?,
            ))
        });
    if let Some((wipe_pipeline, copy_pipeline, mask_pipeline)) = repair_pipelines {
        let pending = 1 - surface.active;
        let mut sync_regions = pending_sync_regions(surface, pending, &mut ctx);
        if sync_regions == damage_regions {
            sync_regions.clear();
        }
        let items = world
            .resource::<RetainedItems>()
            .0
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let mut result = Ok(());
        let mut replayed = 0;
        let mut replayed_quads = 0;
        let has_content = phase_has_drawable_items(phase, world, draw_functions);
        let mut layer_rects = layer_rects.0.lock().unwrap_or_else(PoisonError::into_inner);
        layer_rects.clear();
        let width = size.x as f32;
        let height = size.y as f32;
        layer_rects.extend(sync_regions.iter().chain(&damage_regions).map(|region| {
            [
                region.min_x() as f32 * 2.0 / width - 1.0,
                1.0 - region.min_y() as f32 * 2.0 / height,
                region.max_x() as f32 * 2.0 / width - 1.0,
                1.0 - region.max_y() as f32 * 2.0 / height,
            ]
        }));
        layer_rects.write_buffer(ctx.render_device(), &render_queue);
        let rect_buffer = layer_rects
            .buffer()
            .expect("nonempty damage must allocate a rectangle buffer");
        let sync_instances =
            u32::try_from(sync_regions.len()).expect("sync rectangle count exceeds u32");
        let damage_instances =
            u32::try_from(damage_regions.len()).expect("wipe rectangle count exceeds u32");
        let copy_bind_group = (!sync_regions.is_empty()).then(|| {
            let source = &surface.slots[surface.active].view;
            ctx.render_device().create_bind_group(
                "retained_ui_copy_bind_group",
                &pipeline_cache.get_bind_group_layout(&pipelines.final_layout),
                &BindGroupEntries::sequential((source, source, &blit_pipeline.sampler)),
            )
        });
        let mask_bind_group = ctx.render_device().create_bind_group(
            "retained_ui_damage_mask_bind_group",
            &pipeline_cache.get_bind_group_layout(&pipelines.mask_layout),
            &BindGroupEntries::single(&surface.mask_view),
        );
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
            pass.draw(0..6, sync_instances..sync_instances + damage_instances);
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
        pass.set_scissor_rect(0, 0, size.x, size.y);
        pass.set_vertex_buffer(0, rect_buffer.slice(..));
        if let Some(copy_bind_group) = &copy_bind_group {
            pass.set_render_pipeline(copy_pipeline);
            pass.set_bind_group(0, copy_bind_group, &[]);
            pass.draw(0..6, 0..sync_instances);
        }
        pass.set_render_pipeline(wipe_pipeline);
        pass.draw(0..6, sync_instances..sync_instances + damage_instances);

        if let Some(phase) = phase.filter(|phase| !phase.items.is_empty()) {
            let mut full_run_start = None;
            for index in 0..=phase.items.len() {
                let phase_item =
                    (index < phase.items.len()).then(|| phase.items.get_index(index).unwrap().1);
                let prepared = phase_item
                    .and_then(|item| item_batch_range(world, item, draw_functions))
                    .is_some_and(|range| !range.is_empty());
                let mask_clipped = prepared
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
                    pass.set_scissor_rect(0, 0, size.x, size.y);
                    pass.set_bind_group(1, &mask_bind_group, &[]);
                    for index in run.clone() {
                        let item = phase.items.get_index(index).unwrap().1;
                        replayed += prepared_item_count(world, item, draw_functions);
                        replayed_quads += prepared_quad_count(world, item, draw_functions);
                    }
                    if let Err(err) = phase.render_range(&mut pass, world, ui_view_entity, run) {
                        result = Err(err);
                        break;
                    }
                }

                let Some(item) = phase_item.filter(|_| prepared) else {
                    continue;
                };
                let coverage = items.get(&item.entity()).map(|item| &item.coverage);
                for region in damage_regions.iter().copied().filter(|region| {
                    coverage.is_none_or(|coverage| coverage_intersects(coverage, *region))
                }) {
                    pass.set_scissor_rect(
                        region.min_x() as u32,
                        region.min_y() as u32,
                        (region.max_x() - region.min_x()) as u32,
                        (region.max_y() - region.min_y()) as u32,
                    );
                    replayed += 1;
                    replayed_quads += prepared_quad_count(world, item, draw_functions);
                    if let Err(err) =
                        phase.render_range(&mut pass, world, ui_view_entity, index..index + 1)
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
