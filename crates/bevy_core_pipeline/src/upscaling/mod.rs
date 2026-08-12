use crate::blit::{BlitPipeline, BlitPipelineKey};
use bevy_app::prelude::*;
use bevy_camera::CameraOutputMode;
use bevy_ecs::{entity::EntityHashMap, prelude::*};
use bevy_render::{
    camera::ExtractedCamera, render_resource::*, view::ViewTarget, Render, RenderApp,
    RenderStartup, RenderSystems,
};
use std::sync::{Mutex, PoisonError};

mod node;

pub use node::upscaling;

pub struct UpscalingPlugin;

impl Plugin for UpscalingPlugin {
    fn build(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.add_systems(
                Render,
                // This system should probably technically be run *after* all of the other systems
                // that might modify `PipelineCache` via interior mutability, but for now,
                // we've chosen to simply ignore the ambiguities out of a desire for a better refactor
                // and aversion to extensive and intrusive system ordering.
                // See https://github.com/bevyengine/bevy/issues/14770 for more context.
                prepare_view_upscaling_pipelines
                    .in_set(RenderSystems::Prepare)
                    .ambiguous_with_all(),
            );
            render_app.add_systems(RenderStartup, clear_view_upscaling_pipelines);
            render_app.init_resource::<ViewOutputOverlays>();
        }
    }
}

#[derive(Component)]
pub struct ViewUpscalingPipeline {
    plain: CachedRenderPipelineId,
    overlay: Option<CachedRenderPipelineId>,
    key: BlitPipelineKey,
}

/// Optional premultiplied-alpha layers folded into the existing final blit.
#[derive(Resource, Default)]
pub struct ViewOutputOverlays {
    enabled: bool,
    views: Mutex<EntityHashMap<TextureView>>,
}

impl ViewOutputOverlays {
    /// Enables preparation of the overlay pipeline. Call once during renderer setup.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Selects the layer to composite for one view's current frame.
    pub fn set(&self, view: Entity, overlay: Option<TextureView>) {
        let mut views = self.views.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(overlay) = overlay {
            views.insert(view, overlay);
        } else {
            views.remove(&view);
        }
    }

    fn get(&self, view: Entity) -> Option<TextureView> {
        self.views
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&view)
            .cloned()
    }
}

/// This is not required on first startup but is required during render recovery
fn clear_view_upscaling_pipelines(
    mut commands: Commands,
    views: Query<Entity, With<ViewUpscalingPipeline>>,
) {
    for entity in &views {
        commands.entity(entity).remove::<ViewUpscalingPipeline>();
    }
}

fn prepare_view_upscaling_pipelines(
    mut commands: Commands,
    mut pipeline_cache: ResMut<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<BlitPipeline>>,
    blit_pipeline: Res<BlitPipeline>,
    overlays: Res<ViewOutputOverlays>,
    view_targets: Query<(
        Entity,
        &ViewTarget,
        Option<&ExtractedCamera>,
        Option<&ViewUpscalingPipeline>,
    )>,
) {
    for (entity, view_target, camera, maybe_pipeline) in view_targets.iter() {
        let blend_state = if let Some(extracted_camera) = camera {
            match extracted_camera.output_mode {
                CameraOutputMode::Skip => None,
                CameraOutputMode::Write { blend_state, .. } => {
                    match blend_state {
                        None => {
                            // Auto-detect: the first camera to render to this output
                            // (sorted_camera_index_for_target == 0) uses replace mode;
                            // subsequent cameras default to alpha blending so they don't
                            // accidentally overwrite earlier cameras' output.
                            if extracted_camera.sorted_camera_index_for_target > 0 {
                                Some(BlendState::ALPHA_BLENDING)
                            } else {
                                None
                            }
                        }
                        _ => blend_state,
                    }
                }
            }
        } else {
            None
        };

        let Some(target_format) = view_target.out_texture_view_format() else {
            continue;
        };

        let key = BlitPipelineKey {
            target_format,
            blend_state,
            samples: 1,
            premultiplied_overlay: false,
            source_space: view_target.compositing_space,
        };

        if maybe_pipeline.is_none_or(|pipeline| pipeline.key != key) {
            let plain = pipelines.specialize(&pipeline_cache, &blit_pipeline, key);
            let overlay = overlays.enabled.then(|| {
                pipelines.specialize(
                    &pipeline_cache,
                    &blit_pipeline,
                    BlitPipelineKey {
                        premultiplied_overlay: true,
                        ..key
                    },
                )
            });

            // Ensure the pipeline is loaded before continuing the frame to prevent frames without
            // any GPU work submitted
            pipeline_cache.block_on_render_pipeline(plain);
            if let Some(overlay) = overlay {
                pipeline_cache.block_on_render_pipeline(overlay);
            }

            commands.entity(entity).insert(ViewUpscalingPipeline {
                plain,
                overlay,
                key,
            });
        }
    }
}
