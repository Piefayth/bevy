use crate::{
    core_3d::Transparent3d,
    oit::{resolve::OitResolvePipelineId, OrderIndependentTransparencySettings},
    skybox::{SkyboxBindGroup, SkyboxPipelineId},
};
use bevy_camera::{MainPassResolutionOverride, Viewport};
use bevy_ecs::prelude::*;
use bevy_log::error;
#[cfg(feature = "trace")]
use bevy_log::info_span;
use bevy_render::{
    camera::ExtractedCamera,
    diagnostic::RecordDiagnostics,
    render_phase::ViewSortedRenderPhases,
    render_resource::{PipelineCache, RenderPassDescriptor, StoreOp},
    renderer::{RenderContext, ViewQuery},
    view::{ExtractedView, ViewDepthTexture, ViewTarget, ViewUniformOffset},
};

pub fn main_transparent_pass_3d(
    world: &World,
    view: ViewQuery<(
        &ExtractedCamera,
        &ExtractedView,
        &ViewTarget,
        &ViewDepthTexture,
        Option<&MainPassResolutionOverride>,
        Has<OrderIndependentTransparencySettings>,
        Option<&OitResolvePipelineId>,
        Option<&SkyboxPipelineId>,
        Option<&SkyboxBindGroup>,
        &ViewUniformOffset,
    )>,
    transparent_phases: Res<ViewSortedRenderPhases<Transparent3d>>,
    mut ctx: RenderContext,
) {
    let view_entity = view.entity();

    let (
        camera,
        extracted_view,
        target,
        depth,
        resolution_override,
        has_oit,
        oit_resolve_pipeline_id,
        skybox_pipeline,
        skybox_bind_group,
        view_uniform_offset,
    ) = view.into_inner();

    let Some(transparent_phase) = transparent_phases.get(&extracted_view.retained_view_entity)
    else {
        return;
    };

    // VENDORED CHANGE: this is THE main pass now. The opaque/alpha-mask pass
    // skips itself when it has nothing to draw (in this game: always — the
    // flat pipeline queues everything transparent-phase), so this pass runs
    // unconditionally: it owns the first-use CLEAR of the color and depth
    // attachments, and it draws the skybox first, where the opaque pass used
    // to. One pass over the tile memory instead of two.
    {
        #[cfg(feature = "trace")]
        let _main_transparent_pass_3d_span = info_span!("main_transparent_pass_3d").entered();

        let diagnostics = ctx.diagnostic_recorder();
        let diagnostics = diagnostics.as_deref();

        // We can't run the transparent phase if OitResolvePipelineId is not
        // ready — we'd write `oit_atomic_counter`/`oit_heads` without
        // resetting them, corrupting the linked list on the next pass. But
        // the PASS itself must still begin: it owns the frame's only clear
        // (and the skybox), so warmup skips the phase draws, never the pass.
        let oit_blocked = has_oit
            && !transparent_phase.items.is_empty()
            && !oit_resolve_pipeline_id.is_some_and(|id| {
                world
                    .resource::<PipelineCache>()
                    .get_render_pipeline(id.0)
                    .is_some()
            });

        let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("main_transparent_pass_3d"),
            color_attachments: &[Some(target.get_color_attachment())],
            // NOTE: store is set to `true` as a workaround for issue #3776,
            // https://github.com/bevyengine/bevy/issues/3776
            // so that wgpu does not clear the depth buffer.
            depth_stencil_attachment: Some(depth.get_attachment(StoreOp::Store)),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        let pass_span = diagnostics.pass_span(&mut render_pass, "main_transparent_pass_3d");

        if let Some(viewport) =
            Viewport::from_viewport_and_override(camera.viewport.as_ref(), resolution_override)
        {
            render_pass.set_camera_viewport(&viewport);
        }

        // The sky first, under everything: fullscreen at the far plane, no
        // depth write — phase items paint over it in sorted order exactly as
        // they painted over the opaque pass's skybox before.
        if let (Some(skybox_pipeline), Some(SkyboxBindGroup(skybox_bind_group))) =
            (skybox_pipeline, skybox_bind_group)
        {
            let pipeline_cache = world.resource::<PipelineCache>();
            if let Some(pipeline) = pipeline_cache.get_render_pipeline(skybox_pipeline.0) {
                render_pass.set_render_pipeline(pipeline);
                render_pass.set_bind_group(
                    0,
                    &skybox_bind_group.0,
                    &[view_uniform_offset.offset, skybox_bind_group.1],
                );
                render_pass.draw(0..3, 0..1);
            }
        }

        if !transparent_phase.items.is_empty()
            && !oit_blocked
            && let Err(err) = transparent_phase.render(&mut render_pass, world, view_entity)
        {
            error!("Error encountered while rendering the transparent phase {err:?}");
        }

        pass_span.end(&mut render_pass);
    }

    // WebGL2 quirk: if ending with a render pass with a custom viewport, the viewport isn't
    // reset for the next render pass so add an empty render pass without a custom viewport
    #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
    if camera.viewport.is_some() {
        #[cfg(feature = "trace")]
        let _reset_viewport_pass_3d = info_span!("reset_viewport_pass_3d").entered();
        let pass_descriptor = RenderPassDescriptor {
            label: Some("reset_viewport_pass_3d"),
            color_attachments: &[Some(target.get_color_attachment())],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        };

        ctx.command_encoder().begin_render_pass(&pass_descriptor);
    }
}
