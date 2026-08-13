use crate::{
    blit::BlitPipeline,
    upscaling::{ViewOutputOverlays, ViewUpscalingPipeline},
};
use bevy_camera::{CameraOutputMode, ClearColor, ClearColorConfig};
use bevy_ecs::prelude::*;
use bevy_render::{
    camera::ExtractedCamera,
    diagnostic::RecordDiagnostics,
    render_resource::{BindGroup, PipelineCache, RenderPassDescriptor, TextureViewId},
    renderer::{RenderContext, ViewQuery},
    view::ViewTarget,
};

#[derive(Default)]
pub struct UpscalingBindGroupCache {
    plain: Option<(TextureViewId, BindGroup)>,
    overlay: Option<(TextureViewId, TextureViewId, BindGroup)>,
}

pub fn upscaling(
    view: ViewQuery<(
        Entity,
        &ViewTarget,
        &ViewUpscalingPipeline,
        Option<&ExtractedCamera>,
    )>,
    pipeline_cache: Res<PipelineCache>,
    blit_pipeline: Res<BlitPipeline>,
    overlays: Res<ViewOutputOverlays>,
    clear_color_global: Res<ClearColor>,
    mut cache: Local<UpscalingBindGroupCache>,
    mut ctx: RenderContext,
) {
    let (view_entity, target, upscaling_target, camera) = view.into_inner();

    let clear_color = if let Some(camera) = camera {
        match camera.output_mode {
            CameraOutputMode::Write { clear_color, .. } => clear_color,
            CameraOutputMode::Skip => return,
        }
    } else {
        ClearColorConfig::Default
    };
    let clear_color = match clear_color {
        ClearColorConfig::Default => Some(clear_color_global.0),
        ClearColorConfig::Custom(color) => Some(color),
        ClearColorConfig::None => None,
    };
    let converted_clear_color = clear_color.map(Into::into);

    // texture to be upscaled to the output texture
    let main_texture_view = target.main_texture_view();

    let overlay = overlays.get(view_entity);
    let (pipeline_id, bind_group) =
        if let (Some(overlay), Some(pipeline_id)) = (overlay.as_ref(), upscaling_target.overlay) {
            let bind_group = match &mut cache.overlay {
                Some((main_id, overlay_id, bind_group))
                    if main_texture_view.id() == *main_id && overlay.view.id() == *overlay_id =>
                {
                    bind_group
                }
                cached => {
                    let bind_group = blit_pipeline.create_overlay_bind_group(
                        ctx.render_device(),
                        main_texture_view,
                        &overlay.view,
                        &pipeline_cache,
                    );
                    let (_, _, bind_group) =
                        cached.insert((main_texture_view.id(), overlay.view.id(), bind_group));
                    bind_group
                }
            };
            (pipeline_id, bind_group)
        } else {
            let bind_group = match &mut cache.plain {
                Some((id, bind_group)) if main_texture_view.id() == *id => bind_group,
                cached => {
                    let bind_group = blit_pipeline.create_bind_group(
                        ctx.render_device(),
                        main_texture_view,
                        &pipeline_cache,
                    );
                    let (_, bind_group) = cached.insert((main_texture_view.id(), bind_group));
                    bind_group
                }
            };
            (upscaling_target.plain, bind_group)
        };

    let Some(out_attachment) = target.out_texture_color_attachment(converted_clear_color) else {
        return;
    };

    let pass_descriptor = RenderPassDescriptor {
        label: Some("upscaling"),
        color_attachments: &[Some(out_attachment)],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    };

    let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline_id) else {
        // we need to do some work on the swapchain to avoid pink screen uninit on macos
        #[cfg(target_os = "macos")]
        ctx.command_encoder().begin_render_pass(&pass_descriptor);
        return;
    };

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let time_span = diagnostics.time_span(ctx.command_encoder(), "upscaling");

    {
        let mut render_pass = ctx.command_encoder().begin_render_pass(&pass_descriptor);

        // A full-target overlay must not be scissored to the camera's
        // viewport — the interface owns the whole output even when the
        // world renders letterboxed.
        let full_target_overlay = overlay.as_ref().is_some_and(|overlay| overlay.fills_target);
        if !full_target_overlay
            && let Some(camera) = camera
            && let Some(viewport) = &camera.viewport
        {
            let size = viewport.physical_size;
            let position = viewport.physical_position;
            render_pass.set_scissor_rect(position.x, position.y, size.x, size.y);
        }

        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }

    time_span.end(ctx.command_encoder());
}
