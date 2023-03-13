use std::marker::PhantomData;

use super::{UiBatch, UiImageBindGroups, UiMeta};
use crate::{prelude::UiCameraConfig, DefaultCameraView, UiMaterial, RenderUiMaterials};
use bevy_asset::Handle;
use bevy_ecs::{
    prelude::*,
    system::{lifetimeless::*, SystemParamItem}, query::ROQueryItem,
};
use bevy_render::{
    render_graph::*,
    render_phase::*,
    render_resource::{CachedRenderPipelineId, LoadOp, Operations, RenderPassDescriptor},
    renderer::*,
    view::*,
};
use bevy_log::{warn, debug};
use bevy_utils::FloatOrd;

pub struct UiPassNode {
    ui_view_query: QueryState<
        (
            &'static RenderPhase<TransparentUi>,
            &'static ViewTarget,
            Option<&'static UiCameraConfig>,
        ),
        With<ExtractedView>,
    >,
    default_camera_view_query: QueryState<&'static DefaultCameraView>,
}

impl UiPassNode {
    pub const IN_VIEW: &'static str = "view";

    pub fn new(world: &mut World) -> Self {
        Self {
            ui_view_query: world.query_filtered(),
            default_camera_view_query: world.query(),
        }
    }
}

impl Node for UiPassNode {
    fn input(&self) -> Vec<SlotInfo> {
        vec![SlotInfo::new(UiPassNode::IN_VIEW, SlotType::Entity)]
    }

    fn update(&mut self, world: &mut World) {
        self.ui_view_query.update_archetypes(world);
        self.default_camera_view_query.update_archetypes(world);
    }

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        // get the ViewTarget and the RenderPhase<TransparentUi> from the world
        // then actually calls through to render the queried phase
        let input_view_entity = graph.get_input_entity(Self::IN_VIEW)?;

        let Ok((transparent_phase, target, camera_ui)) =
                self.ui_view_query.get_manual(world, input_view_entity)
             else {
                return Ok(());
            };
        if transparent_phase.items.is_empty() {
            return Ok(());
        }
        // Don't render UI for cameras where it is explicitly disabled
        if matches!(camera_ui, Some(&UiCameraConfig { show_ui: false })) {
            return Ok(());
        }

        // use the "default" view entity if it is defined
        let view_entity = if let Ok(default_view) = self
            .default_camera_view_query
            .get_manual(world, input_view_entity)
        {
            default_view.0
        } else {
            input_view_entity
        };
        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("ui_pass"),
            color_attachments: &[Some(target.get_unsampled_color_attachment(Operations {
                load: LoadOp::Load,
                store: true,
            }))],
            depth_stencil_attachment: None,
        });

        transparent_phase.render(&mut render_pass, world, view_entity);

        Ok(())
    }
}

// TransparentUi is a SINGLE ui entity in the render world to be rendered
// Noting that they have their own pipelines and draw functions? Hmmm
pub struct TransparentUi {
    pub sort_key: FloatOrd,
    pub entity: Entity,
    pub pipeline: CachedRenderPipelineId,
    pub draw_function: DrawFunctionId,
}

impl PhaseItem for TransparentUi {
    type SortKey = FloatOrd;

    #[inline]
    fn entity(&self) -> Entity {
        self.entity
    }

    #[inline]
    fn sort_key(&self) -> Self::SortKey {
        self.sort_key
    }

    #[inline]
    fn draw_function(&self) -> DrawFunctionId {
        self.draw_function
    }
}

impl CachedRenderPipelinePhaseItem for TransparentUi {
    #[inline]
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.pipeline
    }
}

// This is a _render command tuple_
// Each element of it is a render command
// This is exported and consumed by app.add_render_command
// It is then available from the DrawFunctions resource and 
    // it is added to the render graph in mod.queue_uinodes
pub type DrawUi<M> = (
    SetItemPipeline,    // provided, configures our PhaseItem (TransparentUi)
    SetUiViewBindGroup<0>,
    SetUiTextureBindGroup<M, 1>,
    SetUiMaterialBindGroup<M, 2>,
    DrawUiNode<M>,
);

pub struct SetUiViewBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetUiViewBindGroup<I> {
    type Param = SRes<UiMeta>;
    type ViewWorldQuery = Read<ViewUniformOffset>;
    type ItemWorldQuery = ();

    fn render<'w>(
        _item: &P,
        view_uniform: &'w ViewUniformOffset,
        _entity: (),
        ui_meta: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        pass.set_bind_group(
            I,
            ui_meta.into_inner().view_bind_group.as_ref().unwrap(),
            &[view_uniform.offset],
        );
        RenderCommandResult::Success
    }
}
pub struct SetUiTextureBindGroup<M: UiMaterial, const I: usize>(PhantomData<M>);
impl<M: UiMaterial, P: PhaseItem, const I: usize> RenderCommand<P> for SetUiTextureBindGroup<M, I> {
    type Param = SRes<UiImageBindGroups>;
    type ViewWorldQuery = ();
    type ItemWorldQuery = Read<UiBatch>;

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: (),
        batch: &'w UiBatch,
        image_bind_groups: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {

        let image_bind_groups = image_bind_groups.into_inner();
        pass.set_bind_group(I, image_bind_groups.values.get(&batch.image).unwrap(), &[]);
        RenderCommandResult::Success
    }
}

pub struct SetUiMaterialBindGroup<M: UiMaterial, const I: usize>(PhantomData<M>);
impl<P: PhaseItem, M: UiMaterial, const I: usize> RenderCommand<P>
    for SetUiMaterialBindGroup<M, I>
{
    type Param = SRes<RenderUiMaterials<M>>;
    type ViewWorldQuery = ();
    type ItemWorldQuery = Read<UiBatch>;

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: (),
        ui_batch: ROQueryItem<'_, Self::ItemWorldQuery>,
        materials: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        // guaranteed to be the correct material by the submitting plugin
        let ui_material = materials.into_inner().get(&ui_batch.material.clone_weak().typed::<M>()).unwrap();
        pass.set_bind_group(I, &ui_material.bind_group, &[]);
        RenderCommandResult::Success
    }
}

pub struct DrawUiNode<M: UiMaterial>(PhantomData<M>);
impl<M: UiMaterial, P: PhaseItem> RenderCommand<P> for DrawUiNode<M> {
    type Param = SRes<UiMeta>;
    type ViewWorldQuery = ();
    type ItemWorldQuery = Read<UiBatch>;

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: (),
        batch: &'w UiBatch,
        ui_meta: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        // set the vertex buffer to the _entire_ vertex buffer in ui meta
        pass.set_vertex_buffer(0, ui_meta.into_inner().vertices.buffer().unwrap().slice(..));
        
        // submit our specific range to be drawn
        pass.draw(batch.range.clone(), 0..1);
        RenderCommandResult::Success
    }
}
