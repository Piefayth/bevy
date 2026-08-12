//! Damage-clipped instanced drawing for retained box shadows.

use crate::scene::RetainedBoxShadowItem;
use bevy::{
    app::SubApp,
    asset::{load_embedded_asset, AssetServer, Handle},
    color::ColorToComponents,
    ecs::{
        entity::Entity,
        query::With,
        schedule::IntoScheduleConfigs,
        system::{lifetimeless::*, Commands, Res, ResMut, SystemParamItem},
    },
    math::FloatOrd,
    mesh::{VertexBufferLayout, VertexFormat},
    platform::collections::HashMap,
    render::{
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
        view::{ExtractedView, ViewUniformOffset, ViewUniforms},
        GpuResourceAppExt,
    },
    shader::{Shader, ShaderDefVal},
    ui_render::{TransparentUi, UiCameraView, UiPipeline},
};
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuShadowInstance {
    transform: [f32; 4],
    translation: [f32; 2],
    color: [f32; 4],
    size: [f32; 2],
    radius: [f32; 4],
    blur: f32,
    bounds: [f32; 2],
}

struct ShadowRun {
    render_entity: Entity,
    main_entity: MainEntity,
    camera: Entity,
    z_order: f32,
    samples: u32,
    instances: core::ops::Range<usize>,
    items: u32,
}

#[derive(bevy::prelude::Resource, Default)]
pub(crate) struct RetainedShadowRuns {
    runs: Vec<ShadowRun>,
    instances: Vec<GpuShadowInstance>,
}

impl RetainedShadowRuns {
    pub(crate) fn clear(&mut self) {
        self.runs.clear();
        self.instances.clear();
    }

    pub(crate) fn push(
        &mut self,
        current: &mut Option<usize>,
        draw: &crate::scene::RetainedDraw,
        item: &RetainedBoxShadowItem,
    ) {
        let transform = draw.transform.to_cols_array();
        let instance = GpuShadowInstance {
            transform: [transform[0], transform[1], transform[2], transform[3]],
            translation: [transform[4], transform[5]],
            color: item.color().to_f32_array(),
            size: item.size().to_array(),
            radius: item.radius().into(),
            blur: item.blur_radius(),
            bounds: item.bounds().to_array(),
        };
        let start = self.instances.len();
        self.instances.push(instance);
        let end = self.instances.len();

        if let Some(index) = *current {
            let run = &mut self.runs[index];
            if run.camera == draw.camera
                && run.samples == item.samples()
                && run.z_order.to_bits() == draw.z_order.to_bits()
            {
                run.instances.end = end;
                run.items = run
                    .items
                    .checked_add(1)
                    .expect("retained shadow run item count exceeds u32");
                return;
            }
        }
        self.runs.push(ShadowRun {
            render_entity: draw.render_entity,
            main_entity: draw.main_entity,
            camera: draw.camera,
            z_order: draw.z_order,
            samples: item.samples(),
            instances: start..end,
            items: 1,
        });
        *current = Some(self.runs.len() - 1);
    }
}

struct ShadowBatch {
    range: core::ops::Range<u32>,
    items: u32,
}

#[derive(bevy::prelude::Resource)]
pub(crate) struct RetainedShadows {
    instances: RawBufferVec<GpuShadowInstance>,
    batches: HashMap<Entity, ShadowBatch>,
    view_bind_group: Option<BindGroup>,
}

impl Default for RetainedShadows {
    fn default() -> Self {
        Self {
            instances: RawBufferVec::new(BufferUsages::VERTEX),
            batches: HashMap::default(),
            view_bind_group: None,
        }
    }
}

impl RetainedShadows {
    pub(crate) fn batch_range(&self, entity: Entity) -> Option<core::ops::Range<u32>> {
        self.batches.get(&entity).map(|batch| batch.range.clone())
    }

    pub(crate) fn item_count(&self, entity: Entity) -> Option<u32> {
        self.batches.get(&entity).map(|batch| batch.items)
    }
}

#[derive(bevy::prelude::Resource)]
pub(crate) struct ShadowPipeline {
    view_layout: BindGroupLayoutDescriptor,
    mask_layout: BindGroupLayoutDescriptor,
    shader: Handle<Shader>,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub(crate) struct ShadowPipelineKey {
    target_format: bevy::render::render_resource::TextureFormat,
    samples: u32,
}

impl SpecializedRenderPipeline for ShadowPipeline {
    type Key = ShadowPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let layout = VertexBufferLayout::from_vertex_formats(
            VertexStepMode::Instance,
            vec![
                VertexFormat::Float32x4,
                VertexFormat::Float32x2,
                VertexFormat::Float32x4,
                VertexFormat::Float32x2,
                VertexFormat::Float32x4,
                VertexFormat::Float32,
                VertexFormat::Float32x2,
            ],
        );
        let shader_defs = vec![ShaderDefVal::UInt("SHADOW_SAMPLES".into(), key.samples)];
        RenderPipelineDescriptor {
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: shader_defs.clone(),
                buffers: vec![layout],
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
            layout: vec![self.view_layout.clone(), self.mask_layout.clone()],
            label: Some("retained_ui_shadow_pipeline".into()),
            ..Default::default()
        }
    }
}

fn init_pipeline(mut commands: Commands, assets: Res<AssetServer>, ui_pipeline: Res<UiPipeline>) {
    commands.insert_resource(ShadowPipeline {
        view_layout: ui_pipeline.view_layout.clone(),
        mask_layout: crate::mask::layout(),
        shader: load_embedded_asset!(assets.as_ref(), "shadow.wgsl"),
    });
}

pub(crate) fn queue(
    runs: Res<RetainedShadowRuns>,
    pipeline: Res<ShadowPipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<ShadowPipeline>>,
    mut phases: ResMut<ViewSortedRenderPhases<TransparentUi>>,
    render_views: bevy::ecs::system::Query<&UiCameraView, With<ExtractedView>>,
    camera_views: bevy::ecs::system::Query<&ExtractedView>,
    pipeline_cache: Res<PipelineCache>,
    draw_functions: Res<DrawFunctions<TransparentUi>>,
) {
    let draw_function = draw_functions.read().id::<DrawRetainedShadows>();
    for (index, run) in runs.runs.iter().enumerate() {
        let Ok(default_view) = render_views.get(run.camera) else {
            continue;
        };
        let Ok(view) = camera_views.get(default_view.0) else {
            continue;
        };
        let Some(phase) = phases.get_mut(&view.retained_view_entity) else {
            continue;
        };
        phase.add_transient(TransparentUi {
            draw_function,
            pipeline: pipelines.specialize(
                &pipeline_cache,
                &pipeline,
                ShadowPipelineKey {
                    target_format: view.target_format,
                    samples: run.samples,
                },
            ),
            entity: (run.render_entity, run.main_entity),
            sort_key: FloatOrd(run.z_order),
            index,
            batch_range: 0..1,
            extra_index: PhaseItemExtraIndex::None,
            indexed: false,
        });
    }
}

pub(crate) fn prepare(
    runs: Res<RetainedShadowRuns>,
    mut shadows: ResMut<RetainedShadows>,
    view_uniforms: Res<ViewUniforms>,
    pipeline: Res<ShadowPipeline>,
    pipeline_cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    shadows.instances.clear();
    shadows.batches.clear();
    for run in &runs.runs {
        let start = shadows.instances.len() as u32;
        for &instance in &runs.instances[run.instances.clone()] {
            shadows.instances.push(instance);
        }
        let end = shadows.instances.len() as u32;
        shadows.batches.insert(
            run.render_entity,
            ShadowBatch {
                range: start..end,
                items: run.items,
            },
        );
    }
    if shadows.instances.is_empty() {
        shadows.view_bind_group = None;
        return;
    }
    shadows.instances.write_buffer(&device, &queue);
    shadows.view_bind_group = view_uniforms.uniforms.binding().map(|binding| {
        device.create_bind_group(
            "retained_ui_shadow_view_bind_group",
            &pipeline_cache.get_bind_group_layout(&pipeline.view_layout),
            &BindGroupEntries::single(binding),
        )
    });
}

pub(crate) type DrawRetainedShadows = (
    SetItemPipeline,
    SetShadowViewBindGroup<0>,
    DrawShadowInstances,
);

pub(crate) struct SetShadowViewBindGroup<const I: usize>;

impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetShadowViewBindGroup<I> {
    type Param = SRes<RetainedShadows>;
    type ViewQuery = Read<ViewUniformOffset>;
    type ItemQuery = ();

    fn render<'w>(
        _item: &P,
        view: &'w ViewUniformOffset,
        _entity: Option<()>,
        shadows: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(bind_group) = shadows.into_inner().view_bind_group.as_ref() else {
            return RenderCommandResult::Failure("retained shadow view bind group is unavailable");
        };
        pass.set_bind_group(I, bind_group, &[view.offset]);
        RenderCommandResult::Success
    }
}

pub(crate) struct DrawShadowInstances;

impl<P: PhaseItem> RenderCommand<P> for DrawShadowInstances {
    type Param = SRes<RetainedShadows>;
    type ViewQuery = ();
    type ItemQuery = ();

    fn render<'w>(
        item: &P,
        _view: (),
        _entity: Option<()>,
        shadows: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let shadows = shadows.into_inner();
        let Some(batch) = shadows.batches.get(&item.entity()) else {
            return RenderCommandResult::Skip;
        };
        let Some(instances) = shadows.instances.buffer() else {
            return RenderCommandResult::Failure("retained shadow instances are unavailable");
        };
        pass.set_vertex_buffer(0, instances.slice(..));
        pass.draw(0..6, batch.range.clone());
        RenderCommandResult::Success
    }
}

pub(crate) fn register(app: &mut SubApp) {
    app.init_resource::<RetainedShadowRuns>()
        .init_resource::<RetainedShadows>()
        .init_gpu_resource::<SpecializedRenderPipelines<ShadowPipeline>>()
        .add_render_command::<TransparentUi, DrawRetainedShadows>()
        .add_systems(
            bevy::render::RenderStartup,
            init_pipeline.after(bevy::ui_render::init_ui_pipeline),
        );
}
