//! Damage-clipped instanced drawing for retained gradients.

use crate::scene::{RetainedFullRebuilds, RetainedGradientItem};
use bevy::{
    app::SubApp,
    asset::{load_embedded_asset, AssetServer, Handle},
    color::{ColorToComponents, Hsla, Hsva, LinearRgba, Oklaba, Oklcha, Srgba},
    ecs::{
        entity::Entity,
        query::With,
        schedule::IntoScheduleConfigs,
        system::{lifetimeless::*, Commands, Res, ResMut, SystemParamItem},
    },
    math::{ops::sin_cos, FloatOrd, Vec2},
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
    shader::Shader,
    ui::InterpolationColorSpace,
    ui_render::{TransparentUi, UiAntiAlias, UiCameraView, UiPipeline},
};
use bytemuck::{Pod, Zeroable};

const RADIAL: u32 = 16;
const FILL_START: u32 = 32;
const FILL_END: u32 = 64;
const CONIC: u32 = 128;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuGradientInstance {
    transform: [f32; 4],
    translation: [f32; 2],
    size: [f32; 2],
    flags: u32,
    radius: [f32; 4],
    border: [f32; 4],
    g_start: [f32; 2],
    direction: [f32; 2],
    start_color: [f32; 4],
    lengths_hint: [f32; 3],
    end_color: [f32; 4],
    clip: [f32; 4],
}

struct GradientRun {
    render_entity: Entity,
    main_entity: MainEntity,
    camera: Entity,
    z_order: f32,
    color_space: InterpolationColorSpace,
    instances: core::ops::Range<usize>,
    items: u32,
}

#[derive(bevy::prelude::Resource, Default)]
pub(crate) struct RetainedGradientRuns {
    runs: Vec<GradientRun>,
    instances: Vec<GpuGradientInstance>,
}

impl RetainedGradientRuns {
    pub(crate) fn clear(&mut self) {
        self.runs.clear();
        self.instances.clear();
    }

    pub(crate) fn push(
        &mut self,
        current: &mut Option<usize>,
        draw: &crate::scene::RetainedDraw,
        color_space: InterpolationColorSpace,
        prepared: &[GpuGradientInstance],
    ) {
        let start = self.instances.len();
        self.instances.extend_from_slice(prepared);
        let end = self.instances.len();
        if start == end {
            return;
        }

        if let Some(index) = *current {
            let run = &mut self.runs[index];
            if run.camera == draw.camera
                && run.color_space == color_space
                && run.z_order.to_bits() == draw.z_order.to_bits()
            {
                run.instances.end = end;
                run.items = run
                    .items
                    .checked_add(1)
                    .expect("retained gradient run item count exceeds u32");
                return;
            }
        }
        self.runs.push(GradientRun {
            render_entity: draw.render_entity,
            main_entity: draw.main_entity,
            camera: draw.camera,
            z_order: draw.z_order,
            color_space,
            instances: start..end,
            items: 1,
        });
        *current = Some(self.runs.len() - 1);
    }
}

pub(crate) fn prepare_gradient_instances(
    draw: &crate::scene::RetainedDraw,
    item: &RetainedGradientItem,
    instances: &mut Vec<GpuGradientInstance>,
) {
    let rect = item.rect();
    let size = rect.size();
    let corners = [
        Vec2::new(-0.5, -0.5) * size,
        Vec2::new(0.5, -0.5) * size,
        Vec2::new(0.5, 0.5) * size,
        Vec2::new(-0.5, 0.5) * size,
    ];
    let (g_start, direction, gradient_flags) = match item.resolved() {
        bevy::ui_render::gradient::ResolvedGradient::Linear { angle } => {
            let corner = ((angle - core::f32::consts::FRAC_PI_2).rem_euclid(core::f32::consts::TAU)
                / core::f32::consts::FRAC_PI_2) as usize;
            let (sin, cos) = sin_cos(angle);
            (corners[corner].to_array(), [sin, -cos], 0)
        }
        bevy::ui_render::gradient::ResolvedGradient::Conic { center, start } => {
            (center.to_array(), [start, 0.0], CONIC)
        }
        bevy::ui_render::gradient::ResolvedGradient::Radial { center, size } => (
            center.to_array(),
            [if size.y != 0.0 { size.x / size.y } else { 1.0 }, 0.0],
            RADIAL,
        ),
    };
    let mut flags = gradient_flags;
    if let bevy::ui_render::NodeType::Border(border_flags) = item.node_type() {
        flags |= border_flags;
    }
    let transform = draw.transform.to_cols_array();
    let border = item.border();
    let radius: [f32; 4] = item.border_radius().into();
    let base = GpuGradientInstance {
        transform: [transform[0], transform[1], transform[2], transform[3]],
        translation: [transform[4], transform[5]],
        size: size.to_array(),
        flags,
        radius,
        border: [
            border.min_inset.x,
            border.min_inset.y,
            border.max_inset.x,
            border.max_inset.y,
        ],
        g_start,
        direction,
        start_color: [0.0; 4],
        lengths_hint: [0.0; 3],
        end_color: [0.0; 4],
        clip: clip_rect(draw.clip),
    };

    let mut stops = item.stops();
    let stop_count = stops.len();
    let Some(mut start_stop) = stops.next() else {
        return;
    };
    let mut segment_count = 0;
    for (index, end_stop) in stops.enumerate() {
        if start_stop.1 == end_stop.1 {
            if index + 2 == stop_count {
                if segment_count > 0 {
                    start_stop.0 = LinearRgba::NONE;
                }
            } else {
                start_stop = end_stop;
                continue;
            }
        }
        let mut segment_flags = flags;
        if start_stop.1 > 0.0 && (index == 0 || segment_count == 0) {
            segment_flags |= FILL_START;
        }
        if index + 2 == stop_count {
            segment_flags |= FILL_END;
        }
        let segment = GpuGradientInstance {
            flags: segment_flags,
            start_color: convert_color(start_stop.0, item.color_space()),
            lengths_hint: [start_stop.1, end_stop.1, start_stop.2],
            end_color: convert_color(end_stop.0, item.color_space()),
            ..base
        };
        instances.push(segment);
        segment_count += 1;
        start_stop = end_stop;
    }
}

impl GpuGradientInstance {
    pub(crate) fn set_placement(
        &mut self,
        transform: bevy::math::Affine2,
        clip: Option<bevy::math::Rect>,
    ) {
        let transform = transform.to_cols_array();
        self.transform = [transform[0], transform[1], transform[2], transform[3]];
        self.translation = [transform[4], transform[5]];
        self.clip = clip_rect(clip);
    }
}

fn clip_rect(clip: Option<bevy::math::Rect>) -> [f32; 4] {
    clip.map_or([-f32::MAX, -f32::MAX, f32::MAX, f32::MAX], |clip| {
        [clip.min.x, clip.min.y, clip.max.x, clip.max.y]
    })
}

fn convert_color(color: LinearRgba, space: InterpolationColorSpace) -> [f32; 4] {
    match space {
        InterpolationColorSpace::Oklaba => {
            let color: Oklaba = color.into();
            [color.lightness, color.a, color.b, color.alpha]
        }
        InterpolationColorSpace::Oklcha | InterpolationColorSpace::OklchaLong => {
            let color: Oklcha = color.into();
            [
                color.lightness,
                color.chroma,
                color.hue / 360.0,
                color.alpha,
            ]
        }
        InterpolationColorSpace::Srgba => {
            let color: Srgba = color.into();
            [color.red, color.green, color.blue, color.alpha]
        }
        InterpolationColorSpace::LinearRgba => color.to_f32_array(),
        InterpolationColorSpace::Hsla | InterpolationColorSpace::HslaLong => {
            let color: Hsla = color.into();
            [
                color.hue / 360.0,
                color.saturation,
                color.lightness,
                color.alpha,
            ]
        }
        InterpolationColorSpace::Hsva | InterpolationColorSpace::HsvaLong => {
            let color: Hsva = color.into();
            [
                color.hue / 360.0,
                color.saturation,
                color.value,
                color.alpha,
            ]
        }
    }
}

struct GradientBatch {
    range: core::ops::Range<u32>,
    items: u32,
}

#[derive(bevy::prelude::Resource)]
pub(crate) struct RetainedGradients {
    instances: RawBufferVec<GpuGradientInstance>,
    batches: HashMap<Entity, GradientBatch>,
    view_bind_group: Option<BindGroup>,
    prepared_work: HashMap<Entity, (u64, u64)>,
}

impl Default for RetainedGradients {
    fn default() -> Self {
        Self {
            instances: RawBufferVec::new(BufferUsages::VERTEX),
            batches: HashMap::default(),
            view_bind_group: None,
            prepared_work: HashMap::default(),
        }
    }
}

impl RetainedGradients {
    pub(crate) fn batch_range(&self, entity: Entity) -> Option<core::ops::Range<u32>> {
        self.batches.get(&entity).map(|batch| batch.range.clone())
    }

    pub(crate) fn counts(&self, entity: Entity) -> Option<(u32, usize)> {
        self.batches
            .get(&entity)
            .map(|batch| (batch.items, batch.range.len()))
    }

    pub(crate) fn prepared_work(&self, camera: Entity) -> (u64, u64) {
        self.prepared_work.get(&camera).copied().unwrap_or_default()
    }

    pub(crate) fn direct_ready(&self, entity: Entity) -> bool {
        self.batches
            .get(&entity)
            .is_some_and(|batch| !batch.range.is_empty())
            && self.instances.buffer().is_some()
            && self.view_bind_group.is_some()
    }
}

#[derive(bevy::prelude::Resource)]
pub(crate) struct GradientPipeline {
    view_layout: BindGroupLayoutDescriptor,
    mask_layout: BindGroupLayoutDescriptor,
    shader: Handle<Shader>,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub(crate) struct GradientPipelineKey {
    target_format: bevy::render::render_resource::TextureFormat,
    color_space: InterpolationColorSpace,
    anti_alias: bool,
    full_rebuild: bool,
}

impl SpecializedRenderPipeline for GradientPipeline {
    type Key = GradientPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let layout = VertexBufferLayout::from_vertex_formats(
            VertexStepMode::Instance,
            vec![
                VertexFormat::Float32x4,
                VertexFormat::Float32x2,
                VertexFormat::Float32x2,
                VertexFormat::Uint32,
                VertexFormat::Float32x4,
                VertexFormat::Float32x4,
                VertexFormat::Float32x2,
                VertexFormat::Float32x2,
                VertexFormat::Float32x4,
                VertexFormat::Float32x3,
                VertexFormat::Float32x4,
                VertexFormat::Float32x4,
            ],
        );
        let color_space = match key.color_space {
            InterpolationColorSpace::Oklaba => "IN_OKLAB",
            InterpolationColorSpace::Oklcha => "IN_OKLCH",
            InterpolationColorSpace::OklchaLong => "IN_OKLCH_LONG",
            InterpolationColorSpace::Srgba => "IN_SRGB",
            InterpolationColorSpace::LinearRgba => "IN_LINEAR_RGB",
            InterpolationColorSpace::Hsla => "IN_HSL",
            InterpolationColorSpace::HslaLong => "IN_HSL_LONG",
            InterpolationColorSpace::Hsva => "IN_HSV",
            InterpolationColorSpace::HsvaLong => "IN_HSV_LONG",
        };
        let mut shader_defs = vec![color_space.into()];
        if key.anti_alias {
            shader_defs.push("ANTI_ALIAS".into());
        }
        if key.full_rebuild {
            shader_defs.push("FULL_REBUILD".into());
        }
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
            layout: if key.full_rebuild {
                vec![self.view_layout.clone()]
            } else {
                vec![self.view_layout.clone(), self.mask_layout.clone()]
            },
            label: Some("retained_ui_gradient_pipeline".into()),
            ..Default::default()
        }
    }
}

fn init_pipeline(mut commands: Commands, assets: Res<AssetServer>, ui_pipeline: Res<UiPipeline>) {
    commands.insert_resource(GradientPipeline {
        view_layout: ui_pipeline.view_layout.clone(),
        mask_layout: crate::mask::layout(),
        shader: load_embedded_asset!(assets.as_ref(), "gradient_render.wgsl"),
    });
}

pub(crate) fn queue(
    runs: Res<RetainedGradientRuns>,
    pipeline: Res<GradientPipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<GradientPipeline>>,
    mut phases: ResMut<ViewSortedRenderPhases<TransparentUi>>,
    render_views: bevy::ecs::system::Query<
        (&UiCameraView, Option<&UiAntiAlias>),
        With<ExtractedView>,
    >,
    camera_views: bevy::ecs::system::Query<&ExtractedView>,
    cache: Res<PipelineCache>,
    draws: Res<DrawFunctions<TransparentUi>>,
    full_rebuilds: Res<RetainedFullRebuilds>,
) {
    let draw_function = draws.read().id::<DrawRetainedGradients>();
    for (index, run) in runs.runs.iter().enumerate() {
        let Ok((default_view, anti_alias)) = render_views.get(run.camera) else {
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
                &cache,
                &pipeline,
                GradientPipelineKey {
                    target_format: view.target_format,
                    color_space: run.color_space,
                    anti_alias: matches!(anti_alias, None | Some(UiAntiAlias::On)),
                    full_rebuild: full_rebuilds.0.contains(&run.camera),
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
    mut runs: ResMut<RetainedGradientRuns>,
    mut gradients: ResMut<RetainedGradients>,
    uniforms: Res<ViewUniforms>,
    pipeline: Res<GradientPipeline>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    gradients.instances.clear();
    gradients.batches.clear();
    gradients.prepared_work.clear();
    core::mem::swap(gradients.instances.values_mut(), &mut runs.instances);
    for run in &runs.runs {
        let quads =
            u64::try_from(run.instances.len()).expect("retained gradient count exceeds u64");
        let work = gradients.prepared_work.entry(run.camera).or_default();
        work.0 += u64::from(run.items);
        work.1 += quads;
        gradients.batches.insert(
            run.render_entity,
            GradientBatch {
                range: u32::try_from(run.instances.start)
                    .expect("retained gradient count exceeds u32")
                    ..u32::try_from(run.instances.end)
                        .expect("retained gradient count exceeds u32"),
                items: run.items,
            },
        );
    }
    if gradients.instances.is_empty() {
        gradients.view_bind_group = None;
        return;
    }
    gradients.instances.write_buffer(&device, &queue);
    gradients.view_bind_group = uniforms.uniforms.binding().map(|binding| {
        device.create_bind_group(
            "retained_ui_gradient_view_bind_group",
            &cache.get_bind_group_layout(&pipeline.view_layout),
            &BindGroupEntries::single(binding),
        )
    });
}

pub(crate) type DrawRetainedGradients = (
    SetItemPipeline,
    SetGradientViewBindGroup<0>,
    DrawGradientInstances,
);

pub(crate) struct SetGradientViewBindGroup<const I: usize>;

impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetGradientViewBindGroup<I> {
    type Param = SRes<RetainedGradients>;
    type ViewQuery = Read<ViewUniformOffset>;
    type ItemQuery = ();

    fn render<'w>(
        _item: &P,
        view: &'w ViewUniformOffset,
        _entity: Option<()>,
        gradients: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(bind_group) = gradients.into_inner().view_bind_group.as_ref() else {
            return RenderCommandResult::Failure(
                "retained gradient view bind group is unavailable",
            );
        };
        pass.set_bind_group(I, bind_group, &[view.offset]);
        RenderCommandResult::Success
    }
}

pub(crate) struct DrawGradientInstances;

impl<P: PhaseItem> RenderCommand<P> for DrawGradientInstances {
    type Param = SRes<RetainedGradients>;
    type ViewQuery = ();
    type ItemQuery = ();

    fn render<'w>(
        item: &P,
        _view: (),
        _entity: Option<()>,
        gradients: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let gradients = gradients.into_inner();
        let Some(batch) = gradients.batches.get(&item.entity()) else {
            return RenderCommandResult::Skip;
        };
        let Some(instances) = gradients.instances.buffer() else {
            return RenderCommandResult::Failure("retained gradient instances are unavailable");
        };
        pass.set_vertex_buffer(0, instances.slice(..));
        pass.draw(0..6, batch.range.clone());
        RenderCommandResult::Success
    }
}

pub(crate) fn register(app: &mut SubApp) {
    app.init_resource::<RetainedGradientRuns>()
        .init_resource::<RetainedGradients>()
        .init_gpu_resource::<SpecializedRenderPipelines<GradientPipeline>>()
        .add_render_command::<TransparentUi, DrawRetainedGradients>()
        .add_systems(
            bevy::render::RenderStartup,
            init_pipeline.after(bevy::ui_render::init_ui_pipeline),
        );
}
