use core::{
    f32::consts::{FRAC_PI_2, TAU},
    hash::Hash,
};

use super::shader_flags::BORDER_ALL;
use crate::retained::{
    batch_retained_ui, remove_owner, vertex_storage_supported, RemovedUiNode, RetainedBatchItem,
    RetainedUiBatch, UiInstanceArena,
};
use crate::*;
use bevy_asset::*;
use bevy_color::{ColorToComponents, Hsla, Hsva, LinearRgba, Oklaba, Oklcha, Srgba};
use bevy_ecs::{
    entity::EntityHashMap,
    system::{
        lifetimeless::{Read, SRes},
        *,
    },
};
use bevy_math::Affine2;
use bevy_math::{
    ops::{cos, sin},
    FloatOrd, Rect, Vec2,
};
use bevy_mesh::VertexBufferLayout;
use bevy_render::{
    impl_atomic_pod,
    render_phase::*,
    render_resource::{
        binding_types::{storage_buffer_read_only_sized, uniform_buffer},
        *,
    },
    renderer::{RenderDevice, RenderQueue},
    view::*,
    Extract, ExtractSchedule, Render, RenderSystems,
};
use bevy_render::{GpuResourceAppExt, RenderStartup};
use bevy_shader::Shader;
use bevy_sprite::BorderRect;
use bevy_ui::{
    BackgroundGradient, BorderGradient, ColorStop, ComputedStackIndex, ComputedUiRenderTargetInfo,
    ConicGradient, Gradient, InterpolationColorSpace, LinearGradient, RadialGradient,
    ResolvedBorderRadius, Val,
};
use bevy_utils::default;
use bytemuck::{Pod, Zeroable};

pub struct GradientPlugin;

impl Plugin for GradientPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "gradient.wgsl");

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_command::<TransparentUi, DrawGradientFns>()
                .init_resource::<ExtractedGradients>()
                .init_gpu_resource::<GradientMeta>()
                .init_gpu_resource::<SpecializedRenderPipelines<GradientPipeline>>()
                .add_systems(RenderStartup, init_gradient_pipeline)
                .add_systems(
                    ExtractSchedule,
                    extract_gradients
                        .in_set(RenderUiSystems::ExtractGradient)
                        .after(extract_uinode_background_colors),
                )
                .add_systems(
                    Render,
                    (
                        queue_gradient.in_set(RenderSystems::Queue),
                        prepare_gradient.in_set(RenderSystems::PrepareBindGroups),
                    ),
                );
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct GradientStyleInstance {
    flags_and_padding: [u32; 4],
    radius: [[f32; 4]; 2],
    border: [f32; 4],
    gradient: [f32; 4],
    start_color: [f32; 4],
    end_color: [f32; 4],
    params: [f32; 4],
}

impl_atomic_pod!(GradientStyleInstance, GradientStyleInstanceBlob);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GradientCameraState {
    retained_view_entity: RetainedViewEntity,
    target_format: TextureFormat,
    anti_alias: bool,
    storage_buffers: bool,
}

#[derive(Resource)]
pub struct GradientMeta {
    geometry_instances: AtomicSparseBufferVec<UiGeometryInstance>,
    style_instances: AtomicSparseBufferVec<GradientStyleInstance>,
    instance_indices: RawBufferVec<u32>,
    view_bind_group: Option<BindGroup>,
    instance_bind_group: Option<BindGroup>,
    instance_buffer_ids: Option<(BufferId, BufferId)>,
    batches: Vec<RetainedUiBatch<CachedRenderPipelineId>>,
    arena: UiInstanceArena,
    scratch: Vec<(UiGeometryInstance, GradientStyleInstance)>,
    camera_states: EntityHashMap<GradientCameraState>,
    use_storage_buffers: bool,
}

impl Default for GradientMeta {
    fn default() -> Self {
        Self {
            geometry_instances: AtomicSparseBufferVec::new(
                BufferUsages::VERTEX | BufferUsages::STORAGE,
                0,
                "gradient geometry instances".into(),
            ),
            style_instances: AtomicSparseBufferVec::new(
                BufferUsages::VERTEX | BufferUsages::STORAGE,
                0,
                "gradient style instances".into(),
            ),
            instance_indices: RawBufferVec::new(BufferUsages::VERTEX),
            view_bind_group: None,
            instance_bind_group: None,
            instance_buffer_ids: None,
            batches: Vec::new(),
            arena: UiInstanceArena::default(),
            scratch: Vec::new(),
            camera_states: EntityHashMap::default(),
            use_storage_buffers: false,
        }
    }
}

#[derive(Resource)]
pub struct GradientPipeline {
    pub view_layout: BindGroupLayoutDescriptor,
    pub instance_layout: BindGroupLayoutDescriptor,
    pub shader: Handle<Shader>,
}

pub fn init_gradient_pipeline(mut commands: Commands, asset_server: Res<AssetServer>) {
    let view_layout = BindGroupLayoutDescriptor::new(
        "ui_gradient_view_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::VERTEX_FRAGMENT,
            uniform_buffer::<ViewUniform>(true),
        ),
    );
    let instance_layout = BindGroupLayoutDescriptor::new(
        "ui_gradient_instance_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX,
            (
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
            ),
        ),
    );

    commands.insert_resource(GradientPipeline {
        view_layout,
        instance_layout,
        shader: load_embedded_asset!(asset_server.as_ref(), "gradient.wgsl"),
    });
}

pub fn compute_gradient_line_length(angle: f32, size: Vec2) -> f32 {
    let center = 0.5 * size;
    let v = Vec2::new(sin(angle), -cos(angle));

    let (pos_corner, neg_corner) = if v.x >= 0.0 && v.y <= 0.0 {
        (size.with_y(0.), size.with_x(0.))
    } else if v.x >= 0.0 && v.y > 0.0 {
        (size, Vec2::ZERO)
    } else if v.x < 0.0 && v.y <= 0.0 {
        (Vec2::ZERO, size)
    } else {
        (size.with_x(0.), size.with_y(0.))
    };

    let t_pos = (pos_corner - center).dot(v);
    let t_neg = (neg_corner - center).dot(v);

    (t_pos - t_neg).abs()
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct UiGradientPipelineKey {
    anti_alias: bool,
    color_space: InterpolationColorSpace,
    pub target_format: TextureFormat,
    storage_buffers: bool,
}

impl SpecializedRenderPipeline for GradientPipeline {
    type Key = UiGradientPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let buffers = if key.storage_buffers {
            vec![VertexBufferLayout::from_vertex_formats(
                VertexStepMode::Instance,
                vec![VertexFormat::Uint32],
            )]
        } else {
            vec![ui_geometry_vertex_layout(), gradient_style_vertex_layout()]
        };
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

        let mut shader_defs = if key.anti_alias {
            vec![color_space.into(), "ANTI_ALIAS".into()]
        } else {
            vec![color_space.into()]
        };
        let mut layout = vec![self.view_layout.clone()];
        if key.storage_buffers {
            shader_defs.push("UI_STORAGE_INSTANCE".into());
            layout.push(self.instance_layout.clone());
        }

        RenderPipelineDescriptor {
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: shader_defs.clone(),
                buffers,
                ..default()
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs,
                targets: vec![Some(ColorTargetState {
                    format: key.target_format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            layout,
            label: Some("ui_gradient_pipeline".into()),
            ..default()
        }
    }
}

fn gradient_style_vertex_layout() -> VertexBufferLayout {
    VertexBufferLayout::from_vertex_formats(
        VertexStepMode::Instance,
        vec![
            VertexFormat::Uint32x4,
            VertexFormat::Float32x4,
            VertexFormat::Float32x4,
            VertexFormat::Float32x4,
            VertexFormat::Float32x4,
            VertexFormat::Float32x4,
            VertexFormat::Float32x4,
            VertexFormat::Float32x4,
        ],
    )
    .offset_locations_by(8)
}

pub enum ResolvedGradient {
    Linear { angle: f32 },
    Conic { center: Vec2, start: f32 },
    Radial { center: Vec2, size: Vec2 },
}

pub struct ExtractedGradient {
    pub stack_index: u32,
    pub transform: Affine2,
    pub rect: Rect,
    pub clip: Option<Rect>,
    pub stops: Vec<(LinearRgba, f32, f32)>,
    pub node_type: NodeType,
    /// Border radius of the UI node.
    /// Ordering: top left, top right, bottom right, bottom left.
    pub border_radius: ResolvedBorderRadius,
    /// Border thickness of the UI node.
    /// Ordering: left, top, right, bottom.
    pub border: BorderRect,
    pub resolved_gradient: ResolvedGradient,
    pub color_space: InterpolationColorSpace,
}

/// A render-world resource that stores all gradients in the scene.
#[derive(Resource, Default)]
pub struct ExtractedGradients {
    /// The list of gradients grouped by their main-world entity, along with each group's target camera entity.
    ///
    /// This is a two-level data structure so that we can quickly remove all
    /// gradients associated with a main-world entity when it changes.
    pub items: MainEntityHashMap<(Entity, EntityIndexMap<ExtractedGradient>)>,
    pub changed: MainEntityHashSet,
    removed: Vec<RemovedUiNode>,
}

// Interpolate implicit stops (where position is `f32::NAN`)
// If the first and last stops are implicit set them to the `min` and `max` values
// so that we always have explicit start and end points to interpolate between.
fn interpolate_color_stops(stops: &mut [(LinearRgba, f32, f32)], min: f32, max: f32) {
    if stops[0].1.is_nan() {
        stops[0].1 = min;
    }
    if stops.last().unwrap().1.is_nan() {
        stops.last_mut().unwrap().1 = max;
    }

    let mut i = 1;

    while i < stops.len() - 1 {
        let point = stops[i].1;
        if point.is_nan() {
            let start = i;
            let mut end = i + 1;
            while end < stops.len() - 1 && stops[end].1.is_nan() {
                end += 1;
            }
            let start_point = stops[start - 1].1;
            let end_point = stops[end].1;
            let steps = end - start;
            let step = (end_point - start_point) / (steps + 1) as f32;
            for j in 0..steps {
                stops[i + j].1 = start_point + step * (j + 1) as f32;
            }
            i = end;
        }
        i += 1;
    }
}

fn compute_color_stops(
    stops: &[ColorStop],
    scale_factor: f32,
    length: f32,
    target_size: Vec2,
    scratch: &mut Vec<(LinearRgba, f32, f32)>,
) -> Vec<(LinearRgba, f32, f32)> {
    let mut extracted_color_stops = vec![];

    // resolve the physical distances of explicit stops and sort them
    scratch.extend(stops.iter().filter_map(|stop| {
        stop.point
            .resolve(scale_factor, length, target_size)
            .ok()
            .map(|physical_point| (stop.color.to_linear(), physical_point, stop.hint))
    }));
    scratch.sort_by_key(|(_, point, _)| FloatOrd(*point));

    let min = scratch
        .first()
        .map(|(_, min, _)| *min)
        .unwrap_or(0.)
        .min(0.);

    // get the position of the last explicit stop and use the full length of the gradient if no explicit stops
    let max = scratch
        .last()
        .map(|(_, max, _)| *max)
        .unwrap_or(length)
        .max(length);

    let mut sorted_stops_drain = scratch.drain(..);

    // Fill the extracted color stops buffer
    extracted_color_stops.extend(stops.iter().map(|stop| {
        if stop.point == Val::Auto {
            (stop.color.to_linear(), f32::NAN, stop.hint)
        } else {
            sorted_stops_drain.next().unwrap()
        }
    }));

    interpolate_color_stops(&mut extracted_color_stops, min, max);

    extracted_color_stops
}

pub fn extract_gradients(
    mut commands: Commands,
    mut extracted_gradients: ResMut<ExtractedGradients>,
    gradients_query: Extract<
        Query<
            (
                Entity,
                &ComputedNode,
                &ComputedStackIndex,
                &ComputedUiTargetCamera,
                &ComputedUiRenderTargetInfo,
                &UiGlobalTransform,
                &InheritedVisibility,
                Option<&CalculatedClip>,
                AnyOf<(&BackgroundGradient, &BorderGradient)>,
            ),
            Or<(
                Changed<ComputedNode>,
                Changed<ComputedStackIndex>,
                Changed<ComputedUiTargetCamera>,
                Changed<ComputedUiRenderTargetInfo>,
                Changed<UiGlobalTransform>,
                Changed<InheritedVisibility>,
                Changed<CalculatedClip>,
                Changed<BackgroundGradient>,
                Changed<BorderGradient>,
            )>,
        >,
    >,
    (
        mut removed_computed_node_query,
        mut removed_computed_stack_index_query,
        mut removed_computed_ui_target_camera_query,
        mut removed_computed_ui_render_target_info_query,
        mut removed_ui_global_transform_query,
        mut removed_inherited_visibility_query,
        mut removed_calculated_clip_query,
        mut removed_background_gradient_query,
        mut removed_border_gradient_query,
    ): (
        Extract<RemovedComponents<ComputedNode>>,
        Extract<RemovedComponents<ComputedStackIndex>>,
        Extract<RemovedComponents<ComputedUiTargetCamera>>,
        Extract<RemovedComponents<ComputedUiRenderTargetInfo>>,
        Extract<RemovedComponents<UiGlobalTransform>>,
        Extract<RemovedComponents<InheritedVisibility>>,
        Extract<RemovedComponents<CalculatedClip>>,
        Extract<RemovedComponents<BackgroundGradient>>,
        Extract<RemovedComponents<BorderGradient>>,
    ),
    camera_map: Extract<UiCameraMap>,
    mut nodes_processed_this_frame: Local<MainEntityHashSet>,
) {
    nodes_processed_this_frame.clear();
    extracted_gradients.changed.clear();
    extracted_gradients.removed.clear();
    let mut camera_mapper = camera_map.get_mapper();
    let mut sorted_stops = vec![];

    for (
        entity,
        uinode,
        stack_index,
        camera,
        target,
        transform,
        inherited_visibility,
        clip,
        (gradient, gradient_border),
    ) in &gradients_query
    {
        let main_entity = MainEntity::from(entity);
        extracted_gradients.changed.insert(main_entity);

        // If there were any previous gradients for this entity, despawn them.
        let extracted = &mut *extracted_gradients;
        if let Some((_, old_gradients)) =
            remove_owner(&mut extracted.items, main_entity, &mut extracted.removed)
        {
            for render_entity in old_gradients.keys() {
                commands.entity(*render_entity).despawn();
            }
        }

        // Skip invisible images
        if !inherited_visibility.get() {
            continue;
        }

        let Some(extracted_camera_entity) = camera_mapper.map(camera) else {
            continue;
        };
        for (gradients, node_type) in [
            (gradient.map(|g| &g.0), NodeType::Rect),
            (gradient_border.map(|g| &g.0), NodeType::Border(BORDER_ALL)),
        ]
        .iter()
        .filter_map(|(g, n)| g.map(|g| (g, *n)))
        {
            for gradient in gradients.iter() {
                if gradient.is_empty() {
                    continue;
                }

                nodes_processed_this_frame.insert(main_entity);

                if let Some(color) = gradient.get_single() {
                    // With a single color stop there's no gradient, fill the node with the color
                    let length = compute_gradient_line_length(0.0, uinode.size);
                    let extracted_stops = compute_color_stops(
                        &[
                            ColorStop::new(color, Val::Percent(0.0)),
                            ColorStop::new(color, Val::Percent(100.0)),
                        ],
                        target.scale_factor(),
                        length,
                        target.physical_size().as_vec2(),
                        &mut sorted_stops,
                    );
                    extracted_gradients
                        .items
                        .entry(main_entity)
                        .or_insert_with(|| (extracted_camera_entity, Default::default()))
                        .1
                        .insert(
                            commands.spawn_empty().id(),
                            ExtractedGradient {
                                stack_index: stack_index.0,
                                transform: transform.into(),
                                stops: extracted_stops,
                                rect: Rect {
                                    min: Vec2::ZERO,
                                    max: uinode.size,
                                },
                                clip: clip.map(|clip| clip.clip),
                                node_type,
                                border_radius: uinode.border_radius,
                                border: uinode.border,
                                resolved_gradient: ResolvedGradient::Linear { angle: 0.0 },
                                color_space: gradient.get_color_space(),
                            },
                        );
                    continue;
                }
                match gradient {
                    Gradient::Linear(LinearGradient {
                        color_space,
                        angle,
                        stops,
                    }) => {
                        let length = compute_gradient_line_length(*angle, uinode.size);

                        let extracted_stops = compute_color_stops(
                            stops,
                            target.scale_factor(),
                            length,
                            target.physical_size().as_vec2(),
                            &mut sorted_stops,
                        );

                        extracted_gradients
                            .items
                            .entry(main_entity)
                            .or_insert_with(|| (extracted_camera_entity, Default::default()))
                            .1
                            .insert(
                                commands.spawn_empty().id(),
                                ExtractedGradient {
                                    stack_index: stack_index.0,
                                    transform: transform.into(),
                                    stops: extracted_stops,
                                    rect: Rect {
                                        min: Vec2::ZERO,
                                        max: uinode.size,
                                    },
                                    clip: clip.map(|clip| clip.clip),
                                    node_type,
                                    border_radius: uinode.border_radius,
                                    border: uinode.border,
                                    resolved_gradient: ResolvedGradient::Linear { angle: *angle },
                                    color_space: *color_space,
                                },
                            );
                    }
                    Gradient::Radial(RadialGradient {
                        color_space,
                        position: center,
                        shape,
                        stops,
                    }) => {
                        let c = center.resolve(
                            target.scale_factor(),
                            uinode.size,
                            target.physical_size().as_vec2(),
                        );

                        let size = shape.resolve(
                            c,
                            target.scale_factor(),
                            uinode.size,
                            target.physical_size().as_vec2(),
                        );

                        let length = size.x;

                        let computed_stops = compute_color_stops(
                            stops,
                            target.scale_factor(),
                            length,
                            target.physical_size().as_vec2(),
                            &mut sorted_stops,
                        );

                        extracted_gradients
                            .items
                            .entry(main_entity)
                            .or_insert_with(|| (extracted_camera_entity, Default::default()))
                            .1
                            .insert(
                                commands.spawn_empty().id(),
                                ExtractedGradient {
                                    stack_index: stack_index.0,
                                    transform: transform.into(),
                                    stops: computed_stops,
                                    rect: Rect {
                                        min: Vec2::ZERO,
                                        max: uinode.size,
                                    },
                                    clip: clip.map(|clip| clip.clip),
                                    node_type,
                                    border_radius: uinode.border_radius,
                                    border: uinode.border,
                                    resolved_gradient: ResolvedGradient::Radial { center: c, size },
                                    color_space: *color_space,
                                },
                            );
                    }
                    Gradient::Conic(ConicGradient {
                        color_space,
                        start,
                        position: center,
                        stops,
                    }) => {
                        let g_start = center.resolve(
                            target.scale_factor(),
                            uinode.size,
                            target.physical_size().as_vec2(),
                        );

                        // sort the explicit stops
                        sorted_stops.extend(stops.iter().filter_map(|stop| {
                            stop.angle.map(|angle| {
                                (stop.color.to_linear(), angle.clamp(0., TAU), stop.hint)
                            })
                        }));
                        sorted_stops.sort_by_key(|(_, angle, _)| FloatOrd(*angle));
                        let mut sorted_stops_drain = sorted_stops.drain(..);

                        // fill the extracted stops buffer
                        let mut extracted_color_stops: Vec<_> = stops
                            .iter()
                            .map(|stop| {
                                if stop.angle.is_none() {
                                    (stop.color.to_linear(), f32::NAN, stop.hint)
                                } else {
                                    sorted_stops_drain.next().unwrap()
                                }
                            })
                            .collect();

                        interpolate_color_stops(&mut extracted_color_stops, 0., TAU);

                        extracted_gradients
                            .items
                            .entry(main_entity)
                            .or_insert_with(|| (extracted_camera_entity, Default::default()))
                            .1
                            .insert(
                                commands.spawn_empty().id(),
                                ExtractedGradient {
                                    stack_index: stack_index.0,
                                    transform: transform.into(),
                                    stops: extracted_color_stops,
                                    rect: Rect {
                                        min: Vec2::ZERO,
                                        max: uinode.size,
                                    },
                                    clip: clip.map(|clip| clip.clip),
                                    node_type,
                                    border_radius: uinode.border_radius,
                                    border: uinode.border,
                                    resolved_gradient: ResolvedGradient::Conic {
                                        start: *start,
                                        center: g_start,
                                    },
                                    color_space: *color_space,
                                },
                            );
                    }
                }
            }
        }
    }

    // Only remove the render-world data if we didn't handle the node above.
    // It's possible that a relevant component was removed and added in the same
    // frame.
    for main_entity in removed_computed_node_query
        .read()
        .chain(removed_computed_stack_index_query.read())
        .chain(removed_computed_ui_target_camera_query.read())
        .chain(removed_computed_ui_render_target_info_query.read())
        .chain(removed_ui_global_transform_query.read())
        .chain(removed_inherited_visibility_query.read())
        .chain(removed_calculated_clip_query.read())
        .chain(removed_background_gradient_query.read())
        .chain(removed_border_gradient_query.read())
    {
        let main_entity = MainEntity::from(main_entity);
        if nodes_processed_this_frame.contains(&main_entity) {
            continue;
        }
        extracted_gradients.changed.insert(main_entity);
        let extracted = &mut *extracted_gradients;
        let Some((_, extracted_nodes)) =
            remove_owner(&mut extracted.items, main_entity, &mut extracted.removed)
        else {
            continue;
        };
        for render_entity in extracted_nodes.keys() {
            commands.entity(*render_entity).despawn();
        }
    }
}

fn convert_color_to_space(color: LinearRgba, space: InterpolationColorSpace) -> [f32; 4] {
    match space {
        InterpolationColorSpace::Oklaba => {
            let oklaba: Oklaba = color.into();
            [oklaba.lightness, oklaba.a, oklaba.b, oklaba.alpha]
        }
        InterpolationColorSpace::Oklcha | InterpolationColorSpace::OklchaLong => {
            let oklcha: Oklcha = color.into();
            [
                oklcha.lightness,
                oklcha.chroma,
                // The shader expects normalized hues
                oklcha.hue / 360.,
                oklcha.alpha,
            ]
        }
        InterpolationColorSpace::Srgba => {
            let srgba: Srgba = color.into();
            [srgba.red, srgba.green, srgba.blue, srgba.alpha]
        }
        InterpolationColorSpace::LinearRgba => color.to_f32_array(),
        InterpolationColorSpace::Hsla | InterpolationColorSpace::HslaLong => {
            let hsla: Hsla = color.into();
            // The shader expects normalized hues
            [hsla.hue / 360., hsla.saturation, hsla.lightness, hsla.alpha]
        }
        InterpolationColorSpace::Hsva | InterpolationColorSpace::HsvaLong => {
            let hsva: Hsva = color.into();
            // The shader expects normalized hues
            [hsva.hue / 360., hsva.saturation, hsva.value, hsva.alpha]
        }
    }
}

pub fn queue_gradient(
    extracted_gradients: Res<ExtractedGradients>,
    gradient_pipeline: Res<GradientPipeline>,
    mut gradient_meta: ResMut<GradientMeta>,
    mut pipelines: ResMut<SpecializedRenderPipelines<GradientPipeline>>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<TransparentUi>>,
    render_views: Query<(Entity, &UiCameraView, Option<&UiAntiAlias>), With<ExtractedView>>,
    camera_views: Query<&ExtractedView>,
    pipeline_cache: Res<PipelineCache>,
    draw_functions: Res<DrawFunctions<TransparentUi>>,
    render_device: Res<RenderDevice>,
) {
    let draw_function = draw_functions.read().id::<DrawGradientFns>();
    let storage_buffers = vertex_storage_supported(&render_device, 2);
    let mut active_cameras = HashSet::new();
    let mut invalidated_cameras = HashSet::new();

    for (camera_entity, ui_camera_view, ui_anti_alias) in &render_views {
        let Ok(view) = camera_views.get(ui_camera_view.0) else {
            continue;
        };
        let state = GradientCameraState {
            retained_view_entity: view.retained_view_entity,
            target_format: view.target_format,
            anti_alias: matches!(ui_anti_alias, None | Some(UiAntiAlias::On)),
            storage_buffers,
        };
        active_cameras.insert(camera_entity);
        if gradient_meta.camera_states.insert(camera_entity, state) != Some(state) {
            invalidated_cameras.insert(camera_entity);
        }
    }

    for removed in &extracted_gradients.removed {
        let Some(camera_state) = gradient_meta.camera_states.get(&removed.camera_entity) else {
            continue;
        };
        if let Some(phase) = transparent_render_phases.get_mut(&camera_state.retained_view_entity) {
            phase.remove(removed.render_entity, removed.main_entity);
        }
    }

    let mut dirty = extracted_gradients.changed.clone();
    if !invalidated_cameras.is_empty() {
        dirty.extend(extracted_gradients.items.iter().filter_map(
            |(main_entity, (camera_entity, _))| {
                invalidated_cameras
                    .contains(camera_entity)
                    .then_some(*main_entity)
            },
        ));
    }

    for main_entity in dirty {
        let Some((camera_entity, gradients)) = extracted_gradients.items.get(&main_entity) else {
            continue;
        };
        let Some(camera_state) = gradient_meta.camera_states.get(camera_entity) else {
            continue;
        };
        let Some(phase) = transparent_render_phases.get_mut(&camera_state.retained_view_entity)
        else {
            continue;
        };
        for (render_entity, gradient) in gradients {
            let pipeline = pipelines.specialize(
                &pipeline_cache,
                &gradient_pipeline,
                UiGradientPipelineKey {
                    anti_alias: camera_state.anti_alias,
                    color_space: gradient.color_space,
                    target_format: camera_state.target_format,
                    storage_buffers: camera_state.storage_buffers,
                },
            );
            phase.add_retained(TransparentUi {
                draw_function,
                pipeline,
                entity: (*render_entity, main_entity),
                sort_key: FloatOrd(
                    gradient.stack_index as f32
                        + match gradient.node_type {
                            NodeType::Border(_) => stack_z_offsets::BORDER_GRADIENT,
                            _ => stack_z_offsets::GRADIENT,
                        },
                ),
                batch_range: 0..0,
                extra_index: PhaseItemExtraIndex::None,
                indexed: false,
                batch_index: None,
            });
        }
    }
    gradient_meta
        .camera_states
        .retain(|camera, _| active_cameras.contains(camera));
}

fn generate_gradient_instances(
    gradient: &ExtractedGradient,
    scratch: &mut Vec<(UiGeometryInstance, GradientStyleInstance)>,
) {
    scratch.clear();
    let size = gradient.rect.size();
    let (position_diff, culled) =
        clipping_offsets(gradient.transform, Vec2::ZERO, size, gradient.clip);
    if culled {
        return;
    }
    let (position_diff_01, position_diff_23) = pack_corners(position_diff);
    let (uv_01, uv_23) = pack_corners([Vec2::ZERO, Vec2::X, Vec2::ONE, Vec2::Y]);
    let geometry = UiGeometryInstance {
        transform_x: gradient.transform.x_axis.into(),
        transform_y: gradient.transform.y_axis.into(),
        translation: gradient.transform.translation.into(),
        size: size.into(),
        position_diff_01,
        position_diff_23,
        uv_01,
        uv_23,
    };
    let corner_points = QUAD_VERTEX_POSITIONS.map(|position| position * size);
    let mut flags = match gradient.node_type {
        NodeType::Border(borders) => borders,
        _ => 0,
    };
    let (g_start, g_dir, gradient_flags) = match gradient.resolved_gradient {
        ResolvedGradient::Linear { angle } => {
            let corner_index = (angle - FRAC_PI_2).rem_euclid(TAU) / FRAC_PI_2;
            (
                corner_points[corner_index as usize],
                Vec2::new(sin(angle), -cos(angle)),
                0,
            )
        }
        ResolvedGradient::Conic { center, start } => {
            (center, Vec2::new(start, 0.0), shader_flags::CONIC)
        }
        ResolvedGradient::Radial { center, size } => (
            center,
            Vec2::splat(if size.y != 0.0 { size.x / size.y } else { 1.0 }),
            shader_flags::RADIAL,
        ),
    };
    flags |= gradient_flags;

    let mut segment_count = 0;
    for stop_index in 0..gradient.stops.len() - 1 {
        let mut start_stop = gradient.stops[stop_index];
        let end_stop = gradient.stops[stop_index + 1];
        if start_stop.1 == end_stop.1 {
            if stop_index == gradient.stops.len() - 2 {
                if segment_count > 0 {
                    start_stop.0 = LinearRgba::NONE;
                }
            } else {
                continue;
            }
        }
        let mut stop_flags = flags;
        if start_stop.1 > 0.0 && (stop_index == 0 || segment_count == 0) {
            stop_flags |= shader_flags::FILL_START;
        }
        if stop_index == gradient.stops.len() - 2 {
            stop_flags |= shader_flags::FILL_END;
        }
        scratch.push((
            geometry,
            GradientStyleInstance {
                flags_and_padding: [stop_flags, 0, 0, 0],
                radius: gradient.border_radius.into(),
                border: [
                    gradient.border.min_inset.x,
                    gradient.border.min_inset.y,
                    gradient.border.max_inset.x,
                    gradient.border.max_inset.y,
                ],
                gradient: [g_start.x, g_start.y, g_dir.x, g_dir.y],
                start_color: convert_color_to_space(start_stop.0, gradient.color_space),
                end_color: convert_color_to_space(end_stop.0, gradient.color_space),
                params: [start_stop.1, end_stop.1, start_stop.2, 0.0],
            },
        ));
        segment_count += 1;
    }
}

fn rebuild_gradient_owner(
    main_entity: MainEntity,
    meta: &mut GradientMeta,
    extracted_gradients: &ExtractedGradients,
) {
    meta.arena.free_owner(main_entity);
    let Some((_, gradients)) = extracted_gradients.items.get(&main_entity) else {
        return;
    };
    let mut owned = Vec::with_capacity(gradients.len());
    for (render_entity, gradient) in gradients {
        generate_gradient_instances(gradient, &mut meta.scratch);
        if meta.scratch.is_empty() {
            meta.arena.insert_empty(*render_entity);
        } else {
            let count = u32::try_from(meta.scratch.len()).expect("too many gradient segments");
            let (start, capacity) = meta.arena.alloc(count);
            meta.geometry_instances.grow(start + capacity);
            meta.style_instances.grow(start + capacity);
            for (offset, (geometry, style)) in meta.scratch.iter().copied().enumerate() {
                let index = start + offset as u32;
                meta.geometry_instances.set(index, geometry);
                meta.style_instances.set(index, style);
            }
            meta.arena.insert(*render_entity, start, count, capacity);
        }
        owned.push(*render_entity);
    }
    if !owned.is_empty() {
        meta.arena.owners.insert(main_entity, owned);
    }
}

pub fn prepare_gradient(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    mut meta: ResMut<GradientMeta>,
    extracted_gradients: Res<ExtractedGradients>,
    view_uniforms: Res<ViewUniforms>,
    gradient_pipeline: Res<GradientPipeline>,
    mut phases: ResMut<ViewSortedRenderPhases<TransparentUi>>,
    (
        mut sparse_buffer_update_jobs,
        mut sparse_buffer_update_bind_groups,
        sparse_buffer_update_pipelines,
    ): (
        ResMut<SparseBufferUpdateJobs>,
        ResMut<SparseBufferUpdateBindGroups>,
        Res<SparseBufferUpdatePipelines>,
    ),
) {
    meta.use_storage_buffers = vertex_storage_supported(&render_device, 2);
    if meta.arena.needs_compaction() || !meta.arena.initialized {
        meta.arena.reset();
        meta.geometry_instances.clear();
        meta.style_instances.clear();
        for main_entity in extracted_gradients.items.keys().copied() {
            rebuild_gradient_owner(main_entity, &mut meta, &extracted_gradients);
        }
    } else {
        for main_entity in extracted_gradients.changed.iter().copied() {
            rebuild_gradient_owner(main_entity, &mut meta, &extracted_gradients);
        }
    }
    meta.geometry_instances
        .write_buffers(&render_device, &render_queue);
    meta.style_instances
        .write_buffers(&render_device, &render_queue);
    meta.geometry_instances.prepare_to_populate_buffers(
        &render_device,
        &pipeline_cache,
        &mut sparse_buffer_update_jobs,
        &mut sparse_buffer_update_bind_groups,
        &sparse_buffer_update_pipelines,
    );
    meta.style_instances.prepare_to_populate_buffers(
        &render_device,
        &pipeline_cache,
        &mut sparse_buffer_update_jobs,
        &mut sparse_buffer_update_bind_groups,
        &sparse_buffer_update_pipelines,
    );

    let Some(view_binding) = view_uniforms.uniforms.binding() else {
        meta.batches.clear();
        return;
    };
    meta.view_bind_group = Some(render_device.create_bind_group(
        "gradient_view_bind_group",
        &pipeline_cache.get_bind_group_layout(&gradient_pipeline.view_layout),
        &BindGroupEntries::single(view_binding),
    ));
    if meta.use_storage_buffers {
        if let (Some(geometry_buffer), Some(style_buffer)) = (
            meta.geometry_instances.buffer(),
            meta.style_instances.buffer(),
        ) {
            let ids = (geometry_buffer.id(), style_buffer.id());
            if meta.instance_buffer_ids != Some(ids) {
                meta.instance_bind_group = Some(render_device.create_bind_group(
                    "gradient_instance_bind_group",
                    &pipeline_cache.get_bind_group_layout(&gradient_pipeline.instance_layout),
                    &BindGroupEntries::sequential((
                        geometry_buffer.as_entire_binding(),
                        style_buffer.as_entire_binding(),
                    )),
                ));
                meta.instance_buffer_ids = Some(ids);
            }
        } else {
            meta.instance_bind_group = None;
            meta.instance_buffer_ids = None;
        }
    }
    let GradientMeta {
        arena,
        instance_indices,
        use_storage_buffers,
        ..
    } = &mut *meta;
    meta.batches = batch_retained_ui(
        &mut phases,
        instance_indices,
        *use_storage_buffers,
        |item| {
            if extracted_gradients
                .items
                .get(&item.main_entity())
                .and_then(|(_, gradients)| gradients.get(&item.entity()))
                .is_none()
            {
                return RetainedBatchItem::NotOwned;
            }
            let Some(slot) = arena.slots.get(&item.entity()) else {
                return RetainedBatchItem::Culled;
            };
            if slot.instances.count == 0 {
                RetainedBatchItem::Culled
            } else {
                RetainedBatchItem::Drawable {
                    instances: slot.instances,
                    key: item.pipeline,
                }
            }
        },
        PartialEq::eq,
        |_, _| {},
    );
    meta.instance_indices
        .write_buffer(&render_device, &render_queue);
}

pub type DrawGradientFns = (
    SetItemPipeline,
    SetGradientViewBindGroup<0>,
    SetGradientInstanceBindGroup<1>,
    DrawGradient,
);

pub struct SetGradientInstanceBindGroup<const I: usize>;
impl<const I: usize> RenderCommand<TransparentUi> for SetGradientInstanceBindGroup<I> {
    type Param = SRes<GradientMeta>;
    type ViewQuery = ();
    type ItemQuery = ();

    fn render<'w>(
        _item: &TransparentUi,
        _view: (),
        _entity: Option<()>,
        meta: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let meta = meta.into_inner();
        if meta.use_storage_buffers {
            let Some(bind_group) = meta.instance_bind_group.as_ref() else {
                return RenderCommandResult::Failure("missing gradient instance bind group");
            };
            pass.set_bind_group(I, bind_group, &[]);
        }
        RenderCommandResult::Success
    }
}

pub struct SetGradientViewBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetGradientViewBindGroup<I> {
    type Param = SRes<GradientMeta>;
    type ViewQuery = Read<ViewUniformOffset>;
    type ItemQuery = ();

    fn render<'w>(
        _item: &P,
        view_uniform: &'w ViewUniformOffset,
        _entity: Option<()>,
        ui_meta: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(view_bind_group) = ui_meta.into_inner().view_bind_group.as_ref() else {
            return RenderCommandResult::Failure("view_bind_group not available");
        };
        pass.set_bind_group(I, view_bind_group, &[view_uniform.offset]);
        RenderCommandResult::Success
    }
}

pub struct DrawGradient;
impl RenderCommand<TransparentUi> for DrawGradient {
    type Param = SRes<GradientMeta>;
    type ViewQuery = ();
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        item: &TransparentUi,
        _view: (),
        _entity: Option<()>,
        ui_meta: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(batch_index) = item.batch_index else {
            return RenderCommandResult::Skip;
        };
        let meta = ui_meta.into_inner();
        let Some(batch) = meta.batches.get(batch_index as usize) else {
            return RenderCommandResult::Failure("gradient batch index out of range");
        };
        if meta.use_storage_buffers {
            let Some(indices) = meta.instance_indices.buffer() else {
                return RenderCommandResult::Failure("missing gradient instance indices");
            };
            pass.set_vertex_buffer(0, indices.slice(..));
        } else {
            let Some(geometry) = meta.geometry_instances.buffer() else {
                return RenderCommandResult::Failure("missing gradient geometry instances");
            };
            let Some(style) = meta.style_instances.buffer() else {
                return RenderCommandResult::Failure("missing gradient style instances");
            };
            pass.set_vertex_buffer(0, geometry.slice(..));
            pass.set_vertex_buffer(1, style.slice(..));
        }
        pass.draw(0..6, batch.range.clone());
        RenderCommandResult::Success
    }
}

#[cfg(test)]
mod retained_tests {
    use core::mem::{offset_of, size_of};

    use super::*;
    use crate::retained::{preprocess_wgsl_for_test, validate_wgsl_for_test};

    fn gradient(stops: Vec<(LinearRgba, f32, f32)>) -> ExtractedGradient {
        ExtractedGradient {
            stack_index: 0,
            transform: Affine2::IDENTITY,
            rect: Rect::from_center_size(Vec2::ZERO, Vec2::new(100.0, 50.0)),
            clip: None,
            stops,
            node_type: NodeType::Rect,
            border_radius: ResolvedBorderRadius::ZERO,
            border: BorderRect::ZERO,
            resolved_gradient: ResolvedGradient::Linear { angle: 0.0 },
            color_space: InterpolationColorSpace::LinearRgba,
        }
    }

    #[test]
    fn gradient_instance_layout_uses_exactly_the_portable_attribute_budget() {
        let layout = gradient_style_vertex_layout();
        assert_eq!(size_of::<GradientStyleInstance>(), 128);
        assert_eq!(layout.array_stride, 128);
        assert_eq!(layout.attributes.len(), 8);
        assert_eq!(
            layout.attributes[0].offset as usize,
            offset_of!(GradientStyleInstance, flags_and_padding)
        );
        assert_eq!(
            layout.attributes[7].offset as usize,
            offset_of!(GradientStyleInstance, params)
        );
        assert_eq!(
            ui_geometry_vertex_layout().attributes.len() + layout.attributes.len(),
            16
        );
    }

    #[test]
    fn gradient_segments_are_retained_one_instance_per_nonempty_stop_interval() {
        let red = LinearRgba::RED;
        let blue = LinearRgba::BLUE;
        let green = LinearRgba::GREEN;
        let mut scratch = Vec::new();
        generate_gradient_instances(
            &gradient(vec![(red, 0.0, 0.5), (blue, 0.5, 0.25), (green, 1.0, 0.75)]),
            &mut scratch,
        );
        assert_eq!(scratch.len(), 2);
        assert_eq!(scratch[0].1.start_color, red.to_f32_array());
        assert_eq!(scratch[0].1.end_color, blue.to_f32_array());
        assert_eq!(scratch[0].1.params, [0.0, 0.5, 0.5, 0.0]);
        assert_eq!(scratch[1].1.params, [0.5, 1.0, 0.25, 0.0]);
        assert_eq!(
            scratch[1].1.flags_and_padding[0] & shader_flags::FILL_END,
            shader_flags::FILL_END
        );
    }

    #[test]
    fn equal_intermediate_gradient_stops_do_not_allocate_empty_instances() {
        let mut scratch = Vec::new();
        generate_gradient_instances(
            &gradient(vec![
                (LinearRgba::RED, 0.0, 0.5),
                (LinearRgba::BLUE, 0.5, 0.5),
                (LinearRgba::GREEN, 0.5, 0.5),
                (LinearRgba::WHITE, 1.0, 0.5),
            ]),
            &mut scratch,
        );
        assert_eq!(scratch.len(), 2);
    }

    #[test]
    fn gradient_shader_validates_for_every_instance_and_antialias_path() {
        let source = include_str!("gradient.wgsl");
        let prefix = "
            struct View { clip_from_world: mat4x4<f32>, }
            const PI: f32 = 3.141592653589793;
            fn draw_uinode_background(
                color: vec4<f32>, point: vec2<f32>, size: vec2<f32>,
                radius_x: vec4<f32>, radius_y: vec4<f32>,
                border: vec4<f32>, flags: u32,
            ) -> vec4<f32> { return color; }
            fn draw_uinode_border(
                color: vec4<f32>, point: vec2<f32>, size: vec2<f32>,
                radius_x: vec4<f32>, radius_y: vec4<f32>,
                border: vec4<f32>, flags: u32,
            ) -> vec4<f32> { return color; }
        ";
        for storage in [false, true] {
            for anti_alias in [false, true] {
                let mut definitions = vec!["IN_LINEAR_RGB"];
                if storage {
                    definitions.push("UI_STORAGE_INSTANCE");
                }
                if anti_alias {
                    definitions.push("ANTI_ALIAS");
                }
                let processed = preprocess_wgsl_for_test(source, &definitions, prefix);
                validate_wgsl_for_test(
                    &format!("gradient shader (storage={storage}, aa={anti_alias})"),
                    &processed,
                );
            }
        }
    }
}
