//! Change-driven retained extraction for solid borders and outlines.

use crate::{
    boundary::retained_clip,
    scene::{
        coverage_rect, PaintFamily, PaintId, ResourceFingerprint, RetainedDraw, RetainedDrawItem,
        RetainedNodeItem, RetainedUiScene, RetainedUiSurfaces,
    },
};
use bevy::{
    app::Inherited,
    asset::AssetId,
    camera::visibility::InheritedVisibility,
    color::{Alpha, LinearRgba},
    ecs::{
        entity::Entity,
        lifecycle::RemovedComponents,
        query::{Changed, Or, With},
        system::{Commands, Query, Res, SystemParam},
    },
    image::Image,
    math::{Rect, Vec2},
    render::Extract,
    sprite::BorderRect,
    ui::{
        BorderColor, CalculatedClip, ComputedNode, ComputedStackIndex, ComputedUiPaintTarget,
        ComputedUiRenderTargetInfo, ComputedUiTargetCamera, Display, Node, Outline,
        ResolvedBorderRadius, UiGlobalTransform,
    },
    ui_render::{shader_flags, stack_z_offsets, NodeType, UiCameraMap},
};
pub(crate) const EDGE_FLAGS: [u32; 4] = [
    shader_flags::BORDER_LEFT,
    shader_flags::BORDER_TOP,
    shader_flags::BORDER_RIGHT,
    shader_flags::BORDER_BOTTOM,
];

type BorderQueryItem<'a> = (
    Entity,
    &'a Node,
    &'a ComputedNode,
    &'a ComputedStackIndex,
    &'a UiGlobalTransform,
    &'a InheritedVisibility,
    Option<&'a CalculatedClip>,
    Option<&'a Inherited<ComputedUiPaintTarget>>,
    &'a ComputedUiTargetCamera,
    Option<&'a BorderColor>,
    Option<&'a Outline>,
);

#[derive(SystemParam)]
pub(crate) struct RemovedBorderInputs<'w, 's> {
    border: RemovedComponents<'w, 's, BorderColor>,
    outline: RemovedComponents<'w, 's, Outline>,
    clip: RemovedComponents<'w, 's, CalculatedClip>,
    computed_node: RemovedComponents<'w, 's, ComputedNode>,
    node: RemovedComponents<'w, 's, Node>,
    stack: RemovedComponents<'w, 's, ComputedStackIndex>,
    transform: RemovedComponents<'w, 's, UiGlobalTransform>,
    visibility: RemovedComponents<'w, 's, InheritedVisibility>,
    camera: RemovedComponents<'w, 's, ComputedUiTargetCamera>,
}

pub(crate) fn extract_retained_borders(
    mut commands: Commands,
    state: Res<RetainedUiScene>,
    changed: Extract<
        Query<
            BorderQueryItem<'static>,
            (
                Or<(With<BorderColor>, With<Outline>)>,
                Or<(
                    Changed<ComputedNode>,
                    Changed<ComputedStackIndex>,
                    Changed<InheritedVisibility>,
                    Changed<CalculatedClip>,
                    Changed<ComputedUiTargetCamera>,
                    Changed<BorderColor>,
                    Changed<Outline>,
                    Changed<Node>,
                    Changed<ComputedUiRenderTargetInfo>,
                )>,
            ),
        >,
    >,
    all: Extract<Query<BorderQueryItem<'static>, Or<(With<BorderColor>, With<Outline>)>>>,
    camera_map: Extract<UiCameraMap>,
    mut removed: Extract<RemovedBorderInputs>,
) {
    let mut extra_candidates = bevy::platform::collections::HashSet::<Entity>::default();
    let RemovedBorderInputs {
        border,
        outline,
        clip,
        computed_node,
        node,
        stack,
        transform,
        visibility,
        camera,
    } = &mut *removed;
    let mut surfaces = state.lock();

    for entity in computed_node
        .read()
        .chain(node.read())
        .chain(stack.read())
        .chain(transform.read())
        .chain(visibility.read())
        .chain(camera.read())
    {
        remove_edges(&mut surfaces, &mut commands, entity, 0);
        remove_edges(&mut surfaces, &mut commands, entity, 4);
    }
    for entity in border.read() {
        remove_edges(&mut surfaces, &mut commands, entity, 0);
        extra_candidates.insert(entity);
    }
    for entity in outline.read() {
        remove_edges(&mut surfaces, &mut commands, entity, 4);
        extra_candidates.insert(entity);
    }
    extra_candidates.extend(clip.read());
    extra_candidates.retain(|entity| !changed.contains(*entity));

    let mut camera_mapper = camera_map.get_mapper();
    for (
        entity,
        node,
        computed,
        stack,
        transform,
        visibility,
        clip,
        owner,
        target_camera,
        border,
        outline,
    ) in changed.iter().chain(
        extra_candidates
            .into_iter()
            .filter_map(|entity| all.get(entity).ok()),
    ) {
        let Some(camera) = camera_mapper.map(target_camera) else {
            remove_edges(&mut surfaces, &mut commands, entity, 0);
            remove_edges(&mut surfaces, &mut commands, entity, 4);
            continue;
        };
        let visible = visibility.get() && node.display != Display::None && !computed.is_empty();
        let clip = retained_clip(entity, computed, transform, clip, owner);
        let transform = transform.affine();

        if let Some(border) = border {
            let colors = [
                border.left.to_linear(),
                border.top.to_linear(),
                border.right.to_linear(),
                border.bottom.to_linear(),
            ];
            let widths = computed.border();
            if visible
                && [
                    widths.min_inset.x,
                    widths.min_inset.y,
                    widths.max_inset.x,
                    widths.max_inset.y,
                ]
                .into_iter()
                .zip(colors)
                .any(|(width, color)| width > 0.0 && !color.is_fully_transparent())
            {
                upsert_edges(
                    &mut surfaces,
                    &mut commands,
                    entity,
                    camera,
                    stack.0 as f32 + stack_z_offsets::BORDER,
                    transform,
                    clip,
                    computed.size(),
                    widths,
                    computed.border_radius(),
                    colors,
                    true,
                    0,
                );
            } else {
                remove_edges(&mut surfaces, &mut commands, entity, 0);
            }
        } else {
            remove_edges(&mut surfaces, &mut commands, entity, 0);
        }

        if let Some(outline) = outline {
            if visible && computed.outline_width() > 0.0 && !outline.color.is_fully_transparent() {
                upsert_edges(
                    &mut surfaces,
                    &mut commands,
                    entity,
                    camera,
                    stack.0 as f32 + stack_z_offsets::BORDER,
                    transform,
                    clip,
                    computed.outlined_node_size(),
                    BorderRect::all(computed.outline_width()),
                    computed.outline_radius(),
                    [outline.color.to_linear(); 4],
                    true,
                    4,
                );
            } else {
                remove_edges(&mut surfaces, &mut commands, entity, 4);
            }
        } else {
            remove_edges(&mut surfaces, &mut commands, entity, 4);
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "the shared draw geometry is supplied once for four fixed edge records"
)]
fn upsert_edges(
    surfaces: &mut RetainedUiSurfaces,
    commands: &mut Commands,
    entity: Entity,
    camera: Entity,
    z_order: f32,
    transform: bevy::math::Affine2,
    clip: Option<Rect>,
    size: Vec2,
    border: BorderRect,
    border_radius: ResolvedBorderRadius,
    colors: [LinearRgba; 4],
    visible: bool,
    first_ordinal: u32,
) {
    let widths = [
        border.min_inset.x,
        border.min_inset.y,
        border.max_inset.x,
        border.max_inset.y,
    ];
    let radii: [f32; 4] = border_radius.into();
    for edge in 0..4 {
        let painted = visible && widths[edge] > 0.0 && !colors[edge].is_fully_transparent();
        surfaces.upsert(
            commands,
            border_id(entity, first_ordinal + edge as u32),
            camera,
            RetainedDraw {
                render_entity: Entity::PLACEHOLDER,
                camera,
                main_entity: entity.into(),
                z_order,
                paint_order: 0,
                clip,
                image: AssetId::<Image>::default(),
                transform,
                local_translation: Vec2::ZERO,
                item: RetainedDrawItem::Node(RetainedNodeItem {
                    color: colors[edge],
                    rect: Rect::from_corners(Vec2::ZERO, size),
                    atlas_scaling: None,
                    image_extent: None,
                    flip_x: false,
                    flip_y: false,
                    border,
                    border_radius,
                    node_type: NodeType::Border(EDGE_FLAGS[edge]),
                }),
            },
            ResourceFingerprint::None,
            painted
                .then(|| coverage_rect(edge_rect(size, widths[edge], radii, edge), transform, clip))
                .flatten()
                .into_iter()
                .collect(),
            painted,
        );
    }
}

pub(crate) fn edge_rect(size: Vec2, width: f32, radii: [f32; 4], edge: usize) -> Rect {
    let half = size * 0.5;
    let reach = match edge {
        0 => width.max(radii[0]).max(radii[3]),
        1 => width.max(radii[0]).max(radii[1]),
        2 => width.max(radii[1]).max(radii[2]),
        3 => width.max(radii[2]).max(radii[3]),
        _ => unreachable!("UI nodes have exactly four border edges"),
    } + 0.5;
    match edge {
        0 => Rect::from_corners(-half, Vec2::new(-half.x + reach.min(size.x), half.y)),
        1 => Rect::from_corners(-half, Vec2::new(half.x, -half.y + reach.min(size.y))),
        2 => Rect::from_corners(Vec2::new(half.x - reach.min(size.x), -half.y), half),
        3 => Rect::from_corners(Vec2::new(-half.x, half.y - reach.min(size.y)), half),
        _ => unreachable!("UI nodes have exactly four border edges"),
    }
}

fn remove_edges(
    surfaces: &mut RetainedUiSurfaces,
    commands: &mut Commands,
    entity: Entity,
    first_ordinal: u32,
) {
    for edge in 0..4 {
        surfaces.remove(commands, border_id(entity, first_ordinal + edge));
    }
}

fn border_id(entity: Entity, ordinal: u32) -> PaintId {
    PaintId {
        entity,
        family: PaintFamily::Border,
        ordinal,
    }
}
