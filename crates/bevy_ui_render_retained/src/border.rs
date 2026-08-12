//! Change-driven retained extraction for solid borders and outlines.

use crate::{
    boundary::retained_clip,
    scene::{
        coverage_rect, PaintFamily, PaintId, ResourceFingerprint, RetainedBorderItem, RetainedDraw,
        RetainedDrawItem, RetainedUiScene, RetainedUiSurfaces,
    },
};
use bevy::{
    app::Inherited,
    asset::AssetId,
    camera::visibility::InheritedVisibility,
    color::{Alpha, LinearRgba},
    ecs::{
        entity::{Entity, EntityHashMap, EntityHashSet},
        lifecycle::RemovedComponents,
        query::{Changed, Or, With},
        system::{Commands, Query, Res, ResMut, SystemParam},
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
    ui_render::{shader_flags, stack_z_offsets, UiCameraMap},
};

#[derive(bevy::prelude::Resource, Default)]
pub(crate) struct RetainedBorderDependencies {
    active: EntityHashMap<u8>,
}
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
    style_changed: Extract<Query<Entity, Or<(Changed<BorderColor>, Changed<Outline>)>>>,
    geometry_changed: Extract<
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
                    Changed<Node>,
                    Changed<ComputedUiRenderTargetInfo>,
                )>,
            ),
        >,
    >,
    all: Extract<Query<BorderQueryItem<'static>, Or<(With<BorderColor>, With<Outline>)>>>,
    camera_map: Extract<UiCameraMap>,
    mut removed: Extract<RemovedBorderInputs>,
    mut dependencies: ResMut<RetainedBorderDependencies>,
) {
    let mut candidates = EntityHashSet::default();
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
        dependencies.active.remove(&entity);
        surfaces.remove(&mut commands, border_id(entity, 0));
        surfaces.remove(&mut commands, border_id(entity, 1));
        candidates.insert(entity);
    }
    for entity in border.read() {
        surfaces.remove(&mut commands, border_id(entity, 0));
        candidates.insert(entity);
    }
    for entity in outline.read() {
        surfaces.remove(&mut commands, border_id(entity, 1));
        candidates.insert(entity);
    }
    candidates.extend(
        clip.read()
            .filter(|entity| dependencies.active.contains_key(entity)),
    );
    candidates.extend(style_changed.iter());
    let mut camera_mapper = camera_map.get_mapper();
    candidates.retain(|candidate| {
        if all.contains(*candidate) {
            true
        } else {
            dependencies.active.remove(candidate);
            surfaces.remove(&mut commands, border_id(*candidate, 0));
            surfaces.remove(&mut commands, border_id(*candidate, 1));
            false
        }
    });
    for item in candidates.iter().filter_map(|&entity| all.get(entity).ok()) {
        let (entity, _, _, _, _, _, _, _, target_camera, _, _) = item;
        let old = dependencies.active.get(&entity).copied().unwrap_or(0);
        let Some(camera) = camera_mapper.map(target_camera) else {
            surfaces.remove(&mut commands, border_id(entity, 0));
            surfaces.remove(&mut commands, border_id(entity, 1));
            dependencies.active.remove(&entity);
            continue;
        };
        let active = update_borders(&mut surfaces, &mut commands, item, camera, old);
        if active == 0 {
            dependencies.active.remove(&entity);
        } else {
            dependencies.active.insert(entity, active);
        }
    }
    for item in geometry_changed.iter() {
        let entity = item.0;
        if candidates.contains(&entity) || !dependencies.active.contains_key(&entity) {
            continue;
        }
        let old = dependencies.active[&entity];
        let Some(camera) = camera_mapper.map(item.8) else {
            surfaces.remove(&mut commands, border_id(entity, 0));
            surfaces.remove(&mut commands, border_id(entity, 1));
            dependencies.active.remove(&entity);
            continue;
        };
        let active = update_borders(&mut surfaces, &mut commands, item, camera, old);
        if active == 0 {
            dependencies.active.remove(&entity);
        } else {
            dependencies.active.insert(entity, active);
        }
    }
}

fn update_borders(
    surfaces: &mut RetainedUiSurfaces,
    commands: &mut Commands,
    item: BorderQueryItem<'_>,
    camera: Entity,
    old: u8,
) -> u8 {
    let (entity, node, computed, stack, transform, visibility, clip, owner, _, border, outline) =
        item;
    let visible = visibility.get() && node.display != Display::None && !computed.is_empty();
    let clip = retained_clip(entity, computed, transform, clip, owner);
    let transform = transform.affine();
    let mut active = 0;

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
            upsert_border(
                surfaces,
                commands,
                entity,
                camera,
                stack.0 as f32 + stack_z_offsets::BORDER,
                transform,
                clip,
                computed.size(),
                widths,
                computed.border_radius(),
                colors,
                0,
            );
            active |= 1;
        } else if old & 1 != 0 {
            surfaces.remove(commands, border_id(entity, 0));
        }
    } else if old & 1 != 0 {
        surfaces.remove(commands, border_id(entity, 0));
    }

    if let Some(outline) = outline {
        if visible && computed.outline_width() > 0.0 && !outline.color.is_fully_transparent() {
            upsert_border(
                surfaces,
                commands,
                entity,
                camera,
                stack.0 as f32 + stack_z_offsets::BORDER,
                transform,
                clip,
                computed.outlined_node_size(),
                BorderRect::all(computed.outline_width()),
                computed.outline_radius(),
                [outline.color.to_linear(); 4],
                1,
            );
            active |= 2;
        } else if old & 2 != 0 {
            surfaces.remove(commands, border_id(entity, 1));
        }
    } else if old & 2 != 0 {
        surfaces.remove(commands, border_id(entity, 1));
    }
    active
}

#[expect(
    clippy::too_many_arguments,
    reason = "one border has independent target, geometry, and four edge colors"
)]
fn upsert_border(
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
    ordinal: u32,
) {
    let widths = [
        border.min_inset.x,
        border.min_inset.y,
        border.max_inset.x,
        border.max_inset.y,
    ];
    let radii: [f32; 4] = border_radius.into();
    let edge_coverage = core::array::from_fn(|edge| {
        (widths[edge] > 0.0 && !colors[edge].is_fully_transparent())
            .then(|| coverage_rect(edge_rect(size, widths[edge], radii, edge), transform, clip))
            .flatten()
    });
    let coverage = edge_coverage.iter().copied().flatten().collect();
    surfaces.upsert(
        commands,
        border_id(entity, ordinal),
        camera,
        RetainedDraw {
            render_entity: Entity::PLACEHOLDER,
            camera,
            main_entity: entity.into(),
            z_order,
            paint_order: ordinal,
            clip,
            image: AssetId::<Image>::default(),
            transform,
            layout_translation: Vec2::ZERO,
            local_translation: Vec2::ZERO,
            item: RetainedDrawItem::Border(RetainedBorderItem::new(
                Rect::from_corners(Vec2::ZERO, size),
                border,
                border_radius,
                colors,
                edge_coverage,
            )),
        },
        ResourceFingerprint::None,
        coverage,
        true,
    );
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

fn border_id(entity: Entity, ordinal: u32) -> PaintId {
    PaintId {
        entity,
        family: PaintFamily::Border,
        ordinal,
    }
}
