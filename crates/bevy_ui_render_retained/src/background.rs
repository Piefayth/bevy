//! Change-driven retained extraction for UI backgrounds.

use crate::scene::{
    coverage, PaintFamily, PaintId, ResourceFingerprint, RetainedDraw, RetainedDrawItem,
    RetainedNodeItem, RetainedUiScene,
};
use bevy::{
    camera::visibility::InheritedVisibility,
    color::Alpha,
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
        BackgroundColor, CalculatedClip, ComputedNode, ComputedStackIndex,
        ComputedUiRenderTargetInfo, ComputedUiTargetCamera, Node, OuterColor, UiGlobalTransform,
    },
    ui_render::{stack_z_offsets, NodeType, UiCameraMap},
};

type BackgroundQueryItem<'a> = (
    Entity,
    &'a ComputedNode,
    &'a ComputedStackIndex,
    &'a UiGlobalTransform,
    &'a InheritedVisibility,
    Option<&'a CalculatedClip>,
    &'a ComputedUiTargetCamera,
    &'a BackgroundColor,
    Option<&'a OuterColor>,
);

#[derive(SystemParam)]
pub(crate) struct RemovedBackgroundInputs<'w, 's> {
    background: RemovedComponents<'w, 's, BackgroundColor>,
    outer: RemovedComponents<'w, 's, OuterColor>,
    clip: RemovedComponents<'w, 's, CalculatedClip>,
    computed_node: RemovedComponents<'w, 's, ComputedNode>,
    node: RemovedComponents<'w, 's, Node>,
    stack: RemovedComponents<'w, 's, ComputedStackIndex>,
    transform: RemovedComponents<'w, 's, UiGlobalTransform>,
    visibility: RemovedComponents<'w, 's, InheritedVisibility>,
    camera: RemovedComponents<'w, 's, ComputedUiTargetCamera>,
}

pub(crate) fn extract_retained_backgrounds(
    mut commands: Commands,
    state: Res<RetainedUiScene>,
    changed: Extract<
        Query<
            BackgroundQueryItem<'static>,
            (
                With<BackgroundColor>,
                Or<(
                    Changed<ComputedNode>,
                    Changed<ComputedStackIndex>,
                    Changed<UiGlobalTransform>,
                    Changed<InheritedVisibility>,
                    Changed<CalculatedClip>,
                    Changed<ComputedUiTargetCamera>,
                    Changed<BackgroundColor>,
                    Changed<OuterColor>,
                    Changed<Node>,
                    Changed<ComputedUiRenderTargetInfo>,
                )>,
            ),
        >,
    >,
    all: Extract<Query<BackgroundQueryItem<'static>, With<BackgroundColor>>>,
    camera_map: Extract<UiCameraMap>,
    mut removed: Extract<RemovedBackgroundInputs>,
) {
    let mut surfaces = state.lock();

    let RemovedBackgroundInputs {
        background,
        outer,
        clip,
        computed_node,
        node,
        stack,
        transform,
        visibility,
        camera,
    } = &mut *removed;

    for entity in background
        .read()
        .chain(computed_node.read())
        .chain(node.read())
        .chain(stack.read())
        .chain(transform.read())
        .chain(visibility.read())
        .chain(camera.read())
    {
        surfaces.remove(&mut commands, background_id(entity, 0));
        surfaces.remove(&mut commands, background_id(entity, 1));
    }

    for entity in outer.read() {
        surfaces.remove(&mut commands, background_id(entity, 1));
    }

    let mut camera_mapper = camera_map.get_mapper();
    let rebuilt_after_removal = clip
        .read()
        .filter(|entity| !changed.contains(*entity))
        .filter_map(|entity| all.get(entity).ok());
    for (entity, node, stack, transform, visibility, clip, target_camera, background, outer) in
        changed.iter().chain(rebuilt_after_removal)
    {
        let fill_id = background_id(entity, 0);
        let outer_id = background_id(entity, 1);
        let Some(camera) = camera_mapper.map(target_camera) else {
            surfaces.remove(&mut commands, fill_id);
            surfaces.remove(&mut commands, outer_id);
            continue;
        };
        let transform = transform.affine();
        let clip = clip.map(|clip| clip.clip);
        let visible = visibility.get() && !node.is_empty();
        let z_order = stack.0 as f32 + stack_z_offsets::BACKGROUND_COLOR;
        let base_item = RetainedNodeItem {
            color: background.0.into(),
            rect: Rect {
                min: Vec2::ZERO,
                max: node.size,
            },
            atlas_scaling: None,
            flip_x: false,
            flip_y: false,
            border: node.border(),
            border_radius: node.border_radius(),
            node_type: NodeType::Rect,
        };
        let base = RetainedDraw {
            render_entity: Entity::PLACEHOLDER,
            camera,
            main_entity: entity.into(),
            z_order,
            clip,
            image: bevy::asset::AssetId::<Image>::default(),
            transform,
            item: RetainedDrawItem::Node(base_item),
        };
        let fill_painted = visible && !background.is_fully_transparent();
        surfaces.upsert(
            &mut commands,
            fill_id,
            camera,
            base.clone(),
            ResourceFingerprint::None,
            fill_painted
                .then(|| coverage(node.size, transform, clip))
                .flatten()
                .into_iter()
                .collect(),
        );

        if let Some(outer) = outer {
            let outer_painted = visible && !outer.is_fully_transparent();
            surfaces.upsert(
                &mut commands,
                outer_id,
                camera,
                RetainedDraw {
                    item: RetainedDrawItem::Node(RetainedNodeItem {
                        color: outer.0.into(),
                        border: BorderRect::ZERO,
                        node_type: NodeType::Inverted,
                        ..base_item
                    }),
                    ..base
                },
                ResourceFingerprint::None,
                outer_painted
                    .then(|| coverage(node.size, transform, clip))
                    .flatten()
                    .into_iter()
                    .collect(),
            );
        } else {
            surfaces.remove(&mut commands, outer_id);
        }
    }
}

fn background_id(entity: Entity, ordinal: u32) -> PaintId {
    PaintId {
        entity,
        family: PaintFamily::Background,
        ordinal,
    }
}
