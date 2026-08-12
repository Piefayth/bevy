//! Change-driven retained extraction for UI backgrounds.

use crate::{
    boundary::retained_clip,
    scene::{
        coverage, PaintFamily, PaintId, PendingRetainedPaint, ResourceFingerprint, RetainedDraw,
        RetainedDrawItem, RetainedNodeItem,
    },
};
use bevy::{
    app::Inherited,
    camera::visibility::InheritedVisibility,
    color::Alpha,
    ecs::{
        entity::Entity,
        lifecycle::RemovedComponents,
        query::{Changed, Or, With},
        system::{Query, ResMut, SystemParam},
    },
    image::Image,
    math::{Rect, Vec2},
    render::sync_world::MainEntity,
    render::Extract,
    sprite::BorderRect,
    ui::{
        BackgroundColor, CalculatedClip, ComputedNode, ComputedStackIndex, ComputedUiPaintTarget,
        ComputedUiRenderTargetInfo, ComputedUiTargetCamera, Node, OuterColor, UiGlobalTransform,
    },
    ui_render::{stack_z_offsets, NodeType, UiCameraMap},
};

struct PendingBackground {
    id: PaintId,
    camera: Entity,
    main_entity: MainEntity,
    z_order: f32,
    clip: Option<Rect>,
    transform: bevy::math::Affine2,
    item: RetainedNodeItem,
    painted: bool,
}

#[derive(bevy::prelude::Resource, Default)]
pub(crate) struct PendingRetainedBackgrounds {
    upserts: Vec<PendingBackground>,
    removals: Vec<PaintId>,
}

impl PendingBackground {
    fn into_retained(self) -> PendingRetainedPaint {
        let coverage = self
            .painted
            .then(|| coverage(self.item.rect.size(), self.transform, self.clip))
            .flatten()
            .into_iter()
            .collect();
        PendingRetainedPaint {
            id: self.id,
            camera: self.camera,
            draw: RetainedDraw {
                render_entity: Entity::PLACEHOLDER,
                camera: self.camera,
                main_entity: self.main_entity,
                z_order: self.z_order,
                paint_order: 0,
                clip: self.clip,
                image: bevy::asset::AssetId::<Image>::default(),
                transform: self.transform,
                layout_translation: Vec2::ZERO,
                local_translation: Vec2::ZERO,
                item: RetainedDrawItem::Node(self.item),
            },
            resource: ResourceFingerprint::None,
            coverage,
            painted: self.painted,
        }
    }
}

impl PendingRetainedBackgrounds {
    pub(crate) fn removals(&mut self) -> impl Iterator<Item = PaintId> + '_ {
        self.removals.drain(..)
    }

    pub(crate) fn upserts(&mut self) -> impl Iterator<Item = PendingRetainedPaint> + '_ {
        self.upserts.drain(..).map(PendingBackground::into_retained)
    }
}
type BackgroundQueryItem<'a> = (
    Entity,
    &'a ComputedNode,
    &'a ComputedStackIndex,
    &'a UiGlobalTransform,
    &'a InheritedVisibility,
    Option<&'a CalculatedClip>,
    Option<&'a Inherited<ComputedUiPaintTarget>>,
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
    changed: Extract<
        Query<
            BackgroundQueryItem<'static>,
            (
                With<BackgroundColor>,
                Or<(
                    Changed<ComputedNode>,
                    Changed<ComputedStackIndex>,
                    Changed<InheritedVisibility>,
                    Changed<CalculatedClip>,
                    Changed<ComputedUiTargetCamera>,
                    Changed<ComputedUiRenderTargetInfo>,
                    Changed<BackgroundColor>,
                    Changed<OuterColor>,
                )>,
            ),
        >,
    >,
    all: Extract<Query<BackgroundQueryItem<'static>, With<BackgroundColor>>>,
    camera_map: Extract<UiCameraMap>,
    mut removed: Extract<RemovedBackgroundInputs>,
    mut pending: ResMut<PendingRetainedBackgrounds>,
) {
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

    let mut camera_mapper = camera_map.get_mapper();
    let PendingRetainedBackgrounds {
        upserts: staged,
        removals,
    } = &mut *pending;
    staged.reserve(changed.iter().size_hint().0);
    for entity in background
        .read()
        .chain(computed_node.read())
        .chain(node.read())
        .chain(stack.read())
        .chain(transform.read())
        .chain(visibility.read())
        .chain(camera.read())
    {
        removals.push(background_id(entity, 0));
        removals.push(background_id(entity, 1));
    }
    for entity in outer.read() {
        removals.push(background_id(entity, 1));
    }

    staged.reserve(changed.iter().size_hint().0);
    for (
        entity,
        node,
        stack,
        transform,
        visibility,
        clip,
        owner,
        target_camera,
        background,
        outer,
    ) in changed.iter().chain(
        clip.read()
            .filter(|entity| !changed.contains(*entity))
            .filter_map(|entity| all.get(entity).ok()),
    ) {
        let fill_id = background_id(entity, 0);
        let outer_id = background_id(entity, 1);
        let Some(camera) = camera_mapper.map(target_camera) else {
            removals.push(fill_id);
            removals.push(outer_id);
            continue;
        };
        let clip = retained_clip(entity, node, transform, clip, owner);
        let transform = transform.affine();
        let visible = visibility.get() && !node.is_empty();
        let z_order = stack.0 as f32 + stack_z_offsets::BACKGROUND_COLOR;
        let base_item = RetainedNodeItem {
            color: background.0.into(),
            rect: Rect {
                min: Vec2::ZERO,
                max: node.size,
            },
            atlas_scaling: None,
            image_extent: None,
            flip_x: false,
            flip_y: false,
            border: node.border(),
            border_radius: node.border_radius(),
            node_type: NodeType::Rect,
        };
        let fill_painted = visible && !background.is_fully_transparent();
        let outer_painted = outer.is_some_and(|outer| visible && !outer.is_fully_transparent());
        let outer_item = outer.map(|outer| RetainedNodeItem {
            color: outer.0.into(),
            border: BorderRect::ZERO,
            node_type: NodeType::Inverted,
            ..base_item
        });
        if let Some(outer_item) = outer_item {
            staged.push(background_change(
                outer_id,
                camera,
                entity.into(),
                z_order,
                clip,
                transform,
                outer_item,
                outer_painted,
            ));
        }
        staged.push(background_change(
            fill_id,
            camera,
            entity.into(),
            z_order,
            clip,
            transform,
            base_item,
            fill_painted,
        ));
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "a background change contains one canonical retained draw"
)]
fn background_change(
    id: PaintId,
    camera: Entity,
    main_entity: MainEntity,
    z_order: f32,
    clip: Option<Rect>,
    transform: bevy::math::Affine2,
    item: RetainedNodeItem,
    painted: bool,
) -> PendingBackground {
    PendingBackground {
        id,
        camera,
        main_entity,
        z_order,
        clip,
        transform,
        item,
        painted,
    }
}

fn background_id(entity: Entity, ordinal: u32) -> PaintId {
    PaintId {
        entity,
        family: PaintFamily::Background,
        ordinal,
    }
}
