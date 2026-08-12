//! Change-driven retained extraction for UI box shadows.

use crate::{
    boundary::retained_clip,
    scene::{
        coverage, PaintFamily, PaintId, ResourceFingerprint, RetainedBoxShadowItem, RetainedDraw,
        RetainedDrawItem, RetainedUiScene, RetainedUiSurfaces,
    },
};
use bevy::{
    app::Inherited,
    asset::AssetId,
    camera::visibility::InheritedVisibility,
    color::Alpha,
    ecs::{
        entity::Entity,
        lifecycle::RemovedComponents,
        query::{Changed, With},
        system::{Commands, Query, Res, ResMut, SystemParam},
    },
    image::Image,
    math::{Affine2, Vec2},
    render::{sync_world::MainEntity, Extract},
    ui::{
        BoxShadow, CalculatedClip, ComputedNode, ComputedStackIndex, ComputedUiPaintTarget,
        ComputedUiRenderTargetInfo, ComputedUiTargetCamera, Display, Node, UiGlobalTransform,
    },
    ui_render::{box_shadow::resolve_box_shadow, stack_z_offsets, BoxShadowSamples, UiCameraMap},
};
use std::collections::{HashMap, HashSet};

#[derive(bevy::prelude::Resource, Default)]
pub(crate) struct RetainedShadowDependencies {
    entities: HashMap<Entity, HashSet<PaintId>>,
}

impl RetainedShadowDependencies {
    fn set(&mut self, entity: Entity, paints: HashSet<PaintId>) {
        if paints.is_empty() {
            self.entities.remove(&entity);
        } else {
            self.entities.insert(entity, paints);
        }
    }

    fn remove(&mut self, entity: Entity) -> HashSet<PaintId> {
        self.entities.remove(&entity).unwrap_or_default()
    }
}

type ShadowQueryItem<'a> = (
    Entity,
    &'a Node,
    &'a ComputedNode,
    &'a ComputedStackIndex,
    &'a UiGlobalTransform,
    &'a InheritedVisibility,
    &'a BoxShadow,
    Option<&'a CalculatedClip>,
    Option<&'a Inherited<ComputedUiPaintTarget>>,
    &'a ComputedUiTargetCamera,
    &'a ComputedUiRenderTargetInfo,
);

#[derive(SystemParam)]
pub(crate) struct RemovedShadowInputs<'w, 's> {
    shadow: RemovedComponents<'w, 's, BoxShadow>,
    clip: RemovedComponents<'w, 's, CalculatedClip>,
    target: RemovedComponents<'w, 's, ComputedUiRenderTargetInfo>,
    samples: RemovedComponents<'w, 's, BoxShadowSamples>,
    computed_node: RemovedComponents<'w, 's, ComputedNode>,
    node: RemovedComponents<'w, 's, Node>,
    stack: RemovedComponents<'w, 's, ComputedStackIndex>,
    transform: RemovedComponents<'w, 's, UiGlobalTransform>,
    visibility: RemovedComponents<'w, 's, InheritedVisibility>,
    camera: RemovedComponents<'w, 's, ComputedUiTargetCamera>,
}

pub(crate) fn extract_retained_shadows(
    mut commands: Commands,
    state: Res<RetainedUiScene>,
    mut dependencies: ResMut<RetainedShadowDependencies>,
    changed: Extract<
        Query<
            ShadowQueryItem<'static>,
            (
                With<BoxShadow>,
                bevy::ecs::query::Or<(
                    Changed<ComputedNode>,
                    Changed<ComputedStackIndex>,
                    Changed<InheritedVisibility>,
                    Changed<CalculatedClip>,
                    Changed<ComputedUiTargetCamera>,
                    Changed<ComputedUiRenderTargetInfo>,
                    Changed<BoxShadow>,
                    Changed<Node>,
                )>,
            ),
        >,
    >,
    all: Extract<Query<ShadowQueryItem<'static>, With<BoxShadow>>>,
    camera_map: Extract<UiCameraMap>,
    samples: Extract<Query<&'static BoxShadowSamples>>,
    changed_samples: Extract<Query<Entity, Changed<BoxShadowSamples>>>,
    mut removed: Extract<RemovedShadowInputs>,
) {
    let mut extra_candidates = HashSet::new();
    let RemovedShadowInputs {
        shadow,
        clip,
        target,
        samples: removed_samples,
        computed_node,
        node,
        stack,
        transform,
        visibility,
        camera,
    } = &mut *removed;
    extra_candidates.extend(shadow.read());
    extra_candidates.extend(clip.read());
    extra_candidates.extend(target.read());
    let changed_cameras: HashSet<_> = changed_samples
        .iter()
        .chain(removed_samples.read())
        .collect();
    if !changed_cameras.is_empty() {
        extra_candidates.extend(all.iter().filter_map(|item| {
            item.9
                .get()
                .filter(|camera| changed_cameras.contains(camera))
                .map(|_| item.0)
        }));
    }
    extra_candidates.retain(|entity| !changed.contains(*entity));

    let mut surfaces = state.lock();
    for entity in computed_node
        .read()
        .chain(node.read())
        .chain(stack.read())
        .chain(transform.read())
        .chain(visibility.read())
        .chain(camera.read())
    {
        remove_entity(&mut dependencies, &mut surfaces, &mut commands, entity);
    }
    extra_candidates.retain(|entity| {
        if all.contains(*entity) {
            true
        } else {
            remove_entity(&mut dependencies, &mut surfaces, &mut commands, *entity);
            false
        }
    });

    let mut camera_mapper = camera_map.get_mapper();
    for (
        entity,
        source_node,
        node,
        stack,
        transform,
        visibility,
        shadows,
        clip,
        owner,
        target_camera,
        target,
    ) in changed.iter().chain(
        extra_candidates
            .into_iter()
            .filter_map(|entity| all.get(entity).ok()),
    ) {
        let Some(camera) = camera_mapper.map(target_camera) else {
            remove_entity(&mut dependencies, &mut surfaces, &mut commands, entity);
            continue;
        };
        let shadow_samples = target_camera
            .get()
            .and_then(|camera| samples.get(camera).ok())
            .copied()
            .unwrap_or_default()
            .0;
        let visible = visibility.get()
            && source_node.display != Display::None
            && !node.is_empty()
            && !node.size().cmple(Vec2::ZERO).any();
        let clip = retained_clip(entity, node, transform, clip, owner);
        let node_transform = transform.affine();
        let mut paints = HashSet::new();
        for (ordinal, logical) in shadows.iter().enumerate() {
            let Some(shadow) = resolve_box_shadow(
                logical,
                node.size(),
                node.border_radius(),
                target.scale_factor(),
                target.physical_size().as_vec2(),
            ) else {
                continue;
            };
            let id = PaintId {
                entity,
                family: PaintFamily::BoxShadow,
                ordinal: u32::try_from(ordinal).expect("box shadow count exceeds u32"),
            };
            let transform = node_transform * Affine2::from_translation(shadow.offset);
            let painted = visible && !shadow.color.is_fully_transparent();
            let coverage = painted
                .then(|| coverage(shadow.bounds, transform, clip))
                .flatten()
                .into_iter()
                .collect::<crate::PaintCoverage>();
            surfaces.upsert(
                &mut commands,
                id,
                camera,
                RetainedDraw {
                    render_entity: Entity::PLACEHOLDER,
                    camera,
                    main_entity: MainEntity::from(entity),
                    z_order: stack.0 as f32 + stack_z_offsets::BOX_SHADOW,
                    paint_order: u32::try_from(ordinal).expect("box shadow count exceeds u32"),
                    clip,
                    image: AssetId::<Image>::default(),
                    transform,
                    local_translation: shadow.offset,
                    item: RetainedDrawItem::BoxShadow(RetainedBoxShadowItem::new(
                        stack.0,
                        shadow,
                        shadow_samples,
                    )),
                },
                ResourceFingerprint::None,
                coverage,
                painted,
            );
            if painted {
                paints.insert(id);
            }
        }

        for old in dependencies.remove(entity) {
            if !paints.contains(&old) {
                surfaces.remove(&mut commands, old);
            }
        }
        dependencies.set(entity, paints);
    }
}

fn remove_entity(
    dependencies: &mut RetainedShadowDependencies,
    surfaces: &mut RetainedUiSurfaces,
    commands: &mut Commands,
    entity: Entity,
) {
    for id in dependencies.remove(entity) {
        surfaces.remove(commands, id);
    }
}
