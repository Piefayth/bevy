//! Change-driven retained extraction for background and border gradients.

use crate::{
    border::edge_rect,
    boundary::retained_clip,
    scene::{
        coverage, coverage_rect, PaintFamily, PaintId, PendingGradientPaints, PendingRetainedPaint,
        ResourceFingerprint, RetainedDraw, RetainedDrawItem, RetainedGradientItem,
    },
};
use bevy::{
    app::Inherited,
    asset::AssetId,
    camera::visibility::InheritedVisibility,
    color::Alpha,
    ecs::{
        entity::{Entity, EntityHashMap, EntityHashSet},
        lifecycle::RemovedComponents,
        query::{Changed, Or, With},
        system::{Query, ResMut, SystemParam},
    },
    image::Image,
    math::{Rect, Vec2},
    render::{sync_world::MainEntity, Extract},
    ui::{
        BackgroundGradient, BorderGradient, CalculatedClip, ComputedNode, ComputedStackIndex,
        ComputedUiPaintTarget, ComputedUiRenderTargetInfo, ComputedUiTargetCamera, Display, Node,
        UiGlobalTransform,
    },
    ui_render::{gradient::resolve_gradient, stack_z_offsets, UiCameraMap},
};
use smallvec::SmallVec;
#[derive(bevy::prelude::Resource, Default)]
pub(crate) struct RetainedGradientDependencies {
    entities: EntityHashMap<SmallVec<[PaintId; 4]>>,
}

impl RetainedGradientDependencies {
    fn set(&mut self, entity: Entity, paints: SmallVec<[PaintId; 4]>) {
        if paints.is_empty() {
            self.entities.remove(&entity);
        } else {
            self.entities.insert(entity, paints);
        }
    }

    fn remove(&mut self, entity: Entity) -> SmallVec<[PaintId; 4]> {
        self.entities.remove(&entity).unwrap_or_default()
    }
}

type GradientQueryItem<'a> = (
    Entity,
    &'a Node,
    &'a ComputedNode,
    &'a ComputedStackIndex,
    &'a UiGlobalTransform,
    &'a InheritedVisibility,
    Option<&'a CalculatedClip>,
    Option<&'a Inherited<ComputedUiPaintTarget>>,
    &'a ComputedUiTargetCamera,
    &'a ComputedUiRenderTargetInfo,
    Option<&'a BackgroundGradient>,
    Option<&'a BorderGradient>,
);

#[derive(SystemParam)]
pub(crate) struct RemovedGradientInputs<'w, 's> {
    background: RemovedComponents<'w, 's, BackgroundGradient>,
    border: RemovedComponents<'w, 's, BorderGradient>,
    clip: RemovedComponents<'w, 's, CalculatedClip>,
    target: RemovedComponents<'w, 's, ComputedUiRenderTargetInfo>,
    computed_node: RemovedComponents<'w, 's, ComputedNode>,
    node: RemovedComponents<'w, 's, Node>,
    stack: RemovedComponents<'w, 's, ComputedStackIndex>,
    transform: RemovedComponents<'w, 's, UiGlobalTransform>,
    visibility: RemovedComponents<'w, 's, InheritedVisibility>,
    camera: RemovedComponents<'w, 's, ComputedUiTargetCamera>,
}

pub(crate) fn extract_retained_gradients(
    mut dependencies: ResMut<RetainedGradientDependencies>,
    changed: Extract<
        Query<
            GradientQueryItem<'static>,
            (
                Or<(With<BackgroundGradient>, With<BorderGradient>)>,
                Or<(
                    Changed<ComputedNode>,
                    Changed<ComputedStackIndex>,
                    Changed<InheritedVisibility>,
                    Changed<CalculatedClip>,
                    Changed<ComputedUiTargetCamera>,
                    Changed<ComputedUiRenderTargetInfo>,
                    Changed<BackgroundGradient>,
                    Changed<BorderGradient>,
                    Changed<Node>,
                )>,
            ),
        >,
    >,
    all: Extract<
        Query<GradientQueryItem<'static>, Or<(With<BackgroundGradient>, With<BorderGradient>)>>,
    >,
    camera_map: Extract<UiCameraMap>,
    mut removed: Extract<RemovedGradientInputs>,
    mut pending: ResMut<PendingGradientPaints>,
) {
    let mut extra_candidates = EntityHashSet::default();
    let RemovedGradientInputs {
        background,
        border,
        clip,
        target,
        computed_node,
        node,
        stack,
        transform,
        visibility,
        camera,
    } = &mut *removed;
    extra_candidates.extend(background.read());
    extra_candidates.extend(border.read());
    extra_candidates.extend(clip.read());
    extra_candidates.extend(target.read());
    extra_candidates.retain(|entity| !changed.contains(*entity));

    for entity in computed_node
        .read()
        .chain(node.read())
        .chain(stack.read())
        .chain(transform.read())
        .chain(visibility.read())
        .chain(camera.read())
    {
        for id in dependencies.remove(entity) {
            pending.remove(id);
        }
    }
    extra_candidates.retain(|entity| {
        if all.contains(*entity) {
            true
        } else {
            for id in dependencies.remove(*entity) {
                pending.remove(id);
            }
            false
        }
    });

    let mut camera_mapper = camera_map.get_mapper();
    let mut scratch = Vec::new();
    let mut resolved_stops = Vec::new();
    pending.reserve(changed.iter().size_hint().0);
    for (
        entity,
        source_node,
        node,
        stack,
        transform,
        visibility,
        clip,
        owner,
        target_camera,
        target,
        backgrounds,
        borders,
    ) in changed.iter().chain(
        extra_candidates
            .into_iter()
            .filter_map(|entity| all.get(entity).ok()),
    ) {
        let Some(camera) = camera_mapper.map(target_camera) else {
            for id in dependencies.remove(entity) {
                pending.remove(id);
            }
            continue;
        };

        let visible = visibility.get()
            && source_node.display != Display::None
            && !node.is_empty()
            && !node.size().cmple(Vec2::ZERO).any();
        let clip = retained_clip(entity, node, transform, clip, owner);
        let transform = transform.affine();
        let mut paint_ids = SmallVec::new();
        for (gradients, border_gradient) in [
            (backgrounds.map(|gradients| &gradients.0), false),
            (borders.map(|gradients| &gradients.0), true),
        ] {
            let Some(gradients) = gradients else {
                continue;
            };
            for (ordinal, gradient) in gradients.iter().enumerate() {
                if gradient.is_empty() {
                    continue;
                }
                resolved_stops.clear();
                let (resolved, color_space) = resolve_gradient(
                    gradient,
                    target.scale_factor(),
                    node.size(),
                    target.physical_size().as_vec2(),
                    &mut scratch,
                    &mut resolved_stops,
                );
                let painted = visible
                    && resolved_stops
                        .iter()
                        .any(|(color, _, _)| !color.is_fully_transparent());
                let id = PaintId {
                    entity,
                    family: if border_gradient {
                        PaintFamily::BorderGradient
                    } else {
                        PaintFamily::Gradient
                    },
                    ordinal: u32::try_from(ordinal).expect("gradient count exceeds u32"),
                };
                let coverage = gradient_coverage(node, transform, clip, painted, border_gradient);
                pending.upsert(PendingRetainedPaint {
                    id,
                    camera,
                    draw: RetainedDraw {
                        render_entity: Entity::PLACEHOLDER,
                        camera,
                        main_entity: MainEntity::from(entity),
                        z_order: stack.0 as f32
                            + if border_gradient {
                                stack_z_offsets::BORDER_GRADIENT
                            } else {
                                stack_z_offsets::GRADIENT
                            },
                        paint_order: u32::try_from(ordinal).expect("gradient count exceeds u32"),
                        clip,
                        image: AssetId::<Image>::default(),
                        transform,
                        layout_translation: Vec2::ZERO,
                        local_translation: Vec2::ZERO,
                        item: RetainedDrawItem::Gradient(RetainedGradientItem::new(
                            stack.0,
                            Rect::from_corners(Vec2::ZERO, node.size()),
                            node.border_radius(),
                            node.border(),
                            resolved,
                            color_space,
                            &resolved_stops,
                            border_gradient,
                        )),
                    },
                    resource: ResourceFingerprint::None,
                    coverage,
                    painted,
                });
                if painted {
                    paint_ids.push(id);
                }
            }
        }

        for old in dependencies.remove(entity) {
            if !paint_ids.contains(&old) {
                pending.remove(old);
            }
        }
        dependencies.set(entity, paint_ids);
    }
}

fn gradient_coverage(
    node: &ComputedNode,
    transform: bevy::math::Affine2,
    clip: Option<Rect>,
    painted: bool,
    border_gradient: bool,
) -> crate::PaintCoverage {
    if !painted {
        return Default::default();
    }
    if !border_gradient {
        return coverage(node.size(), transform, clip).into_iter().collect();
    }
    let border = node.border();
    let widths = [
        border.min_inset.x,
        border.min_inset.y,
        border.max_inset.x,
        border.max_inset.y,
    ];
    let radii: [f32; 4] = node.border_radius().into();
    widths
        .into_iter()
        .enumerate()
        .filter(|(_, width)| *width > 0.0)
        .filter_map(|(edge, width)| {
            coverage_rect(edge_rect(node.size(), width, radii, edge), transform, clip)
        })
        .collect()
}
