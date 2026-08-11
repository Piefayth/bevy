//! Change-driven retained extraction for UI backgrounds.

use crate::{FloatBits, PaintRecord, PhysicalRect, RepairPlan, RetainedPaint, WorkCounters};
use bevy::{
    camera::visibility::InheritedVisibility,
    color::{Alpha, ColorToComponents, LinearRgba},
    ecs::{
        entity::Entity,
        lifecycle::RemovedComponents,
        query::{Changed, Or, With},
        system::{Commands, Query, Res, ResMut, SystemParam},
    },
    image::Image,
    math::{Affine2, Rect, Vec2},
    render::{sync_world::MainEntity, Extract},
    sprite::BorderRect,
    ui::{
        BackgroundColor, CalculatedClip, ComputedNode, ComputedStackIndex,
        ComputedUiRenderTargetInfo, ComputedUiTargetCamera, Node, OuterColor, ResolvedBorderRadius,
        UiGlobalTransform,
    },
    ui_render::{
        stack_z_offsets, ExtractedUiItem, ExtractedUiNode, ExtractedUiNodes, NodeType, UiCameraMap,
    },
};
use core::sync::atomic::{AtomicU64, Ordering};
use std::{
    collections::HashMap,
    collections::HashSet,
    sync::{Mutex, PoisonError},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum BackgroundKind {
    Fill,
    Outer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct BackgroundId {
    entity: Entity,
    kind: BackgroundKind,
}

#[derive(Clone, Copy)]
struct BackgroundDraw {
    render_entity: Entity,
    camera: Entity,
    main_entity: MainEntity,
    z_order: f32,
    clip: Option<Rect>,
    transform: Affine2,
    color: LinearRgba,
    size: Vec2,
    border: BorderRect,
    border_radius: ResolvedBorderRadius,
    node_type: NodeType,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BackgroundFingerprint {
    camera: Entity,
    z_order: FloatBits,
    clip: Option<[FloatBits; 4]>,
    transform: [FloatBits; 6],
    color: [FloatBits; 4],
    size: [FloatBits; 2],
    border: [FloatBits; 4],
    border_radius: [FloatBits; 4],
    node_type: u8,
}

#[derive(Clone)]
struct BackgroundRecord {
    fingerprint: BackgroundFingerprint,
    render_entity: Entity,
    draw: Option<BackgroundDraw>,
}

impl PartialEq for BackgroundRecord {
    fn eq(&self, other: &Self) -> bool {
        self.fingerprint == other.fingerprint
    }
}

impl BackgroundRecord {
    fn new(draw: BackgroundDraw, visible: bool) -> Self {
        let clip = draw
            .clip
            .map(|clip| [clip.min.x, clip.min.y, clip.max.x, clip.max.y].map(FloatBits::new));
        let transform = draw.transform.to_cols_array().map(FloatBits::new);
        let color = draw.color.to_f32_array().map(FloatBits::new);
        let size = draw.size.to_array().map(FloatBits::new);
        let border = [
            draw.border.min_inset.x,
            draw.border.min_inset.y,
            draw.border.max_inset.x,
            draw.border.max_inset.y,
        ]
        .map(FloatBits::new);
        let border_radius: [f32; 4] = draw.border_radius.into();
        let node_type = match draw.node_type {
            NodeType::Rect => 0,
            NodeType::Inverted => 1,
            NodeType::Border(_) => unreachable!("background extraction does not create borders"),
        };
        Self {
            fingerprint: BackgroundFingerprint {
                camera: draw.camera,
                z_order: FloatBits::new(draw.z_order),
                clip,
                transform,
                color,
                size,
                border,
                border_radius: border_radius.map(FloatBits::new),
                node_type,
            },
            render_entity: draw.render_entity,
            draw: visible.then_some(draw),
        }
    }
}

#[derive(Default)]
struct BackgroundSurfaces {
    paint: HashMap<Entity, RetainedPaint<BackgroundId, BackgroundRecord>>,
    owners: HashMap<BackgroundId, Entity>,
}

#[derive(bevy::prelude::Resource, Default)]
pub(crate) struct RetainedBackgrounds(Mutex<BackgroundSurfaces>);

#[derive(bevy::prelude::Resource, Default)]
pub(crate) struct RetainedItemBounds(pub(crate) Mutex<HashMap<Entity, PhysicalRect>>);

/// Atomic render-world counters for change-driven paint extraction.
#[derive(bevy::prelude::Resource, Default)]
pub struct RetainedUiPaintCounters {
    candidates: AtomicU64,
    records_compared: AtomicU64,
    records_changed: AtomicU64,
    records_removed: AtomicU64,
    damage_events: AtomicU64,
}

impl RetainedUiPaintCounters {
    /// Returns accumulated work without resetting the counters.
    pub fn snapshot(&self) -> WorkCounters {
        WorkCounters {
            candidates: self.candidates.load(Ordering::Relaxed),
            records_compared: self.records_compared.load(Ordering::Relaxed),
            records_changed: self.records_changed.load(Ordering::Relaxed),
            records_removed: self.records_removed.load(Ordering::Relaxed),
            damage_events: self.damage_events.load(Ordering::Relaxed),
        }
    }

    fn add(&self, work: WorkCounters) {
        self.candidates
            .fetch_add(work.candidates, Ordering::Relaxed);
        self.records_compared
            .fetch_add(work.records_compared, Ordering::Relaxed);
        self.records_changed
            .fetch_add(work.records_changed, Ordering::Relaxed);
        self.records_removed
            .fetch_add(work.records_removed, Ordering::Relaxed);
        self.damage_events
            .fetch_add(work.damage_events, Ordering::Relaxed);
    }
}

impl RetainedBackgrounds {
    pub(crate) fn repair_plan(&self, camera: Entity) -> Option<RepairPlan> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .paint
            .get(&camera)
            .and_then(RetainedPaint::repair_plan)
    }

    pub(crate) fn acknowledge(&self, camera: Entity, plan: &RepairPlan) {
        let mut surfaces = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(paint) = surfaces.paint.get_mut(&camera) else {
            return;
        };
        paint.acknowledge(plan);
    }

    pub(crate) fn has_visible_records(&self, camera: Entity, target: PhysicalRect) -> bool {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .paint
            .get(&camera)
            .is_some_and(|paint| {
                paint.iter().any(|(_, record)| {
                    record
                        .coverage
                        .is_some_and(|coverage| coverage.intersection(target).is_some())
                })
            })
    }

    pub(crate) fn invalidate(&self, camera: Entity, coverage: PhysicalRect) {
        let mut surfaces = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(paint) = surfaces.paint.get_mut(&camera) {
            paint.invalidate(coverage);
        }
    }
}

pub(crate) fn cleanup_retained_backgrounds(
    mut commands: Commands,
    state: Res<RetainedBackgrounds>,
    live_render_entities: Query<Entity, With<MainEntity>>,
) {
    let live: HashSet<_> = live_render_entities.iter().collect();
    let mut surfaces = state.0.lock().unwrap_or_else(PoisonError::into_inner);
    let removed_cameras: Vec<_> = surfaces
        .paint
        .keys()
        .copied()
        .filter(|camera| !live.contains(camera))
        .collect();
    for camera in removed_cameras {
        if let Some(paint) = surfaces.paint.remove(&camera) {
            for (_, record) in paint.iter() {
                if let Ok(mut entity) = commands.get_entity(record.value.render_entity) {
                    entity.despawn();
                }
            }
        }
        surfaces.owners.retain(|_, owner| *owner != camera);
    }
}

fn coverage(size: Vec2, transform: Affine2, clip: Option<Rect>) -> Option<PhysicalRect> {
    let half_size = size * 0.5;
    let corners = [
        Vec2::new(-half_size.x, -half_size.y),
        Vec2::new(half_size.x, -half_size.y),
        Vec2::new(half_size.x, half_size.y),
        Vec2::new(-half_size.x, half_size.y),
    ]
    .map(|corner| transform.transform_point2(corner));
    let mut min = corners[0];
    let mut max = corners[0];
    for corner in &corners[1..] {
        min = min.min(*corner);
        max = max.max(*corner);
    }
    if let Some(clip) = clip {
        min = min.max(clip.min);
        max = max.min(clip.max);
    }
    PhysicalRect::from_min_max(
        min.x.floor() as i32,
        min.y.floor() as i32,
        max.x.ceil() as i32,
        max.y.ceil() as i32,
    )
}

fn remove_record(surfaces: &mut BackgroundSurfaces, commands: &mut Commands, id: BackgroundId) {
    let Some(camera) = surfaces.owners.remove(&id) else {
        return;
    };
    let paint = surfaces.paint.get_mut(&camera).unwrap();
    let retained_entity = paint.get(&id).map(|record| record.value.render_entity);
    paint.remove(&id);
    if let Some(render_entity) = retained_entity
        && let Ok(mut entity) = commands.get_entity(render_entity)
    {
        entity.despawn();
    }
}

fn upsert_record(
    surfaces: &mut BackgroundSurfaces,
    commands: &mut Commands,
    id: BackgroundId,
    camera: Entity,
    build: impl FnOnce(Entity) -> PaintRecord<BackgroundRecord>,
) {
    let retained_entity = surfaces
        .owners
        .get(&id)
        .and_then(|old_camera| surfaces.paint.get(old_camera))
        .and_then(|paint| paint.get(&id))
        .map(|record| record.value.render_entity)
        .unwrap_or_else(|| commands.spawn_empty().id());

    if let Some(old_camera) = surfaces.owners.insert(id, camera)
        && old_camera != camera
    {
        surfaces.paint.get_mut(&old_camera).unwrap().remove(&id);
    }
    surfaces
        .paint
        .entry(camera)
        .or_default()
        .upsert(id, build(retained_entity));
}

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

#[expect(
    clippy::too_many_arguments,
    reason = "each removed input is an exact invalidation source"
)]
pub(crate) fn extract_retained_backgrounds(
    mut commands: Commands,
    state: Res<RetainedBackgrounds>,
    counters: Res<RetainedUiPaintCounters>,
    bounds: Res<RetainedItemBounds>,
    mut extracted: ResMut<ExtractedUiNodes>,
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
    let mut surfaces = state.0.lock().unwrap_or_else(PoisonError::into_inner);
    let mut bounds = bounds.0.lock().unwrap_or_else(PoisonError::into_inner);
    bounds.clear();

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
        remove_record(
            &mut surfaces,
            &mut commands,
            BackgroundId {
                entity,
                kind: BackgroundKind::Fill,
            },
        );
        remove_record(
            &mut surfaces,
            &mut commands,
            BackgroundId {
                entity,
                kind: BackgroundKind::Outer,
            },
        );
    }

    for entity in outer.read() {
        remove_record(
            &mut surfaces,
            &mut commands,
            BackgroundId {
                entity,
                kind: BackgroundKind::Outer,
            },
        );
    }

    let mut camera_mapper = camera_map.get_mapper();
    let rebuilt_after_removal = clip
        .read()
        .filter(|entity| !changed.contains(*entity))
        .filter_map(|entity| all.get(entity).ok());
    for (entity, node, stack, transform, visibility, clip, target_camera, background, outer) in
        changed.iter().chain(rebuilt_after_removal)
    {
        let fill_id = BackgroundId {
            entity,
            kind: BackgroundKind::Fill,
        };
        let outer_id = BackgroundId {
            entity,
            kind: BackgroundKind::Outer,
        };
        let Some(camera) = camera_mapper.map(target_camera) else {
            remove_record(&mut surfaces, &mut commands, fill_id);
            remove_record(&mut surfaces, &mut commands, outer_id);
            continue;
        };
        let transform = transform.affine();
        let clip = clip.map(|clip| clip.clip);
        let visible = visibility.get() && !node.is_empty();
        let z_order = stack.0 as f32 + stack_z_offsets::BACKGROUND_COLOR;
        let base = BackgroundDraw {
            render_entity: Entity::PLACEHOLDER,
            camera,
            main_entity: entity.into(),
            z_order,
            clip,
            transform,
            color: background.0.into(),
            size: node.size,
            border: node.border(),
            border_radius: node.border_radius(),
            node_type: NodeType::Rect,
        };
        upsert_record(
            &mut surfaces,
            &mut commands,
            fill_id,
            camera,
            |render_entity| {
                let draw = BackgroundDraw {
                    render_entity,
                    ..base
                };
                PaintRecord {
                    coverage: (visible && !background.is_fully_transparent())
                        .then(|| coverage(node.size, transform, clip))
                        .flatten(),
                    value: BackgroundRecord::new(
                        draw,
                        visible && !background.is_fully_transparent(),
                    ),
                }
            },
        );

        if let Some(outer) = outer {
            upsert_record(
                &mut surfaces,
                &mut commands,
                outer_id,
                camera,
                |render_entity| {
                    let draw = BackgroundDraw {
                        render_entity,
                        color: outer.0.into(),
                        border: BorderRect::ZERO,
                        node_type: NodeType::Inverted,
                        ..base
                    };
                    PaintRecord {
                        coverage: (visible && !outer.is_fully_transparent())
                            .then(|| coverage(node.size, transform, clip))
                            .flatten(),
                        value: BackgroundRecord::new(
                            draw,
                            visible && !outer.is_fully_transparent(),
                        ),
                    }
                },
            );
        } else {
            remove_record(&mut surfaces, &mut commands, outer_id);
        }
    }

    for paint in surfaces.paint.values_mut() {
        counters.add(paint.take_counters());
    }

    for paint in surfaces.paint.values() {
        if paint.repair_plan().is_none() {
            continue;
        }
        let mut records: Vec<_> = paint
            .iter()
            .filter_map(|(id, record)| {
                record
                    .value
                    .draw
                    .zip(record.coverage)
                    .map(|(draw, coverage)| (*id, draw, coverage))
            })
            .collect();
        records.sort_by(|(left_id, left, _), (right_id, right, _)| {
            left.z_order
                .total_cmp(&right.z_order)
                .then_with(|| left_id.kind.cmp(&right_id.kind))
                .then_with(|| left_id.entity.cmp(&right_id.entity))
        });
        for (_, draw, coverage) in records {
            bounds.insert(draw.render_entity, coverage);
            extracted.uinodes.push(ExtractedUiNode {
                render_entity: draw.render_entity,
                z_order: draw.z_order,
                clip: draw.clip,
                image: bevy::asset::AssetId::<Image>::default(),
                extracted_camera_entity: draw.camera,
                transform: draw.transform,
                item: ExtractedUiItem::Node {
                    color: draw.color,
                    rect: Rect {
                        min: Vec2::ZERO,
                        max: draw.size,
                    },
                    atlas_scaling: None,
                    flip_x: false,
                    flip_y: false,
                    border: draw.border,
                    border_radius: draw.border_radius,
                    node_type: draw.node_type,
                },
                main_entity: draw.main_entity,
            });
        }
    }
}
