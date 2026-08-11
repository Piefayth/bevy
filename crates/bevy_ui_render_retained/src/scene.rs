//! Retained UI paint records shared by every node family.

use crate::{
    FloatBits, PaintCoverage, PaintRecord, PhysicalRect, RepairPlan, RetainedPaint, WorkCounters,
};
use bevy::{
    asset::AssetId,
    color::ColorToComponents,
    ecs::{
        entity::Entity,
        query::With,
        system::{Commands, Query, Res, ResMut},
    },
    image::Image,
    math::{Affine2, Rect, Vec2},
    render::sync_world::MainEntity,
    sprite::BorderRect,
    ui::ResolvedBorderRadius,
    ui_render::{ExtractedUiItem, ExtractedUiNode, ExtractedUiNodes, NodeType},
};
use core::sync::atomic::{AtomicU64, Ordering};
use std::{
    collections::{HashMap, HashSet},
    sync::{Mutex, MutexGuard, PoisonError},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum PaintFamily {
    Background,
    Border,
    Image,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct PaintId {
    pub(crate) entity: Entity,
    pub(crate) family: PaintFamily,
    pub(crate) ordinal: u32,
}

#[derive(Clone, Copy)]
pub(crate) struct RetainedNodeDraw {
    pub(crate) render_entity: Entity,
    pub(crate) camera: Entity,
    pub(crate) main_entity: MainEntity,
    pub(crate) z_order: f32,
    pub(crate) clip: Option<Rect>,
    pub(crate) image: AssetId<Image>,
    pub(crate) resource_generation: u64,
    pub(crate) transform: Affine2,
    pub(crate) color: bevy::color::LinearRgba,
    pub(crate) rect: Rect,
    pub(crate) atlas_scaling: Option<Vec2>,
    pub(crate) flip_x: bool,
    pub(crate) flip_y: bool,
    pub(crate) border: BorderRect,
    pub(crate) border_radius: ResolvedBorderRadius,
    pub(crate) node_type: NodeType,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum NodeTypeFingerprint {
    Rect,
    Inverted,
    Border(u32),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RetainedNodeMergeFingerprint {
    camera: Entity,
    z_order: FloatBits,
    clip: Option<[FloatBits; 4]>,
    image: AssetId<Image>,
    resource_generation: u64,
    transform: [FloatBits; 6],
    color: [FloatBits; 4],
    rect: [FloatBits; 4],
    atlas_scaling: Option<[FloatBits; 2]>,
    flip_x: bool,
    flip_y: bool,
    border: [FloatBits; 4],
    border_radius: [FloatBits; 4],
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RetainedNodeFingerprint {
    merge: RetainedNodeMergeFingerprint,
    node_type: NodeTypeFingerprint,
}

#[derive(Clone)]
struct RetainedNodeRecord {
    fingerprint: RetainedNodeFingerprint,
    render_entity: Entity,
    draw: RetainedNodeDraw,
}

impl PartialEq for RetainedNodeRecord {
    fn eq(&self, other: &Self) -> bool {
        self.fingerprint == other.fingerprint
    }
}

impl RetainedNodeRecord {
    fn new(draw: RetainedNodeDraw) -> Self {
        let clip = draw
            .clip
            .map(|clip| [clip.min.x, clip.min.y, clip.max.x, clip.max.y].map(FloatBits::new));
        let rect = [
            draw.rect.min.x,
            draw.rect.min.y,
            draw.rect.max.x,
            draw.rect.max.y,
        ]
        .map(FloatBits::new);
        let border = [
            draw.border.min_inset.x,
            draw.border.min_inset.y,
            draw.border.max_inset.x,
            draw.border.max_inset.y,
        ]
        .map(FloatBits::new);
        let border_radius: [f32; 4] = draw.border_radius.into();
        let node_type = match draw.node_type {
            NodeType::Rect => NodeTypeFingerprint::Rect,
            NodeType::Inverted => NodeTypeFingerprint::Inverted,
            NodeType::Border(flags) => NodeTypeFingerprint::Border(flags),
        };
        Self {
            fingerprint: RetainedNodeFingerprint {
                merge: RetainedNodeMergeFingerprint {
                    camera: draw.camera,
                    z_order: FloatBits::new(draw.z_order),
                    clip,
                    image: draw.image,
                    resource_generation: draw.resource_generation,
                    transform: draw.transform.to_cols_array().map(FloatBits::new),
                    color: draw.color.to_f32_array().map(FloatBits::new),
                    rect,
                    atlas_scaling: draw
                        .atlas_scaling
                        .map(|scaling| scaling.to_array().map(FloatBits::new)),
                    flip_x: draw.flip_x,
                    flip_y: draw.flip_y,
                    border,
                    border_radius: border_radius.map(FloatBits::new),
                },
                node_type,
            },
            render_entity: draw.render_entity,
            draw,
        }
    }
}

#[derive(Default)]
pub(crate) struct RetainedUiSurfaces {
    paint: HashMap<Entity, RetainedPaint<PaintId, RetainedNodeRecord>>,
    owners: HashMap<PaintId, Entity>,
}

impl RetainedUiSurfaces {
    pub(crate) fn remove(&mut self, commands: &mut Commands, id: PaintId) {
        let Some(camera) = self.owners.remove(&id) else {
            return;
        };
        let paint = self.paint.get_mut(&camera).unwrap();
        let retained_entity = paint.get(&id).map(|record| record.value.render_entity);
        paint.remove(&id);
        if let Some(render_entity) = retained_entity
            && let Ok(mut entity) = commands.get_entity(render_entity)
        {
            entity.despawn();
        }
    }

    pub(crate) fn upsert(
        &mut self,
        commands: &mut Commands,
        id: PaintId,
        camera: Entity,
        mut draw: RetainedNodeDraw,
        coverage: PaintCoverage,
    ) {
        if coverage.is_empty() {
            self.remove(commands, id);
            return;
        }
        let render_entity = self
            .owners
            .get(&id)
            .and_then(|old_camera| self.paint.get(old_camera))
            .and_then(|paint| paint.get(&id))
            .map(|record| record.value.render_entity)
            .unwrap_or_else(|| commands.spawn_empty().id());
        draw.render_entity = render_entity;

        if let Some(old_camera) = self.owners.insert(id, camera)
            && old_camera != camera
        {
            self.paint.get_mut(&old_camera).unwrap().remove(&id);
        }
        self.paint.entry(camera).or_default().upsert(
            id,
            PaintRecord {
                coverage,
                value: RetainedNodeRecord::new(draw),
            },
        );
    }
}

pub(crate) fn coverage(size: Vec2, transform: Affine2, clip: Option<Rect>) -> Option<PhysicalRect> {
    coverage_rect(Rect::from_center_size(Vec2::ZERO, size), transform, clip)
}

pub(crate) fn coverage_rect(
    rect: Rect,
    transform: Affine2,
    clip: Option<Rect>,
) -> Option<PhysicalRect> {
    let corners = [
        rect.min,
        Vec2::new(rect.max.x, rect.min.y),
        rect.max,
        Vec2::new(rect.min.x, rect.max.y),
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

#[derive(bevy::prelude::Resource, Default)]
pub(crate) struct RetainedUiScene(Mutex<RetainedUiSurfaces>);

impl RetainedUiScene {
    pub(crate) fn lock(&self) -> MutexGuard<'_, RetainedUiSurfaces> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub(crate) fn repair_plan(&self, camera: Entity) -> Option<RepairPlan> {
        self.lock()
            .paint
            .get(&camera)
            .and_then(RetainedPaint::repair_plan)
    }

    pub(crate) fn acknowledge(&self, camera: Entity, plan: &RepairPlan) {
        let mut surfaces = self.lock();
        let Some(paint) = surfaces.paint.get_mut(&camera) else {
            return;
        };
        paint.acknowledge(plan);
    }

    pub(crate) fn has_visible_records(&self, camera: Entity, target: PhysicalRect) -> bool {
        self.lock().paint.get(&camera).is_some_and(|paint| {
            paint.iter().any(|(_, record)| {
                record
                    .coverage
                    .iter()
                    .any(|coverage| coverage.intersection(target).is_some())
            })
        })
    }

    pub(crate) fn invalidate(&self, camera: Entity, coverage: PhysicalRect) {
        if let Some(paint) = self.lock().paint.get_mut(&camera) {
            paint.invalidate(coverage);
        }
    }
}

#[derive(bevy::prelude::Resource, Default)]
pub(crate) struct RetainedItems(pub(crate) Mutex<HashMap<Entity, RetainedItem>>);

#[derive(Clone)]
pub(crate) struct RetainedItem {
    pub(crate) coverage: PaintCoverage,
    pub(crate) image: AssetId<Image>,
}

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

pub(crate) fn cleanup_retained_ui(
    mut commands: Commands,
    state: Res<RetainedUiScene>,
    live_render_entities: Query<Entity, With<MainEntity>>,
) {
    let live: HashSet<_> = live_render_entities.iter().collect();
    let mut surfaces = state.lock();
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

pub(crate) fn replay_retained_ui(
    state: Res<RetainedUiScene>,
    counters: Res<RetainedUiPaintCounters>,
    items: Res<RetainedItems>,
    mut extracted: ResMut<ExtractedUiNodes>,
) {
    let mut surfaces = state.lock();
    let mut items = items.0.lock().unwrap_or_else(PoisonError::into_inner);
    items.clear();

    for paint in surfaces.paint.values_mut() {
        counters.add(paint.take_counters());
    }

    for paint in surfaces.paint.values() {
        if paint.repair_plan().is_none() {
            continue;
        }
        let mut records: Vec<_> = paint
            .iter()
            .map(|(id, record)| {
                (
                    *id,
                    record.value.draw,
                    record.coverage.clone(),
                    record.value.fingerprint.merge.clone(),
                )
            })
            .collect();
        records.sort_by(|(left_id, left, _, _), (right_id, right, _, _)| {
            left.z_order
                .total_cmp(&right.z_order)
                .then_with(|| left_id.family.cmp(&right_id.family))
                .then_with(|| left_id.entity.cmp(&right_id.entity))
                .then_with(|| left_id.ordinal.cmp(&right_id.ordinal))
        });
        let mut grouped: Vec<(
            PaintId,
            RetainedNodeDraw,
            PaintCoverage,
            RetainedNodeMergeFingerprint,
        )> = Vec::new();
        for (id, draw, coverage, merge) in records {
            if id.family == PaintFamily::Border
                && let NodeType::Border(flags) = draw.node_type
                && let Some((last_id, last_draw, last_coverage, last_merge)) = grouped.last_mut()
                && last_id.entity == id.entity
                && *last_merge == merge
                && let NodeType::Border(last_flags) = &mut last_draw.node_type
            {
                *last_flags |= flags;
                last_coverage.extend(coverage.iter().copied());
                continue;
            }
            grouped.push((id, draw, coverage, merge));
        }
        for (_, draw, coverage, _) in grouped {
            items.insert(
                draw.render_entity,
                RetainedItem {
                    coverage,
                    image: draw.image,
                },
            );
            extracted.uinodes.push(ExtractedUiNode {
                render_entity: draw.render_entity,
                z_order: draw.z_order,
                clip: draw.clip,
                image: draw.image,
                extracted_camera_entity: draw.camera,
                transform: draw.transform,
                item: ExtractedUiItem::Node {
                    color: draw.color,
                    rect: draw.rect,
                    atlas_scaling: draw.atlas_scaling,
                    flip_x: draw.flip_x,
                    flip_y: draw.flip_y,
                    border: draw.border,
                    border_radius: draw.border_radius,
                    node_type: draw.node_type,
                },
                main_entity: draw.main_entity,
            });
        }
    }
}
