//! Retained UI paint records shared by every paint family.

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
    sprite::{BorderRect, SliceScaleMode, SpriteImageMode},
    ui::ResolvedBorderRadius,
    ui_render::{
        ui_texture_slice_pipeline::{ExtractedUiTextureSlice, ExtractedUiTextureSlices},
        ExtractedGlyph, ExtractedUiItem, ExtractedUiNode, ExtractedUiNodes, NodeType,
    },
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
    TextBackground,
    TextShadow,
    TextShadowDecoration,
    TextSelection,
    Text,
    TextDecoration,
    TextPreedit,
    TextCursor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct PaintId {
    pub(crate) entity: Entity,
    pub(crate) family: PaintFamily,
    pub(crate) ordinal: u32,
}

#[derive(Clone)]
pub(crate) struct RetainedDraw {
    pub(crate) render_entity: Entity,
    pub(crate) camera: Entity,
    pub(crate) main_entity: MainEntity,
    pub(crate) z_order: f32,
    pub(crate) paint_order: u32,
    pub(crate) clip: Option<Rect>,
    pub(crate) image: AssetId<Image>,
    pub(crate) transform: Affine2,
    pub(crate) item: RetainedDrawItem,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ResourceFingerprint {
    None,
    Revisions(Box<[u64]>),
}

#[derive(Clone)]
pub(crate) enum RetainedDrawItem {
    Node(RetainedNodeItem),
    Glyphs(Box<[RetainedGlyph]>),
    TextureSlice(RetainedTextureSliceItem),
}

#[derive(Clone, Copy)]
pub(crate) struct RetainedNodeItem {
    pub(crate) color: bevy::color::LinearRgba,
    pub(crate) rect: Rect,
    pub(crate) atlas_scaling: Option<Vec2>,
    pub(crate) flip_x: bool,
    pub(crate) flip_y: bool,
    pub(crate) border: BorderRect,
    pub(crate) border_radius: ResolvedBorderRadius,
    pub(crate) node_type: NodeType,
}

#[derive(Clone)]
pub(crate) struct RetainedTextureSliceItem {
    pub(crate) stack_index: u32,
    pub(crate) rect: Rect,
    pub(crate) atlas_rect: Option<Rect>,
    pub(crate) color: bevy::color::LinearRgba,
    pub(crate) image_scale_mode: SpriteImageMode,
    pub(crate) flip_x: bool,
    pub(crate) flip_y: bool,
    pub(crate) inverse_scale_factor: f32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct RetainedGlyph {
    color: [FloatBits; 4],
    translation: [FloatBits; 2],
    rect: [FloatBits; 4],
}

impl RetainedGlyph {
    pub(crate) fn new(color: bevy::color::LinearRgba, translation: Vec2, rect: Rect) -> Self {
        Self {
            color: color.to_f32_array().map(FloatBits::new),
            translation: translation.to_array().map(FloatBits::new),
            rect: [rect.min.x, rect.min.y, rect.max.x, rect.max.y].map(FloatBits::new),
        }
    }

    fn color(self) -> bevy::color::LinearRgba {
        bevy::color::LinearRgba::new(
            self.color[0].get(),
            self.color[1].get(),
            self.color[2].get(),
            self.color[3].get(),
        )
    }

    pub(crate) fn translation(self) -> Vec2 {
        Vec2::new(self.translation[0].get(), self.translation[1].get())
    }

    pub(crate) fn rect(self) -> Rect {
        Rect::from_corners(
            Vec2::new(self.rect[0].get(), self.rect[1].get()),
            Vec2::new(self.rect[2].get(), self.rect[3].get()),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum NodeTypeFingerprint {
    Rect,
    Inverted,
    Border(u32),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RetainedCommonFingerprint {
    camera: Entity,
    z_order: FloatBits,
    paint_order: u32,
    clip: Option<[FloatBits; 4]>,
    image: AssetId<Image>,
    resource: ResourceFingerprint,
    transform: [FloatBits; 6],
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RetainedNodeMergeFingerprint {
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

#[derive(Clone, Debug, PartialEq, Eq)]
enum RetainedItemFingerprint {
    Node(RetainedNodeFingerprint),
    Glyphs,
    TextureSlice(RetainedTextureSliceFingerprint),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RetainedTextureSliceFingerprint {
    rect: [FloatBits; 4],
    atlas_rect: Option<[FloatBits; 4]>,
    color: [FloatBits; 4],
    image_scale_mode: TextureSliceModeFingerprint,
    flip_x: bool,
    flip_y: bool,
    inverse_scale_factor: FloatBits,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum TextureSliceModeFingerprint {
    Sliced {
        border: [FloatBits; 4],
        center: SliceScaleModeFingerprint,
        sides: SliceScaleModeFingerprint,
        max_corner_scale: FloatBits,
    },
    Tiled {
        tile_x: bool,
        tile_y: bool,
        stretch_value: FloatBits,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum SliceScaleModeFingerprint {
    Stretch,
    Tile { stretch_value: FloatBits },
}

struct RetainedRecord {
    common: RetainedCommonFingerprint,
    item: RetainedItemFingerprint,
    render_entity: Entity,
    draw: RetainedDraw,
}

impl PartialEq for RetainedRecord {
    fn eq(&self, other: &Self) -> bool {
        self.common == other.common
            && self.item == other.item
            && match (&self.draw.item, &other.draw.item) {
                (RetainedDrawItem::Glyphs(left), RetainedDrawItem::Glyphs(right)) => left == right,
                _ => true,
            }
    }
}

impl RetainedRecord {
    fn new(draw: RetainedDraw, resource: ResourceFingerprint) -> Self {
        let clip = draw
            .clip
            .map(|clip| [clip.min.x, clip.min.y, clip.max.x, clip.max.y].map(FloatBits::new));
        let item = match &draw.item {
            RetainedDrawItem::Node(item) => {
                let rect = [
                    item.rect.min.x,
                    item.rect.min.y,
                    item.rect.max.x,
                    item.rect.max.y,
                ]
                .map(FloatBits::new);
                let border = [
                    item.border.min_inset.x,
                    item.border.min_inset.y,
                    item.border.max_inset.x,
                    item.border.max_inset.y,
                ]
                .map(FloatBits::new);
                let border_radius: [f32; 4] = item.border_radius.into();
                let node_type = match item.node_type {
                    NodeType::Rect => NodeTypeFingerprint::Rect,
                    NodeType::Inverted => NodeTypeFingerprint::Inverted,
                    NodeType::Border(flags) => NodeTypeFingerprint::Border(flags),
                };
                RetainedItemFingerprint::Node(RetainedNodeFingerprint {
                    merge: RetainedNodeMergeFingerprint {
                        color: item.color.to_f32_array().map(FloatBits::new),
                        rect,
                        atlas_scaling: item
                            .atlas_scaling
                            .map(|scaling| scaling.to_array().map(FloatBits::new)),
                        flip_x: item.flip_x,
                        flip_y: item.flip_y,
                        border,
                        border_radius: border_radius.map(FloatBits::new),
                    },
                    node_type,
                })
            }
            RetainedDrawItem::Glyphs(_) => RetainedItemFingerprint::Glyphs,
            RetainedDrawItem::TextureSlice(item) => {
                RetainedItemFingerprint::TextureSlice(RetainedTextureSliceFingerprint {
                    rect: rect_fingerprint(item.rect),
                    atlas_rect: item.atlas_rect.map(rect_fingerprint),
                    color: item.color.to_f32_array().map(FloatBits::new),
                    image_scale_mode: texture_slice_mode_fingerprint(&item.image_scale_mode),
                    flip_x: item.flip_x,
                    flip_y: item.flip_y,
                    inverse_scale_factor: FloatBits::new(item.inverse_scale_factor),
                })
            }
        };
        Self {
            common: RetainedCommonFingerprint {
                camera: draw.camera,
                z_order: FloatBits::new(draw.z_order),
                paint_order: draw.paint_order,
                clip,
                image: draw.image,
                resource,
                transform: draw.transform.to_cols_array().map(FloatBits::new),
            },
            item,
            render_entity: draw.render_entity,
            draw,
        }
    }

    fn border_parts(
        &self,
    ) -> Option<(
        &RetainedCommonFingerprint,
        &RetainedNodeMergeFingerprint,
        u32,
    )> {
        let RetainedItemFingerprint::Node(node) = &self.item else {
            return None;
        };
        let NodeTypeFingerprint::Border(flags) = node.node_type else {
            return None;
        };
        Some((&self.common, &node.merge, flags))
    }
}

fn rect_fingerprint(rect: Rect) -> [FloatBits; 4] {
    [rect.min.x, rect.min.y, rect.max.x, rect.max.y].map(FloatBits::new)
}

fn texture_slice_mode_fingerprint(mode: &SpriteImageMode) -> TextureSliceModeFingerprint {
    match mode {
        SpriteImageMode::Sliced(slicer) => TextureSliceModeFingerprint::Sliced {
            border: [
                slicer.border.min_inset.x,
                slicer.border.min_inset.y,
                slicer.border.max_inset.x,
                slicer.border.max_inset.y,
            ]
            .map(FloatBits::new),
            center: slice_scale_mode_fingerprint(slicer.center_scale_mode),
            sides: slice_scale_mode_fingerprint(slicer.sides_scale_mode),
            max_corner_scale: FloatBits::new(slicer.max_corner_scale),
        },
        SpriteImageMode::Tiled {
            tile_x,
            tile_y,
            stretch_value,
        } => TextureSliceModeFingerprint::Tiled {
            tile_x: *tile_x,
            tile_y: *tile_y,
            stretch_value: FloatBits::new(*stretch_value),
        },
        SpriteImageMode::Auto | SpriteImageMode::Scale(_) => {
            unreachable!("retained texture slices require sliced or tiled image mode")
        }
    }
}

fn slice_scale_mode_fingerprint(mode: SliceScaleMode) -> SliceScaleModeFingerprint {
    match mode {
        SliceScaleMode::Stretch => SliceScaleModeFingerprint::Stretch,
        SliceScaleMode::Tile { stretch_value } => SliceScaleModeFingerprint::Tile {
            stretch_value: FloatBits::new(stretch_value),
        },
    }
}

#[derive(Default)]
pub(crate) struct RetainedUiSurfaces {
    paint: HashMap<Entity, RetainedPaint<PaintId, RetainedRecord>>,
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
        mut draw: RetainedDraw,
        resource: ResourceFingerprint,
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
                value: RetainedRecord::new(draw, resource),
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
    mut extracted_slices: ResMut<ExtractedUiTextureSlices>,
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
        let mut records: Vec<_> = paint.iter().collect();
        records.sort_by(|(left_id, left), (right_id, right)| {
            left.value
                .draw
                .z_order
                .total_cmp(&right.value.draw.z_order)
                .then_with(|| left_id.family.cmp(&right_id.family))
                .then_with(|| {
                    left.value
                        .draw
                        .paint_order
                        .cmp(&right.value.draw.paint_order)
                })
                .then_with(|| left_id.entity.cmp(&right_id.entity))
                .then_with(|| left_id.ordinal.cmp(&right_id.ordinal))
        });
        let mut index = 0;
        while index < records.len() {
            let (id, record) = records[index];
            if id.family == PaintFamily::Border
                && let Some((common, node, mut flags)) = record.value.border_parts()
            {
                let mut coverage = record.coverage.clone();
                index += 1;
                while let Some((next_id, next)) = records.get(index).copied()
                    && next_id.family == PaintFamily::Border
                    && next_id.entity == id.entity
                    && let Some((next_common, next_node, next_flags)) = next.value.border_parts()
                    && next_common == common
                    && next_node == node
                {
                    flags |= next_flags;
                    coverage.extend(next.coverage.iter().copied());
                    index += 1;
                }
                push_replayed(
                    &mut extracted,
                    &mut extracted_slices,
                    &mut items,
                    &record.value.draw,
                    coverage,
                    Some(NodeType::Border(flags)),
                );
                continue;
            }

            push_replayed(
                &mut extracted,
                &mut extracted_slices,
                &mut items,
                &record.value.draw,
                record.coverage.clone(),
                None,
            );
            index += 1;
        }
    }
}

fn push_replayed(
    extracted: &mut ExtractedUiNodes,
    extracted_slices: &mut ExtractedUiTextureSlices,
    items: &mut HashMap<Entity, RetainedItem>,
    draw: &RetainedDraw,
    coverage: PaintCoverage,
    node_type: Option<NodeType>,
) {
    let item = match &draw.item {
        RetainedDrawItem::Node(item) => ExtractedUiItem::Node {
            color: item.color,
            rect: item.rect,
            atlas_scaling: item.atlas_scaling,
            flip_x: item.flip_x,
            flip_y: item.flip_y,
            border: item.border,
            border_radius: item.border_radius,
            node_type: node_type.unwrap_or(item.node_type),
        },
        RetainedDrawItem::Glyphs(glyphs) => {
            let start = extracted.glyphs.len();
            extracted
                .glyphs
                .extend(glyphs.iter().map(|glyph| ExtractedGlyph {
                    color: glyph.color(),
                    translation: glyph.translation(),
                    rect: glyph.rect(),
                }));
            ExtractedUiItem::Glyphs {
                range: start..extracted.glyphs.len(),
            }
        }
        RetainedDrawItem::TextureSlice(item) => {
            extracted_slices.slices.push(ExtractedUiTextureSlice {
                stack_index: item.stack_index,
                transform: draw.transform,
                rect: item.rect,
                atlas_rect: item.atlas_rect,
                image: draw.image,
                clip: draw.clip,
                extracted_camera_entity: draw.camera,
                color: item.color,
                image_scale_mode: item.image_scale_mode.clone(),
                flip_x: item.flip_x,
                flip_y: item.flip_y,
                inverse_scale_factor: item.inverse_scale_factor,
                main_entity: draw.main_entity,
                render_entity: draw.render_entity,
            });
            items.insert(
                draw.render_entity,
                RetainedItem {
                    coverage,
                    image: draw.image,
                },
            );
            return;
        }
    };
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
        item,
        main_entity: draw.main_entity,
    });
}
