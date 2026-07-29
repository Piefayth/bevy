//! Shared retained-mode allocation and batching support for UI render pipelines.

use core::ops::Range;

use bevy_ecs::entity::{EntityHashMap, EntityIndexMap};
use bevy_ecs::prelude::Entity;
use bevy_platform::collections::HashMap;
use bevy_render::{
    render_phase::ViewSortedRenderPhases,
    render_resource::{CachedRenderPipelineId, RawBufferVec},
    renderer::RenderDevice,
    sync_world::{MainEntity, MainEntityHashMap, MainEntityHashSet},
};

use crate::TransparentUi;

pub(crate) fn vertex_storage_supported(render_device: &RenderDevice, buffer_count: u32) -> bool {
    render_device.limits().max_storage_buffers_per_shader_stage >= buffer_count
}

pub(crate) const UI_ARENA_COMPACT_MIN_INSTANCES: u32 = 1 << 14;

#[derive(Clone, Copy, Debug)]
pub(crate) struct RemovedUiNode {
    pub(crate) main_entity: MainEntity,
    pub(crate) render_entity: Entity,
    pub(crate) camera_entity: Entity,
}

/// Records every render entity before an extracted owner is rebuilt or removed.
pub(crate) fn remove_owner<I>(
    owners: &mut MainEntityHashMap<(Entity, EntityIndexMap<I>)>,
    main_entity: MainEntity,
    removed: &mut Vec<RemovedUiNode>,
) -> Option<(Entity, EntityIndexMap<I>)> {
    let (camera_entity, items) = owners.remove(&main_entity)?;
    removed.extend(items.keys().copied().map(|render_entity| RemovedUiNode {
        main_entity,
        render_entity,
        camera_entity,
    }));
    Some((camera_entity, items))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ItemInstances {
    pub(crate) start: u32,
    pub(crate) count: u32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ArenaSlot {
    pub(crate) instances: ItemInstances,
    pub(crate) capacity: u32,
}

/// Stable power-of-two allocations for retained UI data.
#[derive(Default)]
pub(crate) struct UiInstanceArena {
    pub(crate) slots: EntityHashMap<ArenaSlot>,
    pub(crate) owners: MainEntityHashMap<Vec<Entity>>,
    pub(crate) pending_assets: MainEntityHashSet,
    pub(crate) free_lists: HashMap<u32, Vec<u32>>,
    pub(crate) top: u32,
    pub(crate) dead_instances: u32,
    pub(crate) initialized: bool,
}

impl UiInstanceArena {
    pub(crate) fn alloc(&mut self, count: u32) -> (u32, u32) {
        debug_assert_ne!(count, 0);
        let capacity = count
            .checked_next_power_of_two()
            .expect("one UI item exceeded the retained arena's capacity");
        let start = match self.free_lists.get_mut(&capacity).and_then(Vec::pop) {
            Some(start) => {
                self.dead_instances -= capacity;
                start
            }
            None => {
                let start = self.top;
                self.top = self
                    .top
                    .checked_add(capacity)
                    .expect("retained UI arena exceeded u32::MAX elements");
                start
            }
        };
        (start, capacity)
    }

    /// Allocates an exact-sized range for fixed-size retained records.
    pub(crate) fn alloc_exact(&mut self, count: u32) -> (u32, u32) {
        debug_assert_ne!(count, 0);
        let start = match self.free_lists.get_mut(&count).and_then(Vec::pop) {
            Some(start) => {
                self.dead_instances -= count;
                start
            }
            None => {
                let start = self.top;
                self.top = self
                    .top
                    .checked_add(count)
                    .expect("retained UI arena exceeded u32::MAX elements");
                start
            }
        };
        (start, count)
    }

    pub(crate) fn insert_empty(&mut self, render_entity: Entity) {
        self.slots.insert(
            render_entity,
            ArenaSlot {
                instances: ItemInstances::default(),
                capacity: 0,
            },
        );
    }

    pub(crate) fn insert(&mut self, render_entity: Entity, start: u32, count: u32, capacity: u32) {
        self.slots.insert(
            render_entity,
            ArenaSlot {
                instances: ItemInstances { start, count },
                capacity,
            },
        );
    }

    pub(crate) fn free(&mut self, render_entity: Entity) {
        if let Some(slot) = self.slots.remove(&render_entity)
            && slot.capacity != 0
        {
            self.free_lists
                .entry(slot.capacity)
                .or_default()
                .push(slot.instances.start);
            self.dead_instances += slot.capacity;
        }
    }

    pub(crate) fn free_owner(&mut self, main_entity: MainEntity) {
        self.pending_assets.remove(&main_entity);
        if let Some(render_entities) = self.owners.remove(&main_entity) {
            for render_entity in render_entities {
                self.free(render_entity);
            }
        }
    }

    pub(crate) fn needs_compaction(&self) -> bool {
        self.top >= UI_ARENA_COMPACT_MIN_INSTANCES
            && self.dead_instances.saturating_mul(8) > self.top
    }

    pub(crate) fn reset(&mut self) {
        self.slots.clear();
        self.owners.clear();
        self.pending_assets.clear();
        self.free_lists.clear();
        self.top = 0;
        self.dead_instances = 0;
        self.initialized = true;
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RetainedUiBatch<K> {
    pub(crate) range: Range<u32>,
    pub(crate) key: K,
}

pub(crate) enum RetainedBatchItem<K> {
    NotOwned,
    Culled,
    Drawable { instances: ItemInstances, key: K },
}

/// Builds draw batches in sorted phase order.
///
/// In indirect mode, `instance_indices` receives one retained slot index per
/// drawable instance. This allows adjacent compatible phase items to batch even
/// when their retained arena allocations aren't contiguous.
pub(crate) fn batch_retained_ui<K, F, C, M>(
    phases: &mut ViewSortedRenderPhases<TransparentUi>,
    instance_indices: &mut RawBufferVec<u32>,
    indirect: bool,
    mut classify: F,
    compatible: C,
    merge_key: M,
) -> Vec<RetainedUiBatch<K>>
where
    K: Clone,
    F: FnMut(&TransparentUi) -> RetainedBatchItem<K>,
    C: Fn(&K, &K) -> bool,
    M: Fn(&mut K, &K),
{
    instance_indices.clear();
    let mut batches: Vec<RetainedUiBatch<K>> = Vec::new();

    for phase in phases.values_mut() {
        let mut active: Option<(usize, usize)> = None;

        for item_index in 0..phase.items.len() {
            let classified = classify(&phase.items[item_index]);
            match classified {
                RetainedBatchItem::NotOwned => {
                    active = None;
                }
                RetainedBatchItem::Culled => {
                    let phase_item_index =
                        u32::try_from(item_index).expect("too many retained UI phase items");
                    let item = &mut phase.items[item_index];
                    item.batch_index = None;
                    item.batch_range = phase_item_index..phase_item_index;
                }
                RetainedBatchItem::Drawable { instances, key } => {
                    let retained_end = instances
                        .start
                        .checked_add(instances.count)
                        .expect("retained UI instance range exceeded u32::MAX");
                    let range = if indirect {
                        let start = u32::try_from(instance_indices.len())
                            .expect("retained UI indirection buffer exceeded u32::MAX elements");
                        for retained_index in instances.start..retained_end {
                            instance_indices.push(retained_index);
                        }
                        start
                            ..start
                                .checked_add(instances.count)
                                .expect("retained UI draw range exceeded u32::MAX")
                    } else {
                        instances.start..retained_end
                    };

                    let can_merge = active.is_some_and(|(batch_index, _)| {
                        let batch = &batches[batch_index];
                        compatible(&batch.key, &key) && (indirect || batch.range.end == range.start)
                    });

                    let (batch_index, first_item_index) = if can_merge {
                        let active = active.expect("active batch checked above");
                        let batch = &mut batches[active.0];
                        batch.range.end = range.end;
                        merge_key(&mut batch.key, &key);
                        active
                    } else {
                        let batch_index = batches.len();
                        batches.push(RetainedUiBatch { range, key });
                        active = Some((batch_index, item_index));
                        (batch_index, item_index)
                    };

                    let item = &mut phase.items[item_index];
                    item.batch_index = (item_index == first_item_index).then_some(
                        u32::try_from(batch_index).expect("too many retained UI batches"),
                    );
                    let phase_item_index =
                        u32::try_from(item_index).expect("too many retained UI phase items");
                    item.batch_range = phase_item_index..phase_item_index;
                    phase.items[first_item_index].batch_range.end = phase_item_index
                        .checked_add(1)
                        .expect("retained UI phase range exceeded u32::MAX");
                }
            }
        }
    }

    batches
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct UiCameraPipelineState {
    pub(crate) retained_view_entity: bevy_render::view::RetainedViewEntity,
    pub(crate) pipeline: CachedRenderPipelineId,
}

#[cfg(test)]
pub(crate) fn preprocess_wgsl_for_test(source: &str, defined: &[&str], prefix: &str) -> String {
    #[derive(Clone, Copy)]
    struct Conditional {
        parent: bool,
        branch_taken: bool,
    }

    let mut output = String::from(prefix);
    let mut conditionals = Vec::new();
    let mut include = true;
    let mut skipping_import = false;
    for line in source.lines() {
        let directive = line.trim();
        if skipping_import {
            if directive == "}" || directive == "};" {
                skipping_import = false;
            }
            continue;
        }
        if directive.starts_with("#import") {
            skipping_import = directive.ends_with('{');
            continue;
        }
        match directive {
            directive if directive.starts_with("#define_import_path") => {}
            directive if directive.starts_with("#ifdef ") => {
                let condition = defined.contains(&directive.trim_start_matches("#ifdef "));
                conditionals.push(Conditional {
                    parent: include,
                    branch_taken: condition,
                });
                include &= condition;
            }
            directive if directive.starts_with("#else ifdef ") => {
                let frame = conditionals.last_mut().expect("#else ifdef without #ifdef");
                let condition = defined.contains(&directive.trim_start_matches("#else ifdef "));
                include = frame.parent && !frame.branch_taken && condition;
                frame.branch_taken |= condition;
            }
            "#else" => {
                let frame = conditionals.last_mut().expect("#else without #ifdef");
                include = frame.parent && !frame.branch_taken;
                frame.branch_taken = true;
            }
            "#endif" => {
                include = conditionals.pop().expect("#endif without #ifdef").parent;
            }
            _ if include => {
                output.push_str(line);
                output.push('\n');
            }
            _ => {}
        }
    }
    assert!(conditionals.is_empty(), "unterminated shader conditional");
    output
}

#[cfg(test)]
pub(crate) fn validate_wgsl_for_test(label: &str, source: &str) {
    let module = naga::front::wgsl::parse_str(source).unwrap_or_else(|error| {
        panic!("{label} failed to parse:\n{}", error.emit_to_string(source))
    });
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|error| panic!("{label} failed validation: {error}"));
}

#[cfg(test)]
mod tests {
    use bevy_math::FloatOrd;
    use bevy_render::{
        render_phase::{DrawFunctionId, PhaseItemExtraIndex},
        render_resource::{BufferUsages, CachedRenderPipelineId},
        view::RetainedViewEntity,
    };

    use super::*;

    fn entity(index: u32) -> Entity {
        Entity::from_bits(index as u64)
    }

    fn item(render: u32, main: u32) -> TransparentUi {
        TransparentUi {
            sort_key: FloatOrd(render as f32),
            entity: (entity(render), MainEntity::from(entity(main))),
            pipeline: CachedRenderPipelineId::INVALID,
            draw_function: DrawFunctionId(0),
            batch_range: 0..0,
            extra_index: PhaseItemExtraIndex::None,
            indexed: false,
            batch_index: None,
        }
    }

    fn phases(
        items: impl IntoIterator<Item = TransparentUi>,
    ) -> ViewSortedRenderPhases<TransparentUi> {
        let main = MainEntity::from(entity(100));
        let retained_view = RetainedViewEntity::new(main, None, 0);
        let mut phases = ViewSortedRenderPhases::default();
        phases.prepare_for_new_frame(retained_view);
        let phase = phases.get_mut(&retained_view).unwrap();
        for item in items {
            phase.add_retained(item);
        }
        phases
    }

    #[test]
    fn indirect_batching_merges_fragmented_retained_slots_in_phase_order() {
        let mut phases = phases([item(1, 11), item(2, 12)]);
        let mut indices = RawBufferVec::new(BufferUsages::VERTEX);
        let batches = batch_retained_ui(
            &mut phases,
            &mut indices,
            true,
            |item| {
                let instances = if item.entity.0 == entity(1) {
                    ItemInstances {
                        start: 10,
                        count: 2,
                    }
                } else {
                    ItemInstances { start: 3, count: 1 }
                };
                RetainedBatchItem::Drawable {
                    instances,
                    key: 7u32,
                }
            },
            PartialEq::eq,
            |_, _| {},
        );
        assert_eq!(
            batches,
            [RetainedUiBatch {
                range: 0..3,
                key: 7
            }]
        );
        assert_eq!(indices.values(), &[10, 11, 3]);
        let phase = phases.values().next().unwrap();
        assert_eq!(phase.items[0].batch_index, Some(0));
        assert_eq!(phase.items[0].batch_range, 0..2);
        assert_eq!(phase.items[1].batch_index, None);
    }

    #[test]
    fn direct_batching_splits_noncontiguous_slots() {
        let mut phases = phases([item(1, 11), item(2, 12)]);
        let mut indices = RawBufferVec::new(BufferUsages::VERTEX);
        let batches = batch_retained_ui(
            &mut phases,
            &mut indices,
            false,
            |item| RetainedBatchItem::Drawable {
                instances: ItemInstances {
                    start: if item.entity.0 == entity(1) { 10 } else { 3 },
                    count: 1,
                },
                key: 7u32,
            },
            PartialEq::eq,
            |_, _| {},
        );
        assert_eq!(batches.len(), 2);
        assert!(indices.is_empty());
    }

    #[test]
    fn nonowned_phase_item_is_a_hard_batch_boundary() {
        let mut phases = phases([item(1, 11), item(2, 12), item(3, 13)]);
        let mut indices = RawBufferVec::new(BufferUsages::VERTEX);
        let batches = batch_retained_ui(
            &mut phases,
            &mut indices,
            true,
            |item| {
                if item.entity.0 == entity(2) {
                    RetainedBatchItem::NotOwned
                } else {
                    RetainedBatchItem::Drawable {
                        instances: ItemInstances {
                            start: if item.entity.0 == entity(1) { 1 } else { 3 },
                            count: 1,
                        },
                        key: 7u32,
                    }
                }
            },
            PartialEq::eq,
            |_, _| {},
        );
        assert_eq!(batches.len(), 2);
        let phase = phases.values().next().unwrap();
        assert_eq!(phase.items[0].batch_range, 0..1);
        assert_eq!(phase.items[2].batch_range, 2..3);
    }

    #[test]
    fn culled_item_can_be_skipped_inside_an_indirect_batch() {
        let mut phases = phases([item(1, 11), item(2, 12), item(3, 13)]);
        let mut indices = RawBufferVec::new(BufferUsages::VERTEX);
        let batches = batch_retained_ui(
            &mut phases,
            &mut indices,
            true,
            |item| {
                if item.entity.0 == entity(2) {
                    RetainedBatchItem::Culled
                } else {
                    RetainedBatchItem::Drawable {
                        instances: ItemInstances {
                            start: if item.entity.0 == entity(1) { 1 } else { 3 },
                            count: 1,
                        },
                        key: 7u32,
                    }
                }
            },
            PartialEq::eq,
            |_, _| {},
        );
        assert_eq!(
            batches,
            [RetainedUiBatch {
                range: 0..2,
                key: 7
            }]
        );
        assert_eq!(indices.values(), &[1, 3]);
        let phase = phases.values().next().unwrap();
        assert_eq!(phase.items[0].batch_range, 0..3);
        assert_eq!(phase.items[1].batch_index, None);
        assert_eq!(phase.items[2].batch_index, None);
    }

    #[test]
    fn exact_arena_allocations_are_contiguous_and_reused() {
        let mut arena = UiInstanceArena::default();
        arena.reset();
        let first = arena.alloc_exact(6);
        let second = arena.alloc_exact(6);
        assert_eq!(first, (0, 6));
        assert_eq!(second, (6, 6));

        arena.insert(entity(1), first.0, 6, first.1);
        arena.free(entity(1));
        assert_eq!(arena.dead_instances, 6);
        assert_eq!(arena.alloc_exact(6), first);
        assert_eq!(arena.dead_instances, 0);
    }

    #[test]
    fn removing_an_owner_records_every_retained_render_entity() {
        let main_entity = MainEntity::from(entity(50));
        let camera_entity = entity(60);
        let mut items = EntityIndexMap::default();
        items.insert(entity(1), 10u32);
        items.insert(entity(2), 20u32);
        let mut owners = MainEntityHashMap::default();
        owners.insert(main_entity, (camera_entity, items));
        let mut removed = Vec::new();

        let (_, removed_items) = remove_owner(&mut owners, main_entity, &mut removed).unwrap();
        assert_eq!(removed_items.len(), 2);
        assert!(!owners.contains_key(&main_entity));
        assert_eq!(removed.len(), 2);
        assert!(removed.iter().all(|record| {
            record.main_entity == main_entity && record.camera_entity == camera_entity
        }));
        assert_eq!(
            removed
                .iter()
                .map(|record| record.render_entity)
                .collect::<Vec<_>>(),
            [entity(1), entity(2)]
        );
    }

    #[test]
    fn every_shared_quad_shader_uses_the_legacy_triangle_corner_order() {
        const EXPECTED: &str = "const QUAD_CORNERS = array(
    vec2(-0.5, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5),
    vec2(-0.5, -0.5),
    vec2(0.5, -0.5),
    vec2(0.5, 0.5),
);";
        for (label, source) in [
            ("core UI", include_str!("ui.wgsl")),
            ("gradient", include_str!("gradient.wgsl")),
            ("box shadow", include_str!("box_shadow.wgsl")),
            ("texture slice", include_str!("ui_texture_slice.wgsl")),
        ] {
            assert!(
                source.contains(EXPECTED),
                "{label} shader diverged from QUAD_INDICES"
            );
        }
    }
}
