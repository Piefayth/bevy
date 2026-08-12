//! Exact physical-pixel damage tracking.

use alloc::sync::Arc;
use bevy::platform::collections::HashMap;
use core::sync::atomic::{AtomicU64, Ordering};

static NEXT_JOURNAL_ID: AtomicU64 = AtomicU64::new(1);

/// A non-empty rectangle of physical pixels with an exclusive maximum edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PhysicalRect {
    min_x: i32,
    min_y: i32,
    max_x: i32,
    max_y: i32,
}

impl PhysicalRect {
    /// Creates a rectangle, returning `None` when either axis is empty.
    pub const fn from_min_max(min_x: i32, min_y: i32, max_x: i32, max_y: i32) -> Option<Self> {
        if min_x < max_x && min_y < max_y {
            Some(Self {
                min_x,
                min_y,
                max_x,
                max_y,
            })
        } else {
            None
        }
    }

    /// Returns the minimum x coordinate.
    pub const fn min_x(&self) -> i32 {
        self.min_x
    }

    /// Returns the minimum y coordinate.
    pub const fn min_y(&self) -> i32 {
        self.min_y
    }

    /// Returns the exclusive maximum x coordinate.
    pub const fn max_x(&self) -> i32 {
        self.max_x
    }

    /// Returns the exclusive maximum y coordinate.
    pub const fn max_y(&self) -> i32 {
        self.max_y
    }

    /// Returns the number of physical pixels in this rectangle.
    pub const fn area(&self) -> u64 {
        let width = (self.max_x as i64 - self.min_x as i64) as u64;
        let height = (self.max_y as i64 - self.min_y as i64) as u64;
        width * height
    }

    /// Returns the pixels shared by both rectangles.
    pub const fn intersection(self, other: Self) -> Option<Self> {
        Self::from_min_max(
            if self.min_x > other.min_x {
                self.min_x
            } else {
                other.min_x
            },
            if self.min_y > other.min_y {
                self.min_y
            } else {
                other.min_y
            },
            if self.max_x < other.max_x {
                self.max_x
            } else {
                other.max_x
            },
            if self.max_y < other.max_y {
                self.max_y
            } else {
                other.max_y
            },
        )
    }
}

#[derive(Clone, Copy, Debug)]
struct DamageEvent {
    epoch: u64,
    rect: PhysicalRect,
}

/// An exact repair plan over all damage owed through one journal epoch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepairPlan {
    journal_id: u64,
    through_epoch: u64,
    regions: Arc<[PhysicalRect]>,
    spatial: Arc<SpatialIndex<()>>,
    damaged_pixels: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SpatialIndex<T> {
    values: Vec<T>,
    nodes: Vec<SpatialNode>,
    parents: Vec<Option<usize>>,
    entry_nodes: Vec<usize>,
    refit_marks: Vec<u32>,
    refit_generation: u32,
    root: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SpatialNode {
    bounds: PhysicalRect,
    contents: SpatialNodeContents,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SpatialNodeContents {
    Entry(usize),
    Branch([usize; 2]),
}

impl<T> SpatialIndex<T> {
    pub(crate) fn new(entries: impl IntoIterator<Item = (PhysicalRect, T)>) -> Self {
        let (regions, values): (Vec<_>, Vec<_>) = entries.into_iter().unzip();
        let mut order: Vec<_> = (0..regions.len()).collect();
        let mut nodes = Vec::with_capacity(regions.len().saturating_mul(2).saturating_sub(1));
        let mut parents = Vec::with_capacity(nodes.capacity());
        let mut entry_nodes = vec![usize::MAX; regions.len()];
        let root = Self::build(
            &regions,
            &mut order,
            0,
            &mut nodes,
            &mut parents,
            &mut entry_nodes,
        );
        let refit_marks = vec![0; nodes.len()];
        Self {
            values,
            nodes,
            parents,
            entry_nodes,
            refit_marks,
            refit_generation: 0,
            root,
        }
    }

    pub(crate) fn query_intersecting<U>(&self, other: &SpatialIndex<U>, mut visit: impl FnMut(T))
    where
        T: Copy,
    {
        if let Some(root) = self.root {
            self.query_intersecting_node(root, other, &mut visit);
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.values.len()
    }

    pub(crate) fn query(&self, region: PhysicalRect, mut visit: impl FnMut(T))
    where
        T: Copy,
    {
        if let Some(root) = self.root {
            self.query_node(root, region, &mut visit);
        }
    }

    pub(crate) fn update_many(&mut self, updates: impl IntoIterator<Item = (usize, PhysicalRect)>) {
        self.refit_generation = self.refit_generation.wrapping_add(1);
        if self.refit_generation == 0 {
            self.refit_marks.fill(0);
            self.refit_generation = 1;
        }
        let generation = self.refit_generation;
        for (entry, bounds) in updates {
            let mut node = self.entry_nodes[entry];
            self.nodes[node].bounds = bounds;
            while let Some(parent) = self.parents[node] {
                if self.refit_marks[parent] == generation {
                    break;
                }
                self.refit_marks[parent] = generation;
                node = parent;
            }
        }
        if let Some(root) = self.root {
            self.refit_node(root, generation);
        }
    }

    fn refit_node(&mut self, node: usize, generation: u32) {
        if self.refit_marks[node] != generation {
            return;
        }
        let SpatialNodeContents::Branch(children) = self.nodes[node].contents else {
            return;
        };
        self.refit_node(children[0], generation);
        self.refit_node(children[1], generation);
        self.nodes[node].bounds = enclosing_rect(
            self.nodes[children[0]].bounds,
            self.nodes[children[1]].bounds,
        );
    }

    fn build(
        regions: &[PhysicalRect],
        order: &mut [usize],
        depth: usize,
        nodes: &mut Vec<SpatialNode>,
        parents: &mut Vec<Option<usize>>,
        entry_nodes: &mut [usize],
    ) -> Option<usize> {
        if order.is_empty() {
            return None;
        }
        if order.len() == 1 {
            let entry = order[0];
            let index = nodes.len();
            nodes.push(SpatialNode {
                bounds: regions[entry],
                contents: SpatialNodeContents::Entry(entry),
            });
            parents.push(None);
            entry_nodes[entry] = index;
            return Some(index);
        }
        let middle = order.len() / 2;
        order.select_nth_unstable_by_key(middle, |&entry| {
            let rect = regions[entry];
            let x = i64::from(rect.min_x()) + i64::from(rect.max_x());
            let y = i64::from(rect.min_y()) + i64::from(rect.max_y());
            if depth.is_multiple_of(2) {
                (x, y)
            } else {
                (y, x)
            }
        });
        let (left, right) = order.split_at_mut(middle);
        let left = Self::build(regions, left, depth + 1, nodes, parents, entry_nodes).unwrap();
        let right = Self::build(regions, right, depth + 1, nodes, parents, entry_nodes).unwrap();
        let left_bounds = nodes[left].bounds;
        let right_bounds = nodes[right].bounds;
        let bounds = enclosing_rect(left_bounds, right_bounds);
        let index = nodes.len();
        nodes.push(SpatialNode {
            bounds,
            contents: SpatialNodeContents::Branch([left, right]),
        });
        parents.push(None);
        parents[left] = Some(index);
        parents[right] = Some(index);
        Some(index)
    }

    fn query_intersecting_node<U>(
        &self,
        node: usize,
        other: &SpatialIndex<U>,
        visit: &mut impl FnMut(T),
    ) where
        T: Copy,
    {
        let node = self.nodes[node];
        if !other.intersects(node.bounds) {
            return;
        }
        match node.contents {
            SpatialNodeContents::Entry(entry) => visit(self.values[entry]),
            SpatialNodeContents::Branch(children) => {
                self.query_intersecting_node(children[0], other, visit);
                self.query_intersecting_node(children[1], other, visit);
            }
        }
    }

    fn intersects(&self, query: PhysicalRect) -> bool {
        self.root
            .is_some_and(|root| self.intersects_node(root, query))
    }

    fn query_node(&self, node: usize, query: PhysicalRect, visit: &mut impl FnMut(T))
    where
        T: Copy,
    {
        let node = self.nodes[node];
        if node.bounds.intersection(query).is_none() {
            return;
        }
        match node.contents {
            SpatialNodeContents::Entry(entry) => visit(self.values[entry]),
            SpatialNodeContents::Branch(children) => {
                self.query_node(children[0], query, visit);
                self.query_node(children[1], query, visit);
            }
        }
    }

    fn intersects_node(&self, node: usize, query: PhysicalRect) -> bool {
        let node = self.nodes[node];
        if node.bounds.intersection(query).is_none() {
            return false;
        }
        match node.contents {
            SpatialNodeContents::Entry(_) => true,
            SpatialNodeContents::Branch(children) => {
                self.intersects_node(children[0], query) || self.intersects_node(children[1], query)
            }
        }
    }
}

fn enclosing_rect(left: PhysicalRect, right: PhysicalRect) -> PhysicalRect {
    PhysicalRect::from_min_max(
        left.min_x().min(right.min_x()),
        left.min_y().min(right.min_y()),
        left.max_x().max(right.max_x()),
        left.max_y().max(right.max_y()),
    )
    .unwrap()
}

impl RepairPlan {
    /// Returns the non-overlapping rectangles whose union is exactly the damage.
    pub fn regions(&self) -> &[PhysicalRect] {
        &self.regions
    }

    pub(crate) fn spatial(&self) -> &SpatialIndex<()> {
        &self.spatial
    }

    pub(crate) const fn through_epoch(&self) -> u64 {
        self.through_epoch
    }

    /// Returns the number of unique damaged physical pixels.
    pub const fn damaged_pixels(&self) -> u64 {
        self.damaged_pixels
    }
}

/// Damage that remains owed until the encoded repair is acknowledged.
#[derive(Debug)]
pub struct DamageJournal {
    id: u64,
    next_epoch: u64,
    events: Vec<DamageEvent>,
    cached_plan: Option<CachedPlan>,
}

#[derive(Debug)]
struct CachedPlan {
    ordered_input: Vec<PhysicalRect>,
    sorted_input: Vec<PhysicalRect>,
    regions: Arc<[PhysicalRect]>,
    spatial: Arc<SpatialIndex<()>>,
    damaged_pixels: u64,
}

impl Default for DamageJournal {
    fn default() -> Self {
        let id = NEXT_JOURNAL_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .expect("retained UI damage journal ID overflowed");
        Self {
            id,
            next_epoch: 0,
            events: Vec::new(),
            cached_plan: None,
        }
    }
}

impl DamageJournal {
    /// Records a damaged physical rectangle.
    pub fn record(&mut self, rect: PhysicalRect) -> u64 {
        self.next_epoch = self
            .next_epoch
            .checked_add(1)
            .expect("retained UI damage epoch overflowed");
        self.events.push(DamageEvent {
            epoch: self.next_epoch,
            rect,
        });
        self.next_epoch
    }

    pub(crate) const fn latest_epoch(&self) -> u64 {
        self.next_epoch
    }

    /// Builds an exact plan for all currently owed damage.
    pub fn plan(&mut self) -> Option<RepairPlan> {
        let through_epoch = self.events.last()?.epoch;
        let matches_ordered_cache = self.cached_plan.as_ref().is_some_and(|cached| {
            cached.ordered_input.len() == self.events.len()
                && cached
                    .ordered_input
                    .iter()
                    .zip(&self.events)
                    .all(|(cached, event)| *cached == event.rect)
        });
        let (regions, spatial, damaged_pixels) = if matches_ordered_cache {
            let cached = self.cached_plan.as_ref().unwrap();
            (
                cached.regions.clone(),
                cached.spatial.clone(),
                cached.damaged_pixels,
            )
        } else {
            let ordered_input: Vec<_> = self.events.iter().map(|event| event.rect).collect();
            let mut sorted_input = ordered_input.clone();
            sorted_input.sort_unstable();
            if self
                .cached_plan
                .as_ref()
                .is_some_and(|cached| cached.sorted_input == sorted_input)
            {
                let cached = self.cached_plan.as_mut().unwrap();
                cached.ordered_input = ordered_input;
                (
                    cached.regions.clone(),
                    cached.spatial.clone(),
                    cached.damaged_pixels,
                )
            } else {
                let regions: Arc<[_]> = exact_union(sorted_input.iter().copied()).into();
                let damaged_pixels = regions.iter().map(PhysicalRect::area).sum();
                let spatial = Arc::new(SpatialIndex::new(
                    regions.iter().copied().map(|rect| (rect, ())),
                ));
                self.cached_plan = Some(CachedPlan {
                    ordered_input,
                    sorted_input,
                    regions: regions.clone(),
                    spatial: spatial.clone(),
                    damaged_pixels,
                });
                (regions, spatial, damaged_pixels)
            }
        };
        Some(RepairPlan {
            journal_id: self.id,
            through_epoch,
            regions,
            spatial,
            damaged_pixels,
        })
    }

    /// Clears only damage included in a repair that was actually encoded.
    pub fn acknowledge(&mut self, plan: &RepairPlan) {
        assert_eq!(
            plan.journal_id, self.id,
            "a repair plan must be acknowledged by its originating damage journal"
        );
        self.events.retain(|event| event.epoch > plan.through_epoch);
    }

    /// Returns whether any damage remains owed.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

pub(crate) fn exact_union(rects: impl IntoIterator<Item = PhysicalRect>) -> Vec<PhysicalRect> {
    let rects: Vec<_> = rects.into_iter().collect();
    if rects.is_empty() {
        return Vec::new();
    }

    let mut y_edges: Vec<_> = rects
        .iter()
        .flat_map(|rect| [rect.min_y(), rect.max_y()])
        .collect();
    y_edges.sort_unstable();
    y_edges.dedup();

    #[derive(Clone, Copy)]
    struct Event {
        x: i32,
        start: usize,
        end: usize,
        delta: i32,
    }

    let mut events = Vec::with_capacity(rects.len() * 2);
    for rect in rects {
        let start = y_edges.binary_search(&rect.min_y()).unwrap();
        let end = y_edges.binary_search(&rect.max_y()).unwrap();
        events.push(Event {
            x: rect.min_x(),
            start,
            end,
            delta: 1,
        });
        events.push(Event {
            x: rect.max_x(),
            start,
            end,
            delta: -1,
        });
    }
    events.sort_unstable_by_key(|event| event.x);

    let mut regions = Vec::<PhysicalRect>::new();
    let mut previous = HashMap::<(i32, i32), usize>::new();
    let mut coverage = CoverageTree::new(y_edges);
    let mut event_index = 0;

    while event_index < events.len() {
        let min_x = events[event_index].x;
        while event_index < events.len() && events[event_index].x == min_x {
            let event = events[event_index];
            coverage.update(event.start, event.end, event.delta);
            event_index += 1;
        }
        let Some(next) = events.get(event_index) else {
            break;
        };
        let max_x = next.x;

        let intervals = coverage.intervals();

        let mut current = HashMap::new();
        for (min_y, max_y) in intervals {
            let index = if let Some(&index) = previous.get(&(min_y, max_y)) {
                regions[index].max_x = max_x;
                index
            } else {
                let index = regions.len();
                regions.push(
                    PhysicalRect::from_min_max(min_x, min_y, max_x, max_y)
                        .expect("sweep strips are non-empty"),
                );
                index
            };
            current.insert((min_y, max_y), index);
        }
        previous = current;
    }

    regions
}

struct CoverageTree {
    y_edges: Vec<i32>,
    cover: Vec<i32>,
}

impl CoverageTree {
    fn new(y_edges: Vec<i32>) -> Self {
        let segment_count = y_edges.len() - 1;
        Self {
            y_edges,
            cover: vec![0; segment_count * 4],
        }
    }

    fn update(&mut self, start: usize, end: usize, delta: i32) {
        self.update_node(1, 0, self.y_edges.len() - 1, start, end, delta);
    }

    fn update_node(
        &mut self,
        node: usize,
        node_start: usize,
        node_end: usize,
        update_start: usize,
        update_end: usize,
        delta: i32,
    ) {
        if update_start <= node_start && node_end <= update_end {
            self.cover[node] += delta;
            debug_assert!(self.cover[node] >= 0);
            return;
        }

        let middle = (node_start + node_end) / 2;
        if update_start < middle {
            self.update_node(
                node * 2,
                node_start,
                middle,
                update_start,
                update_end,
                delta,
            );
        }
        if update_end > middle {
            self.update_node(
                node * 2 + 1,
                middle,
                node_end,
                update_start,
                update_end,
                delta,
            );
        }
    }

    fn intervals(&self) -> Vec<(i32, i32)> {
        let mut intervals = Vec::new();
        self.collect(1, 0, self.y_edges.len() - 1, &mut intervals);
        intervals
    }

    fn collect(
        &self,
        node: usize,
        node_start: usize,
        node_end: usize,
        intervals: &mut Vec<(i32, i32)>,
    ) {
        if self.cover[node] > 0 {
            Self::push_interval(intervals, self.y_edges[node_start], self.y_edges[node_end]);
            return;
        }
        if node_end - node_start == 1 {
            return;
        }

        let middle = (node_start + node_end) / 2;
        self.collect(node * 2, node_start, middle, intervals);
        self.collect(node * 2 + 1, middle, node_end, intervals);
    }

    fn push_interval(intervals: &mut Vec<(i32, i32)>, min_y: i32, max_y: i32) {
        if let Some(previous) = intervals.last_mut()
            && previous.1 == min_y
        {
            previous.1 = max_y;
        } else {
            intervals.push((min_y, max_y));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(min_x: i32, min_y: i32, max_x: i32, max_y: i32) -> PhysicalRect {
        PhysicalRect::from_min_max(min_x, min_y, max_x, max_y).unwrap()
    }

    fn covered(regions: &[PhysicalRect], x: i32, y: i32) -> bool {
        regions.iter().any(|rect| {
            rect.min_x() <= x && x < rect.max_x() && rect.min_y() <= y && y < rect.max_y()
        })
    }

    #[test]
    fn exact_union_never_fills_the_missing_corner_of_an_l_shape() {
        let regions = exact_union([rect(0, 0, 2, 1), rect(0, 1, 1, 2)]);

        assert_eq!(regions.iter().map(PhysicalRect::area).sum::<u64>(), 3);
        assert!(covered(&regions, 0, 0));
        assert!(covered(&regions, 1, 0));
        assert!(covered(&regions, 0, 1));
        assert!(!covered(&regions, 1, 1));
    }

    #[test]
    fn intersection_returns_only_shared_pixels() {
        assert_eq!(
            rect(0, 1, 4, 5).intersection(rect(2, -1, 7, 3)),
            Some(rect(2, 1, 4, 3))
        );
        assert_eq!(rect(0, 0, 1, 1).intersection(rect(1, 0, 2, 1)), None);
    }

    #[test]
    fn spatial_queries_match_exhaustive_intersections() {
        let entries: Vec<_> = (0..257)
            .map(|index| {
                let x = index * 37 % 113 - 20;
                let y = index * 53 % 97 - 15;
                let width = index % 11 + 1;
                let height = index % 7 + 1;
                (rect(x, y, x + width, y + height), index)
            })
            .collect();
        let index = SpatialIndex::new(entries.iter().copied());

        for query_number in 0..101 {
            let x = (query_number * 29 % 127) - 25;
            let y = (query_number * 31 % 109) - 20;
            let query = rect(x, y, x + 9, y + 6);
            let mut actual = Vec::new();
            index.query(query, |entry| actual.push(entry));
            actual.sort_unstable();
            let expected: Vec<_> = entries
                .iter()
                .filter_map(|(region, entry)| region.intersection(query).map(|_| *entry))
                .collect();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn index_to_index_query_emits_each_source_once() {
        let source: Vec<_> = (0..1_000)
            .map(|entry| (rect(entry * 3, 0, entry * 3 + 2, 2), entry))
            .collect();
        let damage = SpatialIndex::new([
            (rect(0, 0, 1_500, 1), ()),
            (rect(750, 1, 2_250, 2), ()),
            (rect(2_700, 0, 3_000, 2), ()),
        ]);
        let index = SpatialIndex::new(source.iter().copied());
        let mut actual = Vec::new();
        index.query_intersecting(&damage, |entry| actual.push(entry));
        actual.sort_unstable();
        let expected: Vec<_> = source
            .iter()
            .filter_map(|(region, entry)| damage.intersects(*region).then_some(*entry))
            .collect();

        assert_eq!(actual, expected);
        assert!(actual.windows(2).all(|pair| pair[0] != pair[1]));
    }

    #[test]
    fn refitting_one_entry_updates_queries_without_rebuilding() {
        let mut index = SpatialIndex::new([
            (rect(0, 0, 2, 2), 0),
            (rect(10, 10, 12, 12), 1),
            (rect(20, 20, 22, 22), 2),
        ]);

        index.update_many([(1, rect(3, 3, 5, 5))]);

        let mut old = Vec::new();
        index.query(rect(10, 10, 12, 12), |entry| old.push(entry));
        assert!(old.is_empty());
        let mut new = Vec::new();
        index.query(rect(3, 3, 5, 5), |entry| new.push(entry));
        assert_eq!(new, [1]);
    }

    #[test]
    fn exact_union_matches_source_coverage_for_overlapping_rectangles() {
        let source = [rect(-2, -1, 2, 2), rect(0, -3, 3, 1), rect(1, 1, 4, 3)];
        let regions = exact_union(source);

        for y in -4..4 {
            for x in -3..5 {
                assert_eq!(
                    covered(&regions, x, y),
                    covered(&source, x, y),
                    "coverage differed at ({x}, {y})"
                );
            }
        }

        for (index, left) in regions.iter().enumerate() {
            for right in &regions[index + 1..] {
                let overlaps = left.min_x() < right.max_x()
                    && right.min_x() < left.max_x()
                    && left.min_y() < right.max_y()
                    && right.min_y() < left.max_y();
                assert!(!overlaps, "union regions must not overlap");
            }
        }
    }

    #[test]
    fn exact_union_handles_many_disjoint_strips_without_pairwise_search() {
        let source = (0..10_000).map(|x| rect(x * 2, 0, x * 2 + 1, 1));
        let regions = exact_union(source);

        assert_eq!(regions.len(), 10_000);
        assert_eq!(regions.iter().map(PhysicalRect::area).sum::<u64>(), 10_000);
    }

    #[test]
    fn unacknowledged_damage_remains_owed() {
        let mut journal = DamageJournal::default();
        journal.record(rect(0, 0, 2, 2));
        let first = journal.plan().unwrap();

        assert_eq!(journal.plan(), Some(first));
        assert!(!journal.is_empty());
    }

    #[test]
    fn acknowledging_a_plan_does_not_clear_later_damage() {
        let mut journal = DamageJournal::default();
        journal.record(rect(0, 0, 2, 2));
        let first = journal.plan().unwrap();
        journal.record(rect(5, 5, 6, 6));

        journal.acknowledge(&first);

        let remaining = journal.plan().unwrap();
        assert_eq!(remaining.regions(), &[rect(5, 5, 6, 6)]);
    }

    #[test]
    #[should_panic(expected = "originating damage journal")]
    fn a_repair_plan_cannot_acknowledge_another_surface() {
        let mut first = DamageJournal::default();
        first.record(rect(0, 0, 1, 1));
        let plan = first.plan().unwrap();

        let mut second = DamageJournal::default();
        second.record(rect(5, 5, 6, 6));
        second.acknowledge(&plan);
    }
}
