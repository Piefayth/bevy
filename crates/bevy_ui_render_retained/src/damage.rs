//! Exact physical-pixel damage tracking.

use core::sync::atomic::{AtomicU64, Ordering};
use std::collections::HashMap;

static NEXT_JOURNAL_ID: AtomicU64 = AtomicU64::new(1);

/// A non-empty rectangle of physical pixels with an exclusive maximum edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
    regions: Vec<PhysicalRect>,
    damaged_pixels: u64,
}

impl RepairPlan {
    /// Returns the non-overlapping rectangles whose union is exactly the damage.
    pub fn regions(&self) -> &[PhysicalRect] {
        &self.regions
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
        }
    }
}

impl DamageJournal {
    /// Records a damaged physical rectangle.
    pub fn record(&mut self, rect: PhysicalRect) {
        self.next_epoch = self
            .next_epoch
            .checked_add(1)
            .expect("retained UI damage epoch overflowed");
        self.events.push(DamageEvent {
            epoch: self.next_epoch,
            rect,
        });
    }

    /// Builds an exact plan for all currently owed damage.
    pub fn plan(&self) -> Option<RepairPlan> {
        let through_epoch = self.events.last()?.epoch;
        let regions = exact_union(self.events.iter().map(|event| event.rect));
        let damaged_pixels = regions.iter().map(PhysicalRect::area).sum();
        Some(RepairPlan {
            journal_id: self.id,
            through_epoch,
            regions,
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
