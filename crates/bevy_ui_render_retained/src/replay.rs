//! Selection of sorted paint items needed to rebuild damaged pixels.

use crate::{PaintCoverage, PhysicalRect};
use core::ops::Range;

/// Visibility information for one item in paint order.
#[derive(Clone, Copy, Debug)]
pub enum ReplayItem<'a> {
    /// The item has no prepared draw this frame.
    Culled,
    /// The item has no exact retained coverage and must be replayed conservatively.
    Unbounded,
    /// The item has exact retained physical-pixel coverage.
    Bounded(&'a PaintCoverage),
}

impl ReplayItem<'_> {
    fn intersects(self, region: PhysicalRect) -> bool {
        match self {
            Self::Culled => false,
            Self::Unbounded => true,
            Self::Bounded(coverage) => coverage
                .iter()
                .any(|coverage| coverage.intersection(region).is_some()),
        }
    }
}

/// Contiguous ranges of sorted phase items needed to rebuild one damage region.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReplayPlan {
    runs: Vec<Range<usize>>,
    item_count: usize,
}

impl ReplayPlan {
    /// Selects intersecting items while preserving their original paint order.
    pub fn for_region<'a>(
        region: PhysicalRect,
        items: impl IntoIterator<Item = ReplayItem<'a>>,
    ) -> Self {
        let mut runs = Vec::new();
        let mut run_start = None;
        let mut item_count = 0;
        let mut len = 0;

        for item in items {
            let intersects = item.intersects(region);
            match (run_start, intersects) {
                (None, true) => run_start = Some(len),
                (Some(start), false) => {
                    runs.push(start..len);
                    run_start = None;
                }
                _ => {}
            }
            item_count += usize::from(intersects);
            len += 1;
        }
        if let Some(start) = run_start {
            runs.push(start..len);
        }

        Self { runs, item_count }
    }

    /// Returns the contiguous sorted-phase ranges to render.
    pub fn runs(&self) -> &[Range<usize>] {
        &self.runs
    }

    /// Returns the number of selected paint items.
    pub const fn item_count(&self) -> usize {
        self.item_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(min_x: i32, min_y: i32, max_x: i32, max_y: i32) -> PhysicalRect {
        PhysicalRect::from_min_max(min_x, min_y, max_x, max_y).unwrap()
    }

    #[test]
    fn selection_preserves_order_and_groups_only_consecutive_items() {
        let hit = PaintCoverage::one(rect(0, 0, 2, 2));
        let miss = PaintCoverage::one(rect(4, 4, 6, 6));
        let plan = ReplayPlan::for_region(
            rect(1, 1, 3, 3),
            [
                ReplayItem::Bounded(&hit),
                ReplayItem::Bounded(&hit),
                ReplayItem::Bounded(&miss),
                ReplayItem::Unbounded,
                ReplayItem::Culled,
                ReplayItem::Bounded(&hit),
            ],
        );

        assert_eq!(plan.runs(), &[0..2, 3..4, 5..6]);
        assert_eq!(plan.item_count(), 4);
    }

    #[test]
    fn no_intersections_produce_no_render_ranges() {
        let miss = PaintCoverage::one(rect(4, 4, 6, 6));
        let plan = ReplayPlan::for_region(
            rect(0, 0, 2, 2),
            [ReplayItem::Culled, ReplayItem::Bounded(&miss)],
        );

        assert!(plan.runs().is_empty());
        assert_eq!(plan.item_count(), 0);
    }
}
