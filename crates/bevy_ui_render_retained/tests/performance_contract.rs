//! Deterministic large-scene work contracts complementing wall-clock benchmarks.

use bevy_ui_render_retained::{
    PaintCoverage, PaintRecord, PhysicalRect, ReplayItem, ReplayPlan, RetainedPaint, WorkCounters,
};

const ITEM_COUNT: u32 = 10_000;

fn rect(min_x: i32, min_y: i32, max_x: i32, max_y: i32) -> PhysicalRect {
    PhysicalRect::from_min_max(min_x, min_y, max_x, max_y).unwrap()
}

fn record(coverage: PhysicalRect, value: u32) -> PaintRecord<u32> {
    PaintRecord {
        coverage: coverage.into(),
        value,
    }
}

fn settled_overlapping_paint() -> RetainedPaint<u32, u32> {
    let mut paint = RetainedPaint::default();
    for id in 0..ITEM_COUNT {
        paint.upsert(id, record(rect(0, 0, 10, 10), 0));
    }
    let initial = paint.repair_plan().unwrap();
    paint.acknowledge(&initial);
    paint.take_counters();
    paint
}

#[test]
fn ten_thousand_static_records_nominate_and_damage_nothing() {
    let mut paint = settled_overlapping_paint();

    assert!(paint.repair_plan().is_none());
    assert_eq!(paint.take_counters(), WorkCounters::default());
}

#[test]
fn one_animation_among_ten_thousand_submits_one_candidate() {
    let mut paint = settled_overlapping_paint();

    paint.upsert(ITEM_COUNT - 1, record(rect(0, 0, 10, 10), 1));

    assert_eq!(
        paint.take_counters(),
        WorkCounters {
            candidates: 1,
            records_compared: 1,
            records_changed: 1,
            damage_events: 2,
            ..Default::default()
        }
    );
    let repair = paint.repair_plan().unwrap();
    assert_eq!(repair.damaged_pixels(), 100);
    assert_eq!(repair.regions(), &[rect(0, 0, 10, 10)]);
}

#[test]
fn ten_thousand_animations_submit_exactly_ten_thousand_candidates() {
    let mut paint = settled_overlapping_paint();

    for id in 0..ITEM_COUNT {
        paint.upsert(id, record(rect(0, 0, 10, 10), 1));
    }

    assert_eq!(
        paint.take_counters(),
        WorkCounters {
            candidates: ITEM_COUNT.into(),
            records_compared: ITEM_COUNT.into(),
            records_changed: ITEM_COUNT.into(),
            damage_events: (ITEM_COUNT * 2).into(),
            ..Default::default()
        }
    );
    let repair = paint.repair_plan().unwrap();
    assert_eq!(repair.damaged_pixels(), 100);
    assert_eq!(repair.regions(), &[rect(0, 0, 10, 10)]);
}

#[test]
fn replay_cardinality_tracks_overlap_not_total_scene_size() {
    let damage = rect(0, 0, 10, 10);
    let hit = PaintCoverage::one(damage);
    let miss = PaintCoverage::one(rect(20, 20, 21, 21));

    let one_hit = ReplayPlan::for_region(
        damage,
        (0..ITEM_COUNT).map(|index| {
            if index + 1 == ITEM_COUNT {
                ReplayItem::Bounded(&hit)
            } else {
                ReplayItem::Bounded(&miss)
            }
        }),
    );
    assert_eq!(one_hit.item_count(), 1);
    assert_eq!(one_hit.runs().len(), 1);
    assert_eq!(one_hit.runs()[0], 9_999..10_000);

    let all_hit =
        ReplayPlan::for_region(damage, (0..ITEM_COUNT).map(|_| ReplayItem::Bounded(&hit)));
    assert_eq!(all_hit.item_count(), ITEM_COUNT as usize);
    assert_eq!(all_hit.runs().len(), 1);
    assert_eq!(all_hit.runs()[0], 0..10_000);

    let fragmented = ReplayPlan::for_region(
        damage,
        (0..ITEM_COUNT).map(|index| {
            if index.is_multiple_of(2) {
                ReplayItem::Bounded(&hit)
            } else {
                ReplayItem::Bounded(&miss)
            }
        }),
    );
    assert_eq!(fragmented.item_count(), ITEM_COUNT as usize / 2);
    assert_eq!(fragmented.runs().len(), ITEM_COUNT as usize / 2);
}

#[test]
fn one_removal_and_replacement_do_constant_record_work() {
    let mut paint = settled_overlapping_paint();
    let id = ITEM_COUNT - 1;

    assert!(paint.remove(&id));
    paint.upsert(id, record(rect(20, 20, 30, 30), 0));

    assert_eq!(
        paint.take_counters(),
        WorkCounters {
            candidates: 1,
            records_changed: 1,
            records_removed: 1,
            damage_events: 2,
            ..Default::default()
        }
    );
    let repair = paint.repair_plan().unwrap();
    assert_eq!(repair.damaged_pixels(), 200);
    assert_eq!(repair.regions().len(), 2);
}
