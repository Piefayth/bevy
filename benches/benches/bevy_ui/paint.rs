use bevy_ui_render_retained::{PaintRecord, PhysicalRect, RetainedPaint};
use criterion::{BenchmarkId, Criterion, criterion_group};
use std::hint::black_box;

fn coverage(index: u32) -> PhysicalRect {
    let x = (index % 100) as i32;
    let y = (index / 100) as i32;
    PhysicalRect::from_min_max(x, y, x + 1, y + 1).unwrap()
}

fn retained_paint(record_count: u32) -> RetainedPaint<u32, u32> {
    let mut paint = RetainedPaint::default();
    for id in 0..record_count {
        paint.upsert(
            id,
            PaintRecord {
                coverage: Some(coverage(id)),
                value: 0,
            },
        );
    }
    let initial = paint.repair_plan().unwrap();
    paint.acknowledge(&initial);
    paint.take_counters();
    paint
}

fn paint(c: &mut Criterion) {
    let mut group = c.benchmark_group("retained_paint");

    for record_count in [100, 1_000, 10_000] {
        let quiet = retained_paint(record_count);
        group.bench_with_input(
            BenchmarkId::new("quiet", record_count),
            &record_count,
            |bencher, _| bencher.iter(|| black_box(quiet.repair_plan())),
        );

        let mut localized = retained_paint(record_count);
        let localized_id = record_count - 1;
        let mut localized_value = 0;
        group.bench_with_input(
            BenchmarkId::new("localized_change", record_count),
            &record_count,
            |bencher, _| {
                bencher.iter(|| {
                    localized_value ^= 1;
                    localized.upsert(
                        localized_id,
                        PaintRecord {
                            coverage: Some(coverage(localized_id)),
                            value: localized_value,
                        },
                    );
                    let repair = localized.repair_plan().unwrap();
                    black_box(repair.damaged_pixels());
                    localized.acknowledge(&repair);
                });
            },
        );

        let mut full = retained_paint(record_count);
        let mut full_value = 0;
        group.bench_with_input(
            BenchmarkId::new("full_change", record_count),
            &record_count,
            |bencher, _| {
                bencher.iter(|| {
                    full_value ^= 1;
                    for id in 0..record_count {
                        full.upsert(
                            id,
                            PaintRecord {
                                coverage: Some(coverage(id)),
                                value: full_value,
                            },
                        );
                    }
                    let repair = full.repair_plan().unwrap();
                    black_box(repair.damaged_pixels());
                    full.acknowledge(&repair);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, paint);
