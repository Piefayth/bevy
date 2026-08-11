use bevy_ui_render_retained::{PaintRecord, PhysicalRect, RetainedPaint};
use criterion::{criterion_group, BenchmarkId, Criterion};
use std::hint::black_box;

#[derive(Clone, Copy)]
enum Coverage {
    Tiled,
    Scattered,
    Overlapping,
}

impl Coverage {
    const fn name(self) -> &'static str {
        match self {
            Self::Tiled => "tiled",
            Self::Scattered => "scattered",
            Self::Overlapping => "overlapping",
        }
    }
}

fn coverage(shape: Coverage, index: u32) -> PhysicalRect {
    let x = (index % 100) as i32;
    let y = (index / 100) as i32;
    match shape {
        Coverage::Tiled => PhysicalRect::from_min_max(x, y, x + 1, y + 1).unwrap(),
        Coverage::Scattered => {
            PhysicalRect::from_min_max(x * 2, y * 2, x * 2 + 1, y * 2 + 1).unwrap()
        }
        Coverage::Overlapping => PhysicalRect::from_min_max(0, 0, 8, 8).unwrap(),
    }
}

fn retained_paint(record_count: u32, shape: Coverage) -> RetainedPaint<u32, u32> {
    let mut paint = RetainedPaint::default();
    for id in 0..record_count {
        paint.upsert(
            id,
            PaintRecord {
                coverage: coverage(shape, id).into(),
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
        group.bench_with_input(
            BenchmarkId::new("quiet", record_count),
            &record_count,
            |bencher, _| {
                let quiet = retained_paint(record_count, Coverage::Tiled);
                bencher.iter(|| black_box(quiet.repair_plan()));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("localized_change", record_count),
            &record_count,
            |bencher, _| {
                let mut localized = retained_paint(record_count, Coverage::Tiled);
                let localized_id = record_count - 1;
                let mut localized_value = 0;
                bencher.iter(|| {
                    localized_value ^= 1;
                    localized.upsert(
                        localized_id,
                        PaintRecord {
                            coverage: coverage(Coverage::Tiled, localized_id).into(),
                            value: localized_value,
                        },
                    );
                    let repair = localized.repair_plan().unwrap();
                    black_box(repair.damaged_pixels());
                    localized.acknowledge(&repair);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("full_change", record_count),
            &record_count,
            |bencher, _| {
                let mut full = retained_paint(record_count, Coverage::Tiled);
                let mut full_value = 0;
                bencher.iter(|| {
                    full_value ^= 1;
                    for id in 0..record_count {
                        full.upsert(
                            id,
                            PaintRecord {
                                coverage: coverage(Coverage::Tiled, id).into(),
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

    let record_count = 10_000;
    for shape in [Coverage::Scattered, Coverage::Overlapping] {
        group.bench_function(format!("full_change_{}/10_000", shape.name()), |bencher| {
            let mut paint = retained_paint(record_count, shape);
            let mut value = 0;
            bencher.iter(|| {
                value ^= 1;
                for id in 0..record_count {
                    paint.upsert(
                        id,
                        PaintRecord {
                            coverage: coverage(shape, id).into(),
                            value,
                        },
                    );
                }
                let repair = paint.repair_plan().unwrap();
                black_box((repair.damaged_pixels(), repair.regions().len()));
                paint.acknowledge(&repair);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, paint);
