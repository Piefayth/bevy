use bevy_ui_render_retained::{PaintCoverage, PhysicalRect, ReplayItem, ReplayPlan};
use criterion::{criterion_group, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

#[derive(Clone, Copy)]
enum Scene {
    Disjoint,
    Clustered,
    Alternating,
    Overlapping,
}

impl Scene {
    const ALL: [Self; 4] = [
        Self::Disjoint,
        Self::Clustered,
        Self::Alternating,
        Self::Overlapping,
    ];

    const fn name(self) -> &'static str {
        match self {
            Self::Disjoint => "disjoint_one_hit",
            Self::Clustered => "clustered_ten_percent_hit",
            Self::Alternating => "alternating_half_hit",
            Self::Overlapping => "overlapping_all_hit",
        }
    }

    fn coverage(self, index: usize, count: usize) -> PhysicalRect {
        let miss_x = 20 + index as i32 * 2;
        match self {
            Self::Disjoint => {
                if index + 1 == count {
                    rect(0, 0, 10, 10)
                } else {
                    rect(miss_x, 0, miss_x + 1, 1)
                }
            }
            Self::Clustered => {
                if index >= count - count / 10 {
                    rect(0, 0, 10, 10)
                } else {
                    rect(miss_x, 0, miss_x + 1, 1)
                }
            }
            Self::Alternating => {
                if index.is_multiple_of(2) {
                    rect(0, 0, 10, 10)
                } else {
                    rect(miss_x, 0, miss_x + 1, 1)
                }
            }
            Self::Overlapping => rect(0, 0, 10, 10),
        }
    }
}

fn rect(min_x: i32, min_y: i32, max_x: i32, max_y: i32) -> PhysicalRect {
    PhysicalRect::from_min_max(min_x, min_y, max_x, max_y).unwrap()
}

fn replay(c: &mut Criterion) {
    let mut group = c.benchmark_group("retained_replay_selection");

    for count in [100, 1_000, 10_000] {
        group.throughput(Throughput::Elements(count as u64));
        for scene in Scene::ALL {
            group.bench_with_input(
                BenchmarkId::new(scene.name(), count),
                &count,
                |bencher, _| {
                    let coverage: Vec<_> = (0..count)
                        .map(|index| PaintCoverage::one(scene.coverage(index, count)))
                        .collect();
                    bencher.iter(|| {
                        let plan = ReplayPlan::for_region(
                            black_box(rect(0, 0, 10, 10)),
                            coverage.iter().map(ReplayItem::Bounded),
                        );
                        black_box((plan.item_count(), plan.runs().len()));
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, replay);
