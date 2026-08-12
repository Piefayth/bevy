use bevy_app::{App, PostUpdate, TaskPoolPlugin};
use bevy_asset::{AssetApp, AssetPlugin};
use bevy_camera::{Camera, Camera2d, ComputedCameraValues, RenderTargetInfo, Viewport};
use bevy_ecs::prelude::*;
use bevy_image::ImagePlugin;
use bevy_math::UVec2;
use bevy_text::TextPlugin;
use bevy_time::TimePlugin;
use bevy_ui::{BorderRadius, LayoutContainment, Node, PositionType, UiPlugin, UiTransform, Val};
use bevy_ui_render_retained::RetainedUiMainWorldPlugin;
use criterion::{
    criterion_group, measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion, Throughput,
};
use std::time::{Duration, Instant};

const TARGET_SIZE: UVec2 = UVec2::new(1024, 1024);
const FOREST_SIZE: usize = 100;

#[derive(Clone, Copy, PartialEq, Eq)]
enum TreeShape {
    Flat,
    Balanced,
    Forest,
    Contained(usize),
    AbsoluteContained(usize),
    Deep,
}

impl TreeShape {
    fn name(self) -> String {
        match self {
            Self::Flat => "flat".into(),
            Self::Balanced => "balanced".into(),
            Self::Forest => "forest_100".into(),
            Self::Contained(size) => format!("contained_{size}"),
            Self::AbsoluteContained(size) => format!("absolute_contained_{size}"),
            Self::Deep => "deep".into(),
        }
    }

    const fn containment_size(self) -> Option<usize> {
        match self {
            Self::Contained(size) | Self::AbsoluteContained(size) => Some(size),
            _ => None,
        }
    }
}

struct LayoutApp {
    app: App,
    nodes: Vec<Entity>,
    roots: Vec<Entity>,
    boundaries: Vec<Entity>,
    contained_leaves: Vec<Entity>,
}

fn node() -> Node {
    Node {
        width: Val::Px(8.0),
        height: Val::Px(8.0),
        ..Default::default()
    }
}

fn root_node() -> Node {
    Node {
        width: Val::Px(TARGET_SIZE.x as f32),
        height: Val::Px(TARGET_SIZE.y as f32),
        ..Default::default()
    }
}

fn node_for(shape: TreeShape, index: usize) -> Node {
    if matches!(shape, TreeShape::AbsoluteContained(_)) {
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px((index % 100) as f32 * 10.0),
            top: Val::Px((index / 100) as f32 * 10.0),
            ..node()
        }
    } else {
        node()
    }
}

fn layout_app(node_count: usize, retained: bool, shape: TreeShape) -> LayoutApp {
    assert!(node_count > 0);
    let mut app = App::new();
    app.add_plugins((
        TaskPoolPlugin::default(),
        TimePlugin,
        AssetPlugin::default(),
        ImagePlugin::default(),
        TextPlugin,
        UiPlugin,
    ))
    .init_asset::<bevy_image::TextureAtlasLayout>();
    if retained {
        app.add_plugins(RetainedUiMainWorldPlugin);
    }

    app.world_mut().spawn((
        Camera2d,
        Camera {
            computed: ComputedCameraValues {
                target_info: Some(RenderTargetInfo {
                    physical_size: TARGET_SIZE,
                    scale_factor: 1.0,
                }),
                ..Default::default()
            },
            viewport: Some(Viewport {
                physical_size: TARGET_SIZE,
                ..Default::default()
            }),
            ..Default::default()
        },
    ));

    let mut nodes = Vec::with_capacity(node_count);
    let mut roots = Vec::new();
    let mut boundaries = Vec::new();
    let mut contained_leaves = Vec::new();
    let containment_size = shape.containment_size();
    if let Some(size) = containment_size {
        assert!(size >= 2);
    }
    for index in 0..node_count {
        let parent = if let Some(size) = containment_size {
            (index > 0).then(|| {
                if (index - 1) % size == 0 {
                    nodes[0]
                } else {
                    nodes[index - (index - 1) % size]
                }
            })
        } else {
            match shape {
                TreeShape::Flat if index > 0 => Some(nodes[0]),
                TreeShape::Balanced if index > 0 => Some(nodes[(index - 1) / 4]),
                TreeShape::Forest if index % FOREST_SIZE != 0 => {
                    Some(nodes[index - index % FOREST_SIZE])
                }
                TreeShape::Deep if index > 0 => nodes.last().copied(),
                _ => None,
            }
        };
        let entity = if let Some(parent) = parent {
            let mut entity = app
                .world_mut()
                .spawn((node_for(shape, index), ChildOf(parent)));
            if containment_size.is_some_and(|size| (index - 1) % size == 0) {
                entity.insert(LayoutContainment);
            }
            let entity = entity.id();
            if let Some(size) = containment_size {
                if (index - 1) % size == 0 {
                    boundaries.push(entity);
                } else {
                    contained_leaves.push(entity);
                }
            }
            entity
        } else {
            let entity = app.world_mut().spawn(root_node()).id();
            roots.push(entity);
            entity
        };
        nodes.push(entity);
    }

    app.world_mut().run_schedule(PostUpdate);
    LayoutApp {
        app,
        nodes,
        roots,
        boundaries,
        contained_leaves,
    }
}

fn measure_updates(
    app: &mut App,
    iterations: u64,
    mut prepare: impl FnMut(&mut World),
) -> Duration {
    let mut elapsed = Duration::ZERO;
    for _ in 0..iterations {
        prepare(app.world_mut());
        let start = Instant::now();
        app.world_mut().run_schedule(PostUpdate);
        elapsed += start.elapsed();
    }
    elapsed
}

fn bench_scene(
    group: &mut BenchmarkGroup<'_, WallTime>,
    node_count: usize,
    retained: bool,
    shape: TreeShape,
    include_full_change: bool,
    include_reparent: bool,
) {
    group.throughput(Throughput::Elements(node_count as u64));

    group.bench_with_input(
        BenchmarkId::new("quiet", node_count),
        &node_count,
        |bencher, _| {
            let mut quiet = layout_app(node_count, retained, shape);
            bencher.iter_custom(|iterations| measure_updates(&mut quiet.app, iterations, |_| {}));
        },
    );

    group.bench_with_input(
        BenchmarkId::new("one_layout_change", node_count),
        &node_count,
        |bencher, _| {
            let mut localized = layout_app(node_count, retained, shape);
            let localized_node = *localized.nodes.last().unwrap();
            let mut localized_width = 8.0;
            bencher.iter_custom(|iterations| {
                measure_updates(&mut localized.app, iterations, |world| {
                    localized_width = if localized_width == 8.0 { 9.0 } else { 8.0 };
                    world.get_mut::<Node>(localized_node).unwrap().width = Val::Px(localized_width);
                })
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("one_placement_change", node_count),
        &node_count,
        |bencher, _| {
            let mut placement = layout_app(node_count, retained, shape);
            let placement_node = *placement.nodes.last().unwrap();
            let mut placement_x = 0.0;
            bencher.iter_custom(|iterations| {
                measure_updates(&mut placement.app, iterations, |world| {
                    placement_x = if placement_x == 0.0 { 1.0 } else { 0.0 };
                    world
                        .get_mut::<UiTransform>(placement_node)
                        .unwrap()
                        .translation
                        .x = Val::Px(placement_x);
                })
            });
        },
    );

    if include_full_change {
        group.bench_with_input(
            BenchmarkId::new("one_equal_node_write", node_count),
            &node_count,
            |bencher, _| {
                let mut equal = layout_app(node_count, retained, shape);
                let entity = *equal.nodes.last().unwrap();
                bencher.iter_custom(|iterations| {
                    measure_updates(&mut equal.app, iterations, |world| {
                        world.get_mut::<Node>(entity).unwrap().width = Val::Px(8.0);
                    })
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("one_node_geometry_change", node_count),
            &node_count,
            |bencher, _| {
                let mut local = layout_app(node_count, retained, shape);
                let entity = *local.nodes.last().unwrap();
                let mut radius = 0.0;
                bencher.iter_custom(|iterations| {
                    measure_updates(&mut local.app, iterations, |world| {
                        radius = if radius == 0.0 { 1.0 } else { 0.0 };
                        world.get_mut::<Node>(entity).unwrap().border_radius =
                            BorderRadius::all(Val::Px(radius));
                    })
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("all_layout_change", node_count),
            &node_count,
            |bencher, _| {
                let mut full = layout_app(node_count, retained, shape);
                let mut full_width = 8.0;
                bencher.iter_custom(|iterations| {
                    measure_updates(&mut full.app, iterations, |world| {
                        full_width = if full_width == 8.0 { 9.0 } else { 8.0 };
                        for &entity in &full.nodes {
                            world.get_mut::<Node>(entity).unwrap().width = Val::Px(full_width);
                        }
                    })
                });
            },
        );
    }

    if include_reparent {
        group.bench_with_input(
            BenchmarkId::new("one_reparent", node_count),
            &node_count,
            |bencher, _| {
                let mut reparent = layout_app(node_count, retained, shape);
                assert!(reparent.roots.len() >= 2);
                let child = *reparent.nodes.last().unwrap();
                let parents = [reparent.roots[0], reparent.roots[1]];
                let mut parent = 0;
                bencher.iter_custom(|iterations| {
                    measure_updates(&mut reparent.app, iterations, |world| {
                        parent ^= 1;
                        world.entity_mut(child).insert(ChildOf(parents[parent]));
                    })
                });
            },
        );
    }
    if let Some(containment_size) = shape.containment_size() {
        group.bench_with_input(
            BenchmarkId::new("one_boundary_layout_change", node_count),
            &node_count,
            |bencher, _| {
                let mut boundary = layout_app(node_count, retained, shape);
                let entity = *boundary.boundaries.last().unwrap();
                let mut width = 8.0;
                bencher.iter_custom(|iterations| {
                    measure_updates(&mut boundary.app, iterations, |world| {
                        width = if width == 8.0 { 9.0 } else { 8.0 };
                        world.get_mut::<Node>(entity).unwrap().width = Val::Px(width);
                    })
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("one_change_per_boundary", node_count),
            &node_count,
            |bencher, _| {
                let mut distributed = layout_app(node_count, retained, shape);
                let entities: Vec<_> = distributed
                    .contained_leaves
                    .chunks(containment_size - 1)
                    .filter_map(|chunk| chunk.last().copied())
                    .collect();
                let mut width = 8.0;
                bencher.iter_custom(|iterations| {
                    measure_updates(&mut distributed.app, iterations, |world| {
                        width = if width == 8.0 { 9.0 } else { 8.0 };
                        for &entity in &entities {
                            world.get_mut::<Node>(entity).unwrap().width = Val::Px(width);
                        }
                    })
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("all_contained_leaves_change", node_count),
            &node_count,
            |bencher, _| {
                let mut all = layout_app(node_count, retained, shape);
                let entities = all.contained_leaves.clone();
                let mut width = 8.0;
                bencher.iter_custom(|iterations| {
                    measure_updates(&mut all.app, iterations, |world| {
                        width = if width == 8.0 { 9.0 } else { 8.0 };
                        for &entity in &entities {
                            world.get_mut::<Node>(entity).unwrap().width = Val::Px(width);
                        }
                    })
                });
            },
        );
    }
}

fn layout(c: &mut Criterion) {
    for retained in [false, true] {
        let renderer = if retained { "retained" } else { "stock" };

        let mut flat =
            c.benchmark_group(format!("ui_layout/{renderer}/{}", TreeShape::Flat.name()));
        for node_count in [100, 1_000, 10_000] {
            bench_scene(
                &mut flat,
                node_count,
                retained,
                TreeShape::Flat,
                true,
                false,
            );
        }
        flat.finish();

        for (shape, node_count, reparent) in [
            (TreeShape::Balanced, 10_000, false),
            (TreeShape::Forest, 10_000, true),
            (TreeShape::Contained(10), 10_000, false),
            (TreeShape::Contained(100), 10_000, false),
            (TreeShape::Contained(1_000), 10_000, false),
            (TreeShape::AbsoluteContained(100), 10_000, false),
            (TreeShape::Deep, 256, false),
        ] {
            let mut group = c.benchmark_group(format!("ui_layout/{renderer}/{}", shape.name()));
            bench_scene(&mut group, node_count, retained, shape, false, reparent);
            group.finish();
        }
    }
}

criterion_group!(benches, layout);
