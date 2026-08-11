use bevy_app::{App, PostUpdate, TaskPoolPlugin};
use bevy_asset::{AssetApp, AssetPlugin};
use bevy_camera::{Camera, Camera2d, ComputedCameraValues, RenderTargetInfo, Viewport};
use bevy_ecs::prelude::*;
use bevy_image::ImagePlugin;
use bevy_math::UVec2;
use bevy_text::TextPlugin;
use bevy_time::TimePlugin;
use bevy_ui::{Node, UiPlugin, Val};
use bevy_ui_render_retained::RetainedUiMainWorldPlugin;
use criterion::{criterion_group, BenchmarkId, Criterion, Throughput};
use std::time::{Duration, Instant};

const TARGET_SIZE: UVec2 = UVec2::new(1024, 1024);

fn layout_app(node_count: usize, retained: bool) -> (App, Vec<Entity>) {
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

    let root = app
        .world_mut()
        .spawn(Node {
            width: Val::Px(TARGET_SIZE.x as f32),
            height: Val::Px(TARGET_SIZE.y as f32),
            ..Default::default()
        })
        .id();
    let mut nodes = Vec::with_capacity(node_count);
    nodes.push(root);
    for _ in 1..node_count {
        nodes.push(
            app.world_mut()
                .spawn((
                    Node {
                        width: Val::Px(8.0),
                        height: Val::Px(8.0),
                        ..Default::default()
                    },
                    ChildOf(root),
                ))
                .id(),
        );
    }

    app.world_mut().run_schedule(PostUpdate);
    (app, nodes)
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

fn layout_group(c: &mut Criterion, name: &str, retained: bool) {
    let mut group = c.benchmark_group(name);

    for node_count in [100, 1_000, 10_000] {
        group.throughput(Throughput::Elements(node_count as u64));

        let (mut quiet_app, _) = layout_app(node_count, retained);
        group.bench_with_input(
            BenchmarkId::new("quiet", node_count),
            &node_count,
            |bencher, _| {
                bencher
                    .iter_custom(|iterations| measure_updates(&mut quiet_app, iterations, |_| {}));
            },
        );

        let (mut localized_app, localized_nodes) = layout_app(node_count, retained);
        let localized = *localized_nodes.last().unwrap();
        let mut localized_width = 8.0;
        group.bench_with_input(
            BenchmarkId::new("localized_change", node_count),
            &node_count,
            |bencher, _| {
                bencher.iter_custom(|iterations| {
                    measure_updates(&mut localized_app, iterations, |world| {
                        localized_width = if localized_width == 8.0 { 9.0 } else { 8.0 };
                        world.get_mut::<Node>(localized).unwrap().width = Val::Px(localized_width);
                    })
                });
            },
        );

        let (mut full_app, full_nodes) = layout_app(node_count, retained);
        let mut full_width = 8.0;
        group.bench_with_input(
            BenchmarkId::new("full_change", node_count),
            &node_count,
            |bencher, _| {
                bencher.iter_custom(|iterations| {
                    measure_updates(&mut full_app, iterations, |world| {
                        full_width = if full_width == 8.0 { 9.0 } else { 8.0 };
                        for entity in full_nodes.iter().skip(1) {
                            world.get_mut::<Node>(*entity).unwrap().width = Val::Px(full_width);
                        }
                    })
                });
            },
        );
    }

    group.finish();
}

fn layout(c: &mut Criterion) {
    layout_group(c, "ui_layout", false);
    layout_group(c, "retained_ui_layout", true);
}

criterion_group!(benches, layout);
