use bevy_ecs::{prelude::*, schedule::Schedule, system::ScheduleSystem};
use criterion::{criterion_group, BenchmarkId, Criterion, Throughput};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

#[derive(Component)]
struct A;
#[derive(Component)]
struct B;
#[derive(Component)]
struct C;
#[derive(Component)]
struct D;
#[derive(Component)]
struct E;
#[derive(Component)]
struct F;
#[derive(Component)]
struct G;
#[derive(Component)]
struct H;

fn empty() {
    black_box(());
}

fn scan_one(query: Query<Entity, Changed<A>>) {
    black_box(query.iter().count());
}

fn scan_eight(
    query: Query<
        Entity,
        Or<(
            Changed<A>,
            Changed<B>,
            Changed<C>,
            Changed<D>,
            Changed<E>,
            Changed<F>,
            Changed<G>,
            Changed<H>,
        )>,
    >,
) {
    black_box(query.iter().count());
}

fn world(entity_count: usize) -> (World, Vec<Entity>) {
    let mut world = World::new();
    let entities = world
        .spawn_batch((0..entity_count).map(|_| (A, B, C, D, E, F, G, H)))
        .collect();
    (world, entities)
}

fn initialized_schedule<M>(
    world: &mut World,
    system: impl IntoScheduleConfigs<ScheduleSystem, M>,
) -> Schedule {
    let mut schedule = Schedule::default();
    schedule.add_systems(system);
    schedule.run(world);
    schedule
}

fn measure_runs(
    world: &mut World,
    schedule: &mut Schedule,
    iterations: u64,
    mut prepare: impl FnMut(&mut World),
) -> Duration {
    let mut elapsed = Duration::ZERO;
    for _ in 0..iterations {
        prepare(world);
        let start = Instant::now();
        schedule.run(world);
        elapsed += start.elapsed();
    }
    elapsed
}

fn bench_scan<M>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    entity_count: usize,
    name: &str,
    system: impl IntoScheduleConfigs<ScheduleSystem, M> + Clone,
) {
    group.bench_with_input(
        BenchmarkId::new(format!("{name}/quiet"), entity_count),
        &entity_count,
        |bencher, _| {
            let (mut quiet_world, _) = world(entity_count);
            let mut quiet_schedule = initialized_schedule(&mut quiet_world, system.clone());
            bencher.iter(|| quiet_schedule.run(&mut quiet_world));
        },
    );

    group.bench_with_input(
        BenchmarkId::new(format!("{name}/one_changed"), entity_count),
        &entity_count,
        |bencher, _| {
            let (mut one_world, one_entities) = world(entity_count);
            let mut one_schedule = initialized_schedule(&mut one_world, system.clone());
            let one_entity = *one_entities.last().unwrap();
            bencher.iter_custom(|iterations| {
                measure_runs(&mut one_world, &mut one_schedule, iterations, |world| {
                    world.get_mut::<A>(one_entity).unwrap().set_changed();
                })
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new(format!("{name}/all_changed"), entity_count),
        &entity_count,
        |bencher, _| {
            let (mut all_world, all_entities) = world(entity_count);
            let mut all_schedule = initialized_schedule(&mut all_world, system.clone());
            bencher.iter_custom(|iterations| {
                measure_runs(&mut all_world, &mut all_schedule, iterations, |world| {
                    for &entity in &all_entities {
                        world.get_mut::<A>(entity).unwrap().set_changed();
                    }
                })
            });
        },
    );
}

fn changed(c: &mut Criterion) {
    let mut group = c.benchmark_group("ui_changed_scan");

    for entity_count in [100, 1_000, 10_000] {
        group.throughput(Throughput::Elements(entity_count as u64));

        group.bench_with_input(
            BenchmarkId::new("schedule_only", entity_count),
            &entity_count,
            |bencher, _| {
                let (mut empty_world, _) = world(entity_count);
                let mut empty_schedule = initialized_schedule(&mut empty_world, empty);
                bencher.iter(|| empty_schedule.run(&mut empty_world));
            },
        );

        bench_scan(&mut group, entity_count, "one_input", scan_one);
        bench_scan(&mut group, entity_count, "eight_inputs", scan_eight);
    }

    group.finish();
}

criterion_group!(benches, changed);
