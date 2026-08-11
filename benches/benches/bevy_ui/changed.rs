use bevy_ecs::{prelude::*, schedule::Schedule, system::ScheduleSystem};
use criterion::{BenchmarkId, Criterion, criterion_group};
use std::hint::black_box;

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

fn world(entity_count: usize) -> World {
    let mut world = World::new();
    world.spawn_batch((0..entity_count).map(|_| (A, B, C, D, E, F, G, H)));
    world
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

fn changed(c: &mut Criterion) {
    let mut group = c.benchmark_group("ui_changed_scan");

    for entity_count in [100, 1_000, 10_000] {
        let mut empty_world = world(entity_count);
        let mut empty_schedule = initialized_schedule(&mut empty_world, empty);
        group.bench_with_input(
            BenchmarkId::new("schedule_only", entity_count),
            &entity_count,
            |bencher, _| bencher.iter(|| empty_schedule.run(&mut empty_world)),
        );

        let mut one_world = world(entity_count);
        let mut one_schedule = initialized_schedule(&mut one_world, scan_one);
        group.bench_with_input(
            BenchmarkId::new("one_input", entity_count),
            &entity_count,
            |bencher, _| bencher.iter(|| one_schedule.run(&mut one_world)),
        );

        let mut eight_world = world(entity_count);
        let mut eight_schedule = initialized_schedule(&mut eight_world, scan_eight);
        group.bench_with_input(
            BenchmarkId::new("eight_inputs", entity_count),
            &entity_count,
            |bencher, _| bencher.iter(|| eight_schedule.run(&mut eight_world)),
        );
    }

    group.finish();
}

criterion_group!(benches, changed);
