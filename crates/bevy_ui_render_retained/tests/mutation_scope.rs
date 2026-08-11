//! Scope proof for O(changes) UI invalidation.

use bevy_ecs::{
    lifecycle::HookContext,
    prelude::*,
    world::{DeferredWorld, World},
};

#[derive(Component)]
struct Value(u32);

#[derive(Resource, Default)]
struct HookRuns {
    inserts: u32,
    discards: u32,
}

fn on_insert(mut world: DeferredWorld, _: HookContext) {
    world.resource_mut::<HookRuns>().inserts += 1;
}

fn on_discard(mut world: DeferredWorld, _: HookContext) {
    world.resource_mut::<HookRuns>().discards += 1;
}

#[test]
fn mutable_component_writes_have_no_public_push_invalidation_hook() {
    let mut world = World::new();
    world.init_resource::<HookRuns>();
    world
        .register_component_hooks::<Value>()
        .on_insert(on_insert)
        .on_discard(on_discard);

    let entity = world.spawn(Value(0)).id();
    *world.resource_mut::<HookRuns>() = HookRuns::default();

    world.get_mut::<Value>(entity).unwrap().0 = 1;
    assert_eq!(world.resource::<HookRuns>().inserts, 0);
    assert_eq!(world.resource::<HookRuns>().discards, 0);

    world.entity_mut(entity).insert(Value(2));
    assert_eq!(world.resource::<HookRuns>().inserts, 1);
    assert_eq!(world.resource::<HookRuns>().discards, 1);
}
