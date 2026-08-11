//! Exact schedule proofs for main-world UI quiescence.

use bevy::{
    app::{App, PostUpdate, TaskPoolPlugin},
    asset::{AssetApp, AssetPlugin},
    camera::{Camera, Camera2d, ComputedCameraValues, RenderTargetInfo, Viewport},
    color::Color,
    ecs::{resource::Resource, schedule::IntoScheduleConfigs, system::ResMut},
    image::{ImagePlugin, TextureAtlasLayout},
    math::{UVec2, Vec2},
    text::TextPlugin,
    time::TimePlugin,
    ui::{
        BackgroundColor, ComputedNode, ComputedStackIndex, Node, OverrideClip, UiGlobalTransform,
        UiStack, UiSystems, UiTransform, Val, ZIndex,
    },
};
use bevy_ui_render_retained::{
    RetainedUiMainWorldCounters, RetainedUiMainWorldPlugin, RetainedUiMainWorldWork,
};

const TARGET_SIZE: UVec2 = UVec2::new(64, 64);

#[derive(Resource, Default)]
struct Runs {
    layout: u32,
    stack: u32,
    post_layout: u32,
}

fn count_layout(mut runs: ResMut<Runs>) {
    runs.layout += 1;
}

fn count_stack(mut runs: ResMut<Runs>) {
    runs.stack += 1;
}

fn count_post_layout(mut runs: ResMut<Runs>) {
    runs.post_layout += 1;
}

fn retained_work(app: &App) -> RetainedUiMainWorldWork {
    app.world()
        .resource::<RetainedUiMainWorldCounters>()
        .snapshot()
}

fn test_app() -> (App, bevy::ecs::entity::Entity, bevy::ecs::entity::Entity) {
    let mut app = App::new();
    app.add_plugins((
        TaskPoolPlugin::default(),
        TimePlugin,
        AssetPlugin::default(),
        ImagePlugin::default(),
        TextPlugin,
        bevy::ui::UiPlugin,
        RetainedUiMainWorldPlugin,
    ))
    .init_asset::<TextureAtlasLayout>()
    .init_resource::<Runs>()
    .add_systems(PostUpdate, count_layout.in_set(UiSystems::Layout))
    .add_systems(PostUpdate, count_stack.in_set(UiSystems::Stack))
    .add_systems(PostUpdate, count_post_layout.in_set(UiSystems::PostLayout));

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
            width: Val::Px(64.0),
            height: Val::Px(64.0),
            ..Default::default()
        })
        .id();
    let leaf = app
        .world_mut()
        .spawn((
            Node {
                width: Val::Px(8.0),
                height: Val::Px(8.0),
                ..Default::default()
            },
            BackgroundColor(Color::WHITE),
            bevy::ecs::hierarchy::ChildOf(root),
        ))
        .id();

    app.world_mut().run_schedule(PostUpdate);
    *app.world_mut().resource_mut::<Runs>() = Runs::default();
    (app, root, leaf)
}

#[test]
fn static_and_paint_only_ui_skip_layout_and_stack_walks() {
    let (mut app, _, leaf) = test_app();
    let initial = retained_work(&app);

    app.world_mut().run_schedule(PostUpdate);
    assert_eq!(retained_work(&app), initial);
    assert_eq!(
        (
            app.world().resource::<Runs>().layout,
            app.world().resource::<Runs>().stack,
            app.world().resource::<Runs>().post_layout,
        ),
        (1, 1, 1),
        "custom systems in the public UI sets must remain ungated"
    );

    app.world_mut().get_mut::<BackgroundColor>(leaf).unwrap().0 = Color::BLACK;
    app.world_mut().run_schedule(PostUpdate);
    assert_eq!(retained_work(&app), initial);
    assert_eq!(
        (
            app.world().resource::<Runs>().layout,
            app.world().resource::<Runs>().stack,
            app.world().resource::<Runs>().post_layout,
        ),
        (2, 2, 2)
    );
}

#[test]
fn layout_placement_and_stack_inputs_wake_only_the_required_domain() {
    let (mut app, _, leaf) = test_app();

    let before = retained_work(&app);
    app.world_mut().get_mut::<Node>(leaf).unwrap().width = Val::Px(12.0);
    app.world_mut().run_schedule(PostUpdate);
    let after = retained_work(&app);
    assert_eq!(after.layout_runs, before.layout_runs + 1);
    assert_eq!(after.geometry_runs, before.geometry_runs + 1);
    assert_eq!(after.stack_runs, before.stack_runs);
    assert_eq!(after.clip_runs, before.clip_runs + 1);
    assert_eq!(
        app.world().get::<ComputedNode>(leaf).unwrap().size().x,
        12.0
    );

    let before = after;
    app.world_mut()
        .get_mut::<UiTransform>(leaf)
        .unwrap()
        .translation
        .x = Val::Px(3.0);
    app.world_mut().run_schedule(PostUpdate);
    let after = retained_work(&app);
    assert_eq!(after.layout_runs, before.layout_runs);
    assert_eq!(after.geometry_runs, before.geometry_runs + 1);
    assert_eq!(after.stack_runs, before.stack_runs);
    assert_eq!(after.clip_runs, before.clip_runs + 1);
    assert_ne!(
        app.world()
            .get::<UiGlobalTransform>(leaf)
            .unwrap()
            .translation,
        Vec2::ZERO
    );

    let before = after;
    app.world_mut().entity_mut(leaf).insert(ZIndex(4));
    app.world_mut().run_schedule(PostUpdate);
    let after = retained_work(&app);
    assert_eq!(after.layout_runs, before.layout_runs);
    assert_eq!(after.geometry_runs, before.geometry_runs);
    assert_eq!(after.stack_runs, before.stack_runs + 1);
    assert_eq!(after.clip_runs, before.clip_runs);
    assert!(app.world().get::<ComputedStackIndex>(leaf).unwrap().0 > 0);
}

#[test]
fn hierarchy_changes_wake_layout_and_stack() {
    let (mut app, root, leaf) = test_app();
    let new_root = app
        .world_mut()
        .spawn(Node {
            width: Val::Px(64.0),
            height: Val::Px(64.0),
            ..Default::default()
        })
        .id();
    app.world_mut().run_schedule(PostUpdate);
    let before = retained_work(&app);

    app.world_mut()
        .entity_mut(leaf)
        .insert(bevy::ecs::hierarchy::ChildOf(new_root));
    app.world_mut().run_schedule(PostUpdate);

    let after = retained_work(&app);
    assert_eq!(after.layout_runs, before.layout_runs + 1);
    assert_eq!(after.geometry_runs, before.geometry_runs + 1);
    assert_eq!(after.stack_runs, before.stack_runs + 1);
    assert_eq!(after.clip_runs, before.clip_runs + 1);
    assert_ne!(root, new_root);
}

#[test]
fn clipping_only_inputs_do_not_wake_layout_or_stack() {
    let (mut app, _, leaf) = test_app();
    let before = retained_work(&app);

    app.world_mut().entity_mut(leaf).insert(OverrideClip);
    app.world_mut().run_schedule(PostUpdate);

    let after = retained_work(&app);
    assert_eq!(after.layout_runs, before.layout_runs);
    assert_eq!(after.geometry_runs, before.geometry_runs);
    assert_eq!(after.stack_runs, before.stack_runs);
    assert_eq!(after.clip_runs, before.clip_runs + 1);
}

#[test]
fn node_removal_wakes_every_recursive_domain_and_cleans_the_stack() {
    let (mut app, _, leaf) = test_app();
    let before = retained_work(&app);

    app.world_mut().entity_mut(leaf).despawn();
    app.world_mut().run_schedule(PostUpdate);

    let after = retained_work(&app);
    assert_eq!(after.layout_runs, before.layout_runs + 1);
    assert_eq!(after.geometry_runs, before.geometry_runs + 1);
    assert_eq!(after.stack_runs, before.stack_runs + 1);
    assert_eq!(after.clip_runs, before.clip_runs + 1);
    assert!(!app.world().resource::<UiStack>().uinodes.contains(&leaf));
}

#[test]
fn every_removal_is_consumed_in_the_frame_that_handles_it() {
    let (mut app, root, first) = test_app();
    let second = app.world_mut().spawn(Node::default()).id();
    app.world_mut().entity_mut(root).add_child(second);
    app.world_mut().run_schedule(PostUpdate);

    app.world_mut().despawn(first);
    app.world_mut().despawn(second);
    app.world_mut().run_schedule(PostUpdate);
    let after_removals = retained_work(&app);

    app.world_mut().run_schedule(PostUpdate);
    assert_eq!(retained_work(&app), after_removals);
}
