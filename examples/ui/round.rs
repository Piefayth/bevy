//! This comment removes an annoying warning.
use bevy::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(Update, update)
        .run();
}

#[derive(Component)]
struct Node1;
#[derive(Component)]
struct Node2;

#[derive(Resource)]
struct NodeAnimation((f32, f32));

fn setup(mut commands: Commands) {
    commands.spawn(Camera2d);

    commands.spawn(Node {
        width: Val::Px(6.0),
        height: Val::Px(6.0),
        top: Val::Px(6.0),
        position_type: PositionType::Absolute,
        ..default()
     })
    .insert(BackgroundColor(Color::WHITE.into()))
    .insert(Node1);

    commands.spawn(Node {
        width: Val::Px(6.0),
        height: Val::Px(6.0),
        top: Val::Px(6.0),
        position_type: PositionType::Absolute,
        ..default()
     })
    .insert(BackgroundColor(Color::BLACK.into()))
    .insert(Node2);

    commands.insert_resource(UiScale(20.0));
    commands.insert_resource(NodeAnimation((0.0, 1.0)));
}

fn cubic_ease_in_out(factor: f32) -> f32 {
    if factor < 0.5 {
        4. * factor * factor * factor
    } else {
        1. - (-2. * factor + 2.).powf(3.) / 2.
    }
}

fn position(factor: f32) -> f32 {
    cubic_ease_in_out(factor) * 30.
}

fn update(
    mut node1_q: Query<&mut Node, With<Node1>>,
    mut node2_q: Query<&mut GlobalTransform, With<Node2>>,
    mut node_animation: ResMut<NodeAnimation>,
    time: Res<Time>,
) {
    let (factor, direction) = &mut node_animation.0;
    *factor += time.delta_secs() * *direction * 0.3;
    if *factor > 1. {
        *direction = -1.;
    } else if *factor < 0. {
        *direction = 1.;
    }

    for mut style in node1_q.iter_mut() {
        style.left = Val::Px(position(*factor));
    }

    for mut t in node2_q.iter_mut() {
        let mut transform = t.compute_transform();
        transform.translation.x = position(*factor) + 30.;
        *t = transform.into();
    }
}
