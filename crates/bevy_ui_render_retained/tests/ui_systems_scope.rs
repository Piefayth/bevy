//! Scope proof for externally controlling Bevy UI system sets.

use bevy_app::{App, Plugin, PostUpdate};
use bevy_ecs::prelude::*;
use bevy_ui::UiSystems;

#[derive(Resource, Default)]
struct Gate(bool);

#[derive(Resource, Default)]
struct Runs(u32);

struct ExistingUiPlugin;

impl Plugin for ExistingUiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Runs>()
            .add_systems(PostUpdate, count_run.in_set(UiSystems::Layout));
    }
}

struct ExternalGatePlugin;

impl Plugin for ExternalGatePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Gate>()
            .configure_sets(PostUpdate, UiSystems::Layout.run_if(gate_is_open));
    }
}

fn count_run(mut runs: ResMut<Runs>) {
    runs.0 += 1;
}

fn gate_is_open(gate: Res<Gate>) -> bool {
    gate.0
}

#[test]
fn an_external_plugin_can_gate_an_existing_ui_system_set() {
    let mut app = App::new();
    app.add_plugins((ExistingUiPlugin, ExternalGatePlugin));

    app.update();
    assert_eq!(app.world().resource::<Runs>().0, 0);

    app.world_mut().resource_mut::<Gate>().0 = true;
    app.update();
    assert_eq!(app.world().resource::<Runs>().0, 1);

    app.world_mut().resource_mut::<Gate>().0 = false;
    app.update();
    assert_eq!(app.world().resource::<Runs>().0, 1);
}
