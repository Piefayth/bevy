//! Scope proof for replacing the final Core 2D writer from another crate.

use bevy_app::{App, Plugin};
use bevy_core_pipeline::{core_2d::Core2dPlugin, upscaling::upscaling, Core2d, Core2dSystems};
use bevy_ecs::schedule::{IntoScheduleConfigs, ScheduleCleanupPolicy};
use bevy_render::{extract_plugin::ExtractPlugin, RenderApp};

struct RetainedFinalWriterPlugin;

impl Plugin for RetainedFinalWriterPlugin {
    fn build(&self, app: &mut App) {
        let render_app = app
            .get_sub_app_mut(RenderApp)
            .expect("ExtractPlugin creates the render app");

        let removed = render_app
            .remove_systems_in_set(
                Core2d,
                upscaling,
                ScheduleCleanupPolicy::RemoveSetAndSystems,
            )
            .expect("Core2dPlugin creates the Core2d schedule");
        assert_eq!(removed, 1, "the stock final writer must exist exactly once");

        render_app.add_systems(
            Core2d,
            retained_final_writer.after(Core2dSystems::PostProcess),
        );
    }
}

fn retained_final_writer() {}

#[test]
fn an_external_plugin_can_replace_the_core_2d_final_writer() {
    let mut app = App::new();
    app.add_plugins((
        ExtractPlugin::default(),
        Core2dPlugin,
        RetainedFinalWriterPlugin,
    ));

    let render_app = app.sub_app_mut(RenderApp);
    let removed_replacement = render_app
        .remove_systems_in_set(
            Core2d,
            retained_final_writer,
            ScheduleCleanupPolicy::RemoveSetAndSystems,
        )
        .expect("the Core2d schedule remains mutable");
    assert_eq!(removed_replacement, 1);
}
