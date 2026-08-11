//! Exact main-world gates for recursive Bevy UI work.

use bevy::{
    app::{App, Plugin, PostUpdate},
    ecs::{
        hierarchy::{ChildOf, Children},
        lifecycle::RemovedComponents,
        query::{Added, Changed, Or},
        schedule::{IntoScheduleConfigs, ScheduleCleanupPolicy},
        system::{Query, Res, SystemParam},
    },
    ui::{
        ui_geometry_system, ui_layout_system, ui_stack_system, CalculatedClip, ComputedNode,
        ComputedUiRenderTargetInfo, ContentSize, GlobalZIndex, IgnoreScroll, LayoutConfig, Node,
        Outline, OverrideClip, ScrollPosition, UiGlobalTransform, UiSystems, UiTransform, ZIndex,
    },
};
use core::sync::atomic::{AtomicU64, Ordering};

/// Main-world recursive work completed since startup.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RetainedUiMainWorldWork {
    /// Taffy layout computations executed.
    pub layout_runs: u64,
    /// Bevy derived-geometry placement walks executed.
    pub geometry_runs: u64,
    /// Bevy paint-stack rebuilds executed.
    pub stack_runs: u64,
    /// Bevy recursive clipping walks executed.
    pub clip_runs: u64,
}

/// Atomic main-world work counters for diagnostics, tests, and benchmarks.
#[derive(bevy::prelude::Resource, Default)]
pub struct RetainedUiMainWorldCounters {
    layout_runs: AtomicU64,
    geometry_runs: AtomicU64,
    stack_runs: AtomicU64,
    clip_runs: AtomicU64,
}

impl RetainedUiMainWorldCounters {
    /// Returns the current monotonic work counts.
    pub fn snapshot(&self) -> RetainedUiMainWorldWork {
        RetainedUiMainWorldWork {
            layout_runs: self.layout_runs.load(Ordering::Relaxed),
            geometry_runs: self.geometry_runs.load(Ordering::Relaxed),
            stack_runs: self.stack_runs.load(Ordering::Relaxed),
            clip_runs: self.clip_runs.load(Ordering::Relaxed),
        }
    }
}

/// Prevents Bevy's recursive layout, geometry, stack, and clipping walks when none of their
/// inputs changed.
///
/// Add this after [`bevy::ui::UiPlugin`]. Only the four named stock systems are replaced; other
/// systems placed in the same [`UiSystems`] sets keep their original schedule behavior.
#[derive(Default)]
pub struct RetainedUiMainWorldPlugin;

impl Plugin for RetainedUiMainWorldPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RetainedUiMainWorldCounters>();

        let removed_layout = app
            .remove_systems_in_set(
                PostUpdate,
                ui_layout_system,
                ScheduleCleanupPolicy::RemoveSystemsOnly,
            )
            .expect("UiPlugin must be added before RetainedUiMainWorldPlugin");
        assert_eq!(
            removed_layout, 1,
            "UiPlugin must contain exactly one stock layout system"
        );
        let removed_geometry = app
            .remove_systems_in_set(
                PostUpdate,
                ui_geometry_system,
                ScheduleCleanupPolicy::RemoveSystemsOnly,
            )
            .expect("UiPlugin must be added before RetainedUiMainWorldPlugin");
        assert_eq!(
            removed_geometry, 1,
            "UiPlugin must contain exactly one stock geometry system"
        );
        let removed_stack = app
            .remove_systems_in_set(
                PostUpdate,
                ui_stack_system,
                ScheduleCleanupPolicy::RemoveSystemsOnly,
            )
            .expect("UiPlugin must be added before RetainedUiMainWorldPlugin");
        assert_eq!(
            removed_stack, 1,
            "UiPlugin must contain exactly one stock stack system"
        );
        let removed_clipping = app
            .remove_systems_in_set(
                PostUpdate,
                bevy::ui::update::update_clipping_system,
                ScheduleCleanupPolicy::RemoveSystemsOnly,
            )
            .expect("UiPlugin must be added before RetainedUiMainWorldPlugin");
        assert_eq!(
            removed_clipping, 1,
            "UiPlugin must contain exactly one stock clipping system"
        );

        app.add_systems(
            PostUpdate,
            ui_layout_system
                .in_set(UiSystems::Layout)
                .ambiguous_with(bevy::sprite::update_text2d_layout)
                .run_if(layout_may_change),
        )
        .add_systems(
            PostUpdate,
            ui_geometry_system
                .in_set(UiSystems::Layout)
                .after(ui_layout_system)
                .ambiguous_with(bevy::sprite::update_text2d_layout)
                .run_if(geometry_may_change),
        )
        .add_systems(
            PostUpdate,
            ui_stack_system
                .in_set(UiSystems::Stack)
                .run_if(stack_may_change),
        )
        .add_systems(
            PostUpdate,
            bevy::ui::update::update_clipping_system
                .in_set(UiSystems::PostLayout)
                .run_if(clipping_may_change),
        );
    }
}

#[derive(SystemParam)]
struct RemovedLayoutInputs<'w, 's> {
    node: RemovedComponents<'w, 's, Node>,
    content_size: RemovedComponents<'w, 's, ContentSize>,
    target: RemovedComponents<'w, 's, ComputedUiRenderTargetInfo>,
    children: RemovedComponents<'w, 's, Children>,
    parent: RemovedComponents<'w, 's, ChildOf>,
}

impl RemovedLayoutInputs<'_, '_> {
    fn any(&mut self) -> bool {
        let Self {
            node,
            content_size,
            target,
            children,
            parent,
        } = self;
        node.read()
            .chain(content_size.read())
            .chain(target.read())
            .chain(children.read())
            .chain(parent.read())
            .next()
            .is_some()
    }
}

fn layout_may_change(
    changed_core: Query<
        (),
        Or<(
            Changed<Node>,
            Changed<ContentSize>,
            Changed<ComputedUiRenderTargetInfo>,
            Changed<Children>,
            Changed<ChildOf>,
        )>,
    >,
    mut removed: RemovedLayoutInputs,
    counters: Res<RetainedUiMainWorldCounters>,
) -> bool {
    let changed = !changed_core.is_empty() || removed.any();
    if changed {
        counters.layout_runs.fetch_add(1, Ordering::Relaxed);
    }
    changed
}

#[derive(SystemParam)]
struct RemovedGeometryInputs<'w, 's> {
    node: RemovedComponents<'w, 's, Node>,
    content_size: RemovedComponents<'w, 's, ContentSize>,
    target: RemovedComponents<'w, 's, ComputedUiRenderTargetInfo>,
    transform: RemovedComponents<'w, 's, UiTransform>,
    config: RemovedComponents<'w, 's, LayoutConfig>,
    outline: RemovedComponents<'w, 's, Outline>,
    scroll: RemovedComponents<'w, 's, ScrollPosition>,
    ignore_scroll: RemovedComponents<'w, 's, IgnoreScroll>,
    children: RemovedComponents<'w, 's, Children>,
    parent: RemovedComponents<'w, 's, ChildOf>,
}

impl RemovedGeometryInputs<'_, '_> {
    fn any(&mut self) -> bool {
        let Self {
            node,
            content_size,
            target,
            transform,
            config,
            outline,
            scroll,
            ignore_scroll,
            children,
            parent,
        } = self;
        node.read()
            .chain(content_size.read())
            .chain(target.read())
            .chain(transform.read())
            .chain(config.read())
            .chain(outline.read())
            .chain(scroll.read())
            .chain(ignore_scroll.read())
            .chain(children.read())
            .chain(parent.read())
            .next()
            .is_some()
    }
}

fn geometry_may_change(
    changed_core: Query<
        (),
        Or<(
            Changed<Node>,
            Changed<ContentSize>,
            Changed<ComputedUiRenderTargetInfo>,
            Changed<UiTransform>,
            Changed<Children>,
            Changed<ChildOf>,
        )>,
    >,
    changed_optional: Query<
        (),
        Or<(
            Changed<LayoutConfig>,
            Changed<Outline>,
            Changed<ScrollPosition>,
            Changed<IgnoreScroll>,
        )>,
    >,
    mut removed: RemovedGeometryInputs,
    counters: Res<RetainedUiMainWorldCounters>,
) -> bool {
    let changed = !changed_core.is_empty() || !changed_optional.is_empty() || removed.any();
    if changed {
        counters.geometry_runs.fetch_add(1, Ordering::Relaxed);
    }
    changed
}

#[derive(SystemParam)]
struct RemovedStackInputs<'w, 's> {
    node: RemovedComponents<'w, 's, Node>,
    global_z: RemovedComponents<'w, 's, GlobalZIndex>,
    local_z: RemovedComponents<'w, 's, ZIndex>,
    children: RemovedComponents<'w, 's, Children>,
    parent: RemovedComponents<'w, 's, ChildOf>,
}

impl RemovedStackInputs<'_, '_> {
    fn any(&mut self) -> bool {
        let Self {
            node,
            global_z,
            local_z,
            children,
            parent,
        } = self;
        node.read()
            .chain(global_z.read())
            .chain(local_z.read())
            .chain(children.read())
            .chain(parent.read())
            .next()
            .is_some()
    }
}

fn stack_may_change(
    added_nodes: Query<(), Added<Node>>,
    changed_order: Query<
        (),
        Or<(
            Changed<GlobalZIndex>,
            Changed<ZIndex>,
            Changed<Children>,
            Changed<ChildOf>,
        )>,
    >,
    mut removed: RemovedStackInputs,
    counters: Res<RetainedUiMainWorldCounters>,
) -> bool {
    let changed = !added_nodes.is_empty() || !changed_order.is_empty() || removed.any();
    if changed {
        counters.stack_runs.fetch_add(1, Ordering::Relaxed);
    }
    changed
}

#[derive(SystemParam)]
struct RemovedClippingInputs<'w, 's> {
    node: RemovedComponents<'w, 's, Node>,
    computed: RemovedComponents<'w, 's, ComputedNode>,
    transform: RemovedComponents<'w, 's, UiGlobalTransform>,
    clip: RemovedComponents<'w, 's, CalculatedClip>,
    override_clip: RemovedComponents<'w, 's, OverrideClip>,
    children: RemovedComponents<'w, 's, Children>,
    parent: RemovedComponents<'w, 's, ChildOf>,
}

impl RemovedClippingInputs<'_, '_> {
    fn any(&mut self) -> bool {
        let Self {
            node,
            computed,
            transform,
            clip,
            override_clip,
            children,
            parent,
        } = self;
        node.read()
            .chain(computed.read())
            .chain(transform.read())
            .chain(clip.read())
            .chain(override_clip.read())
            .chain(children.read())
            .chain(parent.read())
            .next()
            .is_some()
    }
}

fn clipping_may_change(
    changed: Query<
        (),
        Or<(
            Changed<Node>,
            Changed<ComputedNode>,
            Changed<UiGlobalTransform>,
            Changed<CalculatedClip>,
            Changed<OverrideClip>,
            Changed<Children>,
            Changed<ChildOf>,
        )>,
    >,
    mut removed: RemovedClippingInputs,
    counters: Res<RetainedUiMainWorldCounters>,
) -> bool {
    let changed = !changed.is_empty() || removed.any();
    if changed {
        counters.clip_runs.fetch_add(1, Ordering::Relaxed);
    }
    changed
}
