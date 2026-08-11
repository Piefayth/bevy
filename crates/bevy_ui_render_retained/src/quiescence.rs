//! Exact main-world gates for recursive Bevy UI work.

use bevy::{
    app::{App, Plugin, PostUpdate},
    ecs::{
        hierarchy::{ChildOf, Children},
        lifecycle::RemovedComponents,
        query::{Added, Changed, Or},
        schedule::{IntoScheduleConfigs, ScheduleCleanupPolicy},
        system::{Query, Res, ResMut, SystemParam},
    },
    ui::{
        ui_geometry_system, ui_layout_system, ui_stack_system, ComputedNode,
        ComputedUiRenderTargetInfo, ContentSize, GlobalZIndex, IgnoreScroll, LayoutConfig,
        LayoutContainment, Node, Outline, OverrideClip, ScrollPosition, UiGlobalTransform,
        UiSystems, UiTransform, ZIndex,
    },
};
use core::sync::atomic::{AtomicU64, Ordering};

fn drain_removed<T: bevy::ecs::component::Component>(removed: &mut RemovedComponents<T>) -> bool {
    removed.read().count() != 0
}

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
        app.init_resource::<RetainedUiMainWorldCounters>()
            .init_resource::<UiDirtyDomains>();

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
            classify_ui_changes
                .in_set(UiSystems::Layout)
                .before(ui_layout_system)
                .before(ui_geometry_system)
                .before(ui_stack_system)
                .before(bevy::ui::update::update_clipping_system),
        )
        .add_systems(
            PostUpdate,
            ui_layout_system
                .in_set(UiSystems::Layout)
                .ambiguous_with(bevy::sprite::update_text2d_layout)
                .run_if(layout_is_dirty),
        )
        .add_systems(
            PostUpdate,
            ui_geometry_system
                .in_set(UiSystems::Layout)
                .after(ui_layout_system)
                .ambiguous_with(bevy::sprite::update_text2d_layout)
                .run_if(geometry_is_dirty),
        )
        .add_systems(
            PostUpdate,
            ui_stack_system
                .in_set(UiSystems::Stack)
                .run_if(stack_is_dirty),
        )
        .add_systems(
            PostUpdate,
            bevy::ui::update::update_clipping_system
                .in_set(UiSystems::PostLayout)
                .run_if(clipping_may_change),
        );
    }
}

#[derive(bevy::prelude::Resource, Default)]
struct UiDirtyDomains {
    layout: bool,
    geometry: bool,
    stack: bool,
}

#[derive(SystemParam)]
struct RemovedUiInputs<'w, 's> {
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
    containment: RemovedComponents<'w, 's, LayoutContainment>,
    global_z: RemovedComponents<'w, 's, GlobalZIndex>,
    local_z: RemovedComponents<'w, 's, ZIndex>,
}

fn classify_ui_changes(
    layout_changes: Query<
        (),
        Or<(
            Changed<Node>,
            Changed<ContentSize>,
            Changed<ComputedUiRenderTargetInfo>,
            Changed<Children>,
            Changed<ChildOf>,
            Changed<LayoutContainment>,
        )>,
    >,
    geometry_changes: Query<
        (),
        Or<(
            Changed<UiTransform>,
            Changed<LayoutConfig>,
            Changed<Outline>,
            Changed<ScrollPosition>,
            Changed<IgnoreScroll>,
        )>,
    >,
    added_nodes: Query<(), Added<Node>>,
    stack_changes: Query<
        (),
        Or<(
            Changed<GlobalZIndex>,
            Changed<ZIndex>,
            Changed<Children>,
            Changed<ChildOf>,
        )>,
    >,
    mut removed: RemovedUiInputs,
    mut domains: ResMut<UiDirtyDomains>,
    counters: Res<RetainedUiMainWorldCounters>,
) {
    let removed_node = drain_removed(&mut removed.node);
    let removed_content = drain_removed(&mut removed.content_size);
    let removed_target = drain_removed(&mut removed.target);
    let removed_transform = drain_removed(&mut removed.transform);
    let removed_config = drain_removed(&mut removed.config);
    let removed_outline = drain_removed(&mut removed.outline);
    let removed_scroll = drain_removed(&mut removed.scroll);
    let removed_ignore_scroll = drain_removed(&mut removed.ignore_scroll);
    let removed_children = drain_removed(&mut removed.children);
    let removed_parent = drain_removed(&mut removed.parent);
    let removed_containment = drain_removed(&mut removed.containment);
    let removed_global_z = drain_removed(&mut removed.global_z);
    let removed_local_z = drain_removed(&mut removed.local_z);

    let layout_removed = removed_node
        | removed_content
        | removed_target
        | removed_children
        | removed_parent
        | removed_containment;
    let layout = !layout_changes.is_empty() || layout_removed;

    let geometry_removed = layout_removed
        | removed_transform
        | removed_config
        | removed_outline
        | removed_scroll
        | removed_ignore_scroll;
    let geometry = layout || !geometry_changes.is_empty() || geometry_removed;

    let stack = !added_nodes.is_empty()
        || !stack_changes.is_empty()
        || removed_node
        || removed_global_z
        || removed_local_z
        || removed_children
        || removed_parent;

    *domains = UiDirtyDomains {
        layout,
        geometry,
        stack,
    };

    if layout {
        counters.layout_runs.fetch_add(1, Ordering::Relaxed);
    }
    if geometry {
        counters.geometry_runs.fetch_add(1, Ordering::Relaxed);
    }
    if stack {
        counters.stack_runs.fetch_add(1, Ordering::Relaxed);
    }
}

fn layout_is_dirty(domains: Res<UiDirtyDomains>) -> bool {
    domains.layout
}

fn geometry_is_dirty(domains: Res<UiDirtyDomains>) -> bool {
    domains.geometry
}

fn stack_is_dirty(domains: Res<UiDirtyDomains>) -> bool {
    domains.stack
}

#[derive(SystemParam)]
struct RemovedClippingInputs<'w, 's> {
    node: RemovedComponents<'w, 's, Node>,
    computed: RemovedComponents<'w, 's, ComputedNode>,
    transform: RemovedComponents<'w, 's, UiGlobalTransform>,
    override_clip: RemovedComponents<'w, 's, OverrideClip>,
    children: RemovedComponents<'w, 's, Children>,
    parent: RemovedComponents<'w, 's, ChildOf>,
}

impl RemovedClippingInputs<'_, '_> {
    fn any(&mut self) -> bool {
        drain_removed(&mut self.node)
            | drain_removed(&mut self.computed)
            | drain_removed(&mut self.transform)
            | drain_removed(&mut self.override_clip)
            | drain_removed(&mut self.children)
            | drain_removed(&mut self.parent)
    }
}

fn clipping_may_change(
    changed: Query<
        (),
        Or<(
            Changed<Node>,
            Changed<ComputedNode>,
            Changed<UiGlobalTransform>,
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
