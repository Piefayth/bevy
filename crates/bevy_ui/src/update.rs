//! This module contains systems that update the UI when something changes

use crate::{
    experimental::{UiChildren, UiRootNodes},
    ui_transform::UiGlobalTransform,
    CalculatedClip, ComputedUiRenderTargetInfo, ComputedUiTargetCamera, DefaultUiCamera, Display,
    Node, OverrideClip, PaintContainment, UiScale, UiTargetCamera,
};

use super::ComputedNode;
use bevy_app::Propagate;
use bevy_camera::Camera;
use bevy_ecs::{
    entity::{Entity, EntityHashMap, EntityHashSet},
    hierarchy::{ChildOf, Children},
    lifecycle::RemovedComponents,
    query::{Changed, Has, Or, With},
    system::{Commands, Local, Query, Res, SystemParam},
};
use bevy_math::{Rect, UVec2};

/// The complete subset of [`Node`] read by clipping.
///
/// Comparing it prevents unrelated fields, such as border radius, from invalidating a subtree.
#[derive(Clone, Copy, PartialEq)]
struct ClipStyle {
    display: Display,
    overflow: crate::Overflow,
    overflow_clip_margin: crate::OverflowClipMargin,
}

impl From<&Node> for ClipStyle {
    fn from(node: &Node) -> Self {
        Self {
            display: node.display,
            overflow: node.overflow,
            overflow_clip_margin: node.overflow_clip_margin,
        }
    }
}

#[derive(SystemParam)]
#[doc(hidden)]
pub struct ClippingChanges<'w, 's> {
    clip_styles: Local<'s, EntityHashMap<ClipStyle>>,
    nodes: Query<'w, 's, (Entity, &'static Node), Changed<Node>>,
    subtrees: Query<
        'w,
        's,
        Entity,
        (
            With<Node>,
            Or<(
                Changed<ComputedNode>,
                Changed<UiGlobalTransform>,
                Changed<OverrideClip>,
                Changed<PaintContainment>,
            )>,
        ),
    >,
    changed_children: Query<'w, 's, Entity, Changed<Children>>,
    changed_parent: Query<'w, 's, Entity, Changed<ChildOf>>,
    removed_node: RemovedComponents<'w, 's, Node>,
    removed_computed: RemovedComponents<'w, 's, ComputedNode>,
    removed_transform: RemovedComponents<'w, 's, UiGlobalTransform>,
    removed_override: RemovedComponents<'w, 's, OverrideClip>,
    removed_containment: RemovedComponents<'w, 's, PaintContainment>,
    removed_children: RemovedComponents<'w, 's, Children>,
    removed_parent: RemovedComponents<'w, 's, ChildOf>,
}

/// Updates clipping for nodes whose clipping inputs may have changed.
pub fn update_clipping_system(
    mut commands: Commands,
    mut node_query: Query<(
        &Node,
        &ComputedNode,
        &UiGlobalTransform,
        Option<&mut CalculatedClip>,
        Option<&mut DerivedClipState>,
        Has<OverrideClip>,
        Has<PaintContainment>,
    )>,
    ui_children: UiChildren,
    mut changes: ClippingChanges,
    mut dirty_subtrees: Local<EntityHashSet>,
    mut subtree_roots: Local<Vec<Entity>>,
) {
    dirty_subtrees.clear();
    subtree_roots.clear();
    dirty_subtrees.extend(changes.subtrees.iter());
    dirty_subtrees.extend(changes.changed_children.iter());
    dirty_subtrees.extend(changes.changed_parent.iter());
    dirty_subtrees.extend(changes.removed_computed.read());
    dirty_subtrees.extend(changes.removed_transform.read());
    dirty_subtrees.extend(changes.removed_override.read());
    dirty_subtrees.extend(changes.removed_containment.read());
    dirty_subtrees.extend(changes.removed_children.read());
    dirty_subtrees.extend(changes.removed_parent.read());
    for (entity, node) in changes.nodes.iter() {
        let style = ClipStyle::from(node);
        if changes.clip_styles.insert(entity, style) != Some(style) {
            dirty_subtrees.insert(entity);
        }
    }
    for entity in changes.removed_node.read() {
        changes.clip_styles.remove(&entity);
        dirty_subtrees.remove(&entity);
    }

    for &entity in dirty_subtrees.iter() {
        let mut ancestor = ui_children.get_parent(entity);
        let mut covered_by_ancestor = false;
        while let Some(parent) = ancestor {
            if dirty_subtrees.contains(&parent) {
                covered_by_ancestor = true;
                break;
            }
            ancestor = ui_children.get_parent(parent);
        }
        if !covered_by_ancestor {
            subtree_roots.push(entity);
        }
    }

    for entity in subtree_roots.drain(..) {
        let inherited_clip = inherited_clip(entity, &ui_children, &mut node_query);
        update_clipping(
            &mut commands,
            &ui_children,
            &mut node_query,
            entity,
            inherited_clip,
        );
    }
}

fn inherited_clip(
    entity: Entity,
    ui_children: &UiChildren,
    node_query: &mut Query<(
        &Node,
        &ComputedNode,
        &UiGlobalTransform,
        Option<&mut CalculatedClip>,
        Option<&mut DerivedClipState>,
        Has<OverrideClip>,
        Has<PaintContainment>,
    )>,
) -> ClipState {
    let Some(parent) = ui_children.get_parent(entity) else {
        return ClipState::default();
    };
    let Ok((node, computed_node, transform, _, state, _, has_containment)) =
        node_query.get_mut(parent)
    else {
        return ClipState::default();
    };
    children_clip(
        state.as_deref().copied().unwrap_or_default().0,
        node,
        computed_node,
        transform,
        has_containment,
    )
}

fn update_clipping(
    commands: &mut Commands,
    ui_children: &UiChildren,
    node_query: &mut Query<(
        &Node,
        &ComputedNode,
        &UiGlobalTransform,
        Option<&mut CalculatedClip>,
        Option<&mut DerivedClipState>,
        Has<OverrideClip>,
        Has<PaintContainment>,
    )>,
    entity: Entity,
    mut state: ClipState,
) {
    let Ok((
        node,
        computed_node,
        transform,
        maybe_calculated_clip,
        maybe_derived_state,
        has_override_clip,
        has_containment,
    )) = node_query.get_mut(entity)
    else {
        return;
    };

    if has_override_clip {
        state.clear_overridable();
    }
    if node.display == Display::None {
        state.hide();
    }

    match (maybe_calculated_clip, state.effective()) {
        (Some(mut calculated), Some((clip, paint_clip))) => {
            if calculated.clip != clip || calculated.paint_clip != paint_clip {
                *calculated = CalculatedClip { clip, paint_clip };
            }
        }
        (Some(_), None) => {
            commands.entity(entity).remove::<CalculatedClip>();
        }
        (None, Some((clip, paint_clip))) => {
            commands
                .entity(entity)
                .try_insert(CalculatedClip { clip, paint_clip });
        }
        (None, None) => {}
    }

    match (maybe_derived_state, state == ClipState::default()) {
        (Some(_), true) => {
            commands.entity(entity).remove::<DerivedClipState>();
        }
        (Some(mut previous), false) if previous.0 != state => previous.0 = state,
        (Some(_), false) | (None, true) => {}
        (None, false) => {
            commands.entity(entity).try_insert(DerivedClipState(state));
        }
    }

    let children_clip = children_clip(state, node, computed_node, transform, has_containment);

    for child in ui_children.iter_ui_children(entity) {
        update_clipping(commands, ui_children, node_query, child, children_clip);
    }
}

#[derive(Clone, Copy, Default, PartialEq)]
struct ClipAxis {
    inherited: Option<Rect>,
    containment: Option<Rect>,
}

impl ClipAxis {
    fn effective(self) -> Option<Rect> {
        match (self.inherited, self.containment) {
            (Some(inherited), Some(containment)) => Some(inherited.intersect(containment)),
            (Some(clip), None) | (None, Some(clip)) => Some(clip),
            (None, None) => None,
        }
    }

    fn contain(&mut self, clip: Rect) {
        self.containment = Some(
            self.containment
                .map_or(clip, |current| current.intersect(clip)),
        );
    }

    fn clip(&mut self, clip: Rect) {
        self.inherited = Some(
            self.inherited
                .map_or(clip, |current| current.intersect(clip)),
        );
    }
}

#[derive(Clone, Copy, Default, PartialEq)]
struct ClipState {
    global: ClipAxis,
    paint: ClipAxis,
}

impl ClipState {
    fn effective(self) -> Option<(Rect, Rect)> {
        match (self.global.effective(), self.paint.effective()) {
            (Some(global), Some(paint)) => Some((global, paint)),
            (None, None) => None,
            _ => unreachable!("global and paint clip presence must match"),
        }
    }

    fn clear_overridable(&mut self) {
        self.global.inherited = None;
        self.paint.inherited = None;
    }

    fn hide(&mut self) {
        self.global.contain(Rect::default());
        self.paint.contain(Rect::default());
    }
}

#[derive(bevy_ecs::component::Component, Clone, Copy, Default, PartialEq)]
#[doc(hidden)]
pub struct DerivedClipState(ClipState);

fn children_clip(
    mut state: ClipState,
    node: &Node,
    computed_node: &ComputedNode,
    transform: &UiGlobalTransform,
    has_containment: bool,
) -> ClipState {
    if node.display == Display::None {
        state.hide();
        return state;
    }

    if has_containment {
        let boundary = Rect::from_center_size(transform.translation, computed_node.size());
        state.global.contain(boundary);
        state.paint = ClipAxis {
            inherited: None,
            containment: Some(boundary),
        };
    }

    if !node.overflow.is_visible() {
        let mut clip = computed_node.resolve_clip_rect(node.overflow, node.overflow_clip_margin);
        clip.min += transform.translation;
        clip.max += transform.translation;
        state.global.clip(clip);
        state.paint.clip(clip);
    }
    state
}

pub fn propagate_ui_target_cameras(
    mut commands: Commands,
    default_ui_camera: DefaultUiCamera,
    ui_scale: Res<UiScale>,
    camera_query: Query<(&Camera, Has<crate::UiFillsTarget>)>,
    target_camera_query: Query<&UiTargetCamera>,
    ui_root_nodes: UiRootNodes,
    propagate_sources: Query<
        Entity,
        Or<(
            With<Propagate<ComputedUiTargetCamera>>,
            With<Propagate<ComputedUiRenderTargetInfo>>,
        )>,
    >,
) {
    let default_camera_entity = default_ui_camera.get();

    let mut roots = EntityHashSet::default();
    for root_entity in ui_root_nodes.iter() {
        roots.insert(root_entity);
        let camera = target_camera_query
            .get(root_entity)
            .ok()
            .map(UiTargetCamera::entity)
            .or(default_camera_entity)
            .unwrap_or(Entity::PLACEHOLDER);

        commands
            .entity(root_entity)
            .try_insert(Propagate(ComputedUiTargetCamera { camera }));

        let (scale_factor, physical_size) = camera_query
            .get(camera)
            .ok()
            .map(|(camera, fills_target)| {
                (
                    camera.target_scaling_factor().unwrap_or(1.) * ui_scale.0,
                    // A `UiFillsTarget` camera's interface ignores the
                    // viewport and lays out to the whole target.
                    if fills_target {
                        camera.physical_target_size().unwrap_or(UVec2::ZERO)
                    } else {
                        camera.physical_viewport_size().unwrap_or(UVec2::ZERO)
                    },
                )
            })
            .unwrap_or((1., UVec2::ZERO));

        commands
            .entity(root_entity)
            .try_insert(Propagate(ComputedUiRenderTargetInfo {
                scale_factor,
                physical_size,
            }));
    }

    // An EX-ROOT keeps its `Propagate` stamps when reparented under another
    // root, and a `Propagate` holder is a propagation SOURCE — the node
    // (and its subtree) stays pinned to the camera it had as a root.
    // Strip the stamps from anything no longer a root; removal re-triggers
    // inheritance from the new parent chain.
    for entity in &propagate_sources {
        if !roots.contains(&entity) {
            commands.entity(entity).remove::<(
                Propagate<ComputedUiTargetCamera>,
                Propagate<ComputedUiRenderTargetInfo>,
            )>();
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::update::propagate_ui_target_cameras;
    use crate::ComputedUiRenderTargetInfo;
    use crate::ComputedUiTargetCamera;
    use crate::IsDefaultUiCamera;
    use crate::Node;
    use crate::UiScale;
    use crate::UiTargetCamera;
    use bevy_app::App;
    use bevy_app::HierarchyPropagatePlugin;
    use bevy_app::PostUpdate;
    use bevy_app::PropagateSet;
    use bevy_camera::Camera;
    use bevy_camera::Camera2d;
    use bevy_camera::ComputedCameraValues;
    use bevy_camera::RenderTargetInfo;
    use bevy_ecs::hierarchy::ChildOf;
    use bevy_math::UVec2;
    use bevy_utils::default;

    fn setup_test_app() -> App {
        let mut app = App::new();

        app.init_resource::<UiScale>();

        app.add_plugins(HierarchyPropagatePlugin::<ComputedUiTargetCamera>::new(
            PostUpdate,
        ));
        app.configure_sets(
            PostUpdate,
            PropagateSet::<ComputedUiTargetCamera>::default(),
        );

        app.add_plugins(HierarchyPropagatePlugin::<ComputedUiRenderTargetInfo>::new(
            PostUpdate,
        ));
        app.configure_sets(
            PostUpdate,
            PropagateSet::<ComputedUiRenderTargetInfo>::default(),
        );

        app.add_systems(bevy_app::Update, propagate_ui_target_cameras);

        app
    }

    #[test]
    fn update_context_for_single_ui_root() {
        let mut app = setup_test_app();
        let world = app.world_mut();

        let scale_factor = 10.;
        let physical_size = UVec2::new(1000, 500);

        let camera = world
            .spawn((
                Camera2d,
                Camera {
                    computed: ComputedCameraValues {
                        target_info: Some(RenderTargetInfo {
                            physical_size,
                            scale_factor,
                        }),
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ))
            .id();

        let uinode = world.spawn(Node::default()).id();

        app.update();
        let world = app.world_mut();

        assert_eq!(
            *world.get::<ComputedUiTargetCamera>(uinode).unwrap(),
            ComputedUiTargetCamera { camera }
        );

        assert_eq!(
            *world.get::<ComputedUiRenderTargetInfo>(uinode).unwrap(),
            ComputedUiRenderTargetInfo {
                physical_size,
                scale_factor,
            }
        );
    }

    #[test]
    fn update_multiple_context_for_multiple_ui_roots() {
        let mut app = setup_test_app();
        let world = app.world_mut();

        let scale1 = 1.;
        let size1 = UVec2::new(100, 100);
        let scale2 = 2.;
        let size2 = UVec2::new(200, 200);

        let camera1 = world
            .spawn((
                Camera2d,
                IsDefaultUiCamera,
                Camera {
                    computed: ComputedCameraValues {
                        target_info: Some(RenderTargetInfo {
                            physical_size: size1,
                            scale_factor: scale1,
                        }),
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ))
            .id();
        let camera2 = world
            .spawn((
                Camera2d,
                Camera {
                    computed: ComputedCameraValues {
                        target_info: Some(RenderTargetInfo {
                            physical_size: size2,
                            scale_factor: scale2,
                        }),
                        ..Default::default()
                    },
                    ..default()
                },
            ))
            .id();

        let uinode1a = world.spawn(Node::default()).id();
        let uinode2a = world.spawn((Node::default(), UiTargetCamera(camera2))).id();
        let uinode2b = world.spawn((Node::default(), UiTargetCamera(camera2))).id();
        let uinode2c = world.spawn((Node::default(), UiTargetCamera(camera2))).id();
        let uinode1b = world.spawn(Node::default()).id();

        app.update();
        let world = app.world_mut();

        for (uinode, camera, scale_factor, physical_size) in [
            (uinode1a, camera1, scale1, size1),
            (uinode1b, camera1, scale1, size1),
            (uinode2a, camera2, scale2, size2),
            (uinode2b, camera2, scale2, size2),
            (uinode2c, camera2, scale2, size2),
        ] {
            assert_eq!(
                *world.get::<ComputedUiTargetCamera>(uinode).unwrap(),
                ComputedUiTargetCamera { camera }
            );

            assert_eq!(
                *world.get::<ComputedUiRenderTargetInfo>(uinode).unwrap(),
                ComputedUiRenderTargetInfo {
                    physical_size,
                    scale_factor,
                }
            );
        }
    }

    #[test]
    fn update_context_on_changed_camera() {
        let mut app = setup_test_app();
        let world = app.world_mut();

        let scale1 = 1.;
        let size1 = UVec2::new(100, 100);
        let scale2 = 2.;
        let size2 = UVec2::new(200, 200);

        let camera1 = world
            .spawn((
                Camera2d,
                IsDefaultUiCamera,
                Camera {
                    computed: ComputedCameraValues {
                        target_info: Some(RenderTargetInfo {
                            physical_size: size1,
                            scale_factor: scale1,
                        }),
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ))
            .id();
        let camera2 = world
            .spawn((
                Camera2d,
                Camera {
                    computed: ComputedCameraValues {
                        target_info: Some(RenderTargetInfo {
                            physical_size: size2,
                            scale_factor: scale2,
                        }),
                        ..Default::default()
                    },
                    ..default()
                },
            ))
            .id();

        let uinode = world.spawn(Node::default()).id();

        app.update();
        let world = app.world_mut();

        assert_eq!(
            world
                .get::<ComputedUiRenderTargetInfo>(uinode)
                .unwrap()
                .scale_factor,
            scale1
        );

        assert_eq!(
            world
                .get::<ComputedUiRenderTargetInfo>(uinode)
                .unwrap()
                .physical_size,
            size1
        );

        assert_eq!(
            world
                .get::<ComputedUiTargetCamera>(uinode)
                .unwrap()
                .get()
                .unwrap(),
            camera1
        );

        world.entity_mut(uinode).insert(UiTargetCamera(camera2));

        app.update();
        let world = app.world_mut();

        assert_eq!(
            world
                .get::<ComputedUiRenderTargetInfo>(uinode)
                .unwrap()
                .scale_factor,
            scale2
        );

        assert_eq!(
            world
                .get::<ComputedUiRenderTargetInfo>(uinode)
                .unwrap()
                .physical_size,
            size2
        );

        assert_eq!(
            world
                .get::<ComputedUiTargetCamera>(uinode)
                .unwrap()
                .get()
                .unwrap(),
            camera2
        );
    }

    #[test]
    fn update_context_after_parent_removed() {
        let mut app = setup_test_app();
        let world = app.world_mut();

        let scale1 = 1.;
        let size1 = UVec2::new(100, 100);
        let scale2 = 2.;
        let size2 = UVec2::new(200, 200);

        let camera1 = world
            .spawn((
                Camera2d,
                IsDefaultUiCamera,
                Camera {
                    computed: ComputedCameraValues {
                        target_info: Some(RenderTargetInfo {
                            physical_size: size1,
                            scale_factor: scale1,
                        }),
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ))
            .id();
        let camera2 = world
            .spawn((
                Camera2d,
                Camera {
                    computed: ComputedCameraValues {
                        target_info: Some(RenderTargetInfo {
                            physical_size: size2,
                            scale_factor: scale2,
                        }),
                        ..Default::default()
                    },
                    ..default()
                },
            ))
            .id();

        // `UiTargetCamera` is ignored on non-root UI nodes
        let uinode1 = world.spawn((Node::default(), UiTargetCamera(camera2))).id();
        let uinode2 = world.spawn(Node::default()).add_child(uinode1).id();

        app.update();
        let world = app.world_mut();

        assert_eq!(
            world
                .get::<ComputedUiRenderTargetInfo>(uinode1)
                .unwrap()
                .scale_factor(),
            scale1
        );

        assert_eq!(
            world
                .get::<ComputedUiRenderTargetInfo>(uinode1)
                .unwrap()
                .physical_size(),
            size1
        );

        assert_eq!(
            world
                .get::<ComputedUiTargetCamera>(uinode1)
                .unwrap()
                .get()
                .unwrap(),
            camera1
        );

        assert_eq!(
            world
                .get::<ComputedUiTargetCamera>(uinode2)
                .unwrap()
                .get()
                .unwrap(),
            camera1
        );

        // Now `uinode1` is a root UI node its `UiTargetCamera` component will be used and its camera target set to `camera2`.
        world.entity_mut(uinode1).remove::<ChildOf>();

        app.update();
        let world = app.world_mut();

        assert_eq!(
            world
                .get::<ComputedUiRenderTargetInfo>(uinode1)
                .unwrap()
                .scale_factor(),
            scale2
        );

        assert_eq!(
            world
                .get::<ComputedUiRenderTargetInfo>(uinode1)
                .unwrap()
                .physical_size(),
            size2
        );

        assert_eq!(
            world
                .get::<ComputedUiTargetCamera>(uinode1)
                .unwrap()
                .get()
                .unwrap(),
            camera2
        );

        assert_eq!(
            world
                .get::<ComputedUiTargetCamera>(uinode2)
                .unwrap()
                .get()
                .unwrap(),
            camera1
        );
    }

    #[test]
    fn update_great_grandchild() {
        let mut app = setup_test_app();
        let world = app.world_mut();

        let scale = 1.;
        let size = UVec2::new(100, 100);

        let camera = world
            .spawn((
                Camera2d,
                Camera {
                    computed: ComputedCameraValues {
                        target_info: Some(RenderTargetInfo {
                            physical_size: size,
                            scale_factor: scale,
                        }),
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ))
            .id();

        let uinode = world.spawn(Node::default()).id();
        world.spawn(Node::default()).with_children(|builder| {
            builder.spawn(Node::default()).with_children(|builder| {
                builder.spawn(Node::default()).add_child(uinode);
            });
        });

        app.update();
        let world = app.world_mut();

        assert_eq!(
            world
                .get::<ComputedUiRenderTargetInfo>(uinode)
                .unwrap()
                .scale_factor,
            scale
        );

        assert_eq!(
            world
                .get::<ComputedUiRenderTargetInfo>(uinode)
                .unwrap()
                .physical_size,
            size
        );

        assert_eq!(
            world
                .get::<ComputedUiTargetCamera>(uinode)
                .unwrap()
                .get()
                .unwrap(),
            camera
        );

        world.resource_mut::<UiScale>().0 = 2.;

        app.update();
        let world = app.world_mut();

        assert_eq!(
            world
                .get::<ComputedUiRenderTargetInfo>(uinode)
                .unwrap()
                .scale_factor(),
            2.
        );
    }
}
