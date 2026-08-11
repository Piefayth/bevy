use core::fmt;
use core::ops::{Deref, DerefMut};

use bevy_platform::collections::hash_map::Entry;
use taffy::TaffyTree;

use bevy_ecs::{
    entity::{Entity, EntityHashMap, EntityHashSet},
    prelude::Resource,
};
use bevy_math::{UVec2, Vec2};
use bevy_utils::default;

use crate::{layout::convert, LayoutContext, LayoutError, Measure, MeasureArgs, Node, NodeMeasure};
use bevy_text::FontCx;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct LayoutNode {
    // Implicit "viewport" node if this `LayoutNode` corresponds to a root UI node entity
    pub(super) viewport_id: Option<taffy::NodeId>,
    // The id of the node in the taffy tree
    pub(super) id: taffy::NodeId,
    // A disconnected container root for descendants when this node has layout containment.
    pub(super) container_id: Option<taffy::NodeId>,
}

impl From<taffy::NodeId> for LayoutNode {
    fn from(value: taffy::NodeId) -> Self {
        LayoutNode {
            viewport_id: None,
            id: value,
            container_id: None,
        }
    }
}

pub(crate) struct UiTree<T>(TaffyTree<T>);

#[expect(unsafe_code, reason = "TaffyTree is safe as long as calc is not used")]
// SAFETY: Taffy Tree becomes thread unsafe when you use the calc feature, which we do not implement
unsafe impl Send for UiTree<NodeMeasure> {}

#[expect(unsafe_code, reason = "TaffyTree is safe as long as calc is not used")]
// SAFETY: Taffy Tree becomes thread unsafe when you use the calc feature, which we do not implement
unsafe impl Sync for UiTree<NodeMeasure> {}

impl<T> Deref for UiTree<T> {
    type Target = TaffyTree<T>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for UiTree<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Resource)]
pub struct UiSurface {
    pub root_entity_to_viewport_node: EntityHashMap<taffy::NodeId>,
    pub(super) entity_to_taffy: EntityHashMap<LayoutNode>,
    pub(super) taffy: UiTree<NodeMeasure>,
    taffy_children_scratch: Vec<taffy::NodeId>,
    // One-frame handoff from layout computation to derived geometry resolution.
    pub(super) layout_geometry_scopes: EntityHashSet,
    #[cfg(test)]
    pub(super) geometry_visits: usize,
    #[cfg(test)]
    pub(super) layout_computations: usize,
    #[cfg(feature = "ghost_nodes")]
    pub(super) dirty_ghost_children_scratch: EntityHashSet,
}

fn _assert_send_sync_ui_surface_impl_safe() {
    fn _assert_send_sync<T: Send + Sync>() {}
    _assert_send_sync::<EntityHashMap<taffy::NodeId>>();
    _assert_send_sync::<UiTree<NodeMeasure>>();
    _assert_send_sync::<UiSurface>();
}

impl fmt::Debug for UiSurface {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut debug = f.debug_struct("UiSurface");
        debug
            .field("entity_to_taffy", &self.entity_to_taffy)
            .field("taffy_children_scratch", &self.taffy_children_scratch);
        #[cfg(feature = "ghost_nodes")]
        debug.field(
            "dirty_ghost_children_scratch",
            &self.dirty_ghost_children_scratch,
        );
        debug.finish()
    }
}

impl Default for UiSurface {
    fn default() -> Self {
        let taffy: UiTree<NodeMeasure> = UiTree(TaffyTree::new());
        Self {
            root_entity_to_viewport_node: Default::default(),
            entity_to_taffy: Default::default(),
            taffy,
            taffy_children_scratch: Vec::new(),
            layout_geometry_scopes: EntityHashSet::new(),
            #[cfg(test)]
            geometry_visits: 0,
            #[cfg(test)]
            layout_computations: 0,
            #[cfg(feature = "ghost_nodes")]
            dirty_ghost_children_scratch: EntityHashSet::new(),
        }
    }
}

impl UiSurface {
    /// Retrieves the Taffy node associated with the given UI node entity and updates its style.
    /// If no associated Taffy node exists a new Taffy node is inserted into the Taffy layout.
    pub fn upsert_node(
        &mut self,
        layout_context: &LayoutContext,
        entity: Entity,
        node: &Node,
        new_node_context: Option<NodeMeasure>,
    ) -> bool {
        self.upsert_node_with_containment(layout_context, entity, node, new_node_context, false)
    }

    /// Inserts or updates a UI node and its independent descendant layout root.
    pub fn upsert_node_with_containment(
        &mut self,
        layout_context: &LayoutContext,
        entity: Entity,
        node: &Node,
        mut new_node_context: Option<NodeMeasure>,
        contained: bool,
    ) -> bool {
        let style = convert::from_node(node, layout_context);
        let taffy = &mut self.taffy;

        match self.entity_to_taffy.entry(entity) {
            Entry::Occupied(mut entry) => {
                let taffy_node = entry.get_mut();
                let mut changed = false;
                if new_node_context.is_some() {
                    taffy
                        .set_node_context(taffy_node.id, new_node_context)
                        .unwrap();
                    changed = true;
                }

                let style_changed = taffy.style(taffy_node.id).unwrap() != &style;
                if contained {
                    if style_changed {
                        taffy.set_style(taffy_node.id, style.clone()).unwrap();
                        changed = true;
                    }
                    if let Some(container_id) = taffy_node.container_id {
                        if style_changed {
                            taffy
                                .set_style(container_id, containment_style(style, Vec2::ZERO))
                                .unwrap();
                        }
                    } else {
                        let children = taffy.children(taffy_node.id).unwrap();
                        let container_id = taffy
                            .new_with_children(containment_style(style, Vec2::ZERO), &children)
                            .unwrap();
                        taffy.set_children(taffy_node.id, &[]).unwrap();
                        taffy_node.container_id = Some(container_id);
                        changed = true;
                    }
                } else {
                    if style_changed {
                        taffy.set_style(taffy_node.id, style).unwrap();
                        changed = true;
                    }
                    if let Some(container_id) = taffy_node.container_id {
                        let children = taffy.children(container_id).unwrap();
                        taffy.set_children(taffy_node.id, &children).unwrap();
                        taffy.remove(container_id).unwrap();
                        taffy_node.container_id = None;
                        changed = true;
                    }
                }
                changed
            }
            Entry::Vacant(entry) => {
                let container_style =
                    contained.then(|| containment_style(style.clone(), Vec2::ZERO));
                let taffy_node = if let Some(measure) = new_node_context.take() {
                    taffy.new_leaf_with_context(style, measure)
                } else {
                    taffy.new_leaf(style)
                };
                let mut layout_node: LayoutNode = taffy_node.unwrap().into();
                if let Some(container_style) = container_style {
                    layout_node.container_id = Some(taffy.new_leaf(container_style).unwrap());
                }
                entry.insert(layout_node);
                true
            }
        }
    }

    /// Update the `MeasureFunc` of the taffy node corresponding to the given [`Entity`] if the node exists.
    pub fn update_node_context(&mut self, entity: Entity, context: NodeMeasure) -> Option<()> {
        let taffy_node = self.entity_to_taffy.get(&entity)?;
        self.taffy
            .set_node_context(taffy_node.id, Some(context))
            .ok()
    }

    /// Update the children of the taffy node corresponding to the given [`Entity`].
    pub fn update_children(&mut self, entity: Entity, children: impl Iterator<Item = Entity>) {
        self.taffy_children_scratch.clear();

        for child in children {
            if let Some(taffy_node) = self.entity_to_taffy.get_mut(&child) {
                self.taffy_children_scratch.push(taffy_node.id);
                if let Some(viewport_id) = taffy_node.viewport_id.take() {
                    self.taffy.remove(viewport_id).ok();
                }
                self.root_entity_to_viewport_node.remove(&child);
            }
        }

        let taffy_node = self.entity_to_taffy.get(&entity).unwrap();
        let parent_id = taffy_node.container_id.unwrap_or(taffy_node.id);
        self.taffy
            .set_children(parent_id, &self.taffy_children_scratch)
            .unwrap();
    }

    /// Removes children from the entity's taffy node if it exists. Does nothing otherwise.
    pub fn try_remove_children(&mut self, entity: Entity) {
        if let Some(taffy_node) = self.entity_to_taffy.get(&entity) {
            self.taffy
                .set_children(taffy_node.container_id.unwrap_or(taffy_node.id), &[])
                .unwrap();
        }
    }

    /// Removes the measure from the entity's taffy node if it exists. Does nothing otherwise.
    pub fn try_remove_node_context(&mut self, entity: Entity) -> bool {
        if let Some(taffy_node) = self.entity_to_taffy.get(&entity)
            && self.taffy.get_node_context(taffy_node.id).is_some()
        {
            self.taffy.set_node_context(taffy_node.id, None).unwrap();
            return true;
        }
        false
    }

    /// Gets or inserts an implicit taffy viewport node corresponding to the given UI root entity
    pub fn get_or_insert_taffy_viewport_node(&mut self, ui_root_entity: Entity) -> taffy::NodeId {
        *self
            .root_entity_to_viewport_node
            .entry(ui_root_entity)
            .or_insert_with(|| {
                let root_node = self.entity_to_taffy.get_mut(&ui_root_entity).unwrap();
                let implicit_root = self
                    .taffy
                    .new_leaf(taffy::style::Style {
                        display: taffy::style::Display::Grid,
                        // Note: Taffy percentages are floats ranging from 0.0 to 1.0.
                        // So this is setting width:100% and height:100%
                        size: taffy::geometry::Size {
                            width: taffy::style_helpers::percent(1.0_f32),
                            height: taffy::style_helpers::percent(1.0_f32),
                        },
                        align_items: Some(taffy::style::AlignItems::Start),
                        justify_items: Some(taffy::style::JustifyItems::Start),
                        ..default()
                    })
                    .unwrap();
                self.taffy.add_child(implicit_root, root_node.id).unwrap();
                root_node.viewport_id = Some(implicit_root);
                implicit_root
            })
    }

    /// Compute the layout for the given implicit taffy viewport node
    pub fn compute_layout<'a>(
        &mut self,
        ui_root_entity: Entity,
        render_target_resolution: UVec2,
        buffer_query: &'a mut bevy_ecs::prelude::Query<&mut bevy_text::ComputedTextBlock>,
        font_system: &'a mut FontCx,
    ) {
        #[cfg(test)]
        {
            self.layout_computations += 1;
        }
        let implicit_viewport_node = self.get_or_insert_taffy_viewport_node(ui_root_entity);

        let available_space = taffy::geometry::Size {
            width: taffy::style::AvailableSpace::Definite(render_target_resolution.x as f32),
            height: taffy::style::AvailableSpace::Definite(render_target_resolution.y as f32),
        };

        compute_layout_with_measure(
            &mut self.taffy,
            implicit_viewport_node,
            available_space,
            buffer_query,
            font_system,
        );
    }

    /// Computes the disconnected descendant layout rooted at a contained node.
    pub fn compute_contained_layout<'a>(
        &mut self,
        entity: Entity,
        buffer_query: &'a mut bevy_ecs::prelude::Query<&mut bevy_text::ComputedTextBlock>,
        font_system: &'a mut FontCx,
    ) {
        let Some(layout_node) = self.entity_to_taffy.get(&entity).copied() else {
            return;
        };
        let Some(container_id) = layout_node.container_id else {
            return;
        };

        let outer_layout = self.taffy.unrounded_layout(layout_node.id);
        let outer_size = Vec2::new(outer_layout.size.width, outer_layout.size.height);
        let outer_style = self.taffy.style(layout_node.id).unwrap().clone();
        let container_style = containment_style(outer_style, outer_size);
        if self.taffy.style(container_id).unwrap() != &container_style {
            self.taffy.set_style(container_id, container_style).unwrap();
        } else if !self.taffy.dirty(container_id).unwrap() {
            return;
        }

        #[cfg(test)]
        {
            self.layout_computations += 1;
        }

        let available_space = taffy::geometry::Size {
            width: taffy::style::AvailableSpace::Definite(outer_size.x),
            height: taffy::style::AvailableSpace::Definite(outer_size.y),
        };

        compute_layout_with_measure(
            &mut self.taffy,
            container_id,
            available_space,
            buffer_query,
            font_system,
        );
    }

    /// Removes each entity from the internal map and then removes their associated nodes from taffy
    pub fn remove_entities(&mut self, entities: impl IntoIterator<Item = Entity>) {
        for entity in entities {
            if let Some(node) = self.entity_to_taffy.remove(&entity) {
                self.taffy.remove(node.id).unwrap();
                if let Some(container_node) = node.container_id {
                    self.taffy.remove(container_node).ok();
                }
                if let Some(viewport_node) = node.viewport_id {
                    self.taffy.remove(viewport_node).ok();
                }
            }
            self.root_entity_to_viewport_node.remove(&entity);
        }
    }

    /// Get the layout geometry for the taffy node corresponding to the ui node [`Entity`].
    /// Does not compute the layout geometry, `compute_window_layouts` should be run before using this function.
    /// On success returns a pair consisting of the final resolved layout values after rounding
    /// and the size of the node after layout resolution but before rounding.
    pub fn get_layout(
        &mut self,
        entity: Entity,
        use_rounding: bool,
    ) -> Result<(taffy::Layout, Vec2), LayoutError> {
        let Some(taffy_node) = self.entity_to_taffy.get(&entity) else {
            return Err(LayoutError::InvalidHierarchy);
        };

        if use_rounding {
            self.taffy.enable_rounding();
        } else {
            self.taffy.disable_rounding();
        }

        let out = match self.taffy.layout(taffy_node.id).cloned() {
            Ok(mut layout) => {
                if let Some(container_id) = taffy_node.container_id {
                    let container_layout = self.taffy.layout(container_id).unwrap();
                    layout.content_size = container_layout.content_size;
                    layout.scrollbar_size = container_layout.scrollbar_size;
                }
                self.taffy.disable_rounding();
                let taffy_size = self.taffy.layout(taffy_node.id).unwrap().size;
                let unrounded_size = Vec2::new(taffy_size.width, taffy_size.height);
                Ok((layout, unrounded_size))
            }
            Err(taffy_error) => Err(LayoutError::TaffyError(taffy_error)),
        };

        self.taffy.enable_rounding();
        out
    }
}

fn compute_layout_with_measure<'a>(
    taffy: &mut UiTree<NodeMeasure>,
    node: taffy::NodeId,
    available_space: taffy::Size<taffy::AvailableSpace>,
    buffer_query: &'a mut bevy_ecs::prelude::Query<&mut bevy_text::ComputedTextBlock>,
    font_system: &'a mut FontCx,
) {
    taffy
        .compute_layout_with_measure(
            node,
            available_space,
            |known_dimensions: taffy::Size<Option<f32>>,
             available_space: taffy::Size<taffy::AvailableSpace>,
             _node_id: taffy::NodeId,
             context: Option<&mut NodeMeasure>,
             style: &taffy::Style|
             -> taffy::Size<f32> {
                context
                    .map(|ctx| {
                        let buffer = get_text_buffer(
                            crate::widget::TextMeasure::needs_buffer(
                                known_dimensions.height,
                                available_space.width,
                            ),
                            ctx,
                            buffer_query,
                        );
                        let size = ctx.measure(MeasureArgs {
                            known_width: known_dimensions.width,
                            known_height: known_dimensions.height,
                            available_width: available_space.width,
                            available_height: available_space.height,
                            font_system,
                            buffer,
                            style,
                        });
                        taffy::Size {
                            width: size.x,
                            height: size.y,
                        }
                    })
                    .unwrap_or(taffy::Size::ZERO)
            },
        )
        .unwrap();
}

fn containment_style(mut style: taffy::Style, size: Vec2) -> taffy::Style {
    let defaults = taffy::Style::default();
    style.item_is_table = defaults.item_is_table;
    style.item_is_replaced = defaults.item_is_replaced;
    style.box_sizing = defaults.box_sizing;
    style.position = defaults.position;
    style.inset = defaults.inset;
    style.size = taffy::Size {
        width: taffy::style_helpers::length(size.x),
        height: taffy::style_helpers::length(size.y),
    };
    style.min_size = defaults.min_size;
    style.max_size = defaults.max_size;
    style.aspect_ratio = defaults.aspect_ratio;
    style.margin = defaults.margin;
    style.align_self = defaults.align_self;
    style.justify_self = defaults.justify_self;
    style.flex_basis = defaults.flex_basis;
    style.flex_grow = defaults.flex_grow;
    style.flex_shrink = defaults.flex_shrink;
    style.grid_row = defaults.grid_row;
    style.grid_column = defaults.grid_column;
    style
}

pub fn get_text_buffer<'a>(
    needs_buffer: bool,
    ctx: &mut NodeMeasure,
    query: &'a mut bevy_ecs::prelude::Query<&mut bevy_text::ComputedTextBlock>,
) -> Option<&'a mut bevy_text::ComputedTextBlock> {
    // We avoid a query lookup whenever the buffer is not required.
    if !needs_buffer {
        return None;
    }
    let NodeMeasure::Text(crate::widget::TextMeasure { info }) = ctx else {
        return None;
    };
    let Ok(computed) = query.get_mut(info.entity) else {
        return None;
    };
    Some(computed.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContentSize, FixedMeasure};
    use bevy_math::Vec2;
    use taffy::TraversePartialTree;

    #[test]
    fn test_initialization() {
        let ui_surface = UiSurface::default();
        assert!(ui_surface.entity_to_taffy.is_empty());
        assert_eq!(ui_surface.taffy.total_node_count(), 0);
    }

    #[test]
    fn test_upsert() {
        let mut ui_surface = UiSurface::default();
        let root_node_entity = Entity::from_raw_u32(1).unwrap();
        let node = Node::default();

        // standard upsert
        ui_surface.upsert_node(&LayoutContext::TEST_CONTEXT, root_node_entity, &node, None);

        // should be inserted into taffy
        assert_eq!(ui_surface.taffy.total_node_count(), 1);
        assert!(ui_surface.entity_to_taffy.contains_key(&root_node_entity));

        // test duplicate insert 1
        ui_surface.upsert_node(&LayoutContext::TEST_CONTEXT, root_node_entity, &node, None);

        // node count should not have increased
        assert_eq!(ui_surface.taffy.total_node_count(), 1);

        // assign root node to camera
        ui_surface.get_or_insert_taffy_viewport_node(root_node_entity);

        // each root node will create 2 taffy nodes
        assert_eq!(ui_surface.taffy.total_node_count(), 2);

        // test duplicate insert 2
        ui_surface.upsert_node(&LayoutContext::TEST_CONTEXT, root_node_entity, &node, None);

        // node count should not have increased
        assert_eq!(ui_surface.taffy.total_node_count(), 2);
    }

    #[test]
    fn test_remove_entities() {
        let mut ui_surface = UiSurface::default();
        let root_node_entity = Entity::from_raw_u32(1).unwrap();
        let node = Node::default();

        ui_surface.upsert_node(&LayoutContext::TEST_CONTEXT, root_node_entity, &node, None);

        ui_surface.get_or_insert_taffy_viewport_node(root_node_entity);

        assert!(ui_surface.entity_to_taffy.contains_key(&root_node_entity));

        ui_surface.remove_entities([root_node_entity]);
        assert!(!ui_surface.entity_to_taffy.contains_key(&root_node_entity));
    }

    #[test]
    fn test_try_update_measure() {
        let mut ui_surface = UiSurface::default();
        let root_node_entity = Entity::from_raw_u32(1).unwrap();
        let node = Node::default();

        ui_surface.upsert_node(&LayoutContext::TEST_CONTEXT, root_node_entity, &node, None);
        let mut content_size = ContentSize::default();
        content_size.set(NodeMeasure::Fixed(FixedMeasure { size: Vec2::ONE }));
        let measure_func = content_size.measure.take().unwrap();
        assert!(ui_surface
            .update_node_context(root_node_entity, measure_func)
            .is_some());
    }

    #[test]
    fn test_update_children() {
        let mut ui_surface = UiSurface::default();
        let root_node_entity = Entity::from_raw_u32(1).unwrap();
        let child_entity = Entity::from_raw_u32(2).unwrap();
        let node = Node::default();

        ui_surface.upsert_node(&LayoutContext::TEST_CONTEXT, root_node_entity, &node, None);
        ui_surface.upsert_node(&LayoutContext::TEST_CONTEXT, child_entity, &node, None);

        ui_surface.update_children(root_node_entity, vec![child_entity].into_iter());

        let parent_node = *ui_surface.entity_to_taffy.get(&root_node_entity).unwrap();
        let child_node = *ui_surface.entity_to_taffy.get(&child_entity).unwrap();
        assert_eq!(ui_surface.taffy.parent(child_node.id), Some(parent_node.id));
    }

    #[expect(
        unreachable_code,
        reason = "Certain pieces of code tested here cause the test to fail if made reachable; see #16362 for progress on fixing this"
    )]
    #[test]
    fn test_set_camera_children() {
        let mut ui_surface = UiSurface::default();
        let root_node_entity = Entity::from_raw_u32(1).unwrap();
        let child_entity = Entity::from_raw_u32(2).unwrap();
        let node = Node::default();

        ui_surface.upsert_node(&LayoutContext::TEST_CONTEXT, root_node_entity, &node, None);
        ui_surface.upsert_node(&LayoutContext::TEST_CONTEXT, child_entity, &node, None);

        let root_taffy_node = *ui_surface.entity_to_taffy.get(&root_node_entity).unwrap();
        let child_taffy = *ui_surface.entity_to_taffy.get(&child_entity).unwrap();

        // set up the relationship manually
        ui_surface
            .taffy
            .add_child(root_taffy_node.id, child_taffy.id)
            .unwrap();

        ui_surface.get_or_insert_taffy_viewport_node(root_node_entity);

        assert_eq!(
            ui_surface.taffy.parent(child_taffy.id),
            Some(root_taffy_node.id)
        );
        let root_taffy_children = ui_surface.taffy.children(root_taffy_node.id).unwrap();
        assert!(
            root_taffy_children.contains(&child_taffy.id),
            "root node is not a parent of child node"
        );
        assert_eq!(
            ui_surface.taffy.child_count(root_taffy_node.id),
            1,
            "expected root node child count to be 1"
        );

        // clear camera's root nodes
        ui_surface.get_or_insert_taffy_viewport_node(root_node_entity);

        return; // TODO: can't pass the test if we continue - not implemented (remove allow(unreachable_code))

        let root_taffy_children = ui_surface.taffy.children(root_taffy_node.id).unwrap();
        assert!(
            root_taffy_children.contains(&child_taffy.id),
            "root node is not a parent of child node"
        );
        assert_eq!(
            ui_surface.taffy.child_count(root_taffy_node.id),
            1,
            "expected root node child count to be 1"
        );

        // re-associate root node with viewport node
        ui_surface.get_or_insert_taffy_viewport_node(root_node_entity);

        let child_taffy = ui_surface.entity_to_taffy.get(&child_entity).unwrap();
        let root_taffy_children = ui_surface.taffy.children(root_taffy_node.id).unwrap();
        assert!(
            root_taffy_children.contains(&child_taffy.id),
            "root node is not a parent of child node"
        );
        assert_eq!(
            ui_surface.taffy.child_count(root_taffy_node.id),
            1,
            "expected root node child count to be 1"
        );
    }
}
