#![expect(missing_docs, reason = "Not all docs are written yet, see #3492.")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(
    html_logo_url = "https://bevyengine.org/assets/icon.png",
    html_favicon_url = "https://bevyengine.org/assets/icon.png"
)]

//! Provides rendering functionality for `bevy_ui`.

pub mod box_shadow;
mod gradient;
mod image;
mod pipeline;
pub mod render_pass;
mod retained;
mod text;
pub mod ui_material;
mod ui_material_pipeline;
pub mod ui_texture_slice_pipeline;

#[cfg(feature = "bevy_ui_debug")]
mod debug_overlay;

use bevy_a11y::AccessibilitySystems;
use bevy_camera::visibility::InheritedVisibility;
use bevy_camera::{Camera, Camera2d, Camera3d, RenderTarget};
use bevy_ecs::entity::{EntityHashMap, EntityIndexMap};
use bevy_reflect::prelude::ReflectDefault;
use bevy_reflect::Reflect;
use bevy_render::camera::{extract_cameras, CameraMainPassTextureFormats};
use bevy_render::sync_world::{MainEntityHashMap, MainEntityHashSet};
use bevy_shader::load_shader_library;
use bevy_sprite_render::SpriteAssetEvents;
use bevy_ui::widget::{ImageNode, ImageNodeSize, NodeImageMode, Text, TextShadow, ViewportNode};
use bevy_ui::{
    BackgroundColor, BackgroundGradient, BorderColor, BorderGradient, BoxShadow, CalculatedClip,
    ComputedNode, ComputedStackIndex, ComputedUiTargetCamera, Display, Node, OuterColor, Outline,
    ResolvedBorderRadius, UiGlobalTransform, UiSystems, VisualBox,
};

use bevy_app::prelude::*;
use bevy_asset::{AssetEvent, AssetEventSystems, AssetId, Assets};
use bevy_color::{Alpha, ColorToComponents, LinearRgba};
use bevy_core_pipeline::schedule::{Core2d, Core2dSystems, Core3d, Core3dSystems};
use bevy_core_pipeline::upscaling::upscaling;
use bevy_ecs::prelude::*;
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_ecs::system::SystemParam;
use bevy_image::{prelude::*, TRANSPARENT_IMAGE_HANDLE};
use bevy_math::{Affine2, FloatOrd, Mat4, Rect, UVec4, Vec2};
use bevy_render::{
    impl_atomic_pod,
    render_asset::RenderAssets,
    render_phase::{
        sort_phase_system, AddRenderCommand, DrawFunctions, PhaseItem, PhaseItemExtraIndex,
        ViewSortedRenderPhases,
    },
    render_resource::*,
    renderer::{RenderDevice, RenderQueue},
    sync_world::{MainEntity, RenderEntity},
    texture::GpuImage,
    view::{ExtractedView, RetainedViewEntity, ViewUniforms},
    Extract, ExtractSchedule, GpuResourceAppExt, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_sprite::BorderRect;
#[cfg(feature = "bevy_ui_debug")]
pub use debug_overlay::{GlobalUiDebugOptions, UiDebugOptions};

use gradient::GradientPlugin;

use bevy_platform::{
    collections::{hash_map::Entry, HashMap, HashSet},
    sync::Arc,
};
use bevy_text::{
    ComputedTextBlock, EditableText, PositionedGlyph, Strikethrough, StrikethroughColor,
    TextBackgroundColor, TextColor, TextCursorStyle, TextLayoutInfo, TextSpan, Underline,
    UnderlineColor,
};
use bevy_transform::components::GlobalTransform;
use box_shadow::BoxShadowPlugin;
use bytemuck::{Pod, Zeroable};
use core::{mem, ops::Range};

pub use pipeline::*;
pub use render_pass::*;
pub use ui_material_pipeline::*;
use ui_texture_slice_pipeline::UiTextureSlicerPlugin;

use crate::retained::{
    batch_retained_ui, ArenaSlot, ItemInstances, RemovedUiNode, RetainedBatchItem,
    UiCameraPipelineState, UiInstanceArena,
};
use crate::shader_flags::INVERT;
use crate::text::{extract_preedit_underlines, extract_text_cursor};

pub mod prelude {
    #[cfg(feature = "bevy_ui_debug")]
    pub use crate::debug_overlay::{GlobalUiDebugOptions, UiDebugOptions};

    pub use crate::{
        ui_material::*, ui_material_pipeline::UiMaterialPlugin, BoxShadowSamples, UiAntiAlias,
    };
}

/// Local Z offsets of "extracted nodes" for a given entity. These exist to allow rendering multiple "extracted nodes"
/// for a given source entity (ex: render both a background color _and_ a custom material for a given node).
///
/// When possible these offsets should be defined in _this_ module to ensure z-index coordination across contexts.
/// When this is _not_ possible, pick a suitably unique index unlikely to clash with other things (ex: `0.1826823` not `0.1`).
///
/// Offsets should be unique for a given node entity to avoid z fighting.
/// These should pretty much _always_ be larger than -0.5 and smaller than 0.5 to avoid clipping into nodes
/// above / below the current node in the stack.
///
/// A z-index of 0.0 is the baseline, which is used as the primary "background color" of the node.
///
/// Note that nodes "stack" on each other, so a negative offset on the node above could clip _into_
/// a positive offset on a node below.
pub mod stack_z_offsets {
    pub const BOX_SHADOW: f32 = -0.1;
    pub const BACKGROUND_COLOR: f32 = 0.0;
    pub const BORDER: f32 = 0.01;
    pub const GRADIENT: f32 = 0.02;
    pub const BORDER_GRADIENT: f32 = 0.03;
    pub const IMAGE: f32 = 0.04;
    pub const MATERIAL: f32 = 0.05;
    pub const TEXT_SELECTION: f32 = 0.055;
    pub const TEXT: f32 = 0.06;
    pub const TEXT_STRIKETHROUGH: f32 = 0.07;
    pub const TEXT_CURSOR: f32 = 0.08;
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemSet)]
pub enum RenderUiSystems {
    ExtractChanges,
    ExtractCameraViews,
    ExtractBoxShadows,
    ExtractBackgrounds,
    ExtractImages,
    ExtractTextureSlice,
    ExtractBorders,
    ExtractViewportNodes,
    ExtractTextBackgrounds,
    ExtractTextShadows,
    ExtractText,
    ExtractCursor,
    ExtractDebug,
    ExtractGradient,
}

/// Marker for controlling whether UI is rendered with or without anti-aliasing
/// in a camera. By default, UI is always anti-aliased.
///
/// **Note:** This does not affect text anti-aliasing. For that, use the `font_smoothing` property of the [`TextFont`](bevy_text::TextFont) component.
///
/// ```
/// use bevy_camera::prelude::*;
/// use bevy_ecs::prelude::*;
/// use bevy_ui::prelude::*;
/// use bevy_ui_render::prelude::*;
///
/// fn spawn_camera(mut commands: Commands) {
///     commands.spawn((
///         Camera2d,
///         // This will cause all UI in this camera to be rendered without
///         // anti-aliasing
///         UiAntiAlias::Off,
///     ));
/// }
/// ```
#[derive(Component, Clone, Copy, Default, Debug, Reflect, Eq, PartialEq)]
#[reflect(Component, Default, PartialEq, Clone)]
pub enum UiAntiAlias {
    /// UI will render with anti-aliasing
    #[default]
    On,
    /// UI will render without anti-aliasing
    Off,
}

/// Number of shadow samples.
/// A larger value will result in higher quality shadows.
/// Default is 4, values higher than ~10 offer diminishing returns.
///
/// ```
/// use bevy_camera::prelude::*;
/// use bevy_ecs::prelude::*;
/// use bevy_ui::prelude::*;
/// use bevy_ui_render::prelude::*;
///
/// fn spawn_camera(mut commands: Commands) {
///     commands.spawn((
///         Camera2d,
///         BoxShadowSamples(6),
///     ));
/// }
/// ```
#[derive(Component, Clone, Copy, Debug, Reflect, Eq, PartialEq)]
#[reflect(Component, Default, PartialEq, Clone)]
pub struct BoxShadowSamples(pub u32);

impl Default for BoxShadowSamples {
    fn default() -> Self {
        Self(4)
    }
}

#[derive(Default)]
pub struct UiRenderPlugin;

impl Plugin for UiRenderPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "ui.wgsl");

        #[cfg(feature = "bevy_ui_debug")]
        app.init_resource::<GlobalUiDebugOptions>();

        app.add_systems(
            PostUpdate,
            (
                image::mark_images_as_changed_if_their_assets_changed,
                image::update_texture_atlas_layout_components,
            )
                .chain()
                .after(UiSystems::Content)
                .after(AssetEventSystems)
                .after(AccessibilitySystems::Update),
        );

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_gpu_resource::<SpecializedRenderPipelines<UiPipeline>>()
            .init_gpu_resource::<ImageNodeBindGroups>()
            .init_gpu_resource::<UiMeta>()
            .init_resource::<ExtractedUiNodes>()
            .allow_ambiguous_resource::<ExtractedUiNodes>()
            .init_resource::<DrawFunctions<TransparentUi>>()
            .init_resource::<ViewSortedRenderPhases<TransparentUi>>()
            .allow_ambiguous_resource::<ViewSortedRenderPhases<TransparentUi>>()
            .add_render_command::<TransparentUi, DrawUi>()
            .configure_sets(
                ExtractSchedule,
                (
                    RenderUiSystems::ExtractChanges,
                    RenderUiSystems::ExtractCameraViews,
                    RenderUiSystems::ExtractBoxShadows,
                    RenderUiSystems::ExtractBackgrounds,
                    RenderUiSystems::ExtractViewportNodes,
                    RenderUiSystems::ExtractImages,
                    RenderUiSystems::ExtractTextureSlice,
                    RenderUiSystems::ExtractBorders,
                    RenderUiSystems::ExtractTextBackgrounds,
                    RenderUiSystems::ExtractTextShadows,
                    RenderUiSystems::ExtractText,
                    RenderUiSystems::ExtractCursor,
                    RenderUiSystems::ExtractDebug,
                )
                    .chain(),
            )
            .add_systems(RenderStartup, init_ui_pipeline)
            .add_systems(
                ExtractSchedule,
                (
                    extract_uinode_changes.in_set(RenderUiSystems::ExtractChanges),
                    extract_ui_camera_view
                        .after(extract_cameras)
                        .in_set(RenderUiSystems::ExtractCameraViews),
                    extract_uinode_background_colors.in_set(RenderUiSystems::ExtractBackgrounds),
                    extract_uinode_images.in_set(RenderUiSystems::ExtractImages),
                    extract_uinode_borders.in_set(RenderUiSystems::ExtractBorders),
                    extract_viewport_nodes.in_set(RenderUiSystems::ExtractViewportNodes),
                    extract_text_decorations.in_set(RenderUiSystems::ExtractTextBackgrounds),
                    extract_text_shadows.in_set(RenderUiSystems::ExtractTextShadows),
                    extract_text_sections.in_set(RenderUiSystems::ExtractText),
                    extract_text_cursor.in_set(RenderUiSystems::ExtractCursor),
                    extract_preedit_underlines.in_set(RenderUiSystems::ExtractCursor),
                    #[cfg(feature = "bevy_ui_debug")]
                    debug_overlay::extract_debug_overlay.in_set(RenderUiSystems::ExtractDebug),
                ),
            )
            .add_systems(
                Render,
                (
                    queue_uinodes.in_set(RenderSystems::Queue),
                    sort_phase_system::<TransparentUi>.in_set(RenderSystems::PhaseSort),
                    prepare_uinodes.in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(
                Core2d,
                ui_pass.after(Core2dSystems::PostProcess).before(upscaling),
            )
            .add_systems(
                Core3d,
                ui_pass.after(Core3dSystems::PostProcess).before(upscaling),
            );

        app.add_plugins(UiTextureSlicerPlugin);
        app.add_plugins(GradientPlugin);
        app.add_plugins(BoxShadowPlugin);
    }
}

#[derive(SystemParam)]
pub struct UiCameraMap<'w, 's> {
    mapping: Query<'w, 's, RenderEntity>,
}

impl<'w, 's> UiCameraMap<'w, 's> {
    /// Creates a [`UiCameraMapper`] for performing repeated camera-to-render-entity lookups.
    ///
    /// The last successful mapping is cached to avoid redundant queries.
    pub fn get_mapper(&'w self) -> UiCameraMapper<'w, 's> {
        UiCameraMapper {
            mapping: &self.mapping,
            camera_entity: Entity::PLACEHOLDER,
            render_entity: Entity::PLACEHOLDER,
        }
    }
}

/// Helper for mapping UI target camera entities to their corresponding render entities,
/// with caching to avoid repeated lookups for the same camera.
pub struct UiCameraMapper<'w, 's> {
    mapping: &'w Query<'w, 's, RenderEntity>,
    /// Cached camera entity from the last successful `map` call.
    camera_entity: Entity,
    /// Cached camera entity from the last successful `map` call.
    render_entity: Entity,
}

impl<'w, 's> UiCameraMapper<'w, 's> {
    /// Returns the render entity corresponding to the given [`ComputedUiTargetCamera`]'s camera, or none if no corresponding entity was found.
    pub fn map(&mut self, computed_target: &ComputedUiTargetCamera) -> Option<Entity> {
        let camera_entity = computed_target.get()?;
        if self.camera_entity != camera_entity {
            let new_render_camera_entity = self.mapping.get(camera_entity).ok()?;
            self.render_entity = new_render_camera_entity;
            self.camera_entity = camera_entity;
        }

        Some(self.render_entity)
    }

    /// Returns the cached camera entity from the last successful `map` call.
    pub fn current_camera(&self) -> Entity {
        self.camera_entity
    }
}

pub struct ExtractedUiNode {
    pub z_order: f32,
    pub image: AssetId<Image>,
    pub clip: Option<Rect>,
    pub item: ExtractedUiItem,
    pub transform: Affine2,
}

/// The type of UI node.
/// This is used to determine how to render the UI node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NodeType {
    Rect,
    Inverted,
    Border(u32), // shader flags
}

pub enum ExtractedUiItem {
    Node {
        color: LinearRgba,
        rect: Rect,
        atlas_scaling: Option<Vec2>,
        flip_x: bool,
        flip_y: bool,
        /// Border radius of the UI node.
        /// Ordering: top left, top right, bottom right, bottom left.
        border_radius: ResolvedBorderRadius,
        /// Border thickness of the UI node.
        /// Ordering: left, top, right, bottom.
        border: BorderRect,
        node_type: NodeType,
    },
    /// A contiguous sequence of text glyphs from the same section
    Glyphs {
        /// The color, position, and UV rect of each glyph.
        glyphs: Vec<ExtractedGlyph>,
    },
}

pub struct ExtractedGlyph {
    pub color: LinearRgba,
    pub translation: Vec2,
    pub rect: Rect,
}

/// The list of UI nodes, as well as the set of nodes that changed.
///
/// This is a two-level data structure so that we can quickly remove all
/// gradients associated with a main-world entity when it changes.
#[derive(Resource, Default)]
pub struct ExtractedUiNodes {
    /// The list of UI nodes grouped by their main-world entity, along with
    /// each group's target camera entity.
    ///
    /// This is a two-level data structure so that we can quickly remove all UI
    /// nodes associated with a main-world entity when it changes.
    pub uinodes: MainEntityHashMap<(Entity, EntityIndexMap<ExtractedUiNode>)>,
    /// UI nodes that changed this frame.
    pub changed: MainEntityHashSet,
    /// Render entities removed this frame, along with the camera phase that
    /// still contains their retained phase items.
    removed: Vec<RemovedUiNode>,
}

/// A query filter that matches all UI nodes.
type UiNodeQueryFilter = (
    With<ComputedNode>,
    With<ComputedStackIndex>,
    With<UiGlobalTransform>,
    With<InheritedVisibility>,
    With<ComputedUiTargetCamera>,
);

// Note: Whenever you add a new component that affects UI rendering, make sure
// to add a `Changed` query filter and a reference to the `RemovedComponents`
// resource to `extract_uinode_changes` below.
//
// Note: We don't have to match on `AssetChanged` for images or texture atlas
// layouts because the
// `bevy_ui::widget::mark_images_as_changed_as_their_assets_changed` image marks
// the `ImageNode` for us automatically as changed when those assets change.

/// A render-world system that scans for any UI nodes that have changed and
/// removes the render world data associated with them.
pub fn extract_uinode_changes(
    mut commands: Commands,
    mut extracted_uinodes: ResMut<ExtractedUiNodes>,
    all_uinodes_query: Extract<Query<Entity, UiNodeQueryFilter>>,
    changed_uinodes_query: Extract<
        Query<
            Entity,
            (
                UiNodeQueryFilter,
                Or<(
                    Or<(
                        Changed<ComputedNode>,
                        Changed<ComputedStackIndex>,
                        Changed<UiGlobalTransform>,
                        Changed<InheritedVisibility>,
                        Changed<CalculatedClip>,
                        Changed<ComputedUiTargetCamera>,
                        Changed<BackgroundColor>,
                        Changed<OuterColor>,
                    )>,
                    Or<(
                        Changed<ImageNode>,
                        Changed<ImageNodeSize>,
                        Changed<BorderColor>,
                        Changed<Outline>,
                        Changed<ViewportNode>,
                        Changed<ComputedTextBlock>,
                        Changed<TextColor>,
                        Changed<TextLayoutInfo>,
                    )>,
                    Or<(
                        Changed<TextCursorStyle>,
                        Changed<TextShadow>,
                        Changed<BackgroundGradient>,
                        Changed<BorderGradient>,
                        Changed<BoxShadow>,
                        Changed<EditableText>,
                        Changed<Underline>,
                        Changed<Strikethrough>,
                    )>,
                    Or<(Changed<StrikethroughColor>, Changed<UnderlineColor>)>,
                )>,
            ),
        >,
    >,
    #[cfg(feature = "bevy_ui_debug")] changed_debug_options_query: Extract<
        Query<Entity, (UiNodeQueryFilter, Changed<UiDebugOptions>)>,
    >,
    text_span_query: Extract<
        Query<
            Entity,
            (
                With<TextSpan>,
                Or<(
                    Changed<TextColor>,
                    Changed<TextBackgroundColor>,
                    Changed<Underline>,
                    Changed<Strikethrough>,
                    Changed<StrikethroughColor>,
                    Changed<UnderlineColor>,
                )>,
            ),
        >,
    >,
    text_span_parent_query: Extract<Query<&ChildOf, With<TextSpan>>>,
    text_query: Extract<Query<Entity, With<Text>>>,
    (
        mut removed_computed_node_query,
        mut removed_computed_stack_index_query,
        mut removed_ui_global_transform_query,
        mut removed_inherited_visibility_query,
        mut removed_calculated_clip_query,
        mut removed_computed_ui_target_camera_query,
        mut removed_background_color_query,
        mut removed_outer_color_query,
    ): (
        Extract<RemovedComponents<ComputedNode>>,
        Extract<RemovedComponents<ComputedStackIndex>>,
        Extract<RemovedComponents<UiGlobalTransform>>,
        Extract<RemovedComponents<InheritedVisibility>>,
        Extract<RemovedComponents<CalculatedClip>>,
        Extract<RemovedComponents<ComputedUiTargetCamera>>,
        Extract<RemovedComponents<BackgroundColor>>,
        Extract<RemovedComponents<OuterColor>>,
    ),
    (
        mut removed_image_node_query,
        mut removed_image_node_size_query,
        mut removed_border_color_query,
        mut removed_outline_query,
        mut removed_viewport_node_query,
        mut removed_computed_text_block_query,
        mut removed_text_color_query,
        mut removed_text_layout_info_query,
    ): (
        Extract<RemovedComponents<ImageNode>>,
        Extract<RemovedComponents<ImageNodeSize>>,
        Extract<RemovedComponents<BorderColor>>,
        Extract<RemovedComponents<Outline>>,
        Extract<RemovedComponents<ViewportNode>>,
        Extract<RemovedComponents<ComputedTextBlock>>,
        Extract<RemovedComponents<TextColor>>,
        Extract<RemovedComponents<TextLayoutInfo>>,
    ),
    (
        mut removed_text_cursor_style_query,
        mut removed_text_shadow_query,
        mut removed_background_gradient_query,
        mut removed_border_gradient_query,
        mut removed_box_shadow_query,
        mut removed_editable_text_query,
        mut removed_underline_query,
        mut removed_strikethrough_query,
    ): (
        Extract<RemovedComponents<TextCursorStyle>>,
        Extract<RemovedComponents<TextShadow>>,
        Extract<RemovedComponents<BackgroundGradient>>,
        Extract<RemovedComponents<BorderGradient>>,
        Extract<RemovedComponents<BoxShadow>>,
        Extract<RemovedComponents<EditableText>>,
        Extract<RemovedComponents<Underline>>,
        Extract<RemovedComponents<Strikethrough>>,
    ),
    (mut removed_strikethrough_color_query, mut removed_underline_color_query): (
        Extract<RemovedComponents<StrikethroughColor>>,
        Extract<RemovedComponents<UnderlineColor>>,
    ),
    #[cfg(feature = "bevy_ui_debug")] mut removed_debug_options_query: Extract<
        RemovedComponents<UiDebugOptions>,
    >,
    #[cfg(feature = "bevy_ui_debug")] global_ui_debug_options: Extract<Res<GlobalUiDebugOptions>>,
    mut extra_nodes_to_invalidate: Local<MainEntityHashSet>,
) {
    extracted_uinodes.changed.clear();
    extracted_uinodes.removed.clear();

    // If the debug options changed, we wipe everything.
    // That's a bit coarse-grained, but having the debug options change is rare
    // and should only happen in, well, debugging.
    #[cfg(feature = "bevy_ui_debug")]
    let must_wipe_all_nodes = global_ui_debug_options.is_changed();
    #[cfg(not(feature = "bevy_ui_debug"))]
    let must_wipe_all_nodes = false;

    if must_wipe_all_nodes {
        for main_entity in &all_uinodes_query {
            process_changed_entity(
                main_entity.into(),
                &mut commands,
                &text_span_parent_query,
                &text_query,
                &mut extracted_uinodes,
                Some(&mut extra_nodes_to_invalidate),
            );
        }
    } else {
        // Go through all nodes that have changed and invalidate any render world
        // data associated with them.
        for main_entity in changed_uinodes_query
            .iter()
            .chain(text_span_query.iter())
            .chain(removed_computed_node_query.read())
            .chain(removed_computed_stack_index_query.read())
            .chain(removed_ui_global_transform_query.read())
            .chain(removed_inherited_visibility_query.read())
            .chain(removed_calculated_clip_query.read())
            .chain(removed_computed_ui_target_camera_query.read())
            .chain(removed_background_color_query.read())
            .chain(removed_outer_color_query.read())
            .chain(removed_image_node_query.read())
            .chain(removed_image_node_size_query.read())
            .chain(removed_border_color_query.read())
            .chain(removed_outline_query.read())
            .chain(removed_viewport_node_query.read())
            .chain(removed_computed_text_block_query.read())
            .chain(removed_text_color_query.read())
            .chain(removed_text_layout_info_query.read())
            .chain(removed_text_cursor_style_query.read())
            .chain(removed_text_shadow_query.read())
            .chain(removed_background_gradient_query.read())
            .chain(removed_border_gradient_query.read())
            .chain(removed_box_shadow_query.read())
            .chain(removed_editable_text_query.read())
            .chain(removed_underline_query.read())
            .chain(removed_strikethrough_query.read())
            .chain(removed_strikethrough_color_query.read())
            .chain(removed_underline_color_query.read())
        {
            process_changed_entity(
                main_entity.into(),
                &mut commands,
                &text_span_parent_query,
                &text_query,
                &mut extracted_uinodes,
                Some(&mut extra_nodes_to_invalidate),
            );
        }

        // Process nodes that have changed debug options too, if that feature is
        // enabled.
        #[cfg(feature = "bevy_ui_debug")]
        for main_entity in changed_debug_options_query
            .iter()
            .chain(removed_debug_options_query.read())
        {
            process_changed_entity(
                main_entity.into(),
                &mut commands,
                &text_span_parent_query,
                &text_query,
                &mut extracted_uinodes,
                Some(&mut extra_nodes_to_invalidate),
            );
        }
    }

    for main_entity in extra_nodes_to_invalidate.drain() {
        process_changed_entity(
            main_entity,
            &mut commands,
            &text_span_parent_query,
            &text_query,
            &mut extracted_uinodes,
            None,
        );
    }

    fn process_changed_entity(
        mut main_entity: MainEntity,
        commands: &mut Commands,
        text_span_parent_query: &Query<&ChildOf, With<TextSpan>>,
        text_query: &Query<Entity, With<Text>>,
        extracted_uinodes: &mut ExtractedUiNodes,
        maybe_extra_nodes_to_invalidate: Option<&mut MainEntityHashSet>,
    ) {
        // Mark the node as changed so that the other `extract_` systems will
        // know to process it.
        extracted_uinodes.changed.insert(main_entity);

        if let Some((camera_entity, mut render_entities)) =
            extracted_uinodes.uinodes.remove(&main_entity)
        {
            for (render_entity, _) in render_entities.drain(..) {
                extracted_uinodes.removed.push(RemovedUiNode {
                    main_entity,
                    render_entity,
                    camera_entity,
                });
                commands.entity(render_entity).despawn();
            }
        }

        // If this node is a `TextSpan`, then we need to invalidate the ancestor
        // `Text` node too. This is because `extract_text_decorations` only
        // looks at the `text_background_colors_query` for the text spans if the
        // `uinode_query` that it's iterating over matched the ancestor `Text`
        // node.
        if let Some(extra_nodes_to_invalidate) = maybe_extra_nodes_to_invalidate
            && let Ok(parent) = text_span_parent_query.get(main_entity.entity())
        {
            main_entity = parent.parent().into();
            loop {
                if text_query.contains(main_entity.entity()) {
                    extra_nodes_to_invalidate.insert(main_entity);
                    break;
                }
                match text_span_parent_query.get(main_entity.entity()) {
                    Ok(parent) => main_entity = parent.parent().into(),
                    Err(_) => break,
                }
            }
        }
    }
}

pub fn extract_uinode_background_colors(
    mut commands: Commands,
    extracted_uinodes: ResMut<ExtractedUiNodes>,
    uinode_query: Extract<
        Query<(
            Entity,
            &ComputedNode,
            &ComputedStackIndex,
            &UiGlobalTransform,
            &InheritedVisibility,
            Option<&CalculatedClip>,
            &ComputedUiTargetCamera,
            &BackgroundColor,
            Option<&OuterColor>,
        )>,
    >,
    camera_map: Extract<UiCameraMap>,
) {
    let extracted_uinodes = extracted_uinodes.into_inner();
    let mut camera_mapper = camera_map.get_mapper();

    for (
        entity,
        uinode,
        stack_index,
        transform,
        inherited_visibility,
        clip,
        camera,
        background_color,
        maybe_outer_color,
    ) in extracted_uinodes
        .changed
        .iter()
        .flat_map(|main_entity| uinode_query.get(main_entity.entity()).ok())
    {
        // Skip invisible backgrounds
        if !inherited_visibility.get()
            || (background_color.is_fully_transparent()
                && maybe_outer_color.is_none_or(|outer| outer.is_fully_transparent()))
            || uinode.is_empty()
        {
            continue;
        }

        let extracted_sub_uinodes = match extracted_uinodes.uinodes.entry(entity.into()) {
            Entry::Occupied(entry) => &mut entry.into_mut().1,
            Entry::Vacant(entry) => {
                let Some(extracted_camera_entity) = camera_mapper.map(camera) else {
                    continue;
                };
                &mut entry
                    .insert((extracted_camera_entity, Default::default()))
                    .1
            }
        };

        if !background_color.is_fully_transparent() {
            extracted_sub_uinodes.insert(
                commands.spawn_empty().id(),
                ExtractedUiNode {
                    z_order: stack_index.0 as f32 + stack_z_offsets::BACKGROUND_COLOR,
                    clip: clip.map(|clip| clip.clip),
                    image: AssetId::default(),
                    transform: transform.into(),
                    item: ExtractedUiItem::Node {
                        color: background_color.0.into(),
                        rect: Rect {
                            min: Vec2::ZERO,
                            max: uinode.size,
                        },
                        atlas_scaling: None,
                        flip_x: false,
                        flip_y: false,
                        border: uinode.border(),
                        border_radius: uinode.border_radius(),
                        node_type: NodeType::Rect,
                    },
                },
            );
        }

        if let Some(outer_color) = maybe_outer_color
            && !outer_color.0.is_fully_transparent()
        {
            extracted_sub_uinodes.insert(
                commands.spawn_empty().id(),
                ExtractedUiNode {
                    z_order: stack_index.0 as f32 + stack_z_offsets::BACKGROUND_COLOR,
                    clip: clip.map(|clip| clip.clip),
                    image: AssetId::default(),
                    transform: transform.into(),
                    item: ExtractedUiItem::Node {
                        color: outer_color.0.into(),
                        rect: Rect {
                            min: Vec2::ZERO,
                            max: uinode.size,
                        },
                        atlas_scaling: None,
                        flip_x: false,
                        flip_y: false,
                        border: BorderRect::ZERO,
                        border_radius: uinode.border_radius(),
                        node_type: NodeType::Inverted,
                    },
                },
            );
        }
    }
}

pub fn extract_uinode_images(
    mut commands: Commands,
    extracted_uinodes: ResMut<ExtractedUiNodes>,
    texture_atlases: Extract<Res<Assets<TextureAtlasLayout>>>,
    uinode_query: Extract<
        Query<(
            Entity,
            &ComputedNode,
            &ComputedStackIndex,
            &UiGlobalTransform,
            &InheritedVisibility,
            Option<&CalculatedClip>,
            &ComputedUiTargetCamera,
            &ImageNode,
            &ImageNodeSize,
        )>,
    >,
    camera_map: Extract<UiCameraMap>,
) {
    let extracted_uinodes = extracted_uinodes.into_inner();
    let mut camera_mapper = camera_map.get_mapper();

    for (
        entity,
        uinode,
        stack_index,
        transform,
        inherited_visibility,
        clip,
        camera,
        image,
        image_size,
    ) in extracted_uinodes
        .changed
        .iter()
        .flat_map(|main_entity| uinode_query.get(main_entity.entity()).ok())
    {
        let visual_box = match image.visual_box {
            VisualBox::ContentBox => uinode.content_box(),
            VisualBox::PaddingBox => uinode.padding_box(),
            VisualBox::BorderBox => uinode.border_box(),
        };
        // Skip invisible images
        if !inherited_visibility.get()
            || image.color.is_fully_transparent()
            || image.image.id() == TRANSPARENT_IMAGE_HANDLE.id()
            || image.image_mode.uses_slices()
            || visual_box.size().cmple(Vec2::ZERO).any()
        {
            continue;
        }

        let Some(extracted_camera_entity) = camera_mapper.map(camera) else {
            continue;
        };

        let size = if matches!(image.image_mode, NodeImageMode::Auto) {
            let source = image_size.size().as_vec2();
            if source.cmple(Vec2::ZERO).any() {
                visual_box.size()
            } else {
                source * (visual_box.size() / source).min_element()
            }
        } else {
            visual_box.size()
        };

        let atlas_rect = image
            .texture_atlas
            .as_ref()
            .and_then(|s| s.texture_rect(&texture_atlases))
            .map(|r| r.as_rect());

        let mut rect = match (atlas_rect, image.rect) {
            (None, None) => Rect {
                min: Vec2::ZERO,
                max: size,
            },
            (None, Some(image_rect)) => image_rect,
            (Some(atlas_rect), None) => atlas_rect,
            (Some(atlas_rect), Some(mut image_rect)) => {
                image_rect.min += atlas_rect.min;
                image_rect.max += atlas_rect.min;
                image_rect
            }
        };

        let atlas_scaling = if atlas_rect.is_some() || image.rect.is_some() {
            let atlas_scaling = size / rect.size();
            rect.min *= atlas_scaling;
            rect.max *= atlas_scaling;
            Some(atlas_scaling)
        } else {
            None
        };

        extracted_uinodes
            .uinodes
            .entry(entity.into())
            .or_insert_with(|| (extracted_camera_entity, Default::default()))
            .1
            .insert(
                commands.spawn_empty().id(),
                ExtractedUiNode {
                    z_order: stack_index.0 as f32 + stack_z_offsets::IMAGE,
                    clip: clip.map(|clip| clip.clip),
                    image: image.image.id(),
                    transform: Affine2::from(*transform)
                        * Affine2::from_translation(visual_box.center()),
                    item: ExtractedUiItem::Node {
                        color: image.color.into(),
                        rect,
                        atlas_scaling,
                        flip_x: image.flip_x,
                        flip_y: image.flip_y,
                        border: BorderRect::ZERO,
                        border_radius: uinode.border_radius,
                        node_type: NodeType::Rect,
                    },
                },
            );
    }
}

pub fn extract_uinode_borders(
    mut commands: Commands,
    extracted_uinodes: ResMut<ExtractedUiNodes>,
    uinode_query: Extract<
        Query<(
            Entity,
            Option<&Node>,
            &ComputedNode,
            &ComputedStackIndex,
            &UiGlobalTransform,
            &InheritedVisibility,
            Option<&CalculatedClip>,
            &ComputedUiTargetCamera,
            AnyOf<(&BorderColor, &Outline)>,
        )>,
    >,
    camera_map: Extract<UiCameraMap>,
) {
    let extracted_uinodes = extracted_uinodes.into_inner();
    let image = AssetId::<Image>::default();
    let mut camera_mapper = camera_map.get_mapper();

    for (
        entity,
        node,
        computed_node,
        stack_index,
        transform,
        inherited_visibility,
        maybe_clip,
        camera,
        (maybe_border_color, maybe_outline),
    ) in extracted_uinodes
        .changed
        .iter()
        .flat_map(|main_entity| uinode_query.get(main_entity.entity()).ok())
    {
        // Skip invisible borders and removed nodes
        if !inherited_visibility.get() || node.is_some_and(|node| node.display == Display::None) {
            continue;
        }

        let Some(extracted_camera_entity) = camera_mapper.map(camera) else {
            continue;
        };

        // Don't extract borders with zero width along all edges
        if computed_node.border() != BorderRect::ZERO
            && let Some(border_color) = maybe_border_color
        {
            let border_colors = [
                border_color.left.to_linear(),
                border_color.top.to_linear(),
                border_color.right.to_linear(),
                border_color.bottom.to_linear(),
            ];

            const BORDER_FLAGS: [u32; 4] = [
                shader_flags::BORDER_LEFT,
                shader_flags::BORDER_TOP,
                shader_flags::BORDER_RIGHT,
                shader_flags::BORDER_BOTTOM,
            ];
            let mut completed_flags = 0;

            for (i, &color) in border_colors.iter().enumerate() {
                if color.is_fully_transparent() {
                    continue;
                }

                let mut border_flags = BORDER_FLAGS[i];

                if completed_flags & border_flags != 0 {
                    continue;
                }

                for j in i + 1..4 {
                    if color == border_colors[j] {
                        border_flags |= BORDER_FLAGS[j];
                    }
                }
                completed_flags |= border_flags;

                let node = ExtractedUiNode {
                    z_order: stack_index.0 as f32 + stack_z_offsets::BORDER,
                    image,
                    clip: maybe_clip.map(|clip| clip.clip),
                    transform: transform.into(),
                    item: ExtractedUiItem::Node {
                        color,
                        rect: Rect {
                            max: computed_node.size(),
                            ..Default::default()
                        },
                        atlas_scaling: None,
                        flip_x: false,
                        flip_y: false,
                        border: computed_node.border(),
                        border_radius: computed_node.border_radius(),
                        node_type: NodeType::Border(border_flags),
                    },
                };

                extracted_uinodes
                    .uinodes
                    .entry(entity.into())
                    .or_insert_with(|| (extracted_camera_entity, Default::default()))
                    .1
                    .insert(commands.spawn_empty().id(), node);
            }
        }

        if computed_node.outline_width() <= 0. {
            continue;
        }

        if let Some(outline) = maybe_outline.filter(|outline| !outline.color.is_fully_transparent())
        {
            let outline_size = computed_node.outlined_node_size();
            extracted_uinodes
                .uinodes
                .entry(entity.into())
                .or_insert_with(|| (extracted_camera_entity, Default::default()))
                .1
                .insert(
                    commands.spawn_empty().id(),
                    ExtractedUiNode {
                        z_order: stack_index.0 as f32 + stack_z_offsets::BORDER,
                        image,
                        clip: maybe_clip.map(|clip| clip.clip),
                        transform: transform.into(),
                        item: ExtractedUiItem::Node {
                            color: outline.color.into(),
                            rect: Rect {
                                max: outline_size,
                                ..Default::default()
                            },
                            atlas_scaling: None,
                            flip_x: false,
                            flip_y: false,
                            border: BorderRect::all(computed_node.outline_width()),
                            border_radius: computed_node.outline_radius(),
                            node_type: NodeType::Border(shader_flags::BORDER_ALL),
                        },
                    },
                );
        }
    }
}

/// The UI camera is "moved back" by this many units (plus the [`UI_CAMERA_TRANSFORM_OFFSET`]) and also has a view
/// distance of this many units. This ensures that with a left-handed projection,
/// as UI elements are "stacked on top of each other", they are within the camera's view
/// and have room to grow.
// TODO: Consider computing this value at runtime based on the maximum z-value.
const UI_CAMERA_FAR: f32 = 1000.0;

// This value is subtracted from the far distance for the camera's z-position to ensure nodes at z == 0.0 are rendered
// TODO: Evaluate if we still need this.
const UI_CAMERA_TRANSFORM_OFFSET: f32 = -0.1;

/// The ID of the subview associated with a camera on which UI is to be drawn.
///
/// When UI is present, cameras extract to two views: the main 2D/3D one and a
/// UI one. The main 2D or 3D camera gets subview 0, and the corresponding UI
/// camera gets this subview, 1.
const UI_CAMERA_SUBVIEW: u32 = 1;

/// A render-world component that lives on the main render target view and
/// specifies the corresponding UI view.
///
/// For example, if UI is being rendered to a 3D camera, this component lives on
/// the 3D camera and contains the entity corresponding to the UI view.
#[derive(Component)]
/// Entity id of the temporary render entity with the corresponding extracted UI view.
pub struct UiCameraView(pub Entity);

/// A render-world component that lives on the UI view and specifies the
/// corresponding main render target view.
///
/// For example, if the UI is being rendered to a 3D camera, this component
/// lives on the UI view and contains the entity corresponding to the 3D camera.
///
/// This is the inverse of [`UiCameraView`].
#[derive(Component)]
pub struct UiViewTarget(pub Entity);

/// Information that [`extract_ui_camera_view`] maintains about each view that
/// it has seen.
pub struct CachedUiViewData {
    /// The render-world [`ExtractedView`].
    extracted_view_entity: Entity,
    /// The unique, stable identifier for the view across frames.
    retained_view_entity: RetainedViewEntity,
}

/// Extracts all UI elements associated with a camera into the render world.
pub fn extract_ui_camera_view(
    mut commands: Commands,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<TransparentUi>>,
    query: Extract<
        Query<
            (
                Entity,
                RenderEntity,
                &Camera,
                Option<&UiAntiAlias>,
                Option<&BoxShadowSamples>,
            ),
            Or<(With<Camera2d>, With<Camera3d>)>,
        >,
    >,
    main_pass_formats: Res<CameraMainPassTextureFormats>,
    mut live_entities: Local<HashSet<RetainedViewEntity>>,
    mut cached_ui_view_data: Local<MainEntityHashMap<CachedUiViewData>>,
    mut removed_cameras_query: Extract<RemovedComponents<Camera>>,
    mut cameras_updated_this_frame: Local<MainEntityHashSet>,
) {
    cameras_updated_this_frame.clear();
    for (main_entity, render_entity, camera, ui_anti_alias, shadow_samples) in &query {
        let main_entity = MainEntity::from(main_entity);
        let retained_view_entity = RetainedViewEntity::new(main_entity, None, UI_CAMERA_SUBVIEW);

        // ignore inactive cameras
        if let (Some(physical_viewport_rect), Some(target_size), Some(target_format)) = (
            camera.physical_viewport_rect(),
            camera.physical_target_size(),
            main_pass_formats.get(&render_entity).copied(),
        ) && target_size.x != 0
            && target_size.y != 0
            && camera.physical_viewport_size().is_some()
            && camera.is_active
        {
            cameras_updated_this_frame.insert(main_entity);
            transparent_render_phases.prepare_for_new_frame(retained_view_entity);

            // use a projection matrix with the origin in the top left instead of the bottom left that comes with OrthographicProjection
            let projection_matrix = Mat4::orthographic_rh(
                0.0,
                physical_viewport_rect.width() as f32,
                physical_viewport_rect.height() as f32,
                0.0,
                0.0,
                UI_CAMERA_FAR,
            );
            // We use `UI_CAMERA_SUBVIEW` here so as not to conflict with the
            // main 3D or 2D camera, which will have subview index 0.
            // Creates the UI view.
            let extracted_view = ExtractedView {
                retained_view_entity,
                clip_from_view: projection_matrix,
                world_from_view: GlobalTransform::from_xyz(
                    0.0,
                    0.0,
                    UI_CAMERA_FAR + UI_CAMERA_TRANSFORM_OFFSET,
                ),
                clip_from_world: None,
                target_format,
                viewport: UVec4::from((physical_viewport_rect.min, physical_viewport_rect.size())),
                color_grading: Default::default(),
                invert_culling: false,
            };
            // Link to the main camera view.
            let ui_view_target_component = UiViewTarget(render_entity);

            let ui_camera_view = match cached_ui_view_data.get(&main_entity) {
                Some(cached_ui_view_data) => commands
                    .entity(cached_ui_view_data.extracted_view_entity)
                    .insert((extracted_view, ui_view_target_component))
                    .id(),
                None => commands
                    .spawn((extracted_view, ui_view_target_component))
                    .id(),
            };

            let mut entity_commands = commands
                .get_entity(render_entity)
                .expect("Camera entity wasn't synced.");
            // Link from the main 2D/3D camera view to the UI view.
            entity_commands.insert(UiCameraView(ui_camera_view));
            if let Some(ui_anti_alias) = ui_anti_alias {
                entity_commands.insert(*ui_anti_alias);
            }
            if let Some(shadow_samples) = shadow_samples {
                entity_commands.insert(*shadow_samples);
            }

            live_entities.insert(retained_view_entity);
            cached_ui_view_data.insert(
                main_entity,
                CachedUiViewData {
                    extracted_view_entity: ui_camera_view,
                    retained_view_entity,
                },
            );
            continue;
        }

        // If we got here, the camera no longer exists or is no longer
        // renderable. Remove its associated render-world data.
        commands
            .get_entity(render_entity)
            .expect("Camera entity wasn't synced.")
            .remove::<(UiCameraView, UiAntiAlias, BoxShadowSamples)>();
        live_entities.remove(&retained_view_entity);
        if let Some(cached_ui_view_data) = cached_ui_view_data.remove(&main_entity) {
            commands
                .entity(cached_ui_view_data.extracted_view_entity)
                .despawn();
        }
    }

    // Only remove the render-world data for a camera if we didn't handle the
    // camera above.
    // It's possible that the `Camera` component was removed and added in the
    // same frame.
    for main_entity in removed_cameras_query.read() {
        let main_entity = MainEntity::from(main_entity);
        if cameras_updated_this_frame.contains(&main_entity) {
            continue;
        }

        if let Some(cached_ui_view_data) = cached_ui_view_data.remove(&main_entity) {
            commands
                .entity(cached_ui_view_data.extracted_view_entity)
                .despawn();
            live_entities.remove(&cached_ui_view_data.retained_view_entity);
        }
    }

    // Clean up render phases belonging to cameras that no longer exist.
    transparent_render_phases.retain(|entity, _| live_entities.contains(entity));
}

pub fn extract_viewport_nodes(
    mut commands: Commands,
    extracted_uinodes: ResMut<ExtractedUiNodes>,
    camera_query: Extract<Query<(&Camera, &RenderTarget)>>,
    uinode_query: Extract<
        Query<(
            Entity,
            &ComputedNode,
            &ComputedStackIndex,
            &UiGlobalTransform,
            &InheritedVisibility,
            Option<&CalculatedClip>,
            &ComputedUiTargetCamera,
            &ViewportNode,
        )>,
    >,
    camera_map: Extract<UiCameraMap>,
) {
    let extracted_uinodes = extracted_uinodes.into_inner();
    let mut camera_mapper = camera_map.get_mapper();

    for (
        entity,
        uinode,
        stack_index,
        transform,
        inherited_visibility,
        clip,
        camera,
        viewport_node,
    ) in extracted_uinodes
        .changed
        .iter()
        .flat_map(|main_entity| uinode_query.get(main_entity.entity()).ok())
    {
        // Skip invisible images
        if !inherited_visibility.get() || uinode.is_empty() {
            continue;
        }

        let Some(extracted_camera_entity) = camera_mapper.map(camera) else {
            continue;
        };
        let Some(camera_entity) = viewport_node.camera else {
            continue;
        };

        let Some(image) = camera_query
            .get(camera_entity)
            .ok()
            .and_then(|(_, render_target)| render_target.as_image())
        else {
            continue;
        };

        extracted_uinodes
            .uinodes
            .entry(entity.into())
            .or_insert_with(|| (extracted_camera_entity, Default::default()))
            .1
            .insert(
                commands.spawn_empty().id(),
                ExtractedUiNode {
                    z_order: stack_index.0 as f32 + stack_z_offsets::IMAGE,
                    clip: clip.map(|clip| clip.clip),
                    image: image.id(),
                    transform: transform.into(),
                    item: ExtractedUiItem::Node {
                        color: LinearRgba::WHITE,
                        rect: Rect {
                            min: Vec2::ZERO,
                            max: uinode.size,
                        },
                        atlas_scaling: None,
                        flip_x: false,
                        flip_y: false,
                        border: uinode.border(),
                        border_radius: uinode.border_radius(),
                        node_type: NodeType::Rect,
                    },
                },
            );
    }
}

pub fn extract_text_sections(
    mut commands: Commands,
    extracted_uinodes: ResMut<ExtractedUiNodes>,
    uinode_query: Extract<
        Query<(
            Entity,
            &ComputedNode,
            &ComputedStackIndex,
            &UiGlobalTransform,
            &InheritedVisibility,
            Option<&CalculatedClip>,
            &ComputedUiTargetCamera,
            &ComputedTextBlock,
            &TextColor,
            &TextLayoutInfo,
            Option<&EditableText>,
            Option<&TextCursorStyle>,
        )>,
    >,
    text_styles: Extract<Query<&TextColor>>,
    camera_map: Extract<UiCameraMap>,
) {
    let extracted_uinodes = extracted_uinodes.into_inner();
    let mut camera_mapper = camera_map.get_mapper();

    let mut glyphs = vec![];

    for (
        entity,
        uinode,
        stack_index,
        global_transform,
        inherited_visibility,
        maybe_clip,
        camera,
        computed_block,
        text_color,
        text_layout_info,
        editable_text,
        cursor_style,
    ) in extracted_uinodes
        .changed
        .iter()
        .flat_map(|main_entity| uinode_query.get(main_entity.entity()).ok())
    {
        // Skip if not visible or if size is set to zero (e.g. when a parent is set to `Display::None`)
        if !inherited_visibility.get() || uinode.is_empty() {
            continue;
        }

        let Some(extracted_camera_entity) = camera_mapper.map(camera) else {
            continue;
        };

        let transform = Affine2::from(*global_transform)
            * Affine2::from_translation(
                uinode.content_box().min
                    - editable_text.map_or(Vec2::ZERO, |text| text.viewport.offset),
            );

        let clip = if editable_text.is_some() {
            let content_box = uinode.content_box();
            let text_clip = Rect::from_center_size(
                global_transform.affine().translation + content_box.center(),
                content_box.size(),
            );
            Some(maybe_clip.map_or(text_clip, |clip| clip.clip.intersect(text_clip)))
        } else {
            maybe_clip.map(|clip| clip.clip)
        };

        let mut color = text_color.0.to_linear();

        let selected_text_color = cursor_style
            .and_then(|cursor_style| cursor_style.selected_text_color)
            .map(|selected_text_color| selected_text_color.to_linear());

        let mut current_section_index = 0;

        for (
            i,
            PositionedGlyph {
                position,
                atlas_info,
                section_index,
                ..
            },
        ) in text_layout_info.glyphs.iter().enumerate()
        {
            if current_section_index != *section_index
                && let Some(section_entity) = computed_block
                    .entities()
                    .get(*section_index)
                    .map(|t| t.entity)
            {
                color = text_styles
                    .get(section_entity)
                    .map(|text_color| LinearRgba::from(text_color.0))
                    .unwrap_or_default();
                current_section_index = *section_index;
            }

            let color = if !atlas_info.is_alpha_mask {
                LinearRgba::WHITE
            } else if let Some(selected_text_color) = selected_text_color
                && text_layout_info
                    .selection_rects
                    .iter()
                    .any(|selection_rect| {
                        let glyph_rect = Rect::from_center_size(*position, atlas_info.rect.size());
                        selection_rect.contains(glyph_rect.min)
                            && selection_rect.contains(glyph_rect.max)
                    })
            {
                selected_text_color
            } else {
                color
            };

            glyphs.push(ExtractedGlyph {
                color,
                translation: *position,
                rect: atlas_info.rect,
            });

            if text_layout_info
                .glyphs
                .get(i + 1)
                .is_none_or(|info| info.atlas_info.texture != atlas_info.texture)
            {
                extracted_uinodes
                    .uinodes
                    .entry(entity.into())
                    .or_insert_with(|| (extracted_camera_entity, Default::default()))
                    .1
                    .insert(
                        commands.spawn_empty().id(),
                        ExtractedUiNode {
                            z_order: stack_index.0 as f32 + stack_z_offsets::TEXT,
                            image: atlas_info.texture,
                            clip,
                            item: ExtractedUiItem::Glyphs {
                                glyphs: mem::take(&mut glyphs),
                            },
                            transform,
                        },
                    );
            }
        }
    }
}

pub fn extract_text_shadows(
    mut commands: Commands,
    extracted_uinodes: ResMut<ExtractedUiNodes>,
    uinode_query: Extract<
        Query<(
            Entity,
            &ComputedNode,
            &ComputedStackIndex,
            &UiGlobalTransform,
            &ComputedUiTargetCamera,
            &InheritedVisibility,
            Option<&CalculatedClip>,
            &TextLayoutInfo,
            &TextShadow,
            &ComputedTextBlock,
            Option<&EditableText>,
        )>,
    >,
    text_decoration_query: Extract<Query<(Has<Strikethrough>, Has<Underline>)>>,
    camera_map: Extract<UiCameraMap>,
) {
    let extracted_uinodes = extracted_uinodes.into_inner();
    let mut camera_mapper = camera_map.get_mapper();

    let mut glyphs = vec![];

    for (
        entity,
        uinode,
        stack_index,
        global_transform,
        target,
        inherited_visibility,
        maybe_clip,
        text_layout_info,
        shadow,
        computed_block,
        editable_text,
    ) in extracted_uinodes
        .changed
        .iter()
        .flat_map(|main_entity| uinode_query.get(main_entity.entity()).ok())
    {
        // Skip if not visible or if size is set to zero (e.g. when a parent is set to `Display::None`)
        if !inherited_visibility.get() || uinode.is_empty() {
            continue;
        }

        let Some(extracted_camera_entity) = camera_mapper.map(target) else {
            continue;
        };

        let node_transform = Affine2::from(*global_transform)
            * Affine2::from_translation(
                uinode.content_box().min + shadow.offset / uinode.inverse_scale_factor()
                    - editable_text.map_or(Vec2::ZERO, |text| text.viewport.offset),
            );

        let clip = if editable_text.is_some() {
            let content_box = uinode.content_box();
            let text_clip = Rect::from_center_size(
                global_transform.affine().translation + content_box.center(),
                content_box.size(),
            );
            Some(maybe_clip.map_or(text_clip, |clip| clip.clip.intersect(text_clip)))
        } else {
            maybe_clip.map(|clip| clip.clip)
        };

        for (
            i,
            PositionedGlyph {
                position,
                atlas_info,
                section_index,
                ..
            },
        ) in text_layout_info.glyphs.iter().enumerate()
        {
            glyphs.push(ExtractedGlyph {
                color: shadow.color.into(),
                translation: *position,
                rect: atlas_info.rect,
            });

            if text_layout_info.glyphs.get(i + 1).is_none_or(|info| {
                info.section_index != *section_index
                    || info.atlas_info.texture != atlas_info.texture
            }) {
                extracted_uinodes
                    .uinodes
                    .entry(entity.into())
                    .or_insert_with(|| (extracted_camera_entity, Default::default()))
                    .1
                    .insert(
                        commands.spawn_empty().id(),
                        ExtractedUiNode {
                            transform: node_transform,
                            z_order: stack_index.0 as f32 + stack_z_offsets::TEXT,
                            image: atlas_info.texture,
                            clip,
                            item: ExtractedUiItem::Glyphs {
                                glyphs: mem::take(&mut glyphs),
                            },
                        },
                    );
            }
        }

        for run in text_layout_info.run_geometry.iter() {
            let Some(section_entity) = computed_block
                .entities()
                .get(run.section_index)
                .map(|t| t.entity)
            else {
                continue;
            };
            let Ok((has_strikethrough, has_underline)) = text_decoration_query.get(section_entity)
            else {
                continue;
            };

            if has_strikethrough {
                extracted_uinodes
                    .uinodes
                    .entry(entity.into())
                    .or_insert_with(|| (extracted_camera_entity, Default::default()))
                    .1
                    .insert(
                        commands.spawn_empty().id(),
                        ExtractedUiNode {
                            z_order: stack_index.0 as f32 + stack_z_offsets::TEXT,
                            clip,
                            image: AssetId::default(),
                            transform: node_transform
                                * Affine2::from_translation(run.strikethrough_position()),
                            item: ExtractedUiItem::Node {
                                color: shadow.color.into(),
                                rect: Rect {
                                    min: Vec2::ZERO,
                                    max: run.strikethrough_size(),
                                },
                                atlas_scaling: None,
                                flip_x: false,
                                flip_y: false,
                                border: BorderRect::ZERO,
                                border_radius: ResolvedBorderRadius::ZERO,
                                node_type: NodeType::Rect,
                            },
                        },
                    );
            }

            if has_underline {
                extracted_uinodes
                    .uinodes
                    .entry(entity.into())
                    .or_insert_with(|| (extracted_camera_entity, Default::default()))
                    .1
                    .insert(
                        commands.spawn_empty().id(),
                        ExtractedUiNode {
                            z_order: stack_index.0 as f32 + stack_z_offsets::TEXT,
                            clip,
                            image: AssetId::default(),
                            transform: node_transform
                                * Affine2::from_translation(run.underline_position()),
                            item: ExtractedUiItem::Node {
                                color: shadow.color.into(),
                                rect: Rect {
                                    min: Vec2::ZERO,
                                    max: run.underline_size(),
                                },
                                atlas_scaling: None,
                                flip_x: false,
                                flip_y: false,
                                border: BorderRect::ZERO,
                                border_radius: ResolvedBorderRadius::ZERO,
                                node_type: NodeType::Rect,
                            },
                        },
                    );
            }
        }
    }
}

pub fn extract_text_decorations(
    mut commands: Commands,
    extracted_uinodes: ResMut<ExtractedUiNodes>,
    uinode_query: Extract<
        Query<(
            Entity,
            &ComputedNode,
            &ComputedStackIndex,
            &ComputedTextBlock,
            &UiGlobalTransform,
            &InheritedVisibility,
            Option<&CalculatedClip>,
            &ComputedUiTargetCamera,
            &TextLayoutInfo,
            Option<&EditableText>,
        )>,
    >,
    text_background_colors_query: Extract<
        Query<(
            AnyOf<(&TextBackgroundColor, &Strikethrough, &Underline)>,
            &TextColor,
            Option<&StrikethroughColor>,
            Option<&UnderlineColor>,
        )>,
    >,
    camera_map: Extract<UiCameraMap>,
) {
    let extracted_uinodes = extracted_uinodes.into_inner();
    let mut camera_mapper = camera_map.get_mapper();

    for (
        entity,
        uinode,
        stack_index,
        computed_block,
        global_transform,
        inherited_visibility,
        maybe_clip,
        camera,
        text_layout_info,
        editable_text,
    ) in extracted_uinodes
        .changed
        .iter()
        .flat_map(|main_entity| uinode_query.get(main_entity.entity()).ok())
    {
        // Skip if not visible or if size is set to zero (e.g. when a parent is set to `Display::None`)
        if !inherited_visibility.get() || uinode.is_empty() {
            continue;
        }

        let Some(extracted_camera_entity) = camera_mapper.map(camera) else {
            continue;
        };

        let transform = Affine2::from(global_transform)
            * Affine2::from_translation(
                uinode.content_box().min
                    - editable_text.map_or(Vec2::ZERO, |text| text.viewport.offset),
            );

        let clip = if editable_text.is_some() {
            let content_box = uinode.content_box();
            let text_clip = Rect::from_center_size(
                global_transform.affine().translation + content_box.center(),
                content_box.size(),
            );
            Some(maybe_clip.map_or(text_clip, |clip| clip.clip.intersect(text_clip)))
        } else {
            maybe_clip.map(|clip| clip.clip)
        };

        for run in text_layout_info.run_geometry.iter() {
            let Some(section_entity) = computed_block
                .entities()
                .get(run.section_index)
                .map(|t| t.entity)
            else {
                continue;
            };
            let Ok((
                (text_background_color, maybe_strikethrough, maybe_underline),
                text_color,
                maybe_strikethrough_color,
                maybe_underline_color,
            )) = text_background_colors_query.get(section_entity)
            else {
                continue;
            };

            if let Some(text_background_color) = text_background_color {
                extracted_uinodes
                    .uinodes
                    .entry(entity.into())
                    .or_insert_with(|| (extracted_camera_entity, Default::default()))
                    .1
                    .insert(
                        commands.spawn_empty().id(),
                        ExtractedUiNode {
                            z_order: stack_index.0 as f32 + stack_z_offsets::TEXT,
                            clip,
                            image: AssetId::default(),
                            transform: transform * Affine2::from_translation(run.bounds.center()),
                            item: ExtractedUiItem::Node {
                                color: text_background_color.0.to_linear(),
                                rect: Rect {
                                    min: Vec2::ZERO,
                                    max: run.bounds.size(),
                                },
                                atlas_scaling: None,
                                flip_x: false,
                                flip_y: false,
                                border: BorderRect::ZERO,
                                border_radius: ResolvedBorderRadius::ZERO,
                                node_type: NodeType::Rect,
                            },
                        },
                    );
            }

            if maybe_strikethrough.is_some() {
                let color = maybe_strikethrough_color
                    .map(|sc| sc.0)
                    .unwrap_or(text_color.0)
                    .to_linear();

                extracted_uinodes
                    .uinodes
                    .entry(entity.into())
                    .or_insert_with(|| (extracted_camera_entity, Default::default()))
                    .1
                    .insert(
                        commands.spawn_empty().id(),
                        ExtractedUiNode {
                            z_order: stack_index.0 as f32 + stack_z_offsets::TEXT_STRIKETHROUGH,
                            clip,
                            image: AssetId::default(),
                            transform: transform
                                * Affine2::from_translation(run.strikethrough_position()),
                            item: ExtractedUiItem::Node {
                                color,
                                rect: Rect {
                                    min: Vec2::ZERO,
                                    max: run.strikethrough_size(),
                                },
                                atlas_scaling: None,
                                flip_x: false,
                                flip_y: false,
                                border: BorderRect::ZERO,
                                border_radius: ResolvedBorderRadius::ZERO,
                                node_type: NodeType::Rect,
                            },
                        },
                    );
            }

            if maybe_underline.is_some() {
                let color = maybe_underline_color
                    .map(|uc| uc.0)
                    .unwrap_or(text_color.0)
                    .to_linear();

                extracted_uinodes
                    .uinodes
                    .entry(entity.into())
                    .or_insert_with(|| (extracted_camera_entity, Default::default()))
                    .1
                    .insert(
                        commands.spawn_empty().id(),
                        ExtractedUiNode {
                            z_order: stack_index.0 as f32 + stack_z_offsets::TEXT_STRIKETHROUGH,
                            clip,
                            image: AssetId::default(),
                            transform: transform
                                * Affine2::from_translation(run.underline_position()),
                            item: ExtractedUiItem::Node {
                                color,
                                rect: Rect {
                                    min: Vec2::ZERO,
                                    max: run.underline_size(),
                                },
                                atlas_scaling: None,
                                flip_x: false,
                                flip_y: false,
                                border: BorderRect::ZERO,
                                border_radius: ResolvedBorderRadius::ZERO,
                                node_type: NodeType::Rect,
                            },
                        },
                    );
            }
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub(crate) struct UiGeometryInstance {
    /// The columns of the 2D affine transform.
    pub transform_x: [f32; 2],
    pub transform_y: [f32; 2],
    pub translation: [f32; 2],
    /// Size of the UI node before clipping.
    pub size: [f32; 2],
    /// World-space clipping offsets, packed two corners per attribute.
    pub position_diff_01: [f32; 4],
    pub position_diff_23: [f32; 4],
    /// Texture coordinates, packed two corners per attribute.
    pub uv_01: [f32; 4],
    pub uv_23: [f32; 4],
}

impl_atomic_pod!(UiGeometryInstance, UiGeometryInstanceBlob);

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct UiStyleInstance {
    /// Positions relative to the center, packed two corners per attribute.
    pub point_01: [f32; 4],
    pub point_23: [f32; 4],
    pub color: [f32; 4],
    /// Shader flags to determine how to render the UI node.
    /// See [`shader_flags`] for possible values.
    pub flags: u32,
    pub flags_padding: [u32; 3],
    /// Border radius of the UI node.
    /// Ordering: top left, top right, bottom right, bottom left.
    pub radius: [[f32; 4]; 2],
    /// Border thickness of the UI node.
    /// Ordering: left, top, right, bottom.
    pub border: [f32; 4],
}

impl_atomic_pod!(UiStyleInstance, UiStyleInstanceBlob);

#[derive(Copy, Clone, Debug, Default)]
struct UiInstance {
    geometry: UiGeometryInstance,
    style: UiStyleInstance,
}

#[derive(Resource)]
pub struct UiMeta {
    geometry_instances: AtomicSparseBufferVec<UiGeometryInstance>,
    style_instances: AtomicSparseBufferVec<UiStyleInstance>,
    instance_indices: RawBufferVec<u32>,
    instance_bind_group: Option<BindGroup>,
    instance_buffer_ids: Option<(BufferId, BufferId)>,
    use_storage_buffers: bool,
    view_bind_group: Option<BindGroup>,
    batches: Vec<UiBatch>,
    arena: UiInstanceArena,
    instance_scratch: Vec<UiInstance>,
    camera_states: EntityHashMap<UiCameraPipelineState>,
}

impl Default for UiMeta {
    fn default() -> Self {
        Self {
            geometry_instances: AtomicSparseBufferVec::new(
                BufferUsages::VERTEX | BufferUsages::STORAGE,
                0,
                Arc::from("ui geometry instance buffer"),
            ),
            style_instances: AtomicSparseBufferVec::new(
                BufferUsages::VERTEX | BufferUsages::STORAGE,
                0,
                Arc::from("ui style instance buffer"),
            ),
            instance_indices: RawBufferVec::new(BufferUsages::VERTEX),
            instance_bind_group: None,
            instance_buffer_ids: None,
            use_storage_buffers: false,
            view_bind_group: None,
            batches: Vec::new(),
            arena: UiInstanceArena::default(),
            instance_scratch: Vec::new(),
            camera_states: EntityHashMap::default(),
        }
    }
}

pub(crate) const QUAD_VERTEX_POSITIONS: [Vec2; 4] = [
    Vec2::new(-0.5, -0.5),
    Vec2::new(0.5, -0.5),
    Vec2::new(0.5, 0.5),
    Vec2::new(-0.5, 0.5),
];

pub(crate) const QUAD_INDICES: [usize; 6] = [0, 2, 3, 0, 1, 2];

#[derive(Debug)]
pub struct UiBatch {
    pub range: Range<u32>,
    pub image: AssetId<Image>,
}

#[derive(Clone, Copy, Debug)]
struct UiBatchKey {
    pipeline: CachedRenderPipelineId,
    image: AssetId<Image>,
}

fn compatible_ui_batch_keys(left: &UiBatchKey, right: &UiBatchKey) -> bool {
    left.pipeline == right.pipeline
        && (left.image == AssetId::default()
            || right.image == AssetId::default()
            || left.image == right.image)
}

fn merge_ui_batch_key(left: &mut UiBatchKey, right: &UiBatchKey) {
    if left.image == AssetId::default() {
        left.image = right.image;
    }
}

/// The values here should match the values for the constants in `ui.wgsl`
pub mod shader_flags {
    /// Texture should be ignored
    pub const UNTEXTURED: u32 = 0;
    /// Textured
    pub const TEXTURED: u32 = 1;
    /// Ordering: top left, top right, bottom right, bottom left.
    pub const CORNERS: [u32; 4] = [0, 2, 2 | 4, 4];
    pub const RADIAL: u32 = 16;
    pub const FILL_START: u32 = 32;
    pub const FILL_END: u32 = 64;
    pub const CONIC: u32 = 128;
    pub const BORDER_LEFT: u32 = 256;
    pub const BORDER_TOP: u32 = 512;
    pub const BORDER_RIGHT: u32 = 1024;
    pub const BORDER_BOTTOM: u32 = 2048;
    pub const BORDER_ALL: u32 = BORDER_LEFT + BORDER_TOP + BORDER_RIGHT + BORDER_BOTTOM;
    pub const INVERT: u32 = 4096;
}

fn collect_queue_main_entities(
    extracted_uinodes: &ExtractedUiNodes,
    invalidated_cameras: &HashSet<Entity>,
) -> MainEntityHashSet {
    let mut dirty_main_entities: MainEntityHashSet =
        extracted_uinodes.changed.iter().copied().collect();
    if !invalidated_cameras.is_empty() {
        dirty_main_entities.extend(extracted_uinodes.uinodes.iter().filter_map(
            |(main_entity, (camera_entity, _))| {
                invalidated_cameras
                    .contains(camera_entity)
                    .then_some(*main_entity)
            },
        ));
    }
    dirty_main_entities
}

pub fn queue_uinodes(
    extracted_uinodes: Res<ExtractedUiNodes>,
    ui_pipeline: Res<UiPipeline>,
    mut ui_meta: ResMut<UiMeta>,
    mut pipelines: ResMut<SpecializedRenderPipelines<UiPipeline>>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<TransparentUi>>,
    render_views: Query<(Entity, &UiCameraView, Option<&UiAntiAlias>), With<ExtractedView>>,
    camera_views: Query<&ExtractedView>,
    pipeline_cache: Res<PipelineCache>,
    draw_functions: Res<DrawFunctions<TransparentUi>>,
    render_device: Res<RenderDevice>,
) {
    let draw_function = draw_functions.read().id::<DrawUi>();
    let camera_states = &mut ui_meta.camera_states;
    let mut active_cameras = HashSet::new();
    let mut invalidated_cameras = HashSet::new();
    let storage_buffers = retained::vertex_storage_supported(&render_device, 2);

    for (camera_entity, ui_camera_view, ui_anti_alias) in &render_views {
        let Ok(view) = camera_views.get(ui_camera_view.0) else {
            continue;
        };
        let state = UiCameraPipelineState {
            retained_view_entity: view.retained_view_entity,
            pipeline: pipelines.specialize(
                &pipeline_cache,
                &ui_pipeline,
                UiPipelineKey {
                    target_format: view.target_format,
                    anti_alias: matches!(ui_anti_alias, None | Some(UiAntiAlias::On)),
                    storage_buffers,
                },
            ),
        };
        active_cameras.insert(camera_entity);
        if camera_states.insert(camera_entity, state) != Some(state) {
            invalidated_cameras.insert(camera_entity);
        }
    }

    for removed in &extracted_uinodes.removed {
        let Some(camera_state) = camera_states.get(&removed.camera_entity) else {
            continue;
        };
        if let Some(phase) = transparent_render_phases.get_mut(&camera_state.retained_view_entity) {
            phase.remove(removed.render_entity, removed.main_entity);
        }
    }

    for main_entity in collect_queue_main_entities(&extracted_uinodes, &invalidated_cameras) {
        let Some((camera_entity, extracted_sub_uinodes)) =
            extracted_uinodes.uinodes.get(&main_entity)
        else {
            continue;
        };
        let Some(camera_state) = camera_states.get(camera_entity) else {
            continue;
        };
        let Some(transparent_phase) =
            transparent_render_phases.get_mut(&camera_state.retained_view_entity)
        else {
            continue;
        };
        for (render_entity, extracted_uinode) in extracted_sub_uinodes {
            transparent_phase.add_retained(TransparentUi {
                draw_function,
                pipeline: camera_state.pipeline,
                entity: (*render_entity, main_entity),
                sort_key: FloatOrd(extracted_uinode.z_order),
                // batch_range will be calculated in prepare_uinodes
                batch_range: 0..0,
                extra_index: PhaseItemExtraIndex::None,
                indexed: false,
                batch_index: None,
            });
        }
    }

    camera_states.retain(|camera_entity, _| active_cameras.contains(camera_entity));
}

#[derive(Resource, Default)]
pub struct ImageNodeBindGroups {
    pub values: HashMap<AssetId<Image>, BindGroup>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InstanceGeneration {
    Ready,
    Culled,
    PendingImage,
}

pub(crate) fn pack_corners(values: [Vec2; 4]) -> ([f32; 4], [f32; 4]) {
    (
        [values[0].x, values[0].y, values[1].x, values[1].y],
        [values[2].x, values[2].y, values[3].x, values[3].y],
    )
}

fn make_ui_instance(
    transform: Affine2,
    size: Vec2,
    position_diff: [Vec2; 4],
    uvs: [Vec2; 4],
    points: [Vec2; 4],
    color: [f32; 4],
    flags: u32,
    radius: [[f32; 4]; 2],
    border: [f32; 4],
) -> UiInstance {
    let (position_diff_01, position_diff_23) = pack_corners(position_diff);
    let (uv_01, uv_23) = pack_corners(uvs);
    let (point_01, point_23) = pack_corners(points);
    UiInstance {
        geometry: UiGeometryInstance {
            transform_x: transform.matrix2.x_axis.into(),
            transform_y: transform.matrix2.y_axis.into(),
            translation: transform.translation.into(),
            size: size.into(),
            position_diff_01,
            position_diff_23,
            uv_01,
            uv_23,
        },
        style: UiStyleInstance {
            point_01,
            point_23,
            color,
            flags,
            flags_padding: [0; 3],
            radius,
            border,
        },
    }
}

pub(crate) fn clipping_offsets(
    transform: Affine2,
    translation: Vec2,
    size: Vec2,
    clip: Option<Rect>,
) -> ([Vec2; 4], bool) {
    let positions =
        QUAD_VERTEX_POSITIONS.map(|corner| transform.transform_point2(translation + corner * size));
    let position_diff = if let Some(clip) = clip {
        [
            Vec2::new(
                f32::max(clip.min.x - positions[0].x, 0.0),
                f32::max(clip.min.y - positions[0].y, 0.0),
            ),
            Vec2::new(
                f32::min(clip.max.x - positions[1].x, 0.0),
                f32::max(clip.min.y - positions[1].y, 0.0),
            ),
            Vec2::new(
                f32::min(clip.max.x - positions[2].x, 0.0),
                f32::min(clip.max.y - positions[2].y, 0.0),
            ),
            Vec2::new(
                f32::max(clip.min.x - positions[3].x, 0.0),
                f32::min(clip.max.y - positions[3].y, 0.0),
            ),
        ]
    } else {
        [Vec2::ZERO; 4]
    };

    // Exact clipping of a rotated quad can require more than four vertices, so
    // preserve the existing conservative behavior and don't cull that case.
    let transformed_size = transform.transform_vector2(size).abs();
    let culled = transform.x_axis[1] == 0.0
        && (position_diff[0].x - position_diff[1].x >= transformed_size.x
            || position_diff[1].y - position_diff[2].y >= transformed_size.y);
    (position_diff, culled)
}

fn generate_item_instances(
    extracted_uinode: &ExtractedUiNode,
    image_extent: Option<Vec2>,
    scratch: &mut Vec<UiInstance>,
) -> InstanceGeneration {
    scratch.clear();
    match &extracted_uinode.item {
        ExtractedUiItem::Node {
            atlas_scaling,
            flip_x,
            flip_y,
            border_radius,
            border,
            node_type,
            rect,
            color,
        } => {
            let textured = extracted_uinode.image != AssetId::default();
            let mut flags = if textured {
                shader_flags::TEXTURED
            } else {
                shader_flags::UNTEXTURED
            };
            let mut uinode_rect = *rect;
            let size = uinode_rect.size();
            let (position_diff, culled) = clipping_offsets(
                extracted_uinode.transform,
                Vec2::ZERO,
                size,
                extracted_uinode.clip,
            );
            if culled {
                return InstanceGeneration::Culled;
            }
            let points = core::array::from_fn(|index| {
                QUAD_VERTEX_POSITIONS[index] * size + position_diff[index]
            });

            let uvs = if !textured {
                [Vec2::ZERO, Vec2::X, Vec2::ONE, Vec2::Y]
            } else {
                let Some(image_extent) = image_extent else {
                    return InstanceGeneration::PendingImage;
                };
                let atlas_extent = atlas_scaling
                    .map(|scaling| image_extent * scaling)
                    .unwrap_or(uinode_rect.max);
                let mut uv_position_diff = position_diff;
                if *flip_x {
                    mem::swap(&mut uinode_rect.max.x, &mut uinode_rect.min.x);
                    for offset in &mut uv_position_diff {
                        offset.x *= -1.0;
                    }
                }
                if *flip_y {
                    mem::swap(&mut uinode_rect.max.y, &mut uinode_rect.min.y);
                    for offset in &mut uv_position_diff {
                        offset.y *= -1.0;
                    }
                }
                [
                    uinode_rect.min + uv_position_diff[0],
                    Vec2::new(uinode_rect.max.x, uinode_rect.min.y) + uv_position_diff[1],
                    uinode_rect.max + uv_position_diff[2],
                    Vec2::new(uinode_rect.min.x, uinode_rect.max.y) + uv_position_diff[3],
                ]
                .map(|position| position / atlas_extent)
            };

            match *node_type {
                NodeType::Border(border_flags) => flags |= border_flags,
                NodeType::Inverted => flags |= INVERT,
                NodeType::Rect => {}
            }
            scratch.push(make_ui_instance(
                extracted_uinode.transform,
                size,
                position_diff,
                uvs,
                points,
                color.to_f32_array(),
                flags,
                (*border_radius).into(),
                [
                    border.min_inset.x,
                    border.min_inset.y,
                    border.max_inset.x,
                    border.max_inset.y,
                ],
            ));
        }
        ExtractedUiItem::Glyphs { glyphs } => {
            let Some(image_extent) = image_extent else {
                return InstanceGeneration::PendingImage;
            };
            for glyph in glyphs {
                let size = glyph.rect.size();
                let (position_diff, culled) = clipping_offsets(
                    extracted_uinode.transform,
                    glyph.translation,
                    size,
                    extracted_uinode.clip,
                );
                if culled {
                    continue;
                }
                let uvs = [
                    glyph.rect.min + position_diff[0],
                    Vec2::new(glyph.rect.max.x, glyph.rect.min.y) + position_diff[1],
                    glyph.rect.max + position_diff[2],
                    Vec2::new(glyph.rect.min.x, glyph.rect.max.y) + position_diff[3],
                ]
                .map(|position| position / image_extent);
                let mut transform = extracted_uinode.transform;
                transform.translation = transform.transform_point2(glyph.translation);
                scratch.push(make_ui_instance(
                    transform,
                    size,
                    position_diff,
                    uvs,
                    [Vec2::ZERO; 4],
                    glyph.color.to_f32_array(),
                    shader_flags::TEXTURED,
                    [[0.0; 4]; 2],
                    [0.0; 4],
                ));
            }
            if scratch.is_empty() {
                return InstanceGeneration::Culled;
            }
        }
    }
    InstanceGeneration::Ready
}

fn place_item(
    arena: &mut UiInstanceArena,
    geometry_instances: &mut AtomicSparseBufferVec<UiGeometryInstance>,
    style_instances: &mut AtomicSparseBufferVec<UiStyleInstance>,
    scratch: &mut Vec<UiInstance>,
    render_entity: Entity,
    extracted: &ExtractedUiNode,
    gpu_images: &RenderAssets<GpuImage>,
) -> InstanceGeneration {
    let image_extent = gpu_images
        .get(extracted.image)
        .map(|image| image.size_2d().as_vec2());
    let generation = generate_item_instances(extracted, image_extent, scratch);
    let slot = match generation {
        InstanceGeneration::Ready => {
            let count = u32::try_from(scratch.len()).expect("too many UI instances in one item");
            let (start, capacity) = arena.alloc(count);
            geometry_instances.grow(start + capacity);
            style_instances.grow(start + capacity);
            for (offset, instance) in scratch.iter().copied().enumerate() {
                let index = start + offset as u32;
                geometry_instances.set(index, instance.geometry);
                style_instances.set(index, instance.style);
            }
            ArenaSlot {
                instances: ItemInstances { start, count },
                capacity,
            }
        }
        InstanceGeneration::Culled | InstanceGeneration::PendingImage => ArenaSlot {
            instances: ItemInstances::default(),
            capacity: 0,
        },
    };
    arena.slots.insert(render_entity, slot);
    generation
}

fn rebuild_owner_instances(
    main_entity: MainEntity,
    arena: &mut UiInstanceArena,
    geometry_instances: &mut AtomicSparseBufferVec<UiGeometryInstance>,
    style_instances: &mut AtomicSparseBufferVec<UiStyleInstance>,
    scratch: &mut Vec<UiInstance>,
    extracted_uinodes: &ExtractedUiNodes,
    gpu_images: &RenderAssets<GpuImage>,
) {
    arena.free_owner(main_entity);
    let Some((_, sub_uinodes)) = extracted_uinodes.uinodes.get(&main_entity) else {
        return;
    };

    let mut owned = Vec::with_capacity(sub_uinodes.len());
    let mut pending_image = false;
    for (render_entity, extracted) in sub_uinodes {
        pending_image |= matches!(
            place_item(
                arena,
                geometry_instances,
                style_instances,
                scratch,
                *render_entity,
                extracted,
                gpu_images,
            ),
            InstanceGeneration::PendingImage
        );
        owned.push(*render_entity);
    }
    if pending_image {
        arena.pending_assets.insert(main_entity);
    }
    if !owned.is_empty() {
        arena.owners.insert(main_entity, owned);
    }
}

fn collect_dirty_main_entities(
    extracted_uinodes: &ExtractedUiNodes,
    arena: &UiInstanceArena,
    changed_images: &HashSet<AssetId<Image>>,
) -> MainEntityHashSet {
    let mut dirty_main_entities: MainEntityHashSet =
        extracted_uinodes.changed.iter().copied().collect();
    dirty_main_entities.extend(arena.pending_assets.iter().copied());
    if !changed_images.is_empty() {
        dirty_main_entities.extend(extracted_uinodes.uinodes.iter().filter_map(
            |(main_entity, (_, sub_uinodes))| {
                sub_uinodes
                    .values()
                    .any(|node| changed_images.contains(&node.image))
                    .then_some(*main_entity)
            },
        ));
    }
    dirty_main_entities
}

pub fn prepare_uinodes(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    mut ui_meta: ResMut<UiMeta>,
    extracted_uinodes: Res<ExtractedUiNodes>,
    view_uniforms: Res<ViewUniforms>,
    ui_pipeline: Res<UiPipeline>,
    mut image_bind_groups: ResMut<ImageNodeBindGroups>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    mut phases: ResMut<ViewSortedRenderPhases<TransparentUi>>,
    events: Res<SpriteAssetEvents>,
    mut previous_len: Local<usize>,
    (
        mut sparse_buffer_update_jobs,
        mut sparse_buffer_update_bind_groups,
        sparse_buffer_update_pipelines,
    ): (
        ResMut<SparseBufferUpdateJobs>,
        ResMut<SparseBufferUpdateBindGroups>,
        Res<SparseBufferUpdatePipelines>,
    ),
) {
    ui_meta.use_storage_buffers = retained::vertex_storage_supported(&render_device, 2);
    let mut changed_images = HashSet::new();
    for event in &events.images {
        match event {
            AssetEvent::Unused { .. } => {}
            AssetEvent::Added { id }
            | AssetEvent::LoadedWithDependencies { id }
            | AssetEvent::Modified { id }
            | AssetEvent::Removed { id } => {
                image_bind_groups.values.remove(id);
                changed_images.insert(*id);
            }
        }
    }

    if ui_meta.arena.needs_compaction() || !ui_meta.arena.initialized {
        let UiMeta {
            geometry_instances,
            style_instances,
            arena,
            instance_scratch,
            ..
        } = &mut *ui_meta;
        arena.reset();
        geometry_instances.clear();
        style_instances.clear();
        let main_entities: Vec<_> = extracted_uinodes.uinodes.keys().copied().collect();
        for main_entity in main_entities {
            rebuild_owner_instances(
                main_entity,
                arena,
                geometry_instances,
                style_instances,
                instance_scratch,
                &extracted_uinodes,
                &gpu_images,
            );
        }
    } else {
        let dirty_main_entities =
            collect_dirty_main_entities(&extracted_uinodes, &ui_meta.arena, &changed_images);
        let UiMeta {
            geometry_instances,
            style_instances,
            arena,
            instance_scratch,
            ..
        } = &mut *ui_meta;
        for main_entity in dirty_main_entities {
            rebuild_owner_instances(
                main_entity,
                arena,
                geometry_instances,
                style_instances,
                instance_scratch,
                &extracted_uinodes,
                &gpu_images,
            );
        }
    }

    ui_meta
        .geometry_instances
        .write_buffers(&render_device, &render_queue);
    ui_meta
        .style_instances
        .write_buffers(&render_device, &render_queue);
    ui_meta.geometry_instances.prepare_to_populate_buffers(
        &render_device,
        &pipeline_cache,
        &mut sparse_buffer_update_jobs,
        &mut sparse_buffer_update_bind_groups,
        &sparse_buffer_update_pipelines,
    );
    ui_meta.style_instances.prepare_to_populate_buffers(
        &render_device,
        &pipeline_cache,
        &mut sparse_buffer_update_jobs,
        &mut sparse_buffer_update_bind_groups,
        &sparse_buffer_update_pipelines,
    );

    if ui_meta.use_storage_buffers {
        let Some(geometry_buffer) = ui_meta.geometry_instances.buffer() else {
            ui_meta.instance_bind_group = None;
            ui_meta.instance_buffer_ids = None;
            return;
        };
        let Some(style_buffer) = ui_meta.style_instances.buffer() else {
            ui_meta.instance_bind_group = None;
            ui_meta.instance_buffer_ids = None;
            return;
        };
        let buffer_ids = (geometry_buffer.id(), style_buffer.id());
        if ui_meta.instance_buffer_ids != Some(buffer_ids) {
            ui_meta.instance_bind_group = Some(render_device.create_bind_group(
                "ui_instance_bind_group",
                &pipeline_cache.get_bind_group_layout(&ui_pipeline.instance_layout),
                &BindGroupEntries::sequential((
                    geometry_buffer.as_entire_binding(),
                    style_buffer.as_entire_binding(),
                )),
            ));
            ui_meta.instance_buffer_ids = Some(buffer_ids);
        }
    } else {
        ui_meta.instance_bind_group = None;
        ui_meta.instance_buffer_ids = None;
    }

    let Some(view_binding) = view_uniforms.uniforms.binding() else {
        ui_meta.batches.clear();
        return;
    };
    ui_meta.view_bind_group = Some(render_device.create_bind_group(
        "ui_view_bind_group",
        &pipeline_cache.get_bind_group_layout(&ui_pipeline.view_layout),
        &BindGroupEntries::single(view_binding),
    ));

    let UiMeta {
        arena,
        instance_indices,
        use_storage_buffers,
        ..
    } = &mut *ui_meta;
    let retained_batches = batch_retained_ui(
        &mut phases,
        instance_indices,
        *use_storage_buffers,
        |item| {
            let Some(extracted_uinode) = extracted_uinodes
                .uinodes
                .get(&item.main_entity())
                .and_then(|(_, sub_uinodes)| sub_uinodes.get(&item.entity()))
            else {
                return RetainedBatchItem::NotOwned;
            };
            let Some(slot) = arena.slots.get(&item.entity()).copied() else {
                return RetainedBatchItem::Culled;
            };
            if slot.instances.count == 0 || gpu_images.get(extracted_uinode.image).is_none() {
                return RetainedBatchItem::Culled;
            }
            RetainedBatchItem::Drawable {
                instances: slot.instances,
                key: UiBatchKey {
                    pipeline: item.pipeline,
                    image: extracted_uinode.image,
                },
            }
        },
        compatible_ui_batch_keys,
        merge_ui_batch_key,
    );

    ui_meta
        .instance_indices
        .write_buffer(&render_device, &render_queue);
    let batches: Vec<_> = retained_batches
        .into_iter()
        .map(|batch| {
            let image = gpu_images
                .get(batch.key.image)
                .expect("retained UI batch image was validated while batching");
            image_bind_groups
                .values
                .entry(batch.key.image)
                .or_insert_with(|| {
                    render_device.create_bind_group(
                        "ui_material_bind_group",
                        &pipeline_cache.get_bind_group_layout(&ui_pipeline.image_layout),
                        &BindGroupEntries::sequential((&image.texture_view, &image.sampler)),
                    )
                });
            UiBatch {
                range: batch.range,
                image: batch.key.image,
            }
        })
        .collect();
    *previous_len = batches.len();
    ui_meta.batches = batches;
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_asset::uuid::Uuid;
    use bevy_render::render_phase::DrawFunctionId;
    use core::mem::offset_of;

    const EPSILON: f32 = 0.000_01;

    fn entity(index: u32) -> Entity {
        Entity::from_raw_u32(index).unwrap()
    }

    fn image_id(value: u128) -> AssetId<Image> {
        AssetId::Uuid {
            uuid: Uuid::from_u128(value),
        }
    }

    fn assert_vec2_eq(actual: Vec2, expected: Vec2) {
        assert!(
            (actual - expected).abs().max_element() <= EPSILON,
            "expected {expected:?}, got {actual:?}"
        );
    }

    fn unpack(first: [f32; 4], second: [f32; 4], corner: usize) -> Vec2 {
        match corner {
            0 => Vec2::new(first[0], first[1]),
            1 => Vec2::new(first[2], first[3]),
            2 => Vec2::new(second[0], second[1]),
            3 => Vec2::new(second[2], second[3]),
            _ => panic!("invalid corner {corner}"),
        }
    }

    fn instance_world_corner(instance: &UiInstance, corner: usize) -> Vec2 {
        let geometry = &instance.geometry;
        let local = QUAD_VERTEX_POSITIONS[corner] * Vec2::from(geometry.size);
        Vec2::from(geometry.transform_x) * local.x
            + Vec2::from(geometry.transform_y) * local.y
            + Vec2::from(geometry.translation)
            + unpack(geometry.position_diff_01, geometry.position_diff_23, corner)
    }

    fn node(
        image: AssetId<Image>,
        transform: Affine2,
        clip: Option<Rect>,
        flip_x: bool,
        flip_y: bool,
    ) -> ExtractedUiNode {
        ExtractedUiNode {
            z_order: 7.0,
            image,
            clip,
            transform,
            item: ExtractedUiItem::Node {
                color: LinearRgba::new(0.1, 0.2, 0.3, 0.4),
                rect: Rect::new(0.0, 0.0, 100.0, 80.0),
                atlas_scaling: None,
                flip_x,
                flip_y,
                border_radius: ResolvedBorderRadius {
                    top_left: Vec2::new(1.0, 2.0),
                    top_right: Vec2::new(3.0, 4.0),
                    bottom_right: Vec2::new(5.0, 6.0),
                    bottom_left: Vec2::new(7.0, 8.0),
                },
                border: BorderRect {
                    min_inset: Vec2::new(9.0, 10.0),
                    max_inset: Vec2::new(11.0, 12.0),
                },
                node_type: NodeType::Border(shader_flags::BORDER_ALL),
            },
        }
    }

    #[test]
    fn instance_streams_fit_atomic_blobs_and_vertex_attribute_limit() {
        assert_eq!(size_of::<UiGeometryInstance>(), 24 * size_of::<u32>());
        assert_eq!(size_of::<UiStyleInstance>(), 28 * size_of::<u32>());
        // The macro-backed atomic blob currently supports up to 32 words.
        assert!(size_of::<UiGeometryInstance>() <= 32 * size_of::<u32>());
        assert!(size_of::<UiStyleInstance>() <= 32 * size_of::<u32>());
        // Locations 0 through 14 are used; WebGPU guarantees at least 16.
        const UI_VERTEX_ATTRIBUTE_COUNT: usize = 15;
        const {
            assert!(UI_VERTEX_ATTRIBUTE_COUNT <= 16);
        }
    }

    #[test]
    fn instance_memory_layout_exactly_matches_pipeline_attributes() {
        let geometry_layout = ui_geometry_vertex_layout();
        let style_layout = ui_style_vertex_layout();
        assert_eq!(
            geometry_layout.array_stride as usize,
            size_of::<UiGeometryInstance>()
        );
        assert_eq!(
            style_layout.array_stride as usize,
            size_of::<UiStyleInstance>()
        );
        assert_eq!(geometry_layout.step_mode, VertexStepMode::Instance);
        assert_eq!(style_layout.step_mode, VertexStepMode::Instance);

        let geometry_offsets = [
            offset_of!(UiGeometryInstance, transform_x),
            offset_of!(UiGeometryInstance, transform_y),
            offset_of!(UiGeometryInstance, translation),
            offset_of!(UiGeometryInstance, size),
            offset_of!(UiGeometryInstance, position_diff_01),
            offset_of!(UiGeometryInstance, position_diff_23),
            offset_of!(UiGeometryInstance, uv_01),
            offset_of!(UiGeometryInstance, uv_23),
        ];
        let style_offsets = [
            offset_of!(UiStyleInstance, point_01),
            offset_of!(UiStyleInstance, point_23),
            offset_of!(UiStyleInstance, color),
            offset_of!(UiStyleInstance, flags),
            offset_of!(UiStyleInstance, radius),
            offset_of!(UiStyleInstance, radius) + size_of::<[f32; 4]>(),
            offset_of!(UiStyleInstance, border),
        ];

        assert_eq!(geometry_layout.attributes.len(), geometry_offsets.len());
        assert_eq!(style_layout.attributes.len(), style_offsets.len());
        for (location, (attribute, expected_offset)) in geometry_layout
            .attributes
            .iter()
            .zip(geometry_offsets)
            .enumerate()
        {
            assert_eq!(attribute.shader_location, location as u32);
            assert_eq!(attribute.offset as usize, expected_offset);
        }
        for (index, (attribute, expected_offset)) in style_layout
            .attributes
            .iter()
            .zip(style_offsets)
            .enumerate()
        {
            assert_eq!(attribute.shader_location, index as u32 + 8);
            assert_eq!(attribute.offset as usize, expected_offset);
        }
    }

    #[test]
    fn partially_clipped_node_reconstructs_the_original_six_vertices() {
        let extracted = node(
            AssetId::default(),
            Affine2::from_translation(Vec2::new(50.0, 40.0)),
            Some(Rect::new(10.0, 20.0, 90.0, 70.0)),
            false,
            false,
        );
        let mut instances = Vec::new();

        assert_eq!(
            generate_item_instances(&extracted, None, &mut instances),
            InstanceGeneration::Ready
        );
        assert_eq!(instances.len(), 1);
        let expected_corners = [
            Vec2::new(10.0, 20.0),
            Vec2::new(90.0, 20.0),
            Vec2::new(90.0, 70.0),
            Vec2::new(10.0, 70.0),
        ];
        for (vertex, corner) in QUAD_INDICES.into_iter().enumerate() {
            assert_vec2_eq(
                instance_world_corner(&instances[0], corner),
                expected_corners[corner],
            );
            let expected_flag = shader_flags::BORDER_ALL | shader_flags::CORNERS[corner];
            let shader_flag = instances[0].style.flags | shader_flags::CORNERS[corner];
            assert_eq!(shader_flag, expected_flag, "vertex {vertex}");
        }

        assert_eq!(
            instances[0].style.radius,
            [[1.0, 3.0, 5.0, 7.0], [2.0, 4.0, 6.0, 8.0]]
        );
        assert_eq!(instances[0].style.border, [9.0, 10.0, 11.0, 12.0]);
        assert_eq!(instances[0].style.point_01, [-40.0, -20.0, 40.0, -20.0]);
        assert_eq!(instances[0].style.point_23, [40.0, 30.0, -40.0, 30.0]);
    }

    #[test]
    fn fully_clipped_axis_aligned_node_produces_no_instance() {
        let extracted = node(
            AssetId::default(),
            Affine2::from_translation(Vec2::new(50.0, 40.0)),
            Some(Rect::new(100.0, 0.0, 200.0, 80.0)),
            false,
            false,
        );
        let mut instances = vec![UiInstance::default()];

        assert_eq!(
            generate_item_instances(&extracted, None, &mut instances),
            InstanceGeneration::Culled
        );
        assert!(instances.is_empty());
    }

    #[test]
    fn rotated_node_is_conservatively_not_culled() {
        let transform = Affine2::from_translation(Vec2::splat(50.0))
            * Affine2::from_angle(core::f32::consts::FRAC_PI_4);
        let extracted = node(
            AssetId::default(),
            transform,
            Some(Rect::new(500.0, 500.0, 600.0, 600.0)),
            false,
            false,
        );
        let mut instances = Vec::new();

        assert_eq!(
            generate_item_instances(&extracted, None, &mut instances),
            InstanceGeneration::Ready
        );
        assert_eq!(instances.len(), 1);
    }

    #[test]
    fn textured_node_waits_for_its_gpu_image_instead_of_becoming_culled() {
        let extracted = node(image_id(1), Affine2::IDENTITY, None, false, false);
        let mut instances = Vec::new();

        assert_eq!(
            generate_item_instances(&extracted, None, &mut instances),
            InstanceGeneration::PendingImage
        );
        assert!(instances.is_empty());
    }

    #[test]
    fn flipping_texture_coordinates_does_not_flip_clipped_geometry() {
        let extracted = node(
            image_id(2),
            Affine2::from_translation(Vec2::new(50.0, 40.0)),
            Some(Rect::new(10.0, 20.0, 90.0, 70.0)),
            true,
            true,
        );
        let mut instances = Vec::new();
        assert_eq!(
            generate_item_instances(&extracted, Some(Vec2::new(100.0, 80.0)), &mut instances),
            InstanceGeneration::Ready
        );
        let instance = &instances[0];
        let expected_uvs = [
            Vec2::new(0.9, 0.75),
            Vec2::new(0.1, 0.75),
            Vec2::new(0.1, 0.125),
            Vec2::new(0.9, 0.125),
        ];
        for (corner, expected_uv) in expected_uvs.into_iter().enumerate() {
            assert_vec2_eq(
                unpack(instance.geometry.uv_01, instance.geometry.uv_23, corner),
                expected_uv,
            );
        }
        assert_vec2_eq(instance_world_corner(instance, 0), Vec2::new(10.0, 20.0));
        assert_vec2_eq(instance_world_corner(instance, 2), Vec2::new(90.0, 70.0));
    }

    #[test]
    fn glyph_translation_uvs_and_legacy_zero_points_are_preserved() {
        let extracted = ExtractedUiNode {
            z_order: 0.0,
            image: image_id(3),
            clip: None,
            transform: Affine2::from_translation(Vec2::new(50.0, 50.0)),
            item: ExtractedUiItem::Glyphs {
                glyphs: vec![ExtractedGlyph {
                    color: LinearRgba::WHITE,
                    translation: Vec2::new(10.0, 5.0),
                    rect: Rect::new(20.0, 10.0, 40.0, 30.0),
                }],
            },
        };
        let mut instances = Vec::new();

        assert_eq!(
            generate_item_instances(&extracted, Some(Vec2::new(200.0, 100.0)), &mut instances),
            InstanceGeneration::Ready
        );
        assert_eq!(instances.len(), 1);
        let instance = &instances[0];
        assert_vec2_eq(
            Vec2::from(instance.geometry.translation),
            Vec2::new(60.0, 55.0),
        );
        assert_vec2_eq(instance_world_corner(instance, 0), Vec2::new(50.0, 45.0));
        assert_vec2_eq(instance_world_corner(instance, 2), Vec2::new(70.0, 65.0));
        assert_vec2_eq(
            unpack(instance.geometry.uv_01, instance.geometry.uv_23, 0),
            Vec2::new(0.1, 0.1),
        );
        assert_vec2_eq(
            unpack(instance.geometry.uv_01, instance.geometry.uv_23, 2),
            Vec2::new(0.2, 0.3),
        );
        assert_eq!(instance.style.point_01, [0.0; 4]);
        assert_eq!(instance.style.point_23, [0.0; 4]);
    }

    #[test]
    fn glyph_run_keeps_only_glyphs_that_survive_clipping() {
        let extracted = ExtractedUiNode {
            z_order: 0.0,
            image: image_id(4),
            clip: Some(Rect::new(0.0, 0.0, 25.0, 25.0)),
            transform: Affine2::IDENTITY,
            item: ExtractedUiItem::Glyphs {
                glyphs: vec![
                    ExtractedGlyph {
                        color: LinearRgba::WHITE,
                        translation: Vec2::new(10.0, 10.0),
                        rect: Rect::new(0.0, 0.0, 10.0, 10.0),
                    },
                    ExtractedGlyph {
                        color: LinearRgba::WHITE,
                        translation: Vec2::new(100.0, 100.0),
                        rect: Rect::new(10.0, 10.0, 20.0, 20.0),
                    },
                ],
            },
        };
        let mut instances = Vec::new();

        assert_eq!(
            generate_item_instances(&extracted, Some(Vec2::splat(128.0)), &mut instances),
            InstanceGeneration::Ready
        );
        assert_eq!(instances.len(), 1);
    }

    #[test]
    fn arena_reuses_only_exact_power_of_two_capacity_classes() {
        let mut arena = UiInstanceArena::default();
        let (large_start, large_capacity) = arena.alloc(3);
        let (small_start, small_capacity) = arena.alloc(1);
        assert_eq!((large_start, large_capacity), (0, 4));
        assert_eq!((small_start, small_capacity), (4, 1));

        let render_entity = entity(10);
        arena.slots.insert(
            render_entity,
            ArenaSlot {
                instances: ItemInstances {
                    start: large_start,
                    count: 3,
                },
                capacity: large_capacity,
            },
        );
        arena.free(render_entity);
        assert_eq!(arena.dead_instances, 4);

        // A two-element request must not split the four-element free block.
        assert_eq!(arena.alloc(2), (5, 2));
        assert_eq!(arena.dead_instances, 4);
        assert_eq!(arena.alloc(4), (0, 4));
        assert_eq!(arena.dead_instances, 0);
    }

    #[test]
    fn freeing_an_owner_releases_every_slot_and_pending_marker() {
        let mut arena = UiInstanceArena::default();
        let main_entity = MainEntity::from(entity(20));
        let first = entity(21);
        let second = entity(22);
        arena.owners.insert(main_entity, vec![first, second]);
        arena.pending_assets.insert(main_entity);
        arena.slots.insert(
            first,
            ArenaSlot {
                instances: ItemInstances { start: 0, count: 1 },
                capacity: 1,
            },
        );
        arena.slots.insert(
            second,
            ArenaSlot {
                instances: ItemInstances { start: 1, count: 2 },
                capacity: 2,
            },
        );

        arena.free_owner(main_entity);

        assert!(!arena.owners.contains_key(&main_entity));
        assert!(!arena.pending_assets.contains(&main_entity));
        assert!(arena.slots.is_empty());
        assert_eq!(arena.dead_instances, 3);
        assert_eq!(arena.free_lists.get(&1), Some(&vec![0]));
        assert_eq!(arena.free_lists.get(&2), Some(&vec![1]));
    }

    #[test]
    fn compaction_threshold_is_bounded_and_uses_matching_units() {
        let mut arena = UiInstanceArena {
            top: retained::UI_ARENA_COMPACT_MIN_INSTANCES,
            dead_instances: retained::UI_ARENA_COMPACT_MIN_INSTANCES / 8,
            ..Default::default()
        };
        assert!(!arena.needs_compaction());
        arena.dead_instances += 1;
        assert!(arena.needs_compaction());
        arena.top -= 1;
        assert!(!arena.needs_compaction());
    }

    #[test]
    fn unchanged_frame_has_no_dirty_instance_owners() {
        let extracted = ExtractedUiNodes::default();
        let arena = UiInstanceArena {
            initialized: true,
            ..Default::default()
        };
        assert!(
            collect_dirty_main_entities(&extracted, &arena, &HashSet::new()).is_empty(),
            "a true no-op frame must not rewrite any retained instance slots"
        );
    }

    #[test]
    fn unchanged_frame_queues_no_retained_phase_items() {
        let extracted = ExtractedUiNodes::default();
        assert!(
            collect_queue_main_entities(&extracted, &HashSet::new()).is_empty(),
            "a true no-op frame must not requeue retained UI phase items"
        );
    }

    #[test]
    fn camera_pipeline_change_requeues_only_nodes_for_that_camera() {
        let changed_camera = entity(40);
        let unchanged_camera = entity(41);
        let changed_main = MainEntity::from(entity(42));
        let unchanged_main = MainEntity::from(entity(43));
        let mut extracted = ExtractedUiNodes::default();
        extracted
            .uinodes
            .insert(changed_main, (changed_camera, EntityIndexMap::default()));
        extracted.uinodes.insert(
            unchanged_main,
            (unchanged_camera, EntityIndexMap::default()),
        );

        let queued = collect_queue_main_entities(&extracted, &HashSet::from([changed_camera]));

        assert_eq!(queued.len(), 1);
        assert!(queued.contains(&changed_main));
    }

    #[test]
    fn retained_phase_items_survive_frame_cleanup_until_explicitly_removed() {
        let main_entity = MainEntity::from(entity(50));
        let render_entity = entity(51);
        let transient_main = MainEntity::from(entity(52));
        let transient_render = entity(53);
        let retained_view = RetainedViewEntity::new(main_entity, None, 0);
        let mut phases = ViewSortedRenderPhases::<TransparentUi>::default();
        phases.prepare_for_new_frame(retained_view);
        let phase = phases.get_mut(&retained_view).unwrap();
        phase.add_retained(TransparentUi {
            sort_key: FloatOrd(0.0),
            entity: (render_entity, main_entity),
            pipeline: CachedRenderPipelineId::INVALID,
            draw_function: DrawFunctionId(0),
            batch_range: 0..1,
            extra_index: PhaseItemExtraIndex::None,
            indexed: false,
            batch_index: None,
        });
        phase.add_transient(TransparentUi {
            sort_key: FloatOrd(1.0),
            entity: (transient_render, transient_main),
            pipeline: CachedRenderPipelineId::INVALID,
            draw_function: DrawFunctionId(0),
            batch_range: 1..2,
            extra_index: PhaseItemExtraIndex::None,
            indexed: false,
            batch_index: None,
        });
        assert_eq!(phase.items.len(), 2);

        phases.prepare_for_new_frame(retained_view);
        let phase = phases.get_mut(&retained_view).unwrap();
        assert_eq!(phase.items.len(), 1);
        assert!(phase.items.contains_key(&(render_entity, main_entity)));
        phase.remove(render_entity, main_entity);
        assert!(phase.items.is_empty());
    }

    #[test]
    fn pending_and_asset_changed_owners_are_selected_for_retry() {
        let pending_main = MainEntity::from(entity(30));
        let asset_main = MainEntity::from(entity(31));
        let changed_image = image_id(30);
        let mut extracted = ExtractedUiNodes::default();
        extracted.uinodes.insert(
            asset_main,
            (
                entity(32),
                EntityIndexMap::from_iter([(
                    entity(33),
                    node(changed_image, Affine2::IDENTITY, None, false, false),
                )]),
            ),
        );
        let mut arena = UiInstanceArena::default();
        arena.pending_assets.insert(pending_main);

        let dirty =
            collect_dirty_main_entities(&extracted, &arena, &HashSet::from([changed_image]));

        assert_eq!(dirty.len(), 2);
        assert!(dirty.contains(&pending_main));
        assert!(dirty.contains(&asset_main));
    }

    #[test]
    fn untextured_core_instances_merge_without_crossing_pipeline_or_texture_boundaries() {
        let pipeline = CachedRenderPipelineId::INVALID;
        let other_pipeline = CachedRenderPipelineId::new(1);
        let first_image = image_id(40);
        let second_image = image_id(41);
        let untextured = UiBatchKey {
            pipeline,
            image: AssetId::default(),
        };
        let first = UiBatchKey {
            pipeline,
            image: first_image,
        };
        let second = UiBatchKey {
            pipeline,
            image: second_image,
        };
        assert!(compatible_ui_batch_keys(&untextured, &first));
        assert!(compatible_ui_batch_keys(&first, &untextured));
        assert!(!compatible_ui_batch_keys(&first, &second));
        assert!(!compatible_ui_batch_keys(
            &first,
            &UiBatchKey {
                pipeline: other_pipeline,
                image: first_image,
            }
        ));
        let mut merged = untextured;
        merge_ui_batch_key(&mut merged, &first);
        assert_eq!(merged.image, first_image);
    }

    #[test]
    fn ui_shader_parses_and_validates_with_both_anti_alias_variants() {
        fn preprocess(source: &str, anti_alias: bool, storage_instances: bool) -> String {
            let mut output = String::from("struct View { clip_from_world: mat4x4<f32>, }\n");
            let mut conditions = Vec::new();
            let mut include = true;
            for line in source.lines() {
                match line.trim() {
                    directive if directive.starts_with("#define_import_path") => {}
                    directive if directive.starts_with("#import") => {}
                    directive if directive.starts_with("#ifdef ") => {
                        let condition = match directive.trim_start_matches("#ifdef ") {
                            "ANTI_ALIAS" => anti_alias,
                            "UI_STORAGE_INSTANCE" => storage_instances,
                            other => panic!("unexpected shader definition {other}"),
                        };
                        conditions.push((include, condition));
                        include &= condition;
                    }
                    "#else" => {
                        let (parent, condition) =
                            conditions.last().copied().expect("#else without #ifdef");
                        include = parent && !condition;
                    }
                    "#endif" => {
                        let (parent, _) = conditions.pop().expect("#endif without #ifdef");
                        include = parent;
                    }
                    _ if include => {
                        output.push_str(line);
                        output.push('\n');
                    }
                    _ => {}
                }
            }
            output
        }

        let source = include_str!("ui.wgsl");
        assert!(source.contains("const QUAD_CORNER_INDICES = array(0u, 2u, 3u, 0u, 1u, 2u);"));
        assert!(source.contains("const QUAD_CORNER_FLAGS = array(0u, 6u, 4u, 0u, 2u, 6u);"));
        for anti_alias in [false, true] {
            for storage_instances in [false, true] {
                let source = preprocess(source, anti_alias, storage_instances);
                let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|error| {
                    panic!(
                        "UI shader failed to parse with ANTI_ALIAS={anti_alias}, \
                     UI_STORAGE_INSTANCE={storage_instances}:\n{}",
                        error.emit_to_string(&source)
                    )
                });
                naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::all(),
                )
                .validate(&module)
                .unwrap_or_else(|error| {
                    panic!(
                        "UI shader failed validation with ANTI_ALIAS={anti_alias}, \
                     UI_STORAGE_INSTANCE={storage_instances}: {error}"
                    )
                });
                if storage_instances {
                    let mut composer = naga_oil::compose::Composer::default();
                    composer
                        .add_composable_module(naga_oil::compose::ComposableModuleDescriptor {
                            source: &source,
                            file_path: "ui-storage-test.wgsl",
                            as_name: Some(format!("ui_storage_{anti_alias}")),
                            ..Default::default()
                        })
                        .unwrap_or_else(|error| {
                            panic!(
                                "UI storage shader failed composable-module validation with \
                                 ANTI_ALIAS={anti_alias}: {error}"
                            )
                        });
                }
            }
        }
    }
}
