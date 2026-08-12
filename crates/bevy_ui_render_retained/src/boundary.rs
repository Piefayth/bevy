//! Declared raster-cache boundaries and their hierarchy ownership.

use crate::{
    scene::{
        coverage, PaintFamily, PaintId, RetainedBoundaryItem, RetainedDraw, RetainedDrawItem,
        RetainedUiScene,
    },
    PaintCoverage, PhysicalRect,
};
use bevy::{
    app::{Inherited, Propagate},
    camera::Camera,
    ecs::{
        hierarchy::ChildOf,
        lifecycle::{Add, Remove},
        observer::On,
        query::{Changed, Or, With},
        reflect::ReflectComponent,
        system::{Commands, Query, Res, ResMut},
    },
    math::{Mat4, Rect, UVec2, UVec4, Vec2},
    platform::collections::{HashMap, HashSet},
    prelude::{Component, Entity, GlobalTransform, Reflect, Resource},
    reflect::std_traits::ReflectDefault,
    render::{
        camera::CameraMainPassTextureFormats,
        render_phase::ViewSortedRenderPhases,
        render_resource::TextureFormat,
        sync_world::{RenderEntity, SyncToRenderWorld},
        view::{ExtractedView, RetainedViewEntity},
        Extract,
    },
    ui::{
        CalculatedClip, ComputedNode, ComputedStackIndex, ComputedUiPaintTarget,
        ComputedUiRenderTargetInfo, ComputedUiTargetCamera, PaintContainment, UiGlobalTransform,
        UiTransform,
    },
    ui_render::{
        BoxShadowSamples, TransparentUi, UiAntiAlias, UiCameraMap, UiCameraView, UiViewTarget,
        VolatileUiPaintTargets, UI_CAMERA_FAR, UI_CAMERA_TRANSFORM_OFFSET,
    },
};
use bevy::{asset::AssetId, image::Image, math::Affine2, render::sync_world::MainEntity};

/// Caches this node and its descendants in an independent retained surface.
///
/// The boundary is a paint-containment promise and an atomic stacking context. The node's own
/// paint is also clipped to its border box, so effects such as an exterior shadow belong on a
/// parent wrapper. Its
/// [`transform`](Self::transform) and [`opacity`](Self::opacity) are applied when the cached
/// surface is painted into its parent; changing either property never rerasterizes the subtree.
/// Add [`bevy::ui::LayoutContainment`] separately when descendant layout must also be isolated.
#[derive(Component, Clone, Copy, Debug, PartialEq, Reflect)]
#[reflect(Component, Default, PartialEq, Debug, Clone)]
#[require(SyncToRenderWorld, PaintContainment)]
pub struct RepaintBoundary {
    /// Parent-surface placement applied around the boundary's center.
    pub transform: UiTransform,
    /// Group opacity applied after the subtree has been flattened.
    ///
    /// Values are clamped to `0.0..=1.0`; `NaN` is treated as fully opaque. A zero-opacity
    /// boundary suspends raster work until it becomes visible again.
    pub opacity: f32,
}

impl RepaintBoundary {
    /// An identity boundary with fully opaque group composition.
    pub const IDENTITY: Self = Self {
        transform: UiTransform::IDENTITY,
        opacity: 1.0,
    };

    /// Creates a boundary with a responsive parent-surface translation.
    pub const fn from_translation(translation: bevy::ui::Val2) -> Self {
        Self {
            transform: UiTransform::from_translation(translation),
            ..Self::IDENTITY
        }
    }

    /// Creates a boundary with a compositor-only scale.
    pub const fn from_scale(scale: Vec2) -> Self {
        Self {
            transform: UiTransform::from_scale(scale),
            ..Self::IDENTITY
        }
    }
}

impl Default for RepaintBoundary {
    fn default() -> Self {
        Self::IDENTITY
    }
}

pub(crate) fn retained_clip(
    entity: Entity,
    node: &ComputedNode,
    transform: &UiGlobalTransform,
    clip: Option<&CalculatedClip>,
    owner: Option<&Inherited<ComputedUiPaintTarget>>,
) -> Option<Rect> {
    let Some(owner) = owner else {
        return clip.map(|clip| clip.clip);
    };
    if owner.0 .0 == entity {
        Some(Rect::from_center_size(transform.translation, node.size()))
    } else {
        Some(
            clip.expect("paint-contained descendants must have a boundary-local clip")
                .paint_clip,
        )
    }
}

pub(crate) fn install_boundary_source(added: On<Add, RepaintBoundary>, mut commands: Commands) {
    commands
        .entity(added.entity)
        .try_insert(Propagate(ComputedUiPaintTarget(added.entity)));
}

pub(crate) fn remove_boundary_source(removed: On<Remove, RepaintBoundary>, mut commands: Commands) {
    commands
        .entity(removed.entity)
        .try_remove::<Propagate<ComputedUiPaintTarget>>();
}

#[derive(Clone)]
pub(crate) struct BoundaryView {
    pub(crate) main_entity: Entity,
    pub(crate) surface: Entity,
    pub(crate) parent_surface: Entity,
    pub(crate) source_camera: Entity,
    pub(crate) retained_view_entity: RetainedViewEntity,
    pub(crate) rect: PhysicalRect,
    pub(crate) format: TextureFormat,
    visible: bool,
    anti_alias: Option<UiAntiAlias>,
}

impl BoundaryView {
    pub(crate) fn size(&self) -> UVec2 {
        UVec2::new(
            (self.rect.max_x() - self.rect.min_x()) as u32,
            (self.rect.max_y() - self.rect.min_y()) as u32,
        )
    }
}

#[derive(Resource, Default)]
pub(crate) struct BoundaryViews {
    pub(crate) views: HashMap<Entity, BoundaryView>,
    active: Vec<Entity>,
    retired: Vec<RetainedViewEntity>,
}

impl BoundaryViews {
    fn effectively_visible(&self, surface: Entity) -> bool {
        let mut current = surface;
        while let Some(view) = self.views.get(&current) {
            if !view.visible {
                return false;
            }
            current = view.parent_surface;
        }
        true
    }

    fn visible_subtree(&self, root: Entity) -> Vec<(Entity, PhysicalRect)> {
        self.views
            .iter()
            .filter(|(surface, _)| {
                let mut current = **surface;
                loop {
                    if current == root {
                        return self.effectively_visible(**surface);
                    }
                    let Some(view) = self.views.get(&current) else {
                        return false;
                    };
                    current = view.parent_surface;
                }
            })
            .map(|(&surface, view)| (surface, view.rect))
            .collect()
    }

    fn retire(&mut self, view: BoundaryView) {
        self.retired.push(view.retained_view_entity);
    }

    fn remove_surface(&mut self, surface: Entity) {
        if let Some(view) = self.views.remove(&surface) {
            self.retire(view);
        }
    }

    fn remove_main_entity(&mut self, main_entity: Entity) {
        let surface = self
            .views
            .iter()
            .find_map(|(&surface, view)| (view.main_entity == main_entity).then_some(surface));
        if let Some(surface) = surface {
            self.remove_surface(surface);
        }
    }

    fn insert(&mut self, view: BoundaryView) {
        if let Some(previous) = self.views.insert(view.surface, view.clone())
            && (previous.retained_view_entity != view.retained_view_entity
                || (previous.visible && !view.visible))
        {
            self.retire(previous);
        }
    }

    pub(crate) fn take_retired(&mut self) -> Vec<RetainedViewEntity> {
        core::mem::take(&mut self.retired)
    }

    pub(crate) fn active(&self) -> impl Iterator<Item = &BoundaryView> {
        self.active
            .iter()
            .filter_map(|surface| self.views.get(surface))
    }
}

type BoundaryQueryItem<'a> = (
    Entity,
    &'a RenderEntity,
    &'a RepaintBoundary,
    &'a ComputedNode,
    &'a ComputedStackIndex,
    &'a UiGlobalTransform,
    Option<&'a CalculatedClip>,
    &'a ComputedUiTargetCamera,
    &'a ComputedUiRenderTargetInfo,
    Option<&'a ChildOf>,
);

pub(crate) fn extract_boundaries(
    mut commands: Commands,
    state: Res<RetainedUiScene>,
    mut views: ResMut<BoundaryViews>,
    changed: Extract<
        Query<
            BoundaryQueryItem<'static>,
            Or<(
                Changed<RepaintBoundary>,
                Changed<ComputedNode>,
                Changed<ComputedStackIndex>,
                Changed<UiGlobalTransform>,
                Changed<CalculatedClip>,
                Changed<ComputedUiTargetCamera>,
                Changed<ComputedUiRenderTargetInfo>,
                Changed<ChildOf>,
            )>,
        >,
    >,
    owners: Extract<Query<&'static Inherited<ComputedUiPaintTarget>>>,
    cameras: Extract<Query<Option<&'static UiAntiAlias>, With<Camera>>>,
    render_entities: Extract<Query<&'static RenderEntity>>,
    camera_map: Extract<UiCameraMap>,
    formats: Res<CameraMainPassTextureFormats>,
    mut removed: Extract<bevy::ecs::lifecycle::RemovedComponents<RepaintBoundary>>,
) {
    let mut surfaces = state.lock();
    let mut wake_surfaces = Vec::new();
    for entity in removed.read() {
        if let Ok(surface) = render_entities.get(entity) {
            views.remove_surface(surface.id());
        } else {
            views.remove_main_entity(entity);
        }
        surfaces.remove(&mut commands, boundary_id(entity));
    }

    let mut camera_mapper = camera_map.get_mapper();
    for (entity, surface, boundary, node, stack, transform, clip, target, target_info, parent) in
        &changed
    {
        let Some(source_camera) = camera_mapper.map(target) else {
            views.remove_surface(surface.id());
            surfaces.remove(&mut commands, boundary_id(entity));
            continue;
        };
        let Some(format) = formats.0.get(&source_camera).copied() else {
            views.remove_surface(surface.id());
            surfaces.remove(&mut commands, boundary_id(entity));
            continue;
        };
        let rect = Rect::from_center_size(transform.translation, node.size());
        let Some(rect) = PhysicalRect::from_min_max(
            rect.min.x.floor() as i32,
            rect.min.y.floor() as i32,
            rect.max.x.ceil() as i32,
            rect.max.y.ceil() as i32,
        ) else {
            views.remove_surface(surface.id());
            surfaces.remove(&mut commands, boundary_id(entity));
            continue;
        };
        let parent_owner = parent.and_then(|parent| owners.get(parent.parent()).ok());
        let parent_surface = parent_owner
            .and_then(|owner| render_entities.get(owner.0 .0).ok())
            .map_or(source_camera, RenderEntity::id);
        let source_main = target
            .get()
            .expect("a mapped UI target camera must have a main-world entity");
        let anti_alias = cameras.get(source_main).ok().flatten().copied();
        let size = Vec2::new(
            (rect.max_x() - rect.min_x()) as f32,
            (rect.max_y() - rect.min_y()) as f32,
        );
        let node_center = transform.translation;
        let surface_center = Vec2::new(
            (rect.min_x() + rect.max_x()) as f32 * 0.5,
            (rect.min_y() + rect.max_y()) as f32 * 0.5,
        );
        let placement = boundary.transform.compute_affine(
            target_info.scale_factor(),
            node.size(),
            target_info.physical_size().as_vec2(),
        );
        let composite_transform = Affine2::from_translation(node_center)
            * placement
            * Affine2::from_translation(surface_center - node_center);
        let composite_clip = clip.map(|clip| {
            if parent_owner.is_some() {
                clip.paint_clip
            } else {
                clip.clip
            }
        });
        let boundary_item = RetainedBoundaryItem::new(surface.id(), size, boundary.opacity);
        let visible = boundary_item.opacity() > 0.0;
        let had_view = views.views.contains_key(&surface.id());
        let was_effectively_visible = views.effectively_visible(surface.id());
        let composite_coverage: PaintCoverage = visible
            .then(|| coverage(size, composite_transform, composite_clip))
            .flatten()
            .into_iter()
            .collect();
        surfaces.upsert_boundary(
            &mut commands,
            boundary_id(entity),
            parent_surface,
            RetainedDraw {
                render_entity: Entity::PLACEHOLDER,
                camera: parent_surface,
                main_entity: MainEntity::from(entity),
                z_order: stack.0 as f32,
                paint_order: 0,
                clip: composite_clip,
                image: AssetId::<Image>::default(),
                transform: composite_transform,
                layout_translation: Vec2::ZERO,
                local_translation: Vec2::ZERO,
                item: RetainedDrawItem::Boundary(boundary_item),
            },
            composite_coverage,
        );
        views.insert(BoundaryView {
            main_entity: entity,
            surface: surface.id(),
            parent_surface,
            source_camera,
            retained_view_entity: RetainedViewEntity::new(
                entity.into(),
                Some(source_main.into()),
                2,
            ),
            rect,
            format,
            visible,
            anti_alias,
        });
        if had_view && !was_effectively_visible && views.effectively_visible(surface.id()) {
            wake_surfaces.extend(views.visible_subtree(surface.id()));
        }
    }
    drop(surfaces);
    for (surface, rect) in wake_surfaces {
        state.invalidate(surface, rect);
    }
}

fn boundary_id(entity: Entity) -> PaintId {
    PaintId {
        entity,
        family: PaintFamily::Boundary,
        ordinal: 0,
    }
}

pub(crate) fn prepare_boundary_views(
    mut commands: Commands,
    scene: Res<RetainedUiScene>,
    mut views: ResMut<BoundaryViews>,
    mut phases: ResMut<ViewSortedRenderPhases<TransparentUi>>,
) {
    for entity in views.active.drain(..) {
        commands.entity(entity).try_remove::<(
            ExtractedView,
            UiCameraView,
            UiViewTarget,
            UiAntiAlias,
            BoxShadowSamples,
        )>();
    }

    for surface in scene.dirty_surfaces() {
        let Some(view) = views.views.get(&surface) else {
            continue;
        };
        if !views.effectively_visible(surface) {
            scene.discard_damage(surface);
            continue;
        }
        let size = view.size();
        let projection = Mat4::orthographic_rh(
            view.rect.min_x() as f32,
            view.rect.max_x() as f32,
            view.rect.max_y() as f32,
            view.rect.min_y() as f32,
            0.0,
            UI_CAMERA_FAR,
        );
        let mut entity = commands.entity(surface);
        entity.insert((
            ExtractedView {
                retained_view_entity: view.retained_view_entity,
                clip_from_view: projection,
                world_from_view: GlobalTransform::from_xyz(
                    0.0,
                    0.0,
                    UI_CAMERA_FAR + UI_CAMERA_TRANSFORM_OFFSET,
                ),
                clip_from_world: None,
                target_format: view.format,
                viewport: UVec4::new(0, 0, size.x, size.y),
                color_grading: Default::default(),
                invert_culling: false,
            },
            UiCameraView(surface),
            UiViewTarget(surface),
        ));
        if let Some(anti_alias) = view.anti_alias {
            entity.insert(anti_alias);
        }
        phases.prepare_for_new_frame(view.retained_view_entity);
        views.active.push(surface);
    }
}

pub(crate) fn invalidate_volatile_paint_targets(
    scene: Res<RetainedUiScene>,
    views: Res<BoundaryViews>,
    volatile: Res<VolatileUiPaintTargets>,
) {
    for (target, size) in volatile.targets() {
        let coverage = views
            .views
            .get(&target)
            .map(|view| view.rect)
            .or_else(|| PhysicalRect::from_min_max(0, 0, size.x as i32, size.y as i32));
        if let Some(coverage) = coverage {
            scene.invalidate_volatile(target, coverage);
        }
    }
}

pub(crate) fn propagate_boundary_damage(scene: Res<RetainedUiScene>, views: Res<BoundaryViews>) {
    let mut sources = HashSet::new();
    for dirty in scene.dirty_surfaces() {
        let mut surface = dirty;
        while let Some(view) = views.views.get(&surface) {
            if !sources.insert(surface) {
                break;
            }
            surface = view.parent_surface;
        }
    }
    let mut boundaries: Vec<_> = sources
        .into_iter()
        .map(|surface| {
            let view = &views.views[&surface];
            let mut depth = 0;
            let mut parent = view.parent_surface;
            while let Some(parent_view) = views.views.get(&parent) {
                depth += 1;
                parent = parent_view.parent_surface;
            }
            (surface, boundary_id(view.main_entity), view.rect, depth)
        })
        .collect();
    boundaries.sort_by_key(|(_, _, _, depth)| core::cmp::Reverse(*depth));
    let boundaries: Vec<_> = boundaries
        .into_iter()
        .map(|(surface, id, rect, _)| (surface, id, rect))
        .collect();
    scene.propagate_boundaries(&boundaries);
}
