//! Change-driven retained extraction for camera-backed UI viewport nodes.

use crate::{
    sampled_image::{ImageReader, ImageSample, RetainedSampledImages, SampledImageState},
    scene::{
        coverage, PaintFamily, PaintId, ResourceFingerprint, RetainedDraw, RetainedDrawItem,
        RetainedNodeItem, RetainedUiScene, RetainedUiSurfaces,
    },
};
use bevy::{
    asset::{Assets, Handle},
    camera::visibility::InheritedVisibility,
    camera::{Camera, RenderTarget},
    ecs::{
        entity::Entity,
        lifecycle::RemovedComponents,
        query::{Added, Changed, Or, With},
        system::{Commands, Query, Res, ResMut, SystemParam},
    },
    image::Image,
    math::{Rect, Vec2},
    render::{render_resource::DefaultImageSamplerDescriptor, sync_world::MainEntity, Extract},
    ui::{
        widget::ViewportNode, CalculatedClip, ComputedNode, ComputedStackIndex,
        ComputedUiTargetCamera, Display, Node, UiGlobalTransform,
    },
    ui_render::{stack_z_offsets, NodeType, UiCameraMap},
};
use std::collections::{HashMap, HashSet};

#[derive(bevy::prelude::Resource, Default)]
pub(crate) struct RetainedViewportDependencies {
    sources: HashMap<Entity, Entity>,
    readers: HashMap<Entity, HashSet<Entity>>,
}

impl RetainedViewportDependencies {
    fn set_source(&mut self, viewport: Entity, source: Option<Entity>) {
        self.remove(viewport);
        if let Some(source) = source {
            self.sources.insert(viewport, source);
            self.readers.entry(source).or_default().insert(viewport);
        }
    }

    fn remove(&mut self, viewport: Entity) {
        let Some(source) = self.sources.remove(&viewport) else {
            return;
        };
        let Some(readers) = self.readers.get_mut(&source) else {
            return;
        };
        readers.remove(&viewport);
        if readers.is_empty() {
            self.readers.remove(&source);
        }
    }

    fn source_changed(&self, source: Entity, candidates: &mut HashSet<Entity>) {
        if let Some(readers) = self.readers.get(&source) {
            candidates.extend(readers);
        }
    }
}

type ViewportQueryItem<'a> = (
    Entity,
    &'a Node,
    &'a ComputedNode,
    &'a ComputedStackIndex,
    &'a UiGlobalTransform,
    &'a InheritedVisibility,
    Option<&'a CalculatedClip>,
    &'a ComputedUiTargetCamera,
    &'a ViewportNode,
);

#[derive(SystemParam)]
pub(crate) struct RemovedViewportInputs<'w, 's> {
    viewport: RemovedComponents<'w, 's, ViewportNode>,
    clip: RemovedComponents<'w, 's, CalculatedClip>,
    computed_node: RemovedComponents<'w, 's, ComputedNode>,
    node: RemovedComponents<'w, 's, Node>,
    stack: RemovedComponents<'w, 's, ComputedStackIndex>,
    transform: RemovedComponents<'w, 's, UiGlobalTransform>,
    visibility: RemovedComponents<'w, 's, InheritedVisibility>,
    camera: RemovedComponents<'w, 's, ComputedUiTargetCamera>,
    source_target: RemovedComponents<'w, 's, RenderTarget>,
    source_camera: RemovedComponents<'w, 's, Camera>,
}

#[expect(
    clippy::too_many_arguments,
    reason = "node, source-camera, image, and lifecycle inputs independently nominate viewport changes"
)]
pub(crate) fn extract_retained_viewports(
    mut commands: Commands,
    state: Res<RetainedUiScene>,
    mut dependencies: ResMut<RetainedViewportDependencies>,
    sampled_images: Res<RetainedSampledImages>,
    default_sampler: Res<DefaultImageSamplerDescriptor>,
    images: Extract<Res<Assets<Image>>>,
    changed: Extract<
        Query<
            ViewportQueryItem<'static>,
            (
                With<ViewportNode>,
                Or<(
                    Changed<ComputedNode>,
                    Changed<ComputedStackIndex>,
                    Changed<InheritedVisibility>,
                    Changed<CalculatedClip>,
                    Changed<ComputedUiTargetCamera>,
                    Changed<ViewportNode>,
                    Changed<Node>,
                )>,
            ),
        >,
    >,
    all: Extract<Query<ViewportQueryItem<'static>, With<ViewportNode>>>,
    source_cameras: Extract<Query<(&'static Camera, &'static RenderTarget)>>,
    changed_sources: Extract<Query<Entity, Or<(Changed<RenderTarget>, Added<Camera>)>>>,
    camera_map: Extract<UiCameraMap>,
    mut removed: Extract<RemovedViewportInputs>,
) {
    let mut sampled_images = sampled_images.lock();
    let mut extra_candidates: HashSet<_> = sampled_images.take_viewports().into_iter().collect();
    let RemovedViewportInputs {
        viewport,
        clip,
        computed_node,
        node,
        stack,
        transform,
        visibility,
        camera,
        source_target,
        source_camera,
    } = &mut *removed;
    for source in changed_sources
        .iter()
        .chain(source_target.read())
        .chain(source_camera.read())
    {
        dependencies.source_changed(source, &mut extra_candidates);
    }
    extra_candidates.extend(viewport.read());
    extra_candidates.extend(clip.read());
    extra_candidates.retain(|entity| !changed.contains(*entity));

    let mut surfaces = state.lock();
    for entity in computed_node
        .read()
        .chain(node.read())
        .chain(stack.read())
        .chain(transform.read())
        .chain(visibility.read())
        .chain(camera.read())
    {
        remove_viewport(
            &mut dependencies,
            &mut sampled_images,
            &mut surfaces,
            &mut commands,
            entity,
        );
    }
    extra_candidates.retain(|entity| {
        if all.contains(*entity) {
            true
        } else {
            remove_viewport(
                &mut dependencies,
                &mut sampled_images,
                &mut surfaces,
                &mut commands,
                *entity,
            );
            false
        }
    });

    let mut camera_mapper = camera_map.get_mapper();
    for (entity, source_node, node, stack, transform, visibility, clip, target_camera, viewport) in
        changed.iter().chain(
            extra_candidates
                .into_iter()
                .filter_map(|entity| all.get(entity).ok()),
        )
    {
        dependencies.set_source(entity, viewport.camera);
        let id = viewport_id(entity);
        let reader = ImageReader::Viewport(entity);
        let Some(camera) = camera_mapper.map(target_camera) else {
            surfaces.remove(&mut commands, id);
            sampled_images.remove_reader(reader);
            continue;
        };
        let Some(source) = viewport.camera else {
            surfaces.remove(&mut commands, id);
            sampled_images.remove_reader(reader);
            continue;
        };
        let Some(image) = source_cameras
            .get(source)
            .ok()
            .and_then(|(_, target)| target.as_image())
            .map(Handle::id)
        else {
            surfaces.remove(&mut commands, id);
            sampled_images.remove_reader(reader);
            continue;
        };

        let transform = transform.affine();
        let clip = clip.map(|clip| clip.clip);
        let painted = visibility.get()
            && source_node.display != Display::None
            && !node.is_empty()
            && !node.size().cmple(Vec2::ZERO).any();
        let resource = if painted {
            let sample = ImageSample::all(image);
            sampled_images.replace_reader(reader, [sample], &images, &default_sampler);
            sampled_images.mark_pending(image, images.get(image));
            ResourceFingerprint::Revisions(sampled_images.revisions(image, [sample]))
        } else {
            sampled_images.remove_reader(reader);
            ResourceFingerprint::None
        };
        surfaces.upsert(
            &mut commands,
            id,
            camera,
            RetainedDraw {
                render_entity: Entity::PLACEHOLDER,
                camera,
                main_entity: MainEntity::from(entity),
                z_order: stack.0 as f32 + stack_z_offsets::IMAGE,
                paint_order: 0,
                clip,
                image,
                transform,
                local_translation: Vec2::ZERO,
                item: RetainedDrawItem::Node(RetainedNodeItem {
                    color: bevy::color::LinearRgba::WHITE,
                    rect: Rect::from_corners(Vec2::ZERO, node.size()),
                    atlas_scaling: None,
                    image_extent: None,
                    flip_x: false,
                    flip_y: false,
                    border: node.border(),
                    border_radius: node.border_radius(),
                    node_type: NodeType::Rect,
                }),
            },
            resource,
            painted
                .then(|| coverage(node.size(), transform, clip))
                .flatten()
                .into_iter()
                .collect(),
            painted,
        );
    }
}

fn viewport_id(entity: Entity) -> PaintId {
    PaintId {
        entity,
        family: PaintFamily::Viewport,
        ordinal: 0,
    }
}

fn remove_viewport(
    dependencies: &mut RetainedViewportDependencies,
    sampled_images: &mut SampledImageState,
    surfaces: &mut RetainedUiSurfaces,
    commands: &mut Commands,
    entity: Entity,
) {
    dependencies.remove(entity);
    sampled_images.remove_reader(ImageReader::Viewport(entity));
    surfaces.remove(commands, viewport_id(entity));
}
