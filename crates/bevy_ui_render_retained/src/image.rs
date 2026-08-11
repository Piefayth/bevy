//! Change-driven retained extraction for ordinary UI images.

use crate::scene::{
    coverage, PaintFamily, PaintId, ResourceFingerprint, RetainedDraw, RetainedDrawItem,
    RetainedNodeItem, RetainedUiScene,
};
use bevy::{
    asset::{AssetEvent, AssetId, Assets, RenderAssetUsages},
    camera::visibility::InheritedVisibility,
    color::Alpha,
    ecs::{
        entity::Entity,
        lifecycle::RemovedComponents,
        message::MessageReader,
        query::{Changed, Or, With},
        system::{Commands, Query, Res, ResMut, SystemParam},
    },
    image::{Image, TextureAtlasLayout, TRANSPARENT_IMAGE_HANDLE},
    math::{Affine2, Rect, Vec2},
    render::{render_asset::RenderAssets, sync_world::MainEntity, texture::GpuImage, Extract},
    sprite::BorderRect,
    ui::{
        widget::{ImageNode, ImageNodeSize, NodeImageMode},
        CalculatedClip, ComputedNode, ComputedStackIndex, ComputedUiRenderTargetInfo,
        ComputedUiTargetCamera, Node, UiGlobalTransform, VisualBox,
    },
    ui_render::{stack_z_offsets, NodeType, UiCameraMap},
};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy)]
struct ImageDependencies {
    image: AssetId<Image>,
    atlas: Option<AssetId<TextureAtlasLayout>>,
}

#[derive(bevy::prelude::Resource, Default)]
pub(crate) struct RetainedImageDependencies {
    entities: HashMap<Entity, ImageDependencies>,
    image_readers: HashMap<AssetId<Image>, HashSet<Entity>>,
    atlas_readers: HashMap<AssetId<TextureAtlasLayout>, HashSet<Entity>>,
    image_generations: HashMap<AssetId<Image>, u64>,
    pending_images: HashSet<AssetId<Image>>,
    next_generation: u64,
}

impl RetainedImageDependencies {
    fn remove_entity(&mut self, entity: Entity) {
        let Some(old) = self.entities.remove(&entity) else {
            return;
        };
        remove_reader(&mut self.image_readers, old.image, entity);
        if let Some(atlas) = old.atlas {
            remove_reader(&mut self.atlas_readers, atlas, entity);
        }
    }

    fn set_entity(&mut self, entity: Entity, dependencies: ImageDependencies) {
        self.remove_entity(entity);
        self.entities.insert(entity, dependencies);
        self.image_readers
            .entry(dependencies.image)
            .or_default()
            .insert(entity);
        if let Some(atlas) = dependencies.atlas {
            self.atlas_readers.entry(atlas).or_default().insert(entity);
        }
    }

    fn image_changed(
        &mut self,
        id: AssetId<Image>,
        pending: bool,
        candidates: &mut HashSet<Entity>,
    ) {
        let Some(readers) = self.image_readers.get(&id) else {
            return;
        };
        self.next_generation = self
            .next_generation
            .checked_add(1)
            .expect("retained image generation exhausted");
        self.image_generations.insert(id, self.next_generation);
        if pending {
            self.pending_images.insert(id);
        } else {
            self.pending_images.remove(&id);
        }
        candidates.extend(readers);
    }

    pub(crate) fn is_pending(&self, id: AssetId<Image>) -> bool {
        self.pending_images.contains(&id)
    }

    fn atlas_changed(&self, id: AssetId<TextureAtlasLayout>, candidates: &mut HashSet<Entity>) {
        if let Some(readers) = self.atlas_readers.get(&id) {
            candidates.extend(readers);
        }
    }
}

fn remove_reader<A: bevy::asset::Asset>(
    readers: &mut HashMap<AssetId<A>, HashSet<Entity>>,
    asset: AssetId<A>,
    entity: Entity,
) {
    let Some(entities) = readers.get_mut(&asset) else {
        return;
    };
    entities.remove(&entity);
    if entities.is_empty() {
        readers.remove(&asset);
    }
}

type ImageQueryItem<'a> = (
    Entity,
    &'a ComputedNode,
    &'a ComputedStackIndex,
    &'a UiGlobalTransform,
    &'a InheritedVisibility,
    Option<&'a CalculatedClip>,
    &'a ComputedUiTargetCamera,
    &'a ImageNode,
    &'a ImageNodeSize,
);

#[derive(SystemParam)]
pub(crate) struct RemovedImageInputs<'w, 's> {
    image: RemovedComponents<'w, 's, ImageNode>,
    image_size: RemovedComponents<'w, 's, ImageNodeSize>,
    clip: RemovedComponents<'w, 's, CalculatedClip>,
    computed_node: RemovedComponents<'w, 's, ComputedNode>,
    node: RemovedComponents<'w, 's, Node>,
    stack: RemovedComponents<'w, 's, ComputedStackIndex>,
    transform: RemovedComponents<'w, 's, UiGlobalTransform>,
    visibility: RemovedComponents<'w, 's, InheritedVisibility>,
    camera: RemovedComponents<'w, 's, ComputedUiTargetCamera>,
}

#[expect(
    clippy::too_many_arguments,
    reason = "component, asset, and removal inputs independently nominate exact image changes"
)]
pub(crate) fn extract_retained_images(
    mut commands: Commands,
    state: Res<RetainedUiScene>,
    mut dependencies: ResMut<RetainedImageDependencies>,
    images: Extract<Res<Assets<Image>>>,
    texture_atlases: Extract<Res<Assets<TextureAtlasLayout>>>,
    mut image_events: Extract<MessageReader<AssetEvent<Image>>>,
    mut atlas_events: Extract<MessageReader<AssetEvent<TextureAtlasLayout>>>,
    changed: Extract<
        Query<
            ImageQueryItem<'static>,
            (
                With<ImageNode>,
                Or<(
                    Changed<ComputedNode>,
                    Changed<ComputedStackIndex>,
                    Changed<UiGlobalTransform>,
                    Changed<InheritedVisibility>,
                    Changed<CalculatedClip>,
                    Changed<ComputedUiTargetCamera>,
                    Changed<ImageNode>,
                    Changed<ImageNodeSize>,
                    Changed<Node>,
                    Changed<ComputedUiRenderTargetInfo>,
                )>,
            ),
        >,
    >,
    all: Extract<Query<ImageQueryItem<'static>, With<ImageNode>>>,
    camera_map: Extract<UiCameraMap>,
    mut removed: Extract<RemovedImageInputs>,
) {
    let mut candidates: HashSet<_> = changed.iter().map(|item| item.0).collect();
    for event in image_events.read() {
        match *event {
            AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                dependencies.image_changed(id, true, &mut candidates);
            }
            AssetEvent::Unused { id } => {
                dependencies.image_changed(id, false, &mut candidates);
            }
            AssetEvent::Removed { .. } | AssetEvent::LoadedWithDependencies { .. } => {}
        }
    }
    for event in atlas_events.read() {
        let id = match *event {
            AssetEvent::Added { id } | AssetEvent::Modified { id } | AssetEvent::Removed { id } => {
                id
            }
            AssetEvent::Unused { .. } | AssetEvent::LoadedWithDependencies { .. } => continue,
        };
        dependencies.atlas_changed(id, &mut candidates);
    }

    let RemovedImageInputs {
        image,
        image_size,
        clip,
        computed_node,
        node,
        stack,
        transform,
        visibility,
        camera,
    } = &mut *removed;
    let mut surfaces = state.lock();
    for entity in image
        .read()
        .chain(image_size.read())
        .chain(computed_node.read())
        .chain(node.read())
        .chain(stack.read())
        .chain(transform.read())
        .chain(visibility.read())
        .chain(camera.read())
    {
        dependencies.remove_entity(entity);
        surfaces.remove(&mut commands, image_id(entity));
    }
    candidates.extend(clip.read());

    let mut camera_mapper = camera_map.get_mapper();
    for entity in candidates {
        let Ok((
            entity,
            node,
            stack,
            transform,
            visibility,
            clip,
            target_camera,
            image,
            image_size,
        )) = all.get(entity)
        else {
            dependencies.remove_entity(entity);
            surfaces.remove(&mut commands, image_id(entity));
            continue;
        };

        if image.image_mode.uses_slices() {
            dependencies.remove_entity(entity);
            surfaces.remove(&mut commands, image_id(entity));
            continue;
        }

        let image_asset = image.image.id();
        let atlas_asset = image.texture_atlas.as_ref().map(|atlas| atlas.layout.id());
        dependencies.set_entity(
            entity,
            ImageDependencies {
                image: image_asset,
                atlas: atlas_asset,
            },
        );
        if let Some(asset) = images.get(image_asset) {
            if asset.asset_usage.contains(RenderAssetUsages::RENDER_WORLD) {
                dependencies.pending_images.insert(image_asset);
            } else {
                dependencies.pending_images.remove(&image_asset);
            }
        }

        let Some(camera) = camera_mapper.map(target_camera) else {
            surfaces.remove(&mut commands, image_id(entity));
            continue;
        };
        let visual_box = match image.visual_box {
            VisualBox::ContentBox => node.content_box(),
            VisualBox::PaddingBox => node.padding_box(),
            VisualBox::BorderBox => node.border_box(),
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
            .and_then(|atlas| atlas.texture_rect(&texture_atlases))
            .map(|rect| rect.as_rect());
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
            let scaling = size / rect.size();
            rect.min *= scaling;
            rect.max *= scaling;
            Some(scaling)
        } else {
            None
        };
        let transform = transform.affine() * Affine2::from_translation(visual_box.center());
        let clip = clip.map(|clip| clip.clip);
        let painted = visibility.get()
            && !image.color.is_fully_transparent()
            && image_asset != TRANSPARENT_IMAGE_HANDLE.id()
            && !node.is_empty()
            && !visual_box.size().cmple(Vec2::ZERO).any();
        let image_generation = dependencies
            .image_generations
            .get(&image_asset)
            .copied()
            .unwrap_or_default();
        surfaces.upsert(
            &mut commands,
            image_id(entity),
            camera,
            RetainedDraw {
                render_entity: Entity::PLACEHOLDER,
                camera,
                main_entity: MainEntity::from(entity),
                z_order: stack.0 as f32 + stack_z_offsets::IMAGE,
                paint_order: 0,
                clip,
                image: image_asset,
                transform,
                item: RetainedDrawItem::Node(RetainedNodeItem {
                    color: image.color.into(),
                    rect,
                    atlas_scaling,
                    flip_x: image.flip_x,
                    flip_y: image.flip_y,
                    border: BorderRect::ZERO,
                    border_radius: node.border_radius,
                    node_type: NodeType::Rect,
                }),
            },
            ResourceFingerprint::Generation(image_generation),
            painted
                .then(|| coverage(size, transform, clip))
                .flatten()
                .into_iter()
                .collect(),
        );
    }
}

fn image_id(entity: Entity) -> PaintId {
    PaintId {
        entity,
        family: PaintFamily::Image,
        ordinal: 0,
    }
}

pub(crate) fn resolve_ready_images(
    mut dependencies: ResMut<RetainedImageDependencies>,
    gpu_images: Res<RenderAssets<GpuImage>>,
) {
    let mut pending = core::mem::take(&mut dependencies.pending_images);
    pending.retain(|image| {
        dependencies.image_readers.contains_key(image) && gpu_images.get(*image).is_none()
    });
    dependencies.pending_images = pending;
}
