//! Change-driven retained extraction for ordinary UI images.

use crate::sampled_image::{ImageReader, ImageSample, RetainedSampledImages};
use crate::scene::{
    coverage, PaintFamily, PaintId, ResourceFingerprint, RetainedDraw, RetainedDrawItem,
    RetainedNodeItem, RetainedTextureSliceItem, RetainedUiScene,
};
use bevy::{
    asset::{AssetEvent, AssetId, Assets},
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
    render::{render_resource::DefaultImageSamplerDescriptor, sync_world::MainEntity, Extract},
    sprite::{BorderRect, SpriteImageMode},
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
    atlas: Option<AssetId<TextureAtlasLayout>>,
}

#[derive(bevy::prelude::Resource, Default)]
pub(crate) struct RetainedImageDependencies {
    entities: HashMap<Entity, ImageDependencies>,
    atlas_readers: HashMap<AssetId<TextureAtlasLayout>, HashSet<Entity>>,
}

impl RetainedImageDependencies {
    fn remove_entity(&mut self, entity: Entity) {
        let Some(old) = self.entities.remove(&entity) else {
            return;
        };
        if let Some(atlas) = old.atlas {
            remove_reader(&mut self.atlas_readers, atlas, entity);
        }
    }

    fn set_entity(&mut self, entity: Entity, dependencies: ImageDependencies) {
        self.remove_entity(entity);
        self.entities.insert(entity, dependencies);
        if let Some(atlas) = dependencies.atlas {
            self.atlas_readers.entry(atlas).or_default().insert(entity);
        }
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
    mut sampled_images: ResMut<RetainedSampledImages>,
    default_sampler: Res<DefaultImageSamplerDescriptor>,
    images: Extract<Res<Assets<Image>>>,
    texture_atlases: Extract<Res<Assets<TextureAtlasLayout>>>,
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
    candidates.extend(sampled_images.take_nodes());
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
        sampled_images.remove_reader(ImageReader::Node(entity));
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
            sampled_images.remove_reader(ImageReader::Node(entity));
            surfaces.remove(&mut commands, image_id(entity));
            continue;
        };

        let image_asset = image.image.id();
        let atlas_asset = image.texture_atlas.as_ref().map(|atlas| atlas.layout.id());
        let Some(camera) = camera_mapper.map(target_camera) else {
            dependencies.remove_entity(entity);
            sampled_images.remove_reader(ImageReader::Node(entity));
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
        let source_rect = match (atlas_rect, image.rect) {
            (None, None) => None,
            (None, Some(image_rect)) => Some(image_rect),
            (Some(atlas_rect), None) => Some(atlas_rect),
            (Some(atlas_rect), Some(mut image_rect)) => {
                image_rect.min += atlas_rect.min;
                image_rect.max += atlas_rect.min;
                Some(image_rect)
            }
        };
        let item = match &image.image_mode {
            NodeImageMode::Sliced(slicer) => {
                RetainedDrawItem::TextureSlice(RetainedTextureSliceItem {
                    stack_index: stack.0,
                    rect: Rect {
                        min: Vec2::ZERO,
                        max: size,
                    },
                    atlas_rect: source_rect,
                    color: image.color.into(),
                    image_scale_mode: SpriteImageMode::Sliced(slicer.clone()),
                    flip_x: image.flip_x,
                    flip_y: image.flip_y,
                    inverse_scale_factor: node.inverse_scale_factor,
                })
            }
            NodeImageMode::Tiled {
                tile_x,
                tile_y,
                stretch_value,
            } => RetainedDrawItem::TextureSlice(RetainedTextureSliceItem {
                stack_index: stack.0,
                rect: Rect {
                    min: Vec2::ZERO,
                    max: size,
                },
                atlas_rect: source_rect,
                color: image.color.into(),
                image_scale_mode: SpriteImageMode::Tiled {
                    tile_x: *tile_x,
                    tile_y: *tile_y,
                    stretch_value: *stretch_value,
                },
                flip_x: image.flip_x,
                flip_y: image.flip_y,
                inverse_scale_factor: node.inverse_scale_factor,
            }),
            NodeImageMode::Auto | NodeImageMode::Stretch => {
                let mut rect = source_rect.unwrap_or(Rect {
                    min: Vec2::ZERO,
                    max: size,
                });
                let atlas_scaling = if source_rect.is_some() {
                    let scaling = size / rect.size();
                    rect.min *= scaling;
                    rect.max *= scaling;
                    Some(scaling)
                } else {
                    None
                };
                RetainedDrawItem::Node(RetainedNodeItem {
                    color: image.color.into(),
                    rect,
                    atlas_scaling,
                    flip_x: image.flip_x,
                    flip_y: image.flip_y,
                    border: BorderRect::ZERO,
                    border_radius: node.border_radius,
                    node_type: NodeType::Rect,
                })
            }
        };
        let sample = source_rect.map_or_else(
            || ImageSample::all(image_asset),
            |rect| ImageSample::rect(image_asset, rect),
        );
        let transform = transform.affine() * Affine2::from_translation(visual_box.center());
        let clip = clip.map(|clip| clip.clip);
        let painted = visibility.get()
            && !image.color.is_fully_transparent()
            && image_asset != TRANSPARENT_IMAGE_HANDLE.id()
            && !node.is_empty()
            && !visual_box.size().cmple(Vec2::ZERO).any();
        let resources = if painted {
            dependencies.set_entity(entity, ImageDependencies { atlas: atlas_asset });
            sampled_images.replace_reader(
                ImageReader::Node(entity),
                [sample],
                &images,
                &default_sampler,
            );
            sampled_images.mark_pending(image_asset, images.get(image_asset));
            ResourceFingerprint::Revisions(sampled_images.revisions(image_asset, [sample]))
        } else {
            dependencies.remove_entity(entity);
            sampled_images.remove_reader(ImageReader::Node(entity));
            ResourceFingerprint::None
        };
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
                item,
            },
            resources,
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
