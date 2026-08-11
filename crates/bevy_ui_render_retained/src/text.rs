//! Change-driven retained extraction for ordinary UI text glyphs.

use crate::scene::{
    coverage, PaintFamily, PaintId, ResourceFingerprint, RetainedDraw, RetainedDrawItem,
    RetainedGlyph, RetainedUiScene, RetainedUiSurfaces,
};
use bevy::{
    asset::{AssetEvent, AssetId, Assets, RenderAssetUsages},
    camera::visibility::InheritedVisibility,
    color::{Alpha, LinearRgba},
    ecs::{
        entity::Entity,
        lifecycle::RemovedComponents,
        message::MessageReader,
        query::{Changed, Or, With},
        system::{Commands, Query, Res, ResMut, SystemParam},
    },
    image::{Image, ImageSampler},
    math::{Affine2, Rect, UVec3, Vec2},
    render::{
        render_asset::RenderAssets, render_resource::TextureUsages, sync_world::MainEntity,
        texture::GpuImage, Extract,
    },
    text::{ComputedTextBlock, PositionedGlyph, TextColor, TextCursorStyle, TextLayoutInfo},
    ui::{
        widget::{Text, TextScroll},
        CalculatedClip, ComputedNode, ComputedStackIndex, ComputedUiRenderTargetInfo,
        ComputedUiTargetCamera, Node, UiGlobalTransform,
    },
    ui_render::{stack_z_offsets, UiCameraMap},
};
use std::collections::{HashMap, HashSet};

#[derive(Default)]
struct TextRootDependencies {
    sections: Vec<Entity>,
    images: Vec<AssetId<Image>>,
    samples: Vec<AtlasSample>,
    runs: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct AtlasSample {
    image: AssetId<Image>,
    min: [i64; 2],
    max: [i64; 2],
}

impl AtlasSample {
    fn new(image: AssetId<Image>, rect: Rect) -> Self {
        Self {
            image,
            min: [
                (rect.min.x.floor() as i64).saturating_sub(1),
                (rect.min.y.floor() as i64).saturating_sub(1),
            ],
            max: [
                (rect.max.x.ceil() as i64).saturating_add(1),
                (rect.max.y.ceil() as i64).saturating_add(1),
            ],
        }
    }
}

struct AtlasSampleState {
    pixels: Option<Box<[u8]>>,
    revision: u64,
    readers: HashSet<Entity>,
}

struct AtlasMetadataState {
    value: Option<Image>,
    revision: u64,
}

#[derive(bevy::prelude::Resource, Default)]
pub(crate) struct RetainedTextDependencies {
    roots: HashMap<Entity, TextRootDependencies>,
    section_roots: HashMap<Entity, HashSet<Entity>>,
    image_readers: HashMap<AssetId<Image>, HashSet<Entity>>,
    image_metadata: HashMap<AssetId<Image>, AtlasMetadataState>,
    samples: HashMap<AtlasSample, AtlasSampleState>,
    pending_images: HashSet<AssetId<Image>>,
    next_revision: u64,
}

impl RetainedTextDependencies {
    fn remove_root(&mut self, root: Entity) -> u32 {
        let runs = self.detach_root(root);
        self.prune_unused();
        runs
    }

    fn detach_root(&mut self, root: Entity) -> u32 {
        let Some(old) = self.roots.remove(&root) else {
            return 0;
        };
        for section in old.sections {
            remove_reader(&mut self.section_roots, section, root);
        }
        for image in old.images {
            remove_reader(&mut self.image_readers, image, root);
        }
        for sample in old.samples {
            if let Some(state) = self.samples.get_mut(&sample) {
                state.readers.remove(&root);
            }
        }
        old.runs
    }

    fn prune_unused(&mut self) {
        self.image_metadata
            .retain(|image, _| self.image_readers.contains_key(image));
        self.pending_images
            .retain(|image| self.image_readers.contains_key(image));
        self.samples.retain(|_, state| !state.readers.is_empty());
    }

    fn set_root(
        &mut self,
        root: Entity,
        sections: impl IntoIterator<Item = Entity>,
        images: impl IntoIterator<Item = AssetId<Image>>,
        samples: impl IntoIterator<Item = AtlasSample>,
        runs: u32,
        assets: &Assets<Image>,
    ) -> u32 {
        let old_runs = self.detach_root(root);
        let mut dependencies = TextRootDependencies {
            runs,
            ..Default::default()
        };
        for section in sections {
            if !dependencies.sections.contains(&section) {
                dependencies.sections.push(section);
                self.section_roots.entry(section).or_default().insert(root);
            }
        }
        for image in images {
            if !dependencies.images.contains(&image) {
                dependencies.images.push(image);
                self.image_readers.entry(image).or_default().insert(root);
                self.image_metadata
                    .entry(image)
                    .or_insert_with(|| AtlasMetadataState {
                        value: assets.get(image).map(image_metadata),
                        revision: 0,
                    });
            }
        }
        for sample in samples {
            if !dependencies.samples.contains(&sample) {
                dependencies.samples.push(sample);
                self.samples
                    .entry(sample)
                    .or_insert_with(|| AtlasSampleState {
                        pixels: assets
                            .get(sample.image)
                            .and_then(|image| sample_pixels(image, sample)),
                        revision: 0,
                        readers: HashSet::default(),
                    })
                    .readers
                    .insert(root);
            }
        }
        self.roots.insert(root, dependencies);
        self.prune_unused();
        old_runs
    }

    fn image_changed(
        &mut self,
        image: AssetId<Image>,
        asset: Option<&Image>,
        pending: bool,
        candidates: &mut HashSet<Entity>,
    ) {
        let Some(readers) = self.image_readers.get(&image).cloned() else {
            return;
        };
        let mut relevant_change = false;
        let metadata = asset.map(image_metadata);
        let metadata_changed = self
            .image_metadata
            .get(&image)
            .is_none_or(|old| metadata.is_none() || old.value != metadata);
        if metadata_changed {
            relevant_change = true;
            let revision = self.new_revision();
            self.image_metadata.insert(
                image,
                AtlasMetadataState {
                    value: metadata,
                    revision,
                },
            );
            candidates.extend(&readers);
        }

        let samples: Vec<_> = self
            .samples
            .keys()
            .filter(|sample| sample.image == image)
            .copied()
            .collect();
        for sample in samples {
            let changed = self
                .samples
                .get(&sample)
                .is_some_and(|old| match (&old.pixels, asset) {
                    (Some(old), Some(image)) => {
                        !sample_matches(image, sample, old).unwrap_or(false)
                    }
                    _ => true,
                });
            if changed {
                let pixels = asset.and_then(|image| sample_pixels(image, sample));
                relevant_change = true;
                let revision = self.new_revision();
                let state = self.samples.get_mut(&sample).unwrap();
                state.pixels = pixels;
                state.revision = revision;
                candidates.extend(&state.readers);
            }
        }
        if pending && relevant_change {
            self.pending_images.insert(image);
        } else if !pending {
            self.pending_images.remove(&image);
        }
    }

    fn revisions(
        &self,
        image: AssetId<Image>,
        samples: impl IntoIterator<Item = AtlasSample>,
    ) -> Box<[u64]> {
        core::iter::once(
            self.image_metadata
                .get(&image)
                .map_or(0, |state| state.revision),
        )
        .chain(
            samples
                .into_iter()
                .map(|sample| self.samples.get(&sample).map_or(0, |state| state.revision)),
        )
        .collect()
    }

    fn new_revision(&mut self) -> u64 {
        self.next_revision = self
            .next_revision
            .checked_add(1)
            .expect("retained text atlas revision exhausted");
        self.next_revision
    }

    pub(crate) fn is_pending(&self, image: AssetId<Image>) -> bool {
        self.pending_images.contains(&image)
    }
}

fn image_metadata(image: &Image) -> Image {
    let mut metadata = Image {
        data: None,
        data_order: image.data_order,
        texture_descriptor: image.texture_descriptor.clone(),
        sampler: image.sampler.clone(),
        texture_view_descriptor: image.texture_view_descriptor.clone(),
        asset_usage: image.asset_usage & RenderAssetUsages::RENDER_WORLD,
        copy_on_resize: false,
    };
    metadata.texture_descriptor.label = None;
    metadata.texture_descriptor.usage = TextureUsages::empty();
    if let ImageSampler::Descriptor(sampler) = &mut metadata.sampler {
        sampler.label = None;
    }
    if let Some(view) = &mut metadata.texture_view_descriptor {
        view.label = None;
        view.usage = None;
    }
    metadata
}

fn sample_pixels(image: &Image, sample: AtlasSample) -> Option<Box<[u8]>> {
    let [min_x, min_y, max_x, max_y] = sample_bounds(image, sample);
    let mut pixels = Vec::new();
    for y in min_y..max_y {
        for x in min_x..max_x {
            pixels.extend_from_slice(image.pixel_bytes(UVec3::new(x, y, 0)).ok()?);
        }
    }
    Some(pixels.into_boxed_slice())
}

fn sample_matches(image: &Image, sample: AtlasSample, old: &[u8]) -> Option<bool> {
    let [min_x, min_y, max_x, max_y] = sample_bounds(image, sample);
    let mut offset = 0;
    for y in min_y..max_y {
        for x in min_x..max_x {
            let pixel = image.pixel_bytes(UVec3::new(x, y, 0)).ok()?;
            let end = offset + pixel.len();
            if old.get(offset..end) != Some(pixel) {
                return Some(false);
            }
            offset = end;
        }
    }
    Some(offset == old.len())
}

fn sample_bounds(image: &Image, sample: AtlasSample) -> [u32; 4] {
    let size = image.texture_descriptor.size;
    [
        sample.min[0].clamp(0, i64::from(size.width)) as u32,
        sample.min[1].clamp(0, i64::from(size.height)) as u32,
        sample.max[0].clamp(0, i64::from(size.width)) as u32,
        sample.max[1].clamp(0, i64::from(size.height)) as u32,
    ]
}

fn remove_reader<K: Eq + core::hash::Hash + Copy>(
    readers: &mut HashMap<K, HashSet<Entity>>,
    key: K,
    entity: Entity,
) {
    let Some(entities) = readers.get_mut(&key) else {
        return;
    };
    entities.remove(&entity);
    if entities.is_empty() {
        readers.remove(&key);
    }
}

type TextQueryItem<'a> = (
    Entity,
    &'a Node,
    &'a ComputedNode,
    &'a ComputedStackIndex,
    &'a UiGlobalTransform,
    &'a InheritedVisibility,
    Option<&'a CalculatedClip>,
    &'a ComputedUiTargetCamera,
    &'a ComputedTextBlock,
    &'a TextColor,
    &'a TextLayoutInfo,
    Option<&'a TextScroll>,
    Option<&'a TextCursorStyle>,
);

#[derive(SystemParam)]
pub(crate) struct RemovedTextInputs<'w, 's> {
    text: RemovedComponents<'w, 's, Text>,
    layout: RemovedComponents<'w, 's, TextLayoutInfo>,
    computed_text: RemovedComponents<'w, 's, ComputedTextBlock>,
    clip: RemovedComponents<'w, 's, CalculatedClip>,
    scroll: RemovedComponents<'w, 's, TextScroll>,
    cursor: RemovedComponents<'w, 's, TextCursorStyle>,
    color: RemovedComponents<'w, 's, TextColor>,
    computed_node: RemovedComponents<'w, 's, ComputedNode>,
    node: RemovedComponents<'w, 's, Node>,
    stack: RemovedComponents<'w, 's, ComputedStackIndex>,
    transform: RemovedComponents<'w, 's, UiGlobalTransform>,
    visibility: RemovedComponents<'w, 's, InheritedVisibility>,
    camera: RemovedComponents<'w, 's, ComputedUiTargetCamera>,
}

#[expect(
    clippy::too_many_arguments,
    reason = "component, span, atlas, and lifecycle inputs independently nominate text changes"
)]
pub(crate) fn extract_retained_text(
    mut commands: Commands,
    state: Res<RetainedUiScene>,
    mut dependencies: ResMut<RetainedTextDependencies>,
    images: Extract<Res<Assets<Image>>>,
    mut image_events: Extract<MessageReader<AssetEvent<Image>>>,
    changed: Extract<
        Query<
            TextQueryItem<'static>,
            (
                With<Text>,
                Or<(
                    Changed<ComputedNode>,
                    Changed<ComputedStackIndex>,
                    Changed<UiGlobalTransform>,
                    Changed<InheritedVisibility>,
                    Changed<CalculatedClip>,
                    Changed<ComputedUiTargetCamera>,
                    Changed<ComputedTextBlock>,
                    Changed<TextColor>,
                    Changed<TextLayoutInfo>,
                    Changed<TextScroll>,
                    Changed<TextCursorStyle>,
                    Changed<Node>,
                    Changed<ComputedUiRenderTargetInfo>,
                )>,
            ),
        >,
    >,
    all: Extract<Query<TextQueryItem<'static>, With<Text>>>,
    changed_colors: Extract<Query<Entity, Changed<TextColor>>>,
    text_styles: Extract<Query<&TextColor>>,
    camera_map: Extract<UiCameraMap>,
    mut removed: Extract<RemovedTextInputs>,
) {
    let mut candidates: HashSet<_> = changed.iter().map(|item| item.0).collect();
    for section in &changed_colors {
        if let Some(roots) = dependencies.section_roots.get(&section) {
            candidates.extend(roots);
        }
    }
    for event in image_events.read() {
        match *event {
            AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                dependencies.image_changed(id, images.get(id), true, &mut candidates);
            }
            AssetEvent::Unused { id } => {
                dependencies.image_changed(id, images.get(id), false, &mut candidates);
            }
            AssetEvent::Removed { .. } | AssetEvent::LoadedWithDependencies { .. } => {}
        }
    }

    let RemovedTextInputs {
        text,
        layout,
        computed_text,
        clip,
        scroll,
        cursor,
        color,
        computed_node,
        node,
        stack,
        transform,
        visibility,
        camera,
    } = &mut *removed;
    for section in color.read() {
        if let Some(roots) = dependencies.section_roots.get(&section) {
            candidates.extend(roots);
        }
    }

    let mut surfaces = state.lock();
    for root in text
        .read()
        .chain(layout.read())
        .chain(computed_text.read())
        .chain(computed_node.read())
        .chain(node.read())
        .chain(stack.read())
        .chain(transform.read())
        .chain(visibility.read())
        .chain(camera.read())
    {
        remove_text_root(&mut dependencies, &mut surfaces, &mut commands, root);
    }
    candidates.extend(clip.read());
    candidates.extend(scroll.read());
    candidates.extend(cursor.read());

    let mut camera_mapper = camera_map.get_mapper();
    for root in candidates {
        let Ok((
            root,
            _,
            node,
            stack,
            global_transform,
            visibility,
            clip,
            target_camera,
            computed_block,
            text_color,
            layout,
            scroll,
            cursor_style,
        )) = all.get(root)
        else {
            remove_text_root(&mut dependencies, &mut surfaces, &mut commands, root);
            continue;
        };

        let transform = global_transform.affine()
            * Affine2::from_translation(
                node.content_box().min - scroll.map_or(Vec2::ZERO, |scroll| scroll.0),
            );
        let clip = if scroll.is_some() {
            let content_box = node.content_box();
            let text_clip = Rect::from_center_size(
                global_transform.affine().translation + content_box.center(),
                content_box.size(),
            );
            Some(clip.map_or(text_clip, |clip| clip.clip.intersect(text_clip)))
        } else {
            clip.map(|clip| clip.clip)
        };
        let camera = camera_mapper.map(target_camera);
        let visible = camera.is_some() && visibility.get() && !node.is_empty();
        let selected_text_color = cursor_style
            .and_then(|style| style.selected_text_color)
            .map(|color| color.to_linear());
        let mut color = text_color.0.to_linear();
        let mut section_index = 0;
        let mut runs: Vec<(AssetId<Image>, Vec<RetainedGlyph>, Vec<AtlasSample>)> = Vec::new();

        for PositionedGlyph {
            position,
            atlas_info,
            section_index: glyph_section,
            ..
        } in &layout.glyphs
        {
            if section_index != *glyph_section {
                if let Some(section) = computed_block.entities().get(*glyph_section) {
                    color = text_styles
                        .get(section.entity)
                        .map(|color| LinearRgba::from(color.0))
                        .unwrap_or_default();
                }
                section_index = *glyph_section;
            }
            let glyph_color = if !atlas_info.is_alpha_mask {
                LinearRgba::WHITE
            } else if let Some(selected) = selected_text_color
                && layout.selection_rects.iter().any(|selection| {
                    let glyph = Rect::from_center_size(*position, atlas_info.rect.size());
                    selection.contains(glyph.min) && selection.contains(glyph.max)
                })
            {
                selected
            } else {
                color
            };
            if glyph_color.is_fully_transparent() {
                continue;
            }
            let glyph = RetainedGlyph::new(glyph_color, *position, atlas_info.rect);
            let sample = AtlasSample::new(atlas_info.texture, atlas_info.rect);
            if let Some((texture, glyphs, samples)) = runs.last_mut()
                && *texture == atlas_info.texture
            {
                glyphs.push(glyph);
                if !samples.contains(&sample) {
                    samples.push(sample);
                }
            } else {
                runs.push((atlas_info.texture, vec![glyph], vec![sample]));
            }
        }

        let image_ids: Vec<_> = runs.iter().map(|(image, _, _)| *image).collect();
        let atlas_samples: Vec<_> = runs
            .iter()
            .flat_map(|(_, _, samples)| samples.iter().copied())
            .collect();
        let old_runs = dependencies.set_root(
            root,
            computed_block
                .entities()
                .iter()
                .map(|section| section.entity),
            image_ids.iter().copied(),
            atlas_samples,
            runs.len() as u32,
            &images,
        );
        for &image in &image_ids {
            if images
                .get(image)
                .is_some_and(|asset| asset.asset_usage.contains(RenderAssetUsages::RENDER_WORLD))
            {
                dependencies.pending_images.insert(image);
            }
        }

        for ordinal in runs.len() as u32..old_runs {
            surfaces.remove(&mut commands, text_id(root, ordinal));
        }
        let Some(camera) = camera.filter(|_| visible) else {
            for ordinal in 0..runs.len() as u32 {
                surfaces.remove(&mut commands, text_id(root, ordinal));
            }
            continue;
        };

        for (ordinal, (image, glyphs, samples)) in runs.into_iter().enumerate() {
            let glyph_coverage = glyphs
                .iter()
                .filter_map(|glyph| {
                    coverage(
                        glyph.rect().size(),
                        transform * Affine2::from_translation(glyph.translation()),
                        clip,
                    )
                })
                .collect();
            let revisions = dependencies.revisions(image, samples);
            surfaces.upsert(
                &mut commands,
                text_id(root, ordinal as u32),
                camera,
                RetainedDraw {
                    render_entity: Entity::PLACEHOLDER,
                    camera,
                    main_entity: MainEntity::from(root),
                    z_order: stack.0 as f32 + stack_z_offsets::TEXT,
                    clip,
                    image,
                    transform,
                    item: RetainedDrawItem::Glyphs(glyphs.into_boxed_slice()),
                },
                ResourceFingerprint::Revisions(revisions),
                glyph_coverage,
            );
        }
    }
}

fn remove_text_root(
    dependencies: &mut RetainedTextDependencies,
    surfaces: &mut RetainedUiSurfaces,
    commands: &mut Commands,
    root: Entity,
) {
    let runs = dependencies.remove_root(root);
    for ordinal in 0..runs {
        surfaces.remove(commands, text_id(root, ordinal));
    }
}

fn text_id(entity: Entity, ordinal: u32) -> PaintId {
    PaintId {
        entity,
        family: PaintFamily::Text,
        ordinal,
    }
}

pub(crate) fn resolve_ready_text_atlases(
    mut dependencies: ResMut<RetainedTextDependencies>,
    gpu_images: Res<RenderAssets<GpuImage>>,
) {
    let mut pending = core::mem::take(&mut dependencies.pending_images);
    pending.retain(|image| {
        dependencies.image_readers.contains_key(image) && gpu_images.get(*image).is_none()
    });
    dependencies.pending_images = pending;
}
