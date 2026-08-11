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
        query::{Changed, Has, Or, With},
        system::{Commands, Query, Res, ResMut, SystemParam},
    },
    image::{Image, ImageSampler},
    input_focus::InputFocus,
    math::{Affine2, Rect, UVec3, Vec2},
    render::{
        render_asset::RenderAssets, render_resource::TextureUsages, sync_world::MainEntity,
        texture::GpuImage, Extract,
    },
    sprite::BorderRect,
    text::{
        ComputedTextBlock, EditableText, PositionedGlyph, Strikethrough, StrikethroughColor,
        TextBackgroundColor, TextColor, TextCursorStyle, TextLayoutInfo, Underline, UnderlineColor,
    },
    ui::{
        widget::{Text, TextScroll, TextShadow},
        CalculatedClip, ComputedNode, ComputedStackIndex, ComputedUiRenderTargetInfo,
        ComputedUiTargetCamera, Node, ResolvedBorderRadius, UiGlobalTransform,
    },
    ui_render::{stack_z_offsets, UiCameraMap},
};
use std::collections::{HashMap, HashSet};

#[derive(Default)]
struct TextRootDependencies {
    sections: HashSet<Entity>,
    images: HashSet<AssetId<Image>>,
    samples: HashSet<AtlasSample>,
    paints: HashSet<PaintId>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
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

struct GlyphRun {
    section: Entity,
    section_ordinal: u32,
    paint_order: u32,
    image: AssetId<Image>,
    glyphs: Vec<RetainedGlyph>,
    samples: HashSet<AtlasSample>,
}

struct TextNodePaint {
    id: PaintId,
    z_order: f32,
    paint_order: u32,
    transform: Affine2,
    color: LinearRgba,
    size: Vec2,
}

fn push_glyph_run(
    runs: &mut Vec<GlyphRun>,
    section: Entity,
    section_ordinal: u32,
    paint_order: u32,
    image: AssetId<Image>,
    glyph: RetainedGlyph,
    sample: AtlasSample,
) {
    if let Some(run) = runs.last_mut()
        && run.image == image
        && run.section == section
    {
        run.glyphs.push(glyph);
        run.samples.insert(sample);
    } else {
        runs.push(GlyphRun {
            section,
            section_ordinal,
            paint_order,
            image,
            glyphs: vec![glyph],
            samples: HashSet::from([sample]),
        });
    }
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
    focused: Option<Entity>,
}

impl RetainedTextDependencies {
    fn remove_root(&mut self, root: Entity) -> HashSet<PaintId> {
        let paints = self.detach_root(root);
        self.prune_unused();
        paints
    }

    fn detach_root(&mut self, root: Entity) -> HashSet<PaintId> {
        let Some(old) = self.roots.remove(&root) else {
            return HashSet::new();
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
        old.paints
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
        paints: HashSet<PaintId>,
        assets: &Assets<Image>,
    ) -> HashSet<PaintId> {
        let old_paints = self.detach_root(root);
        let mut dependencies = TextRootDependencies {
            paints,
            ..Default::default()
        };
        for section in sections {
            if dependencies.sections.insert(section) {
                self.section_roots.entry(section).or_default().insert(root);
            }
        }
        for image in images {
            if dependencies.images.insert(image) {
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
            if dependencies.samples.insert(sample) {
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
        old_paints
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
        let mut samples: Vec<_> = samples.into_iter().collect();
        samples.sort_unstable();
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
    Option<&'a TextShadow>,
    Has<EditableText>,
);

#[derive(SystemParam)]
pub(crate) struct RemovedTextInputs<'w, 's> {
    text: RemovedComponents<'w, 's, Text>,
    editable: RemovedComponents<'w, 's, EditableText>,
    layout: RemovedComponents<'w, 's, TextLayoutInfo>,
    computed_text: RemovedComponents<'w, 's, ComputedTextBlock>,
    clip: RemovedComponents<'w, 's, CalculatedClip>,
    scroll: RemovedComponents<'w, 's, TextScroll>,
    cursor: RemovedComponents<'w, 's, TextCursorStyle>,
    shadow: RemovedComponents<'w, 's, TextShadow>,
    color: RemovedComponents<'w, 's, TextColor>,
    background: RemovedComponents<'w, 's, TextBackgroundColor>,
    strikethrough: RemovedComponents<'w, 's, Strikethrough>,
    strikethrough_color: RemovedComponents<'w, 's, StrikethroughColor>,
    underline: RemovedComponents<'w, 's, Underline>,
    underline_color: RemovedComponents<'w, 's, UnderlineColor>,
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
                Or<(With<Text>, With<EditableText>)>,
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
                    Changed<TextShadow>,
                    Changed<EditableText>,
                    Changed<Node>,
                    Changed<ComputedUiRenderTargetInfo>,
                )>,
            ),
        >,
    >,
    all: Extract<Query<TextQueryItem<'static>, Or<(With<Text>, With<EditableText>)>>>,
    changed_sections: Extract<
        Query<
            Entity,
            Or<(
                Changed<TextColor>,
                Changed<TextBackgroundColor>,
                Changed<Strikethrough>,
                Changed<StrikethroughColor>,
                Changed<Underline>,
                Changed<UnderlineColor>,
            )>,
        >,
    >,
    section_styles: Extract<
        Query<(
            &TextColor,
            Option<&TextBackgroundColor>,
            Has<Strikethrough>,
            Option<&StrikethroughColor>,
            Has<Underline>,
            Option<&UnderlineColor>,
        )>,
    >,
    camera_map: Extract<UiCameraMap>,
    input_focus: Extract<Option<Res<InputFocus>>>,
    mut removed: Extract<RemovedTextInputs>,
) {
    let mut candidates: HashSet<_> = changed.iter().map(|item| item.0).collect();
    let focused = input_focus.as_ref().and_then(|focus| focus.get());
    if dependencies.focused != focused {
        candidates.extend(dependencies.focused);
        candidates.extend(focused);
        dependencies.focused = focused;
    }
    for section in &changed_sections {
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
        editable,
        layout,
        computed_text,
        clip,
        scroll,
        cursor,
        shadow,
        color,
        background,
        strikethrough,
        strikethrough_color,
        underline,
        underline_color,
        computed_node,
        node,
        stack,
        transform,
        visibility,
        camera,
    } = &mut *removed;
    for section in color
        .read()
        .chain(background.read())
        .chain(strikethrough.read())
        .chain(strikethrough_color.read())
        .chain(underline.read())
        .chain(underline_color.read())
    {
        if let Some(roots) = dependencies.section_roots.get(&section) {
            candidates.extend(roots);
        }
    }

    let mut surfaces = state.lock();
    for root in text
        .read()
        .chain(editable.read())
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
    candidates.extend(shadow.read());

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
            shadow,
            editable,
        )) = all.get(root)
        else {
            remove_text_root(&mut dependencies, &mut surfaces, &mut commands, root);
            continue;
        };

        let transform = global_transform.affine()
            * Affine2::from_translation(
                node.content_box().min - scroll.map_or(Vec2::ZERO, |scroll| scroll.0),
            );
        let shadow_transform = shadow.map(|shadow| {
            global_transform.affine()
                * Affine2::from_translation(
                    node.content_box().min + shadow.offset / node.inverse_scale_factor()
                        - scroll.map_or(Vec2::ZERO, |scroll| scroll.0),
                )
        });
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
        let mut runs = Vec::new();
        let mut shadow_runs = Vec::new();
        let mut section_ordinals = HashMap::<Entity, u32>::new();
        let mut current_source_run = None;
        let shadow_color = shadow.map(|shadow| LinearRgba::from(shadow.color));

        for (
            glyph_index,
            PositionedGlyph {
                position,
                atlas_info,
                section_index: glyph_section,
                ..
            },
        ) in layout.glyphs.iter().enumerate()
        {
            if section_index != *glyph_section {
                if let Some(section) = computed_block.entities().get(*glyph_section) {
                    color = section_styles
                        .get(section.entity)
                        .map(|style| LinearRgba::from(style.0 .0))
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
            let section = computed_block
                .entities()
                .get(*glyph_section)
                .map_or(root, |section| section.entity);
            let source_key = (section, atlas_info.texture);
            let section_ordinal = if let Some((current_key, ordinal)) = current_source_run
                && current_key == source_key
            {
                ordinal
            } else {
                let next = section_ordinals.entry(section).or_default();
                let ordinal = *next;
                *next = next
                    .checked_add(1)
                    .expect("text section glyph run count exceeds u32");
                current_source_run = Some((source_key, ordinal));
                ordinal
            };
            let paint_order = u32::try_from(glyph_index).expect("text glyph count exceeds u32");
            let sample = AtlasSample::new(atlas_info.texture, atlas_info.rect);
            if !glyph_color.is_fully_transparent() {
                push_glyph_run(
                    &mut runs,
                    section,
                    section_ordinal,
                    paint_order,
                    atlas_info.texture,
                    RetainedGlyph::new(glyph_color, *position, atlas_info.rect),
                    sample,
                );
            }
            if let Some(shadow_color) = shadow_color.filter(|color| !color.is_fully_transparent()) {
                push_glyph_run(
                    &mut shadow_runs,
                    section,
                    section_ordinal,
                    paint_order,
                    atlas_info.texture,
                    RetainedGlyph::new(shadow_color, *position, atlas_info.rect),
                    sample,
                );
            }
        }

        let mut node_paints = Vec::new();
        let mut decoration_ordinals = HashMap::<Entity, u32>::new();
        for (run_index, run) in layout.run_geometry.iter().enumerate() {
            let Some(section) = computed_block.entities().get(run.section_index) else {
                continue;
            };
            let Ok((
                text_color,
                background,
                has_strikethrough,
                strikethrough_color,
                has_underline,
                underline_color,
            )) = section_styles.get(section.entity)
            else {
                continue;
            };
            let paint_order = u32::try_from(run_index).expect("text run count exceeds u32");
            let next_ordinal = decoration_ordinals.entry(section.entity).or_default();
            let run_ordinal = *next_ordinal;
            *next_ordinal = run_ordinal
                .checked_add(1)
                .expect("text section decoration count exceeds u32");

            if let Some(background) = background
                && !background.0.is_fully_transparent()
            {
                node_paints.push(TextNodePaint {
                    id: PaintId {
                        entity: section.entity,
                        family: PaintFamily::TextBackground,
                        ordinal: run_ordinal,
                    },
                    z_order: stack.0 as f32 + stack_z_offsets::TEXT,
                    paint_order,
                    transform: transform * Affine2::from_translation(run.bounds.center()),
                    color: background.0.to_linear(),
                    size: run.bounds.size(),
                });
            }

            let strikethrough_color = strikethrough_color
                .map_or(text_color.0, |color| color.0)
                .to_linear();
            if has_strikethrough && !strikethrough_color.is_fully_transparent() {
                node_paints.push(TextNodePaint {
                    id: PaintId {
                        entity: section.entity,
                        family: PaintFamily::TextDecoration,
                        ordinal: decoration_ordinal(run_ordinal, 0),
                    },
                    z_order: stack.0 as f32 + stack_z_offsets::TEXT_STRIKETHROUGH,
                    paint_order: decoration_ordinal(paint_order, 0),
                    transform: transform * Affine2::from_translation(run.strikethrough_position()),
                    color: strikethrough_color,
                    size: run.strikethrough_size(),
                });
            }

            let underline_color = underline_color
                .map_or(text_color.0, |color| color.0)
                .to_linear();
            if has_underline && !underline_color.is_fully_transparent() {
                node_paints.push(TextNodePaint {
                    id: PaintId {
                        entity: section.entity,
                        family: PaintFamily::TextDecoration,
                        ordinal: decoration_ordinal(run_ordinal, 1),
                    },
                    z_order: stack.0 as f32 + stack_z_offsets::TEXT_STRIKETHROUGH,
                    paint_order: decoration_ordinal(paint_order, 1),
                    transform: transform * Affine2::from_translation(run.underline_position()),
                    color: underline_color,
                    size: run.underline_size(),
                });
            }

            if let (Some(shadow_transform), Some(shadow_color)) = (shadow_transform, shadow_color)
                && !shadow_color.is_fully_transparent()
            {
                if has_strikethrough {
                    node_paints.push(TextNodePaint {
                        id: PaintId {
                            entity: section.entity,
                            family: PaintFamily::TextShadowDecoration,
                            ordinal: decoration_ordinal(run_ordinal, 0),
                        },
                        z_order: stack.0 as f32 + stack_z_offsets::TEXT,
                        paint_order: decoration_ordinal(paint_order, 0),
                        transform: shadow_transform
                            * Affine2::from_translation(run.strikethrough_position()),
                        color: shadow_color,
                        size: run.strikethrough_size(),
                    });
                }
                if has_underline {
                    node_paints.push(TextNodePaint {
                        id: PaintId {
                            entity: section.entity,
                            family: PaintFamily::TextShadowDecoration,
                            ordinal: decoration_ordinal(run_ordinal, 1),
                        },
                        z_order: stack.0 as f32 + stack_z_offsets::TEXT,
                        paint_order: decoration_ordinal(paint_order, 1),
                        transform: shadow_transform
                            * Affine2::from_translation(run.underline_position()),
                        color: shadow_color,
                        size: run.underline_size(),
                    });
                }
            }
        }

        if let Some(cursor_style) = cursor_style {
            let selection_color = if focused == Some(root) {
                cursor_style.selection_color
            } else {
                cursor_style.unfocused_selection_color
            };
            if !selection_color.is_fully_transparent() {
                for (index, selection) in layout.selection_rects.iter().enumerate() {
                    node_paints.push(TextNodePaint {
                        id: PaintId {
                            entity: root,
                            family: PaintFamily::TextSelection,
                            ordinal: u32::try_from(index)
                                .expect("text selection count exceeds u32"),
                        },
                        z_order: stack.0 as f32 + stack_z_offsets::TEXT_SELECTION,
                        paint_order: u32::try_from(index)
                            .expect("text selection count exceeds u32"),
                        transform: transform * Affine2::from_translation(selection.center()),
                        color: selection_color.to_linear(),
                        size: selection.size(),
                    });
                }
            }
            if let Some((true, cursor)) = layout.cursor
                && !cursor.is_empty()
                && !cursor_style.color.is_fully_transparent()
            {
                node_paints.push(TextNodePaint {
                    id: PaintId {
                        entity: root,
                        family: PaintFamily::TextCursor,
                        ordinal: 0,
                    },
                    z_order: stack.0 as f32 + stack_z_offsets::TEXT_CURSOR,
                    paint_order: 0,
                    transform: transform * Affine2::from_translation(cursor.center()),
                    color: cursor_style.color.to_linear(),
                    size: cursor.size(),
                });
            }
        }

        if editable && !text_color.0.is_fully_transparent() {
            for (index, rect) in layout.preedit_underline_rects.iter().enumerate() {
                node_paints.push(TextNodePaint {
                    id: PaintId {
                        entity: root,
                        family: PaintFamily::TextPreedit,
                        ordinal: u32::try_from(index).expect("preedit underline count exceeds u32"),
                    },
                    z_order: stack.0 as f32 + stack_z_offsets::TEXT_STRIKETHROUGH,
                    paint_order: u32::try_from(index).expect("preedit underline count exceeds u32"),
                    transform: transform * Affine2::from_translation(rect.center()),
                    color: text_color.0.to_linear(),
                    size: rect.size(),
                });
            }
        }

        let image_ids: Vec<_> = runs
            .iter()
            .chain(&shadow_runs)
            .map(|run| run.image)
            .collect();
        let atlas_samples: Vec<_> = runs
            .iter()
            .chain(&shadow_runs)
            .flat_map(|run| run.samples.iter().copied())
            .collect();
        let paint_ids: HashSet<_> = shadow_runs
            .iter()
            .map(|run| glyph_run_id(run, PaintFamily::TextShadow))
            .chain(runs.iter().map(|run| glyph_run_id(run, PaintFamily::Text)))
            .chain(node_paints.iter().map(|paint| paint.id))
            .collect();
        let old_paints = dependencies.set_root(
            root,
            computed_block
                .entities()
                .iter()
                .map(|section| section.entity),
            image_ids.iter().copied(),
            atlas_samples,
            paint_ids.clone(),
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

        for id in old_paints {
            if !paint_ids.contains(&id) {
                surfaces.remove(&mut commands, id);
            }
        }
        let Some(camera) = camera.filter(|_| visible) else {
            for id in paint_ids {
                surfaces.remove(&mut commands, id);
            }
            continue;
        };

        if shadow.is_some() {
            upsert_glyph_runs(
                &mut surfaces,
                &mut commands,
                &dependencies,
                root,
                camera,
                stack.0 as f32 + stack_z_offsets::TEXT,
                clip,
                shadow_transform.expect("a text shadow must have a shadow transform"),
                PaintFamily::TextShadow,
                shadow_runs,
            );
        }
        upsert_glyph_runs(
            &mut surfaces,
            &mut commands,
            &dependencies,
            root,
            camera,
            stack.0 as f32 + stack_z_offsets::TEXT,
            clip,
            transform,
            PaintFamily::Text,
            runs,
        );
        for paint in node_paints {
            upsert_text_node(&mut surfaces, &mut commands, root, camera, clip, paint);
        }
    }
}

fn upsert_text_node(
    surfaces: &mut RetainedUiSurfaces,
    commands: &mut Commands,
    root: Entity,
    camera: Entity,
    clip: Option<Rect>,
    paint: TextNodePaint,
) {
    let paint_coverage = coverage(paint.size, paint.transform, clip)
        .into_iter()
        .collect();
    surfaces.upsert(
        commands,
        paint.id,
        camera,
        RetainedDraw {
            render_entity: Entity::PLACEHOLDER,
            camera,
            main_entity: MainEntity::from(root),
            z_order: paint.z_order,
            paint_order: paint.paint_order,
            clip,
            image: AssetId::<Image>::default(),
            transform: paint.transform,
            item: RetainedDrawItem::Node(crate::scene::RetainedNodeItem {
                color: paint.color,
                rect: Rect {
                    min: Vec2::ZERO,
                    max: paint.size,
                },
                atlas_scaling: None,
                flip_x: false,
                flip_y: false,
                border: BorderRect::ZERO,
                border_radius: ResolvedBorderRadius::ZERO,
                node_type: bevy::ui_render::NodeType::Rect,
            }),
        },
        ResourceFingerprint::None,
        paint_coverage,
    );
}

fn decoration_ordinal(run: u32, kind: u32) -> u32 {
    run.checked_mul(2)
        .and_then(|ordinal| ordinal.checked_add(kind))
        .expect("text decoration ordinal exceeds u32")
}

#[expect(
    clippy::too_many_arguments,
    reason = "a retained glyph run needs the complete canonical draw command"
)]
fn upsert_glyph_runs(
    surfaces: &mut RetainedUiSurfaces,
    commands: &mut Commands,
    dependencies: &RetainedTextDependencies,
    root: Entity,
    camera: Entity,
    z_order: f32,
    clip: Option<Rect>,
    transform: Affine2,
    family: PaintFamily,
    runs: Vec<GlyphRun>,
) {
    for run in runs {
        let id = glyph_run_id(&run, family);
        let glyph_coverage = run
            .glyphs
            .iter()
            .filter_map(|glyph| {
                coverage(
                    glyph.rect().size(),
                    transform * Affine2::from_translation(glyph.translation()),
                    clip,
                )
            })
            .collect();
        let revisions = dependencies.revisions(run.image, run.samples);
        surfaces.upsert(
            commands,
            id,
            camera,
            RetainedDraw {
                render_entity: Entity::PLACEHOLDER,
                camera,
                main_entity: MainEntity::from(root),
                z_order,
                paint_order: run.paint_order,
                clip,
                image: run.image,
                transform,
                item: RetainedDrawItem::Glyphs(run.glyphs.into_boxed_slice()),
            },
            ResourceFingerprint::Revisions(revisions),
            glyph_coverage,
        );
    }
}

fn remove_text_root(
    dependencies: &mut RetainedTextDependencies,
    surfaces: &mut RetainedUiSurfaces,
    commands: &mut Commands,
    root: Entity,
) {
    for id in dependencies.remove_root(root) {
        surfaces.remove(commands, id);
    }
}

fn glyph_run_id(run: &GlyphRun, family: PaintFamily) -> PaintId {
    PaintId {
        entity: run.section,
        family,
        ordinal: run.section_ordinal,
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
