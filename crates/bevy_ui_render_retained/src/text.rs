//! Change-driven retained extraction for ordinary UI text glyphs.

use crate::{
    boundary::retained_clip,
    sampled_image::{ImageReader, ImageSample, RetainedSampledImages, SampledImageState},
    scene::{
        coverage, PaintFamily, PaintId, ResourceFingerprint, RetainedDraw, RetainedDrawItem,
        RetainedGlyph, RetainedUiScene, RetainedUiSurfaces,
    },
};
use bevy::{
    app::Inherited,
    asset::{AssetId, Assets},
    camera::visibility::InheritedVisibility,
    color::{Alpha, LinearRgba},
    ecs::{
        entity::{Entity, EntityHashMap},
        lifecycle::RemovedComponents,
        query::{Changed, Has, Or, With},
        system::{Commands, ParamSet, Query, Res, ResMut, SystemParam},
    },
    image::Image,
    input_focus::InputFocus,
    math::{Affine2, Rect, Vec2},
    platform::collections::{HashMap, HashSet},
    render::{render_resource::DefaultImageSamplerDescriptor, sync_world::MainEntity, Extract},
    sprite::BorderRect,
    text::{
        ComputedTextBlock, EditableText, PositionedGlyph, Strikethrough, StrikethroughColor,
        TextBackgroundColor, TextColor, TextCursorStyle, TextLayoutInfo, Underline, UnderlineColor,
    },
    ui::{
        widget::{Text, TextScroll, TextShadow},
        CalculatedClip, ComputedNode, ComputedStackIndex, ComputedUiPaintTarget,
        ComputedUiRenderTargetInfo, ComputedUiTargetCamera, Node, ResolvedBorderRadius,
        UiGlobalTransform,
    },
    ui_render::{stack_z_offsets, UiCameraMap},
};
use smallvec::{smallvec, SmallVec};

#[derive(Default)]
struct TextRootDependencies {
    sections: SmallVec<[Entity; 2]>,
    paints: SmallVec<[PaintId; 4]>,
    text_paints: SmallVec<[PaintId; 1]>,
    layout: TextLayoutPaintFingerprint,
    placement: TextPlacementFingerprint,
    camera: Option<Entity>,
    fast_color: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct TextPlacementFingerprint {
    transform: [u32; 6],
    clip: Option<[u32; 4]>,
}

impl TextPlacementFingerprint {
    fn new(transform: Affine2, clip: Option<Rect>) -> Self {
        Self {
            transform: transform.to_cols_array().map(f32::to_bits),
            clip: clip.map(rect_bits),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct TextLayoutPaintFingerprint {
    glyphs: Box<[GlyphPaintFingerprint]>,
    runs: Box<[RunPaintFingerprint]>,
    cursor: Option<(bool, [u32; 4])>,
    selection: Box<[[u32; 4]]>,
    preedit: Box<[[u32; 4]]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GlyphPaintFingerprint {
    position: [u32; 2],
    texture: AssetId<Image>,
    rect: [u32; 4],
    section: usize,
    alpha_mask: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RunPaintFingerprint {
    section: usize,
    bounds: [u32; 4],
    strikethrough_y: u32,
    strikethrough_thickness: u32,
    underline_y: u32,
    underline_thickness: u32,
}

impl TextLayoutPaintFingerprint {
    fn new(layout: &TextLayoutInfo) -> Self {
        Self {
            glyphs: layout
                .glyphs
                .iter()
                .map(|glyph| GlyphPaintFingerprint {
                    position: vec2_bits(glyph.position),
                    texture: glyph.atlas_info.texture,
                    rect: rect_bits(glyph.atlas_info.rect),
                    section: glyph.section_index,
                    alpha_mask: glyph.atlas_info.is_alpha_mask,
                })
                .collect(),
            runs: layout
                .run_geometry
                .iter()
                .map(|run| RunPaintFingerprint {
                    section: run.section_index,
                    bounds: rect_bits(run.bounds),
                    strikethrough_y: run.strikethrough_y.to_bits(),
                    strikethrough_thickness: run.strikethrough_thickness.to_bits(),
                    underline_y: run.underline_y.to_bits(),
                    underline_thickness: run.underline_thickness.to_bits(),
                })
                .collect(),
            cursor: layout
                .cursor
                .map(|(visible, rect)| (visible, rect_bits(rect))),
            selection: layout
                .selection_rects
                .iter()
                .copied()
                .map(rect_bits)
                .collect(),
            preedit: layout
                .preedit_underline_rects
                .iter()
                .copied()
                .map(rect_bits)
                .collect(),
        }
    }
}

fn vec2_bits(value: Vec2) -> [u32; 2] {
    value.to_array().map(f32::to_bits)
}

fn rect_bits(value: Rect) -> [u32; 4] {
    [
        value.min.x.to_bits(),
        value.min.y.to_bits(),
        value.max.x.to_bits(),
        value.max.y.to_bits(),
    ]
}

struct GlyphRun {
    section: Entity,
    section_ordinal: u32,
    paint_order: u32,
    image: AssetId<Image>,
    glyphs: SmallVec<[RetainedGlyph; 4]>,
    samples: SmallVec<[ImageSample; 4]>,
}

struct TextNodePaint {
    id: PaintId,
    z_order: f32,
    paint_order: u32,
    transform: Affine2,
    layout_translation: Vec2,
    local_translation: Vec2,
    color: LinearRgba,
    size: Vec2,
}

fn push_glyph_run(
    runs: &mut SmallVec<[GlyphRun; 1]>,
    section: Entity,
    section_ordinal: u32,
    paint_order: u32,
    image: AssetId<Image>,
    glyph: RetainedGlyph,
    sample: ImageSample,
) {
    if let Some(run) = runs.last_mut()
        && run.image == image
        && run.section == section
    {
        run.glyphs.push(glyph);
        if !run.samples.contains(&sample) {
            run.samples.push(sample);
        }
    } else {
        runs.push(GlyphRun {
            section,
            section_ordinal,
            paint_order,
            image,
            glyphs: smallvec![glyph],
            samples: smallvec![sample],
        });
    }
}

#[derive(bevy::prelude::Resource, Default)]
pub(crate) struct RetainedTextDependencies {
    roots: EntityHashMap<TextRootDependencies>,
    section_roots: EntityHashMap<HashSet<Entity>>,
    focused: Option<Entity>,
}

impl RetainedTextDependencies {
    fn remove_root(&mut self, root: Entity) -> SmallVec<[PaintId; 4]> {
        self.detach_root(root)
    }

    fn detach_root(&mut self, root: Entity) -> SmallVec<[PaintId; 4]> {
        let Some(old) = self.roots.remove(&root) else {
            return SmallVec::new();
        };
        for section in old.sections {
            remove_reader(&mut self.section_roots, section, root);
        }
        old.paints
    }

    fn set_root(
        &mut self,
        root: Entity,
        sections: SmallVec<[Entity; 2]>,
        paints: SmallVec<[PaintId; 4]>,
        layout: TextLayoutPaintFingerprint,
        placement: TextPlacementFingerprint,
        camera: Option<Entity>,
        fast_color: bool,
    ) -> SmallVec<[PaintId; 4]> {
        if let Some(existing) = self.roots.get_mut(&root)
            && existing.sections == sections
            && existing.paints == paints
        {
            existing.layout = layout;
            existing.placement = placement;
            existing.camera = camera;
            existing.fast_color = fast_color;
            return SmallVec::new();
        }
        let old_paints = self.detach_root(root);
        let dependencies = TextRootDependencies {
            sections,
            layout,
            placement,
            text_paints: paints
                .iter()
                .filter(|id| id.family == PaintFamily::Text)
                .copied()
                .collect(),
            paints,
            camera,
            fast_color,
        };
        for &section in &dependencies.sections {
            self.section_roots.entry(section).or_default().insert(root);
        }
        self.roots.insert(root, dependencies);
        old_paints
    }
}

fn remove_reader(readers: &mut EntityHashMap<HashSet<Entity>>, key: Entity, entity: Entity) {
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

type TextGeometryItem<'a> = (
    Entity,
    &'a Node,
    &'a ComputedNode,
    &'a UiGlobalTransform,
    &'a InheritedVisibility,
    Option<&'a CalculatedClip>,
    &'a ComputedUiTargetCamera,
    Option<&'a TextScroll>,
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
    sampled_images: Res<RetainedSampledImages>,
    default_sampler: Res<DefaultImageSamplerDescriptor>,
    images: Extract<Res<Assets<Image>>>,
    changed: Extract<
        Query<
            TextQueryItem<'static>,
            (
                Or<(With<Text>, With<EditableText>)>,
                Or<(
                    Changed<ComputedStackIndex>,
                    Changed<InheritedVisibility>,
                    Changed<CalculatedClip>,
                    Changed<ComputedUiTargetCamera>,
                    Changed<TextScroll>,
                    Changed<TextCursorStyle>,
                    Changed<TextShadow>,
                    Changed<EditableText>,
                    Changed<ComputedUiRenderTargetInfo>,
                )>,
            ),
        >,
    >,
    mut derived_changes: Extract<
        ParamSet<(
            Query<
                TextGeometryItem<'static>,
                (Or<(With<Text>, With<EditableText>)>, Changed<ComputedNode>),
            >,
            Query<
                (Entity, &'static ComputedTextBlock),
                (
                    Or<(With<Text>, With<EditableText>)>,
                    Changed<ComputedTextBlock>,
                ),
            >,
            Query<
                (Entity, &'static TextLayoutInfo),
                (
                    Or<(With<Text>, With<EditableText>)>,
                    Changed<TextLayoutInfo>,
                ),
            >,
        )>,
    >,
    all: Extract<Query<TextQueryItem<'static>, Or<(With<Text>, With<EditableText>)>>>,
    changed_sections: Extract<
        Query<
            Entity,
            Or<(
                Changed<TextBackgroundColor>,
                Changed<Strikethrough>,
                Changed<StrikethroughColor>,
                Changed<Underline>,
                Changed<UnderlineColor>,
            )>,
        >,
    >,
    changed_colors: Extract<Query<(Entity, &TextColor), Changed<TextColor>>>,
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
    owners: Extract<Query<&'static Inherited<ComputedUiPaintTarget>>>,
    input_focus: Extract<Option<Res<InputFocus>>>,
    mut removed: Extract<RemovedTextInputs>,
) {
    let mut sampled_images = sampled_images.lock();
    let mut removed_paints = Vec::new();
    let mut retints = Vec::new();
    let mut repositions = Vec::new();
    let mut extra_candidates = sampled_images.take_text();
    let focused = input_focus.as_ref().and_then(|focus| focus.get());
    if dependencies.focused != focused {
        extra_candidates.extend(dependencies.focused);
        extra_candidates.extend(focused);
        dependencies.focused = focused;
    }
    for section in &changed_sections {
        if let Some(roots) = dependencies.section_roots.get(&section) {
            extra_candidates.extend(roots);
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
            extra_candidates.extend(roots);
        }
    }

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
        sampled_images.remove_reader(ImageReader::Text(root));
        removed_paints.extend(dependencies.remove_root(root));
    }
    extra_candidates.extend(clip.read());
    extra_candidates.extend(scroll.read());
    extra_candidates.extend(cursor.read());
    extra_candidates.extend(shadow.read());
    extra_candidates.retain(|root| !changed.contains(*root));

    for (section, color) in &changed_colors {
        let color = color.0.to_linear();
        if let Some(root_dependencies) = dependencies.roots.get(&section)
            && !changed.contains(section)
            && !extra_candidates.contains(&section)
        {
            if root_dependencies.fast_color {
                if root_dependencies.camera.is_some() {
                    retints.push((root_dependencies.text_paints.clone(), section, color));
                }
            } else {
                extra_candidates.insert(section);
            }
        }
        if let Some(roots) = dependencies.section_roots.get(&section) {
            for &root in roots {
                if root == section || changed.contains(root) || extra_candidates.contains(&root) {
                    continue;
                }
                let Some(root_dependencies) = dependencies.roots.get(&root) else {
                    continue;
                };
                if root_dependencies.fast_color {
                    if root_dependencies.camera.is_some() {
                        retints.push((root_dependencies.text_paints.clone(), section, color));
                    }
                } else {
                    extra_candidates.insert(root);
                }
            }
        }
    }

    for (root, computed) in &derived_changes.p1() {
        if changed.contains(root) || extra_candidates.contains(&root) {
            continue;
        }
        let mut sections: SmallVec<[Entity; 2]> = core::iter::once(root)
            .chain(computed.entities().iter().map(|section| section.entity))
            .collect();
        sections.sort_unstable();
        sections.dedup();
        if dependencies
            .roots
            .get(&root)
            .is_none_or(|dependencies| dependencies.sections != sections)
        {
            extra_candidates.insert(root);
        }
    }

    for (root, layout) in &derived_changes.p2() {
        if changed.contains(root) || extra_candidates.contains(&root) {
            continue;
        }
        let fingerprint = TextLayoutPaintFingerprint::new(layout);
        if dependencies
            .roots
            .get(&root)
            .is_none_or(|dependencies| dependencies.layout != fingerprint)
        {
            extra_candidates.insert(root);
        }
    }

    let mut camera_mapper = camera_map.get_mapper();
    for (root, _, node, global_transform, visibility, clip, target_camera, scroll) in
        &derived_changes.p0()
    {
        if changed.contains(root) || extra_candidates.contains(&root) {
            continue;
        }
        let camera = camera_mapper.map(target_camera);
        let visible_camera = camera.filter(|_| visibility.get() && !node.is_empty());
        let Some(root_dependencies) = dependencies.roots.get(&root) else {
            extra_candidates.insert(root);
            continue;
        };
        if root_dependencies.camera != visible_camera {
            extra_candidates.insert(root);
            continue;
        }
        let content_translation =
            node.content_box().min - scroll.map_or(Vec2::ZERO, |scroll| scroll.0);
        let owner = owners.get(root).ok();
        let clip = resolved_text_clip(root, node, global_transform, clip, owner, scroll);
        let placement = TextPlacementFingerprint::new(
            global_transform.affine() * Affine2::from_translation(content_translation),
            clip,
        );
        if root_dependencies.placement == placement {
            continue;
        }
        repositions.push((
            root.into(),
            global_transform.affine(),
            clip,
            content_translation,
        ));
        dependencies
            .roots
            .get_mut(&root)
            .expect("checked above")
            .placement = placement;
    }
    if removed_paints.is_empty()
        && retints.is_empty()
        && repositions.is_empty()
        && changed.is_empty()
        && extra_candidates.is_empty()
    {
        return;
    }
    let mut surfaces = state.lock();
    for id in removed_paints {
        surfaces.remove(&mut commands, id);
    }
    for (paints, section, color) in retints {
        surfaces.retint_text(paints, section, color);
    }
    for (root, transform, clip, content_translation) in repositions {
        surfaces.reposition_layout(root, transform, clip, Some(content_translation));
    }
    for (
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
    ) in changed.iter().chain(
        extra_candidates
            .into_iter()
            .filter_map(|root| all.get(root).ok()),
    ) {
        let owner = owners.get(root).ok();
        let content_translation =
            node.content_box().min - scroll.map_or(Vec2::ZERO, |scroll| scroll.0);
        let transform = global_transform.affine() * Affine2::from_translation(content_translation);
        let shadow_offset = shadow.map(|shadow| shadow.offset / node.inverse_scale_factor());
        let shadow_translation = shadow_offset.map(|offset| content_translation + offset);
        let shadow_transform = shadow_translation
            .map(|translation| global_transform.affine() * Affine2::from_translation(translation));
        let clip = resolved_text_clip(root, node, global_transform, clip, owner, scroll);
        let camera = camera_mapper.map(target_camera);
        let visible = camera.is_some() && visibility.get() && !node.is_empty();
        let selected_text_color = cursor_style
            .and_then(|style| style.selected_text_color)
            .map(|color| color.to_linear());
        let mut color = text_color.0.to_linear();
        let mut section_index = 0;
        let mut runs = SmallVec::<[GlyphRun; 1]>::new();
        let mut shadow_runs = SmallVec::<[GlyphRun; 1]>::new();
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
            let selected = selected_text_color.is_some_and(|_| {
                layout.selection_rects.iter().any(|selection| {
                    let glyph = Rect::from_center_size(*position, atlas_info.rect.size());
                    selection.contains(glyph.min) && selection.contains(glyph.max)
                })
            });
            let glyph_color = if !atlas_info.is_alpha_mask {
                LinearRgba::WHITE
            } else if selected {
                selected_text_color.unwrap()
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
            let sample = ImageSample::rect(atlas_info.texture, atlas_info.rect);
            let atlas_extent = images.get(atlas_info.texture).map(Image::size_f32);
            if !glyph_color.is_fully_transparent() {
                push_glyph_run(
                    &mut runs,
                    section,
                    section_ordinal,
                    paint_order,
                    atlas_info.texture,
                    RetainedGlyph::new(
                        glyph_color,
                        *position,
                        atlas_info.rect,
                        atlas_extent,
                        atlas_info.is_alpha_mask && !selected,
                    ),
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
                    RetainedGlyph::new(
                        shadow_color,
                        *position,
                        atlas_info.rect,
                        atlas_extent,
                        false,
                    ),
                    sample,
                );
            }
        }

        let mut node_paints = SmallVec::<[TextNodePaint; 2]>::new();
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
                    layout_translation: content_translation,
                    local_translation: run.bounds.center(),
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
                    layout_translation: content_translation,
                    local_translation: run.strikethrough_position(),
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
                    layout_translation: content_translation,
                    local_translation: run.underline_position(),
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
                        layout_translation: content_translation,
                        local_translation: shadow_offset.unwrap() + run.strikethrough_position(),
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
                        layout_translation: content_translation,
                        local_translation: shadow_offset.unwrap() + run.underline_position(),
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
                        layout_translation: content_translation,
                        local_translation: selection.center(),
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
                    layout_translation: content_translation,
                    local_translation: cursor.center(),
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
                    layout_translation: content_translation,
                    local_translation: rect.center(),
                    color: text_color.0.to_linear(),
                    size: rect.size(),
                });
            }
        }

        let mut paint_ids: SmallVec<[PaintId; 4]> = shadow_runs
            .iter()
            .map(|run| glyph_run_id(run, PaintFamily::TextShadow))
            .chain(runs.iter().map(|run| glyph_run_id(run, PaintFamily::Text)))
            .chain(node_paints.iter().map(|paint| paint.id))
            .collect();
        paint_ids.sort_unstable();
        paint_ids.dedup();
        let mut sections: SmallVec<[Entity; 2]> = core::iter::once(root)
            .chain(
                computed_block
                    .entities()
                    .iter()
                    .map(|section| section.entity),
            )
            .collect();
        sections.sort_unstable();
        sections.dedup();
        let old_paints = dependencies.set_root(
            root,
            sections,
            paint_ids.clone(),
            TextLayoutPaintFingerprint::new(layout),
            TextPlacementFingerprint::new(transform, clip),
            camera.filter(|_| visible),
            !editable && node_paints.is_empty(),
        );
        sampled_images.replace_reader(
            ImageReader::Text(root),
            runs.iter()
                .chain(&shadow_runs)
                .flat_map(|run| run.samples.iter().copied()),
            &images,
            &default_sampler,
        );
        for image in runs.iter().chain(&shadow_runs).map(|run| run.image) {
            sampled_images.mark_pending(image, images.get(image));
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
                &sampled_images,
                root,
                camera,
                stack.0 as f32 + stack_z_offsets::TEXT,
                clip,
                shadow_transform.expect("a text shadow must have a shadow transform"),
                content_translation,
                shadow_offset.expect("a text shadow must have a local translation"),
                PaintFamily::TextShadow,
                shadow_runs,
            );
        }
        upsert_glyph_runs(
            &mut surfaces,
            &mut commands,
            &sampled_images,
            root,
            camera,
            stack.0 as f32 + stack_z_offsets::TEXT,
            clip,
            transform,
            content_translation,
            Vec2::ZERO,
            PaintFamily::Text,
            runs,
        );
        for paint in node_paints {
            upsert_text_node(&mut surfaces, &mut commands, root, camera, clip, paint);
        }
    }
}

fn resolved_text_clip(
    root: Entity,
    node: &ComputedNode,
    transform: &UiGlobalTransform,
    clip: Option<&CalculatedClip>,
    owner: Option<&Inherited<ComputedUiPaintTarget>>,
    scroll: Option<&TextScroll>,
) -> Option<Rect> {
    let clip = retained_clip(root, node, transform, clip, owner);
    if scroll.is_none() {
        return clip;
    }
    let content_box = node.content_box();
    let text_clip = Rect::from_center_size(
        transform.affine().translation + content_box.center(),
        content_box.size(),
    );
    Some(clip.map_or(text_clip, |clip| clip.intersect(text_clip)))
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
            layout_translation: paint.layout_translation,
            local_translation: paint.local_translation,
            item: RetainedDrawItem::Node(crate::scene::RetainedNodeItem {
                color: paint.color,
                rect: Rect {
                    min: Vec2::ZERO,
                    max: paint.size,
                },
                atlas_scaling: None,
                image_extent: None,
                flip_x: false,
                flip_y: false,
                border: BorderRect::ZERO,
                border_radius: ResolvedBorderRadius::ZERO,
                node_type: bevy::ui_render::NodeType::Rect,
            }),
        },
        ResourceFingerprint::None,
        paint_coverage,
        true,
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
    sampled_images: &SampledImageState,
    root: Entity,
    camera: Entity,
    z_order: f32,
    clip: Option<Rect>,
    transform: Affine2,
    layout_translation: Vec2,
    local_translation: Vec2,
    family: PaintFamily,
    runs: SmallVec<[GlyphRun; 1]>,
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
        let revisions = sampled_images.revisions(run.image, run.samples);
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
                layout_translation,
                local_translation,
                item: RetainedDrawItem::Glyphs(run.glyphs.into_vec().into_boxed_slice()),
            },
            ResourceFingerprint::Revisions(revisions),
            glyph_coverage,
            true,
        );
    }
}

fn glyph_run_id(run: &GlyphRun, family: PaintFamily) -> PaintId {
    PaintId {
        entity: run.section,
        family,
        ordinal: run.section_ordinal,
    }
}
