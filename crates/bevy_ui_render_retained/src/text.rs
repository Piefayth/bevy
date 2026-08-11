//! Change-driven retained extraction for ordinary UI text glyphs.

use crate::sampled_image::{ImageReader, ImageSample, RetainedSampledImages};
use crate::scene::{
    coverage, PaintFamily, PaintId, ResourceFingerprint, RetainedDraw, RetainedDrawItem,
    RetainedGlyph, RetainedUiScene, RetainedUiSurfaces,
};
use bevy::{
    asset::{AssetId, Assets},
    camera::visibility::InheritedVisibility,
    color::{Alpha, LinearRgba},
    ecs::{
        entity::Entity,
        lifecycle::RemovedComponents,
        query::{Changed, Has, Or, With},
        system::{Commands, Query, Res, ResMut, SystemParam},
    },
    image::Image,
    input_focus::InputFocus,
    math::{Affine2, Rect, Vec2},
    render::{render_resource::DefaultImageSamplerDescriptor, sync_world::MainEntity, Extract},
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
    paints: HashSet<PaintId>,
}

struct GlyphRun {
    section: Entity,
    section_ordinal: u32,
    paint_order: u32,
    image: AssetId<Image>,
    glyphs: Vec<RetainedGlyph>,
    samples: HashSet<ImageSample>,
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
    sample: ImageSample,
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
    focused: Option<Entity>,
}

impl RetainedTextDependencies {
    fn remove_root(&mut self, root: Entity) -> HashSet<PaintId> {
        self.detach_root(root)
    }

    fn detach_root(&mut self, root: Entity) -> HashSet<PaintId> {
        let Some(old) = self.roots.remove(&root) else {
            return HashSet::new();
        };
        for section in old.sections {
            remove_reader(&mut self.section_roots, section, root);
        }
        old.paints
    }

    fn set_root(
        &mut self,
        root: Entity,
        sections: impl IntoIterator<Item = Entity>,
        paints: HashSet<PaintId>,
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
        self.roots.insert(root, dependencies);
        old_paints
    }
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
    mut sampled_images: ResMut<RetainedSampledImages>,
    default_sampler: Res<DefaultImageSamplerDescriptor>,
    images: Extract<Res<Assets<Image>>>,
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
    candidates.extend(sampled_images.take_text());
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
        remove_text_root(
            &mut dependencies,
            &mut sampled_images,
            &mut surfaces,
            &mut commands,
            root,
        );
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
            remove_text_root(
                &mut dependencies,
                &mut sampled_images,
                &mut surfaces,
                &mut commands,
                root,
            );
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
            let sample = ImageSample::rect(atlas_info.texture, atlas_info.rect);
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
            paint_ids.clone(),
        );
        sampled_images.replace_reader(
            ImageReader::Text(root),
            atlas_samples,
            &images,
            &default_sampler,
        );
        for &image in &image_ids {
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
    sampled_images: &RetainedSampledImages,
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
                item: RetainedDrawItem::Glyphs(run.glyphs.into_boxed_slice()),
            },
            ResourceFingerprint::Revisions(revisions),
            glyph_coverage,
        );
    }
}

fn remove_text_root(
    dependencies: &mut RetainedTextDependencies,
    sampled_images: &mut RetainedSampledImages,
    surfaces: &mut RetainedUiSurfaces,
    commands: &mut Commands,
    root: Entity,
) {
    sampled_images.remove_reader(ImageReader::Text(root));
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
