//! Retained UI paint records shared by every paint family.

use crate::{
    border::{edge_rect, EDGE_FLAGS},
    boundary::{retained_clip, RepaintBoundary},
    core::{
        prepare_glyph_instance, prepare_persistent_instance, GpuUiInstance, RetainedCoreRuns,
        RetainedCoreSource,
    },
    damage::SpatialIndex,
    gradient_render::RetainedGradientRuns,
    material::RetainedPendingMaterials,
    shadow_render::RetainedShadowRuns,
    FloatBits, PaintCoverage, PaintRecord, PhysicalRect, RepairPlan, RetainedPaint, WorkCounters,
};
use bevy::{
    app::Inherited,
    asset::{AssetId, UntypedAssetId},
    color::ColorToComponents,
    ecs::{
        entity::Entity,
        lifecycle::RemovedComponents,
        query::{Changed, With},
        system::{Commands, Query, Res, ResMut},
    },
    image::Image,
    math::{Affine2, Rect, Vec2},
    platform::collections::{hash_map::Entry, HashMap, HashSet},
    render::{
        sync_world::{MainEntity, RenderEntity},
        Extract,
    },
    sprite::{BorderRect, SliceScaleMode, SpriteImageMode},
    ui::{
        widget::TextScroll, CalculatedClip, ComputedNode, ComputedUiPaintTarget,
        ComputedUiTargetCamera, Node, ResolvedBorderRadius, UiGlobalTransform,
    },
    ui_render::{
        box_shadow::ResolvedBoxShadow,
        gradient::ResolvedGradient,
        ui_texture_slice_pipeline::{ExtractedUiTextureSlice, ExtractedUiTextureSlices},
        NodeType,
    },
};
use core::{
    any::TypeId,
    sync::atomic::{AtomicU64, Ordering},
};
use smallvec::SmallVec;
use std::sync::{Mutex, MutexGuard, PoisonError};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum PaintFamily {
    Boundary,
    BoxShadow,
    Background,
    Border,
    Image,
    Viewport,
    Material(TypeId),
    Gradient,
    BorderGradient,
    TextBackground,
    TextShadow,
    TextShadowDecoration,
    TextSelection,
    Text,
    TextDecoration,
    TextPreedit,
    TextCursor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct PaintId {
    pub(crate) entity: Entity,
    pub(crate) family: PaintFamily,
    pub(crate) ordinal: u32,
}

#[derive(Clone)]
pub(crate) struct RetainedDraw {
    pub(crate) render_entity: Entity,
    pub(crate) camera: Entity,
    pub(crate) main_entity: MainEntity,
    pub(crate) z_order: f32,
    pub(crate) paint_order: u32,
    pub(crate) clip: Option<Rect>,
    pub(crate) image: AssetId<Image>,
    pub(crate) transform: Affine2,
    /// Kept separately so placement never needs to invert a potentially singular transform.
    pub(crate) local_translation: Vec2,
    pub(crate) item: RetainedDrawItem,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ResourceFingerprint {
    None,
    Revisions(SmallVec<[u64; 4]>),
}

#[derive(Clone)]
pub(crate) enum RetainedDrawItem {
    Boundary(RetainedBoundaryItem),
    BoxShadow(RetainedBoxShadowItem),
    Gradient(RetainedGradientItem),
    Material(RetainedMaterialItem),
    Node(RetainedNodeItem),
    Glyphs(Box<[RetainedGlyph]>),
    TextureSlice(RetainedTextureSliceItem),
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct RetainedBoundaryItem {
    pub(crate) surface: Entity,
    size: [FloatBits; 2],
    opacity: FloatBits,
}

impl RetainedBoundaryItem {
    pub(crate) fn new(surface: Entity, size: Vec2, opacity: f32) -> Self {
        let opacity = if opacity.is_nan() {
            1.0
        } else {
            opacity.clamp(0.0, 1.0)
        };
        Self {
            surface,
            size: size.to_array().map(FloatBits::new),
            opacity: FloatBits::new(opacity),
        }
    }

    pub(crate) fn size(&self) -> Vec2 {
        Vec2::new(self.size[0].get(), self.size[1].get())
    }

    pub(crate) fn opacity(&self) -> f32 {
        self.opacity.get()
    }
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct RetainedMaterialItem {
    pub(crate) material_type: TypeId,
    pub(crate) material: UntypedAssetId,
    pub(crate) stack_index: u32,
    rect: [FloatBits; 4],
    border: [FloatBits; 4],
    border_radius: [FloatBits; 4],
    pub(crate) sampled_images: Box<[AssetId<Image>]>,
    target_coverage: bool,
}

impl RetainedMaterialItem {
    pub(crate) fn new<M: bevy::ui_render::ui_material::UiMaterial>(
        material: AssetId<M>,
        stack_index: u32,
        rect: Rect,
        border: BorderRect,
        border_radius: ResolvedBorderRadius,
        sampled_images: Box<[AssetId<Image>]>,
        target_coverage: bool,
    ) -> Self {
        Self {
            material_type: TypeId::of::<M>(),
            material: material.untyped(),
            stack_index,
            rect: rect_fingerprint(rect),
            border: [
                border.min_inset.x,
                border.min_inset.y,
                border.max_inset.x,
                border.max_inset.y,
            ]
            .map(FloatBits::new),
            border_radius: <[f32; 4]>::from(border_radius).map(FloatBits::new),
            sampled_images,
            target_coverage,
        }
    }

    pub(crate) fn rect(&self) -> Rect {
        rect_from_fingerprint(self.rect)
    }

    pub(crate) fn border(&self) -> BorderRect {
        BorderRect {
            min_inset: Vec2::new(self.border[0].get(), self.border[1].get()),
            max_inset: Vec2::new(self.border[2].get(), self.border[3].get()),
        }
    }

    pub(crate) fn border_radius(&self) -> [f32; 4] {
        self.border_radius.map(FloatBits::get)
    }
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct RetainedBoxShadowItem {
    stack_index: u32,
    samples: u32,
    bounds: [FloatBits; 2],
    color: [FloatBits; 4],
    radius: [FloatBits; 4],
    blur_radius: FloatBits,
    size: [FloatBits; 2],
}

impl RetainedBoxShadowItem {
    pub(crate) fn new(stack_index: u32, shadow: ResolvedBoxShadow, samples: u32) -> Self {
        Self {
            stack_index,
            samples,
            bounds: shadow.bounds.to_array().map(FloatBits::new),
            color: shadow.color.to_f32_array().map(FloatBits::new),
            radius: <[f32; 4]>::from(shadow.radius).map(FloatBits::new),
            blur_radius: FloatBits::new(shadow.blur_radius),
            size: shadow.size.to_array().map(FloatBits::new),
        }
    }

    pub(crate) fn bounds(&self) -> Vec2 {
        Vec2::new(self.bounds[0].get(), self.bounds[1].get())
    }

    pub(crate) fn color(&self) -> bevy::color::LinearRgba {
        bevy::color::LinearRgba::new(
            self.color[0].get(),
            self.color[1].get(),
            self.color[2].get(),
            self.color[3].get(),
        )
    }

    pub(crate) fn radius(&self) -> ResolvedBorderRadius {
        ResolvedBorderRadius {
            top_left: self.radius[0].get(),
            top_right: self.radius[1].get(),
            bottom_right: self.radius[2].get(),
            bottom_left: self.radius[3].get(),
        }
    }

    pub(crate) fn size(&self) -> Vec2 {
        Vec2::new(self.size[0].get(), self.size[1].get())
    }

    pub(crate) fn blur_radius(&self) -> f32 {
        self.blur_radius.get()
    }

    pub(crate) const fn samples(&self) -> u32 {
        self.samples
    }
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct RetainedGradientItem {
    stack_index: u32,
    rect: [FloatBits; 4],
    border_radius: [FloatBits; 4],
    border: [FloatBits; 4],
    resolved: RetainedResolvedGradient,
    color_space: bevy::ui::InterpolationColorSpace,
    stops: Box<[RetainedGradientStop]>,
    border_gradient: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RetainedResolvedGradient {
    Linear {
        angle: FloatBits,
    },
    Conic {
        center: [FloatBits; 2],
        start: FloatBits,
    },
    Radial {
        center: [FloatBits; 2],
        size: [FloatBits; 2],
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct RetainedGradientStop {
    color: [FloatBits; 4],
    position: FloatBits,
    hint: FloatBits,
}

impl RetainedGradientItem {
    #[expect(
        clippy::too_many_arguments,
        reason = "the constructor captures one complete canonical gradient command"
    )]
    pub(crate) fn new(
        stack_index: u32,
        rect: Rect,
        border_radius: ResolvedBorderRadius,
        border: BorderRect,
        resolved: ResolvedGradient,
        color_space: bevy::ui::InterpolationColorSpace,
        stops: &[(bevy::color::LinearRgba, f32, f32)],
        border_gradient: bool,
    ) -> Self {
        let resolved = match resolved {
            ResolvedGradient::Linear { angle } => RetainedResolvedGradient::Linear {
                angle: FloatBits::new(angle),
            },
            ResolvedGradient::Conic { center, start } => RetainedResolvedGradient::Conic {
                center: center.to_array().map(FloatBits::new),
                start: FloatBits::new(start),
            },
            ResolvedGradient::Radial { center, size } => RetainedResolvedGradient::Radial {
                center: center.to_array().map(FloatBits::new),
                size: size.to_array().map(FloatBits::new),
            },
        };
        Self {
            stack_index,
            rect: rect_fingerprint(rect),
            border_radius: <[f32; 4]>::from(border_radius).map(FloatBits::new),
            border: [
                border.min_inset.x,
                border.min_inset.y,
                border.max_inset.x,
                border.max_inset.y,
            ]
            .map(FloatBits::new),
            resolved,
            color_space,
            stops: stops
                .iter()
                .map(|(color, position, hint)| RetainedGradientStop {
                    color: color.to_f32_array().map(FloatBits::new),
                    position: FloatBits::new(*position),
                    hint: FloatBits::new(*hint),
                })
                .collect(),
            border_gradient,
        }
    }

    pub(crate) fn rect(&self) -> Rect {
        rect_from_fingerprint(self.rect)
    }

    pub(crate) fn border_radius(&self) -> ResolvedBorderRadius {
        ResolvedBorderRadius {
            top_left: self.border_radius[0].get(),
            top_right: self.border_radius[1].get(),
            bottom_right: self.border_radius[2].get(),
            bottom_left: self.border_radius[3].get(),
        }
    }

    pub(crate) fn border(&self) -> BorderRect {
        BorderRect {
            min_inset: Vec2::new(self.border[0].get(), self.border[1].get()),
            max_inset: Vec2::new(self.border[2].get(), self.border[3].get()),
        }
    }

    pub(crate) fn resolved(&self) -> ResolvedGradient {
        match self.resolved {
            RetainedResolvedGradient::Linear { angle } => {
                ResolvedGradient::Linear { angle: angle.get() }
            }
            RetainedResolvedGradient::Conic { center, start } => ResolvedGradient::Conic {
                center: Vec2::new(center[0].get(), center[1].get()),
                start: start.get(),
            },
            RetainedResolvedGradient::Radial { center, size } => ResolvedGradient::Radial {
                center: Vec2::new(center[0].get(), center[1].get()),
                size: Vec2::new(size[0].get(), size[1].get()),
            },
        }
    }

    pub(crate) fn node_type(&self) -> NodeType {
        if self.border_gradient {
            NodeType::Border(bevy::ui_render::shader_flags::BORDER_ALL)
        } else {
            NodeType::Rect
        }
    }

    pub(crate) const fn color_space(&self) -> bevy::ui::InterpolationColorSpace {
        self.color_space
    }

    pub(crate) fn stops(
        &self,
    ) -> impl ExactSizeIterator<Item = (bevy::color::LinearRgba, f32, f32)> + '_ {
        self.stops.iter().map(|stop| {
            (
                bevy::color::LinearRgba::new(
                    stop.color[0].get(),
                    stop.color[1].get(),
                    stop.color[2].get(),
                    stop.color[3].get(),
                ),
                stop.position.get(),
                stop.hint.get(),
            )
        })
    }
}

#[derive(Clone, Copy)]
pub(crate) struct RetainedNodeItem {
    pub(crate) color: bevy::color::LinearRgba,
    pub(crate) rect: Rect,
    pub(crate) atlas_scaling: Option<Vec2>,
    pub(crate) image_extent: Option<Vec2>,
    pub(crate) flip_x: bool,
    pub(crate) flip_y: bool,
    pub(crate) border: BorderRect,
    pub(crate) border_radius: ResolvedBorderRadius,
    pub(crate) node_type: NodeType,
}

#[derive(Clone)]
pub(crate) struct RetainedTextureSliceItem {
    pub(crate) stack_index: u32,
    pub(crate) rect: Rect,
    pub(crate) atlas_rect: Option<Rect>,
    pub(crate) color: bevy::color::LinearRgba,
    pub(crate) image_scale_mode: SpriteImageMode,
    pub(crate) flip_x: bool,
    pub(crate) flip_y: bool,
    pub(crate) inverse_scale_factor: f32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct RetainedGlyph {
    color: [FloatBits; 4],
    translation: [FloatBits; 2],
    rect: [FloatBits; 4],
    atlas_extent: Option<[FloatBits; 2]>,
    tinted: bool,
}

impl RetainedGlyph {
    pub(crate) fn new(
        color: bevy::color::LinearRgba,
        translation: Vec2,
        rect: Rect,
        atlas_extent: Option<Vec2>,
        tinted: bool,
    ) -> Self {
        Self {
            color: color.to_f32_array().map(FloatBits::new),
            translation: translation.to_array().map(FloatBits::new),
            rect: [rect.min.x, rect.min.y, rect.max.x, rect.max.y].map(FloatBits::new),
            atlas_extent: atlas_extent.map(|extent| extent.to_array().map(FloatBits::new)),
            tinted,
        }
    }

    fn retint(&mut self, color: bevy::color::LinearRgba) -> bool {
        if !self.tinted {
            return false;
        }
        let color = color.to_f32_array().map(FloatBits::new);
        if self.color == color {
            return false;
        }
        self.color = color;
        true
    }

    pub(crate) fn color(self) -> bevy::color::LinearRgba {
        bevy::color::LinearRgba::new(
            self.color[0].get(),
            self.color[1].get(),
            self.color[2].get(),
            self.color[3].get(),
        )
    }

    pub(crate) fn translation(self) -> Vec2 {
        Vec2::new(self.translation[0].get(), self.translation[1].get())
    }

    pub(crate) fn rect(self) -> Rect {
        Rect::from_corners(
            Vec2::new(self.rect[0].get(), self.rect[1].get()),
            Vec2::new(self.rect[2].get(), self.rect[3].get()),
        )
    }

    fn atlas_extent(self) -> Option<Vec2> {
        self.atlas_extent
            .map(|extent| Vec2::new(extent[0].get(), extent[1].get()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum TextureSliceModeFingerprint {
    Sliced {
        border: [FloatBits; 4],
        center: SliceScaleModeFingerprint,
        sides: SliceScaleModeFingerprint,
        max_corner_scale: FloatBits,
    },
    Tiled {
        tile_x: bool,
        tile_y: bool,
        stretch_value: FloatBits,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum SliceScaleModeFingerprint {
    Stretch,
    Tile { stretch_value: FloatBits },
}

struct RetainedRecord {
    resource: ResourceFingerprint,
    render_entity: Entity,
    prepared_core: Option<PreparedCore>,
    draw: RetainedDraw,
}

enum PreparedCore {
    One(GpuUiInstance),
    Many(Box<[GpuUiInstance]>),
}

impl PreparedCore {
    fn as_slice(&self) -> &[GpuUiInstance] {
        match self {
            Self::One(instance) => core::slice::from_ref(instance),
            Self::Many(instances) => instances,
        }
    }
}

impl PartialEq for RetainedRecord {
    fn eq(&self, other: &Self) -> bool {
        self.resource == other.resource
            && retained_common_eq(&self.draw, &other.draw)
            && retained_item_eq(&self.draw.item, &other.draw.item)
    }
}

impl RetainedRecord {
    fn new(draw: RetainedDraw, resource: ResourceFingerprint) -> Self {
        let prepared_core = prepare_core(&draw);
        Self {
            resource,
            render_entity: draw.render_entity,
            prepared_core,
            draw,
        }
    }

    fn reposition(
        &mut self,
        source_transform: Affine2,
        clip: Option<Rect>,
        previous_coverage: &PaintCoverage,
    ) -> Option<PaintCoverage> {
        let transform = source_transform * Affine2::from_translation(self.draw.local_translation);
        if affine_bits(self.draw.transform) == affine_bits(transform)
            && self.draw.clip.map(rect_fingerprint) == clip.map(rect_fingerprint)
        {
            return None;
        }
        self.draw.transform = transform;
        self.draw.clip = clip;
        self.prepared_core = prepare_core(&self.draw);
        Some(draw_coverage(&self.draw, previous_coverage))
    }

    fn border_parts(&self) -> Option<(&RetainedNodeItem, u32)> {
        let RetainedDrawItem::Node(node) = &self.draw.item else {
            return None;
        };
        let NodeType::Border(flags) = node.node_type else {
            return None;
        };
        Some((node, flags))
    }

    fn can_merge_border(&self, other: &Self) -> bool {
        self.resource == other.resource
            && retained_common_eq(&self.draw, &other.draw)
            && match (&self.draw.item, &other.draw.item) {
                (RetainedDrawItem::Node(left), RetainedDrawItem::Node(right)) => {
                    retained_node_merge_eq(left, right)
                }
                _ => false,
            }
    }

    fn retint_glyphs(&mut self, color: bevy::color::LinearRgba) -> bool {
        let RetainedDrawItem::Glyphs(glyphs) = &mut self.draw.item else {
            return false;
        };
        let mut changed = false;
        if let Some(PreparedCore::Many(instances)) = &mut self.prepared_core {
            for (glyph, instance) in glyphs.iter_mut().zip(instances) {
                if glyph.retint(color) {
                    instance.set_color(color);
                    changed = true;
                }
            }
        } else {
            for glyph in glyphs {
                changed |= glyph.retint(color);
            }
        }
        changed
    }

    fn retint_node(&mut self, color: bevy::color::LinearRgba) -> bool {
        let RetainedDrawItem::Node(node) = &mut self.draw.item else {
            return false;
        };
        if node.color == color {
            return false;
        }
        node.color = color;
        if let Some(PreparedCore::One(instance)) = &mut self.prepared_core {
            instance.set_color(color);
        }
        true
    }

    fn core_source(&self, id: PaintId) -> RetainedCoreSource<'_> {
        self.prepared_core
            .as_ref()
            .map_or(RetainedCoreSource::Deferred(id), |prepared| {
                RetainedCoreSource::Prepared(prepared.as_slice())
            })
    }
}

fn prepare_core(draw: &RetainedDraw) -> Option<PreparedCore> {
    match &draw.item {
        RetainedDrawItem::Node(_) => prepare_persistent_instance(draw).map(PreparedCore::One),
        RetainedDrawItem::Glyphs(glyphs) => glyphs
            .iter()
            .map(|&glyph| {
                glyph
                    .atlas_extent()
                    .map(|extent| prepare_glyph_instance(draw, glyph, extent))
            })
            .collect::<Option<Vec<_>>>()
            .map(|instances| PreparedCore::Many(instances.into_boxed_slice())),
        _ => None,
    }
}

fn draw_coverage(draw: &RetainedDraw, previous: &PaintCoverage) -> PaintCoverage {
    match &draw.item {
        RetainedDrawItem::Boundary(boundary) => {
            coverage(boundary.size(), draw.transform, draw.clip)
                .into_iter()
                .collect()
        }
        RetainedDrawItem::BoxShadow(shadow) => coverage(shadow.bounds(), draw.transform, draw.clip)
            .into_iter()
            .collect(),
        RetainedDrawItem::Gradient(gradient) => {
            if gradient.node_type() == NodeType::Rect {
                return coverage(gradient.rect().size(), draw.transform, draw.clip)
                    .into_iter()
                    .collect();
            }
            let border = gradient.border();
            let widths = [
                border.min_inset.x,
                border.min_inset.y,
                border.max_inset.x,
                border.max_inset.y,
            ];
            let radii: [f32; 4] = gradient.border_radius().into();
            widths
                .into_iter()
                .enumerate()
                .filter(|(_, width)| *width > 0.0)
                .filter_map(|(edge, width)| {
                    coverage_rect(
                        edge_rect(gradient.rect().size(), width, radii, edge),
                        draw.transform,
                        draw.clip,
                    )
                })
                .collect()
        }
        RetainedDrawItem::Material(material) => {
            if material.target_coverage {
                previous.clone()
            } else {
                coverage(material.rect().size(), draw.transform, draw.clip)
                    .into_iter()
                    .collect()
            }
        }
        RetainedDrawItem::Node(node) => {
            let NodeType::Border(flags) = node.node_type else {
                return coverage(node.rect.size(), draw.transform, draw.clip)
                    .into_iter()
                    .collect();
            };
            let widths = [
                node.border.min_inset.x,
                node.border.min_inset.y,
                node.border.max_inset.x,
                node.border.max_inset.y,
            ];
            let radii: [f32; 4] = node.border_radius.into();
            widths
                .into_iter()
                .enumerate()
                .filter(|(edge, width)| *width > 0.0 && flags & EDGE_FLAGS[*edge] != 0)
                .filter_map(|(edge, width)| {
                    coverage_rect(
                        edge_rect(node.rect.size(), width, radii, edge),
                        draw.transform,
                        draw.clip,
                    )
                })
                .collect()
        }
        RetainedDrawItem::Glyphs(glyphs) => glyphs
            .iter()
            .filter_map(|glyph| {
                coverage(
                    glyph.rect().size(),
                    draw.transform * Affine2::from_translation(glyph.translation()),
                    draw.clip,
                )
            })
            .collect(),
        RetainedDrawItem::TextureSlice(slice) => {
            coverage(slice.rect.size(), draw.transform, draw.clip)
                .into_iter()
                .collect()
        }
    }
}

fn affine_bits(transform: Affine2) -> [FloatBits; 6] {
    transform.to_cols_array().map(FloatBits::new)
}

fn retained_common_eq(left: &RetainedDraw, right: &RetainedDraw) -> bool {
    left.camera == right.camera
        && FloatBits::new(left.z_order) == FloatBits::new(right.z_order)
        && left.paint_order == right.paint_order
        && left.clip.map(rect_fingerprint) == right.clip.map(rect_fingerprint)
        && left.image == right.image
        && affine_bits(left.transform) == affine_bits(right.transform)
        && left.local_translation.to_array().map(FloatBits::new)
            == right.local_translation.to_array().map(FloatBits::new)
}

fn retained_node_merge_eq(left: &RetainedNodeItem, right: &RetainedNodeItem) -> bool {
    left.color.to_f32_array().map(FloatBits::new) == right.color.to_f32_array().map(FloatBits::new)
        && rect_fingerprint(left.rect) == rect_fingerprint(right.rect)
        && left
            .image_extent
            .map(|value| value.to_array().map(FloatBits::new))
            == right
                .image_extent
                .map(|value| value.to_array().map(FloatBits::new))
        && left
            .atlas_scaling
            .map(|value| value.to_array().map(FloatBits::new))
            == right
                .atlas_scaling
                .map(|value| value.to_array().map(FloatBits::new))
        && left.flip_x == right.flip_x
        && left.flip_y == right.flip_y
        && [
            left.border.min_inset.x,
            left.border.min_inset.y,
            left.border.max_inset.x,
            left.border.max_inset.y,
        ]
        .map(FloatBits::new)
            == [
                right.border.min_inset.x,
                right.border.min_inset.y,
                right.border.max_inset.x,
                right.border.max_inset.y,
            ]
            .map(FloatBits::new)
        && <[f32; 4]>::from(left.border_radius).map(FloatBits::new)
            == <[f32; 4]>::from(right.border_radius).map(FloatBits::new)
}

fn retained_item_eq(left: &RetainedDrawItem, right: &RetainedDrawItem) -> bool {
    match (left, right) {
        (RetainedDrawItem::Boundary(left), RetainedDrawItem::Boundary(right)) => left == right,
        (RetainedDrawItem::BoxShadow(left), RetainedDrawItem::BoxShadow(right)) => left == right,
        (RetainedDrawItem::Gradient(left), RetainedDrawItem::Gradient(right)) => left == right,
        (RetainedDrawItem::Material(left), RetainedDrawItem::Material(right)) => left == right,
        (RetainedDrawItem::Node(left), RetainedDrawItem::Node(right)) => {
            retained_node_merge_eq(left, right) && left.node_type == right.node_type
        }
        (RetainedDrawItem::Glyphs(left), RetainedDrawItem::Glyphs(right)) => left == right,
        (RetainedDrawItem::TextureSlice(left), RetainedDrawItem::TextureSlice(right)) => {
            rect_fingerprint(left.rect) == rect_fingerprint(right.rect)
                && left.atlas_rect.map(rect_fingerprint) == right.atlas_rect.map(rect_fingerprint)
                && left.color.to_f32_array().map(FloatBits::new)
                    == right.color.to_f32_array().map(FloatBits::new)
                && texture_slice_mode_fingerprint(&left.image_scale_mode)
                    == texture_slice_mode_fingerprint(&right.image_scale_mode)
                && left.flip_x == right.flip_x
                && left.flip_y == right.flip_y
                && FloatBits::new(left.inverse_scale_factor)
                    == FloatBits::new(right.inverse_scale_factor)
        }
        _ => false,
    }
}

fn rect_fingerprint(rect: Rect) -> [FloatBits; 4] {
    [rect.min.x, rect.min.y, rect.max.x, rect.max.y].map(FloatBits::new)
}

fn rect_from_fingerprint(rect: [FloatBits; 4]) -> Rect {
    Rect::from_corners(
        Vec2::new(rect[0].get(), rect[1].get()),
        Vec2::new(rect[2].get(), rect[3].get()),
    )
}

fn texture_slice_mode_fingerprint(mode: &SpriteImageMode) -> TextureSliceModeFingerprint {
    match mode {
        SpriteImageMode::Sliced(slicer) => TextureSliceModeFingerprint::Sliced {
            border: [
                slicer.border.min_inset.x,
                slicer.border.min_inset.y,
                slicer.border.max_inset.x,
                slicer.border.max_inset.y,
            ]
            .map(FloatBits::new),
            center: slice_scale_mode_fingerprint(slicer.center_scale_mode),
            sides: slice_scale_mode_fingerprint(slicer.sides_scale_mode),
            max_corner_scale: FloatBits::new(slicer.max_corner_scale),
        },
        SpriteImageMode::Tiled {
            tile_x,
            tile_y,
            stretch_value,
        } => TextureSliceModeFingerprint::Tiled {
            tile_x: *tile_x,
            tile_y: *tile_y,
            stretch_value: FloatBits::new(*stretch_value),
        },
        SpriteImageMode::Auto | SpriteImageMode::Scale(_) => {
            unreachable!("retained texture slices require sliced or tiled image mode")
        }
    }
}

fn slice_scale_mode_fingerprint(mode: SliceScaleMode) -> SliceScaleModeFingerprint {
    match mode {
        SliceScaleMode::Stretch => SliceScaleModeFingerprint::Stretch,
        SliceScaleMode::Tile { stretch_value } => SliceScaleModeFingerprint::Tile {
            stretch_value: FloatBits::new(stretch_value),
        },
    }
}

#[derive(Default)]
pub(crate) struct RetainedUiSurfaces {
    paint: HashMap<Entity, RetainedPaint<PaintId, RetainedRecord>>,
    owners: HashMap<PaintId, PaintOwner>,
    by_main_entity: HashMap<MainEntity, SmallVec<[PaintId; 8]>>,
    surface_by_main_entity: HashMap<MainEntity, Entity>,
    dirty_surfaces: HashSet<Entity>,
    touched_surfaces: HashSet<Entity>,
    propagated_epochs: HashMap<Entity, u64>,
    order: HashMap<Entity, PaintOrder>,
}

struct PaintOwner {
    camera: Entity,
    main_entity: Option<MainEntity>,
    render_entity: Entity,
    order: (FloatBits, u32),
    group: Option<usize>,
}

#[derive(Default)]
struct PaintOrder {
    ids: Vec<PaintId>,
    groups: Vec<core::ops::Range<usize>>,
    group_by_id: HashMap<PaintId, usize>,
    groups_by_entity: HashMap<Entity, SmallVec<[usize; 4]>>,
    spatial: Option<SpatialIndex<usize>>,
    bounds_dirty: HashSet<usize>,
    direct_groups: Vec<usize>,
    direct_epochs: Vec<Option<[u64; 2]>>,
    candidates: Vec<usize>,
    candidate_marks: Vec<u32>,
    candidate_generation: u32,
    order_dirty: bool,
    group_dirty: bool,
}

impl PaintOrder {
    fn note_direct(&mut self, group: Option<usize>, epoch: u64) {
        let Some(group) = group else {
            return;
        };
        match &mut self.direct_epochs[group] {
            Some([_, latest]) => {
                debug_assert!(*latest <= epoch);
                *latest = epoch;
            }
            slot @ None => {
                *slot = Some([epoch, epoch]);
                self.direct_groups.push(group);
            }
        }
    }

    fn acknowledge(&mut self, through_epoch: u64) {
        self.direct_groups.retain(|&group| {
            let [earliest, latest] = self.direct_epochs[group]
                .expect("direct groups have an outstanding epoch interval");
            if earliest > through_epoch {
                return true;
            }
            if latest > through_epoch {
                self.direct_epochs[group] = Some([latest, latest]);
                true
            } else {
                self.direct_epochs[group] = None;
                false
            }
        });
    }
}

impl RetainedUiSurfaces {
    fn set_entity_surface(&mut self, entity: MainEntity, surface: Option<Entity>, camera: Entity) {
        let target = surface.unwrap_or(camera);
        match surface {
            Some(surface) => {
                self.surface_by_main_entity.insert(entity, surface);
            }
            None => {
                self.surface_by_main_entity.remove(&entity);
            }
        }
        let Some(ids) = self.by_main_entity.get(&entity).cloned() else {
            return;
        };
        for id in ids {
            let old_camera = self.owners[&id].camera;
            if old_camera == target {
                continue;
            }
            let mut record = self
                .paint
                .get_mut(&old_camera)
                .expect("owned retained paint camera must exist")
                .take(&id)
                .expect("owned retained paint must exist");
            record.value.draw.camera = target;
            self.paint.entry(target).or_default().upsert(id, record);
            self.dirty_surfaces.extend([old_camera, target]);
            self.touched_surfaces.extend([old_camera, target]);
            let owner = self
                .owners
                .get_mut(&id)
                .expect("main-entity index must contain an owner");
            owner.camera = target;
            owner.group = None;
            for camera in [old_camera, target] {
                let order = self.order.entry(camera).or_default();
                order.order_dirty = true;
                order.spatial = None;
            }
        }
    }

    fn remove_main_entity(&mut self, commands: &mut Commands, entity: MainEntity) {
        let Some(ids) = self.by_main_entity.get(&entity).cloned() else {
            self.surface_by_main_entity.remove(&entity);
            return;
        };
        for id in ids {
            self.remove(commands, id);
        }
        self.surface_by_main_entity.remove(&entity);
    }

    pub(crate) fn core_draw(&self, id: PaintId) -> Option<&RetainedDraw> {
        let camera = self.owners.get(&id)?.camera;
        let record = self.paint.get(&camera)?.get(&id)?;
        Some(&record.value.draw)
    }

    pub(crate) fn retint_text(
        &mut self,
        paints: impl IntoIterator<Item = PaintId>,
        section: Entity,
        color: bevy::color::LinearRgba,
    ) {
        for id in paints {
            if id.entity != section || id.family != PaintFamily::Text {
                continue;
            }
            let Some(owner) = self.owners.get(&id) else {
                continue;
            };
            let camera = owner.camera;
            let group = owner.group;
            let paint = self
                .paint
                .get_mut(&camera)
                .expect("owned text camera must exist");
            self.touched_surfaces.insert(camera);
            let outcome = paint.update_in_place(&id, |record| record.retint_glyphs(color));
            if outcome.is_some_and(|outcome| outcome != crate::UpdateOutcome::Unchanged) {
                let epoch = paint.latest_damage_epoch();
                self.order
                    .entry(camera)
                    .or_default()
                    .note_direct(group, epoch);
                self.dirty_surfaces.insert(camera);
            }
        }
    }

    pub(crate) fn retint_owned_node(
        &mut self,
        id: PaintId,
        color: bevy::color::LinearRgba,
    ) -> bool {
        let Some(owner) = self.owners.get(&id) else {
            return false;
        };
        let outcome = self
            .paint
            .get_mut(&owner.camera)
            .and_then(|paint| paint.update_in_place(&id, |record| record.retint_node(color)));
        self.touched_surfaces.insert(owner.camera);
        if outcome.is_some_and(|outcome| outcome != crate::UpdateOutcome::Unchanged) {
            let camera = owner.camera;
            let group = owner.group;
            let epoch = self.paint[&camera].latest_damage_epoch();
            self.order
                .entry(camera)
                .or_default()
                .note_direct(group, epoch);
            self.dirty_surfaces.insert(camera);
        }
        outcome.is_some()
    }

    fn reposition(&mut self, entity: MainEntity, transform: Affine2, clip: Option<Rect>) {
        let Some(ids) = self.by_main_entity.get(&entity).cloned() else {
            return;
        };
        let camera = self.owners[&ids[0]].camera;
        let paint = self
            .paint
            .get_mut(&camera)
            .expect("owned paint camera must exist");
        self.touched_surfaces.insert(camera);
        let order = self.order.entry(camera).or_default();
        let mut changed = false;
        for id in ids {
            let owner = self
                .owners
                .get(&id)
                .expect("main-entity index must contain an owned paint record");
            debug_assert_eq!(owner.camera, camera);
            let group = owner.group;
            let mut visibility_changed = false;
            let outcome = paint.update_with_coverage(&id, |record, previous| {
                let coverage = record.reposition(transform, clip, previous)?;
                visibility_changed = previous.is_empty() != coverage.is_empty();
                Some(coverage)
            });
            let Some(outcome) =
                outcome.filter(|outcome| *outcome != crate::UpdateOutcome::Unchanged)
            else {
                continue;
            };
            let epoch = paint.latest_damage_epoch();
            changed = true;
            order.note_direct(group, epoch);
            if visibility_changed {
                order.order_dirty = true;
                order.spatial = None;
            } else if outcome.coverage_changed() {
                if let Some(group) = group
                    && order.spatial.is_some()
                    && !order.order_dirty
                    && !order.group_dirty
                {
                    order.bounds_dirty.insert(group);
                } else {
                    order.spatial = None;
                }
            }
        }
        if changed {
            self.dirty_surfaces.insert(camera);
        }
    }

    pub(crate) fn remove(&mut self, commands: &mut Commands, id: PaintId) {
        let Some(owner) = self.detach_owner(id) else {
            return;
        };

        let paint = self.paint.get_mut(&owner.camera).unwrap();
        paint.remove(&id);
        self.dirty_surfaces.insert(owner.camera);
        self.touched_surfaces.insert(owner.camera);
        let index = self.order.entry(owner.camera).or_default();
        index.order_dirty = true;
        index.spatial = None;
        if let Ok(mut entity) = commands.get_entity(owner.render_entity) {
            entity.despawn();
        }
    }

    fn detach_owner(&mut self, id: PaintId) -> Option<PaintOwner> {
        let owner = self.owners.remove(&id)?;
        if let Some(main_entity) = owner.main_entity
            && let Some(ids) = self.by_main_entity.get_mut(&main_entity)
        {
            ids.retain(|candidate| *candidate != id);
            if ids.is_empty() {
                self.by_main_entity.remove(&main_entity);
            }
        }
        Some(owner)
    }

    pub(crate) fn upsert(
        &mut self,
        commands: &mut Commands,
        id: PaintId,
        camera: Entity,
        mut draw: RetainedDraw,
        resource: ResourceFingerprint,
        coverage: PaintCoverage,
        painted: bool,
    ) {
        if !painted {
            self.remove(commands, id);
            return;
        }
        let camera = self
            .surface_by_main_entity
            .get(&draw.main_entity)
            .copied()
            .unwrap_or(camera);
        draw.camera = camera;
        let main_entity = draw.main_entity;
        self.upsert_at(
            commands,
            id,
            camera,
            draw,
            resource,
            coverage,
            Some(main_entity),
        );
    }

    pub(crate) fn upsert_boundary(
        &mut self,
        commands: &mut Commands,
        id: PaintId,
        camera: Entity,
        mut draw: RetainedDraw,
        coverage: PaintCoverage,
    ) {
        draw.camera = camera;
        self.upsert_at(
            commands,
            id,
            camera,
            draw,
            ResourceFingerprint::None,
            coverage,
            None,
        );
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "one canonical record has independent identity, target, draw, and coverage"
    )]
    fn upsert_at(
        &mut self,
        commands: &mut Commands,
        id: PaintId,
        camera: Entity,
        mut draw: RetainedDraw,
        resource: ResourceFingerprint,
        coverage: PaintCoverage,
        index_owner: Option<MainEntity>,
    ) {
        self.touched_surfaces.insert(camera);
        let new_order = (FloatBits::new(draw.z_order), draw.paint_order);
        let (old_camera, render_entity, old_order, old_index_owner) = match self.owners.entry(id) {
            Entry::Occupied(mut owner) => {
                let old_camera = owner.get().camera;
                let previous = (
                    Some(old_camera),
                    owner.get().render_entity,
                    Some(owner.get().order),
                    Some(owner.get().main_entity),
                );
                owner.get_mut().camera = camera;
                owner.get_mut().main_entity = index_owner;
                owner.get_mut().order = new_order;
                if old_camera != camera || previous.2 != Some(new_order) {
                    owner.get_mut().group = None;
                }
                previous
            }
            Entry::Vacant(owner) => {
                let render_entity = commands.spawn_empty().id();
                owner.insert(PaintOwner {
                    camera,
                    main_entity: index_owner,
                    render_entity,
                    order: new_order,
                    group: None,
                });
                (None, render_entity, None, None)
            }
        };
        if old_index_owner != Some(index_owner) {
            if let Some(Some(old_main_entity)) = old_index_owner
                && let Some(ids) = self.by_main_entity.get_mut(&old_main_entity)
            {
                ids.retain(|candidate| *candidate != id);
                if ids.is_empty() {
                    self.by_main_entity.remove(&old_main_entity);
                }
            }
            if let Some(index_owner) = index_owner {
                self.by_main_entity.entry(index_owner).or_default().push(id);
            }
        }
        draw.render_entity = render_entity;

        if let Some(old_camera) = old_camera
            && old_camera != camera
        {
            self.paint.get_mut(&old_camera).unwrap().remove(&id);
            self.dirty_surfaces.insert(old_camera);
            self.touched_surfaces.insert(old_camera);
            let index = self.order.entry(old_camera).or_default();
            index.order_dirty = true;
            index.spatial = None;
        }
        let visibility_changed = old_camera == Some(camera)
            && self
                .paint
                .get(&camera)
                .and_then(|paint| paint.get(&id))
                .is_some_and(|record| record.coverage.is_empty() != coverage.is_empty());
        let outcome = self.paint.entry(camera).or_default().upsert(
            id,
            PaintRecord {
                coverage,
                value: RetainedRecord::new(draw, resource),
            },
        );
        if outcome != crate::UpdateOutcome::Unchanged {
            self.dirty_surfaces.insert(camera);
            let epoch = self.paint[&camera].latest_damage_epoch();
            let group = self.owners[&id].group;
            self.order
                .entry(camera)
                .or_default()
                .note_direct(group, epoch);
        }
        let coverage_changed = old_camera != Some(camera) || outcome.coverage_changed();
        if old_camera != Some(camera) || old_order != Some(new_order) || visibility_changed {
            let index = self.order.entry(camera).or_default();
            index.order_dirty = true;
            index.spatial = None;
        }
        if id.family == PaintFamily::Border && outcome != crate::UpdateOutcome::Unchanged {
            let index = self.order.entry(camera).or_default();
            index.group_dirty = true;
            index.spatial = None;
        }
        if coverage_changed {
            let index = self.order.entry(camera).or_default();
            if let Some(&group) = index.group_by_id.get(&id)
                && index.spatial.is_some()
                && !index.order_dirty
                && !index.group_dirty
            {
                index.bounds_dirty.insert(group);
            } else {
                index.spatial = None;
            }
        }
    }
}

pub(crate) fn coverage(size: Vec2, transform: Affine2, clip: Option<Rect>) -> Option<PhysicalRect> {
    coverage_rect(Rect::from_center_size(Vec2::ZERO, size), transform, clip)
}

pub(crate) fn coverage_rect(
    rect: Rect,
    transform: Affine2,
    clip: Option<Rect>,
) -> Option<PhysicalRect> {
    let corners = [
        rect.min,
        Vec2::new(rect.max.x, rect.min.y),
        rect.max,
        Vec2::new(rect.min.x, rect.max.y),
    ]
    .map(|corner| transform.transform_point2(corner));
    let mut min = corners[0];
    let mut max = corners[0];
    for corner in &corners[1..] {
        min = min.min(*corner);
        max = max.max(*corner);
    }
    if let Some(clip) = clip {
        min = min.max(clip.min);
        max = max.min(clip.max);
    }
    PhysicalRect::from_min_max(
        min.x.floor() as i32,
        min.y.floor() as i32,
        max.x.ceil() as i32,
        max.y.ceil() as i32,
    )
}

#[derive(bevy::prelude::Resource, Default)]
pub(crate) struct RetainedUiScene(Mutex<RetainedUiSurfaces>);

impl RetainedUiScene {
    pub(crate) fn lock(&self) -> MutexGuard<'_, RetainedUiSurfaces> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub(crate) fn acknowledge(&self, camera: Entity, plan: &RepairPlan) {
        let mut surfaces = self.lock();
        let Some(paint) = surfaces.paint.get_mut(&camera) else {
            return;
        };
        paint.acknowledge(plan);
        let clean = !paint.has_damage();
        surfaces
            .order
            .entry(camera)
            .or_default()
            .acknowledge(plan.through_epoch());
        if clean {
            surfaces.dirty_surfaces.remove(&camera);
        }
    }

    pub(crate) fn has_visible_records(&self, camera: Entity, target: PhysicalRect) -> bool {
        self.lock().paint.get(&camera).is_some_and(|paint| {
            paint.iter().any(|(_, record)| {
                record
                    .coverage
                    .iter()
                    .any(|coverage| coverage.intersection(target).is_some())
            })
        })
    }

    pub(crate) fn invalidate(&self, camera: Entity, coverage: PhysicalRect) {
        let mut surfaces = self.lock();
        if let Some(paint) = surfaces.paint.get_mut(&camera) {
            paint.invalidate(coverage);
            surfaces.dirty_surfaces.insert(camera);
            surfaces.touched_surfaces.insert(camera);
        }
    }

    pub(crate) fn invalidate_volatile(&self, camera: Entity, coverage: PhysicalRect) {
        let mut surfaces = self.lock();
        let paint = surfaces.paint.entry(camera).or_default();
        paint.invalidate(coverage);
        surfaces.dirty_surfaces.insert(camera);
        surfaces.touched_surfaces.insert(camera);
    }

    pub(crate) fn dirty_surfaces(&self) -> Vec<Entity> {
        self.lock().dirty_surfaces.iter().copied().collect()
    }

    pub(crate) fn discard_damage(&self, camera: Entity) {
        let mut surfaces = self.lock();
        let Some(paint) = surfaces.paint.get_mut(&camera) else {
            return;
        };
        if let Some(plan) = paint.repair_plan() {
            paint.acknowledge(&plan);
        }
        if !paint.has_damage() {
            surfaces.dirty_surfaces.remove(&camera);
        }
    }

    pub(crate) fn propagate_boundaries(&self, boundaries: &[(Entity, PaintId, PhysicalRect)]) {
        let mut surfaces = self.lock();
        for &(source, id, source_bounds) in boundaries {
            let Some(source_plan) = surfaces
                .paint
                .get_mut(&source)
                .and_then(RetainedPaint::repair_plan)
            else {
                continue;
            };
            let source_epoch = source_plan.through_epoch();
            if surfaces.propagated_epochs.get(&source) == Some(&source_epoch) {
                continue;
            }
            surfaces.propagated_epochs.insert(source, source_epoch);
            let Some(owner) = surfaces.owners.get(&id) else {
                continue;
            };
            let parent = owner.camera;
            let group = owner.group;
            let boundary_draw = &surfaces.paint[&parent]
                .get(&id)
                .expect("owned boundary paint must exist")
                .value
                .draw;
            let RetainedDrawItem::Boundary(boundary) = &boundary_draw.item else {
                unreachable!("boundary ownership must point to a boundary draw");
            };
            if boundary.opacity() == 0.0 {
                continue;
            }
            let source_size = Vec2::new(
                (source_bounds.max_x() - source_bounds.min_x()) as f32,
                (source_bounds.max_y() - source_bounds.min_y()) as f32,
            );
            let output_size = boundary.size();
            let mapped: Vec<_> = source_plan
                .regions()
                .iter()
                .filter_map(|region| region.intersection(source_bounds))
                .filter_map(|region| {
                    let local_min = Vec2::new(
                        (region.min_x() - source_bounds.min_x()) as f32,
                        (region.min_y() - source_bounds.min_y()) as f32,
                    );
                    let local_max = Vec2::new(
                        (region.max_x() - source_bounds.min_x()) as f32,
                        (region.max_y() - source_bounds.min_y()) as f32,
                    );
                    let rect = Rect::from_corners(
                        (local_min / source_size - Vec2::splat(0.5)) * output_size,
                        (local_max / source_size - Vec2::splat(0.5)) * output_size,
                    );
                    coverage_rect(rect, boundary_draw.transform, boundary_draw.clip)
                })
                .collect();
            if mapped.is_empty() {
                continue;
            }
            let paint = surfaces
                .paint
                .get_mut(&parent)
                .expect("owned boundary paint target must exist");
            for region in mapped {
                paint.invalidate(region);
            }
            let epoch = paint.latest_damage_epoch();
            surfaces
                .order
                .entry(parent)
                .or_default()
                .note_direct(group, epoch);
            surfaces.touched_surfaces.insert(parent);
            surfaces.dirty_surfaces.insert(parent);
        }
    }
}

#[derive(bevy::prelude::Resource, Default)]
pub(crate) struct RetainedRepairPlans(pub(crate) HashMap<Entity, RepairPlan>);

#[derive(bevy::prelude::Resource, Default)]
pub(crate) struct RetainedItems {
    pub(crate) items: Mutex<HashMap<Entity, RetainedItem>>,
    pub(crate) boundaries: Mutex<HashMap<Entity, RetainedBoundaryDraw>>,
}

#[derive(Clone)]
pub(crate) struct RetainedItem {
    pub(crate) coverage: PaintCoverage,
    pub(crate) sampled_images: Box<[AssetId<Image>]>,
}

#[derive(Clone)]
pub(crate) struct RetainedBoundaryDraw {
    pub(crate) render_entity: Entity,
    pub(crate) main_entity: MainEntity,
    pub(crate) camera: Entity,
    pub(crate) z_order: f32,
    pub(crate) surface: Entity,
    pub(crate) transform: Affine2,
    pub(crate) size: Vec2,
    pub(crate) opacity: f32,
}

#[derive(Clone)]
pub(crate) struct RetainedMaterialReplay {
    pub(crate) draw: RetainedDraw,
    pub(crate) item: RetainedMaterialItem,
}

#[derive(bevy::prelude::Resource, Default)]
pub(crate) struct RetainedMaterialReplays(pub(crate) Vec<RetainedMaterialReplay>);

/// Atomic render-world counters for change-driven paint extraction.
#[derive(bevy::prelude::Resource, Default)]
pub struct RetainedUiPaintCounters {
    candidates: AtomicU64,
    records_compared: AtomicU64,
    records_changed: AtomicU64,
    records_staged: AtomicU64,
    records_removed: AtomicU64,
    damage_events: AtomicU64,
}

impl RetainedUiPaintCounters {
    /// Returns accumulated work without resetting the counters.
    pub fn snapshot(&self) -> WorkCounters {
        WorkCounters {
            candidates: self.candidates.load(Ordering::Relaxed),
            records_compared: self.records_compared.load(Ordering::Relaxed),
            records_changed: self.records_changed.load(Ordering::Relaxed),
            records_staged: self.records_staged.load(Ordering::Relaxed),
            records_removed: self.records_removed.load(Ordering::Relaxed),
            damage_events: self.damage_events.load(Ordering::Relaxed),
        }
    }

    fn add(&self, work: WorkCounters) {
        self.candidates
            .fetch_add(work.candidates, Ordering::Relaxed);
        self.records_compared
            .fetch_add(work.records_compared, Ordering::Relaxed);
        self.records_changed
            .fetch_add(work.records_changed, Ordering::Relaxed);
        self.records_staged
            .fetch_add(work.records_staged, Ordering::Relaxed);
        self.records_removed
            .fetch_add(work.records_removed, Ordering::Relaxed);
        self.damage_events
            .fetch_add(work.damage_events, Ordering::Relaxed);
    }

    fn add_staged(&self, count: usize) {
        self.records_staged.fetch_add(
            u64::try_from(count).expect("staged retained record count exceeds u64"),
            Ordering::Relaxed,
        );
    }
}

pub(crate) fn cleanup_retained_ui(
    mut commands: Commands,
    state: Res<RetainedUiScene>,
    mut removed_main_entities: RemovedComponents<MainEntity>,
) {
    let mut surfaces = state.lock();
    let removed_cameras: Vec<_> = removed_main_entities
        .read()
        .filter(|camera| surfaces.paint.contains_key(camera))
        .collect();
    for camera in removed_cameras {
        if let Some(paint) = surfaces.paint.remove(&camera) {
            for (id, record) in paint.iter() {
                if let Ok(mut entity) = commands.get_entity(record.value.render_entity) {
                    entity.despawn();
                }
                let owner = surfaces
                    .detach_owner(*id)
                    .expect("retained paint must have an owner");
                debug_assert_eq!(owner.camera, camera);
            }
        }
        surfaces.dirty_surfaces.remove(&camera);
        surfaces.touched_surfaces.remove(&camera);
        surfaces.propagated_epochs.remove(&camera);
        surfaces.order.remove(&camera);
    }
}

pub(crate) fn extract_boundary_ownership(
    mut commands: Commands,
    state: Res<RetainedUiScene>,
    changed: Extract<
        Query<
            (
                Entity,
                &'static Inherited<ComputedUiPaintTarget>,
                &'static ComputedUiTargetCamera,
            ),
            Changed<Inherited<ComputedUiPaintTarget>>,
        >,
    >,
    targets: Extract<Query<&'static ComputedUiTargetCamera, With<Node>>>,
    boundary_entities: Extract<Query<&'static RenderEntity, With<RepaintBoundary>>>,
    camera_map: Extract<bevy::ui_render::UiCameraMap>,
    mut removed: Extract<RemovedComponents<Inherited<ComputedUiPaintTarget>>>,
) {
    let mut camera_mapper = camera_map.get_mapper();
    let mut surfaces = state.lock();
    for (entity, owner, target) in &changed {
        let Some(camera) = camera_mapper.map(target) else {
            surfaces.remove_main_entity(&mut commands, entity.into());
            continue;
        };
        let surface = boundary_entities
            .get(owner.0 .0)
            .expect("repaint boundaries must be synchronized to the render world")
            .id();
        surfaces.set_entity_surface(entity.into(), Some(surface), camera);
    }
    for entity in removed.read() {
        let Ok(target) = targets.get(entity) else {
            surfaces.remove_main_entity(&mut commands, entity.into());
            continue;
        };
        let Some(camera) = camera_mapper.map(target) else {
            surfaces.remove_main_entity(&mut commands, entity.into());
            continue;
        };
        surfaces.set_entity_surface(entity.into(), None, camera);
    }
}

pub(crate) fn extract_retained_placements(
    state: Res<RetainedUiScene>,
    changed: Extract<
        Query<
            (
                Entity,
                &'static UiGlobalTransform,
                Option<&'static CalculatedClip>,
                Option<&'static Inherited<ComputedUiPaintTarget>>,
                &'static ComputedNode,
                Option<&'static TextScroll>,
            ),
            (With<Node>, Changed<UiGlobalTransform>),
        >,
    >,
) {
    let mut surfaces = state.lock();
    for (entity, transform, clip, owner, node, scroll) in &changed {
        let clip = retained_clip(entity, node, transform, clip, owner);
        let transform = transform.affine();
        let clip = if scroll.is_some() {
            let content_box = node.content_box();
            let text_clip = Rect::from_center_size(
                transform.translation + content_box.center(),
                content_box.size(),
            );
            Some(clip.map_or(text_clip, |clip| clip.intersect(text_clip)))
        } else {
            clip
        };
        surfaces.reposition(entity.into(), transform, clip);
    }
}

fn group_bounds(
    paint: &RetainedPaint<PaintId, RetainedRecord>,
    ids: &[PaintId],
    positions: core::ops::Range<usize>,
) -> PhysicalRect {
    positions
        .flat_map(|position| {
            paint
                .get(&ids[position])
                .expect("paint group only contains retained records")
                .coverage
                .iter()
                .copied()
        })
        .reduce(enclosing_rect)
        .expect("retained paint groups have visible coverage")
}

fn enclosing_rect(left: PhysicalRect, right: PhysicalRect) -> PhysicalRect {
    PhysicalRect::from_min_max(
        left.min_x().min(right.min_x()),
        left.min_y().min(right.min_y()),
        left.max_x().max(right.max_x()),
        left.max_y().max(right.max_y()),
    )
    .unwrap()
}

pub(crate) fn replay_retained_ui(
    state: Res<RetainedUiScene>,
    counters: Res<RetainedUiPaintCounters>,
    mut core_runs: ResMut<RetainedCoreRuns>,
    mut gradient_runs: ResMut<RetainedGradientRuns>,
    mut shadow_runs: ResMut<RetainedShadowRuns>,
    retained_items: Res<RetainedItems>,
    mut extracted_slices: ResMut<ExtractedUiTextureSlices>,
    mut extracted_materials: ResMut<RetainedMaterialReplays>,
    pending_materials: Res<RetainedPendingMaterials>,
    mut repair_plans: ResMut<RetainedRepairPlans>,
) {
    let mut surfaces = state.lock();
    core_runs.clear();
    gradient_runs.clear();
    shadow_runs.clear();
    let mut items = retained_items
        .items
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    items.clear();
    let mut boundary_draws = retained_items
        .boundaries
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    boundary_draws.clear();
    extracted_materials.0.clear();
    pending_materials.clear();
    repair_plans.0.clear();
    let mut buffers = ReplayBuffers {
        slices: &mut extracted_slices,
        materials: &mut extracted_materials,
    };

    let touched_surfaces = core::mem::take(&mut surfaces.touched_surfaces);
    for camera in touched_surfaces {
        if let Some(paint) = surfaces.paint.get_mut(&camera) {
            counters.add(paint.take_counters());
        }
    }
    let dirty_surfaces: Vec<_> = surfaces.dirty_surfaces.iter().copied().collect();

    let RetainedUiSurfaces {
        paint,
        owners,
        by_main_entity: _,
        surface_by_main_entity: _,
        dirty_surfaces: _,
        touched_surfaces: _,
        propagated_epochs: _,
        order,
    } = &mut *surfaces;
    for camera in dirty_surfaces {
        let paint = paint
            .get_mut(&camera)
            .expect("dirty retained surface must have paint state");
        let Some(repair) = paint.repair_plan() else {
            continue;
        };
        repair_plans.0.insert(camera, repair.clone());
        let order = order.entry(camera).or_default();
        if order.order_dirty {
            order.ids.clear();
            order.ids.extend(
                paint
                    .iter()
                    .filter(|(_, record)| !record.coverage.is_empty())
                    .map(|(id, _)| *id),
            );
            order.ids.sort_by(|left_id, right_id| {
                let left = paint
                    .get(left_id)
                    .expect("ordered retained record must exist");
                let right = paint
                    .get(right_id)
                    .expect("ordered retained record must exist");
                left.value
                    .draw
                    .z_order
                    .total_cmp(&right.value.draw.z_order)
                    .then_with(|| left_id.family.cmp(&right_id.family))
                    .then_with(|| {
                        left.value
                            .draw
                            .paint_order
                            .cmp(&right.value.draw.paint_order)
                    })
                    .then_with(|| left_id.entity.cmp(&right_id.entity))
                    .then_with(|| left_id.ordinal.cmp(&right_id.ordinal))
            });
            order.group_dirty = true;
            order.order_dirty = false;
        }
        if order.group_dirty {
            order.groups.clear();
            order.group_by_id.clear();
            order.groups_by_entity.clear();
            let mut start = 0;
            while start < order.ids.len() {
                let id = order.ids[start];
                let mut end = start + 1;
                if id.family == PaintFamily::Border {
                    let record = paint.get(&id).expect("ordered retained record must exist");
                    while let Some(next_id) = order.ids.get(end).copied()
                        && next_id.family == PaintFamily::Border
                        && next_id.entity == id.entity
                        && record.value.can_merge_border(
                            &paint
                                .get(&next_id)
                                .expect("ordered retained record must exist")
                                .value,
                        )
                    {
                        end += 1;
                    }
                }
                let group = order.groups.len();
                for &id in &order.ids[start..end] {
                    order.group_by_id.insert(id, group);
                    owners
                        .get_mut(&id)
                        .expect("ordered paint must have an owner")
                        .group = Some(group);
                }
                order
                    .groups_by_entity
                    .entry(id.entity)
                    .or_default()
                    .push(group);
                order.groups.push(start..end);
                start = end;
            }
            order.group_dirty = false;
            order.spatial = None;
        }
        if order.spatial.is_none() {
            let mut entries = Vec::new();
            for (group, positions) in order.groups.iter().enumerate() {
                entries.push((group_bounds(paint, &order.ids, positions.clone()), group));
            }
            order.spatial = Some(SpatialIndex::new(entries));
            order.candidate_marks.resize(order.groups.len(), 0);
            order.direct_groups.clear();
            order.direct_epochs.clear();
            order.direct_epochs.resize(order.groups.len(), None);
            order.bounds_dirty.clear();
        }
        order.candidate_generation = order.candidate_generation.wrapping_add(1);
        if order.candidate_generation == 0 {
            order.candidate_marks.fill(0);
            order.candidate_generation = 1;
        }
        let generation = order.candidate_generation;
        order.candidates.clear();
        for &direct_group in &order.direct_groups {
            if order.direct_epochs[direct_group]
                .is_some_and(|[earliest, _]| earliest <= repair.through_epoch())
            {
                let entity = order.ids[order.groups[direct_group].start].entity;
                for &group in &order.groups_by_entity[&entity] {
                    if order.candidate_marks[group] != generation {
                        order.candidate_marks[group] = generation;
                        order.candidates.push(group);
                    }
                }
            }
        }
        if order.candidates.len() != order.groups.len() {
            let updates: Vec<_> = order
                .bounds_dirty
                .drain()
                .map(|group| {
                    (
                        group,
                        group_bounds(paint, &order.ids, order.groups[group].clone()),
                    )
                })
                .collect();
            order
                .spatial
                .as_mut()
                .expect("retained spatial index was just built")
                .update_many(updates);
            let spatial = order
                .spatial
                .as_ref()
                .expect("retained spatial index was just built");
            let mut mark = |position| {
                if order.candidate_marks[position] != generation {
                    order.candidate_marks[position] = generation;
                    order.candidates.push(position);
                }
            };
            if repair.regions().len() < spatial.len() {
                for &region in repair.regions() {
                    spatial.query(region, &mut mark);
                }
            } else {
                spatial.query_intersecting(repair.spatial(), &mut mark);
            }
        }
        if order.candidates.len() == order.groups.len() {
            order.candidates.clear();
            order.candidates.extend(0..order.groups.len());
        } else {
            order.candidates.sort_unstable();
        }
        counters.add_staged(order.candidates.len());
        let mut core_run = None;
        let mut gradient_run = None;
        let mut shadow_run = None;
        for &group in &order.candidates {
            let positions = order.groups[group].clone();
            let id = order.ids[positions.start];
            let record = paint
                .get(&id)
                .expect("spatial index only contains retained records");
            if let RetainedDrawItem::Boundary(boundary) = &record.value.draw.item {
                core_run = None;
                gradient_run = None;
                shadow_run = None;
                boundary_draws.insert(
                    record.value.draw.render_entity,
                    RetainedBoundaryDraw {
                        render_entity: record.value.draw.render_entity,
                        main_entity: record.value.draw.main_entity,
                        camera: record.value.draw.camera,
                        z_order: record.value.draw.z_order,
                        surface: boundary.surface,
                        transform: record.value.draw.transform,
                        size: boundary.size(),
                        opacity: boundary.opacity(),
                    },
                );
                let coverage = PaintCoverage::from_regions(positions.flat_map(|position| {
                    paint
                        .get(&order.ids[position])
                        .expect("paint group only contains retained records")
                        .coverage
                        .iter()
                        .copied()
                }));
                items.insert(
                    record.value.draw.render_entity,
                    RetainedItem {
                        coverage,
                        sampled_images: Box::default(),
                    },
                );
                continue;
            }
            let border_flags = (id.family == PaintFamily::Border).then(|| {
                positions
                    .clone()
                    .filter_map(|position| {
                        paint
                            .get(&order.ids[position])
                            .and_then(|record| record.value.border_parts())
                            .map(|(_, flags)| flags)
                    })
                    .fold(0, |combined, flags| combined | flags)
            });
            if let RetainedDrawItem::BoxShadow(item) = &record.value.draw.item {
                core_run = None;
                gradient_run = None;
                shadow_runs.push(&mut shadow_run, &record.value.draw, item);
                continue;
            }
            shadow_run = None;
            if let RetainedDrawItem::Gradient(item) = &record.value.draw.item {
                core_run = None;
                gradient_runs.push(&mut gradient_run, &record.value.draw, item);
                continue;
            }
            gradient_run = None;
            if matches!(
                record.value.draw.item,
                RetainedDrawItem::Node(_) | RetainedDrawItem::Glyphs(_)
            ) {
                push_core_replayed(
                    &mut core_runs,
                    &mut core_run,
                    record.value.core_source(id),
                    border_flags,
                    &record.value.draw,
                );
                continue;
            }
            core_run = None;
            let coverage = PaintCoverage::from_regions(positions.flat_map(|position| {
                paint
                    .get(&order.ids[position])
                    .expect("paint group only contains retained records")
                    .coverage
                    .iter()
                    .copied()
            }));
            push_replayed(&mut buffers, &mut items, &record.value.draw, coverage);
        }
    }
}

fn push_core_replayed(
    runs: &mut RetainedCoreRuns,
    current_run: &mut Option<usize>,
    source: RetainedCoreSource<'_>,
    border_flags: Option<u32>,
    draw: &RetainedDraw,
) {
    if let Some(index) = *current_run
        && runs.image(index) == draw.image
    {
        runs.push(index, source, border_flags);
        return;
    }
    let index = runs.start(
        draw.render_entity,
        draw.main_entity,
        draw.camera,
        draw.z_order,
        draw.image,
        source,
        border_flags,
    );
    *current_run = Some(index);
}

struct ReplayBuffers<'a> {
    slices: &'a mut ExtractedUiTextureSlices,
    materials: &'a mut RetainedMaterialReplays,
}

fn push_replayed(
    buffers: &mut ReplayBuffers,
    items: &mut HashMap<Entity, RetainedItem>,
    draw: &RetainedDraw,
    coverage: PaintCoverage,
) {
    match &draw.item {
        RetainedDrawItem::Boundary(_) => unreachable!("boundaries use retained composition runs"),
        RetainedDrawItem::BoxShadow(_) => unreachable!("box shadows use retained instancing"),
        RetainedDrawItem::Gradient(_) => unreachable!("gradients use retained instancing"),
        RetainedDrawItem::Material(item) => {
            buffers.materials.0.push(RetainedMaterialReplay {
                draw: draw.clone(),
                item: item.clone(),
            });
            items.insert(
                draw.render_entity,
                RetainedItem {
                    coverage,
                    sampled_images: item.sampled_images.clone(),
                },
            );
        }
        RetainedDrawItem::Node(_) => unreachable!("ordinary nodes use persistent preparation"),
        RetainedDrawItem::Glyphs(_) => unreachable!("glyphs use persistent preparation"),
        RetainedDrawItem::TextureSlice(item) => {
            buffers.slices.slices.push(ExtractedUiTextureSlice {
                stack_index: item.stack_index,
                transform: draw.transform,
                rect: item.rect,
                atlas_rect: item.atlas_rect,
                image: draw.image,
                clip: draw.clip,
                extracted_camera_entity: draw.camera,
                color: item.color,
                image_scale_mode: item.image_scale_mode.clone(),
                flip_x: item.flip_x,
                flip_y: item.flip_y,
                inverse_scale_factor: item.inverse_scale_factor,
                main_entity: draw.main_entity,
                render_entity: draw.render_entity,
            });
            items.insert(
                draw.render_entity,
                RetainedItem {
                    coverage,
                    sampled_images: core_sampled_images(draw.image),
                },
            );
        }
    }
}

fn core_sampled_images(image: AssetId<Image>) -> Box<[AssetId<Image>]> {
    (image != AssetId::default())
        .then_some(image)
        .into_iter()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_damage_keeps_later_work_in_fixed_size_epoch_intervals() {
        let mut order = PaintOrder {
            direct_epochs: vec![None],
            ..Default::default()
        };

        order.note_direct(Some(0), 2);
        order.note_direct(Some(0), 4);
        order.note_direct(Some(0), 5);
        assert_eq!(order.direct_groups, [0]);
        assert_eq!(order.direct_epochs, [Some([2, 5])]);

        order.acknowledge(4);
        assert_eq!(order.direct_groups, [0]);
        assert_eq!(order.direct_epochs, [Some([5, 5])]);

        order.acknowledge(5);
        assert!(order.direct_groups.is_empty());
        assert_eq!(order.direct_epochs, [None]);
    }
}
