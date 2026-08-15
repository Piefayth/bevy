//! Retained UI paint records shared by every paint family.

use crate::{
    background::PendingRetainedBackgrounds,
    border::{edge_rect, EDGE_FLAGS},
    boundary::{retained_clip, RepaintBoundary},
    core::{
        prepare_glyph_instance, prepare_node_instance, prepare_persistent_instance, GpuUiInstance,
        RetainedCoreRuns, RetainedCoreSource,
    },
    damage::{DamageJournal, SpatialIndex},
    gradient_render::{prepare_gradient_instances, GpuGradientInstance, RetainedGradientRuns},
    material::RetainedPendingMaterials,
    paint::PaintState,
    shadow_render::{prepare_shadow_instance, GpuShadowInstance, RetainedShadowRuns},
    FloatBits, PaintCoverage, PaintRecord, PhysicalRect, RepairPlan, WorkCounters,
};
use bevy::{
    app::Inherited,
    asset::{AssetId, UntypedAssetId},
    color::{Alpha, ColorToComponents},
    ecs::{
        change_detection::{DetectChanges, Ref},
        entity::{Entity, EntityHashMap, EntityHashSet},
        lifecycle::RemovedComponents,
        query::{Changed, With},
        system::{Commands, Query, Res, ResMut},
    },
    image::Image,
    math::{Affine2, Rect, Vec2},
    platform::collections::{HashMap, HashSet},
    render::{
        sync_world::{MainEntity, MainEntityHashMap, RenderEntity},
        view::{ExtractedView, RetainedViewEntity},
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
        NodeType, UiViewTarget,
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
    /// Layout and item offsets stay separate so layout movement never requires transform inversion.
    pub(crate) layout_translation: Vec2,
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
    Border(RetainedBorderItem),
    BoxShadow(RetainedBoxShadowItem),
    Gradient(RetainedGradientItem),
    Material(RetainedMaterialItem),
    Node(RetainedNodeItem),
    Glyphs(Box<[RetainedGlyph]>),
    TextureSlice(RetainedTextureSliceItem),
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct RetainedBorderItem {
    rect: [FloatBits; 4],
    border: [FloatBits; 4],
    border_radius: [FloatBits; 4],
    colors: [[FloatBits; 4]; 4],
    edge_coverage: [Option<PhysicalRect>; 4],
}

impl RetainedBorderItem {
    pub(crate) fn new(
        rect: Rect,
        border: BorderRect,
        border_radius: ResolvedBorderRadius,
        mut colors: [bevy::color::LinearRgba; 4],
        edge_coverage: [Option<PhysicalRect>; 4],
    ) -> Self {
        let widths = [
            border.min_inset.x,
            border.min_inset.y,
            border.max_inset.x,
            border.max_inset.y,
        ];
        for (edge, color) in colors.iter_mut().enumerate() {
            if widths[edge] <= 0.0 || color.is_fully_transparent() {
                *color = bevy::color::LinearRgba::NONE;
            }
        }
        Self {
            rect: rect_fingerprint(rect),
            border: widths.map(FloatBits::new),
            border_radius: <[f32; 4]>::from(border_radius).map(FloatBits::new),
            colors: colors.map(|color| color.to_f32_array().map(FloatBits::new)),
            edge_coverage,
        }
    }

    fn rect(&self) -> Rect {
        rect_from_fingerprint(self.rect)
    }

    fn border(&self) -> BorderRect {
        BorderRect {
            min_inset: Vec2::new(self.border[0].get(), self.border[1].get()),
            max_inset: Vec2::new(self.border[2].get(), self.border[3].get()),
        }
    }

    fn border_radius(&self) -> ResolvedBorderRadius {
        ResolvedBorderRadius {
            top_left: self.border_radius[0].get(),
            top_right: self.border_radius[1].get(),
            bottom_right: self.border_radius[2].get(),
            bottom_left: self.border_radius[3].get(),
        }
    }

    fn color(&self, edge: usize) -> bevy::color::LinearRgba {
        let color = self.colors[edge];
        bevy::color::LinearRgba::new(
            color[0].get(),
            color[1].get(),
            color[2].get(),
            color[3].get(),
        )
    }

    pub(crate) fn grouped_nodes(&self) -> [Option<RetainedNodeItem>; 4] {
        let mut nodes = [None; 4];
        let mut completed = 0;
        let border = self.border();
        for edge in 0..4 {
            if self.colors[edge][3].get() == 0.0 || completed & EDGE_FLAGS[edge] != 0 {
                continue;
            }
            let mut flags = EDGE_FLAGS[edge];
            for (next, _) in EDGE_FLAGS.iter().enumerate().skip(edge + 1) {
                if self.colors[edge] == self.colors[next] {
                    flags |= EDGE_FLAGS[next];
                }
            }
            completed |= flags;
            nodes[edge] = Some(RetainedNodeItem {
                color: self.color(edge),
                rect: self.rect(),
                atlas_scaling: None,
                image_extent: None,
                flip_x: false,
                flip_y: false,
                border,
                border_radius: self.border_radius(),
                node_type: NodeType::Border(flags),
            });
        }
        nodes
    }
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
    stops: SmallVec<[RetainedGradientStop; 2]>,
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
    prepared: Option<PreparedDraw>,
    draw: RetainedDraw,
}

enum PreparedDraw {
    Core(PreparedCore),
    Gradient {
        color_space: bevy::ui::InterpolationColorSpace,
        instances: Vec<GpuGradientInstance>,
    },
    Shadow {
        samples: u32,
        instance: GpuShadowInstance,
    },
}

enum PreparedCore {
    One(GpuUiInstance),
    Many(Vec<GpuUiInstance>),
}

impl PreparedCore {
    fn as_slice(&self) -> &[GpuUiInstance] {
        match self {
            Self::One(instance) => core::slice::from_ref(instance),
            Self::Many(instances) => instances,
        }
    }

    fn reposition(&mut self, draw: &RetainedDraw) {
        match (self, &draw.item) {
            (Self::One(instance), RetainedDrawItem::Node(_)) => {
                instance.set_placement(draw.transform, draw.clip, Vec2::ZERO);
            }
            (Self::Many(instances), RetainedDrawItem::Glyphs(glyphs)) => {
                for (instance, glyph) in instances.iter_mut().zip(glyphs) {
                    instance.set_placement(draw.transform, draw.clip, glyph.translation());
                }
            }
            (Self::Many(instances), RetainedDrawItem::Border(_)) => {
                for instance in instances {
                    instance.set_placement(draw.transform, draw.clip, Vec2::ZERO);
                }
            }
            _ => unreachable!("prepared core data must match its canonical draw family"),
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
        Self {
            resource,
            render_entity: draw.render_entity,
            prepared: None,
            draw,
        }
    }

    fn prepare(&mut self) {
        self.prepared = prepare_draw(&self.draw);
    }

    fn refresh_prepared(&mut self) {
        refresh_prepared(&self.draw, &mut self.prepared);
    }

    fn reposition(
        &mut self,
        source_transform: Affine2,
        clip: Option<Rect>,
        previous_coverage: &PaintCoverage,
    ) -> Option<PaintCoverage> {
        let transform = source_transform
            * Affine2::from_translation(self.draw.layout_translation + self.draw.local_translation);
        if affine_bits(self.draw.transform) == affine_bits(transform)
            && self.draw.clip.map(rect_fingerprint) == clip.map(rect_fingerprint)
        {
            return None;
        }
        self.draw.transform = transform;
        self.draw.clip = clip;
        refresh_border_coverage(&mut self.draw);
        if let Some(prepared) = &mut self.prepared {
            match prepared {
                PreparedDraw::Core(prepared) => prepared.reposition(&self.draw),
                PreparedDraw::Gradient { instances, .. } => {
                    for instance in instances {
                        instance.set_placement(self.draw.transform, self.draw.clip);
                    }
                }
                PreparedDraw::Shadow { instance, .. } => {
                    instance.set_placement(self.draw.transform, self.draw.clip);
                }
            }
        }
        Some(draw_coverage(&self.draw, previous_coverage))
    }

    fn retint_glyphs(&mut self, color: bevy::color::LinearRgba) -> bool {
        let RetainedDrawItem::Glyphs(glyphs) = &mut self.draw.item else {
            return false;
        };
        let mut changed = false;
        if let Some(PreparedDraw::Core(PreparedCore::Many(instances))) = &mut self.prepared {
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

    fn core_source(&self, id: PaintId) -> RetainedCoreSource<'_> {
        match &self.prepared {
            Some(PreparedDraw::Core(prepared)) => RetainedCoreSource::Prepared(prepared.as_slice()),
            _ => RetainedCoreSource::Deferred(id),
        }
    }

    fn gradient_source(
        &self,
    ) -> Option<(bevy::ui::InterpolationColorSpace, &[GpuGradientInstance])> {
        match &self.prepared {
            Some(PreparedDraw::Gradient {
                color_space,
                instances,
            }) => Some((*color_space, instances)),
            _ => None,
        }
    }

    fn shadow_source(&self) -> Option<(u32, GpuShadowInstance)> {
        match &self.prepared {
            Some(PreparedDraw::Shadow { samples, instance }) => Some((*samples, *instance)),
            _ => None,
        }
    }
}

fn prepare_draw(draw: &RetainedDraw) -> Option<PreparedDraw> {
    match &draw.item {
        RetainedDrawItem::Border(border) => {
            let mut instances = Vec::new();
            prepare_border_instances(draw, border, &mut instances);
            Some(PreparedDraw::Core(PreparedCore::Many(instances)))
        }
        RetainedDrawItem::Node(_) => prepare_persistent_instance(draw)
            .map(PreparedCore::One)
            .map(PreparedDraw::Core),
        RetainedDrawItem::Glyphs(glyphs) => glyphs
            .iter()
            .map(|&glyph| {
                glyph
                    .atlas_extent()
                    .map(|extent| prepare_glyph_instance(draw, glyph, extent))
            })
            .collect::<Option<Vec<_>>>()
            .map(PreparedCore::Many)
            .map(PreparedDraw::Core),
        RetainedDrawItem::Gradient(item) => {
            let mut instances = Vec::new();
            prepare_gradient_instances(draw, item, &mut instances);
            Some(PreparedDraw::Gradient {
                color_space: item.color_space(),
                instances,
            })
        }
        RetainedDrawItem::BoxShadow(item) => Some(PreparedDraw::Shadow {
            samples: item.samples(),
            instance: prepare_shadow_instance(draw, item),
        }),
        _ => None,
    }
}

fn prepare_border_instances(
    draw: &RetainedDraw,
    border: &RetainedBorderItem,
    instances: &mut Vec<GpuUiInstance>,
) {
    instances.extend(
        border
            .grouped_nodes()
            .into_iter()
            .flatten()
            .map(|node| prepare_node_instance(draw, &node, Vec2::ONE)),
    );
}

fn refresh_prepared(draw: &RetainedDraw, prepared: &mut Option<PreparedDraw>) {
    match (prepared.as_mut(), &draw.item) {
        (Some(PreparedDraw::Core(PreparedCore::One(instance))), RetainedDrawItem::Node(_)) => {
            if let Some(next) = prepare_persistent_instance(draw) {
                *instance = next;
            } else {
                *prepared = None;
            }
        }
        (
            Some(PreparedDraw::Core(PreparedCore::Many(instances))),
            RetainedDrawItem::Border(border),
        ) => {
            instances.clear();
            prepare_border_instances(draw, border, instances);
        }
        (
            Some(PreparedDraw::Core(PreparedCore::Many(instances))),
            RetainedDrawItem::Glyphs(glyphs),
        ) => {
            instances.clear();
            for &glyph in glyphs {
                let Some(extent) = glyph.atlas_extent() else {
                    *prepared = None;
                    return;
                };
                instances.push(prepare_glyph_instance(draw, glyph, extent));
            }
        }
        (
            Some(PreparedDraw::Gradient {
                color_space,
                instances,
            }),
            RetainedDrawItem::Gradient(item),
        ) => {
            *color_space = item.color_space();
            instances.clear();
            prepare_gradient_instances(draw, item, instances);
        }
        (Some(PreparedDraw::Shadow { samples, instance }), RetainedDrawItem::BoxShadow(item)) => {
            *samples = item.samples();
            *instance = prepare_shadow_instance(draw, item);
        }
        _ => *prepared = prepare_draw(draw),
    }
}

fn draw_coverage(draw: &RetainedDraw, previous: &PaintCoverage) -> PaintCoverage {
    match &draw.item {
        RetainedDrawItem::Boundary(boundary) => {
            coverage(boundary.size(), draw.transform, draw.clip)
                .into_iter()
                .collect()
        }
        RetainedDrawItem::Border(border) => border_coverage(border),
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

fn border_coverage(border: &RetainedBorderItem) -> PaintCoverage {
    border.edge_coverage.into_iter().flatten().collect()
}

fn refresh_border_coverage(draw: &mut RetainedDraw) {
    let RetainedDrawItem::Border(border) = &mut draw.item else {
        return;
    };
    let widths = border.border.map(FloatBits::get);
    let radii = border.border_radius.map(FloatBits::get);
    let size = border.rect().size();
    border.edge_coverage = core::array::from_fn(|edge| {
        (border.colors[edge][3].get() != 0.0)
            .then(|| {
                coverage_rect(
                    edge_rect(size, widths[edge], radii, edge),
                    draw.transform,
                    draw.clip,
                )
            })
            .flatten()
    });
}

fn changed_border_damage(
    previous_draw: &RetainedDraw,
    previous: &RetainedBorderItem,
    draw: &RetainedDraw,
    border: &RetainedBorderItem,
) -> PaintCoverage {
    let all_edges = !retained_common_eq(previous_draw, draw)
        || previous.rect != border.rect
        || previous.border != border.border
        || previous.border_radius != border.border_radius;
    (0..4)
        .filter(|&edge| all_edges || previous.colors[edge] != border.colors[edge])
        .flat_map(|edge| {
            [previous.edge_coverage[edge], border.edge_coverage[edge]]
                .into_iter()
                .flatten()
        })
        .collect()
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
        && left.layout_translation.to_array().map(FloatBits::new)
            == right.layout_translation.to_array().map(FloatBits::new)
}

fn retained_node_merge_eq(left: &RetainedNodeItem, right: &RetainedNodeItem) -> bool {
    left.color.to_f32_array().map(FloatBits::new) == right.color.to_f32_array().map(FloatBits::new)
        && retained_node_shape_eq(left, right)
}

fn retained_node_geometry_eq(left: &RetainedNodeItem, right: &RetainedNodeItem) -> bool {
    retained_node_shape_eq(left, right) && left.node_type == right.node_type
}

fn retained_node_shape_eq(left: &RetainedNodeItem, right: &RetainedNodeItem) -> bool {
    rect_fingerprint(left.rect) == rect_fingerprint(right.rect)
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
        (RetainedDrawItem::Border(left), RetainedDrawItem::Border(right)) => left == right,
        (RetainedDrawItem::BoxShadow(left), RetainedDrawItem::BoxShadow(right)) => left == right,
        (RetainedDrawItem::Gradient(left), RetainedDrawItem::Gradient(right)) => left == right,
        (RetainedDrawItem::Material(left), RetainedDrawItem::Material(right)) => left == right,
        (RetainedDrawItem::Node(left), RetainedDrawItem::Node(right)) => {
            retained_node_merge_eq(left, right)
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
    paint: EntityHashMap<PaintState>,
    records: RecordArena,
    by_main_entity: MainEntityHashMap<SmallVec<[usize; 8]>>,
    surface_by_main_entity: MainEntityHashMap<Entity>,
    dirty_surfaces: Vec<Entity>,
    touched_surfaces: Vec<Entity>,
    propagated_epochs: EntityHashMap<u64>,
    order: EntityHashMap<PaintOrder>,
    surface_origins: EntityHashMap<Vec2>,
    surface_z_origins: EntityHashMap<f32>,
    surface_bounds: EntityHashMap<PhysicalRect>,
    run_by_view: EntityHashMap<(Entity, PaintRunKey)>,
    created_run_views: Vec<RetainedViewEntity>,
    retired_run_views: Vec<RetainedViewEntity>,
}

struct OwnedPaint {
    id: PaintId,
    camera: Entity,
    main_entity: Option<MainEntity>,
    order: (FloatBits, u32),
    group: Option<usize>,
    record: PaintRecord<RetainedRecord>,
}

#[derive(Default)]
struct RecordArena {
    slots: Vec<Option<OwnedPaint>>,
    by_entity: EntityHashMap<SmallVec<[EntityRecordSlot; 8]>>,
    free: Vec<usize>,
}

#[derive(Clone, Copy)]
struct EntityRecordSlot {
    key: (PaintFamily, u32),
    slot: usize,
}

impl RecordArena {
    fn get(&self, id: &PaintId) -> Option<&OwnedPaint> {
        self.slot(id).map(|slot| &self[slot])
    }

    fn slot(&self, id: &PaintId) -> Option<usize> {
        let entries = self.by_entity.get(&id.entity)?;
        let index = entries
            .binary_search_by_key(&(id.family, id.ordinal), |entry| entry.key)
            .ok()?;
        Some(entries[index].slot)
    }

    fn insert(&mut self, owned: OwnedPaint) -> usize {
        let id = owned.id;
        let slot = self.free.pop().unwrap_or(self.slots.len());
        if slot == self.slots.len() {
            self.slots.push(Some(owned));
        } else {
            self.slots[slot] = Some(owned);
        }
        let entries = self.by_entity.entry(id.entity).or_default();
        let index = entries
            .binary_search_by_key(&(id.family, id.ordinal), |entry| entry.key)
            .expect_err("a retained paint identity must have one arena slot");
        entries.insert(
            index,
            EntityRecordSlot {
                key: (id.family, id.ordinal),
                slot,
            },
        );
        slot
    }

    fn remove(&mut self, id: &PaintId) -> Option<(usize, OwnedPaint)> {
        let slot = self.slot(id)?;
        let owned = self.slots[slot]
            .take()
            .expect("record index must point to an occupied arena slot");
        let entity_slots = self
            .by_entity
            .get_mut(&id.entity)
            .expect("occupied record must have an entity index");
        let index = entity_slots
            .binary_search_by_key(&(id.family, id.ordinal), |entry| entry.key)
            .expect("occupied record must have an entity index entry");
        debug_assert_eq!(entity_slots[index].slot, slot);
        entity_slots.remove(index);
        if entity_slots.is_empty() {
            self.by_entity.remove(&id.entity);
        }
        self.free.push(slot);
        Some((slot, owned))
    }

    fn iter(&self) -> impl Iterator<Item = (usize, &OwnedPaint)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(slot, owned)| owned.as_ref().map(|owned| (slot, owned)))
    }
}

impl core::ops::Index<usize> for RecordArena {
    type Output = OwnedPaint;

    fn index(&self, slot: usize) -> &Self::Output {
        self.slots[slot]
            .as_ref()
            .expect("retained paint slot must be occupied")
    }
}

impl core::ops::IndexMut<usize> for RecordArena {
    fn index_mut(&mut self, slot: usize) -> &mut Self::Output {
        self.slots[slot]
            .as_mut()
            .expect("retained paint slot must be occupied")
    }
}

#[derive(Default)]
struct PaintOrder {
    slots: Vec<usize>,
    groups: Vec<core::ops::Range<usize>>,
    spatial: Option<SpatialIndex<usize>>,
    spatial_revision: u64,
    bounds_dirty: Vec<usize>,
    bounds_dirty_marks: Vec<bool>,
    bounds_updates: Vec<(usize, PhysicalRect)>,
    direct_groups: Vec<usize>,
    direct_epochs: Vec<[u64; 2]>,
    direct_damage: Vec<Vec<DirectDamageEvent>>,
    latest_direct_epoch: u64,
    acknowledged_direct_epoch: u64,
    candidates: Vec<usize>,
    cached_candidate_revision: u64,
    cached_candidate_repair: Option<RepairPlan>,
    cached_direct_groups: Vec<usize>,
    candidate_marks: Vec<u32>,
    candidate_generation: u32,
    compositor: Option<OrderedCompositor>,
    volatile: bool,
    order_dirty: bool,
}

#[derive(Clone, Copy)]
struct DirectDamageEvent {
    epoch: u64,
    rect: PhysicalRect,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum PaintRunKey {
    Before(PaintId),
    Tail,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum OrderedCompositorEntry {
    PaintRun {
        key: PaintRunKey,
        positions: core::ops::Range<usize>,
    },
    Boundary {
        position: usize,
    },
}

struct PaintRunState {
    view_entity: Entity,
    retained_view_entity: RetainedViewEntity,
    bounds: PhysicalRect,
    records: Vec<PaintId>,
    damage: DamageJournal,
}

struct OrderedCompositor {
    entries: Vec<OrderedCompositorEntry>,
    runs: HashMap<PaintRunKey, PaintRunState>,
    group_runs: Vec<Option<PaintRunKey>>,
    group_entries: Vec<usize>,
}

fn ordered_compositor_entries(
    records: &RecordArena,
    slots: &[usize],
) -> Option<Vec<OrderedCompositorEntry>> {
    let mut entries = Vec::new();
    let mut run_start = 0;
    for (position, &slot) in slots.iter().enumerate() {
        let owned = &records[slot];
        if !matches!(owned.record.value.draw.item, RetainedDrawItem::Boundary(_)) {
            continue;
        }
        if run_start < position {
            entries.push(OrderedCompositorEntry::PaintRun {
                key: PaintRunKey::Before(owned.id),
                positions: run_start..position,
            });
        }
        entries.push(OrderedCompositorEntry::Boundary { position });
        run_start = position + 1;
    }
    if entries.is_empty() {
        return None;
    }
    if run_start < slots.len() {
        entries.push(OrderedCompositorEntry::PaintRun {
            key: PaintRunKey::Tail,
            positions: run_start..slots.len(),
        });
    }
    Some(entries)
}

fn rebuild_ordered_compositor(
    commands: &mut Commands,
    parent: Entity,
    records: &RecordArena,
    slots: &[usize],
    previous: Option<OrderedCompositor>,
    run_by_view: &mut EntityHashMap<(Entity, PaintRunKey)>,
    created_run_views: &mut Vec<RetainedViewEntity>,
    retired_run_views: &mut Vec<RetainedViewEntity>,
) -> Option<OrderedCompositor> {
    let Some(entries) = ordered_compositor_entries(records, slots) else {
        if let Some(previous) = previous {
            for run in previous.runs.into_values() {
                run_by_view.remove(&run.view_entity);
                retired_run_views.push(run.retained_view_entity);
                commands.entity(run.view_entity).despawn();
            }
        }
        return None;
    };

    let mut previous_runs = previous.map_or_else(HashMap::default, |state| state.runs);
    let mut runs = HashMap::default();
    let mut group_runs = vec![None; slots.len()];
    for entry in &entries {
        let OrderedCompositorEntry::PaintRun { key, positions } = entry else {
            continue;
        };
        let bounds = positions
            .clone()
            .flat_map(|position| records[slots[position]].record.coverage.iter().copied())
            .reduce(enclosing_rect)
            .expect("a retained paint run contains visible records");
        let record_ids: Vec<_> = positions
            .clone()
            .map(|position| records[slots[position]].id)
            .collect();
        let mut run = previous_runs.remove(key).unwrap_or_else(|| {
            let view_entity = commands.spawn_empty().id();
            let run = PaintRunState {
                view_entity,
                retained_view_entity: RetainedViewEntity::new(view_entity.into(), None, 3),
                bounds,
                records: Vec::new(),
                damage: DamageJournal::default(),
            };
            run_by_view.insert(view_entity, (parent, *key));
            created_run_views.push(run.retained_view_entity);
            run
        });
        if run.records != record_ids {
            let old_records: HashSet<_> = run.records.iter().copied().collect();
            let new_records: HashSet<_> = record_ids.iter().copied().collect();
            for id in old_records.symmetric_difference(&new_records) {
                if let Some(record) = records.get(id) {
                    for &region in &record.record.coverage {
                        run.damage.record(region);
                    }
                }
            }
        }
        run.bounds = enclosing_rect(run.bounds, bounds);
        run.records = record_ids;
        for position in positions.clone() {
            group_runs[position] = Some(*key);
        }
        runs.insert(*key, run);
    }
    for run in previous_runs.into_values() {
        run_by_view.remove(&run.view_entity);
        retired_run_views.push(run.retained_view_entity);
        commands.entity(run.view_entity).despawn();
    }
    let mut group_entries = vec![usize::MAX; slots.len()];
    for (entry_index, entry) in entries.iter().enumerate() {
        match entry {
            OrderedCompositorEntry::PaintRun { positions, .. } => {
                group_entries[positions.clone()].fill(entry_index);
            }
            OrderedCompositorEntry::Boundary { position } => {
                group_entries[*position] = entry_index;
            }
        }
    }
    debug_assert!(group_entries.iter().all(|entry| *entry != usize::MAX));
    Some(OrderedCompositor {
        entries,
        runs,
        group_runs,
        group_entries,
    })
}

impl PaintOrder {
    fn note_bounds_dirty(&mut self, group: usize) {
        if !self.bounds_dirty_marks[group] {
            self.bounds_dirty_marks[group] = true;
            self.bounds_dirty.push(group);
        }
    }

    fn note_direct(&mut self, group: Option<usize>, epoch: u64) {
        let Some(group) = group else {
            return;
        };
        self.latest_direct_epoch = self.latest_direct_epoch.max(epoch);
        match &mut self.direct_epochs[group] {
            [_, latest] if *latest > self.acknowledged_direct_epoch => {
                debug_assert!(*latest <= epoch);
                *latest = epoch;
            }
            slot => {
                *slot = [epoch, epoch];
                self.direct_groups.push(group);
            }
        }
    }

    fn note_source_damage(
        &mut self,
        group: Option<usize>,
        damage: impl IntoIterator<Item = PhysicalRect>,
    ) {
        let Some(compositor) = &mut self.compositor else {
            return;
        };
        let Some(key) = group.and_then(|group| compositor.group_runs[group]) else {
            return;
        };
        let run = compositor
            .runs
            .get_mut(&key)
            .expect("a compositor group must name a retained paint run");
        for region in damage {
            run.bounds = enclosing_rect(run.bounds, region);
            run.damage.record(region);
        }
    }

    fn note_direct_damage(
        &mut self,
        group: Option<usize>,
        epoch: u64,
        damage: impl IntoIterator<Item = PhysicalRect>,
    ) {
        self.note_direct(group, epoch);
        let Some(group) = group else {
            return;
        };
        self.direct_damage[group].extend(
            damage
                .into_iter()
                .map(|rect| DirectDamageEvent { epoch, rect }),
        );
    }

    fn acknowledge(&mut self, through_epoch: u64) {
        self.acknowledged_direct_epoch = self.acknowledged_direct_epoch.max(through_epoch);
        for &group in &self.direct_groups {
            self.direct_damage[group].retain(|event| event.epoch > through_epoch);
        }
        if self.latest_direct_epoch <= through_epoch {
            self.direct_groups.clear();
            return;
        }
        self.direct_groups.retain(|&group| {
            let [earliest, latest] = self.direct_epochs[group];
            debug_assert!(latest > self.acknowledged_direct_epoch);
            if earliest > through_epoch {
                return true;
            }
            if latest > through_epoch {
                self.direct_epochs[group] = [latest, latest];
                true
            } else {
                false
            }
        });
    }
}

impl RetainedUiSurfaces {
    fn localize_transform(&self, camera: Entity, transform: Affine2) -> Affine2 {
        self.surface_origins
            .get(&camera)
            .map_or(transform, |origin| {
                Affine2::from_translation(-*origin) * transform
            })
    }

    fn localize_clip(&self, camera: Entity, clip: Option<Rect>) -> Option<Rect> {
        let Some(origin) = self.surface_origins.get(&camera) else {
            return clip;
        };
        clip.map(|clip| Rect::from_corners(clip.min - *origin, clip.max - *origin))
    }

    fn localize_paint(
        &self,
        camera: Entity,
        draw: &mut RetainedDraw,
        supplied_coverage: &PaintCoverage,
    ) -> PaintCoverage {
        let Some(origin) = self.surface_origins.get(&camera) else {
            return supplied_coverage.clone();
        };
        draw.transform = Affine2::from_translation(-*origin) * draw.transform;
        draw.z_order -= self.surface_z_origins.get(&camera).copied().unwrap_or(0.0);
        draw.clip = draw
            .clip
            .map(|clip| Rect::from_corners(clip.min - *origin, clip.max - *origin));
        refresh_border_coverage(draw);
        if matches!(&draw.item, RetainedDrawItem::Material(material) if material.target_coverage) {
            self.surface_bounds
                .get(&camera)
                .copied()
                .into_iter()
                .collect()
        } else {
            draw_coverage(draw, &PaintCoverage::empty())
        }
    }

    pub(crate) fn set_surface_space(
        &mut self,
        surface: Entity,
        origin: Vec2,
        z_origin: f32,
        bounds: PhysicalRect,
    ) {
        let previous_origin = self.surface_origins.insert(surface, origin);
        let previous_z_origin = self.surface_z_origins.insert(surface, z_origin);
        let previous_bounds = self.surface_bounds.insert(surface, bounds);
        if previous_bounds.is_none() || previous_bounds == Some(bounds) {
            return;
        }

        let offset = previous_origin.unwrap_or(origin) - origin;
        let z_offset = previous_z_origin.unwrap_or(z_origin) - z_origin;
        let slots: Vec<_> = self
            .records
            .iter()
            .filter_map(|(slot, owned)| (owned.camera == surface).then_some(slot))
            .collect();
        for slot in slots {
            let owned = &mut self.records[slot];
            owned.record.value.draw.transform =
                Affine2::from_translation(offset) * owned.record.value.draw.transform;
            owned.record.value.draw.z_order += z_offset;
            owned.order = (
                FloatBits::new(owned.record.value.draw.z_order),
                owned.record.value.draw.paint_order,
            );
            owned.record.value.draw.clip = owned
                .record
                .value
                .draw
                .clip
                .map(|clip| Rect::from_corners(clip.min + offset, clip.max + offset));
            refresh_border_coverage(&mut owned.record.value.draw);
            owned.record.coverage = if matches!(
                &owned.record.value.draw.item,
                RetainedDrawItem::Material(material) if material.target_coverage
            ) {
                bounds.into()
            } else {
                draw_coverage(&owned.record.value.draw, &owned.record.coverage)
            };
            owned.record.value.refresh_prepared();
        }
        let paint = self.paint.entry(surface).or_default();
        paint.invalidate(bounds);
        let order = self.order.entry(surface).or_default();
        order.order_dirty = true;
        order.spatial = None;
        self.dirty(surface);
        self.touch(surface);
    }

    pub(crate) fn set_target_bounds(&mut self, target: Entity, bounds: PhysicalRect) {
        self.surface_bounds.insert(target, bounds);
    }

    fn retire_surface_space_if_unowned(&mut self, surface: Entity) {
        if self
            .surface_by_main_entity
            .values()
            .any(|owned| *owned == surface)
            || self
                .records
                .iter()
                .any(|(_, owned)| owned.camera == surface)
        {
            return;
        }
        self.surface_origins.remove(&surface);
        self.surface_z_origins.remove(&surface);
        self.surface_bounds.remove(&surface);
    }

    fn touch(&mut self, camera: Entity) {
        if self.paint.entry(camera).or_default().list_touched() {
            self.touched_surfaces.push(camera);
        }
    }

    fn dirty(&mut self, camera: Entity) {
        if self.paint.entry(camera).or_default().list_dirty() {
            self.dirty_surfaces.push(camera);
        }
    }

    fn clear_dirty(&mut self, camera: Entity) {
        if let Some(paint) = self.paint.get_mut(&camera) {
            paint.clear_dirty_listing();
        }
        self.dirty_surfaces.retain(|listed| *listed != camera);
    }

    fn invalidate_compositor_sources(&mut self, camera: Entity, damage: PhysicalRect) {
        let Some(compositor) = self
            .order
            .get_mut(&camera)
            .and_then(|order| order.compositor.as_mut())
        else {
            return;
        };
        for run in compositor.runs.values_mut() {
            for id in &run.records {
                let Some(record) = self.records.get(id) else {
                    continue;
                };
                for region in record
                    .record
                    .coverage
                    .iter()
                    .filter_map(|region| region.intersection(damage))
                {
                    run.damage.record(region);
                }
            }
        }
    }

    fn set_volatile(&mut self, camera: Entity, volatile: bool) {
        let order = self.order.entry(camera).or_default();
        if order.volatile != volatile {
            order.volatile = volatile;
            order.order_dirty = true;
            order.spatial = None;
        }
    }

    fn set_entity_surface(&mut self, entity: MainEntity, surface: Option<Entity>, camera: Entity) {
        let target = surface.unwrap_or(camera);
        let previous_surface = self.surface_by_main_entity.get(&entity).copied();
        match surface {
            Some(surface) => {
                self.surface_by_main_entity.insert(entity, surface);
            }
            None => {
                self.surface_by_main_entity.remove(&entity);
            }
        }
        let slots = self
            .by_main_entity
            .get(&entity)
            .cloned()
            .unwrap_or_default();
        for slot in slots {
            let old_camera = self.records[slot].camera;
            if old_camera == target {
                continue;
            }
            let previous_coverage = self.records[slot].record.coverage.clone();
            self.paint
                .get_mut(&old_camera)
                .expect("owned retained paint camera must exist")
                .remove(&previous_coverage);
            let old_origin = self
                .surface_origins
                .get(&old_camera)
                .copied()
                .unwrap_or(Vec2::ZERO);
            let target_origin = self
                .surface_origins
                .get(&target)
                .copied()
                .unwrap_or(Vec2::ZERO);
            let old_z_origin = self
                .surface_z_origins
                .get(&old_camera)
                .copied()
                .unwrap_or(0.0);
            let target_z_origin = self.surface_z_origins.get(&target).copied().unwrap_or(0.0);
            let offset = old_origin - target_origin;
            let owned = &mut self.records[slot];
            owned.camera = target;
            owned.group = None;
            owned.record.value.draw.camera = target;
            owned.record.value.draw.transform =
                Affine2::from_translation(offset) * owned.record.value.draw.transform;
            owned.record.value.draw.z_order += old_z_origin - target_z_origin;
            owned.order = (
                FloatBits::new(owned.record.value.draw.z_order),
                owned.record.value.draw.paint_order,
            );
            owned.record.value.draw.clip = owned
                .record
                .value
                .draw
                .clip
                .map(|clip| Rect::from_corners(clip.min + offset, clip.max + offset));
            refresh_border_coverage(&mut owned.record.value.draw);
            owned.record.coverage = if matches!(
                &owned.record.value.draw.item,
                RetainedDrawItem::Material(material) if material.target_coverage
            ) {
                self.surface_bounds
                    .get(&target)
                    .copied()
                    .into_iter()
                    .collect()
            } else {
                draw_coverage(&owned.record.value.draw, &owned.record.coverage)
            };
            owned.record.value.refresh_prepared();
            self.paint
                .entry(target)
                .or_default()
                .upsert(None, &owned.record);
            self.dirty(old_camera);
            self.dirty(target);
            self.touch(old_camera);
            self.touch(target);
            for camera in [old_camera, target] {
                let order = self.order.entry(camera).or_default();
                order.order_dirty = true;
                order.spatial = None;
            }
        }
        if let Some(previous_surface) = previous_surface.filter(|old| Some(*old) != surface) {
            self.retire_surface_space_if_unowned(previous_surface);
        }
    }

    fn remove_main_entity(&mut self, commands: &mut Commands, entity: MainEntity) {
        let Some(slots) = self.by_main_entity.get(&entity).cloned() else {
            self.surface_by_main_entity.remove(&entity);
            return;
        };
        for slot in slots {
            let id = self.records[slot].id;
            self.remove(commands, id);
        }
        self.surface_by_main_entity.remove(&entity);
    }

    pub(crate) fn core_draw(&self, id: PaintId) -> Option<&RetainedDraw> {
        Some(&self.records.get(&id)?.record.value.draw)
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
            let Some(slot) = self.records.slot(&id) else {
                continue;
            };
            let camera = self.records[slot].camera;
            let group = self.records[slot].group;
            self.touch(camera);
            let owned = &mut self.records[slot];
            let paint = self
                .paint
                .get_mut(&camera)
                .expect("owned text camera must exist");
            let changed = owned.record.value.retint_glyphs(color);
            let outcome = if changed {
                paint.update(&owned.record.coverage, &owned.record.coverage)
            } else {
                paint.unchanged()
            };
            if outcome != crate::UpdateOutcome::Unchanged {
                let epoch = paint.latest_damage_epoch();
                let damage = owned.record.coverage.clone();
                let order = self.order.entry(camera).or_default();
                order.note_direct_damage(group, epoch, damage.iter().copied());
                order.note_source_damage(group, damage.iter().copied());
                self.dirty(camera);
            }
        }
    }

    pub(crate) fn retint_owned_node(
        &mut self,
        id: PaintId,
        color: bevy::color::LinearRgba,
    ) -> bool {
        let Some(slot) = self.records.slot(&id) else {
            return false;
        };
        let camera = self.records[slot].camera;
        let group = self.records[slot].group;
        self.touch(camera);
        let owned = &mut self.records[slot];
        let paint = self
            .paint
            .get_mut(&camera)
            .expect("owned node camera must exist");
        let RetainedDrawItem::Node(node) = &mut owned.record.value.draw.item else {
            return false;
        };
        let changed = node.color != color;
        if changed {
            node.color = color;
            if let Some(PreparedDraw::Core(PreparedCore::One(instance))) =
                &mut owned.record.value.prepared
            {
                instance.set_color(color);
            }
        }
        let outcome = if changed {
            paint.update(&owned.record.coverage, &owned.record.coverage)
        } else {
            paint.unchanged()
        };
        if outcome != crate::UpdateOutcome::Unchanged {
            let epoch = paint.latest_damage_epoch();
            let damage = owned.record.coverage.clone();
            let order = self.order.entry(camera).or_default();
            order.note_direct_damage(group, epoch, damage.iter().copied());
            order.note_source_damage(group, damage.iter().copied());
            self.dirty(camera);
        }
        true
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "image layout has independent box, transform, clip, and scale outputs"
    )]
    pub(crate) fn relayout_image(
        &mut self,
        commands: &mut Commands,
        entity: Entity,
        size: Vec2,
        center: Vec2,
        source_transform: Affine2,
        clip: Option<Rect>,
        border_radius: ResolvedBorderRadius,
        inverse_scale_factor: f32,
        painted: bool,
    ) -> bool {
        let id = PaintId {
            entity,
            family: PaintFamily::Image,
            ordinal: 0,
        };
        let transform = source_transform * Affine2::from_translation(center);
        self.update_geometry(
            commands,
            id,
            transform,
            clip,
            center,
            painted,
            |item| match item {
                RetainedDrawItem::Node(node) => {
                    let old_rect = rect_fingerprint(node.rect);
                    let old_scaling = node
                        .atlas_scaling
                        .map(|value| value.to_array().map(FloatBits::new));
                    let old_radius = <[f32; 4]>::from(node.border_radius).map(FloatBits::new);
                    if let Some(old_scale) = node.atlas_scaling {
                        let source = Rect::from_corners(
                            node.rect.min / old_scale,
                            node.rect.max / old_scale,
                        );
                        let scale = size / source.size();
                        node.rect = Rect::from_corners(source.min * scale, source.max * scale);
                        node.atlas_scaling = Some(scale);
                    } else {
                        node.rect = Rect::from_corners(Vec2::ZERO, size);
                    }
                    node.border_radius = border_radius;
                    Some(
                        old_rect != rect_fingerprint(node.rect)
                            || old_scaling
                                != node
                                    .atlas_scaling
                                    .map(|value| value.to_array().map(FloatBits::new))
                            || old_radius
                                != <[f32; 4]>::from(node.border_radius).map(FloatBits::new),
                    )
                }
                RetainedDrawItem::TextureSlice(slice) => {
                    let old_rect = rect_fingerprint(slice.rect);
                    let old_scale = FloatBits::new(slice.inverse_scale_factor);
                    slice.rect = Rect::from_corners(Vec2::ZERO, size);
                    slice.inverse_scale_factor = inverse_scale_factor;
                    Some(
                        old_rect != rect_fingerprint(slice.rect)
                            || old_scale != FloatBits::new(slice.inverse_scale_factor),
                    )
                }
                _ => None,
            },
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "canonical geometry has independent identity, placement, visibility, and item data"
    )]
    fn update_geometry(
        &mut self,
        commands: &mut Commands,
        id: PaintId,
        transform: Affine2,
        clip: Option<Rect>,
        local_translation: Vec2,
        painted: bool,
        update_item: impl FnOnce(&mut RetainedDrawItem) -> Option<bool>,
    ) -> bool {
        if !painted {
            self.remove(commands, id);
            return true;
        }
        let Some(slot) = self.records.slot(&id) else {
            return false;
        };
        let camera = self.records[slot].camera;
        let transform = self.localize_transform(camera, transform);
        let clip = self.localize_clip(camera, clip);
        let group = self.records[slot].group;
        self.touch(camera);

        let owned = &mut self.records[slot];
        let draw = &mut owned.record.value.draw;
        let placement_changed = affine_bits(draw.transform) != affine_bits(transform)
            || draw.clip.map(rect_fingerprint) != clip.map(rect_fingerprint)
            || draw.local_translation.to_array().map(FloatBits::new)
                != local_translation.to_array().map(FloatBits::new);
        let Some(item_changed) = update_item(&mut draw.item) else {
            return false;
        };
        draw.transform = transform;
        draw.clip = clip;
        draw.local_translation = local_translation;
        if !placement_changed && !item_changed {
            self.paint
                .get_mut(&camera)
                .expect("owned geometry camera must have paint state")
                .unchanged();
            return true;
        }

        let previous_coverage = owned.record.coverage.clone();
        let coverage = draw_coverage(draw, &previous_coverage);
        let visibility_changed = owned.record.coverage.is_empty() != coverage.is_empty();
        let paint = self
            .paint
            .get_mut(&camera)
            .expect("owned geometry camera must have paint state");
        let outcome = paint.update(&owned.record.coverage, &coverage);
        owned.record.coverage = coverage.clone();
        owned.record.value.refresh_prepared();
        let epoch = paint.latest_damage_epoch();
        let list_dirty = paint.list_dirty();
        let order = self.order.entry(camera).or_default();
        order.note_direct_damage(
            group,
            epoch,
            previous_coverage.iter().chain(&coverage).copied(),
        );
        order.note_source_damage(group, previous_coverage.iter().chain(&coverage).copied());
        if visibility_changed {
            order.order_dirty = true;
            order.spatial = None;
        } else if outcome.coverage_changed() {
            if let Some(group) = group
                && order.spatial.is_some()
                && !order.order_dirty
            {
                order.note_bounds_dirty(group);
            } else {
                order.spatial = None;
            }
        }
        if list_dirty {
            self.dirty_surfaces.push(camera);
        }
        true
    }

    fn reposition(&mut self, entity: MainEntity, transform: Affine2, clip: Option<Rect>) {
        self.reposition_layout(entity, transform, clip, None);
    }

    pub(crate) fn reposition_layout(
        &mut self,
        entity: MainEntity,
        transform: Affine2,
        clip: Option<Rect>,
        layout_translation: Option<Vec2>,
    ) {
        let Some(slots) = self.by_main_entity.get(&entity).cloned() else {
            return;
        };
        let camera = self.records[slots[0]].camera;
        let transform = self.localize_transform(camera, transform);
        let clip = self.localize_clip(camera, clip);
        self.touch(camera);
        let paint = self
            .paint
            .get_mut(&camera)
            .expect("owned paint camera must exist");
        let order = self.order.entry(camera).or_default();
        let mut changed = false;
        for slot in slots {
            let owned = &mut self.records[slot];
            debug_assert_eq!(owned.camera, camera);
            if let Some(layout_translation) = layout_translation {
                owned.record.value.draw.layout_translation = layout_translation;
            }
            let group = owned.group;
            let Some(coverage) =
                owned
                    .record
                    .value
                    .reposition(transform, clip, &owned.record.coverage)
            else {
                paint.unchanged();
                continue;
            };
            let previous_coverage = owned.record.coverage.clone();
            let visibility_changed = owned.record.coverage.is_empty() != coverage.is_empty();
            let outcome = paint.update(&owned.record.coverage, &coverage);
            owned.record.coverage = coverage.clone();
            let epoch = paint.latest_damage_epoch();
            changed = true;
            order.note_direct_damage(
                group,
                epoch,
                previous_coverage.iter().chain(&coverage).copied(),
            );
            order.note_source_damage(group, previous_coverage.iter().chain(&coverage).copied());
            if visibility_changed {
                order.order_dirty = true;
                order.spatial = None;
            } else if outcome.coverage_changed() {
                if let Some(group) = group
                    && order.spatial.is_some()
                    && !order.order_dirty
                {
                    order.note_bounds_dirty(group);
                } else {
                    order.spatial = None;
                }
            }
        }
        if changed {
            self.dirty(camera);
        }
    }

    pub(crate) fn remove(&mut self, commands: &mut Commands, id: PaintId) {
        let Some(owned) = self.detach_record(id) else {
            return;
        };

        self.paint
            .get_mut(&owned.camera)
            .expect("owned paint camera must exist")
            .remove(&owned.record.coverage);
        self.dirty(owned.camera);
        self.touch(owned.camera);
        let index = self.order.entry(owned.camera).or_default();
        index.note_source_damage(owned.group, owned.record.coverage.iter().copied());
        index.order_dirty = true;
        index.spatial = None;
        if let Ok(mut entity) = commands.get_entity(owned.record.value.render_entity) {
            entity.despawn();
        }
    }

    fn detach_record(&mut self, id: PaintId) -> Option<OwnedPaint> {
        let (slot, owned) = self.records.remove(&id)?;
        if let Some(main_entity) = owned.main_entity
            && let Some(slots) = self.by_main_entity.get_mut(&main_entity)
        {
            slots.retain(|candidate| *candidate != slot);
            if slots.is_empty() {
                self.by_main_entity.remove(&main_entity);
            }
        }
        Some(owned)
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
        let coverage = self.localize_paint(camera, &mut draw, &coverage);
        let main_entity = draw.main_entity;
        if self.can_update(id, camera, &draw, &resource, Some(main_entity)) {
            self.update_existing(id, camera, draw, coverage);
            return;
        }
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

    fn can_update(
        &self,
        id: PaintId,
        camera: Entity,
        draw: &RetainedDraw,
        resource: &ResourceFingerprint,
        index_owner: Option<MainEntity>,
    ) -> bool {
        let Some(slot) = self.records.slot(&id) else {
            return false;
        };
        let owned = &self.records[slot];
        let previous = &owned.record.value;
        let order = (FloatBits::new(draw.z_order), draw.paint_order);
        if owned.camera != camera
            || owned.main_entity != index_owner
            || owned.order != order
            || previous.resource != *resource
            || previous.draw.main_entity != draw.main_entity
            || previous.draw.image != draw.image
            || previous
                .draw
                .layout_translation
                .to_array()
                .map(FloatBits::new)
                != draw.layout_translation.to_array().map(FloatBits::new)
        {
            return false;
        }
        true
    }

    fn update_existing(
        &mut self,
        id: PaintId,
        camera: Entity,
        mut draw: RetainedDraw,
        coverage: PaintCoverage,
    ) {
        let slot = self
            .records
            .slot(&id)
            .expect("an updateable paint must already exist");
        let previous = &self.records[slot].record.value;
        let previous_coverage = self.records[slot].record.coverage.clone();
        let same_common = retained_common_eq(&previous.draw, &draw);
        let same_pixels = same_common && retained_item_eq(&previous.draw.item, &draw.item);
        let same_coverage = self.records[slot].record.coverage == coverage;
        let exact_border_damage = match (&previous.draw.item, &draw.item) {
            (RetainedDrawItem::Border(previous_border), RetainedDrawItem::Border(border)) => Some(
                changed_border_damage(&previous.draw, previous_border, &draw, border),
            ),
            _ => None,
        };
        let color_only_node_change = same_common
            && same_coverage
            && matches!(
                (&previous.draw.item, &draw.item),
                (RetainedDrawItem::Node(previous), RetainedDrawItem::Node(next))
                    if retained_node_geometry_eq(previous, next)
            );
        self.records[slot].record.value.draw.local_translation = draw.local_translation;
        let paint = self
            .paint
            .get_mut(&camera)
            .expect("owned paint camera must have paint state");
        if paint.list_touched() {
            self.touched_surfaces.push(camera);
        }
        if same_pixels && same_coverage {
            paint.unchanged();
            return;
        }

        let owned = &mut self.records[slot];
        let visibility_changed = owned.record.coverage.is_empty() != coverage.is_empty();
        let outcome = if let Some(damage) = &exact_border_damage {
            paint.update_exact(&previous_coverage, &coverage, damage)
        } else {
            paint.update(&previous_coverage, &coverage)
        };
        owned.record.coverage = coverage.clone();
        draw.render_entity = owned.record.value.render_entity;
        let color = color_only_node_change.then(|| match &draw.item {
            RetainedDrawItem::Node(node) => node.color,
            _ => unreachable!("a color-only node change must contain a node"),
        });
        owned.record.value.draw = draw;
        let group = owned.group;
        let epoch = paint.latest_damage_epoch();
        let list_dirty = paint.list_dirty();
        if let Some(color) = color {
            if let Some(PreparedDraw::Core(PreparedCore::One(instance))) =
                &mut owned.record.value.prepared
            {
                instance.set_color(color);
            } else {
                owned.record.value.refresh_prepared();
            }
        } else {
            owned.record.value.refresh_prepared();
        }
        let order = self.order.entry(camera).or_default();
        if let Some(damage) = &exact_border_damage {
            order.note_direct_damage(group, epoch, damage.iter().copied());
        } else {
            order.note_direct_damage(
                group,
                epoch,
                previous_coverage.iter().chain(&coverage).copied(),
            );
        }
        if let Some(damage) = exact_border_damage {
            order.note_source_damage(group, damage.iter().copied());
        } else {
            order.note_source_damage(group, previous_coverage.iter().chain(&coverage).copied());
        }
        if visibility_changed {
            order.order_dirty = true;
            order.spatial = None;
        } else if outcome.coverage_changed() {
            if let Some(group) = group
                && order.spatial.is_some()
                && !order.order_dirty
            {
                order.note_bounds_dirty(group);
            } else {
                order.spatial = None;
            }
        }
        if list_dirty {
            self.dirty_surfaces.push(camera);
        }
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
        let coverage = self.localize_paint(camera, &mut draw, &coverage);
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
        let new_order = (FloatBits::new(draw.z_order), draw.paint_order);
        let existing_slot = self.records.slot(&id);
        let existing = existing_slot.map(|slot| &self.records[slot]);
        let old_camera = existing.map(|owned| owned.camera);
        let old_order = existing.map(|owned| owned.order);
        let old_index_owner = existing.and_then(|owned| owned.main_entity);
        let previous_coverage = existing
            .map(|owned| owned.record.coverage.clone())
            .unwrap_or_default();
        let render_entity = existing.map_or_else(
            || commands.spawn_empty().id(),
            |owned| owned.record.value.render_entity,
        );
        if old_index_owner != index_owner
            && let Some(old_main_entity) = old_index_owner
            && let Some(slots) = self.by_main_entity.get_mut(&old_main_entity)
        {
            slots.retain(|candidate| Some(*candidate) != existing_slot);
            if slots.is_empty() {
                self.by_main_entity.remove(&old_main_entity);
            }
        }
        draw.render_entity = render_entity;
        let mut record = PaintRecord {
            coverage,
            value: RetainedRecord::new(draw, resource),
        };
        let visibility_changed = existing
            .is_some_and(|owned| owned.record.coverage.is_empty() != record.coverage.is_empty());
        let previous_group = existing.and_then(|owned| owned.group);
        let (outcome, epoch, list_dirty) =
            if let Some(old_camera) = old_camera.filter(|old| *old != camera) {
                let previous = &self.records[existing_slot.expect("moved paint must exist")].record;
                self.paint
                    .get_mut(&old_camera)
                    .expect("owned paint camera must exist")
                    .remove(&previous.coverage);
                self.dirty(old_camera);
                self.touch(old_camera);
                let index = self.order.entry(old_camera).or_default();
                index.note_source_damage(previous_group, previous_coverage.iter().copied());
                index.order_dirty = true;
                index.spatial = None;
                let paint = self.paint.entry(camera).or_default();
                if paint.list_touched() {
                    self.touched_surfaces.push(camera);
                }
                let outcome = paint.upsert(None, &record);
                let epoch = paint.latest_damage_epoch();
                let list_dirty = outcome != crate::UpdateOutcome::Unchanged && paint.list_dirty();
                (outcome, epoch, list_dirty)
            } else {
                let paint = self.paint.entry(camera).or_default();
                if paint.list_touched() {
                    self.touched_surfaces.push(camera);
                }
                let outcome = paint.upsert(existing.map(|owned| &owned.record), &record);
                let epoch = paint.latest_damage_epoch();
                let list_dirty = outcome != crate::UpdateOutcome::Unchanged && paint.list_dirty();
                (outcome, epoch, list_dirty)
            };
        if outcome != crate::UpdateOutcome::Unchanged {
            record.value.prepare();
        }
        let slot = if let Some(slot) = existing_slot {
            let owned = &mut self.records[slot];
            owned.camera = camera;
            owned.main_entity = index_owner;
            owned.order = new_order;
            if old_camera != Some(camera) || old_order != Some(new_order) {
                owned.group = None;
            }
            if outcome != crate::UpdateOutcome::Unchanged || old_camera != Some(camera) {
                owned.record = record;
            } else {
                owned.record.value.draw.layout_translation = record.value.draw.layout_translation;
                owned.record.value.draw.local_translation = record.value.draw.local_translation;
            }
            slot
        } else {
            self.records.insert(OwnedPaint {
                id,
                camera,
                main_entity: index_owner,
                order: new_order,
                group: None,
                record,
            })
        };
        if old_index_owner != index_owner
            && let Some(index_owner) = index_owner
        {
            self.by_main_entity
                .entry(index_owner)
                .or_default()
                .push(slot);
        }
        if outcome != crate::UpdateOutcome::Unchanged {
            if list_dirty {
                self.dirty_surfaces.push(camera);
            }
            let group = self.records[slot].group;
            let current_coverage = &self.records[slot].record.coverage;
            let order = self.order.entry(camera).or_default();
            order.note_direct_damage(
                group,
                epoch,
                previous_coverage.iter().chain(current_coverage).copied(),
            );
            order.note_source_damage(
                group,
                previous_coverage.iter().chain(current_coverage).copied(),
            );
        }
        let coverage_changed = old_camera != Some(camera) || outcome.coverage_changed();
        if old_camera != Some(camera) || old_order != Some(new_order) || visibility_changed {
            let index = self.order.entry(camera).or_default();
            index.order_dirty = true;
            index.spatial = None;
        }
        if coverage_changed {
            let index = self.order.entry(camera).or_default();
            if let Some(group) = previous_group
                && index.spatial.is_some()
                && !index.order_dirty
            {
                index.note_bounds_dirty(group);
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
            let Some(&(parent, key)) = surfaces.run_by_view.get(&camera) else {
                return;
            };
            if let Some(run) = surfaces
                .order
                .get_mut(&parent)
                .and_then(|order| order.compositor.as_mut())
                .and_then(|compositor| compositor.runs.get_mut(&key))
            {
                run.damage.acknowledge(plan);
            }
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
            surfaces.clear_dirty(camera);
        }
    }

    pub(crate) fn has_visible_records(&self, camera: Entity, target: PhysicalRect) -> bool {
        self.has_visible_records_in(camera, core::slice::from_ref(&target))
    }

    pub(crate) fn has_visible_records_in(&self, camera: Entity, targets: &[PhysicalRect]) -> bool {
        let surfaces = self.lock();
        if let Some(&(parent, key)) = surfaces.run_by_view.get(&camera)
            && let Some(run) = surfaces
                .order
                .get(&parent)
                .and_then(|order| order.compositor.as_ref())
                .and_then(|compositor| compositor.runs.get(&key))
        {
            return run
                .records
                .iter()
                .filter_map(|id| surfaces.records.get(id))
                .flat_map(|record| record.record.coverage.iter())
                .any(|coverage| {
                    targets
                        .iter()
                        .any(|target| coverage.intersection(*target).is_some())
                });
        }
        if let Some(order) = surfaces.order.get(&camera)
            && let Some(spatial) = &order.spatial
        {
            for &target in targets {
                let mut intersects = false;
                spatial.query(target, |group| {
                    if !intersects {
                        intersects = group_intersects(
                            &surfaces.records,
                            &order.slots,
                            order.groups[group].clone(),
                            target,
                        );
                    }
                });
                if !intersects {
                    // Direct damage can make a group drawable before its
                    // changed bounds have settled into the spatial index.
                    intersects = order.direct_groups.iter().any(|&group| {
                        group_intersects(
                            &surfaces.records,
                            &order.slots,
                            order.groups[group].clone(),
                            target,
                        )
                    });
                }
                if intersects {
                    return true;
                }
            }
            return false;
        }
        surfaces.records.iter().any(|(_, owned)| {
            owned.camera == camera
                && owned.record.coverage.iter().any(|coverage| {
                    targets
                        .iter()
                        .any(|target| coverage.intersection(*target).is_some())
                })
        })
    }

    pub(crate) fn compositor(
        &self,
        camera: Entity,
        damage: &RepairPlan,
    ) -> Option<RetainedCompositor> {
        let surfaces = self.lock();
        let order = surfaces.order.get(&camera)?;
        let compositor = order.compositor.as_ref()?;
        let spatial = order.spatial.as_ref()?;
        let mut candidates: HashMap<usize, PhysicalRect> = HashMap::default();
        for &region in damage.regions() {
            spatial.query(region, |group| {
                if group_intersects(
                    &surfaces.records,
                    &order.slots,
                    order.groups[group].clone(),
                    region,
                ) {
                    let bounds =
                        group_bounds(&surfaces.records, &order.slots, order.groups[group].clone())
                            .intersection(region)
                            .expect("an intersecting paint group has nonempty repair bounds");
                    candidates
                        .entry(compositor.group_entries[group])
                        .and_modify(|repair| *repair = enclosing_rect(*repair, bounds))
                        .or_insert(bounds);
                }
            });
        }
        for &group in &order.direct_groups {
            if order.direct_epochs[group][0] > damage.through_epoch() {
                continue;
            }
            for event in order.direct_damage[group]
                .iter()
                .filter(|event| event.epoch <= damage.through_epoch())
            {
                if !group_intersects(
                    &surfaces.records,
                    &order.slots,
                    order.groups[group].clone(),
                    event.rect,
                ) {
                    continue;
                }
                let bounds =
                    group_bounds(&surfaces.records, &order.slots, order.groups[group].clone())
                        .intersection(event.rect)
                        .expect("direct paint damage intersects its current group");
                candidates
                    .entry(compositor.group_entries[group])
                    .and_modify(|repair| *repair = enclosing_rect(*repair, bounds))
                    .or_insert(bounds);
            }
        }
        let mut candidates: Vec<_> = candidates.into_iter().collect();
        candidates.sort_unstable_by_key(|(entry, _)| *entry);
        let mut entries = Vec::with_capacity(candidates.len());
        for (index, repair_bounds) in candidates {
            let entry = &compositor.entries[index];
            match entry {
                OrderedCompositorEntry::PaintRun { key, .. } => {
                    let run = &compositor.runs[key];
                    entries.push(RetainedCompositorEntry::PaintRun(RetainedPaintRun {
                        parent: camera,
                        view_entity: run.view_entity,
                        retained_view_entity: run.retained_view_entity,
                        bounds: run.bounds,
                        repair_bounds,
                    }));
                }
                OrderedCompositorEntry::Boundary { position } => {
                    let record = &surfaces.records[order.slots[*position]].record;
                    let RetainedDrawItem::Boundary(boundary) = &record.value.draw.item else {
                        unreachable!("a compositor boundary entry must point to a boundary record")
                    };
                    entries.push(RetainedCompositorEntry::Boundary {
                        draw: RetainedBoundaryDraw {
                            render_entity: record.value.draw.render_entity,
                            main_entity: record.value.draw.main_entity,
                            camera: record.value.draw.camera,
                            z_order: record.value.draw.z_order,
                            surface: boundary.surface,
                            transform: record.value.draw.transform,
                            size: boundary.size(),
                            opacity: boundary.opacity(),
                        },
                        repair_bounds,
                    });
                }
            }
        }
        Some(RetainedCompositor { entries })
    }

    pub(crate) fn take_run_view_changes(
        &self,
    ) -> (Vec<RetainedViewEntity>, Vec<RetainedViewEntity>) {
        let mut surfaces = self.lock();
        (
            core::mem::take(&mut surfaces.created_run_views),
            core::mem::take(&mut surfaces.retired_run_views),
        )
    }

    pub(crate) fn dirty_paint_runs(&self) -> Vec<RetainedPaintRun> {
        let surfaces = self.lock();
        surfaces
            .dirty_surfaces
            .iter()
            .filter_map(|parent| Some((*parent, surfaces.order.get(parent)?.compositor.as_ref()?)))
            .flat_map(|(parent, compositor)| {
                compositor
                    .runs
                    .values()
                    .filter(|run| !run.damage.is_empty())
                    .map(move |run| RetainedPaintRun {
                        parent,
                        view_entity: run.view_entity,
                        retained_view_entity: run.retained_view_entity,
                        bounds: run.bounds,
                        repair_bounds: run.bounds,
                    })
            })
            .collect()
    }

    pub(crate) fn has_damage(&self, surface: Entity) -> bool {
        let surfaces = self.lock();
        if let Some(paint) = surfaces.paint.get(&surface) {
            return paint.has_damage();
        }
        let Some(&(parent, key)) = surfaces.run_by_view.get(&surface) else {
            return false;
        };
        surfaces
            .order
            .get(&parent)
            .and_then(|order| order.compositor.as_ref())
            .and_then(|compositor| compositor.runs.get(&key))
            .is_some_and(|run| !run.damage.is_empty())
    }

    pub(crate) fn invalidate(&self, camera: Entity, coverage: PhysicalRect) {
        let mut surfaces = self.lock();
        if let Some(paint) = surfaces.paint.get_mut(&camera) {
            paint.invalidate(coverage);
            surfaces.invalidate_compositor_sources(camera, coverage);
            surfaces.dirty(camera);
            surfaces.touch(camera);
        }
    }

    pub(crate) fn invalidate_volatile(&self, camera: Entity, coverage: PhysicalRect) {
        let mut surfaces = self.lock();
        surfaces.set_volatile(camera, true);
        let paint = surfaces.paint.entry(camera).or_default();
        paint.invalidate(coverage);
        surfaces.invalidate_compositor_sources(camera, coverage);
        surfaces.dirty(camera);
        surfaces.touch(camera);
    }

    pub(crate) fn invalidate_departed_volatile(&self, camera: Entity, coverage: PhysicalRect) {
        let mut surfaces = self.lock();
        surfaces.set_volatile(camera, false);
        let paint = surfaces.paint.entry(camera).or_default();
        paint.invalidate(coverage);
        surfaces.invalidate_compositor_sources(camera, coverage);
        surfaces.dirty(camera);
        surfaces.touch(camera);
    }

    pub(crate) fn dirty_surfaces(&self) -> Vec<Entity> {
        self.lock().dirty_surfaces.clone()
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
            surfaces.clear_dirty(camera);
        }
    }

    pub(crate) fn propagate_boundaries(&self, boundaries: &[(Entity, PaintId, PhysicalRect)]) {
        let mut surfaces = self.lock();
        for &(source, id, source_bounds) in boundaries {
            let Some(source_plan) = surfaces
                .paint
                .get_mut(&source)
                .and_then(PaintState::repair_plan)
            else {
                continue;
            };
            let source_epoch = source_plan.through_epoch();
            if surfaces.propagated_epochs.get(&source) == Some(&source_epoch) {
                continue;
            }
            surfaces.propagated_epochs.insert(source, source_epoch);
            let Some(owned) = surfaces.records.get(&id) else {
                continue;
            };
            let parent = owned.camera;
            let group = owned.group;
            let boundary_draw = &owned.record.value.draw;
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
            for &region in &mapped {
                paint.invalidate(region);
            }
            let epoch = paint.latest_damage_epoch();
            surfaces
                .order
                .entry(parent)
                .or_default()
                .note_direct_damage(group, epoch, mapped);
            surfaces.touch(parent);
            surfaces.dirty(parent);
        }
    }
}

pub(crate) struct PendingRetainedPaint {
    pub(crate) id: PaintId,
    pub(crate) camera: Entity,
    pub(crate) draw: RetainedDraw,
    pub(crate) resource: ResourceFingerprint,
    pub(crate) coverage: PaintCoverage,
    pub(crate) painted: bool,
}

#[derive(bevy::prelude::Resource)]
pub(crate) struct PendingRetainedPaints<const SOURCE: u8> {
    upserts: Vec<PendingRetainedPaint>,
    removals: Vec<PaintId>,
}

impl<const SOURCE: u8> Default for PendingRetainedPaints<SOURCE> {
    fn default() -> Self {
        Self {
            upserts: Vec::new(),
            removals: Vec::new(),
        }
    }
}

impl<const SOURCE: u8> PendingRetainedPaints<SOURCE> {
    pub(crate) fn reserve(&mut self, additional: usize) {
        self.upserts.reserve(additional);
    }

    pub(crate) fn upsert(&mut self, paint: PendingRetainedPaint) {
        self.upserts.push(paint);
    }

    pub(crate) fn remove(&mut self, id: PaintId) {
        self.removals.push(id);
    }
}

pub(crate) type PendingGradientPaints = PendingRetainedPaints<0>;
pub(crate) type PendingImagePaints = PendingRetainedPaints<1>;
pub(crate) type PendingShadowPaints = PendingRetainedPaints<2>;
pub(crate) type PendingViewportPaints = PendingRetainedPaints<3>;

pub(crate) fn apply_retained_paints(
    mut commands: Commands,
    state: Res<RetainedUiScene>,
    mut backgrounds: ResMut<PendingRetainedBackgrounds>,
    mut gradients: ResMut<PendingGradientPaints>,
    mut images: ResMut<PendingImagePaints>,
    mut shadows: ResMut<PendingShadowPaints>,
    mut viewports: ResMut<PendingViewportPaints>,
) {
    let mut surfaces = state.lock();
    for id in backgrounds
        .removals()
        .chain(gradients.removals.drain(..))
        .chain(images.removals.drain(..))
        .chain(shadows.removals.drain(..))
        .chain(viewports.removals.drain(..))
    {
        surfaces.remove(&mut commands, id);
    }
    for paint in backgrounds
        .upserts()
        .chain(gradients.upserts.drain(..))
        .chain(images.upserts.drain(..))
        .chain(shadows.upserts.drain(..))
        .chain(viewports.upserts.drain(..))
    {
        surfaces.upsert(
            &mut commands,
            paint.id,
            paint.camera,
            paint.draw,
            paint.resource,
            paint.coverage,
            paint.painted,
        );
    }
}

#[derive(bevy::prelude::Resource, Default)]
pub(crate) struct RetainedRepairPlans(pub(crate) EntityHashMap<RepairPlan>);

#[derive(bevy::prelude::Resource, Default)]
pub(crate) struct RetainedFullRebuilds(pub(crate) EntityHashSet);

#[derive(bevy::prelude::Resource, Default)]
pub(crate) struct RetainedItems {
    pub(crate) items: Mutex<EntityHashMap<RetainedItem>>,
    pub(crate) boundaries: Mutex<EntityHashMap<RetainedBoundaryDraw>>,
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
pub(crate) struct RetainedPaintRun {
    pub(crate) parent: Entity,
    pub(crate) view_entity: Entity,
    pub(crate) retained_view_entity: RetainedViewEntity,
    pub(crate) bounds: PhysicalRect,
    pub(crate) repair_bounds: PhysicalRect,
}

#[derive(Clone)]
pub(crate) enum RetainedCompositorEntry {
    PaintRun(RetainedPaintRun),
    Boundary {
        draw: RetainedBoundaryDraw,
        repair_bounds: PhysicalRect,
    },
}

#[derive(Clone)]
pub(crate) struct RetainedCompositor {
    pub(crate) entries: Vec<RetainedCompositorEntry>,
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
        surfaces.paint.remove(&camera);
        let ids: Vec<_> = surfaces
            .records
            .iter()
            .filter_map(|(_, owned)| (owned.camera == camera).then_some(owned.id))
            .collect();
        for id in ids {
            let owned = surfaces
                .detach_record(id)
                .expect("retained paint must have an owned record");
            if let Ok(mut entity) = commands.get_entity(owned.record.value.render_entity) {
                entity.despawn();
            }
        }
        surfaces.dirty_surfaces.retain(|listed| *listed != camera);
        surfaces.touched_surfaces.retain(|listed| *listed != camera);
        surfaces.propagated_epochs.remove(&camera);
        surfaces.surface_bounds.remove(&camera);
        if let Some(order) = surfaces.order.remove(&camera)
            && let Some(compositor) = order.compositor
        {
            for run in compositor.runs.into_values() {
                surfaces.run_by_view.remove(&run.view_entity);
                surfaces.retired_run_views.push(run.retained_view_entity);
                commands.entity(run.view_entity).despawn();
            }
        }
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
                Ref<'static, ComputedNode>,
                Option<&'static TextScroll>,
            ),
            (With<Node>, Changed<UiGlobalTransform>),
        >,
    >,
) {
    let mut surfaces = state.lock();
    for (entity, transform, clip, owner, node, scroll) in &changed {
        if node.is_changed() {
            continue;
        }
        let clip = retained_clip(entity, &node, transform, clip, owner);
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
    records: &RecordArena,
    slots: &[usize],
    positions: core::ops::Range<usize>,
) -> PhysicalRect {
    positions
        .flat_map(|position| records[slots[position]].record.coverage.iter().copied())
        .reduce(enclosing_rect)
        .expect("retained paint groups have visible coverage")
}

fn group_intersects(
    records: &RecordArena,
    slots: &[usize],
    mut positions: core::ops::Range<usize>,
    region: PhysicalRect,
) -> bool {
    positions.any(|position| {
        records[slots[position]]
            .record
            .coverage
            .iter()
            .any(|coverage| coverage.intersection(region).is_some())
    })
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

fn repair_covers_target(repair: &RepairPlan, target: PhysicalRect) -> bool {
    repair
        .regions()
        .iter()
        .filter_map(|region| region.intersection(target))
        .map(|region| region.area())
        .sum::<u64>()
        == target.area()
}

pub(crate) fn replay_retained_ui(
    mut commands: Commands,
    state: Res<RetainedUiScene>,
    counters: Res<RetainedUiPaintCounters>,
    mut core_runs: ResMut<RetainedCoreRuns>,
    mut gradient_runs: ResMut<RetainedGradientRuns>,
    mut shadow_runs: ResMut<RetainedShadowRuns>,
    retained_items: Res<RetainedItems>,
    mut extracted_slices: ResMut<ExtractedUiTextureSlices>,
    mut extracted_materials: ResMut<RetainedMaterialReplays>,
    pending_materials: Res<RetainedPendingMaterials>,
    ui_views: Query<(&ExtractedView, &UiViewTarget)>,
    mut repair_plans: ResMut<RetainedRepairPlans>,
    mut full_rebuilds: ResMut<RetainedFullRebuilds>,
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
    full_rebuilds.0.clear();
    let mut buffers = ReplayBuffers {
        slices: &mut extracted_slices,
        materials: &mut extracted_materials,
    };

    let touched_surfaces = core::mem::take(&mut surfaces.touched_surfaces);
    for camera in touched_surfaces {
        if let Some(paint) = surfaces.paint.get_mut(&camera) {
            paint.clear_touched_listing();
            counters.add(paint.take_counters());
        }
    }
    let dirty_surfaces = surfaces.dirty_surfaces.clone();

    let RetainedUiSurfaces {
        paint,
        records,
        by_main_entity: _,
        surface_by_main_entity: _,
        dirty_surfaces: _,
        touched_surfaces: _,
        propagated_epochs: _,
        order,
        surface_origins: _,
        surface_z_origins: _,
        surface_bounds: _,
        run_by_view,
        created_run_views,
        retired_run_views,
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
            // Every record on this camera surrenders its group BEFORE the
            // new ordering assigns them: a record excluded below (empty
            // coverage this frame) would otherwise keep an index into the
            // PREVIOUS ordering, and when it comes back, `note_direct` /
            // `note_bounds_dirty` index past the rebuilt per-group vectors
            // (observed live: len 273, index 274, from
            // `extract_retained_text`) — or worse, alias a live group.
            for owned in records.slots.iter_mut().flatten() {
                if owned.camera == camera {
                    owned.group = None;
                }
            }
            order.slots.clear();
            order.slots.extend(
                records
                    .iter()
                    .filter(|(_, owned)| {
                        owned.camera == camera && !owned.record.coverage.is_empty()
                    })
                    .map(|(slot, _)| slot),
            );
            order.slots.sort_by(|&left_slot, &right_slot| {
                let left_owned = &records[left_slot];
                let right_owned = &records[right_slot];
                let left = &left_owned.record.value.draw;
                let right = &right_owned.record.value.draw;
                left.z_order
                    .total_cmp(&right.z_order)
                    .then_with(|| left_owned.id.family.cmp(&right_owned.id.family))
                    .then_with(|| left.paint_order.cmp(&right.paint_order))
                    .then_with(|| left_owned.id.entity.cmp(&right_owned.id.entity))
                    .then_with(|| left_owned.id.ordinal.cmp(&right_owned.id.ordinal))
            });
            order.groups.clear();
            for (position, &slot) in order.slots.iter().enumerate() {
                let group = order.groups.len();
                records[slot].group = Some(group);
                order.groups.push(position..position + 1);
            }
            let previous_compositor = order.compositor.take();
            let compositor_slots = if order.volatile {
                &[][..]
            } else {
                order.slots.as_slice()
            };
            order.compositor = rebuild_ordered_compositor(
                &mut commands,
                camera,
                records,
                compositor_slots,
                previous_compositor,
                run_by_view,
                created_run_views,
                retired_run_views,
            );
            order.spatial = None;
            order.order_dirty = false;
        }
        if order.spatial.is_none() {
            let mut entries = Vec::new();
            for (group, positions) in order.groups.iter().enumerate() {
                entries.push((
                    group_bounds(records, &order.slots, positions.clone()),
                    group,
                ));
            }
            order.spatial = Some(SpatialIndex::new(entries));
            order.spatial_revision = order
                .spatial_revision
                .checked_add(1)
                .expect("retained UI spatial revision overflowed");
            order.candidate_marks.resize(order.groups.len(), 0);
            order.direct_groups.clear();
            order.direct_epochs.clear();
            order.direct_epochs.resize(order.groups.len(), [0, 0]);
            order.direct_damage.clear();
            order
                .direct_damage
                .resize_with(order.groups.len(), Vec::new);
            order.bounds_dirty.clear();
            order.bounds_dirty_marks.clear();
            order.bounds_dirty_marks.resize(order.groups.len(), false);
        }
        let all_groups_changed = order.direct_groups.len() == order.groups.len();
        if !all_groups_changed && !order.bounds_dirty.is_empty() {
            order.bounds_updates.clear();
            for group in order.bounds_dirty.drain(..) {
                order.bounds_dirty_marks[group] = false;
                order.bounds_updates.push((
                    group,
                    group_bounds(records, &order.slots, order.groups[group].clone()),
                ));
            }
            order
                .spatial
                .as_mut()
                .expect("retained spatial index was just built")
                .update_many(order.bounds_updates.drain(..));
            order.spatial_revision = order
                .spatial_revision
                .checked_add(1)
                .expect("retained UI spatial revision overflowed");
        }
        let candidates_cached = order.cached_candidate_revision == order.spatial_revision
            && order.cached_direct_groups == order.direct_groups
            && order
                .cached_candidate_repair
                .as_ref()
                .is_some_and(|cached| cached.shares_regions_with(&repair));
        if !candidates_cached {
            order.candidates.clear();
            if all_groups_changed {
                order.candidates.extend(0..order.groups.len());
            } else {
                order.candidate_generation = order.candidate_generation.wrapping_add(1);
                if order.candidate_generation == 0 {
                    order.candidate_marks.fill(0);
                    order.candidate_generation = 1;
                }
                let generation = order.candidate_generation;
                for &direct_group in &order.direct_groups {
                    if order.direct_epochs[direct_group][0] <= repair.through_epoch()
                        && order.candidate_marks[direct_group] != generation
                    {
                        order.candidate_marks[direct_group] = generation;
                        order.candidates.push(direct_group);
                    }
                }
                if order.candidates.len() != order.groups.len() {
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
            }
            order.cached_candidate_revision = order.spatial_revision;
            order.cached_candidate_repair = Some(repair.clone());
            order.cached_direct_groups.clone_from(&order.direct_groups);
        }
        let mut run_repairs: HashMap<PaintRunKey, RepairPlan> = HashMap::default();
        if let Some(compositor) = &mut order.compositor {
            for (&key, run) in &mut compositor.runs {
                let Some(run_repair) = run.damage.plan() else {
                    continue;
                };
                repair_plans.0.insert(run.view_entity, run_repair.clone());
                if repair_covers_target(&run_repair, run.bounds) {
                    full_rebuilds.0.insert(run.view_entity);
                }
                run_repairs.insert(key, run_repair);
            }
        }
        order.candidate_generation = order.candidate_generation.wrapping_add(1);
        if order.candidate_generation == 0 {
            order.candidate_marks.fill(0);
            order.candidate_generation = 1;
        }
        let replay_generation = order.candidate_generation;
        for &group in &order.candidates {
            order.candidate_marks[group] = replay_generation;
        }
        let mut core_run = None;
        let mut gradient_run = None;
        let mut shadow_run = None;
        let mut current_camera = None;
        let mut contains_boundary = order.compositor.is_some();
        let mut staged = 0;
        let mut candidate_index = 0;
        while let Some(&group) = order.candidates.get(candidate_index) {
            candidate_index += 1;
            if order.candidate_marks[group] != replay_generation {
                continue;
            }
            order.candidate_marks[group] = 0;
            let positions = order.groups[group].clone();
            let slot = order.slots[positions.start];
            let id = records[slot].id;
            let record = &records[slot].record;
            let run = order
                .compositor
                .as_ref()
                .and_then(|compositor| compositor.group_runs[group])
                .and_then(|key| {
                    let repair = run_repairs.get(&key)?;
                    record
                        .coverage
                        .iter()
                        .any(|coverage| {
                            repair
                                .regions()
                                .iter()
                                .any(|damage| coverage.intersection(*damage).is_some())
                        })
                        .then(|| &order.compositor.as_ref().unwrap().runs[&key])
                });
            if let RetainedDrawItem::Boundary(boundary) = &record.value.draw.item {
                contains_boundary = true;
                if order.compositor.is_some() {
                    continue;
                }
                core_run = None;
                gradient_run = None;
                shadow_run = None;
                current_camera = None;
                staged += 1;
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
                    records[order.slots[position]]
                        .record
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
            let draw = if let Some(run) = run {
                let mut draw = record.value.draw.clone();
                draw.camera = run.view_entity;
                draw
            } else if order.compositor.is_some() {
                continue;
            } else {
                record.value.draw.clone()
            };
            if current_camera != Some(draw.camera) {
                core_run = None;
                gradient_run = None;
                shadow_run = None;
                current_camera = Some(draw.camera);
            }
            staged += 1;
            if let RetainedDrawItem::BoxShadow(item) = &record.value.draw.item {
                core_run = None;
                gradient_run = None;
                let (samples, instance) = record
                    .value
                    .shadow_source()
                    .expect("retained shadows are prepared when their record changes");
                debug_assert_eq!(samples, item.samples());
                shadow_runs.push(&mut shadow_run, &draw, samples, instance);
                continue;
            }
            shadow_run = None;
            if let RetainedDrawItem::Gradient(item) = &record.value.draw.item {
                core_run = None;
                let (color_space, instances) = record
                    .value
                    .gradient_source()
                    .expect("retained gradients are prepared when their record changes");
                debug_assert_eq!(color_space, item.color_space());
                gradient_runs.push(&mut gradient_run, &draw, color_space, instances);
                continue;
            }
            gradient_run = None;
            if matches!(
                record.value.draw.item,
                RetainedDrawItem::Border(_)
                    | RetainedDrawItem::Node(_)
                    | RetainedDrawItem::Glyphs(_)
            ) {
                push_core_replayed(
                    &mut core_runs,
                    &mut core_run,
                    record.value.core_source(id),
                    None,
                    &draw,
                );
                continue;
            }
            core_run = None;
            let coverage = PaintCoverage::from_regions(positions.flat_map(|position| {
                records[order.slots[position]]
                    .record
                    .coverage
                    .iter()
                    .copied()
            }));
            push_replayed(&mut buffers, &mut items, &draw, coverage);
        }
        counters.add_staged(staged);
        let target = ui_views.iter().find_map(|(view, target)| {
            (target.0 == camera).then(|| {
                PhysicalRect::from_min_max(0, 0, view.viewport.z as i32, view.viewport.w as i32)
                    .expect("UI views have nonzero physical viewports")
            })
        });
        if !contains_boundary && target.is_some_and(|target| repair_covers_target(&repair, target))
        {
            full_rebuilds.0.insert(camera);
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
    items: &mut EntityHashMap<RetainedItem>,
    draw: &RetainedDraw,
    coverage: PaintCoverage,
) {
    match &draw.item {
        RetainedDrawItem::Boundary(_) => unreachable!("boundaries use retained composition runs"),
        RetainedDrawItem::Border(_) => unreachable!("borders use persistent preparation"),
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
            direct_epochs: vec![[0, 0]],
            direct_damage: vec![Vec::new()],
            ..Default::default()
        };
        let damage = PhysicalRect::from_min_max(0, 0, 1, 1).unwrap();

        order.note_direct_damage(Some(0), 2, [damage]);
        order.note_direct_damage(Some(0), 4, [damage]);
        order.note_direct_damage(Some(0), 5, [damage]);
        assert_eq!(order.direct_groups, [0]);
        assert_eq!(order.direct_epochs, [[2, 5]]);
        assert_eq!(order.direct_damage[0].len(), 3);

        order.acknowledge(4);
        assert_eq!(order.direct_groups, [0]);
        assert_eq!(order.direct_epochs, [[5, 5]]);
        assert_eq!(order.direct_damage[0].len(), 1);

        order.acknowledge(5);
        assert!(order.direct_groups.is_empty());
        assert_eq!(order.direct_epochs, [[5, 5]]);
        assert!(order.direct_damage[0].is_empty());
    }

    #[test]
    fn full_rebuild_requires_exact_target_coverage() {
        let target = PhysicalRect::from_min_max(0, 0, 10, 10).unwrap();
        let mut damage = DamageJournal::default();
        damage.record(PhysicalRect::from_min_max(0, 0, 4, 10).unwrap());
        damage.record(PhysicalRect::from_min_max(6, 0, 10, 10).unwrap());
        assert!(!repair_covers_target(&damage.plan().unwrap(), target));

        damage.record(PhysicalRect::from_min_max(4, -5, 6, 15).unwrap());
        assert!(repair_covers_target(&damage.plan().unwrap(), target));
    }
}
