//! Exact retained contracts and extraction for custom UI materials.

use crate::{
    boundary::retained_clip,
    sampled_image::{ImageReader, ImageSample, RetainedSampledImages, SampledImageState},
    scene::{
        coverage, PaintFamily, PaintId, ResourceFingerprint, RetainedDraw, RetainedDrawItem,
        RetainedMaterialItem, RetainedMaterialReplays, RetainedUiScene, RetainedUiSurfaces,
    },
    PhysicalRect,
};
use bevy::ui_render::ui_material::{MaterialNode, UiMaterial};
use bevy::{
    app::{App, Inherited, Plugin},
    asset::{AssetEvent, AssetId, Assets},
    camera::visibility::InheritedVisibility,
    ecs::{
        entity::Entity,
        lifecycle::RemovedComponents,
        message::MessageReader,
        query::{Changed, Or, With},
        schedule::IntoScheduleConfigs,
        system::{Commands, Query, Res, ResMut, SystemParam},
    },
    image::Image,
    math::{Rect, Vec2},
    render::{
        render_asset::RenderAssets, render_resource::DefaultImageSamplerDescriptor, Extract,
        ExtractSchedule, Render, RenderApp, RenderSystems,
    },
    ui::{
        CalculatedClip, ComputedNode, ComputedStackIndex, ComputedUiPaintTarget,
        ComputedUiRenderTargetInfo, ComputedUiTargetCamera, Display, Node, UiGlobalTransform,
    },
    ui_render::{
        queue_ui_material_nodes, ExtractedUiMaterialNode, ExtractedUiMaterialNodes,
        PreparedUiMaterial, RenderUiSystems, UiCameraMap, UiMaterialBatch, UiMaterialBatchRange,
        UiMaterialInfrastructurePlugin,
    },
};
use core::{any::TypeId, hash::Hash, marker::PhantomData};
use std::{
    collections::{HashMap, HashSet},
    sync::{Mutex, PoisonError},
};

/// Whether an exact material can be retained or must repaint whenever it is visible.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RetainedUiMaterialKey<K> {
    /// This value changes whenever non-image shader output may change.
    Exact(K),
    /// The shader depends on state such as time or globals that changes every frame.
    Volatile,
}

/// Conservative raster coverage promised by a retained UI material.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetainedUiMaterialCoverage {
    /// The vertex shader cannot rasterize outside the node's transformed quad.
    Node,
    /// The vertex shader may rasterize anywhere in the target.
    Target,
}

/// One image region that a retained UI material may sample.
#[derive(Clone, Copy, Debug)]
pub struct RetainedUiMaterialImage {
    image: AssetId<Image>,
    region: Option<Rect>,
}

impl RetainedUiMaterialImage {
    /// Declares that the material may sample any texel in `image`.
    pub fn all(image: impl Into<AssetId<Image>>) -> Self {
        Self {
            image: image.into(),
            region: None,
        }
    }

    /// Declares a physical-texel read rectangle in `image`.
    pub fn region(image: impl Into<AssetId<Image>>, region: Rect) -> Self {
        Self {
            image: image.into(),
            region: Some(region),
        }
    }
}

/// Complete retained-paint declaration for one custom material value.
#[derive(Clone)]
pub struct RetainedUiMaterialSnapshot<K> {
    /// Exact non-image paint state, or explicit per-frame volatility.
    pub key: RetainedUiMaterialKey<K>,
    /// Maximum possible raster coverage.
    pub coverage: RetainedUiMaterialCoverage,
    /// Every image region the shader may read.
    pub images: Vec<RetainedUiMaterialImage>,
}

impl<K> RetainedUiMaterialSnapshot<K> {
    /// Creates an exact declaration. `key` must cover every non-image shader input.
    pub fn exact(
        key: K,
        coverage: RetainedUiMaterialCoverage,
        images: Vec<RetainedUiMaterialImage>,
    ) -> Self {
        Self {
            key: RetainedUiMaterialKey::Exact(key),
            coverage,
            images,
        }
    }

    /// Creates an explicitly volatile declaration for time/global-dependent paint.
    pub fn volatile(
        coverage: RetainedUiMaterialCoverage,
        images: Vec<RetainedUiMaterialImage>,
    ) -> Self {
        Self {
            key: RetainedUiMaterialKey::Volatile,
            coverage,
            images,
        }
    }
}

/// Opt-in correctness contract for retaining a custom [`UiMaterial`].
///
/// `retained_ui` must declare every shader input not already represented by node geometry,
/// target, transform, clip, and material asset identity. Under-declaration is a correctness bug;
/// use [`RetainedUiMaterialKey::Volatile`] or [`RetainedUiMaterialCoverage::Target`] whenever an
/// exact narrower promise cannot be made.
pub trait RetainedUiMaterial: UiMaterial {
    /// Exact, collision-free application value used to compare non-image paint state.
    type PaintKey: Clone + Eq + Send + Sync + 'static;

    /// Returns the complete retained-paint declaration for this material value.
    fn retained_ui(&self) -> RetainedUiMaterialSnapshot<Self::PaintKey>;
}

/// Retained replacement for [`bevy::ui_render::UiMaterialPlugin`].
pub struct RetainedUiMaterialPlugin<M: RetainedUiMaterial>(PhantomData<M>);

impl<M: RetainedUiMaterial> Default for RetainedUiMaterialPlugin<M> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<M: RetainedUiMaterial> Plugin for RetainedUiMaterialPlugin<M>
where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    fn build(&self, app: &mut App) {
        app.add_plugins(UiMaterialInfrastructurePlugin::<M>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<RetainedMaterialDependencies<M>>()
            .init_resource::<RetainedPendingMaterials>()
            .init_resource::<RetainedMaterialReplays>()
            .add_systems(
                ExtractSchedule,
                extract_retained_materials::<M>
                    .in_set(RenderUiSystems::ExtractBackgrounds)
                    .before(crate::scene::replay_retained_ui),
            )
            .add_systems(
                ExtractSchedule,
                replay_retained_materials::<M>.after(crate::scene::replay_retained_ui),
            )
            .add_systems(
                Render,
                mark_pending_materials::<M>
                    .in_set(RenderSystems::Queue)
                    .after(queue_ui_material_nodes::<M>),
            );
    }
}

#[derive(Clone, PartialEq, Eq)]
enum StoredMaterialKey<K> {
    Exact(K),
    Volatile,
}

impl<K> From<RetainedUiMaterialKey<K>> for StoredMaterialKey<K> {
    fn from(key: RetainedUiMaterialKey<K>) -> Self {
        match key {
            RetainedUiMaterialKey::Exact(key) => Self::Exact(key),
            RetainedUiMaterialKey::Volatile => Self::Volatile,
        }
    }
}

struct MaterialAssetState<K> {
    snapshot: Option<StoredMaterialSnapshot<K>>,
    revision: u64,
    readers: HashSet<Entity>,
}

#[derive(Clone, PartialEq, Eq)]
struct StoredMaterialSnapshot<K> {
    key: StoredMaterialKey<K>,
    coverage: RetainedUiMaterialCoverage,
    samples: Box<[ImageSample]>,
}

impl<K> StoredMaterialSnapshot<K> {
    fn new(snapshot: RetainedUiMaterialSnapshot<K>) -> Self {
        let mut samples: Vec<_> = snapshot
            .images
            .into_iter()
            .map(|image| {
                image.region.map_or_else(
                    || ImageSample::all(image.image),
                    |region| ImageSample::rect(image.image, region),
                )
            })
            .collect();
        samples.sort_unstable();
        samples.dedup();
        Self {
            key: snapshot.key.into(),
            coverage: snapshot.coverage,
            samples: samples.into_boxed_slice(),
        }
    }
}

#[derive(bevy::prelude::Resource)]
struct RetainedMaterialDependencies<M: RetainedUiMaterial> {
    entities: HashMap<Entity, AssetId<M>>,
    assets: HashMap<AssetId<M>, MaterialAssetState<M::PaintKey>>,
    volatile: HashSet<AssetId<M>>,
    next_revision: u64,
}

impl<M: RetainedUiMaterial> Default for RetainedMaterialDependencies<M> {
    fn default() -> Self {
        Self {
            entities: HashMap::new(),
            assets: HashMap::new(),
            volatile: HashSet::new(),
            next_revision: 0,
        }
    }
}

impl<M: RetainedUiMaterial> RetainedMaterialDependencies<M> {
    fn new_revision(&mut self) -> u64 {
        self.next_revision = self
            .next_revision
            .checked_add(1)
            .expect("retained UI material revision exhausted");
        self.next_revision
    }

    fn remove_entity(&mut self, entity: Entity) {
        let Some(material) = self.entities.remove(&entity) else {
            return;
        };
        let Some(state) = self.assets.get_mut(&material) else {
            return;
        };
        state.readers.remove(&entity);
        if state.readers.is_empty() {
            self.assets.remove(&material);
            self.volatile.remove(&material);
        }
    }

    fn set_entity(
        &mut self,
        entity: Entity,
        material: AssetId<M>,
        snapshot: Option<StoredMaterialSnapshot<M::PaintKey>>,
    ) {
        if self.entities.get(&entity) != Some(&material) {
            self.remove_entity(entity);
            self.entities.insert(entity, material);
            if snapshot
                .as_ref()
                .is_some_and(|snapshot| matches!(&snapshot.key, StoredMaterialKey::Volatile))
            {
                self.volatile.insert(material);
            }
            self.assets
                .entry(material)
                .or_insert_with(|| MaterialAssetState {
                    snapshot,
                    revision: 0,
                    readers: HashSet::new(),
                })
                .readers
                .insert(entity);
        }
    }

    fn process_asset(
        &mut self,
        material: AssetId<M>,
        value: Option<&M>,
        candidates: &mut HashSet<Entity>,
    ) {
        let new_snapshot = value.map(|value| StoredMaterialSnapshot::new(value.retained_ui()));
        let changed = self
            .assets
            .get(&material)
            .is_some_and(|state| state.snapshot != new_snapshot);
        if !changed {
            return;
        }
        let revision = self.new_revision();
        if new_snapshot
            .as_ref()
            .is_some_and(|snapshot| matches!(&snapshot.key, StoredMaterialKey::Volatile))
        {
            self.volatile.insert(material);
        } else {
            self.volatile.remove(&material);
        }
        let state = self.assets.get_mut(&material).unwrap();
        state.snapshot = new_snapshot;
        state.revision = revision;
        candidates.extend(&state.readers);
    }

    fn nominate_volatile(&mut self, candidates: &mut HashSet<Entity>) {
        let Self {
            assets,
            volatile,
            next_revision,
            ..
        } = self;
        for material in volatile.iter() {
            *next_revision = next_revision
                .checked_add(1)
                .expect("retained UI material revision exhausted");
            let state = assets.get_mut(material).unwrap();
            state.revision = *next_revision;
            candidates.extend(&state.readers);
        }
    }

    fn revision(&self, material: AssetId<M>) -> u64 {
        self.assets.get(&material).map_or(0, |state| state.revision)
    }

    fn prepared_matches(&self, material: AssetId<M>, prepared: &M) -> bool {
        let snapshot = StoredMaterialSnapshot::new(prepared.retained_ui());
        self.assets
            .get(&material)
            .is_some_and(|state| state.snapshot.as_ref() == Some(&snapshot))
    }
}

type MaterialQueryItem<'a, M> = (
    Entity,
    &'a Node,
    &'a ComputedNode,
    &'a ComputedStackIndex,
    &'a UiGlobalTransform,
    &'a MaterialNode<M>,
    &'a InheritedVisibility,
    Option<&'a CalculatedClip>,
    Option<&'a Inherited<ComputedUiPaintTarget>>,
    &'a ComputedUiTargetCamera,
    &'a ComputedUiRenderTargetInfo,
);

#[derive(SystemParam)]
struct RemovedMaterialInputs<'w, 's, M: RetainedUiMaterial> {
    material: RemovedComponents<'w, 's, MaterialNode<M>>,
    clip: RemovedComponents<'w, 's, CalculatedClip>,
    target: RemovedComponents<'w, 's, ComputedUiRenderTargetInfo>,
    computed_node: RemovedComponents<'w, 's, ComputedNode>,
    node: RemovedComponents<'w, 's, Node>,
    stack: RemovedComponents<'w, 's, ComputedStackIndex>,
    transform: RemovedComponents<'w, 's, UiGlobalTransform>,
    visibility: RemovedComponents<'w, 's, InheritedVisibility>,
    camera: RemovedComponents<'w, 's, ComputedUiTargetCamera>,
}

#[expect(
    clippy::too_many_arguments,
    reason = "node, material asset, sampled images, and lifecycle inputs independently nominate paint"
)]
fn extract_retained_materials<M: RetainedUiMaterial>(
    mut commands: Commands,
    state: Res<RetainedUiScene>,
    mut dependencies: ResMut<RetainedMaterialDependencies<M>>,
    sampled_images: Res<RetainedSampledImages>,
    default_sampler: Res<DefaultImageSamplerDescriptor>,
    materials: Extract<Res<Assets<M>>>,
    images: Extract<Res<Assets<Image>>>,
    mut events: Extract<MessageReader<AssetEvent<M>>>,
    changed: Extract<
        Query<
            MaterialQueryItem<'static, M>,
            (
                With<MaterialNode<M>>,
                Or<(
                    Changed<ComputedNode>,
                    Changed<ComputedStackIndex>,
                    Changed<MaterialNode<M>>,
                    Changed<InheritedVisibility>,
                    Changed<CalculatedClip>,
                    Changed<ComputedUiTargetCamera>,
                    Changed<ComputedUiRenderTargetInfo>,
                    Changed<Node>,
                )>,
            ),
        >,
    >,
    all: Extract<Query<MaterialQueryItem<'static, M>, With<MaterialNode<M>>>>,
    camera_map: Extract<UiCameraMap>,
    mut removed: Extract<RemovedMaterialInputs<M>>,
) where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    let mut sampled_images = sampled_images.lock();
    let mut extra_candidates: HashSet<_> =
        sampled_images.take_materials::<M>().into_iter().collect();
    for event in events.read() {
        let material = match *event {
            AssetEvent::Added { id }
            | AssetEvent::Modified { id }
            | AssetEvent::Unused { id }
            | AssetEvent::Removed { id } => id,
            AssetEvent::LoadedWithDependencies { .. } => continue,
        };
        dependencies.process_asset(material, materials.get(material), &mut extra_candidates);
    }
    dependencies.nominate_volatile(&mut extra_candidates);

    let RemovedMaterialInputs {
        material,
        clip,
        target,
        computed_node,
        node,
        stack,
        transform,
        visibility,
        camera,
    } = &mut *removed;
    extra_candidates.extend(material.read());
    extra_candidates.extend(clip.read());
    extra_candidates.extend(target.read());
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
        remove_material::<M>(
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
            remove_material::<M>(
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
    for (
        entity,
        source_node,
        node,
        stack,
        transform,
        handle,
        visibility,
        clip,
        owner,
        target_camera,
        target,
    ) in changed.iter().chain(
        extra_candidates
            .into_iter()
            .filter_map(|entity| all.get(entity).ok()),
    ) {
        let value = materials.get(handle);
        let snapshot = value.map(RetainedUiMaterial::retained_ui);
        dependencies.set_entity(
            entity,
            handle.id(),
            snapshot
                .as_ref()
                .map(|snapshot| StoredMaterialSnapshot::new(snapshot.clone())),
        );
        let id = material_id::<M>(entity);
        let reader = ImageReader::Material(TypeId::of::<M>(), entity);
        let Some(camera) = camera_mapper.map(target_camera) else {
            surfaces.remove(&mut commands, id);
            sampled_images.remove_reader(reader);
            continue;
        };
        let Some(snapshot) = snapshot else {
            surfaces.remove(&mut commands, id);
            sampled_images.remove_reader(reader);
            continue;
        };
        let clip = retained_clip(entity, node, transform, clip, owner);
        let transform = transform.affine();
        let visible = visibility.get()
            && source_node.display != Display::None
            && !node.is_empty()
            && !node.size().cmple(Vec2::ZERO).any();
        let samples: Vec<_> = snapshot
            .images
            .iter()
            .map(|sample| {
                sample.region.map_or_else(
                    || ImageSample::all(sample.image),
                    |region| ImageSample::rect(sample.image, region),
                )
            })
            .collect();
        let mut sampled_ids: Vec<_> = samples.iter().map(|sample| sample.image()).collect();
        sampled_ids.sort_unstable();
        sampled_ids.dedup();
        let resource = if visible {
            sampled_images.replace_reader(
                reader,
                samples.iter().copied(),
                &images,
                &default_sampler,
            );
            for image in &sampled_ids {
                sampled_images.mark_pending(*image, images.get(*image));
            }
            let mut revisions = vec![dependencies.revision(handle.id())];
            for image in &sampled_ids {
                revisions.extend(
                    sampled_images.revisions(
                        *image,
                        samples
                            .iter()
                            .copied()
                            .filter(|sample| sample.image() == *image),
                    ),
                );
            }
            ResourceFingerprint::Revisions(revisions.into_iter().collect())
        } else {
            sampled_images.remove_reader(reader);
            ResourceFingerprint::None
        };
        let coverage = if !visible {
            Default::default()
        } else {
            match snapshot.coverage {
                RetainedUiMaterialCoverage::Node => {
                    coverage(node.size(), transform, clip).into_iter().collect()
                }
                RetainedUiMaterialCoverage::Target => PhysicalRect::from_min_max(
                    0,
                    0,
                    target.physical_size().x as i32,
                    target.physical_size().y as i32,
                )
                .into_iter()
                .collect(),
            }
        };
        surfaces.upsert(
            &mut commands,
            id,
            camera,
            RetainedDraw {
                render_entity: Entity::PLACEHOLDER,
                camera,
                main_entity: entity.into(),
                z_order: stack.0 as f32 + M::stack_z_offset(),
                paint_order: 0,
                clip,
                image: AssetId::<Image>::default(),
                transform,
                layout_translation: Vec2::ZERO,
                local_translation: Vec2::ZERO,
                item: RetainedDrawItem::Material(RetainedMaterialItem::new(
                    handle.id(),
                    stack.0,
                    Rect::from_corners(Vec2::ZERO, node.size()),
                    node.border(),
                    node.border_radius(),
                    sampled_ids.into_boxed_slice(),
                    snapshot.coverage == RetainedUiMaterialCoverage::Target,
                )),
            },
            resource,
            coverage,
            visible,
        );
    }
}

fn material_id<M: RetainedUiMaterial>(entity: Entity) -> PaintId {
    PaintId {
        entity,
        family: PaintFamily::Material(TypeId::of::<M>()),
        ordinal: 0,
    }
}

fn remove_material<M: RetainedUiMaterial>(
    dependencies: &mut RetainedMaterialDependencies<M>,
    sampled_images: &mut SampledImageState,
    surfaces: &mut RetainedUiSurfaces,
    commands: &mut Commands,
    entity: Entity,
) {
    dependencies.remove_entity(entity);
    sampled_images.remove_reader(ImageReader::Material(TypeId::of::<M>(), entity));
    surfaces.remove(commands, material_id::<M>(entity));
}

fn replay_retained_materials<M: RetainedUiMaterial>(
    mut commands: Commands,
    replays: Res<RetainedMaterialReplays>,
    mut extracted: ResMut<ExtractedUiMaterialNodes<M>>,
) {
    for replay in &replays.0 {
        if replay.item.material_type != TypeId::of::<M>() {
            continue;
        }
        if let Ok(mut entity) = commands.get_entity(replay.draw.render_entity) {
            entity.remove::<(UiMaterialBatch<M>, UiMaterialBatchRange)>();
        }
        extracted.uinodes.push(ExtractedUiMaterialNode {
            stack_index: replay.item.stack_index,
            transform: replay.draw.transform,
            rect: replay.item.rect(),
            border: replay.item.border(),
            border_radius: replay.item.border_radius(),
            material: replay.item.material.typed::<M>(),
            clip: replay.draw.clip,
            extracted_camera_entity: replay.draw.camera,
            main_entity: replay.draw.main_entity,
            render_entity: replay.draw.render_entity,
        });
    }
}

#[derive(bevy::prelude::Resource, Default)]
pub(crate) struct RetainedPendingMaterials(Mutex<HashSet<Entity>>);

impl RetainedPendingMaterials {
    pub(crate) fn clear(&self) {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
    }

    fn insert(&self, camera: Entity) {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(camera);
    }

    pub(crate) fn contains(&self, camera: Entity) -> bool {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .contains(&camera)
    }
}

fn mark_pending_materials<M: RetainedUiMaterial>(
    replays: Res<RetainedMaterialReplays>,
    dependencies: Res<RetainedMaterialDependencies<M>>,
    prepared: Res<RenderAssets<PreparedUiMaterial<M>>>,
    pending: Res<RetainedPendingMaterials>,
) {
    for replay in &replays.0 {
        if replay.item.material_type != TypeId::of::<M>() {
            continue;
        }
        let material = replay.item.material.typed::<M>();
        if prepared
            .get(material)
            .is_none_or(|value| !dependencies.prepared_matches(material, &value.source))
        {
            pending.insert(replay.draw.camera);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(
        coverage: RetainedUiMaterialCoverage,
        images: Vec<RetainedUiMaterialImage>,
    ) -> StoredMaterialSnapshot<u8> {
        StoredMaterialSnapshot::new(RetainedUiMaterialSnapshot::exact(7, coverage, images))
    }

    #[test]
    fn prepared_identity_includes_coverage_and_image_bindings() {
        let first = snapshot(
            RetainedUiMaterialCoverage::Node,
            vec![RetainedUiMaterialImage::all(AssetId::<Image>::default())],
        );
        let changed_image = snapshot(
            RetainedUiMaterialCoverage::Node,
            vec![RetainedUiMaterialImage::all(AssetId::<Image>::invalid())],
        );
        let changed_coverage = snapshot(
            RetainedUiMaterialCoverage::Target,
            vec![RetainedUiMaterialImage::all(AssetId::<Image>::default())],
        );

        assert!(first != changed_image);
        assert!(first != changed_coverage);
    }

    #[test]
    fn image_declaration_order_and_duplicates_are_not_paint_state() {
        let first = AssetId::<Image>::default();
        let second = AssetId::<Image>::invalid();
        let left = snapshot(
            RetainedUiMaterialCoverage::Node,
            vec![
                RetainedUiMaterialImage::all(first),
                RetainedUiMaterialImage::all(second),
            ],
        );
        let right = snapshot(
            RetainedUiMaterialCoverage::Node,
            vec![
                RetainedUiMaterialImage::all(second),
                RetainedUiMaterialImage::all(first),
                RetainedUiMaterialImage::all(first),
            ],
        );

        assert!(left == right);
    }
}
