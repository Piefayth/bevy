use std::{marker::PhantomData, ops::Range, any::{Any, TypeId}};

use bevy_app::{App, IntoSystemAppConfigs, Plugin};
use bevy_asset::{AddAsset, AssetEvent, Assets, Handle, HandleUntyped, HandleId};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::prelude::*;
use bevy_math::{Mat4, Rect, Vec2, Vec3};
use bevy_reflect::TypeUuid;
use bevy_render::{
    extract_component::ExtractComponentPlugin,
    prelude::Color,
    render_asset::{PrepareAssetSet, RenderAssets},
    render_phase::{AddRenderCommand, DrawFunctions, RenderPhase},
    render_resource::{
        AsBindGroup, AsBindGroupError, BindGroup, BindGroupDescriptor, BindGroupEntry,
        BindingResource, OwnedBindingResource, PipelineCache, RenderPipelineDescriptor, ShaderRef,
        SpecializedRenderPipelines,
    },
    renderer::RenderDevice,
    texture::{FallbackImage, Image, DEFAULT_IMAGE_HANDLE},
    view::{ComputedVisibility, ExtractedView, ViewUniforms},
    Extract, ExtractSchedule, RenderApp, RenderSet,
};
use bevy_sprite::{SpriteAssetEvents, TextureAtlas};
#[cfg(feature = "bevy_text")]
use bevy_text::{Text, TextLayoutInfo};
use bevy_transform::prelude::GlobalTransform;
use bevy_utils::{FloatOrd, HashMap, HashSet};
#[cfg(feature = "bevy_text")]
use bevy_window::{PrimaryWindow, Window};

use std::hash::Hash;

use crate::{
    BackgroundColor, CalculatedClip, DrawUi, Node,
    RenderUiSystem, TransparentUi, UiImage, UiMeta, UiPipeline, UiPipelineKey, UiStack,
};

use super::reset_extracted_ui_nodes;

/// Docs for UiMaterial!
pub trait UiMaterial: AsBindGroup + Send + Sync + Clone + TypeUuid + Sized + 'static {
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Default
    }

    #[allow(unused_variables)]
    #[inline]
    fn specialize(descriptor: &mut RenderPipelineDescriptor, key: UiPipelineKey<Self>) -> () {
        let _ = descriptor.label.insert("ui_pipeline".into());
        ()
    }
}

pub struct UiMaterialPlugin<M: UiMaterial>(pub PhantomData<M>);

impl<M: UiMaterial> Plugin for UiMaterialPlugin<M>
where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    fn build(&self, app: &mut App) {
        app.add_asset::<M>()
            .add_plugin(ExtractComponentPlugin::<Handle<M>>::extract_visible());

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<UiPipeline<M>>()
                .init_resource::<SpecializedRenderPipelines<UiPipeline<M>>>()
                .init_resource::<ExtractedUiMaterials<M>>()
                .init_resource::<RenderUiMaterials<M>>()
                .add_render_command::<TransparentUi, DrawUi<M>>()
                .add_systems(
                    (
                        extract_uinodes::<M>.in_set(RenderUiSystem::ExtractNode).after(reset_extracted_ui_nodes),
                        extract_ui_materials::<M>,
                        #[cfg(feature = "bevy_text")]
                        extract_text_uinodes::<M>.after(RenderUiSystem::ExtractNode),
                    )
                        .in_schedule(ExtractSchedule),
                )
                .add_system(
                    prepare_ui_materials::<M>
                        .in_set(RenderSet::Prepare)
                        .after(PrepareAssetSet::PreAssetPrepare),
                )
                .add_system(queue_uinodes::<M>.in_set(RenderSet::Queue));
        }
    }
}

impl<M: UiMaterial> Default for UiMaterialPlugin<M> {
    fn default() -> Self {
        Self(Default::default())
    }
}

/// Used to expose [`UiMaterial`] updates from the main world to the render world
#[derive(Resource)]
struct ExtractedUiMaterials<M: UiMaterial> {
    extracted: Vec<(Handle<M>, M)>,
    removed: Vec<Handle<M>>,
}

impl<M: UiMaterial> Default for ExtractedUiMaterials<M> {
    fn default() -> Self {
        Self {
            extracted: Default::default(),
            removed: Default::default(),
        }
    }
}

/// Data prepared for a [`UiMaterial`] instance.
pub struct PreparedUiMaterial<T: UiMaterial> {
    pub bindings: Vec<OwnedBindingResource>,
    pub bind_group: BindGroup,
    pub key: T::Data,
}

/// Stores all prepared representations of [`UiMaterial`] assets for as long as they exist.
#[derive(Resource, Deref, DerefMut)]
pub struct RenderUiMaterials<T: UiMaterial>(HashMap<Handle<T>, PreparedUiMaterial<T>>);

impl<T: UiMaterial> Default for RenderUiMaterials<T> {
    fn default() -> Self {
        Self(Default::default())
    }
}

/// This system extracts all created or modified assets of the corresponding [`Material2d`] type
/// into the "render world".
fn extract_ui_materials<M: UiMaterial>(
    mut commands: Commands,
    mut events: Extract<EventReader<AssetEvent<M>>>,
    assets: Extract<Res<Assets<M>>>,
) {
    let mut changed_assets = HashSet::default();
    let mut removed = Vec::new();
    for event in events.iter() {
        match event {
            AssetEvent::Created { handle } | AssetEvent::Modified { handle } => {
                changed_assets.insert(handle.clone_weak());
            }
            AssetEvent::Removed { handle } => {
                changed_assets.remove(handle);
                removed.push(handle.clone_weak());
            }
        }
    }

    let mut extracted_assets = Vec::new();
    for handle in changed_assets.drain() {
        if let Some(asset) = assets.get(&handle) {
            extracted_assets.push((handle, asset.clone()));
        }
    }

    commands.insert_resource(ExtractedUiMaterials {
        extracted: extracted_assets,
        removed,
    });
}

/// All [`UiMaterial`] values of a given type that should be prepared next frame.
pub struct PrepareNextFrameMaterials<M: UiMaterial> {
    assets: Vec<(Handle<M>, M)>,
}

impl<M: UiMaterial> Default for PrepareNextFrameMaterials<M> {
    fn default() -> Self {
        Self {
            assets: Default::default(),
        }
    }
}

/// This system prepares all assets of the corresponding [`UiMaterial`] types
/// which where extracted this frame for the GPU.
fn prepare_ui_materials<M: UiMaterial>(
    mut prepare_next_frame: Local<PrepareNextFrameMaterials<M>>,
    mut extracted_assets: ResMut<ExtractedUiMaterials<M>>,
    mut render_materials: ResMut<RenderUiMaterials<M>>,
    render_device: Res<RenderDevice>,
    images: Res<RenderAssets<Image>>,
    fallback_image: Res<FallbackImage>,
    pipeline: Res<UiPipeline<M>>,
) {
    let queued_assets = std::mem::take(&mut prepare_next_frame.assets);
    for (handle, material) in queued_assets {
        match prepare_ui_material(
            &material,
            &render_device,
            &images,
            &fallback_image,
            &pipeline,
        ) {
            Ok(prepared_asset) => {
                render_materials.insert(handle, prepared_asset);
            }
            Err(AsBindGroupError::RetryNextUpdate) => {
                prepare_next_frame.assets.push((handle, material));
            }
        }
    }

    for removed in std::mem::take(&mut extracted_assets.removed) {
        render_materials.remove(&removed);
    }

    for (handle, material) in std::mem::take(&mut extracted_assets.extracted) {
        match prepare_ui_material(
            &material,
            &render_device,
            &images,
            &fallback_image,
            &pipeline,
        ) {
            Ok(prepared_asset) => {
                render_materials.insert(handle, prepared_asset);
            }
            Err(AsBindGroupError::RetryNextUpdate) => {
                prepare_next_frame.assets.push((handle, material));
            }
        }
    }
}

fn prepare_ui_material<M: UiMaterial>(
    material: &M,
    render_device: &RenderDevice,
    images: &RenderAssets<Image>,
    fallback_image: &FallbackImage,
    pipeline: &UiPipeline<M>,
) -> Result<PreparedUiMaterial<M>, AsBindGroupError> {
    let prepared = material.as_bind_group(
        &pipeline.material_layout,
        render_device,
        images,
        fallback_image,
    )?;
    Ok(PreparedUiMaterial {
        bindings: prepared.bindings,
        bind_group: prepared.bind_group,
        key: prepared.data,
    })
}




/* END region "PrepareMaterials" */

/* BEGIN region "ExtractNodes" */




pub struct ExtractedUiNode {
    pub stack_index: usize,
    pub transform: Mat4,
    pub color: Color,
    pub rect: Rect,
    pub image: Handle<Image>,
    pub material: HandleUntyped,
    pub atlas_size: Option<Vec2>,
    pub clip: Option<Rect>,
    pub flip_x: bool,
    pub flip_y: bool,
}

#[derive(Resource)]
pub struct ExtractedUiNodes {
    pub uinodes: Vec<ExtractedUiNode>,
}

impl Default for ExtractedUiNodes {
    fn default() -> Self {
        Self {
            uinodes: Default::default(),
        }
    }
}

pub fn extract_uinodes<M: UiMaterial>(
    mut extracted_uinodes: ResMut<ExtractedUiNodes>,
    images: Extract<Res<Assets<Image>>>,
    ui_stack: Extract<Res<UiStack>>,
    uinode_query: Extract<
        Query<(
            &Node,
            &GlobalTransform,
            &BackgroundColor,
            Option<&UiImage>,
            &Handle<M>,
            &ComputedVisibility,
            Option<&CalculatedClip>,
        )>,
    >,
) {
    for (stack_index, entity) in ui_stack.uinodes.iter().enumerate() {
        // only get entities that have a handle to the UiMaterial M
        if let Ok((uinode, transform, color, maybe_image, material, visibility, clip)) =
            uinode_query.get(*entity)
        {
            // Skip invisible and completely transparent nodes
            if !visibility.is_visible() || color.0.a() == 0.0 {
                continue;
            }

            let (image, flip_x, flip_y) = if let Some(image) = maybe_image {
                // Skip loading images
                if !images.contains(&image.texture) {
                    continue;
                }
                (image.texture.clone_weak(), image.flip_x, image.flip_y)
            } else {
                (DEFAULT_IMAGE_HANDLE.typed().clone_weak(), false, false)
            };
            
            extracted_uinodes.uinodes.push(ExtractedUiNode {
                stack_index,
                transform: transform.compute_matrix(),
                color: color.0,
                rect: Rect {
                    min: Vec2::ZERO,
                    max: uinode.calculated_size,
                },
                image,
                material: material.clone_weak_untyped(),
                atlas_size: None,
                clip: clip.map(|clip| clip.clip),
                flip_x,
                flip_y,
            });
        }
    }
}

#[cfg(feature = "bevy_text")]
pub fn extract_text_uinodes<M: UiMaterial>(
    mut extracted_uinodes: ResMut<ExtractedUiNodes>,
    texture_atlases: Extract<Res<Assets<TextureAtlas>>>,
    windows: Extract<Query<&Window, With<PrimaryWindow>>>,
    ui_stack: Extract<Res<UiStack>>,
    uinode_query: Extract<
        Query<(
            &Node,
            &GlobalTransform,
            &Text,
            &TextLayoutInfo,
            &Handle<M>,
            &ComputedVisibility,
            Option<&CalculatedClip>,
        )>,
    >,
) {
    // TODO: Support window-independent UI scale: https://github.com/bevyengine/bevy/issues/5621
    let scale_factor = windows
        .get_single()
        .map(|window| window.resolution.scale_factor() as f32)
        .unwrap_or(1.0);

    for (stack_index, entity) in ui_stack.uinodes.iter().enumerate() {
        if let Ok((uinode, global_transform, text, text_layout_info, material, visibility, clip)) =
            uinode_query.get(*entity)
        {
            if !visibility.is_visible() {
                continue;
            }
            // Skip if size is set to zero (e.g. when a parent is set to `Display::None`)
            if uinode.size() == Vec2::ZERO {
                continue;
            }

            let text_glyphs = &text_layout_info.glyphs;
            let alignment_offset = (uinode.size() / -2.0).extend(0.0);

            let mut color = Color::WHITE;
            let mut current_section = usize::MAX;
            for text_glyph in text_glyphs {
                if text_glyph.section_index != current_section {
                    color = text.sections[text_glyph.section_index]
                        .style
                        .color
                        .as_rgba_linear();
                    current_section = text_glyph.section_index;
                }
                let atlas = texture_atlases
                    .get(&text_glyph.atlas_info.texture_atlas)
                    .unwrap();
                let texture = atlas.texture.clone_weak();
                let index = text_glyph.atlas_info.glyph_index;
                let rect = atlas.textures[index];
                let atlas_size = Some(atlas.size);

                // NOTE: Should match `bevy_text::text2d::extract_text2d_sprite`
                let extracted_transform = global_transform.compute_matrix()
                    * Mat4::from_scale(Vec3::splat(scale_factor.recip()))
                    * Mat4::from_translation(
                        alignment_offset * scale_factor + text_glyph.position.extend(0.),
                    );

                extracted_uinodes.uinodes.push(ExtractedUiNode {
                    stack_index,
                    transform: extracted_transform,
                    color,
                    rect,
                    image: texture,
                    material: material.clone_weak_untyped(), // TODO!!!
                    atlas_size,
                    clip: clip.map(|clip| clip.clip),
                    flip_x: false,
                    flip_y: false,
                });
            }
        }
    }
}

/* END region "ExtractNodes" */
/* BEGIN region "NodesAndBatches" */


#[derive(Component)]
pub struct UiBatch {
    pub range: Range<u32>,
    pub image: Handle<Image>,
    pub material: HandleUntyped,
    pub z: f32,
}

#[derive(Resource, Default)]
pub struct UiImageBindGroups {
    pub values: HashMap<Handle<Image>, BindGroup>,
}

#[allow(clippy::too_many_arguments)]
pub fn queue_uinodes<M: UiMaterial>(
    draw_functions: Res<DrawFunctions<TransparentUi>>,
    render_device: Res<RenderDevice>,
    mut ui_meta: ResMut<UiMeta>,
    view_uniforms: Res<ViewUniforms>,
    ui_pipeline: Res<UiPipeline<M>>,
    mut pipelines: ResMut<SpecializedRenderPipelines<UiPipeline<M>>>,
    pipeline_cache: Res<PipelineCache>,
    mut image_bind_groups: ResMut<UiImageBindGroups>,
    render_materials: Res<RenderUiMaterials<M>>,
    gpu_images: Res<RenderAssets<Image>>,
    ui_batches: Query<(Entity, &UiBatch)>,
    mut views: Query<(&ExtractedView, &mut RenderPhase<TransparentUi>)>,
    events: Res<SpriteAssetEvents>,
) where
    M::Data: PartialEq + Eq + Hash + Clone,
{   
    // If an image has changed, the GpuImage has (probably) changed
    for event in &events.images {
        match event {
            AssetEvent::Created { .. } => None,
            AssetEvent::Modified { handle } | AssetEvent::Removed { handle } => {
                image_bind_groups.values.remove(handle)
            }
        };
    }

    if let Some(view_binding) = view_uniforms.uniforms.binding() {
        ui_meta.view_bind_group = Some(render_device.create_bind_group(&BindGroupDescriptor {
            entries: &[BindGroupEntry {
                binding: 0,
                resource: view_binding,
            }],
            label: Some("ui_view_bind_group"),
            layout: &ui_pipeline.view_layout,
        }));

        let draw_ui_function = draw_functions.read().id::<DrawUi<M>>();

        for (view, mut transparent_phase) in &mut views {
            // this utilizes the UiBatches created in prepare_uinodes
            // should we actually still cache the image bind groups?
            // or should we cache the bind groups of the CustomUiMaterialPipeline type?
            for (entity, batch) in &ui_batches {
                if let HandleId::Id(type_uuid, _) = batch.material.id() {
                    if M::TYPE_UUID != type_uuid {
                        // batch isn't for this material
                        continue;
                    }
                }

                let typed_material = &batch.material.clone_weak().typed::<M>();

                // gets the render pipeline (UiPipeline) from the cache (keyed on UiPipelineKey) or creates it
                if let Some(ui_material) = render_materials.get(typed_material) {
                    let pipeline = pipelines.specialize(
                        &pipeline_cache,
                        &ui_pipeline,
                        UiPipelineKey {
                            hdr: view.hdr,
                            bind_group_data: ui_material.key.clone(),
                        },
                    );
                    image_bind_groups // these are just cached frame to frame
                        .values
                        .entry(batch.image.clone_weak()) // getting the bind group keyed on the image handle
                        .or_insert_with(|| {
                            // otherwise we have to create a bind group for the unseen image
                            let gpu_image = gpu_images.get(&batch.image).unwrap();
                            // these two bind group entries are specific to images, shader creators probably
                            // want to be able to do this kind of stuff
                            // we are assigning the color as a vertex color in prepare_uinodes
                            render_device.create_bind_group(&BindGroupDescriptor {
                                entries: &[
                                    BindGroupEntry {
                                        binding: 0,
                                        resource: BindingResource::TextureView(
                                            &gpu_image.texture_view,
                                        ),
                                    },
                                    BindGroupEntry {
                                        binding: 1,
                                        resource: BindingResource::Sampler(&gpu_image.sampler),
                                    },
                                ],
                                label: Some("ui_material_bind_group"),
                                layout: &ui_pipeline.image_layout,
                            })
                        });

                    transparent_phase.add(TransparentUi {
                        draw_function: draw_ui_function,
                        pipeline,
                        entity,
                        sort_key: FloatOrd(batch.z),
                    });
                } else {
                    // this batch had a different material
                }
            }
        }
    }
}
