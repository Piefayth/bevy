//! Exact reverse dependencies for image-backed retained paint.

use bevy::{
    asset::{AssetEvent, AssetId, Assets, RenderAssetUsages},
    ecs::{entity::Entity, message::MessageReader, system::ResMut},
    image::{Image, ImageAddressMode, ImageSampler, ImageSamplerDescriptor},
    math::{Rect, UVec3},
    render::{
        render_asset::RenderAssets,
        render_resource::{DefaultImageSamplerDescriptor, TextureDimension, TextureUsages},
        texture::GpuImage,
        Extract,
    },
};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ImageReader {
    Node(Entity),
    Text(Entity),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct ImageSample {
    image: AssetId<Image>,
    region: SampleRegion,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum SampleRegion {
    All,
    Rect { min: [i64; 2], max: [i64; 2] },
}

impl ImageSample {
    pub(crate) fn all(image: AssetId<Image>) -> Self {
        Self {
            image,
            region: SampleRegion::All,
        }
    }

    pub(crate) fn rect(image: AssetId<Image>, rect: Rect) -> Self {
        if !rect.min.is_finite()
            || !rect.max.is_finite()
            || rect.min.x > rect.max.x
            || rect.min.y > rect.max.y
        {
            return Self::all(image);
        }
        Self {
            image,
            region: SampleRegion::Rect {
                min: [
                    (rect.min.x.floor() as i64).saturating_sub(1),
                    (rect.min.y.floor() as i64).saturating_sub(1),
                ],
                max: [
                    (rect.max.x.ceil() as i64).saturating_add(1),
                    (rect.max.y.ceil() as i64).saturating_add(1),
                ],
            },
        }
    }

    pub(crate) fn image(self) -> AssetId<Image> {
        self.image
    }
}

#[derive(Default)]
struct ReaderDependencies {
    images: HashSet<AssetId<Image>>,
    samples: HashSet<ImageSample>,
}

struct SampleState {
    pixels: Option<Box<[u8]>>,
    revision: u64,
    readers: HashSet<ImageReader>,
}

struct MetadataState {
    value: Option<Image>,
    revision: u64,
}

#[derive(bevy::prelude::Resource, Default)]
pub(crate) struct RetainedSampledImages {
    readers: HashMap<ImageReader, ReaderDependencies>,
    image_readers: HashMap<AssetId<Image>, HashSet<ImageReader>>,
    metadata: HashMap<AssetId<Image>, MetadataState>,
    samples: HashMap<ImageSample, SampleState>,
    pending: HashSet<AssetId<Image>>,
    nominated: HashSet<ImageReader>,
    next_revision: u64,
}

impl RetainedSampledImages {
    pub(crate) fn replace_reader(
        &mut self,
        reader: ImageReader,
        samples: impl IntoIterator<Item = ImageSample>,
        assets: &Assets<Image>,
        default_sampler: &DefaultImageSamplerDescriptor,
    ) {
        self.detach_reader(reader);
        let mut dependencies = ReaderDependencies::default();
        for requested_sample in samples {
            let image = requested_sample.image();
            let sample = assets.get(image).map_or(requested_sample, |asset| {
                proven_sample(requested_sample, asset, default_sampler)
            });
            if dependencies.samples.insert(sample) {
                self.samples
                    .entry(sample)
                    .or_insert_with(|| SampleState {
                        pixels: assets
                            .get(sample.image())
                            .and_then(|image| sample_pixels(image, sample)),
                        revision: 0,
                        readers: HashSet::default(),
                    })
                    .readers
                    .insert(reader);
            }
            if dependencies.images.insert(image) {
                self.image_readers.entry(image).or_default().insert(reader);
                self.metadata.entry(image).or_insert_with(|| MetadataState {
                    value: assets.get(image).map(image_metadata),
                    revision: 0,
                });
            }
        }
        self.readers.insert(reader, dependencies);
        self.prune_unused();
    }

    pub(crate) fn remove_reader(&mut self, reader: ImageReader) {
        self.detach_reader(reader);
        self.nominated.remove(&reader);
        self.prune_unused();
    }

    fn detach_reader(&mut self, reader: ImageReader) {
        let Some(old) = self.readers.remove(&reader) else {
            return;
        };
        for image in old.images {
            remove_reverse_reader(&mut self.image_readers, image, reader);
        }
        for sample in old.samples {
            if let Some(state) = self.samples.get_mut(&sample) {
                state.readers.remove(&reader);
            }
        }
    }

    fn prune_unused(&mut self) {
        self.metadata
            .retain(|image, _| self.image_readers.contains_key(image));
        self.pending
            .retain(|image| self.image_readers.contains_key(image));
        self.samples.retain(|_, state| !state.readers.is_empty());
    }

    pub(crate) fn mark_pending(&mut self, image: AssetId<Image>, asset: Option<&Image>) {
        if asset.is_some_and(|asset| asset.asset_usage.contains(RenderAssetUsages::RENDER_WORLD)) {
            self.pending.insert(image);
        } else {
            self.pending.remove(&image);
        }
    }

    fn image_changed(&mut self, image: AssetId<Image>, asset: Option<&Image>, pending: bool) {
        let Some(readers) = self.image_readers.get(&image).cloned() else {
            return;
        };
        let mut relevant_change = false;
        let metadata = asset.map(image_metadata);
        let metadata_changed = self
            .metadata
            .get(&image)
            .is_none_or(|old| metadata.is_none() || old.value != metadata);
        if metadata_changed {
            relevant_change = true;
            let revision = self.new_revision();
            self.metadata.insert(
                image,
                MetadataState {
                    value: metadata,
                    revision,
                },
            );
            self.nominated.extend(&readers);
        }

        let samples: Vec<_> = self
            .samples
            .keys()
            .filter(|sample| sample.image() == image)
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
                relevant_change = true;
                let pixels = asset.and_then(|image| sample_pixels(image, sample));
                let revision = self.new_revision();
                let state = self.samples.get_mut(&sample).unwrap();
                state.pixels = pixels;
                state.revision = revision;
                self.nominated.extend(&state.readers);
            }
        }

        if pending && relevant_change {
            self.pending.insert(image);
        } else if !pending {
            self.pending.remove(&image);
        }
    }

    pub(crate) fn take_nodes(&mut self) -> HashSet<Entity> {
        self.take(|reader| match reader {
            ImageReader::Node(entity) => Some(entity),
            ImageReader::Text(_) => None,
        })
    }

    pub(crate) fn take_text(&mut self) -> HashSet<Entity> {
        self.take(|reader| match reader {
            ImageReader::Text(entity) => Some(entity),
            ImageReader::Node(_) => None,
        })
    }

    fn take(&mut self, map: impl Fn(ImageReader) -> Option<Entity>) -> HashSet<Entity> {
        let mut entities = HashSet::new();
        self.nominated.retain(|reader| {
            if let Some(entity) = map(*reader) {
                entities.insert(entity);
                false
            } else {
                true
            }
        });
        entities
    }

    pub(crate) fn revisions(
        &self,
        image: AssetId<Image>,
        samples: impl IntoIterator<Item = ImageSample>,
    ) -> Box<[u64]> {
        let mut samples: Vec<_> = samples.into_iter().collect();
        samples.sort_unstable();
        core::iter::once(self.metadata.get(&image).map_or(0, |state| state.revision))
            .chain(
                samples
                    .into_iter()
                    .map(|sample| self.samples.get(&sample).map_or(0, |state| state.revision)),
            )
            .collect()
    }

    pub(crate) fn is_pending(&self, image: AssetId<Image>) -> bool {
        self.pending.contains(&image)
    }

    pub(crate) fn resolve_ready(&mut self, gpu_images: &RenderAssets<GpuImage>) {
        self.pending.retain(|image| {
            self.image_readers.contains_key(image) && gpu_images.get(*image).is_none()
        });
    }

    fn new_revision(&mut self) -> u64 {
        self.next_revision = self
            .next_revision
            .checked_add(1)
            .expect("retained sampled-image revision exhausted");
        self.next_revision
    }
}

pub(crate) fn extract_sampled_image_changes(
    mut retained: ResMut<RetainedSampledImages>,
    images: Extract<bevy::ecs::system::Res<Assets<Image>>>,
    mut events: Extract<MessageReader<AssetEvent<Image>>>,
) {
    for event in events.read() {
        match *event {
            AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                retained.image_changed(id, images.get(id), true);
            }
            AssetEvent::Unused { id } => retained.image_changed(id, images.get(id), false),
            AssetEvent::Removed { .. } | AssetEvent::LoadedWithDependencies { .. } => {}
        }
    }
}

pub(crate) fn resolve_ready_sampled_images(
    mut retained: ResMut<RetainedSampledImages>,
    gpu_images: bevy::ecs::system::Res<RenderAssets<GpuImage>>,
) {
    retained.resolve_ready(&gpu_images);
}

fn remove_reverse_reader(
    readers: &mut HashMap<AssetId<Image>, HashSet<ImageReader>>,
    image: AssetId<Image>,
    reader: ImageReader,
) {
    let Some(entries) = readers.get_mut(&image) else {
        return;
    };
    entries.remove(&reader);
    if entries.is_empty() {
        readers.remove(&image);
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

fn proven_sample(
    sample: ImageSample,
    image: &Image,
    default_sampler: &DefaultImageSamplerDescriptor,
) -> ImageSample {
    if matches!(sample.region, SampleRegion::All) {
        return sample;
    }
    let descriptor = match &image.sampler {
        ImageSampler::Default => &default_sampler.0,
        ImageSampler::Descriptor(descriptor) => descriptor,
    };
    if image.texture_descriptor.dimension == TextureDimension::D2
        && image.texture_descriptor.size.depth_or_array_layers == 1
        && image.texture_descriptor.mip_level_count == 1
        && sampler_has_local_footprint(descriptor)
    {
        sample
    } else {
        ImageSample::all(sample.image())
    }
}

fn sampler_has_local_footprint(descriptor: &ImageSamplerDescriptor) -> bool {
    descriptor.address_mode_u == ImageAddressMode::ClampToEdge
        && descriptor.address_mode_v == ImageAddressMode::ClampToEdge
        && descriptor.anisotropy_clamp == 1
        && descriptor.compare.is_none()
}

fn sample_pixels(image: &Image, sample: ImageSample) -> Option<Box<[u8]>> {
    match sample.region {
        SampleRegion::All => Some(image.data.as_ref()?.clone().into_boxed_slice()),
        SampleRegion::Rect { .. } => {
            let [min_x, min_y, max_x, max_y] = sample_bounds(image, sample);
            let mut pixels = Vec::new();
            for y in min_y..max_y {
                for x in min_x..max_x {
                    pixels.extend_from_slice(image.pixel_bytes(UVec3::new(x, y, 0)).ok()?);
                }
            }
            Some(pixels.into_boxed_slice())
        }
    }
}

fn sample_matches(image: &Image, sample: ImageSample, old: &[u8]) -> Option<bool> {
    match sample.region {
        SampleRegion::All => Some(image.data.as_deref()? == old),
        SampleRegion::Rect { .. } => {
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
    }
}

fn sample_bounds(image: &Image, sample: ImageSample) -> [u32; 4] {
    let SampleRegion::Rect { min, max } = sample.region else {
        unreachable!("whole-image samples do not have rectangle bounds");
    };
    let size = image.texture_descriptor.size;
    let [min_x, max_x] = clamp_sample_axis(min[0], max[0], size.width);
    let [min_y, max_y] = clamp_sample_axis(min[1], max[1], size.height);
    [min_x, min_y, max_x, max_y]
}

fn clamp_sample_axis(min: i64, max: i64, size: u32) -> [u32; 2] {
    let size = i64::from(size);
    if size == 0 {
        return [0, 0];
    }
    if max <= 0 {
        return [0, 1];
    }
    if min >= size {
        return [(size - 1) as u32, size as u32];
    }
    [min.clamp(0, size) as u32, max.clamp(0, size) as u32]
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::{asset::RenderAssetUsages, math::Vec2, render::render_resource::Extent3d};

    fn test_image() -> Image {
        Image::new_fill(
            Extent3d {
                width: 8,
                height: 2,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            &[0; 8 * 2 * 4],
            bevy::render::render_resource::TextureFormat::Rgba8Unorm,
            RenderAssetUsages::default(),
        )
    }

    #[test]
    fn rectangle_sample_includes_linear_filter_reach() {
        let image = test_image();
        let sample = ImageSample::rect(
            AssetId::default(),
            Rect::from_corners(Vec2::X, Vec2::new(3.0, 1.0)),
        );

        assert_eq!(sample_bounds(&image, sample), [0, 0, 4, 2]);
    }

    #[test]
    fn clamped_outside_sample_still_tracks_the_edge_texel() {
        let image = test_image();
        let sample = ImageSample::rect(
            AssetId::default(),
            Rect::from_corners(Vec2::new(20.0, -9.0), Vec2::new(24.0, -4.0)),
        );

        assert_eq!(sample_bounds(&image, sample), [7, 0, 8, 1]);
    }

    #[test]
    fn repeating_sampler_falls_back_to_the_whole_image() {
        let mut image = test_image();
        let mut descriptor = ImageSamplerDescriptor::linear();
        descriptor.set_address_mode(ImageAddressMode::Repeat);
        image.sampler = ImageSampler::Descriptor(descriptor);
        let sample = ImageSample::rect(
            AssetId::default(),
            Rect::from_corners(Vec2::ZERO, Vec2::ONE),
        );
        let default_sampler = DefaultImageSamplerDescriptor(ImageSamplerDescriptor::linear());

        assert!(matches!(
            proven_sample(sample, &image, &default_sampler).region,
            SampleRegion::All
        ));
    }

    #[test]
    fn mipmapped_image_falls_back_to_the_whole_image() {
        let mut image = test_image();
        image.texture_descriptor.mip_level_count = 2;
        let sample = ImageSample::rect(
            AssetId::default(),
            Rect::from_corners(Vec2::ZERO, Vec2::ONE),
        );
        let default_sampler = DefaultImageSamplerDescriptor(ImageSamplerDescriptor::linear());

        assert!(matches!(
            proven_sample(sample, &image, &default_sampler).region,
            SampleRegion::All
        ));
    }
}
