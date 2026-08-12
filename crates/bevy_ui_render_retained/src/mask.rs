//! Exact damage-mask binding shared by retained pipelines.

use bevy::render::render_resource::{
    binding_types::texture_2d, BindGroupLayoutDescriptor, BindGroupLayoutEntries, ShaderStages,
    TextureSampleType,
};

pub(crate) fn layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "retained_ui_damage_mask_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::FRAGMENT,
            texture_2d(TextureSampleType::Float { filterable: false }),
        ),
    )
}
