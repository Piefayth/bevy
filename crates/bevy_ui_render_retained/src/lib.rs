#![doc(
    html_logo_url = "https://bevyengine.org/assets/icon.png",
    html_favicon_url = "https://bevyengine.org/assets/icon.png"
)]

//! A damage-tracked retained renderer for Bevy UI.
//!
//! The implementation is being built capability by capability. The design and
//! verification contract live in `docs/retained_ui.md` at the workspace root.

#![forbid(unsafe_code)]

extern crate alloc;

mod background;
mod border;
mod core;
mod damage;
mod gradient;
mod gradient_render;
mod image;
mod layer;
mod mask;
mod material;
mod paint;
mod quiescence;
mod sampled_image;
mod scene;
mod shadow;
mod shadow_render;
mod text;
mod viewport;

pub use damage::{DamageJournal, PhysicalRect, RepairPlan};
pub use layer::{RetainedUiLayerCounters, RetainedUiLayerWork, RetainedUiRenderPlugin};
pub use material::{
    RetainedUiMaterial, RetainedUiMaterialCoverage, RetainedUiMaterialImage, RetainedUiMaterialKey,
    RetainedUiMaterialPlugin, RetainedUiMaterialSnapshot,
};
pub use paint::{
    FloatBits, PaintCoverage, PaintRecord, RetainedPaint, UpdateOutcome, WorkCounters,
};
pub use quiescence::{
    RetainedUiMainWorldCounters, RetainedUiMainWorldPlugin, RetainedUiMainWorldWork,
};
pub use sampled_image::RetainedUiImageWrites;
pub use scene::RetainedUiPaintCounters;
