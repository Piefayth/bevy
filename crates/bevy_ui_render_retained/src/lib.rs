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
mod damage;
mod image;
mod layer;
mod paint;
mod scene;

pub use damage::{DamageJournal, PhysicalRect, RepairPlan};
pub use layer::{RetainedUiLayerCounters, RetainedUiLayerWork, RetainedUiRenderPlugin};
pub use paint::{FloatBits, PaintRecord, RetainedPaint, UpdateOutcome, WorkCounters};
pub use scene::RetainedUiPaintCounters;
