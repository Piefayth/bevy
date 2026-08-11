use criterion::criterion_main;

mod changed;
mod layout;
mod paint;
mod replay;

criterion_main!(
    changed::benches,
    layout::benches,
    paint::benches,
    replay::benches
);
