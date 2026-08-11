use criterion::criterion_main;

mod changed;
mod layout;
mod paint;

criterion_main!(changed::benches, layout::benches, paint::benches);
