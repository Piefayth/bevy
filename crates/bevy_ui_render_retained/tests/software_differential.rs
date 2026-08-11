//! Differential proof for exact retained repair against full repaint.

use bevy_ui_render_retained::{PaintRecord, PhysicalRect, RepairPlan, RetainedPaint};

const WIDTH: i32 = 16;
const HEIGHT: i32 = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Command {
    order: i32,
    color: [u8; 4],
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Canvas(Vec<[u8; 4]>);

impl Canvas {
    fn transparent() -> Self {
        Self(vec![[0; 4]; (WIDTH * HEIGHT) as usize])
    }

    fn clear(&mut self, rect: PhysicalRect) {
        self.for_each_pixel(rect, |pixel| *pixel = [0; 4]);
    }

    fn draw(&mut self, rect: PhysicalRect, color: [u8; 4]) {
        self.for_each_pixel(rect, |destination| {
            *destination = blend(color, *destination);
        });
    }

    fn for_each_pixel(&mut self, rect: PhysicalRect, mut operation: impl FnMut(&mut [u8; 4])) {
        for y in rect.min_y().max(0)..rect.max_y().min(HEIGHT) {
            for x in rect.min_x().max(0)..rect.max_x().min(WIDTH) {
                operation(&mut self.0[(y * WIDTH + x) as usize]);
            }
        }
    }
}

fn blend(source: [u8; 4], destination: [u8; 4]) -> [u8; 4] {
    let source_alpha = u32::from(source[3]);
    let inverse_alpha = 255 - source_alpha;
    let mut output = [0; 4];
    for channel in 0..3 {
        output[channel] = ((u32::from(source[channel]) * source_alpha
            + u32::from(destination[channel]) * inverse_alpha
            + 127)
            / 255) as u8;
    }
    output[3] = (source_alpha + (u32::from(destination[3]) * inverse_alpha + 127) / 255) as u8;
    output
}

fn rect(min_x: i32, min_y: i32, max_x: i32, max_y: i32) -> PhysicalRect {
    PhysicalRect::from_min_max(min_x, min_y, max_x, max_y).unwrap()
}

fn record(coverage: PhysicalRect, order: i32, color: [u8; 4]) -> PaintRecord<Command> {
    PaintRecord {
        coverage: coverage.into(),
        value: Command { order, color },
    }
}

fn ordered_records(paint: &RetainedPaint<u32, Command>) -> Vec<(u32, &PaintRecord<Command>)> {
    let mut records: Vec<_> = paint.iter().map(|(&id, record)| (id, record)).collect();
    records.sort_by_key(|(id, record)| (record.value.order, *id));
    records
}

fn full_repaint(paint: &RetainedPaint<u32, Command>) -> Canvas {
    let mut canvas = Canvas::transparent();
    for (_, record) in ordered_records(paint) {
        for &coverage in record.coverage.iter() {
            canvas.draw(coverage, record.value.color);
        }
    }
    canvas
}

fn repair(canvas: &mut Canvas, paint: &RetainedPaint<u32, Command>, plan: &RepairPlan) {
    let records = ordered_records(paint);
    for &region in plan.regions() {
        canvas.clear(region);
        for (_, record) in &records {
            for &coverage in record.coverage.iter() {
                if let Some(scissored_coverage) = coverage.intersection(region) {
                    canvas.draw(scissored_coverage, record.value.color);
                }
            }
        }
    }
}

fn repair_and_compare(canvas: &mut Canvas, paint: &mut RetainedPaint<u32, Command>) {
    let plan = paint.repair_plan().expect("the mutation must cause damage");
    repair(canvas, paint, &plan);
    paint.acknowledge(&plan);
    assert_eq!(*canvas, full_repaint(paint));
}

#[test]
fn exact_repairs_match_full_repaint_through_an_adversarial_sequence() {
    let mut paint = RetainedPaint::default();
    let mut retained = Canvas::transparent();

    paint.upsert(0, record(rect(0, 0, 16, 12), 0, [20, 30, 80, 255]));
    paint.upsert(1, record(rect(2, 2, 11, 9), 10, [220, 40, 30, 128]));
    paint.upsert(2, record(rect(5, 4, 9, 7), 20, [20, 240, 90, 192]));
    repair_and_compare(&mut retained, &mut paint);

    paint.upsert(1, record(rect(8, 1, 15, 8), 10, [220, 40, 30, 128]));
    repair_and_compare(&mut retained, &mut paint);

    paint.remove(&2);
    repair_and_compare(&mut retained, &mut paint);

    paint.upsert(3, record(rect(6, 3, 13, 10), 15, [240, 210, 10, 96]));
    repair_and_compare(&mut retained, &mut paint);

    paint.upsert(1, record(rect(8, 1, 15, 8), 30, [220, 40, 30, 128]));
    repair_and_compare(&mut retained, &mut paint);
}
