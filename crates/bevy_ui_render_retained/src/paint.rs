//! Canonical retained paint records.

use crate::{DamageJournal, PhysicalRect, RepairPlan};
use core::hash::Hash;
use std::collections::{hash_map::Entry, HashMap};

/// Exact physical regions covered by one retained paint record.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PaintCoverage(CoverageStorage);

#[derive(Clone, Debug, Default, PartialEq, Eq)]
enum CoverageStorage {
    #[default]
    Empty,
    One(PhysicalRect),
    Many(Box<[PhysicalRect]>),
}

impl PaintCoverage {
    /// No visible pixels.
    pub fn empty() -> Self {
        Self(CoverageStorage::Empty)
    }

    /// One visible rectangle without a heap allocation.
    pub fn one(region: PhysicalRect) -> Self {
        Self(CoverageStorage::One(region))
    }

    /// Builds coverage without allocating for zero or one rectangle.
    pub fn from_regions(regions: impl IntoIterator<Item = PhysicalRect>) -> Self {
        let mut regions = regions.into_iter();
        let Some(first) = regions.next() else {
            return Self::empty();
        };
        let Some(second) = regions.next() else {
            return Self::one(first);
        };
        let mut many = Vec::with_capacity(regions.size_hint().0.saturating_add(2));
        many.extend([first, second]);
        many.extend(regions);
        Self(CoverageStorage::Many(many.into_boxed_slice()))
    }

    /// Returns whether the record covers no pixels.
    pub fn is_empty(&self) -> bool {
        matches!(self.0, CoverageStorage::Empty)
    }

    /// Iterates over exact coverage rectangles.
    pub fn iter(&self) -> impl Iterator<Item = &PhysicalRect> {
        self.as_slice().iter()
    }

    /// Appends exact coverage rectangles.
    pub fn extend(&mut self, regions: impl IntoIterator<Item = PhysicalRect>) {
        let previous = core::mem::take(self);
        *self = Self::from_regions(previous.iter().copied().chain(regions));
    }

    fn as_slice(&self) -> &[PhysicalRect] {
        match &self.0 {
            CoverageStorage::Empty => &[],
            CoverageStorage::One(region) => core::slice::from_ref(region),
            CoverageStorage::Many(regions) => regions,
        }
    }
}

impl From<PhysicalRect> for PaintCoverage {
    fn from(region: PhysicalRect) -> Self {
        Self::one(region)
    }
}

impl FromIterator<PhysicalRect> for PaintCoverage {
    fn from_iter<T: IntoIterator<Item = PhysicalRect>>(iter: T) -> Self {
        Self::from_regions(iter)
    }
}

impl<'a> IntoIterator for &'a PaintCoverage {
    type Item = &'a PhysicalRect;
    type IntoIter = core::slice::Iter<'a, PhysicalRect>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

/// The exact bit representation of a paint-input `f32`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FloatBits(u32);

impl FloatBits {
    /// Stores a float without normalizing zeros or NaNs.
    pub const fn new(value: f32) -> Self {
        Self(value.to_bits())
    }

    /// Returns the stored float.
    pub const fn get(self) -> f32 {
        f32::from_bits(self.0)
    }
}

/// Everything required to prove whether one paint family produces the same pixels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PaintRecord<T> {
    /// Conservative physical-pixel regions; empty when nothing is visible.
    pub coverage: PaintCoverage,
    /// Canonical commands, ordering, clips, effects, and resource generations.
    pub value: T,
}

/// The result of applying one candidate paint record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpdateOutcome {
    /// The canonical record was byte-for-byte equivalent.
    Unchanged,
    /// A new record was inserted.
    Inserted,
    /// An existing record changed.
    Changed,
}

/// Deterministic work performed since counters were last taken.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkCounters {
    /// Candidate records submitted by change-driven extraction.
    pub candidates: u64,
    /// Existing canonical records compared.
    pub records_compared: u64,
    /// Records inserted or changed.
    pub records_changed: u64,
    /// Records removed.
    pub records_removed: u64,
    /// Old or new coverage rectangles added to owed damage.
    pub damage_events: u64,
}

/// Canonical paint state keyed by stable entity-and-family identity.
#[derive(Debug)]
pub struct RetainedPaint<K, V> {
    records: HashMap<K, PaintRecord<V>>,
    damage: DamageJournal,
    counters: WorkCounters,
}

impl<K, V> Default for RetainedPaint<K, V> {
    fn default() -> Self {
        Self {
            records: HashMap::new(),
            damage: DamageJournal::default(),
            counters: WorkCounters::default(),
        }
    }
}

impl<K: Eq + Hash, V: PartialEq> RetainedPaint<K, V> {
    /// Inserts or replaces one candidate after comparing its canonical record.
    pub fn upsert(&mut self, id: K, record: PaintRecord<V>) -> UpdateOutcome {
        let Self {
            records,
            damage,
            counters,
        } = self;
        counters.candidates += 1;
        match records.entry(id) {
            Entry::Occupied(mut entry) => {
                counters.records_compared += 1;
                if entry.get() == &record {
                    return UpdateOutcome::Unchanged;
                }
                record_damage(damage, counters, entry.get().coverage.as_slice());
                record_damage(damage, counters, record.coverage.as_slice());
                entry.insert(record);
                counters.records_changed += 1;
                UpdateOutcome::Changed
            }
            Entry::Vacant(entry) => {
                record_damage(damage, counters, record.coverage.as_slice());
                entry.insert(record);
                counters.records_changed += 1;
                UpdateOutcome::Inserted
            }
        }
    }

    /// Removes one paint record and damages the pixels it occupied.
    pub fn remove(&mut self, id: &K) -> bool {
        let Some(record) = self.records.remove(id) else {
            return false;
        };
        record_damage(
            &mut self.damage,
            &mut self.counters,
            record.coverage.as_slice(),
        );
        self.counters.records_removed += 1;
        true
    }

    /// Returns a canonical paint record.
    pub fn get(&self, id: &K) -> Option<&PaintRecord<V>> {
        self.records.get(id)
    }

    /// Iterates over all canonical records in unspecified order.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &PaintRecord<V>)> {
        self.records.iter()
    }

    /// Plans all currently owed physical-pixel repairs.
    pub fn repair_plan(&self) -> Option<RepairPlan> {
        self.damage.plan()
    }

    /// Acknowledges damage only after its complete repair was encoded.
    pub fn acknowledge(&mut self, plan: &RepairPlan) {
        self.damage.acknowledge(plan);
    }

    /// Invalidates pixels because the surface holding otherwise unchanged records was lost.
    pub fn invalidate(&mut self, coverage: PhysicalRect) {
        record_damage(&mut self.damage, &mut self.counters, &[coverage]);
    }

    /// Takes and resets deterministic work counters.
    pub fn take_counters(&mut self) -> WorkCounters {
        core::mem::take(&mut self.counters)
    }
}

fn record_damage(
    damage: &mut DamageJournal,
    counters: &mut WorkCounters,
    coverage: &[PhysicalRect],
) {
    for &rect in coverage {
        damage.record(rect);
        counters.damage_events += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(min_x: i32, min_y: i32, max_x: i32, max_y: i32) -> PhysicalRect {
        PhysicalRect::from_min_max(min_x, min_y, max_x, max_y).unwrap()
    }

    fn record(coverage: PhysicalRect, value: u32) -> PaintRecord<u32> {
        PaintRecord {
            coverage: coverage.into(),
            value,
        }
    }

    #[test]
    fn identical_candidate_does_not_damage_pixels() {
        let mut paint = RetainedPaint::default();
        let original = record(rect(0, 0, 3, 2), 7);
        assert_eq!(paint.upsert(1, original.clone()), UpdateOutcome::Inserted);
        let initial = paint.repair_plan().unwrap();
        paint.acknowledge(&initial);
        paint.take_counters();

        assert_eq!(paint.upsert(1, original), UpdateOutcome::Unchanged);
        assert!(paint.repair_plan().is_none());
        assert_eq!(
            paint.take_counters(),
            WorkCounters {
                candidates: 1,
                records_compared: 1,
                ..Default::default()
            }
        );
    }

    #[test]
    fn changed_record_damages_exact_old_union_new_coverage() {
        let mut paint = RetainedPaint::default();
        paint.upsert(1, record(rect(0, 0, 2, 2), 7));
        let initial = paint.repair_plan().unwrap();
        paint.acknowledge(&initial);
        paint.take_counters();

        assert_eq!(
            paint.upsert(1, record(rect(1, 1, 3, 3), 8)),
            UpdateOutcome::Changed
        );

        let repair = paint.repair_plan().unwrap();
        assert_eq!(repair.damaged_pixels(), 7);
        assert_eq!(
            paint.take_counters(),
            WorkCounters {
                candidates: 1,
                records_compared: 1,
                records_changed: 1,
                damage_events: 2,
                ..Default::default()
            }
        );
    }

    #[test]
    fn removal_damages_vacated_coverage() {
        let mut paint = RetainedPaint::default();
        paint.upsert(1, record(rect(-2, 4, 3, 6), 7));
        let initial = paint.repair_plan().unwrap();
        paint.acknowledge(&initial);
        paint.take_counters();

        assert!(paint.remove(&1));

        let repair = paint.repair_plan().unwrap();
        assert_eq!(repair.regions(), &[rect(-2, 4, 3, 6)]);
        assert_eq!(repair.damaged_pixels(), 10);
    }

    #[test]
    fn disjoint_record_coverage_does_not_damage_its_gap() {
        let mut paint = RetainedPaint::default();
        paint.upsert(
            1,
            PaintRecord {
                coverage: PaintCoverage::from_regions([rect(0, 0, 2, 2), rect(6, 0, 8, 2)]),
                value: 7,
            },
        );
        let initial = paint.repair_plan().unwrap();
        assert_eq!(initial.damaged_pixels(), 8);
        assert_eq!(initial.regions().len(), 2);
        assert!(initial
            .regions()
            .iter()
            .all(|region| region.intersection(rect(2, 0, 6, 2)).is_none()));
    }

    #[test]
    fn surface_loss_invalidates_unchanged_records() {
        let mut paint = RetainedPaint::<u32, u32>::default();
        paint.upsert(1, record(rect(0, 0, 2, 2), 7));
        let initial = paint.repair_plan().unwrap();
        paint.acknowledge(&initial);
        paint.take_counters();

        paint.invalidate(rect(0, 0, 8, 6));

        let repair = paint.repair_plan().unwrap();
        assert_eq!(repair.regions(), &[rect(0, 0, 8, 6)]);
        assert_eq!(
            paint.take_counters(),
            WorkCounters {
                damage_events: 1,
                ..Default::default()
            }
        );
    }

    #[test]
    fn float_comparison_uses_bits() {
        assert_ne!(FloatBits::new(0.0), FloatBits::new(-0.0));
        let nan = f32::from_bits(0x7fc0_0001);
        assert_eq!(FloatBits::new(nan), FloatBits::new(nan));
    }
}
