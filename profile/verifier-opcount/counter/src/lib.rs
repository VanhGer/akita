//! Global operation counters for measuring one Akita verification.
//!
//! The patched `jolt-field` and `blake2` copies under `opcount/vendor/` and a
//! few verifier call sites call [`bump`]. Counting is off until [`enable`]
//! runs, so the prover's work is not counted.

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;

static ENABLED: AtomicBool = AtomicBool::new(false);

/// fp128 multiply with reduction mod q (`a * b`, `mul_add`, `square`).
pub static FP128_MUL: AtomicU64 = AtomicU64::new(0);
/// fp128 128x128 -> 256-bit product without reduction (`mul_unreduced`).
pub static FP128_MUL_WIDE: AtomicU64 = AtomicU64::new(0);
/// fp128 x u64 product without reduction (`mul_u64_unreduced`).
pub static FP128_MUL_U64: AtomicU64 = AtomicU64::new(0);
/// fp128 x multi-limb integer product (`mul_wide_limbs`).
pub static FP128_MUL_LIMBS: AtomicU64 = AtomicU64::new(0);
/// fp128 x i32 lane scaling (`scale_wide`).
pub static FP128_SCALE_I32: AtomicU64 = AtomicU64::new(0);
/// fp128 addition, subtraction and negation mod q.
pub static FP128_ADD_SUB: AtomicU64 = AtomicU64::new(0);
/// Reduction of an unreduced accumulator back to fp128.
pub static FP128_REDUCE: AtomicU64 = AtomicU64::new(0);
/// fp128 inversion (its internal squarings also land in `FP128_MUL`).
pub static FP128_INVERSE: AtomicU64 = AtomicU64::new(0);
/// `mul_base_unreduced` calls: a ring coefficient times an alpha power. Each
/// one is also counted in `FP128_MUL_WIDE`. Inside the setup scan the ring
/// coefficient is a public setup entry.
pub static MUL_BASE_UNREDUCED: AtomicU64 = AtomicU64::new(0);
/// SIMD-packed field multiplies. These bypass the scalar fp128 counters, so a
/// verification must leave this at zero for the totals to be complete.
pub static PACKED_MUL: AtomicU64 = AtomicU64::new(0);
/// BLAKE2 compression-function calls (Fiat-Shamir transcript).
pub static BLAKE2_COMPRESS: AtomicU64 = AtomicU64::new(0);
/// Keccak-f[1600] permutations in the fold-challenge sampler.
pub static KECCAK_F1600: AtomicU64 = AtomicU64::new(0);

/// Count one operation if counting is enabled.
#[inline(always)]
pub fn bump(counter: &AtomicU64) {
    if ENABLED.load(Relaxed) {
        counter.fetch_add(1, Relaxed);
    }
}

/// A snapshot of every counter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub fp128_mul: u64,
    pub fp128_mul_wide: u64,
    pub fp128_mul_u64: u64,
    pub fp128_mul_limbs: u64,
    pub fp128_scale_i32: u64,
    pub fp128_add_sub: u64,
    pub fp128_reduce: u64,
    pub fp128_inverse: u64,
    pub mul_base_unreduced: u64,
    pub packed_mul: u64,
    pub blake2_compress: u64,
    pub keccak_f1600: u64,
}

impl Counts {
    /// All 128-bit multiplications: reduced plus wide.
    pub fn fp128_mul_total(&self) -> u64 {
        self.fp128_mul + self.fp128_mul_wide
    }

    fn delta(self, start: Self) -> Self {
        Self {
            fp128_mul: self.fp128_mul - start.fp128_mul,
            fp128_mul_wide: self.fp128_mul_wide - start.fp128_mul_wide,
            fp128_mul_u64: self.fp128_mul_u64 - start.fp128_mul_u64,
            fp128_mul_limbs: self.fp128_mul_limbs - start.fp128_mul_limbs,
            fp128_scale_i32: self.fp128_scale_i32 - start.fp128_scale_i32,
            fp128_add_sub: self.fp128_add_sub - start.fp128_add_sub,
            fp128_reduce: self.fp128_reduce - start.fp128_reduce,
            fp128_inverse: self.fp128_inverse - start.fp128_inverse,
            mul_base_unreduced: self.mul_base_unreduced - start.mul_base_unreduced,
            packed_mul: self.packed_mul - start.packed_mul,
            blake2_compress: self.blake2_compress - start.blake2_compress,
            keccak_f1600: self.keccak_f1600 - start.keccak_f1600,
        }
    }
}

impl fmt::Display for Counts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "fp128 mul (reduced)      {:>12}", self.fp128_mul)?;
        writeln!(f, "fp128 mul (wide)         {:>12}", self.fp128_mul_wide)?;
        writeln!(f, "fp128 mul total          {:>12}", self.fp128_mul_total())?;
        writeln!(f, "fp128 x u64              {:>12}", self.fp128_mul_u64)?;
        writeln!(f, "fp128 x limbs            {:>12}", self.fp128_mul_limbs)?;
        writeln!(f, "fp128 x i32 scale        {:>12}", self.fp128_scale_i32)?;
        writeln!(f, "fp128 add/sub            {:>12}", self.fp128_add_sub)?;
        writeln!(f, "fp128 reduce             {:>12}", self.fp128_reduce)?;
        writeln!(f, "fp128 inverse            {:>12}", self.fp128_inverse)?;
        writeln!(
            f,
            "  of wide: ring x alpha  {:>12}",
            self.mul_base_unreduced
        )?;
        writeln!(f, "packed (must be 0)       {:>12}", self.packed_mul)?;
        writeln!(f, "blake2 compressions      {:>12}", self.blake2_compress)?;
        write!(f, "keccak-f1600             {:>12}", self.keccak_f1600)
    }
}

/// Read every counter.
pub fn snapshot() -> Counts {
    Counts {
        fp128_mul: FP128_MUL.load(Relaxed),
        fp128_mul_wide: FP128_MUL_WIDE.load(Relaxed),
        fp128_mul_u64: FP128_MUL_U64.load(Relaxed),
        fp128_mul_limbs: FP128_MUL_LIMBS.load(Relaxed),
        fp128_scale_i32: FP128_SCALE_I32.load(Relaxed),
        fp128_add_sub: FP128_ADD_SUB.load(Relaxed),
        fp128_reduce: FP128_REDUCE.load(Relaxed),
        fp128_inverse: FP128_INVERSE.load(Relaxed),
        mul_base_unreduced: MUL_BASE_UNREDUCED.load(Relaxed),
        packed_mul: PACKED_MUL.load(Relaxed),
        blake2_compress: BLAKE2_COMPRESS.load(Relaxed),
        keccak_f1600: KECCAK_F1600.load(Relaxed),
    }
}

static SECTIONS: Mutex<Vec<(String, Counts)>> = Mutex::new(Vec::new());

/// Zero every counter and section, then start counting.
pub fn enable() {
    for counter in [
        &FP128_MUL,
        &FP128_MUL_WIDE,
        &FP128_MUL_U64,
        &FP128_MUL_LIMBS,
        &FP128_SCALE_I32,
        &FP128_ADD_SUB,
        &FP128_REDUCE,
        &FP128_INVERSE,
        &MUL_BASE_UNREDUCED,
        &PACKED_MUL,
        &BLAKE2_COMPRESS,
        &KECCAK_F1600,
    ] {
        counter.store(0, Relaxed);
    }
    SECTIONS.lock().unwrap().clear();
    ENABLED.store(true, Relaxed);
}

/// Stop counting and return the totals plus every recorded section, in the
/// order the sections finished.
pub fn disable() -> (Counts, Vec<(String, Counts)>) {
    ENABLED.store(false, Relaxed);
    (snapshot(), std::mem::take(&mut *SECTIONS.lock().unwrap()))
}

/// Records the counter delta between its creation and its drop.
pub struct Section {
    label: String,
    start: Counts,
}

/// Start a labelled section; its delta is recorded when it drops.
pub fn section(label: impl Into<String>) -> Section {
    Section {
        label: label.into(),
        start: snapshot(),
    }
}

impl Drop for Section {
    fn drop(&mut self) {
        if ENABLED.load(Relaxed) {
            let delta = snapshot().delta(self.start);
            SECTIONS
                .lock()
                .unwrap()
                .push((std::mem::take(&mut self.label), delta));
        }
    }
}
