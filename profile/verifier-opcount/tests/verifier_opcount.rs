//! Count the operations of one Akita verification.
//!
//! Proves one fp128 opening, then verifies it once on a single thread with
//! the counters on. Only `batched_verify` is counted: setup, commit, prove,
//! and building the verifier are not.
//!
//! ```text
//! cd profile/verifier-opcount
//! cargo test --release -- --nocapture                          # dense, nv = 16
//! OPCOUNT_NV=24 cargo test --release -- --nocapture            # dense, nv = 24
//! OPCOUNT_POLY=onehot OPCOUNT_NV=20 cargo test --release -- --nocapture
//! ```
//!
//! `OPCOUNT_NV` must be a size the checked-in schedule catalog ships
//! (dense: 14, 16, 24, 26; one-hot: 12, 15, 20, 28, 32).
//!
//! The counts are exact. The AND-gate column multiplies them by estimated
//! garbled-circuit costs per operation, and leaves out the terminal's integer
//! CRT/NTT product, which the field counters do not see.
//!
//! The run checks its own counting:
//! - `hooks_count_each_operation_once` runs known fp128 work and requires
//!   every counter to match it exactly.
//! - The verification runs on one thread and again on four, and both must
//!   report the same counts.
//! - The SIMD-packed multiply, which bypasses the scalar counters, must stay
//!   at zero.
//!
//! Inside the setup scan, `mul_base_unreduced` multiplies a public setup
//! coefficient by an alpha power, so those products are priced as
//! constant x variable. Every other product has two variable operands.

#[path = "../../../crates/akita-pcs/examples/support/workspace_schedules.rs"]
mod workspace_schedules;

use akita_config::proof_optimized::fp128;
use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_cpu_backend::{CommitOutput, CommitmentHandle, CpuBackend, DensePoly, GroupContext};
use akita_cpu_backend::{OneHotPoly, SourceHandle};
use akita_params::{BasisMode, ChunkedWitnessCfg, CommittedGroupBatchProfile};
use akita_prover::SelectedProverOpeningData;
use akita_types::{CommittedGroup, GroupBatchStatement, OpeningClaims, PolynomialGroupClaims};
use jolt_field::{CanonicalEncoding, One, Zero};
use opcount::Counts;
use workspace_schedules::load_workspace_scheme;

type F = fp128::Field;
const STACK_SIZE: usize = 256 * 1024 * 1024;

/// Estimated AND gates per operation in a garbled circuit (free-XOR).
const AND_MUL: u64 = 25_000; // 128x128 multiply + reduce mod 2^128 - 275
const AND_MUL_WIDE: u64 = 24_000; // 128x128 multiply, 256-bit accumulate
const AND_SCAN: u64 = 8_000; // public constant x variable, ~64 shifted adds
const AND_ADD_SUB: u64 = 384; // add + compare + conditional subtract
const AND_REDUCE: u64 = 1_500; // Solinas fold of an accumulator
const AND_BLAKE2B: u64 = 36_288; // 12 rounds x 8 G x 6 64-bit adds x 63
const AND_KECCAK: u64 = 38_400; // 24 rounds x 1600 chi ANDs

/// The counters are process-global, so the tests must not overlap.
static COUNTERS: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn hooks_count_each_operation_once() {
    use jolt_field::{Field, Ring, Unreduced};
    let _guard = COUNTERS.lock().unwrap_or_else(|e| e.into_inner());
    let a = F::from_u128_reduced(0x1234_5678_9abc_def0_1357_9bdf_2468_ace0);
    let b = F::from_u128_reduced(0x0fed_cba9_8765_4321_0246_8ace_1357_9bdf);
    let mut x = a;
    opcount::enable();
    for _ in 0..1000 {
        x = std::hint::black_box(x * b);
    }
    for _ in 0..100 {
        x = std::hint::black_box(x.square());
    }
    for _ in 0..50 {
        x = std::hint::black_box(x.mul_add(a, b));
    }
    for _ in 0..300 {
        x = std::hint::black_box(x + a);
        x = std::hint::black_box(x - b);
    }
    let mut accum = <F as Unreduced>::Product::zero();
    for _ in 0..200 {
        accum += std::hint::black_box(x).mul_unreduced(a);
    }
    x = std::hint::black_box(x + F::reduce_product(accum));
    let (counts, _) = opcount::disable();
    std::hint::black_box(x);
    assert_eq!(counts.fp128_mul, 1000 + 100 + 50, "reduced multiplies");
    assert_eq!(counts.fp128_mul_wide, 200, "wide multiplies");
    assert_eq!(counts.fp128_reduce, 1, "reductions");
    assert_eq!(counts.fp128_add_sub, 600 + 1, "additions and subtractions");
    assert_eq!(counts.packed_mul, 0, "packed multiplies");
}

#[test]
fn count_one_verification() {
    let _guard = COUNTERS.lock().unwrap_or_else(|e| e.into_inner());
    std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(run)
        .expect("spawn")
        .join()
        .expect("opcount run panicked");
}

fn run() {
    let poly = std::env::var("OPCOUNT_POLY").unwrap_or_else(|_| "dense".into());
    let nv: usize = std::env::var("OPCOUNT_NV")
        .map(|v| v.parse().expect("OPCOUNT_NV must be an integer"))
        .unwrap_or(16);
    let (counts, sections) = match poly.as_str() {
        "dense" => count_dense(nv),
        "onehot" => count_onehot(nv),
        other => panic!("OPCOUNT_POLY must be `dense` or `onehot`, got `{other}`"),
    };
    report(&format!("fp128 {poly}, nv = {nv}"), &counts, &sections);
}

fn count_dense(nv: usize) -> (Counts, Vec<(String, Counts)>) {
    let mut state = 0x7e57_0000_u64 ^ nv as u64;
    let evals: Vec<F> = (0..1usize << nv)
        .map(|_| F::from_u128_reduced(u128::from(splitmix64(&mut state))))
        .collect();
    let point = random_point(nv, 0xcafe_0000 ^ nv as u64);
    let opening = dense_opening(&evals, &point);
    let poly = DensePoly::<F>::from_field_evals(nv, &evals).expect("dense poly");
    prove_and_count::<fp128::Dense>(nv, point, opening, |stack| {
        stack.import_source(vec![poly]).expect("source")
    })
}

fn count_onehot(nv: usize) -> (Counts, Vec<(String, Counts)>) {
    let k = akita_config::unit_onehot_source_chunk_size::<fp128::OneHot>()
        .expect("unit one-hot config");
    let mut state = 0x0bee_0000_u64 ^ nv as u64;
    let indices: Vec<Option<u8>> = (0..(1usize << nv) / k)
        .map(|_| Some((splitmix64(&mut state) % k as u64) as u8))
        .collect();
    let point = random_point(nv, 0xcafe_0000 ^ nv as u64);
    let opening = indices
        .iter()
        .enumerate()
        .filter_map(|(chunk, hot)| hot.map(|idx| chunk * k + usize::from(idx)))
        .fold(F::zero(), |acc, index| acc + lagrange_weight(&point, index));
    let poly = OneHotPoly::<F, u8>::new(k, indices).expect("one-hot poly");
    prove_and_count::<fp128::OneHot>(nv, point, opening, |stack| {
        stack.import_source(vec![poly]).expect("source")
    })
}

fn prove_and_count<Cfg>(
    nv: usize,
    point: Vec<F>,
    opening: F,
    source: impl FnOnce(&CpuBackend<F, F>) -> SourceHandle<F, F>,
) -> (Counts, Vec<(String, Counts)>)
where
    Cfg: CommitmentConfig<Field = F, ExtField = F>,
{
    let label = b"opcount/one-verification";
    let scheme = load_workspace_scheme::<Cfg>(ChunkedWitnessCfg::default_non_chunked())
        .expect("workspace schedule catalog");
    let setup = scheme.setup_prover(nv, 1).expect("prover setup");
    let stack = CpuBackend::new(setup.expanded.clone()).expect("backend");
    let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");
    let CommitOutput {
        committed_group: commitment,
        private_handle: hint,
    } = stack
        .commit(
            scheme.schedules(),
            &source(&stack),
            GroupContext::scheduler_without_precommitted_groups(),
        )
        .expect("commit");
    let prover_data = prove_input::<Cfg>(&point, opening, &commitment, hint, scheme.schedules());
    let proof = scheme
        .batched_prove(&setup, prover_data, &stack, label, BasisMode::Lagrange)
        .expect("prove");

    let verifier = scheme.verifier(verifier_setup).expect("verifier");
    let openings = [opening];
    let statement = verify_input::<Cfg>(&point, &openings, &commitment, scheme.schedules());
    let count_on = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .stack_size(STACK_SIZE)
            .build()
            .expect("thread pool");
        pool.install(|| {
            opcount::enable();
            let result =
                verifier.batched_verify(&proof, label, statement.clone(), BasisMode::Lagrange);
            let counted = opcount::disable();
            result.expect("verification must accept the honest proof");
            counted
        })
    };
    let (total, sections) = count_on(1);
    // Parallel folds merge partial sums with a few extra additions, so the
    // single-threaded add/sub count is the canonical one. Everything else
    // must match exactly.
    let (parallel_total, _) = count_on(4);
    assert_eq!(
        Counts {
            fp128_add_sub: 0,
            ..total
        },
        Counts {
            fp128_add_sub: 0,
            ..parallel_total
        },
        "counts must not depend on the thread count"
    );
    println!(
        "4-thread rerun: identical except add/sub {} vs {} (parallel merge additions)",
        parallel_total.fp128_add_sub, total.fp128_add_sub
    );
    assert_eq!(
        total.packed_mul, 0,
        "SIMD-packed multiplies bypass the scalar counters"
    );
    (total, sections)
}

fn report(title: &str, total: &Counts, sections: &[(String, Counts)]) {
    println!("\n=== Akita verifier operation counts: {title} ===\n");
    println!("{total}\n");

    println!(
        "{:<44} {:>10} {:>10} {:>10} {:>8} {:>7}",
        "section", "mul", "mul wide", "add/sub", "blake2", "keccak"
    );
    for (label, c) in sections {
        let indent = if label.starts_with("setup scan") {
            "  └ "
        } else {
            ""
        };
        println!(
            "{:<44} {:>10} {:>10} {:>10} {:>8} {:>7}",
            format!("{indent}{label}"),
            c.fp128_mul,
            c.fp128_mul_wide,
            c.fp128_add_sub,
            c.blake2_compress,
            c.keccak_f1600
        );
    }

    // Only setup coefficient x alpha power inside the scan has a public
    // operand; the scan's other wide products multiply two variables.
    let in_scan = |pick: fn(&Counts) -> u64| -> u64 {
        sections
            .iter()
            .filter(|(label, _)| label.starts_with("setup scan"))
            .map(|(_, c)| pick(c))
            .sum()
    };
    let scan_wide = in_scan(|c| c.fp128_mul_wide);
    let scan_const = in_scan(|c| c.mul_base_unreduced);
    let top_level = sections
        .iter()
        .filter(|(label, _)| !label.starts_with("setup scan"))
        .fold(0u64, |sum, (_, c)| sum + c.fp128_mul_total());
    println!(
        "\nsetup scan wide products: {scan_wide} = {scan_const} setup coeff x alpha power + {} var x var",
        scan_wide - scan_const
    );
    println!(
        "fp128 multiplies outside every step section: {}",
        total.fp128_mul_total() - top_level
    );
    let rows = [
        ("fp128 mul, var x var, reduced", total.fp128_mul, AND_MUL),
        (
            "fp128 mul, var x var, wide",
            total.fp128_mul_wide - scan_const,
            AND_MUL_WIDE,
        ),
        ("fp128 mul, const x var (setup scan)", scan_const, AND_SCAN),
        ("fp128 x u64", total.fp128_mul_u64, AND_MUL),
        ("fp128 x limbs", total.fp128_mul_limbs, AND_MUL),
        ("fp128 x i32 scale", total.fp128_scale_i32, AND_MUL),
        ("fp128 add/sub", total.fp128_add_sub, AND_ADD_SUB),
        ("fp128 reduce", total.fp128_reduce, AND_REDUCE),
        ("blake2b compression", total.blake2_compress, AND_BLAKE2B),
        ("keccak-f1600", total.keccak_f1600, AND_KECCAK),
    ];
    println!(
        "\n{:<40} {:>12} {:>12} {:>16}",
        "operation", "count", "AND each*", "AND total*"
    );
    let mut and_total = 0u64;
    for (name, count, each) in rows {
        and_total += count * each;
        println!(
            "{:<40} {:>12} {:>12} {:>16}",
            name,
            count,
            each,
            count * each
        );
    }
    println!(
        "{:<40} {:>12} {:>12} {:>16}",
        "total (without terminal CRT/NTT)", "", "", and_total
    );
    println!(
        "\n* estimated AND gates per operation; garbled size ≈ AND total × 16 B ≈ {:.1} GB",
        and_total as f64 * 16.0 / 1e9
    );
}

#[allow(clippy::type_complexity)]
fn prove_input<'a, Cfg>(
    point: &'a [F],
    opening: F,
    commitment: &'a CommittedGroup<F>,
    hint: CommitmentHandle<F, F>,
    schedules: &TrustedScheduleCatalog<Cfg>,
) -> SelectedProverOpeningData<'a, F, CommitmentHandle<F, F>, F>
where
    Cfg: CommitmentConfig<Field = F, ExtField = F>,
{
    let group = PolynomialGroupClaims::new(point.to_vec(), vec![opening], commitment.clone())
        .expect("prover claims group");
    let claims = OpeningClaims::from_groups(vec![group]).expect("prover claims");
    SelectedProverOpeningData::from_committed_claims::<Cfg>(claims, vec![hint], schedules)
        .expect("prover opening data")
}

fn verify_input<'a, Cfg>(
    point: &'a [F],
    openings: &'a [F],
    commitment: &'a CommittedGroup<F>,
    schedules: &TrustedScheduleCatalog<Cfg>,
) -> GroupBatchStatement<'a, F, F>
where
    Cfg: CommitmentConfig<Field = F, ExtField = F>,
{
    let group = PolynomialGroupClaims::new(point.to_vec(), openings.to_vec(), commitment)
        .expect("verifier claims group");
    let claims = OpeningClaims::from_groups(vec![group]).expect("verifier claims");
    let profiles = CommittedGroupBatchProfile {
        final_group: *commitment.profile(),
        precommitteds: Vec::new(),
    };
    let selection = schedules
        .resolve_profiles(&profiles)
        .expect("verifier schedule")
        .selection();
    GroupBatchStatement::new(selection, claims).expect("verifier statement")
}

fn random_point(nv: usize, seed: u64) -> Vec<F> {
    let mut state = seed;
    (0..nv)
        .map(|_| {
            let lo = u128::from(splitmix64(&mut state));
            let hi = u128::from(splitmix64(&mut state));
            F::from_u128_reduced(hi << 64 | lo)
        })
        .collect()
}

/// `Σ_x eq(point, x) · evals[x]`, folding the lowest variable first.
fn dense_opening(evals: &[F], point: &[F]) -> F {
    let mut layer = evals.to_vec();
    for &coordinate in point {
        layer = layer
            .chunks_exact(2)
            .map(|pair| pair[0] + (pair[1] - pair[0]) * coordinate)
            .collect();
    }
    layer[0]
}

fn lagrange_weight(point: &[F], index: usize) -> F {
    point
        .iter()
        .enumerate()
        .fold(F::one(), |acc, (bit, &coordinate)| {
            if (index >> bit) & 1 == 1 {
                acc * coordinate
            } else {
                acc * (F::one() - coordinate)
            }
        })
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}
