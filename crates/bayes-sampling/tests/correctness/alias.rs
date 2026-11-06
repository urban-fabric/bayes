//! The alias table against the weights it was built from.
//!
//! Pearson's chi-squared is the right instrument: the outcomes *are* the bins, so
//! the expected count in cell `k` is `n·pmf(k)` straight from the table. The
//! parameter sets lean on uneven weights, because that is where the worklist
//! construction could mis-pair a column and still look plausible.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use bayes_distribution_ext::Categorical;
use bayes_distribution_ext::{Discrete, Moments};
use bayes_sampling::AliasTable;
use bayes_utils::{Failures, chi_squared_gof_pvalue, mean_tolerance};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const SEED: u64 = 20_260_105;
const N: usize = 40_000;

// ---------------------------------------------------------------------------
// The alias table
// ---------------------------------------------------------------------------

/// Draws must follow the weights, including when the weights are wildly uneven —
/// which is exactly where the worklist construction could mis-pair a column.
#[test]
fn alias_draws_follow_their_weights() {
    let mut f = Failures::default();
    let cases: Vec<(&str, Vec<f64>)> = vec![
        ("uniform-3", vec![1.0, 1.0, 1.0]),
        ("skewed", vec![0.2, 0.5, 0.3]),
        ("unnormalized", vec![2.0, 5.0, 3.0]),
        ("one-outcome", vec![1.0]),
        ("two-way", vec![0.01, 0.99]),
        ("with-zeros", vec![1.0, 0.0, 1.0, 0.0, 2.0]),
        ("steep", (1..=20).map(|i| 1.0 / i as f64).collect()),
        (
            "long-tail",
            (1..=200).map(|i| (i as f64).powi(-2)).collect(),
        ),
        ("near-equal", vec![1.0 + 1e-12, 1.0, 1.0 - 1e-12]),
    ];
    for (name, weights) in cases {
        let t = AliasTable::new(&weights).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(SEED);
        let mut counts = vec![0.0; t.len()];
        for _ in 0..N {
            counts[t.sample(&mut rng)] += 1.0;
        }
        let expected: Vec<f64> = (0..t.len()).map(|k| t.pmf(k) * N as f64).collect();
        let p = chi_squared_gof_pvalue(&counts, &expected, 5.0);
        f.check(!p.is_finite() || p > 1e-4, || {
            format!("{name}: alias draws disagree with the weights (p = {p:.3e})")
        });
        // Nothing with zero weight may ever come back.
        for (k, &c) in counts.iter().enumerate() {
            f.check(t.pmf(k) > 0.0 || c == 0.0, || {
                format!("{name}: outcome {k} has zero weight but was drawn {c} times")
            });
        }
    }
    f.finish("alias goodness of fit");
}

/// The alias table must agree with the `Categorical` it can replace — same
/// probabilities, and draws that pass a goodness-of-fit test against its PMF.
#[test]
fn alias_matches_the_categorical_it_replaces() {
    let mut f = Failures::default();
    for weights in [
        vec![0.2, 0.5, 0.3],
        vec![2.0, 5.0, 3.0],
        (1..=64).map(|i| i as f64).collect::<Vec<_>>(),
    ] {
        let cat = Categorical::new(&weights).unwrap();
        let t = AliasTable::new(&weights).unwrap();
        for k in 0..weights.len() {
            f.close(
                || format!("alias pmf({k}) vs Categorical"),
                t.pmf(k),
                Discrete::pmf(&cat, k as u64),
                1e-14,
            );
        }
        // And the draws follow the Categorical's own mass function.
        let mut rng = ChaCha8Rng::seed_from_u64(SEED);
        let mut counts = vec![0.0; weights.len()];
        for _ in 0..N {
            counts[t.sample(&mut rng)] += 1.0;
        }
        let expected: Vec<f64> = (0..weights.len())
            .map(|k| Discrete::pmf(&cat, k as u64) * N as f64)
            .collect();
        let p = chi_squared_gof_pvalue(&counts, &expected, 5.0);
        f.check(!p.is_finite() || p > 1e-4, || {
            format!("alias draws disagree with Categorical's pmf (p = {p:.3e})")
        });
        // The mean is a second, independent summary of the same agreement.
        let sample_mean = counts
            .iter()
            .enumerate()
            .map(|(k, &c)| k as f64 * c)
            .sum::<f64>()
            / N as f64;
        let tol = mean_tolerance(Moments::variance(&cat), N, 6.0);
        f.check((sample_mean - Moments::mean(&cat)).abs() <= tol, || {
            format!(
                "alias sample mean {sample_mean} vs Categorical mean {}",
                Moments::mean(&cat)
            )
        });
    }
    f.finish("alias vs Categorical");
}

#[test]
fn alias_is_deterministic_in_the_seed() {
    let t = AliasTable::new(&[0.3, 0.5, 0.2]).unwrap();
    let draw = |seed: u64| {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        t.sample_n(&mut rng, 256)
    };
    assert_eq!(draw(SEED), draw(SEED), "same seed, different draws");
    assert_ne!(
        draw(SEED),
        draw(SEED + 1),
        "different seeds, identical draws"
    );
}
