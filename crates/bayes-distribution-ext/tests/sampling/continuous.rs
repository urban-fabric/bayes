//! Continuous samplers against their own CDFs.
//!
//! The load-bearing check is the Kolmogorov–Smirnov test: `sup|F_n − F|` over a
//! large seeded draw, judged by the asymptotic Kolmogorov series. It is the
//! right instrument here because it is sensitive to the *whole shape* rather
//! than to a summary — a sampler with the right mean and variance but the wrong
//! skew passes a moment check and fails this.

use bayes_distribution_ext::{Continuous, VonMises};
use bayes_utils::families::{
    ContCase, continuous_cases, quantile_unrepresentable_at, quantiles_are_unrepresentable,
};
use bayes_utils::{
    Failures, chi_squared_gof_pvalue, ks_pvalue, ks_statistic, mean_tolerance, variance_tolerance,
};
use rand::SeedableRng;
use rand::distr::Distribution;
use rand_chacha::ChaCha8Rng;

use crate::SEED;

/// How many draws per family. Large enough for KS to notice a 1% shape error,
/// small enough that 60 families still run in a couple of seconds.
const N: usize = 20_000;

fn draw(case: &ContCase, n: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n).map(|_| case.dist.sample(&mut rng)).collect()
}

/// Is a meaningful fraction of this family's probability unreachable in `f64`?
///
/// `Gamma(1e-3, 1)` has `cdf(5e-324) = 0.475`: nearly half its mass lies below
/// the smallest positive double, so a *correct* sampler must pile 47.5% of its
/// draws onto zero and the empirical CDF must differ from the analytic one by
/// that much. The observed `D = 0.470` is exactly that figure, not a defect.
/// KS has nothing to say about such a family.
fn mass_is_partly_unreachable(d: &dyn bayes_utils::ContView) -> bool {
    quantile_unrepresentable_at(d, 1e-4) || quantile_unrepresentable_at(d, 1.0 - 1e-4)
}

/// Draws must follow the family's own CDF.
#[test]
fn draws_follow_the_family_cdf() {
    let mut f = Failures::default();
    for case in continuous_cases() {
        // A family whose quantiles are not representable cannot be sampled
        // meaningfully either: every draw lands on the same double.
        if quantiles_are_unrepresentable(&*case.dist) || mass_is_partly_unreachable(&*case.dist) {
            continue;
        }
        let xs = draw(&case, N, SEED);
        let d = ks_statistic(&xs, |x| case.dist.cdf(x));
        let p = ks_pvalue(d, N);
        // 1e-4 rather than 0.05: with ~60 families a 5% threshold would fire on
        // about three correct samplers every run. At 1e-4 a false failure is a
        // once-in-150-runs event across the whole sweep, and a sampler that is
        // actually drawing from the wrong law fails by orders of magnitude.
        f.check(p > 1e-4, || {
            format!(
                "{}: KS against its own cdf D = {d:.5}, p = {p:.3e} over {N} draws",
                case.name
            )
        });
    }
    f.finish("continuous KS");
}

/// Draws must lie inside the declared support, and be finite.
#[test]
fn draws_lie_within_the_declared_support() {
    let mut f = Failures::default();
    for case in continuous_cases() {
        let xs = draw(&case, 5_000, SEED);
        let (lo, hi) = (case.dist.lo(), case.dist.hi());
        let mut outside = 0;
        let mut nonfinite = 0;
        for &x in &xs {
            if !x.is_finite() {
                nonfinite += 1;
            } else if x < lo || x > hi {
                outside += 1;
            }
        }
        f.check(nonfinite == 0, || {
            format!("{}: {nonfinite} of 5000 draws were not finite", case.name)
        });
        f.check(outside == 0, || {
            format!(
                "{}: {outside} of 5000 draws fell outside the support [{lo:.6e}, {hi:.6e}]",
                case.name
            )
        });
    }
    f.finish("support containment");
}

/// Sample moments against the analytic ones, at central-limit tolerances.
#[test]
fn sample_moments_match_the_analytic_moments() {
    let mut f = Failures::default();
    for case in continuous_cases() {
        let (mu, var) = (case.dist.mean(), case.dist.variance());
        // A family whose mean or variance does not exist has nothing to compare;
        // `Cauchy` and `StudentT(1)` report `NaN` by contract.
        if !mu.is_finite() || !var.is_finite() || quantiles_are_unrepresentable(&*case.dist) {
            continue;
        }
        let xs = draw(&case, N, SEED);
        let sample_mean = xs.iter().sum::<f64>() / N as f64;
        let tol = mean_tolerance(var, N, 6.0);
        f.check((sample_mean - mu).abs() <= tol, || {
            format!(
                "{}: sample mean {sample_mean:.9e} vs analytic {mu:.9e} \
                 (off by {:.3e}, 6-sigma bound {tol:.3e})",
                case.name,
                (sample_mean - mu).abs()
            )
        });

        // The variance bound needs the fourth moment, so it only applies where
        // the kurtosis exists and is not so large that the bound is vacuous.
        let kurt = case.dist.kurtosis();
        // A bound wider than the variance itself is vacuous, so skip it rather
        // than assert something that cannot fail.
        if let Some(vtol) = variance_tolerance(var, kurt, N, 6.0).filter(|&v| v < var) {
            {
                let sample_var =
                    xs.iter().map(|x| (x - sample_mean).powi(2)).sum::<f64>() / (N as f64 - 1.0);
                f.check((sample_var - var).abs() <= vtol, || {
                    format!(
                        "{}: sample variance {sample_var:.9e} vs analytic {var:.9e} \
                         (off by {:.3e}, 6-sigma bound {vtol:.3e})",
                        case.name,
                        (sample_var - var).abs()
                    )
                });
            }
        }
    }
    f.finish("continuous sample moments");
}

/// The median splits the draws in half — a distribution-free check that needs no
/// moments at all, so it covers the heavy-tailed families the moment check skips.
#[test]
fn the_analytic_median_splits_the_draws_in_half() {
    let mut f = Failures::default();
    for case in continuous_cases() {
        let m = case.dist.median();
        if !m.is_finite() || quantiles_are_unrepresentable(&*case.dist) {
            continue;
        }
        let xs = draw(&case, N, SEED);
        let below = xs.iter().filter(|&&x| x <= m).count() as f64 / N as f64;
        // The count below the median is Binomial(N, ½), so its standard error is
        // ½/√N. Six of those is the bound.
        let tol = 6.0 * 0.5 / (N as f64).sqrt();
        f.check((below - 0.5).abs() <= tol, || {
            format!(
                "{}: {:.4} of draws fell at or below the analytic median {m:.6e} \
                 (6-sigma bound {tol:.4})",
                case.name, below
            )
        });
    }
    f.finish("median split");
}

/// Quantiles must partition the draws into equal buckets. This is the KS check's
/// complement: it bins by *probability* rather than by value, so it is sensitive
/// to a sampler that gets the body right and the tails wrong.
#[test]
fn quantile_buckets_receive_equal_shares_of_the_draws() {
    let mut f = Failures::default();
    const BUCKETS: usize = 20;
    for case in continuous_cases() {
        if quantiles_are_unrepresentable(&*case.dist) || mass_is_partly_unreachable(&*case.dist) {
            continue;
        }
        // Cut points at every 1/20th of probability.
        let edges: Vec<f64> = (1..BUCKETS)
            .map(|i| case.dist.inverse_cdf(i as f64 / BUCKETS as f64))
            .collect();
        if edges.iter().any(|e| !e.is_finite()) || edges.windows(2).any(|w| w[0] >= w[1]) {
            continue; // unrepresentable or degenerate cut points
        }
        let xs = draw(&case, N, SEED);
        let mut counts = vec![0.0; BUCKETS];
        for &x in &xs {
            let idx = edges.partition_point(|&e| e < x);
            counts[idx.min(BUCKETS - 1)] += 1.0;
        }
        let expected = vec![N as f64 / BUCKETS as f64; BUCKETS];
        let p = chi_squared_gof_pvalue(&counts, &expected, 5.0);
        f.check(!p.is_finite() || p > 1e-4, || {
            format!(
                "{}: draws are not uniform across its own quantile buckets (p = {p:.3e})",
                case.name
            )
        });
    }
    f.finish("quantile buckets");
}

/// A seeded sampler must be reproducible, and two different seeds must not agree.
#[test]
fn sampling_is_deterministic_in_the_seed() {
    let mut f = Failures::default();
    for case in continuous_cases() {
        let a = draw(&case, 64, SEED);
        let b = draw(&case, 64, SEED);
        f.check(a == b, || {
            format!("{}: the same seed produced different draws", case.name)
        });
        let c = draw(&case, 64, SEED + 1);
        // A degenerate family can legitimately return the same values from any
        // seed; only complain when there is spread to distinguish.
        let spread = case.dist.inverse_cdf(0.99) - case.dist.inverse_cdf(0.01);
        if spread.is_finite() && spread > 0.0 && !quantiles_are_unrepresentable(&*case.dist) {
            f.check(a != c, || {
                format!(
                    "{}: two different seeds produced identical draws",
                    case.name
                )
            });
        }
    }
    f.finish("seed determinism");
}

/// `VonMises` has no CDF to KS against, so its sampler is checked against the
/// circular moments and by binning the density numerically.
#[test]
fn von_mises_draws_follow_its_density() {
    use std::f64::consts::PI;
    let mut f = Failures::default();
    for kappa in [0.5, 2.0, 8.0] {
        let d = VonMises::new(0.0, kappa).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(SEED);
        let xs: Vec<f64> = (0..N).map(|_| d.sample(&mut rng)).collect();

        f.check(xs.iter().all(|x| x.is_finite()), || {
            format!("VonMises(0,{kappa}): a draw was not finite")
        });
        // Draws must land on the circle.
        f.check(
            xs.iter().all(|&x| (-PI - 1e-9..=PI + 1e-9).contains(&x)),
            || format!("VonMises(0,{kappa}): a draw left [-pi, pi]"),
        );

        // Bin the circle and compare against the integrated density per bin.
        const BINS: usize = 24;
        let width = 2.0 * PI / BINS as f64;
        let mut counts = vec![0.0; BINS];
        for &x in &xs {
            let idx = (((x + PI) / width) as usize).min(BINS - 1);
            counts[idx] += 1.0;
        }
        // Each bin's expected count is the *integral* of the density over it, not
        // the midpoint times the width. At kappa = 8 the density's curvature makes
        // the midpoint rule err by 2.3% per bin — enough to make a correct sampler
        // look biased, which it did before this was fixed.
        let expected: Vec<f64> = (0..BINS)
            .map(|i| {
                let a = -PI + width * i as f64;
                bayes_utils::simpson(|t| Continuous::pdf(&d, t), a, a + width, 64) * N as f64
            })
            .collect();
        let p = chi_squared_gof_pvalue(&counts, &expected, 5.0);
        f.check(p > 1e-4, || {
            format!("VonMises(0,{kappa}): binned draws disagree with the density (p = {p:.3e})")
        });

        // The circular mean direction is the location, so the mean resultant
        // vector should point at angle 0.
        let (sx, sy): (f64, f64) = xs
            .iter()
            .fold((0.0, 0.0), |(cx, cy), &t| (cx + t.cos(), cy + t.sin()));
        let angle = sy.atan2(sx);
        f.check(angle.abs() < 0.05, || {
            format!("VonMises(0,{kappa}): mean direction {angle:.4} is not 0")
        });
    }
    f.finish("von Mises sampling");
}
