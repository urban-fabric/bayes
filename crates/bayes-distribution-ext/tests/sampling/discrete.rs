//! Discrete samplers against their own PMFs.
//!
//! The instrument here is Pearson's chi-squared rather than KS, because the
//! support is already a natural set of bins: the expected count in cell `k` is
//! `n·pmf(k)`, straight from the family. Sparse tail cells are pooled before the
//! statistic is formed — an unpooled cell with an expected count below about 5
//! is the classic way to get a significant result out of a correct sampler.

use bayes_distribution_ext::{Bernoulli, DiscreteUniform, IntegerDiscrete, Moments, Support};
use bayes_utils::families::{DiscSampleCase, discrete_sampler_cases, summation_limit};
use bayes_utils::{Failures, chi_squared_gof_pvalue, mean_tolerance};
use rand::SeedableRng;
use rand::distr::Distribution;
use rand_chacha::ChaCha8Rng;

use crate::SEED;

const N: usize = 40_000;

fn draw(case: &DiscSampleCase, n: usize, seed: u64) -> Vec<u64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n).map(|_| case.dist.sample(&mut rng)).collect()
}

/// Draws must follow the family's own PMF.
#[test]
fn draws_follow_the_family_pmf() {
    let mut f = Failures::default();
    for case in discrete_sampler_cases() {
        let top = summation_limit(&*case.dist).min(case.dist.lo() + 400);
        let ks = draw(&case, N, SEED);

        // Observed counts over [lo, top], with everything above pooled last.
        let span = (top - case.dist.lo() + 2) as usize;
        let mut observed = vec![0.0; span];
        for &k in &ks {
            let idx = if k > top {
                span - 1
            } else {
                (k - case.dist.lo()) as usize
            };
            observed[idx] += 1.0;
        }
        let mut expected: Vec<f64> = (0..span - 1)
            .map(|i| case.dist.pmf(case.dist.lo() + i as u64) * N as f64)
            .collect();
        // The final cell pools the whole tail above `top`.
        expected.push(case.dist.sf(top) * N as f64);

        let p = chi_squared_gof_pvalue(&observed, &expected, 5.0);
        f.check(!p.is_finite() || p > 1e-4, || {
            format!(
                "{}: draws disagree with its own pmf (chi-squared p = {p:.3e} over {N} draws)",
                case.name
            )
        });
    }
    f.finish("discrete chi-squared");
}

/// Draws must lie inside the declared support.
#[test]
fn draws_lie_within_the_declared_support() {
    let mut f = Failures::default();
    for case in discrete_sampler_cases() {
        let ks = draw(&case, 10_000, SEED);
        let (lo, hi) = (case.dist.lo(), case.dist.hi());
        let bad = ks.iter().filter(|&&k| k < lo || k > hi).count();
        f.check(bad == 0, || {
            format!(
                "{}: {bad} of 10000 draws fell outside the support [{lo}, {hi}]",
                case.name
            )
        });
        // And every value drawn must carry nonzero mass — a sampler that can
        // return a `k` with `pmf(k) == 0` is inconsistent with its own density.
        let impossible = ks.iter().filter(|&&k| case.dist.pmf(k) == 0.0).count();
        f.check(impossible == 0, || {
            format!(
                "{}: {impossible} draws landed on a value with zero mass",
                case.name
            )
        });
    }
    f.finish("discrete support containment");
}

/// Sample moments against the analytic ones at central-limit tolerances.
#[test]
fn sample_moments_match_the_analytic_moments() {
    let mut f = Failures::default();
    for case in discrete_sampler_cases() {
        let (mu, var) = (case.dist.mean(), case.dist.variance());
        if !mu.is_finite() || !var.is_finite() {
            continue;
        }
        let ks = draw(&case, N, SEED);
        let sample_mean = ks.iter().map(|&k| k as f64).sum::<f64>() / N as f64;
        let tol = mean_tolerance(var, N, 6.0);
        f.check((sample_mean - mu).abs() <= tol, || {
            format!(
                "{}: sample mean {sample_mean:.9} vs analytic {mu:.9} \
                 (off by {:.3e}, 6-sigma bound {tol:.3e})",
                case.name,
                (sample_mean - mu).abs()
            )
        });
        let sample_var = ks
            .iter()
            .map(|&k| (k as f64 - sample_mean).powi(2))
            .sum::<f64>()
            / (N as f64 - 1.0);
        // Without a kurtosis to hand, use the loose but valid bound that a
        // sample variance of a bounded-fourth-moment variable concentrates at
        // rate 1/sqrt(n); 15% at n = 40000 is generous and still catches a
        // sampler drawing from the wrong spread.
        f.check((sample_var - var).abs() <= 0.15 * var.max(1e-12), || {
            format!(
                "{}: sample variance {sample_var:.9} vs analytic {var:.9}",
                case.name
            )
        });
    }
    f.finish("discrete sample moments");
}

#[test]
fn sampling_is_deterministic_in_the_seed() {
    let mut f = Failures::default();
    for case in discrete_sampler_cases() {
        let a = draw(&case, 128, SEED);
        let b = draw(&case, 128, SEED);
        f.check(a == b, || {
            format!("{}: the same seed produced different draws", case.name)
        });
    }
    f.finish("discrete seed determinism");
}

/// `Bernoulli` draws into `bool` and `DiscreteUniform` into `i64`, so neither
/// joins the sweep above. Both are simple enough to check exactly.
#[test]
fn bernoulli_and_discrete_uniform_samplers_are_correct() {
    let mut f = Failures::default();

    for p in [0.0, 0.25, 0.5, 0.9, 1.0] {
        let d = Bernoulli::new(p).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(SEED);
        let n = 40_000;
        let hits = (0..n).filter(|_| d.sample(&mut rng)).count() as f64 / n as f64;
        // The hit rate is Binomial(n, p)/n, so its standard error is √(p(1−p)/n).
        let tol = mean_tolerance(p * (1.0 - p), n, 6.0).max(0.0);
        f.check((hits - p).abs() <= tol + 1e-12, || {
            format!("Bernoulli({p}): drew {hits:.5} successes, expected {p} (bound {tol:.5})")
        });
    }

    // The half-open range `[lo, hi)` is the convention this family documents, so
    // `hi` itself must never be drawn — which is exactly the off-by-one a
    // sampler is most likely to get wrong.
    for (lo, hi) in [(0_i64, 6_i64), (-3, 3), (10, 11)] {
        let d = DiscreteUniform::new(lo, hi).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(SEED);
        let n = 40_000;
        let ks: Vec<i64> = (0..n).map(|_| d.sample(&mut rng)).collect();
        let span = (hi - lo) as usize;
        f.check(ks.iter().all(|&k| k >= lo && k < hi), || {
            format!("DiscreteUniform({lo},{hi}): a draw left the half-open range")
        });
        f.check(!ks.contains(&hi), || {
            format!("DiscreteUniform({lo},{hi}): drew the excluded upper bound {hi}")
        });
        let mut counts = vec![0.0; span];
        for &k in &ks {
            counts[(k - lo) as usize] += 1.0;
        }
        let expected = vec![n as f64 / span as f64; span];
        let p = chi_squared_gof_pvalue(&counts, &expected, 5.0);
        f.check(!p.is_finite() || p > 1e-4, || {
            format!("DiscreteUniform({lo},{hi}): draws are not uniform (p = {p:.3e})")
        });
        let sample_mean = ks.iter().map(|&k| k as f64).sum::<f64>() / n as f64;
        let mu = Moments::mean(&d);
        let tol = mean_tolerance(Moments::variance(&d), n, 6.0);
        f.check((sample_mean - mu).abs() <= tol, || {
            format!("DiscreteUniform({lo},{hi}): sample mean {sample_mean} vs analytic {mu}")
        });
        // The support's declared top is `hi - 1`, and the mass there is nonzero.
        let top = <DiscreteUniform as Support<i64>>::max(&d);
        f.check(top == hi - 1, || {
            format!(
                "DiscreteUniform({lo},{hi}): Support::max() is {top}, expected {}",
                hi - 1
            )
        });
        f.check(IntegerDiscrete::pmf(&d, top) > 0.0, || {
            format!("DiscreteUniform({lo},{hi}): the support top carries no mass")
        });
    }

    f.finish("Bernoulli / DiscreteUniform sampling");
}
