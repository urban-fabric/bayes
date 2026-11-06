//! Rejection and adaptive rejection against the exact CDFs bayes already has.
//!
//! Kolmogorov–Smirnov rather than a moment check, because KS is sensitive to the
//! whole shape: a sampler with the right mean and variance but the wrong skew
//! passes a moment comparison and fails this. The moment check is kept as a
//! second, differently-sensitive summary.
//!
//! The last test here is the important negative one. Adaptive rejection is only
//! valid for a log-concave density, which cannot be verified in general — so a
//! deliberately bimodal target is fed in to confirm the breach flag fires rather
//! than the sampler quietly returning biased draws.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use bayes_distribution_ext::{Beta, Gamma, Normal};
use bayes_distribution_ext::{Continuous, ContinuousCdf, Moments};
use bayes_sampling::{AdaptiveRejection, RejectionSampler, RngCoreShim};
use bayes_utils::{Failures, ks_pvalue, ks_statistic, mean_tolerance};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const SEED: u64 = 20_260_105;
const N: usize = 40_000;

// ---------------------------------------------------------------------------
// Rejection sampling
// ---------------------------------------------------------------------------

/// A uniform envelope over a bounded target, judged by KS against the exact
/// distribution bayes computes for the same law.
#[test]
fn rejection_draws_a_beta_under_a_uniform_envelope() {
    let mut f = Failures::default();
    // Beta(2,4) is bounded with a peak density of ~2.07, so a uniform envelope
    // scaled by 2.2 dominates it comfortably.
    for (a, b, bound) in [(2.0, 4.0, 2.5), (3.0, 3.0, 2.0), (2.0, 2.0, 1.6)] {
        let target = Beta::new(a, b).unwrap();
        let ln_target = move |x: f64| {
            if (0.0..=1.0).contains(&x) {
                Continuous::ln_pdf(&target, x)
            } else {
                f64::NEG_INFINITY
            }
        };
        let mut s = RejectionSampler::new(
            ln_target,
            |_x: f64| 0.0, // uniform(0,1) has log-density 0
            |r: &mut dyn RngCoreShim| r.uniform(),
            bound,
        )
        .unwrap();

        let mut rng = ChaCha8Rng::seed_from_u64(SEED);
        let xs: Vec<f64> = (0..N)
            .map(|_| {
                s.draw(&mut rng, 10_000)
                    .expect("acceptance is likely")
                    .value
            })
            .collect();

        f.check(!s.envelope_was_breached(), || {
            format!(
                "Beta({a},{b}): envelope breached by {:.3e} in log units",
                s.worst_log_excess()
            )
        });
        let d = ks_statistic(&xs, |x| ContinuousCdf::cdf(&target, x));
        let p = ks_pvalue(d, N);
        f.check(p > 1e-4, || {
            format!("Beta({a},{b}): KS against its own cdf D = {d:.5}, p = {p:.3e}")
        });
        // The acceptance rate should be near 1/bound, which is the theory.
        let predicted = 1.0 / bound;
        f.check((s.acceptance_rate() - predicted).abs() < 0.05, || {
            format!(
                "Beta({a},{b}): acceptance {:.4} vs predicted 1/bound = {predicted:.4}",
                s.acceptance_rate()
            )
        });
    }
    f.finish("rejection vs Beta");
}

// ---------------------------------------------------------------------------
// Adaptive rejection
// ---------------------------------------------------------------------------

/// The headline check: adaptive rejection against four log-concave families
/// whose exact CDFs bayes already provides.
#[test]
fn adaptive_rejection_draws_log_concave_families_exactly() {
    let mut f = Failures::default();

    // --- Normal(mu, sigma): h = -(x-mu)^2 / (2 sigma^2), h' = -(x-mu)/sigma^2
    for (mu, sigma) in [(0.0, 1.0), (-2.0, 0.5), (3.0, 2.0)] {
        let exact = Normal::new(mu, sigma).unwrap();
        let h = move |x: f64| -0.5 * ((x - mu) / sigma).powi(2);
        let dh = move |x: f64| -(x - mu) / (sigma * sigma);
        let mut s = AdaptiveRejection::new(
            h,
            dh,
            f64::NEG_INFINITY,
            f64::INFINITY,
            &[mu - sigma, mu + sigma],
        )
        .unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(SEED);
        let xs: Vec<f64> = (0..N)
            .map(|_| s.draw(&mut rng, 1_000).expect("acceptance is likely"))
            .collect();

        f.check(!s.hull_was_breached(), || {
            format!("Normal({mu},{sigma}): the hull was breached, so h is not concave")
        });
        let d = ks_statistic(&xs, |x| ContinuousCdf::cdf(&exact, x));
        let p = ks_pvalue(d, N);
        f.check(p > 1e-4, || {
            format!("Normal({mu},{sigma}): KS D = {d:.5}, p = {p:.3e}")
        });
        // Adaptation must actually pay off: acceptance well above the 50% a
        // two-point hull would give, and far fewer density evaluations than draws.
        f.check(s.acceptance_rate() > 0.9, || {
            format!(
                "Normal({mu},{sigma}): acceptance only {:.3} — the hull is not adapting",
                s.acceptance_rate()
            )
        });
        f.check(s.density_evaluations() < N / 4, || {
            format!(
                "Normal({mu},{sigma}): {} density evaluations for {N} draws — the squeeze \
                 test is not saving work",
                s.density_evaluations()
            )
        });
    }

    // --- Gamma(shape >= 1, rate) on (0, inf): h = (k-1) ln x - r x
    for (shape, rate) in [(2.0, 1.0), (5.0, 2.0), (1.5, 0.5)] {
        let exact = Gamma::new(shape, rate).unwrap();
        let h = move |x: f64| (shape - 1.0) * x.ln() - rate * x;
        let dh = move |x: f64| (shape - 1.0) / x - rate;
        // The mode is (k-1)/r; seed either side of it.
        let mode = (shape - 1.0) / rate;
        let mut s = AdaptiveRejection::new(
            h,
            dh,
            0.0,
            f64::INFINITY,
            &[(mode * 0.5).max(1e-3), mode + 1.0 / rate],
        )
        .unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(SEED);
        let xs: Vec<f64> = (0..N)
            .map(|_| s.draw(&mut rng, 1_000).expect("acceptance is likely"))
            .collect();

        f.check(!s.hull_was_breached(), || {
            format!("Gamma({shape},{rate}): hull breached")
        });
        f.check(xs.iter().all(|&x| x > 0.0 && x.is_finite()), || {
            format!("Gamma({shape},{rate}): a draw left the support")
        });
        let d = ks_statistic(&xs, |x| ContinuousCdf::cdf(&exact, x));
        let p = ks_pvalue(d, N);
        f.check(p > 1e-4, || {
            format!("Gamma({shape},{rate}): KS D = {d:.5}, p = {p:.3e}")
        });
        f.check(s.acceptance_rate() > 0.85, || {
            format!(
                "Gamma({shape},{rate}): acceptance only {:.3}",
                s.acceptance_rate()
            )
        });
    }

    // --- Beta(a >= 1, b >= 1) on (0, 1), a bounded support on both sides
    for (a, b) in [(2.0, 3.0), (4.0, 2.0), (2.0, 2.0)] {
        let exact = Beta::new(a, b).unwrap();
        let h = move |x: f64| (a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln();
        let dh = move |x: f64| (a - 1.0) / x - (b - 1.0) / (1.0 - x);
        let mut s = AdaptiveRejection::new(h, dh, 0.0, 1.0, &[0.25, 0.75]).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(SEED);
        let xs: Vec<f64> = (0..N)
            .map(|_| s.draw(&mut rng, 1_000).expect("acceptance is likely"))
            .collect();

        f.check(!s.hull_was_breached(), || {
            format!("Beta({a},{b}): hull breached")
        });
        f.check(xs.iter().all(|&x| (0.0..=1.0).contains(&x)), || {
            format!("Beta({a},{b}): a draw left [0,1]")
        });
        let d = ks_statistic(&xs, |x| ContinuousCdf::cdf(&exact, x));
        let p = ks_pvalue(d, N);
        f.check(p > 1e-4, || {
            format!("Beta({a},{b}): KS D = {d:.5}, p = {p:.3e}")
        });
    }

    f.finish("adaptive rejection exactness");
}

/// Sample moments against the analytic ones, at central-limit tolerances — a
/// second summary of the same agreement, sensitive to different errors than KS.
#[test]
fn adaptive_rejection_reproduces_the_analytic_moments() {
    let mut f = Failures::default();
    for (shape, rate) in [(2.0, 1.0), (5.0, 2.0)] {
        let exact = Gamma::new(shape, rate).unwrap();
        let h = move |x: f64| (shape - 1.0) * x.ln() - rate * x;
        let dh = move |x: f64| (shape - 1.0) / x - rate;
        let mode = (shape - 1.0) / rate;
        let mut s = AdaptiveRejection::new(
            h,
            dh,
            0.0,
            f64::INFINITY,
            &[(mode * 0.5).max(1e-3), mode + 1.0],
        )
        .unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(SEED);
        let xs: Vec<f64> = (0..N).map(|_| s.draw(&mut rng, 1_000).unwrap()).collect();

        let got = xs.iter().sum::<f64>() / N as f64;
        let want = Moments::mean(&exact);
        let tol = mean_tolerance(Moments::variance(&exact), N, 6.0);
        f.check((got - want).abs() <= tol, || {
            format!("Gamma({shape},{rate}): sample mean {got:.6} vs analytic {want:.6}")
        });

        let var = xs.iter().map(|x| (x - got).powi(2)).sum::<f64>() / (N as f64 - 1.0);
        f.check(
            (var - Moments::variance(&exact)).abs() <= 0.1 * Moments::variance(&exact),
            || {
                format!(
                    "Gamma({shape},{rate}): sample variance {var:.6} vs analytic {:.6}",
                    Moments::variance(&exact)
                )
            },
        );
    }
    f.finish("adaptive rejection moments");
}

/// A density that is **not** log-concave must be reported, not silently
/// mis-sampled. This is the failure mode the flag exists for.
#[test]
fn a_non_log_concave_density_is_detected() {
    // A two-component mixture far enough apart to be bimodal: ln f is convex
    // between the modes, so the "upper" hull cuts below the density there.
    let h = |x: f64| {
        let a = (-0.5 * (x - 3.0_f64).powi(2)).exp();
        let b = (-0.5 * (x + 3.0_f64).powi(2)).exp();
        (a + b).ln()
    };
    let dh = |x: f64| {
        let a = (-0.5 * (x - 3.0_f64).powi(2)).exp();
        let b = (-0.5 * (x + 3.0_f64).powi(2)).exp();
        (-(x - 3.0) * a - (x + 3.0) * b) / (a + b)
    };
    let mut s =
        AdaptiveRejection::new(h, dh, f64::NEG_INFINITY, f64::INFINITY, &[-4.0, 4.0]).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    for _ in 0..4_000 {
        let _ = s.draw(&mut rng, 100);
    }
    assert!(
        s.hull_was_breached(),
        "a bimodal density went undetected: acceptance {:.3}, hull size {}",
        s.acceptance_rate(),
        s.hull_size()
    );
}

#[test]
fn adaptive_rejection_rejects_unusable_configurations() {
    let h = |x: f64| -0.5 * x * x;
    let dh = |x: f64| -x;
    // Fewer than two seeds.
    assert!(AdaptiveRejection::new(h, dh, -10.0, 10.0, &[0.0]).is_err());
    // An empty interval.
    assert!(AdaptiveRejection::new(h, dh, 1.0, 1.0, &[0.5, 0.6]).is_err());
    // A seed outside the interval.
    assert!(AdaptiveRejection::new(h, dh, 0.0, 1.0, &[0.5, 2.0]).is_err());
    // On an unbounded right tail, every seed slopes the wrong way: the envelope
    // would have infinite mass, so there is no sampler to build.
    let uphill = |x: f64| x;
    let d_uphill = |_x: f64| 1.0;
    assert!(
        AdaptiveRejection::new(uphill, d_uphill, 0.0, f64::INFINITY, &[1.0, 2.0]).is_err(),
        "an unbounded tail with positive slope must be refused"
    );
}
