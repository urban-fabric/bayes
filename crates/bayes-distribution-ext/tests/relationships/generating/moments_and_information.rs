//! L-moments and Fisher information, against independent routes to the same
//! quantity.
//!
//! The L-moment check is the more interesting of the two: `l_scale` and the
//! sample `gini_mean_difference` reach `½·E|X₁ − X₂|` from opposite ends — one
//! from the quantile function analytically, the other from the pairwise absolute
//! differences of a seeded draw. Nothing is shared but the answer.
//!
//! Fisher information is checked against `E[(∂ ln f/∂θ)²]` with the score taken
//! numerically, which exercises the density's parameter dependence rather than
//! its argument dependence — a direction nothing else in the suite probes.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use crate::{Failures, derivative, simpson};
use bayes_distribution_ext::{ContinuousCdf, Moments, *};

/// `λ₁ = E[X]`, and `λ₂ = ½·E|X₁ − X₂|` — the latter reached independently
/// through the sample Gini mean difference of a large seeded draw.
#[test]
fn l_moments_agree_with_the_moments_and_the_gini_mean_difference() {
    use rand::SeedableRng;
    use rand::distr::Distribution as _;
    use rand_chacha::ChaCha8Rng;

    let mut f = Failures::default();

    macro_rules! check {
        ($label:literal, $dist:expr, $tol:expr) => {{
            let d = $dist;
            // The first L-moment is the mean, exactly.
            f.close(
                || format!("{}: l_location vs mean", $label),
                d.l_location(),
                Moments::mean(&d),
                1e-9,
            );

            // The second L-moment is half the expected absolute difference of two
            // independent draws. Estimate that from a seeded sample: with n = 4000
            // the standard error of the Gini mean difference is well under 1%.
            let mut rng = ChaCha8Rng::seed_from_u64(20_260_105);
            let xs: Vec<f64> = (0..4000).map(|_| d.sample(&mut rng)).collect();
            let mut acc = 0.0;
            let mut pairs = 0_u64;
            for i in 0..xs.len() {
                for j in (i + 1)..xs.len() {
                    acc += (xs[i] - xs[j]).abs();
                    pairs += 1;
                }
            }
            let gini = acc / pairs as f64;
            f.close(
                || format!("{}: l_scale vs ½·E|X1-X2| from a sample", $label),
                d.l_scale(),
                0.5 * gini,
                $tol,
            );

            // L-skewness and L-kurtosis are ratios, so they live in bounded ranges.
            let t3 = d.l_skewness();
            let t4 = d.l_kurtosis();
            f.check(t3.is_nan() || t3.abs() <= 1.0, || {
                format!("{}: l_skewness = {t3} outside [-1,1]", $label)
            });
            f.check(t4.is_nan() || (-0.25..=1.0).contains(&t4), || {
                format!("{}: l_kurtosis = {t4} outside [-0.25,1]", $label)
            });
        }};
    }

    check!("Normal(0,1)", Normal::new(0.0, 1.0).unwrap(), 0.03);
    check!("Exponential(1)", Exponential::new(1.0).unwrap(), 0.03);
    check!("Uniform(0,1)", Uniform::new(0.0, 1.0).unwrap(), 0.03);
    check!("Logistic(0,1)", Logistic::new(0.0, 1.0).unwrap(), 0.04);

    f.finish("L-moments");
}

/// Fisher information computed in closed form against `E[(∂ ln f / ∂θ)²]`,
/// evaluated by quadrature on a numerical score.
#[test]
fn fisher_information_matches_the_expected_squared_score() {
    let mut f = Failures::default();

    // Bernoulli(p): I(p) = 1/(p(1-p)).
    for p in [0.1, 0.25, 0.5, 0.8] {
        let d = Bernoulli::new(p).unwrap();
        f.close(
            || format!("Bernoulli({p}) fisher_information = 1/(p(1-p))"),
            d.fisher_information(),
            1.0 / (p * (1.0 - p)),
            1e-10,
        );
    }

    // Poisson(λ): I(λ) = 1/λ.
    for lambda in [0.5, 4.0, 100.0] {
        let d = Poisson::new(lambda).unwrap();
        f.close(
            || format!("Poisson({lambda}) fisher_information = 1/λ"),
            d.fisher_information(),
            1.0 / lambda,
            1e-10,
        );
    }

    // Exponential(rate): I(rate) = 1/rate².
    for rate in [0.5, 2.0, 10.0] {
        let d = Exponential::new(rate).unwrap();
        f.close(
            || format!("Exponential({rate}) fisher_information = 1/rate²"),
            d.fisher_information(),
            1.0 / (rate * rate),
            1e-10,
        );
    }

    // Rayleigh(σ): I(σ) = 4/σ². Checked against the expected squared score,
    // ∫ (∂_σ ln f)² f dx, with the score taken numerically.
    for sigma in [0.5, 1.0, 3.0] {
        let d = Rayleigh::new(sigma).unwrap();
        let score_sq = simpson(
            |x| {
                if x <= 0.0 {
                    return 0.0;
                }
                let h = sigma * 1e-5;
                let score = derivative(
                    |s| {
                        Rayleigh::new(s)
                            .map(|r| Continuous::ln_pdf(&r, x))
                            .unwrap_or(f64::NAN)
                    },
                    sigma,
                    h,
                );
                score * score * Continuous::pdf(&d, x)
            },
            1e-9,
            d.inverse_cdf(1.0 - 1e-12),
            100_000,
        );
        f.close(
            || format!("Rayleigh({sigma}) fisher_information vs E[score²]"),
            d.fisher_information(),
            score_sq,
            1e-4,
        );
    }

    f.finish("Fisher information");
}
