//! Each closed-form posterior against **Bayes' rule computed by quadrature**.
//!
//! A conjugate update is a shortcut: it claims that adding sufficient statistics
//! to natural hyperparameters produces the same distribution that
//!
//! ```text
//!     p(θ | x) = prior(θ)·Π likelihood(xᵢ | θ) / ∫ prior·Π likelihood dθ
//! ```
//!
//! produces. Testing that claim does not require trusting the shortcut: the
//! right-hand side can be evaluated using only the prior's density and the
//! likelihood's, with the evidence integral done numerically.
//!
//! This is the deepest edge in the network, because it ties the conjugate layer
//! to the density layer to the quadrature — three things that could each be
//! wrong separately but cannot agree while wrong.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use crate::{Failures, simpson};
use bayes_distribution_ext::{Continuous, Moments, *};

/// Normalize `f` over `[a, b]` and compare it to `g` pointwise — the shape of
/// every check in this file.
fn posterior_matches(
    f: &mut Failures,
    label: &str,
    unnormalized: impl Fn(f64) -> f64,
    closed_form: impl Fn(f64) -> f64,
    window: (f64, f64),
    grid: &[f64],
    tol: f64,
) {
    let (a, b) = window;
    let z = simpson(&unnormalized, a, b, 200_000);
    f.check(z > 0.0 && z.is_finite(), || {
        format!("{label}: numerical evidence integral is {z}")
    });
    for &theta in grid {
        f.close(
            || format!("{label}: posterior density at θ={theta:.6}"),
            closed_form(theta),
            unnormalized(theta) / z,
            tol,
        );
    }
}

#[test]
fn beta_bernoulli_posterior_is_bayes_rule() {
    let mut f = Failures::default();
    let data = [true, true, false, true, true, false, true];
    let successes = data.iter().filter(|&&b| b).count() as f64;
    let failures = data.len() as f64 - successes;

    for (a0, b0) in [(1.0, 1.0), (2.0, 5.0), (0.5, 0.5), (10.0, 3.0)] {
        let model = BetaBernoulli::new(Beta::new(a0, b0).unwrap());
        let post = model.posterior(&data);

        // The closed form claims Beta(a + s, b + f).
        let expected = Beta::new(a0 + successes, b0 + failures).unwrap();
        for p in [0.05, 0.25, 0.5, 0.75, 0.95] {
            f.close(
                || format!("BetaBernoulli({a0},{b0}) posterior pdf at {p} vs Beta(a+s,b+f)"),
                post.pdf(p),
                expected.pdf(p),
                1e-10,
            );
        }

        // And Bayes' rule, computed without using the shortcut at all.
        let prior = Beta::new(a0, b0).unwrap();
        posterior_matches(
            &mut f,
            &format!("BetaBernoulli({a0},{b0})"),
            |theta| {
                let bern = Bernoulli::new(theta).expect("theta in [0,1]");
                let lik: f64 = data
                    .iter()
                    .map(|&b| Discrete::pmf(&bern, u64::from(b)))
                    .product();
                prior.pdf(theta) * lik
            },
            |theta| post.pdf(theta),
            (1e-9, 1.0 - 1e-9),
            &[0.1, 0.3, 0.5, 0.7, 0.9],
            1e-6,
        );

        // The posterior predictive for one more trial is Bernoulli at the
        // posterior mean, which is also ∫ θ·posterior(θ) dθ.
        let pred = model.posterior_predictive(&data);
        let mean_theta = simpson(|t| t * post.pdf(t), 1e-9, 1.0 - 1e-9, 200_000);
        f.close(
            || format!("BetaBernoulli({a0},{b0}) predictive P(success) vs ∫θ·posterior"),
            Discrete::pmf(&pred, 1_u64),
            mean_theta,
            1e-7,
        );
        f.close(
            || format!("BetaBernoulli({a0},{b0}) predictive P(success) vs posterior mean"),
            Discrete::pmf(&pred, 1_u64),
            post.mean(),
            1e-10,
        );
    }
    f.finish("BetaBernoulli = Bayes' rule");
}

#[test]
fn gamma_poisson_posterior_is_bayes_rule() {
    let mut f = Failures::default();
    let data: [u64; 6] = [2, 0, 3, 1, 4, 2];
    let total = data.iter().sum::<u64>() as f64;
    let n = data.len() as f64;

    for (shape, rate) in [(1.0, 1.0), (2.0, 0.5), (5.0, 3.0)] {
        let model = GammaPoisson::new(Gamma::new(shape, rate).unwrap());
        let post = model.posterior(&data);

        // The closed form claims Gamma(shape + Σx, rate + n).
        let expected = Gamma::new(shape + total, rate + n).unwrap();
        for lambda in [0.2, 1.0, 2.0, 4.0, 8.0] {
            f.close(
                || {
                    format!(
                        "GammaPoisson({shape},{rate}) posterior pdf at {lambda} \
                 vs Gamma(a+Σx, b+n)"
                    )
                },
                post.pdf(lambda),
                expected.pdf(lambda),
                1e-10,
            );
        }

        let prior = Gamma::new(shape, rate).unwrap();
        posterior_matches(
            &mut f,
            &format!("GammaPoisson({shape},{rate})"),
            |lambda| {
                if lambda <= 0.0 {
                    return 0.0;
                }
                let pois = Poisson::new(lambda).expect("lambda > 0");
                let lik: f64 = data.iter().map(|&k| Discrete::pmf(&pois, k)).product();
                prior.pdf(lambda) * lik
            },
            |lambda| post.pdf(lambda),
            (1e-9, 40.0),
            &[0.3, 1.0, 2.0, 3.0, 6.0],
            1e-6,
        );

        // The predictive is negative binomial; check it against
        // ∫ Poisson(k | λ)·posterior(λ) dλ.
        let pred = model.posterior_predictive(&data);
        for k in 0_u64..=6 {
            let mixed = simpson(
                |lambda| {
                    if lambda <= 0.0 {
                        return 0.0;
                    }
                    let pois = Poisson::new(lambda).expect("lambda > 0");
                    Discrete::pmf(&pois, k) * post.pdf(lambda)
                },
                1e-9,
                60.0,
                200_000,
            );
            f.close(
                || {
                    format!(
                        "GammaPoisson({shape},{rate}) predictive pmf({k}) \
                 vs ∫Poisson·posterior"
                    )
                },
                Discrete::pmf(&pred, k),
                mixed,
                1e-6,
            );
        }
    }
    f.finish("GammaPoisson = Bayes' rule");
}

#[test]
fn gamma_exponential_posterior_is_bayes_rule() {
    let mut f = Failures::default();
    let data = [0.4, 1.2, 0.7, 2.1, 0.3];
    let total: f64 = data.iter().sum();
    let n = data.len() as f64;

    for (shape, rate) in [(1.0, 1.0), (3.0, 2.0)] {
        let model = GammaExponential::new(Gamma::new(shape, rate).unwrap());
        let post = model.posterior(&data);

        let expected = Gamma::new(shape + n, rate + total).unwrap();
        for lambda in [0.1, 0.5, 1.0, 2.0, 5.0] {
            f.close(
                || {
                    format!(
                        "GammaExponential({shape},{rate}) posterior pdf at {lambda} \
                 vs Gamma(a+n, b+Σx)"
                    )
                },
                post.pdf(lambda),
                expected.pdf(lambda),
                1e-10,
            );
        }

        let prior = Gamma::new(shape, rate).unwrap();
        posterior_matches(
            &mut f,
            &format!("GammaExponential({shape},{rate})"),
            |lambda| {
                if lambda <= 0.0 {
                    return 0.0;
                }
                let e = Exponential::new(lambda).expect("lambda > 0");
                let lik: f64 = data.iter().map(|&x| e.pdf(x)).product();
                prior.pdf(lambda) * lik
            },
            |lambda| post.pdf(lambda),
            (1e-9, 30.0),
            &[0.2, 0.6, 1.0, 1.8, 3.0],
            1e-6,
        );

        // The predictive is a Lomax; check against ∫ Exponential(x|λ)·posterior(λ) dλ.
        let pred = model.posterior_predictive(&data);
        for x in [0.1, 0.5, 1.0, 3.0] {
            let mixed = simpson(
                |lambda| {
                    if lambda <= 0.0 {
                        return 0.0;
                    }
                    let e = Exponential::new(lambda).expect("lambda > 0");
                    e.pdf(x) * post.pdf(lambda)
                },
                1e-9,
                60.0,
                200_000,
            );
            f.close(
                || {
                    format!(
                        "GammaExponential({shape},{rate}) predictive pdf({x}) \
                 vs ∫Exponential·posterior"
                    )
                },
                pred.pdf(x),
                mixed,
                1e-6,
            );
        }
    }
    f.finish("GammaExponential = Bayes' rule");
}

#[test]
fn normal_known_variance_posterior_is_bayes_rule() {
    let mut f = Failures::default();
    let data = [1.2, 0.8, 1.5, 0.9, 1.1];
    let n = data.len() as f64;
    let sample_mean: f64 = data.iter().sum::<f64>() / n;

    for (prior_mean, prior_sd, sigma) in [(0.0, 1.0, 0.5), (1.0, 0.25, 1.0), (-2.0, 3.0, 0.75)] {
        let model = NormalKnownVariance::new(Normal::new(prior_mean, prior_sd).unwrap(), sigma);
        let post = model.posterior(&data);

        // Precision-weighted closed form: τ₁ = τ₀ + n/σ², μ₁ = (τ₀μ₀ + nx̄/σ²)/τ₁.
        let tau0 = 1.0 / (prior_sd * prior_sd);
        let tau1 = tau0 + n / (sigma * sigma);
        let mu1 = (tau0 * prior_mean + n * sample_mean / (sigma * sigma)) / tau1;
        f.close(
            || format!("NormalKnownVariance({prior_mean},{prior_sd},{sigma}) posterior mean"),
            post.mean(),
            mu1,
            1e-10,
        );
        f.close(
            || format!("NormalKnownVariance({prior_mean},{prior_sd},{sigma}) posterior variance"),
            post.variance(),
            1.0 / tau1,
            1e-10,
        );

        let prior = Normal::new(prior_mean, prior_sd).unwrap();
        posterior_matches(
            &mut f,
            &format!("NormalKnownVariance({prior_mean},{prior_sd},{sigma})"),
            |mu| {
                let lik_dist = Normal::new(mu, sigma).expect("sigma > 0");
                let lik: f64 = data.iter().map(|&x| lik_dist.pdf(x)).product();
                prior.pdf(mu) * lik
            },
            |mu| post.pdf(mu),
            (mu1 - 12.0 / tau1.sqrt(), mu1 + 12.0 / tau1.sqrt()),
            &[mu1 - 1.0 / tau1.sqrt(), mu1, mu1 + 1.5 / tau1.sqrt()],
            1e-7,
        );
    }
    f.finish("NormalKnownVariance = Bayes' rule");
}
