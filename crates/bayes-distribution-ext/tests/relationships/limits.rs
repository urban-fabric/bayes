//! **Limiting relations.** Where one family becomes another as a parameter runs
//! to an extreme.
//!
//! These are approximations, so the tolerance is loose and chosen from the known
//! rate of convergence rather than from machine precision. They earn their place
//! anyway: a limit exercises a family at exactly the parameter values where its
//! closed form is most likely to lose precision — a `Binomial` with `n = 10⁶`,
//! a `StudentT` with `ν = 10⁸` — and it checks the *answer*, not just that the
//! code returns without panicking.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use crate::Failures;
use bayes_distribution_ext::{Continuous, ContinuousCdf, Moments, *};

/// `Binomial(n, λ/n) → Poisson(λ)`. The error is `O(λ²/n)`.
#[test]
fn the_binomial_converges_to_the_poisson_as_trials_grow() {
    let mut f = Failures::default();
    for lambda in [0.5, 2.0, 8.0] {
        let pois = Poisson::new(lambda).unwrap();
        for n in [10_000_u64, 1_000_000] {
            let bin = Binomial::new(n, lambda / n as f64).unwrap();
            // Rate of convergence is ~λ²/n; allow a generous multiple of it.
            let tol = 20.0 * lambda * lambda / n as f64;
            for k in 0_u64..=12 {
                f.close(
                    || format!("Binomial({n},{lambda}/n).pmf({k}) vs Poisson({lambda}).pmf({k})"),
                    Discrete::pmf(&bin, k),
                    Discrete::pmf(&pois, k),
                    tol.max(1e-12),
                );
            }
            f.close(
                || format!("Binomial({n},{lambda}/n) mean vs Poisson mean"),
                Moments::mean(&bin),
                Moments::mean(&pois),
                1e-12,
            );
        }
    }
    f.finish("Binomial -> Poisson");
}

/// `NegativeBinomial(r, r/(r+λ)) → Poisson(λ)` as `r → ∞`: the gamma mixing
/// distribution concentrates, so the overdispersion vanishes.
#[test]
fn the_negative_binomial_converges_to_the_poisson_as_the_shape_grows() {
    let mut f = Failures::default();
    for lambda in [1.0, 5.0] {
        let pois = Poisson::new(lambda).unwrap();
        for r in [1e5, 1e7] {
            let nb = NegativeBinomial::new(r, r / (r + lambda)).unwrap();
            let tol = 50.0 * lambda / r;
            for k in 0_u64..=10 {
                f.close(
                    || format!("NegativeBinomial({r:.0e}, r/(r+{lambda})).pmf({k}) vs Poisson"),
                    Discrete::pmf(&nb, k),
                    Discrete::pmf(&pois, k),
                    tol.max(1e-12),
                );
            }
        }
    }
    f.finish("NegativeBinomial -> Poisson");
}

/// `Hypergeometric(N, K, n) → Binomial(n, K/N)` as the population grows with the
/// success fraction held fixed: sampling without replacement stops mattering.
#[test]
fn the_hypergeometric_converges_to_the_binomial_as_the_population_grows() {
    let mut f = Failures::default();
    let draws = 10_u64;
    for frac in [0.1, 0.5] {
        for population in [100_000_u64, 10_000_000] {
            let successes = (population as f64 * frac) as u64;
            let hg = Hypergeometric::new(population, successes, draws).unwrap();
            let bin = Binomial::new(draws, frac).unwrap();
            // The correction term is O(draws/population).
            let tol = 50.0 * draws as f64 / population as f64;
            for k in 0..=draws {
                f.close(
                    || {
                        format!(
                            "Hypergeometric({population},{frac}·N,{draws}).pmf({k}) vs \
                     Binomial({draws},{frac}).pmf({k})"
                        )
                    },
                    Discrete::pmf(&hg, k),
                    Discrete::pmf(&bin, k),
                    tol,
                );
            }
        }
    }
    f.finish("Hypergeometric -> Binomial");
}

/// `BetaBinomial(n, a, b) → Binomial(n, a/(a+b))` as `a + b → ∞`: the Beta prior
/// concentrates on its mean.
#[test]
fn the_beta_binomial_converges_to_the_binomial_as_the_prior_concentrates() {
    let mut f = Failures::default();
    let n = 10_u64;
    for mean_p in [0.25, 0.5] {
        for conc in [1e5, 1e7] {
            let (a, b) = (mean_p * conc, (1.0 - mean_p) * conc);
            let bb = BetaBinomial::new(n, a, b).unwrap();
            let bin = Binomial::new(n, mean_p).unwrap();
            let tol = 100.0 / conc;
            for k in 0..=n {
                f.close(
                    || {
                        format!(
                            "BetaBinomial({n},{a:.0e},{b:.0e}).pmf({k}) vs \
                     Binomial({n},{mean_p}).pmf({k})"
                        )
                    },
                    Discrete::pmf(&bb, k),
                    Discrete::pmf(&bin, k),
                    tol,
                );
            }
        }
    }
    f.finish("BetaBinomial -> Binomial");
}

/// `StudentT(ν) → Normal(0,1)` as `ν → ∞`. The error is `O(1/ν)`.
#[test]
fn student_t_converges_to_the_standard_normal_as_freedom_grows() {
    let mut f = Failures::default();
    let n = Normal::new(0.0, 1.0).unwrap();
    for nu in [1e4, 1e6, 1e8] {
        let t = StudentT::new(nu).unwrap();
        let tol = 50.0 / nu;
        for x in [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0] {
            f.close(
                || format!("StudentT({nu:.0e}).pdf({x}) vs Normal(0,1).pdf({x})"),
                t.pdf(x),
                n.pdf(x),
                tol.max(1e-14),
            );
            f.close(
                || format!("StudentT({nu:.0e}).cdf({x}) vs Normal(0,1).cdf({x})"),
                t.cdf(x),
                n.cdf(x),
                tol.max(1e-14),
            );
        }
        f.close(
            || format!("StudentT({nu:.0e}) variance -> 1"),
            t.variance(),
            1.0,
            tol,
        );
    }
    f.finish("StudentT -> Normal");
}

/// A standardized `ChiSquared(k)` approaches the standard normal, since it is a
/// sum of `k` i.i.d. squares.
#[test]
fn a_standardized_chi_squared_approaches_the_normal_as_freedom_grows() {
    let mut f = Failures::default();
    let n = Normal::new(0.0, 1.0).unwrap();
    for k in [1e5, 1e7] {
        let chi = ChiSquared::new(k).unwrap();
        // Standardize by the exact mean k and standard deviation √(2k).
        let (mu, sd) = (k, (2.0 * k).sqrt());
        // The skewness of χ²_k is √(8/k), which bounds the CDF error.
        let tol = 20.0 * (8.0 / k).sqrt();
        for z in [-2.0, -1.0, 0.0, 1.0, 2.0] {
            f.close(
                || format!("ChiSquared({k:.0e}) standardized cdf at z={z} vs Normal"),
                chi.cdf(mu + z * sd),
                n.cdf(z),
                tol,
            );
        }
        f.close(
            || format!("ChiSquared({k:.0e}) mean = k"),
            chi.mean(),
            k,
            1e-12,
        );
        f.close(
            || format!("ChiSquared({k:.0e}) variance = 2k"),
            chi.variance(),
            2.0 * k,
            1e-12,
        );
    }
    f.finish("ChiSquared -> Normal");
}

/// `FisherSnedecor(d1, d2) → ChiSquared(d1)/d1` as `d2 → ∞`: the denominator
/// chi-square concentrates on 1.
#[test]
fn fisher_snedecor_converges_to_a_scaled_chi_squared_as_the_denominator_grows() {
    let mut f = Failures::default();
    for d1 in [2.0, 6.0] {
        let chi = ChiSquared::new(d1).unwrap();
        for d2 in [1e5, 1e7] {
            let fd = FisherSnedecor::new(d1, d2).unwrap();
            let tol = 200.0 / d2;
            for x in [0.2, 0.5, 1.0, 2.0, 4.0] {
                // P(F ≤ x) -> P(χ²_{d1}/d1 ≤ x) = P(χ²_{d1} ≤ d1·x)
                f.close(
                    || {
                        format!(
                            "F({d1},{d2:.0e}).cdf({x}) vs ChiSquared({d1}).cdf({})",
                            d1 * x
                        )
                    },
                    fd.cdf(x),
                    chi.cdf(d1 * x),
                    tol,
                );
            }
        }
    }
    f.finish("FisherSnedecor -> scaled ChiSquared");
}

/// A `Gamma` with large shape approaches a normal with the same mean and
/// variance — the shape parameter acts like a sample size.
#[test]
fn a_large_shape_gamma_approaches_the_normal() {
    let mut f = Failures::default();
    let n = Normal::new(0.0, 1.0).unwrap();
    for shape in [1e5, 1e7] {
        let g = Gamma::new(shape, 1.0).unwrap();
        let (mu, sd) = (shape, shape.sqrt());
        let tol = 20.0 * 2.0 / shape.sqrt();
        for z in [-2.0, -1.0, 0.0, 1.0, 2.0] {
            f.close(
                || format!("Gamma({shape:.0e},1) standardized cdf at z={z} vs Normal"),
                g.cdf(mu + z * sd),
                n.cdf(z),
                tol,
            );
        }
    }
    f.finish("Gamma -> Normal");
}

/// `VonMises(μ, κ) → Normal(μ, 1/√κ)` as `κ → ∞`: a tightly concentrated
/// circular distribution is locally Gaussian.
#[test]
fn von_mises_approaches_the_wrapped_normal_at_high_concentration() {
    let mut f = Failures::default();
    for kappa in [1e4, 1e6] {
        let vm = VonMises::new(0.0, kappa).unwrap();
        let n = Normal::new(0.0, 1.0 / kappa.sqrt()).unwrap();
        let tol = 50.0 / kappa;
        for z in [-2.0, -1.0, 0.0, 1.0, 2.0] {
            let x = z / kappa.sqrt();
            f.close(
                || format!("VonMises(0,{kappa:.0e}).pdf({x:.6e}) vs Normal(0,1/√κ)"),
                Continuous::pdf(&vm, x),
                n.pdf(x),
                tol * n.pdf(x).max(1.0),
            );
        }
    }
    f.finish("VonMises -> Normal");
}
