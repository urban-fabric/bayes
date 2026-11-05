//! One family is a **transform** of another.
//!
//! Folding, taking a logarithm, taking a reciprocal, shifting the origin, or
//! applying a power — each maps one family onto another, and each carries a
//! Jacobian that the density must pick up. `LogNormal(μ,σ).pdf(x)` has to equal
//! `Normal(μ,σ).pdf(ln x)/x`, and the `/x` is precisely the part that gets
//! dropped by accident.
//!
//! The exponential sits at the centre of this web: `Pareto`, `Weibull` and
//! `Gumbel` are all transformed exponentials, so all three can be checked
//! against one survival function.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use crate::{Failures, quantile_grid};
use bayes_distribution_ext::{Continuous, ContinuousCdf, Moments, *};

const TOL: f64 = 1e-11;

#[test]
fn half_normal_is_the_folded_normal() {
    let mut f = Failures::default();
    for sigma in [0.25, 1.0, 12.0] {
        let hn = HalfNormal::new(sigma).unwrap();
        let n = Normal::new(0.0, sigma).unwrap();
        for x in quantile_grid(|p| hn.inverse_cdf(p)) {
            // Folding doubles the density and maps the CDF through 2Φ − 1.
            f.close(
                || format!("HalfNormal({sigma}) pdf = 2·Normal pdf at {x:.6e}"),
                hn.pdf(x),
                2.0 * n.pdf(x),
                TOL,
            );
            f.close(
                || format!("HalfNormal({sigma}) cdf = 2Φ−1 at {x:.6e}"),
                hn.cdf(x),
                2.0 * n.cdf(x) - 1.0,
                1e-10,
            );
        }
        // E|X| for a centred normal is σ√(2/π).
        f.close(
            || format!("HalfNormal({sigma}) mean = σ√(2/π)"),
            hn.mean(),
            sigma * (2.0 / std::f64::consts::PI).sqrt(),
            TOL,
        );
    }
    f.finish("HalfNormal = |Normal(0,σ)|");
}

#[test]
fn lognormal_is_the_normal_of_the_logarithm() {
    let mut f = Failures::default();
    for (mu, sigma) in [(0.0, 1.0), (-2.0, 0.5), (3.0, 2.0)] {
        let ln_d = LogNormal::new(mu, sigma).unwrap();
        let n = Normal::new(mu, sigma).unwrap();
        for x in quantile_grid(|p| ln_d.inverse_cdf(p)) {
            if x <= 0.0 {
                continue;
            }
            f.close(
                || format!("LogNormal({mu},{sigma}) cdf = Normal cdf of ln at {x:.6e}"),
                ln_d.cdf(x),
                n.cdf(x.ln()),
                1e-10,
            );
            // The Jacobian of the log transform is 1/x.
            f.close(
                || format!("LogNormal({mu},{sigma}) pdf = Normal pdf of ln / x at {x:.6e}"),
                ln_d.pdf(x),
                n.pdf(x.ln()) / x,
                TOL,
            );
        }
    }
    f.finish("LogNormal = exp(Normal)");
}

#[test]
fn inverse_gamma_is_the_reciprocal_of_a_gamma() {
    let mut f = Failures::default();
    for (shape, scale) in [(2.0, 1.0), (4.0, 3.0), (0.75, 0.5)] {
        let ig = InverseGamma::new(shape, scale).unwrap();
        // X ~ InverseGamma(α, β) means 1/X ~ Gamma(α, rate β), so
        // P(X ≤ x) = P(1/X ≥ 1/x) = sf_Gamma(1/x).
        let g = Gamma::new(shape, scale).unwrap();
        for x in quantile_grid(|p| ig.inverse_cdf(p)) {
            if x <= 0.0 {
                continue;
            }
            f.close(
                || format!("InverseGamma({shape},{scale}) cdf = Gamma sf of 1/x at {x:.6e}"),
                ig.cdf(x),
                g.sf(1.0 / x),
                1e-9,
            );
        }
    }
    f.finish("InverseGamma = 1/Gamma");
}

#[test]
fn lomax_is_a_pareto_shifted_to_the_origin() {
    let mut f = Failures::default();
    for (shape, scale) in [(3.0, 2.0), (1.5, 0.5), (8.0, 10.0)] {
        let l = Lomax::new(shape, scale).unwrap();
        // Lomax(α, λ) is Pareto(scale = λ, shape = α) translated by −λ.
        let p = Pareto::new(scale, shape).unwrap();
        for x in quantile_grid(|q| l.inverse_cdf(q)) {
            f.close(
                || format!("Lomax({shape},{scale}) cdf = Pareto cdf shifted at {x:.6e}"),
                l.cdf(x),
                p.cdf(x + scale),
                1e-11,
            );
            f.close(
                || format!("Lomax({shape},{scale}) pdf = Pareto pdf shifted at {x:.6e}"),
                l.pdf(x),
                p.pdf(x + scale),
                1e-11,
            );
        }
    }
    f.finish("Lomax = shifted Pareto");
}

#[test]
fn frechet_is_the_reciprocal_of_a_weibull() {
    let mut f = Failures::default();
    for (shape, scale) in [(3.0, 2.0), (0.8, 1.0), (5.0, 0.4)] {
        let fr = Frechet::new(shape, scale).unwrap();
        // X ~ Frechet(α, s) means 1/X ~ Weibull(scale = 1/s, shape = α), so
        // P(X ≤ x) = P(1/X ≥ 1/x) = sf_Weibull(1/x).
        let w = Weibull::new(1.0 / scale, shape).unwrap();
        for x in quantile_grid(|q| fr.inverse_cdf(q)) {
            if x <= 0.0 {
                continue;
            }
            f.close(
                || format!("Frechet({shape},{scale}) cdf = Weibull sf of 1/x at {x:.6e}"),
                fr.cdf(x),
                w.sf(1.0 / x),
                1e-10,
            );
        }
    }
    f.finish("Frechet = 1/Weibull");
}

#[test]
fn pareto_and_weibull_and_gumbel_are_transformed_exponentials() {
    let mut f = Failures::default();
    let unit = Exponential::new(1.0).unwrap();

    // ln(X/scale) ~ Exponential(shape) for a Pareto.
    for (scale, shape) in [(1.0, 2.5), (3.0, 0.75)] {
        let p = Pareto::new(scale, shape).unwrap();
        let e = Exponential::new(shape).unwrap();
        for x in quantile_grid(|q| p.inverse_cdf(q)) {
            f.close(
                || format!("Pareto({scale},{shape}) sf = Exponential sf of ln(x/s) at {x:.6e}"),
                p.sf(x),
                e.sf((x / scale).ln()),
                1e-10,
            );
        }
    }

    // (X/λ)^k ~ Exponential(1) for a Weibull.
    for (lambda, k) in [(1.0, 2.0), (4.0, 0.5), (0.3, 7.0)] {
        let w = Weibull::new(lambda, k).unwrap();
        for x in quantile_grid(|q| w.inverse_cdf(q)) {
            f.close(
                || format!("Weibull({lambda},{k}) sf = Exponential(1) sf of (x/λ)^k at {x:.6e}"),
                w.sf(x),
                unit.sf((x / lambda).powf(k)),
                1e-10,
            );
        }
    }

    // μ − β·ln E ~ Gumbel(μ, β) for E ~ Exponential(1), so
    // cdf_Gumbel(x) = sf_Exp(exp(−(x−μ)/β)).
    for (mu, beta) in [(0.0, 1.0), (-3.0, 0.25), (5.0, 9.0)] {
        let g = Gumbel::new(mu, beta).unwrap();
        for x in quantile_grid(|q| g.inverse_cdf(q)) {
            f.close(
                || format!("Gumbel({mu},{beta}) cdf = Exponential(1) sf of exp(−z) at {x:.6e}"),
                g.cdf(x),
                unit.sf((-(x - mu) / beta).exp()),
                1e-10,
            );
        }
    }

    // |X − μ| ~ Exponential(1/b) for a Laplace, so the upper tail is half of it.
    for (mu, b) in [(0.0, 1.0), (2.0, 0.5)] {
        let l = Laplace::new(mu, b).unwrap();
        let e = Exponential::new(1.0 / b).unwrap();
        for t in [0.0, 0.25, 1.0, 4.0, 20.0] {
            f.close(
                || format!("Laplace({mu},{b}) sf(μ+{t}) = ½·Exponential(1/b) sf({t})"),
                l.sf(mu + t),
                0.5 * e.sf(t),
                1e-11,
            );
        }
    }

    f.finish("transformed exponentials");
}
