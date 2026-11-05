//! One family **is** another at particular parameters.
//!
//! The purest form of the relation: no transform, no limit, just a different
//! name for the same law. `ChiSquared(k)` is `Gamma(k/2, ½)`, `Exponential(λ)`
//! is `Gamma(1, λ)` and `Weibull(1/λ, 1)`, `StudentT(1)` is the standard
//! Cauchy, `Beta(1,1)` is `Uniform(0,1)`, `Rayleigh(σ)` is `Weibull(σ√2, 2)`.
//!
//! These pin down *parameterization*, which is the one thing a doc comment can
//! get wrong without any test noticing. If `Gamma`'s second argument were a
//! scale rather than a rate, the first relation here would need `2.0` instead
//! of `0.5` — so the test decides the convention, not the comment.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use crate::{Failures, quantile_grid};
use bayes_distribution_ext::{Continuous, ContinuousCdf, Moments, *};

const TOL: f64 = 1e-11;

#[test]
fn chi_squared_is_a_gamma_with_half_the_freedom_and_rate_one_half() {
    let mut f = Failures::default();
    for k in [1.0, 2.0, 3.0, 7.5, 40.0, 1000.0] {
        let chi = ChiSquared::new(k).unwrap();
        // ChiSquared(k) = Gamma(shape = k/2, rate = 1/2). If the Gamma's second
        // parameter were a *scale* this would need to be 2.0, so the relation
        // also pins down which convention `Gamma` uses.
        let gamma = Gamma::new(k / 2.0, 0.5).unwrap();
        for x in quantile_grid(|p| chi.inverse_cdf(p)) {
            f.close(
                || format!("ChiSquared({k}) vs Gamma({},½) pdf at {x:.6e}", k / 2.0),
                chi.pdf(x),
                gamma.pdf(x),
                TOL,
            );
            f.close(
                || format!("ChiSquared({k}) vs Gamma cdf at {x:.6e}"),
                chi.cdf(x),
                gamma.cdf(x),
                TOL,
            );
        }
        f.close(
            || format!("ChiSquared({k}) vs Gamma mean"),
            chi.mean(),
            gamma.mean(),
            TOL,
        );
        f.close(
            || format!("ChiSquared({k}) vs Gamma variance"),
            chi.variance(),
            gamma.variance(),
            TOL,
        );
        f.close(
            || format!("ChiSquared({k}) vs Gamma entropy"),
            chi.entropy(),
            gamma.entropy(),
            1e-9,
        );
    }
    f.finish("ChiSquared = Gamma(k/2, 1/2)");
}

#[test]
fn exponential_is_a_unit_shape_gamma_and_a_unit_shape_weibull() {
    let mut f = Failures::default();
    for rate in [0.25, 1.0, 3.0, 100.0] {
        let e = Exponential::new(rate).unwrap();
        let g = Gamma::new(1.0, rate).unwrap();
        // Weibull(scale = 1/rate, shape = 1) is the same law.
        let w = Weibull::new(1.0 / rate, 1.0).unwrap();
        for x in quantile_grid(|p| e.inverse_cdf(p)) {
            f.close(
                || format!("Exponential({rate}) vs Gamma(1,{rate}) pdf at {x:.6e}"),
                e.pdf(x),
                g.pdf(x),
                TOL,
            );
            f.close(
                || format!("Exponential({rate}) vs Weibull(1/{rate},1) pdf at {x:.6e}"),
                e.pdf(x),
                w.pdf(x),
                TOL,
            );
            f.close(
                || format!("Exponential({rate}) vs Weibull cdf at {x:.6e}"),
                e.cdf(x),
                w.cdf(x),
                TOL,
            );
        }
        f.close(
            || format!("Exponential({rate}) vs Gamma mean"),
            e.mean(),
            g.mean(),
            TOL,
        );
        f.close(
            || format!("Exponential({rate}) vs Weibull mean"),
            e.mean(),
            w.mean(),
            1e-9,
        );
    }
    // ChiSquared(2) is Exponential(1/2) — the one place the two chains meet.
    let chi2 = ChiSquared::new(2.0).unwrap();
    let exp_half = Exponential::new(0.5).unwrap();
    for x in quantile_grid(|p| chi2.inverse_cdf(p)) {
        f.close(
            || format!("ChiSquared(2) vs Exponential(½) cdf at {x:.6e}"),
            chi2.cdf(x),
            exp_half.cdf(x),
            TOL,
        );
    }
    f.finish("Exponential reparameterizations");
}

#[test]
fn student_t_with_one_degree_of_freedom_is_the_standard_cauchy() {
    let mut f = Failures::default();
    let t = StudentT::new(1.0).unwrap();
    let c = Cauchy::new(0.0, 1.0).unwrap();
    for x in [-50.0, -5.0, -1.0, -0.25, 0.0, 0.25, 1.0, 5.0, 50.0] {
        f.close(
            || format!("StudentT(1) vs Cauchy(0,1) pdf at {x}"),
            t.pdf(x),
            c.pdf(x),
            TOL,
        );
        f.close(
            || format!("StudentT(1) vs Cauchy(0,1) cdf at {x}"),
            t.cdf(x),
            c.cdf(x),
            1e-10,
        );
    }
    for p in [0.01, 0.1, 0.5, 0.9, 0.99] {
        f.close(
            || format!("StudentT(1) vs Cauchy(0,1) quantile at {p}"),
            t.inverse_cdf(p),
            c.inverse_cdf(p),
            1e-7,
        );
    }
    f.finish("StudentT(1) = Cauchy(0,1)");
}

#[test]
fn unit_beta_is_the_standard_uniform() {
    let mut f = Failures::default();
    let b = Beta::new(1.0, 1.0).unwrap();
    let u = Uniform::new(0.0, 1.0).unwrap();
    for x in [0.0, 0.1, 0.5, 0.9, 1.0] {
        f.close(
            || format!("Beta(1,1) vs Uniform(0,1) pdf at {x}"),
            b.pdf(x),
            u.pdf(x),
            TOL,
        );
        f.close(
            || format!("Beta(1,1) vs Uniform(0,1) cdf at {x}"),
            b.cdf(x),
            u.cdf(x),
            TOL,
        );
    }
    f.close(
        || "Beta(1,1) vs Uniform(0,1) mean".into(),
        b.mean(),
        u.mean(),
        TOL,
    );
    f.close(
        || "Beta(1,1) vs Uniform(0,1) variance".into(),
        b.variance(),
        u.variance(),
        TOL,
    );
    f.close(
        || "Beta(1,1) vs Uniform(0,1) entropy".into(),
        b.entropy(),
        u.entropy(),
        TOL,
    );
    f.finish("Beta(1,1) = Uniform(0,1)");
}

#[test]
fn rayleigh_is_a_weibull_of_shape_two() {
    let mut f = Failures::default();
    for sigma in [0.5, 1.0, 7.0] {
        let r = Rayleigh::new(sigma).unwrap();
        // Rayleigh(σ) = Weibull(scale = σ√2, shape = 2).
        let w = Weibull::new(sigma * 2.0_f64.sqrt(), 2.0).unwrap();
        for x in quantile_grid(|p| r.inverse_cdf(p)) {
            f.close(
                || format!("Rayleigh({sigma}) vs Weibull(σ√2,2) pdf at {x:.6e}"),
                r.pdf(x),
                w.pdf(x),
                TOL,
            );
            f.close(
                || format!("Rayleigh({sigma}) vs Weibull(σ√2,2) cdf at {x:.6e}"),
                r.cdf(x),
                w.cdf(x),
                TOL,
            );
        }
        f.close(
            || format!("Rayleigh({sigma}) vs Weibull mean"),
            r.mean(),
            w.mean(),
            1e-9,
        );
        f.close(
            || format!("Rayleigh({sigma}) vs Weibull variance"),
            r.variance(),
            w.variance(),
            1e-9,
        );
    }
    f.finish("Rayleigh = Weibull(σ√2, 2)");
}
