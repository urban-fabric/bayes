//! Hostile arguments and parameter validation.
//!
//! An infinite argument must saturate the CDF and vanish the density rather than
//! producing `NaN` — eight families used to return `NaN` from `pdf(±inf)`
//! because the kernel evaluates `inf − inf` there. A `NaN` argument must never
//! come back as a confident *nonzero* probability, though which of the two
//! defensible answers each family gives is only characterized, not asserted; see
//! the test's own comment.
//!
//! `VonMises` has no closed-form CDF, so it sits outside the main sweep and gets
//! its density normalized over the circle here instead.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

// The glob brings every family constructor into scope for the validation sweep.
use bayes_distribution_ext::{Continuous, Moments, VonMises, *};
use bayes_utils::families::{ContCase, continuous_cases};
use bayes_utils::{Failures, rel_error as rel, simpson};

// ---------------------------------------------------------------------------
// Shape and range invariants
// ---------------------------------------------------------------------------

#[test]
fn infinite_arguments_saturate_the_cdf_and_vanish_the_density() {
    let mut f = Failures::default();
    for ContCase { name, dist } in continuous_cases() {
        let c_neg = dist.cdf(f64::NEG_INFINITY);
        f.check(c_neg.abs() < 1e-300, || {
            format!("{name}: cdf(-inf) = {c_neg}, expected 0")
        });
        let c_pos = dist.cdf(f64::INFINITY);
        f.check((c_pos - 1.0).abs() < 1e-12, || {
            format!("{name}: cdf(+inf) = {c_pos}, expected 1")
        });
        for x in [f64::NEG_INFINITY, f64::INFINITY] {
            let p = dist.pdf(x);
            f.check(p == 0.0, || format!("{name}: pdf({x}) = {p}, expected 0"));
        }
    }
    f.finish("infinite arguments");
}

/// Characterizes what a `NaN` argument does. **This pins current behavior; it is
/// not an endorsement of it.**
///
/// The safety property that matters is asserted: a `NaN` argument never produces
/// a confident *nonzero* probability. What is only recorded, not asserted, is
/// that the library is **inconsistent** about which of the two defensible
/// answers it gives:
///
/// * `Exponential`, `Gamma`, `LogNormal`, `Weibull`, … return `cdf(NaN) = 0`
/// * `Uniform`, `Beta`, `Triangular` return `cdf(NaN) = NaN`
///
/// Both follow from the mask-based support test. `x.ge(lo).and(x.le(hi))` is
/// *false* for `NaN`, so the point is reported as "definitely outside the
/// support" and the `NaN` disappears; a family whose CDF has no such mask lets
/// it through. Deciding on one convention is a contract change across ~20
/// families, so it is surfaced here rather than made silently.
#[test]
fn nan_argument_never_yields_a_confident_probability() {
    let mut f = Failures::default();
    let (mut propagating, mut swallowing) = (Vec::new(), Vec::new());
    for ContCase { name, dist } in continuous_cases() {
        let c = dist.cdf(f64::NAN);
        let p = dist.pdf(f64::NAN);
        // The density may vanish or go NaN, but must never be a positive number,
        // and the CDF must never be an interior probability.
        f.check(p.is_nan() || p == 0.0, || {
            format!("{name}: pdf(NaN) = {p}, a positive density for a NaN argument")
        });
        f.check(c.is_nan() || c == 0.0 || c == 1.0, || {
            format!("{name}: cdf(NaN) = {c}, an interior probability for a NaN argument")
        });
        if c.is_nan() {
            propagating.push(name);
        } else {
            swallowing.push(name);
        }
    }
    assert!(
        !propagating.is_empty() && !swallowing.is_empty(),
        "the inconsistency this test documents has changed: {} families propagate \
         NaN through `cdf`, {} swallow it. If the contract was settled one way, \
         replace this test with an assertion of it.",
        propagating.len(),
        swallowing.len()
    );
    f.finish("NaN argument");
}

// ---------------------------------------------------------------------------
// Parameter validation
// ---------------------------------------------------------------------------

#[test]
fn constructors_reject_non_finite_and_out_of_domain_parameters() {
    let mut f = Failures::default();
    let mut bad = |label: &str, rejected: bool| {
        f.check(rejected, || format!("{label} was accepted"));
    };
    bad("Normal(NaN,1)", Normal::new(f64::NAN, 1.0).is_err());
    bad("Normal(0,NaN)", Normal::new(0.0, f64::NAN).is_err());
    bad("Normal(0,inf)", Normal::new(0.0, f64::INFINITY).is_err());
    bad("Normal(inf,1)", Normal::new(f64::INFINITY, 1.0).is_err());
    bad("Normal(0,0)", Normal::new(0.0, 0.0).is_err());
    bad("Normal(0,-1)", Normal::new(0.0, -1.0).is_err());
    bad("Exponential(0)", Exponential::new(0.0).is_err());
    bad("Exponential(-1)", Exponential::new(-1.0).is_err());
    bad("Exponential(NaN)", Exponential::new(f64::NAN).is_err());
    bad("Exponential(inf)", Exponential::new(f64::INFINITY).is_err());
    bad("Gamma(0,1)", Gamma::new(0.0, 1.0).is_err());
    bad("Gamma(1,0)", Gamma::new(1.0, 0.0).is_err());
    bad("Gamma(inf,1)", Gamma::new(f64::INFINITY, 1.0).is_err());
    bad("Gamma(NaN,1)", Gamma::new(f64::NAN, 1.0).is_err());
    bad("Gamma(-1,1)", Gamma::new(-1.0, 1.0).is_err());
    bad("Beta(0,1)", Beta::new(0.0, 1.0).is_err());
    bad("Beta(1,-1)", Beta::new(1.0, -1.0).is_err());
    bad("Beta(NaN,1)", Beta::new(f64::NAN, 1.0).is_err());
    bad("Beta(inf,1)", Beta::new(f64::INFINITY, 1.0).is_err());
    bad("Uniform(1,1)", Uniform::new(1.0, 1.0).is_err());
    bad("Uniform(2,1)", Uniform::new(2.0, 1.0).is_err());
    bad("Uniform(0,inf)", Uniform::new(0.0, f64::INFINITY).is_err());
    bad(
        "Uniform(-inf,0)",
        Uniform::new(f64::NEG_INFINITY, 0.0).is_err(),
    );
    bad("Uniform(NaN,1)", Uniform::new(f64::NAN, 1.0).is_err());
    bad("ChiSquared(0)", ChiSquared::new(0.0).is_err());
    bad("ChiSquared(-1)", ChiSquared::new(-1.0).is_err());
    bad("ChiSquared(inf)", ChiSquared::new(f64::INFINITY).is_err());
    bad("StudentT(0)", StudentT::new(0.0).is_err());
    bad("StudentT(-1)", StudentT::new(-1.0).is_err());
    bad("StudentT(inf)", StudentT::new(f64::INFINITY).is_err());
    bad("Cauchy(0,0)", Cauchy::new(0.0, 0.0).is_err());
    bad("Cauchy(0,-1)", Cauchy::new(0.0, -1.0).is_err());
    bad("Cauchy(NaN,1)", Cauchy::new(f64::NAN, 1.0).is_err());
    bad("Laplace(0,-1)", Laplace::new(0.0, -1.0).is_err());
    bad("Laplace(0,0)", Laplace::new(0.0, 0.0).is_err());
    bad("Logistic(0,0)", Logistic::new(0.0, 0.0).is_err());
    bad("Rayleigh(0)", Rayleigh::new(0.0).is_err());
    bad("Rayleigh(-1)", Rayleigh::new(-1.0).is_err());
    bad("HalfNormal(-1)", HalfNormal::new(-1.0).is_err());
    bad("HalfNormal(0)", HalfNormal::new(0.0).is_err());
    bad("Weibull(0,1)", Weibull::new(0.0, 1.0).is_err());
    bad("Weibull(1,0)", Weibull::new(1.0, 0.0).is_err());
    bad("Weibull(inf,1)", Weibull::new(f64::INFINITY, 1.0).is_err());
    bad("Pareto(0,1)", Pareto::new(0.0, 1.0).is_err());
    bad("Pareto(1,0)", Pareto::new(1.0, 0.0).is_err());
    bad("Lomax(0,1)", Lomax::new(0.0, 1.0).is_err());
    bad("Lomax(1,0)", Lomax::new(1.0, 0.0).is_err());
    bad("InverseGamma(0,1)", InverseGamma::new(0.0, 1.0).is_err());
    bad("InverseGamma(1,0)", InverseGamma::new(1.0, 0.0).is_err());
    bad(
        "FisherSnedecor(0,1)",
        FisherSnedecor::new(0.0, 1.0).is_err(),
    );
    bad(
        "FisherSnedecor(1,0)",
        FisherSnedecor::new(1.0, 0.0).is_err(),
    );
    bad("Gumbel(0,0)", Gumbel::new(0.0, 0.0).is_err());
    bad("Gumbel(0,-1)", Gumbel::new(0.0, -1.0).is_err());
    bad("Frechet(0,1)", Frechet::new(0.0, 1.0).is_err());
    bad("Frechet(1,0)", Frechet::new(1.0, 0.0).is_err());
    bad("LogNormal(0,0)", LogNormal::new(0.0, 0.0).is_err());
    bad("LogNormal(0,-1)", LogNormal::new(0.0, -1.0).is_err());
    bad("LogNormal(NaN,1)", LogNormal::new(f64::NAN, 1.0).is_err());
    bad("Wald(0,1)", Wald::new(0.0, 1.0).is_err());
    bad("Wald(1,0)", Wald::new(1.0, 0.0).is_err());
    bad("Wald(-1,1)", Wald::new(-1.0, 1.0).is_err());
    bad("VonMises(0,-1)", VonMises::new(0.0, -1.0).is_err());
    bad("VonMises(NaN,1)", VonMises::new(f64::NAN, 1.0).is_err());
    bad(
        "VonMises(0,inf)",
        VonMises::new(0.0, f64::INFINITY).is_err(),
    );
    bad("Triangular(0,1,2)", Triangular::new(0.0, 1.0, 2.0).is_err());
    bad(
        "Triangular(1,0,0.5)",
        Triangular::new(1.0, 0.0, 0.5).is_err(),
    );
    bad(
        "Triangular(0,1,NaN)",
        Triangular::new(0.0, 1.0, f64::NAN).is_err(),
    );
    bad(
        "Triangular(0,inf,1)",
        Triangular::new(0.0, f64::INFINITY, 1.0).is_err(),
    );
    f.finish("parameter validation");
}

// ---------------------------------------------------------------------------
// VonMises has no closed-form CDF, so it gets its own density/moment check.
// ---------------------------------------------------------------------------

#[test]
fn von_mises_density_normalizes_over_the_circle() {
    use std::f64::consts::PI;
    let mut f = Failures::default();
    for kappa in [1e-6, 1e-3, 0.5, 2.0, 50.0, 500.0] {
        let d = VonMises::new(0.0, kappa).unwrap();
        let mass = simpson(|t| Continuous::pdf(&d, t), -PI, PI, 200_000);
        f.check((mass - 1.0).abs() < 1e-6, || {
            format!("VonMises(0,{kappa}): ∫pdf over [-π,π] = {mass:.12}, expected 1")
        });
        for i in 0..=100 {
            let t = -PI + 2.0 * PI * i as f64 / 100.0;
            let p = Continuous::pdf(&d, t);
            f.check(p >= 0.0 && p.is_finite(), || {
                format!("VonMises(0,{kappa}): pdf({t}) = {p}")
            });
            // `ln_pdf` is the primitive and stays finite where `pdf` has already
            // underflowed, so only compare where `pdf` carries information.
            if p > 1e-300 {
                f.check(rel(Continuous::ln_pdf(&d, t), p.ln()) < 1e-9, || {
                    format!(
                        "VonMises(0,{kappa}): ln_pdf({t}) = {} but pdf.ln() = {}",
                        Continuous::ln_pdf(&d, t),
                        p.ln()
                    )
                });
            }
        }
        let m = Moments::mean(&d);
        f.check(m.abs() < 1e-9, || {
            format!("VonMises(0,{kappa}): mean = {m}, expected 0")
        });
    }
    f.finish("von Mises density");
}
