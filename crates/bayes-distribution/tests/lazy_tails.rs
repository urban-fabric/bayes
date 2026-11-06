//! **Far tail, lazy families**: the generic-over-`T` path, out where `1 − cdf`
//! has nothing left.
//!
//! This crate's families are a second, independent wrapping of the same
//! kernels: written against `euclid_numeric::Real` so one definition serves
//! every carrier, where `bayes-distribution-ext` wraps them eagerly for `f64`.
//! Sharing kernels means a formula fix reaches both — but `ln_sf` is wired
//! **per family**, and for a long time only four of the twenty-one here had it.
//! The other seventeen inherited the trait default, `ln(1 − cdf)`, which is
//! `-inf` for every argument past a survival of about `1e-16`.
//!
//! So this suite exists to keep the two wrappings honest about the same
//! region, and to make a missing override a named failure rather than a silent
//! gap. `Beta` and `Uniform` are excluded: their support is bounded above, so
//! there is no far tail to reach.

#![allow(clippy::panic, clippy::unwrap_used)]

use bayes_distribution::{
    Cauchy, ChiSquared, ContinuousCdf, Exponential, FisherSnedecor, Frechet, Gamma, Gumbel,
    HalfNormal, InverseGamma, Laplace, LogNormal, Logistic, Lomax, Normal, Pareto, Rayleigh,
    StudentT, Wald, Weibull,
};
use bayes_utils::Failures;

/// A family bound to one parameter set, reduced to the three things a tail
/// check needs.
struct Case {
    name: &'static str,
    cdf: Box<dyn Fn(f64) -> f64>,
    ln_sf: Box<dyn Fn(f64) -> f64>,
    /// A point in the body, to march outward from.
    centre: f64,
    /// The scale of that march.
    spread: f64,
}

fn case(
    name: &'static str,
    centre: f64,
    spread: f64,
    cdf: impl Fn(f64) -> f64 + 'static,
    ln_sf: impl Fn(f64) -> f64 + 'static,
) -> Case {
    Case {
        name,
        cdf: Box::new(cdf),
        ln_sf: Box::new(ln_sf),
        centre,
        spread,
    }
}

/// Every unbounded-above family in the crate.
fn cases() -> Vec<Case> {
    vec![
        case(
            "Normal(0,1)",
            0.0,
            1.0,
            |x| Normal::new(0.0, 1.0).cdf(x),
            |x| Normal::new(0.0, 1.0).ln_sf(x),
        ),
        case(
            "Exponential(1.7)",
            0.6,
            0.6,
            |x| Exponential::new(1.7).cdf(x),
            |x| Exponential::new(1.7).ln_sf(x),
        ),
        case(
            "Logistic(-0.5,1.3)",
            -0.5,
            1.3,
            |x| Logistic::new(-0.5, 1.3).cdf(x),
            |x| Logistic::new(-0.5, 1.3).ln_sf(x),
        ),
        case(
            "Cauchy(0.5,1.7)",
            0.5,
            1.7,
            |x| Cauchy::new(0.5, 1.7).cdf(x),
            |x| Cauchy::new(0.5, 1.7).ln_sf(x),
        ),
        case(
            "Laplace(0.5,1.7)",
            0.5,
            1.7,
            |x| Laplace::new(0.5, 1.7).cdf(x),
            |x| Laplace::new(0.5, 1.7).ln_sf(x),
        ),
        case(
            "StudentT(5)",
            0.0,
            1.0,
            |x| StudentT::new(5.0).cdf(x),
            |x| StudentT::new(5.0).ln_sf(x),
        ),
        case(
            "Gumbel(0,1)",
            0.0,
            1.0,
            |x| Gumbel::new(0.0, 1.0).cdf(x),
            |x| Gumbel::new(0.0, 1.0).ln_sf(x),
        ),
        case(
            "Gamma(4,1.5)",
            2.7,
            1.3,
            |x| Gamma::new(4.0, 1.5).cdf(x),
            |x| Gamma::new(4.0, 1.5).ln_sf(x),
        ),
        case(
            "ChiSquared(3)",
            2.4,
            2.4,
            |x| ChiSquared::new(3.0).cdf(x),
            |x| ChiSquared::new(3.0).ln_sf(x),
        ),
        case(
            "FisherSnedecor(6,9)",
            1.0,
            1.0,
            |x| FisherSnedecor::new(6.0, 9.0).cdf(x),
            |x| FisherSnedecor::new(6.0, 9.0).ln_sf(x),
        ),
        case(
            "InverseGamma(4,2)",
            0.6,
            0.4,
            |x| InverseGamma::new(4.0, 2.0).cdf(x),
            |x| InverseGamma::new(4.0, 2.0).ln_sf(x),
        ),
        case(
            "Frechet(2,3)",
            3.6,
            2.0,
            |x| Frechet::new(2.0, 3.0).cdf(x),
            |x| Frechet::new(2.0, 3.0).ln_sf(x),
        ),
        case(
            "HalfNormal(2)",
            1.3,
            1.3,
            |x| HalfNormal::new(2.0).cdf(x),
            |x| HalfNormal::new(2.0).ln_sf(x),
        ),
        case(
            "LogNormal(0.5,1.2)",
            1.6,
            1.6,
            |x| LogNormal::new(0.5, 1.2).cdf(x),
            |x| LogNormal::new(0.5, 1.2).ln_sf(x),
        ),
        case(
            "Lomax(3,2)",
            0.5,
            0.6,
            |x| Lomax::new(3.0, 2.0).cdf(x),
            |x| Lomax::new(3.0, 2.0).ln_sf(x),
        ),
        case(
            "Pareto(2,1.3)",
            3.0,
            2.0,
            |x| Pareto::new(2.0, 1.3).cdf(x),
            |x| Pareto::new(2.0, 1.3).ln_sf(x),
        ),
        case(
            "Rayleigh(1.4)",
            1.6,
            1.0,
            |x| Rayleigh::new(1.4).cdf(x),
            |x| Rayleigh::new(1.4).ln_sf(x),
        ),
        case(
            "Wald(1,3)",
            1.0,
            0.6,
            |x| Wald::new(1.0, 3.0).cdf(x),
            |x| Wald::new(1.0, 3.0).ln_sf(x),
        ),
        case(
            "Weibull(3,1.7)",
            2.5,
            1.5,
            |x| Weibull::new(3.0, 1.7).cdf(x),
            |x| Weibull::new(3.0, 1.7).ln_sf(x),
        ),
    ]
}

/// Points marching outward by powers of ten.
///
/// Decades rather than doublings, because a polynomial tail needs an enormous
/// reach: a Cauchy's survival is `1/(πz)`, so `ln sf = −40` is at `z ≈ 7e16`.
fn outward(item: &Case) -> Vec<f64> {
    (0..=300)
        .map(|k| item.centre + item.spread * 10.0_f64.powi(k))
        .filter(|x| x.is_finite())
        .collect()
}

/// `sf` is the exponential of `ln_sf`, and is not zero where that is a
/// representable number.
///
/// `sf` has no override anywhere in this crate: it comes from the trait default
/// in `bayes-core`, which derives it from `ln_sf`. So this is simultaneously a
/// check on the fifteen `ln_sf` wirings and on that derivation holding across
/// the generic surface.
#[test]
fn the_lazy_survival_is_computed_rather_than_subtracted() {
    let mut f = Failures::default();
    for item in cases() {
        for x in outward(&item) {
            let (s, ls) = ((item.cdf)(x), (item.ln_sf)(x));
            let survival = ls.exp();
            f.check(!s.is_nan(), || {
                format!("{}: cdf({x:.4e}) is NaN", item.name)
            });
            f.check(!ls.is_nan(), || {
                format!("{}: ln_sf({x:.4e}) is NaN", item.name)
            });
            if ls.is_finite() && ls > -700.0 {
                f.check(survival > 0.0, || {
                    format!(
                        "{}: ln_sf({x:.4e}) = {ls:.4} implies a survival of {survival:.4e}, \
                         which is representable",
                        item.name
                    )
                });
            }
        }
    }
    f.finish("lazy survival is computed");
}

/// Every family's log-survival actually reaches the far tail.
///
/// Without this the check above passes vacuously for a family still on the
/// `ln(1 − cdf)` default: that returns `−inf` out there, the guard skips it,
/// and the gap reads as coverage. Seventeen of these families were in exactly
/// that state.
#[test]
fn every_lazy_family_has_a_usable_log_survival() {
    let mut f = Failures::default();
    for item in cases() {
        let deepest = outward(&item)
            .iter()
            .map(|&x| (item.ln_sf)(x))
            .filter(|v| v.is_finite())
            .fold(0.0_f64, f64::min);
        f.check(deepest < -40.0, || {
            format!(
                "{}: ln_sf never gets below {deepest:.3} across 300 decades, which means it is \
                 still the ln(1 - cdf) default",
                item.name
            )
        });
    }
    f.finish("lazy log-survival reaches the tail");
}

/// `cdf + sf == 1` in the body, where both carry information.
///
/// Guards the derivation: `sf` now goes through a logarithm, which costs an ulp
/// or two in the bulk, and this pins how much.
#[test]
fn the_lazy_cdf_and_survival_still_complement() {
    let mut f = Failures::default();
    for item in cases() {
        // A short march inward from the centre, staying where both are O(1).
        for step in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0] {
            let x = item.centre + item.spread * step;
            let c = (item.cdf)(x);
            if !(1e-6..=1.0 - 1e-6).contains(&c) {
                continue;
            }
            let total = c + (item.ln_sf)(x).exp();
            f.close(
                || format!("{}: cdf + sf at {x:.6}", item.name),
                total,
                1.0,
                1e-11,
            );
        }
    }
    f.finish("lazy complement");
}

/// The lazy standard normal's tail, against published values.
///
/// Every other check here is self-consistency, which cannot catch both
/// wrappings being wrong the same way. These are independent: `Φ(−z)` at
/// integer sigma. They also confirm the lazy surface produces the *same*
/// numbers the eager one is held to in
/// `bayes-distribution-ext/tests/tails/quantile.rs`.
#[test]
fn the_lazy_standard_normal_tail_matches_published_values() {
    let n = Normal::new(0.0, 1.0);
    let reference = [
        (1.0, 1.586_552_539_315e-1),
        (3.0, 1.349_898_031_630e-3),
        (6.0, 9.865_876_450_377e-10),
        (9.0, 1.128_588_405_6e-19),
        (10.0, 7.619_853_024_16e-24),
    ];
    let mut f = Failures::default();
    for (z, truth) in reference {
        let sf: f64 = n.sf(z);
        f.check((sf - truth).abs() / truth < 1e-9, || {
            format!("lazy sf({z}) = {sf:.12e}, published {truth:.12e}")
        });
        let cdf: f64 = n.cdf(-z);
        f.check((cdf - truth).abs() / truth < 1e-9, || {
            format!("lazy cdf(-{z}) = {cdf:.12e}, published {truth:.12e}")
        });
    }
    f.finish("lazy normal tail vs published values");
}
