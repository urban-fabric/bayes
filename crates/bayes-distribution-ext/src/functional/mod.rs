//! Functional adapters and combinators for the distribution trait spine.
//!
//! The concrete distribution types cover common named families. This module covers
//! the other functional use case: lift pure functions into the same traits, then
//! compose them with log-likelihood, interval probability, truncation, and
//! bijective transforms.
//!
//! * `continuous` / `discrete` — closure-backed PDF/PMF (and optional CDF/quantile)
//!   adapters: [`ContinuousDensity`], [`ContinuousFn`], [`DiscreteMass`], [`DiscreteFn`].
//! * `ext` — blanket extension traits ([`ContinuousDistributionExt`],
//!   [`ContinuousCdfExt`], [`DiscreteDistributionExt`]) that compose any distribution.

mod continuous;
mod discrete;
mod ext;

pub use continuous::{ContinuousDensity, ContinuousFn};
pub use discrete::{DiscreteFn, DiscreteMass};
pub use ext::{ContinuousCdfExt, ContinuousDistributionExt, DiscreteDistributionExt};

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::{Continuous, ContinuousCdf, Discrete, DiscreteCdf};

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn closure_backed_continuous_distribution_implements_the_spine() {
        let unit = ContinuousFn::new(
            0.0,
            1.0,
            |x| if (0.0..=1.0).contains(&x) { 1.0 } else { 0.0 },
            |x: f64| x.clamp(0.0, 1.0),
            |p: f64| p.clamp(0.0, 1.0),
        )
        .unwrap();

        close(unit.pdf(0.25), 1.0, 1e-12);
        close(unit.cdf(-1.0), 0.0, 1e-12);
        close(unit.cdf(0.25), 0.25, 1e-12);
        close(unit.cdf(2.0), 1.0, 1e-12);
        close(unit.inverse_cdf(0.75), 0.75, 1e-12);
        close(unit.ln_likelihood(&[0.2, 0.4, 0.8]), 0.0, 1e-12);
        close(unit.probability_between(0.2, 0.7), 0.5, 1e-12);
        close(unit.ln_probability_between(0.2, 0.7).exp(), 0.5, 1e-12);
    }

    #[test]
    fn interval_probability_propagates_malformed_cdfs() {
        let bad = ContinuousFn::new(0.0, 1.0, |_: f64| 1.0, |_: f64| f64::NAN, |p: f64| p).unwrap();

        assert!(bad.probability_between(0.2, 0.7).is_nan());
        assert!(bad.ln_probability_between(0.2, 0.7).is_nan());
    }

    #[test]
    fn extension_combinators_apply_existing_distribution_adapters() {
        let unit = ContinuousFn::new(
            0.0,
            1.0,
            |_| 1.0,
            |x: f64| x.clamp(0.0, 1.0),
            |p: f64| p.clamp(0.0, 1.0),
        )
        .unwrap();

        let shifted = unit.affine(2.0, -1.0).unwrap();
        close(shifted.pdf(0.0), 0.5, 1e-12);

        let truncated = unit.truncated(0.25, 0.75).unwrap();
        close(truncated.pdf(0.5), 2.0, 1e-12);
        close(truncated.cdf(0.5), 0.5, 1e-12);

        let max = unit.maximum_of(4).unwrap();
        close(max.cdf(0.5), 0.5_f64.powi(4), 1e-12);
        let median = unit.order_statistic(5, 3).unwrap();
        close(median.pdf(0.5), 1.875, 1e-12);
        close(unit.cumulative_hazard(0.5), -0.5_f64.ln(), 1e-12);
        close(unit.hazard(0.5), 2.0, 1e-12);
        close(unit.ln_hazard(0.5), 2.0_f64.ln(), 1e-12);
    }

    #[test]
    fn closure_backed_discrete_distribution_implements_the_spine() {
        let coin = DiscreteFn::new(
            0u64,
            1u64,
            |k: u64| match k {
                0 => 0.25,
                1 => 0.75,
                _ => 0.0,
            },
            |k: u64| if k == 0 { 0.25 } else { 1.0 },
            |p: f64| if p <= 0.25 { 0 } else { 1 },
        );

        close(coin.pmf(1), 0.75, 1e-12);
        close(coin.cdf(0), 0.25, 1e-12);
        close(coin.cdf(7), 1.0, 1e-12);
        close(coin.inverse_cdf(0.9) as f64, 1.0, 1e-12);
        close(
            coin.ln_likelihood(&[1, 1, 0]).exp(),
            0.75 * 0.75 * 0.25,
            1e-12,
        );
    }
}
