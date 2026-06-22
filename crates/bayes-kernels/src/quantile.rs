//! Quantile kernel view plus the generic bisection inverse-CDF helper.
//!
//! Closed-form family quantiles live in [`crate::families`]. The iterative
//! bisection helper remains here because it is an algorithm shared by families
//! without elementary inverse CDFs.

use euclid::numeric::{Boolean, Real};

pub use crate::families::{
    cauchy::cauchy_quantile, exponential::exponential_quantile, frechet::frechet_quantile,
    gumbel::gumbel_quantile, half_normal::half_normal_quantile, laplace::laplace_quantile,
    logistic::logistic_quantile, lognormal::lognormal_quantile, lomax::lomax_quantile,
    normal::normal_quantile, pareto::pareto_quantile, rayleigh::rayleigh_quantile,
    triangular::triangular_quantile, uniform::uniform_quantile, weibull::weibull_quantile,
};

/// Inverse CDF by bisection, write-once over [`Real`].
pub fn inverse_cdf_bisect<T, F>(cdf: F, lo: T, hi: T, p: T) -> T
where
    T: Real,
    F: Fn(T) -> T,
{
    let cdf = &cdf;
    let p = &p;

    let a0 = T::select(&lo.is_finite(), lo.clone(), -T::one());
    let b0 = T::select(&hi.is_finite(), hi.clone(), T::one());

    let widen_trips = T::scalar(64.0);
    let (a, b, _) = T::while_loop(
        (a0, b0, T::zero()),
        |s: &(T, T, T)| {
            let low_bad = cdf(s.0.clone()).gt(p);
            let high_bad = cdf(s.1.clone()).lt(p);
            s.2.lt(&widen_trips).and(&low_bad.or(&high_bad))
        },
        |(a, b, k): (T, T, T)| {
            let span = (b.clone() - a.clone()).abs().max(T::one());
            let low_bad = cdf(a.clone()).gt(p);
            let high_bad = cdf(b.clone()).lt(p);
            let a = T::select(&low_bad, a.clone() - span.clone(), a);
            let b = T::select(&high_bad, b.clone() + span, b);
            (a, b, k + T::one())
        },
    );

    let bisect_trips = T::scalar(100.0);
    let (a, b, _) = T::while_loop(
        (a, b, T::zero()),
        |s: &(T, T, T)| s.2.lt(&bisect_trips),
        |(a, b, k): (T, T, T)| {
            let mid = (a.clone() + b.clone()) * T::half();
            let upper_half = cdf(mid.clone()).ge(p);
            let b = T::select(&upper_half, mid.clone(), b);
            let a = T::select(&upper_half, a, mid);
            (a, b, k + T::one())
        },
    );

    let quantile = (a + b) * T::half();
    let at_or_below = T::select(&p.le(&T::zero()), lo, quantile);
    T::select(&p.ge(&T::one()), hi, at_or_below)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        exponential_cdf, exponential_ln_cdf, exponential_ln_sf, gamma_cdf, gamma_cdf_on_support,
        lognormal_cdf, normal_cdf, normal_ln_cdf, normal_ln_sf, pareto_cdf, weibull_cdf,
    };

    #[test]
    fn full_domain_cdfs_floor_below_support() {
        assert_eq!(gamma_cdf(-1.0_f64, 2.0, 1.0), 0.0);
        assert_eq!(exponential_cdf(-0.5_f64, 2.0), 0.0);
        assert_eq!(lognormal_cdf(-1.0_f64, 0.0, 1.0), 0.0);
        assert_eq!(weibull_cdf(-1.0_f64, 1.5, 2.0), 0.0);
        assert_eq!(pareto_cdf(0.5_f64, 1.0, 2.0), 0.0);

        let g = gamma_cdf(2.0_f64, 3.0, 1.0);
        assert!((g - gamma_cdf_on_support(2.0, 3.0, 1.0)).abs() < 1e-15);

        assert_eq!(exponential_ln_cdf(-1.0_f64, 2.0), f64::NEG_INFINITY);
        assert_eq!(exponential_ln_sf(-1.0_f64, 2.0), 0.0);
    }

    #[test]
    fn bisection_inverts_gamma_cdf_on_half_line() {
        let cdf = |x: f64| gamma_cdf(x, 3.5, 1.5);
        for &p in &[0.05, 0.25, 0.5, 0.75, 0.95] {
            let x = inverse_cdf_bisect(cdf, 0.0, f64::INFINITY, p);
            assert!((cdf(x) - p).abs() < 1e-8, "p={p} cdf(x)={}", cdf(x));
        }
        assert_eq!(inverse_cdf_bisect(cdf, 0.0, f64::INFINITY, 0.0), 0.0);
        assert!(inverse_cdf_bisect(cdf, 0.0, f64::INFINITY, 1.0).is_infinite());
    }

    #[test]
    fn bisection_inverts_normal_cdf_on_real_line() {
        let cdf = |x: f64| normal_cdf(x, 0.0, 1.0);
        for &p in &[0.1, 0.5, 0.9] {
            let x = inverse_cdf_bisect(cdf, f64::NEG_INFINITY, f64::INFINITY, p);
            assert!((cdf(x) - p).abs() < 1e-8);
        }
    }

    #[test]
    fn normal_log_tails_match_log_of_cdf() {
        for &x in &[-3.0_f64, -0.5, 0.0, 0.5, 3.0] {
            let lc = normal_ln_cdf(x, 0.0, 1.0);
            assert!((lc - normal_cdf(x, 0.0, 1.0).ln()).abs() < 1e-9, "x={x}");
            let ls = normal_ln_sf(x, 0.0, 1.0);
            assert!(
                (ls - (1.0 - normal_cdf(x, 0.0, 1.0)).ln()).abs() < 1e-9,
                "x={x}"
            );
        }
    }
}
