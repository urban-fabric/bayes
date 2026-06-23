//! The generic bisection inverse-CDF helper.
//!
//! Closed-form family quantiles live in [`crate::families`] (and re-export at the
//! crate root). This iterative bisection helper lives here because it is an
//! algorithm shared by families without elementary inverse CDFs.

use euclid::numeric::{Compare, Element, Natural, Real};

/// Fixed bracket-widening steps (the bracket is valid long before this; extra steps are
/// idempotent because the `select` updates freeze once `cdf(a) ≤ p ≤ cdf(b)`).
const WIDEN_TRIPS: usize = 64;
/// Fixed bisection steps — 100 halvings resolves the bracket to machine precision.
const BISECT_TRIPS: usize = 100;
/// Integer-bisection steps for a discrete quantile — 64 spans the whole `u64` count range.
const DISCRETE_BISECT_TRIPS: usize = 64;

/// Inverse CDF for a **discrete** distribution: the smallest count `k ∈ [k_min, k_max]` with
/// `cdf(k) ≥ p`.
///
/// The integer bisection is carried out in the **real** surface (`Natural` is a semiring with no
/// subtraction, and `k_max` may be `u64::MAX`, which would overflow integer midpoints), bridged
/// back to a count with [`Natural::from_real`]. Like [`inverse_cdf_bisect`] it is a fixed `for`-
/// loop with branchless `select`, so it is carrier-agnostic — a host search on a scalar, an
/// unrolled graph on a tensor of counts.
pub fn discrete_inverse_cdf<N, F>(k_min: N, k_max: N, p: N::Real, cdf: F) -> N
where
    N: Natural + Compare,
    N::Real: Real + Compare<Mask = <N as Compare>::Mask>,
    F: Fn(N) -> N::Real,
{
    let p = &p;
    let one = N::Real::one();
    let two = one.clone() + one.clone();
    let mut lo = k_min.to_real();
    let mut hi = k_max.to_real();
    for _ in 0..DISCRETE_BISECT_TRIPS {
        let mid = ((lo.clone() + hi.clone()) / two.clone()).floor();
        // cdf(mid) < p ⇒ the quantile is strictly above mid; else mid is a valid upper bound.
        let below = cdf(N::from_real(mid.clone())).lt(p);
        lo = <N::Real as Compare>::select(&below, mid.clone() + one.clone(), lo);
        hi = <N::Real as Compare>::select(&below, hi, mid);
    }
    N::from_real(lo)
}

/// Inverse CDF by bisection, write-once over [`Real`].
///
/// Both phases are **bounded `for`-loops with branchless `select`**, not host `while`-loops, so
/// the routine is carrier-agnostic: on an eager scalar it runs the iterations directly; on a
/// deferred / tensor carrier it unrolls into the computation graph (a host `while` over a tensor
/// mask can't lower — the predicate is a mask, not a host `bool`). Each step is idempotent once
/// converged, so a fixed trip count matches the original early-stopping loop.
pub fn inverse_cdf_bisect<T, F>(cdf: F, lo: T, hi: T, p: T) -> T
where
    T: Real,
    F: Fn(T) -> T,
{
    let cdf = &cdf;
    let p = &p;

    let mut a = T::select(&lo.is_finite(), lo.clone(), -T::one());
    let mut b = T::select(&hi.is_finite(), hi.clone(), T::one());

    // Widen the bracket until cdf(a) ≤ p ≤ cdf(b).
    for _ in 0..WIDEN_TRIPS {
        let span = (b.clone() - a.clone()).abs().max(T::one());
        let low_bad = cdf(a.clone()).gt(p);
        let high_bad = cdf(b.clone()).lt(p);
        a = T::select(&low_bad, a.clone() - span.clone(), a);
        b = T::select(&high_bad, b.clone() + span, b);
    }

    // Bisect.
    for _ in 0..BISECT_TRIPS {
        let mid = (a.clone() + b.clone()) * T::half();
        let upper_half = cdf(mid.clone()).ge(p);
        b = T::select(&upper_half, mid.clone(), b);
        a = T::select(&upper_half, a, mid);
    }

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
