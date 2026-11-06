//! The generic bisection inverse-CDF helper.
//!
//! Closed-form family quantiles live in [`crate::families`] (and re-export at the
//! crate root). This iterative bisection helper lives here because it is an
//! algorithm shared by families without elementary inverse CDFs.

use euclid_numeric::{Compare, Element, Natural, Real};

/// Fixed bracket-widening steps (the bracket is valid long before this; extra steps are
/// idempotent because the `select` updates freeze once `cdf(a) ≤ p ≤ cdf(b)`).
const WIDEN_TRIPS: usize = 64;
/// Fixed bisection steps. 100 halvings of the search coordinate resolves the quantile to
/// ~1.2e-27 *relative* to its distance from the nearer bracket endpoint; see
/// [`inverse_cdf_bisect`] for why the coordinate rather than `x` itself is bisected.
const BISECT_TRIPS: usize = 100;
/// Half-width of the bisection coordinate. `exp(-745)` is the smallest positive `f64`, so this
/// is the widest range of scales a double can express at all.
const COORD_HALF_WIDTH: f64 = 745.0;
/// Integer-bisection steps for a discrete quantile.
///
/// 64 *spans* the `u64` count range but does not *resolve* it, which is a different thing. The
/// loop keeps the invariant `cdf(lo − 1) < p ≤ cdf(hi)` and returns `lo`, so it needs the
/// bracket closed to `lo == hi`. Starting from `[0, 2^64]`, 64 halvings are all consumed
/// bringing `hi` down and leave `lo` and `hi` one apart — undecided — so an unbounded family
/// returned `lo` unchanged: `Poisson(2.5).inverse_cdf(0.1)` gave `0` when `cdf(0) = 0.082 < 0.1`
/// and the answer is `1`. The extra trips close the bracket; they are idempotent once it has
/// closed, so overshooting costs only iterations.
const DISCRETE_BISECT_TRIPS: usize = 68;

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
///
/// # Why the coordinate, not `x`, is bisected
///
/// Halving `[a, b]` in `x` gives **absolute** resolution `(b−a)/2^n`, which is the wrong kind of
/// resolution for a quantile. `Gamma(10⁻³, 1)` has `cdf(x) ≈ x^0.001`, so its median sits at
/// `0.5^1000 ≈ 9.3e-302` while its bracket is `[0, 1]`: 100 halvings reach only `~7.9e-31`, and
/// *every* `p` from `1e-6` to `0.9` used to come back as the same `~3.9e-31` — a quantile
/// function that ignored its argument.
///
/// So the search runs over a coordinate `v ∈ [−745, 745]` that is exponential at both ends:
///
/// ```text
///     x(v) = a + (w/2)·exp(v)    for v ≤ 0        (w = b − a)
///     x(v) = b − (w/2)·exp(−v)   for v > 0
/// ```
///
/// This is continuous and monotone — both branches meet at the bracket midpoint `v = 0` — and it
/// reaches `a` and `b` only in the limits. Near either endpoint `d x / x = d v`, so 100 halvings
/// of a 1490-wide coordinate give ~1.2e-27 *relative* precision, and `exp(±745)` spans every
/// scale an `f64` can represent. In the middle of the bracket the resolution is `(w/2)·1.2e-27`,
/// far below the `w·2.2e-16` ulp there, so well-scaled quantiles are unaffected.
///
/// What this cannot fix is a quantile that is not representable: `Beta(0.01, 0.01)` has
/// `Q(0.75) = 1 − 1e-30`, and `1 − 1e-30` *is* `1.0` in `f64`. Such a quantile returns the
/// support bound.
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

    // The exponential coordinate described above. Both branches are always evaluated and the
    // `select` picks one, keeping the mapping usable on a carrier with no host branching.
    let half_span = (b.clone() - a.clone()) * T::half();
    let point = |v: T| -> T {
        let lower = a.clone() + half_span.clone() * v.clone().exp();
        let upper = b.clone() - half_span.clone() * (-v.clone()).exp();
        T::select(&v.le(&T::zero()), lower, upper)
    };

    let mut va = -T::scalar(COORD_HALF_WIDTH);
    let mut vb = T::scalar(COORD_HALF_WIDTH);
    for _ in 0..BISECT_TRIPS {
        let vm = (va.clone() + vb.clone()) * T::half();
        let upper_half = cdf(point(vm.clone())).ge(p);
        vb = T::select(&upper_half, vm.clone(), vb);
        va = T::select(&upper_half, va, vm);
    }

    // If widening never managed to bracket `p`, there is nothing to bisect.
    //
    // That happens when the CDF never reaches `p` at all — which it does for
    // real: a family whose CDF plateaus one ulp below 1 can never satisfy
    // `cdf(b) >= 1 - 1.1e-16`, so the doubling above runs its full trip count
    // and leaves `b` at roughly `2^64`. Returning the midpoint of that bracket
    // reported `1.8446744e19` as the quantile of `InverseGamma(0.5, 1)` and
    // `FisherSnedecor(1, 1)`, a number with no meaning whatsoever.
    //
    // The support bound is the honest answer and the one this routine's own
    // documentation promises for an unrepresentable quantile.
    let unbracketed_above = cdf(b.clone()).lt(p);
    let unbracketed_below = cdf(a.clone()).gt(p);

    let quantile = point((va + vb) * T::half());
    let quantile = T::select(&unbracketed_above, hi.clone(), quantile);
    let quantile = T::select(&unbracketed_below, lo.clone(), quantile);
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
