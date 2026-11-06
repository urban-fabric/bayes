//! Quantile invariants.
//!
//! `cdf(inverse_cdf(p)) == p` is the headline, and it is where the bisection
//! quantile's absolute-resolution defect surfaced: `Gamma(1e-3,1)` returned the
//! same value for every `p` from `1e-6` to `0.9`. The boundary cases matter just
//! as much — `inverse_cdf(0)` must reach the support infimum rather than
//! stopping at some finite value, and an out-of-range probability must not come
//! back as a plausible interior point.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

// The glob brings every family constructor into scope for the validation sweep.
use bayes_utils::Failures;
use bayes_utils::families::{
    ContCase, continuous_cases, quantile_unrepresentable_at, quantiles_are_unrepresentable,
};

// ---------------------------------------------------------------------------
// Shape and range invariants
// ---------------------------------------------------------------------------

#[test]
fn quantile_inverts_the_cdf() {
    let mut f = Failures::default();
    for ContCase { name, dist } in continuous_cases() {
        if quantiles_are_unrepresentable(&*dist) {
            continue;
        }
        for &p in &[
            1e-6,
            1e-3,
            0.01,
            0.1,
            0.25,
            0.5,
            0.75,
            0.9,
            0.99,
            0.999,
            1.0 - 1e-6,
        ] {
            if quantile_unrepresentable_at(&*dist, p) {
                continue;
            }
            let x = dist.inverse_cdf(p);
            if !x.is_finite() {
                continue;
            }
            let back = dist.cdf(x);
            f.check((back - p).abs() < 1e-6, || {
                format!(
                    "{name}: cdf(inverse_cdf({p})) = {back}, off by {:.3e}",
                    (back - p).abs()
                )
            });
        }
    }
    f.finish("quantile inverts cdf");
}

#[test]
fn quantile_is_monotone_in_p() {
    let mut f = Failures::default();
    for ContCase { name, dist } in continuous_cases() {
        let mut prev = f64::NEG_INFINITY;
        for i in 1..1000 {
            let p = i as f64 / 1000.0;
            let x = dist.inverse_cdf(p);
            f.check(x >= prev - 1e-9 * prev.abs().max(1.0), || {
                format!("{name}: inverse_cdf decreased at p={p}: {prev} -> {x}")
            });
            prev = x;
        }
    }
    f.finish("quantile monotonicity");
}

#[test]
fn quantile_reaches_the_support_bounds_at_p_zero_and_one() {
    let mut f = Failures::default();
    for ContCase { name, dist } in continuous_cases() {
        let (lo, hi) = (dist.lo(), dist.hi());
        let at0 = dist.inverse_cdf(0.0);
        let at1 = dist.inverse_cdf(1.0);
        // An infinite bound must be reached exactly; `lo + tol` is NaN there,
        // which would make any comparison below silently false.
        let ok0 = if lo.is_infinite() {
            at0 == lo
        } else {
            at0 <= lo + 1e-9 * lo.abs().max(1.0)
        };
        let ok1 = if hi.is_infinite() {
            at1 == hi
        } else {
            at1 >= hi - 1e-9 * hi.abs().max(1.0)
        };
        f.check(ok0, || {
            format!("{name}: inverse_cdf(0) = {at0:.6e}, support min is {lo:.6e}")
        });
        f.check(ok1, || {
            format!("{name}: inverse_cdf(1) = {at1:.6e}, support max is {hi:.6e}")
        });
    }
    f.finish("quantile at p=0 and p=1");
}

#[test]
fn out_of_range_probabilities_are_not_extrapolated() {
    let mut f = Failures::default();
    for ContCase { name, dist } in continuous_cases() {
        for &p in &[-0.5, -1e-12, 1.0 + 1e-12, 1.5] {
            let x = dist.inverse_cdf(p);
            let interior = x > dist.lo() && x < dist.hi() && x.is_finite();
            f.check(!interior, || {
                format!(
                    "{name}: inverse_cdf({p}) = {x:.6e}, an interior point for an \
                     out-of-range probability"
                )
            });
        }
    }
    f.finish("out-of-range quantile");
}

// ---------------------------------------------------------------------------
// The load-bearing cross-checks: independent code paths for the same quantity
// ---------------------------------------------------------------------------

#[test]
fn median_splits_the_mass_in_half() {
    let mut f = Failures::default();
    for ContCase { name, dist } in continuous_cases() {
        let m = dist.median();
        if !m.is_finite() || quantiles_are_unrepresentable(&*dist) {
            continue;
        }
        let c = dist.cdf(m);
        f.check((c - 0.5).abs() < 1e-6, || {
            format!("{name}: cdf(median={m:.6e}) = {c}, expected 0.5")
        });
    }
    f.finish("median");
}

// ---------------------------------------------------------------------------
// Non-finite inputs
// ---------------------------------------------------------------------------
