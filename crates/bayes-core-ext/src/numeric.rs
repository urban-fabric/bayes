//! Shared numeric helpers for distributions without a closed-form quantile.

/// Numeric inverse CDF by bisection on a continuous, monotone non-decreasing
/// `cdf` over `[lo, hi]` (either bound may be infinite). Used by Gamma, Beta,
/// Student's t, F, Wald, and Inverse-Gamma — distributions whose quantile has no
/// elementary form.
pub fn inverse_cdf_bisect(cdf: impl Fn(f64) -> f64, lo: f64, hi: f64, p: f64) -> f64 {
    if p <= 0.0 {
        return lo;
    }
    if p >= 1.0 {
        return hi;
    }
    let mut a = if lo.is_finite() { lo } else { -1.0 };
    let mut b = if hi.is_finite() { hi } else { 1.0 };
    let mut step = 1.0;
    while !lo.is_finite() && cdf(a) > p {
        a -= step;
        step *= 2.0;
    }
    step = 1.0;
    while !hi.is_finite() && cdf(b) < p {
        b += step;
        step *= 2.0;
    }
    let mut mid = 0.5 * (a + b);
    for _ in 0..200 {
        mid = 0.5 * (a + b);
        if cdf(mid) < p {
            a = mid;
        } else {
            b = mid;
        }
        if (b - a).abs() <= 1e-13 * (1.0 + mid.abs()) {
            break;
        }
    }
    mid
}
