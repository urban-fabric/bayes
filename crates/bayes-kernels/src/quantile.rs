use super::*;
use euclid::numeric::LazyBool;

// --- closed-form quantiles (inverse CDFs) ---
//
// The inverse-CDF maps `p ∈ (0, 1)` to the support — a total formula on the open
// interval, so (as with `logit`) clamping `p` and the `p = 0`/`p = 1` boundary are
// the caller's branch. Variable powers `(·)^{1/shape}` go through
// `exp((1/shape)·ln(·))` because the lazy `powf` takes a *constant* exponent.

/// Exponential quantile: `−ln(1 − p) / rate`.
pub fn exponential_quantile<T: LazyReal>(p: T, rate: T) -> T {
    -((T::one() - p).ln()) / rate
}

/// Logistic quantile: `location + scale · ln(p / (1 − p))`.
pub fn logistic_quantile<T: LazyReal>(p: T, location: T, scale: T) -> T {
    location + scale * (p.clone() / (T::one() - p)).ln()
}

/// Cauchy quantile: `location + scale · tan(π·(p − ½))`.
pub fn cauchy_quantile<T: LazyReal>(p: T, location: T, scale: T) -> T {
    location + scale * (T::pi() * (p - T::half())).tan()
}

/// Laplace quantile: `location − scale · sign(q) · ln(1 − 2|q|)`, `q = p − ½`.
pub fn laplace_quantile<T: LazyReal>(p: T, location: T, scale: T) -> T {
    let q = p - T::half();
    location - scale * q.clone().signum() * (T::one() - T::two() * q.abs()).ln()
}

/// Uniform quantile: `lo + width · p`, with `width = hi − lo`.
pub fn uniform_quantile<T: LazyReal>(p: T, lo: T, width: T) -> T {
    lo + width * p
}

/// Weibull quantile: `scale · (−ln(1 − p))^{1/shape}`.
pub fn weibull_quantile<T: LazyReal>(p: T, scale: T, shape: T) -> T {
    let base = -((T::one() - p).ln());
    scale * ((T::one() / shape) * base.ln()).exp()
}

/// Gumbel quantile: `location − scale · ln(−ln p)`.
pub fn gumbel_quantile<T: LazyReal>(p: T, location: T, scale: T) -> T {
    location - scale * (-(p.ln())).ln()
}

/// Fréchet quantile: `scale · (−ln p)^{−1/shape}`.
pub fn frechet_quantile<T: LazyReal>(p: T, scale: T, shape: T) -> T {
    let base = -(p.ln());
    scale * ((-(T::one() / shape)) * base.ln()).exp()
}

/// Pareto quantile: `minimum · (1 − p)^{−1/shape}`.
pub fn pareto_quantile<T: LazyReal>(p: T, minimum: T, shape: T) -> T {
    let base = T::one() - p;
    minimum * ((-(T::one() / shape)) * base.ln()).exp()
}

/// Lomax quantile: `scale · ((1 − p)^{−1/shape} − 1)`.
pub fn lomax_quantile<T: LazyReal>(p: T, scale: T, shape: T) -> T {
    let base = T::one() - p;
    scale * (((-(T::one() / shape)) * base.ln()).exp() - T::one())
}

/// Rayleigh quantile: `scale · √(−2·ln(1 − p))`.
pub fn rayleigh_quantile<T: LazyReal>(p: T, scale: T) -> T {
    scale * (-T::two() * (T::one() - p).ln()).sqrt()
}

/// Normal quantile: `mean + std·√2·erf⁻¹(2p − 1)`.
pub fn normal_quantile<T: LazyReal>(p: T, mean: T, std_dev: T) -> T {
    mean + std_dev * T::sqrt_2() * (T::two() * p - T::one()).erf_inv()
}

/// Half-normal quantile: `scale·√2·erf⁻¹(p)`.
pub fn half_normal_quantile<T: LazyReal>(p: T, scale: T) -> T {
    scale * T::sqrt_2() * p.erf_inv()
}

/// Log-normal quantile: the Normal quantile on the log scale, exponentiated —
/// `exp(mean_log + σ_log·√2·erf⁻¹(2p − 1))`.
pub fn lognormal_quantile<T: LazyReal>(p: T, mean_log: T, std_log: T) -> T {
    normal_quantile(p, mean_log, std_log).exp()
}

// --- iterative quantile (no elementary inverse) ---
//
// The families whose CDF has no closed-form inverse — Gamma, Beta, Student's t, F, Wald,
// Inverse-Gamma — invert it by bisection. Written once over [`LazyReal`] through
// [`LazyCompare::while_loop`], the same routine is a host `while` loop on `f64`/`f32` and a
// graph `While` on a tensor backend, so the quantile is no longer host-only.

/// Inverse CDF by bisection, write-once over [`LazyReal`]. `cdf` must be monotone
/// non-decreasing; `lo`/`hi` are the support bounds (either may be `±∞`). An infinite end is
/// widened geometrically into a finite bracket, then the bracket is bisected for a fixed
/// trip count (≈ full `f64` precision). The `p ≤ 0` / `p ≥ 1` ends return `lo` / `hi`.
pub fn inverse_cdf_bisect<T, F>(cdf: F, lo: T, hi: T, p: T) -> T
where
    T: LazyReal,
    F: Fn(T) -> T,
{
    let cdf = &cdf;
    let p = &p;

    // Seed a finite working bracket; an infinite end starts at ∓1 and is widened below.
    let a0 = T::select(&lo.is_finite(), lo.clone(), -T::one());
    let b0 = T::select(&hi.is_finite(), hi.clone(), T::one());

    // Widen until `cdf(a) ≤ p ≤ cdf(b)`, pushing the offending end out by the current span
    // (≥ 1) each trip; 64 trips cover any finite double-precision quantile.
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

    // Bisect the now-bracketing `[a, b]` for a fixed trip count.
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

    #[test]
    fn full_domain_cdfs_floor_below_support() {
        assert_eq!(gamma_cdf(-1.0_f64, 2.0, 1.0), 0.0);
        assert_eq!(exponential_cdf(-0.5_f64, 2.0), 0.0);
        assert_eq!(lognormal_cdf(-1.0_f64, 0.0, 1.0), 0.0);
        assert_eq!(weibull_cdf(-1.0_f64, 1.5, 2.0), 0.0);
        assert_eq!(pareto_cdf(0.5_f64, 1.0, 2.0), 0.0);
        // The interior agrees with the on-support kernel where both apply.
        let g = gamma_cdf(2.0_f64, 3.0, 1.0);
        assert!((g - gamma_cdf_on_support(2.0, 3.0, 1.0)).abs() < 1e-15);
        // Stable log-CDF/SF tails floor to ±∞ / 0 off support.
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
