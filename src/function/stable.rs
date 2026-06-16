//! Numerically **stable log-domain helpers** — the small building blocks that keep
//! likelihoods, mixtures, and tail probabilities accurate when ordinary arithmetic
//! would underflow or cancel.
//!
//! * [`ln_sum_exp`] / [`ln_add_exp`] — `ln Σ eˣⁱ` via the max-shift, for mixture and
//!   categorical log-densities.
//! * [`softmax`] / [`log_softmax`] / [`ln_mean_exp`] — normalized log-weights,
//!   log-probabilities, and log averages using the same shift-invariant reduction.
//! * [`ln_1p_exp`] (softplus) / [`ln_1m_exp`] — `ln(1 ± eˣ)` without losing digits.
//! * [`xlogy`] — `x·ln y` with the `0·ln 0 = 0` convention (entropy / KL terms).
//! * [`erfcx`] — the scaled complementary error function `eˣ² erfc(x)`, which stays
//!   finite and accurate in the right tail where `erfc` underflows to zero.

use super::erf::erfc;
use std::f64::consts::PI;

/// `ln(eᵃ + eᵇ)`, computed without overflow by factoring out the larger term.
///
/// Treats `-∞` as the log of zero, so `ln_add_exp(x, -∞) == x`.
pub fn ln_add_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    hi + (lo - hi).exp().ln_1p()
}

/// The **log-sum-exp** `ln Σᵢ eˣⁱ`, stable via the max-shift
/// `m + ln Σᵢ e^{xᵢ − m}`. Empty input and all-`-∞` input both give `-∞`.
pub fn ln_sum_exp(xs: &[f64]) -> f64 {
    let mut max = f64::NEG_INFINITY;
    for &x in xs {
        if x > max {
            max = x;
        }
    }
    if max == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    if max == f64::INFINITY {
        return f64::INFINITY;
    }
    let sum: f64 = xs.iter().map(|&x| (x - max).exp()).sum();
    max + sum.ln()
}

/// The **log-mean-exp** `ln((1/n) Σᵢ eˣⁱ)`, computed as
/// [`ln_sum_exp`] minus `ln n`.
///
/// Empty input returns `-∞`, matching [`ln_sum_exp`]'s "log zero" convention.
pub fn ln_mean_exp(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NEG_INFINITY;
    }
    ln_sum_exp(xs) - (xs.len() as f64).ln()
}

/// The stable **log-softmax** `xᵢ - ln Σⱼ eˣʲ`.
///
/// This is the log-domain form of normalized weights. It is shift-invariant by
/// construction (`log_softmax(x + c) = log_softmax(x)`), the exact fact proved in
/// riemann `RiemannLean/Markov/LogSumExp.lean` as `logSumExp_add_const`.
/// Returns an empty vector for empty input, `NaN` input, or all-`-∞` input. If
/// one or more coordinates are `+∞`, the limiting distribution is uniform over
/// those coordinates and zero elsewhere.
pub fn log_softmax(xs: &[f64]) -> Vec<f64> {
    if xs.is_empty() || xs.iter().any(|x| x.is_nan()) {
        return Vec::new();
    }

    let pos_inf = xs.iter().filter(|&&x| x == f64::INFINITY).count();
    if pos_inf > 0 {
        let log_mass = -(pos_inf as f64).ln();
        return xs
            .iter()
            .map(|&x| {
                if x == f64::INFINITY {
                    log_mass
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect();
    }

    let log_z = ln_sum_exp(xs);
    if log_z == f64::NEG_INFINITY || log_z.is_nan() {
        return Vec::new();
    }
    xs.iter().map(|&x| x - log_z).collect()
}

/// The stable **softmax** `exp(xᵢ) / Σⱼ exp(xⱼ)`.
///
/// The returned vector is a probability vector whenever it is non-empty: entries
/// are nonnegative and sum to one up to floating-point roundoff. This is the
/// Rust counterpart of riemann `softmax_isDistribution`.
pub fn softmax(xs: &[f64]) -> Vec<f64> {
    log_softmax(xs).into_iter().map(f64::exp).collect()
}

/// `ln(eᵃ − eᵇ)`, stable when `a ≥ b`.
///
/// Returns `NaN` if `a < b`, and `-∞` when the two terms are exactly equal.
pub fn ln_sub_exp(a: f64, b: f64) -> f64 {
    if a < b {
        return f64::NAN;
    }
    if a == b {
        return f64::NEG_INFINITY;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    if a == f64::INFINITY {
        return f64::INFINITY;
    }
    a + ln_1m_exp(b - a)
}

/// Softplus `ln(1 + eˣ)`, accurate and overflow-free across the whole range
/// (Mächler's branch points).
pub fn ln_1p_exp(x: f64) -> f64 {
    if x <= -37.0 {
        x.exp()
    } else if x <= 18.0 {
        x.exp().ln_1p()
    } else if x <= 33.3 {
        x + (-x).exp()
    } else {
        x
    }
}

/// `ln(1 − eˣ)` for `x < 0`, via Mächler's two-region split (returns `NaN` for
/// `x ≥ 0`, where the argument is non-positive).
pub fn ln_1m_exp(x: f64) -> f64 {
    if x >= 0.0 {
        return f64::NAN;
    }
    if x > -std::f64::consts::LN_2 {
        (-x.exp_m1()).ln()
    } else {
        (-x.exp()).ln_1p()
    }
}

/// `x·ln y` with the convention `0·ln 0 = 0` (and `0·ln y = 0` for any `y`),
/// the form that keeps entropy and divergence sums finite.
pub fn xlogy(x: f64, y: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x * y.ln() }
}

/// The **scaled complementary error function** `erfcx(x) = eˣ² erfc(x)`.
///
/// For `x ≳ 4`, `erfc(x)` underflows but `erfcx(x) ~ 1/(x√π)` stays finite, so this
/// is the right primitive for stable Normal tail log-probabilities. Computed
/// directly from `erfc` near the origin and by a continued fraction in the tail.
pub fn erfcx(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 * (x * x).exp() - erfcx(-x);
    }
    if x < 4.0 {
        return (x * x).exp() * erfc(x);
    }
    let mut t = x;
    for k in (1..=60).rev() {
        t = x + (k as f64) * 0.5 / t;
    }
    1.0 / (PI.sqrt() * t)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn ln_sum_exp_matches_naive_in_safe_range() {
        let xs = [-1.0_f64, 0.5, 2.0, -3.0];
        let naive = xs.iter().map(|&x| x.exp()).sum::<f64>().ln();
        close(ln_sum_exp(&xs), naive, 1e-12);
        assert_eq!(ln_sum_exp(&[]), f64::NEG_INFINITY);
        assert_eq!(
            ln_sum_exp(&[f64::NEG_INFINITY, f64::NEG_INFINITY]),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn ln_sum_exp_survives_extreme_magnitudes() {
        let xs = [1000.0, 1000.0];
        close(ln_sum_exp(&xs), 1000.0 + 2.0_f64.ln(), 1e-9);
        let xs = [-1000.0, -1001.0];
        close(ln_sum_exp(&xs), ln_add_exp(-1000.0, -1001.0), 1e-12);
    }

    #[test]
    fn ln_mean_exp_and_softmax_are_stable_under_large_shifts() {
        let xs = [1000.0, 1000.0];
        close(ln_mean_exp(&xs), 1000.0, 1e-12);

        let p = softmax(&xs);
        assert_eq!(p.len(), 2);
        close(p[0], 0.5, 1e-12);
        close(p[1], 0.5, 1e-12);

        let shifted = [1200.0, 1200.0];
        assert_eq!(softmax(&xs), softmax(&shifted));
    }

    #[test]
    fn log_softmax_returns_log_probabilities() {
        let lp = log_softmax(&[0.0, 0.0, f64::NEG_INFINITY]);
        assert_eq!(lp.len(), 3);
        close(lp[0], -2.0_f64.ln(), 1e-12);
        close(lp[1], -2.0_f64.ln(), 1e-12);
        assert_eq!(lp[2], f64::NEG_INFINITY);

        let p = softmax(&[f64::INFINITY, 1.0, f64::INFINITY]);
        close(p[0], 0.5, 1e-12);
        close(p[1], 0.0, 1e-12);
        close(p[2], 0.5, 1e-12);

        assert!(softmax(&[]).is_empty());
        assert!(softmax(&[f64::NEG_INFINITY, f64::NEG_INFINITY]).is_empty());
        assert!(softmax(&[0.0, f64::NAN]).is_empty());
    }

    #[test]
    fn ln_add_exp_basic() {
        close(ln_add_exp(0.0, 0.0), 2.0_f64.ln(), 1e-12);
        assert_eq!(ln_add_exp(3.0, f64::NEG_INFINITY), 3.0);
    }

    #[test]
    fn ln_sub_exp_basic() {
        close(
            ln_sub_exp(2.0, 1.0).exp(),
            2.0_f64.exp() - 1.0_f64.exp(),
            1e-12,
        );
        assert_eq!(ln_sub_exp(3.0, f64::NEG_INFINITY), 3.0);
        assert_eq!(ln_sub_exp(3.0, 3.0), f64::NEG_INFINITY);
        assert!(ln_sub_exp(1.0, 2.0).is_nan());
    }

    #[test]
    fn softplus_and_log1m() {
        close(ln_1p_exp(0.0), 2.0_f64.ln(), 1e-12);
        close(ln_1p_exp(50.0), 50.0, 1e-9);
        for &x in &[-0.1, -0.5, -1.0, -5.0] {
            close(ln_1m_exp(x), (1.0 - x.exp()).ln(), 1e-9);
        }
        assert!(ln_1m_exp(0.5).is_nan());
    }

    #[test]
    fn xlogy_zero_convention() {
        assert_eq!(xlogy(0.0, 0.0), 0.0);
        assert_eq!(xlogy(0.0, 5.0), 0.0);
        close(xlogy(2.0, 3.0), 2.0 * 3.0_f64.ln(), 1e-12);
    }

    #[test]
    fn erfcx_matches_definition_near_origin() {
        assert_eq!(erfcx(0.0), 1.0);
        for &x in &[0.25, 0.5, 1.0, 2.0, 3.0] {
            close(erfcx(x) * (-x * x).exp(), erfc(x), 1e-9);
        }
    }

    #[test]
    fn erfcx_tail_is_finite_and_asymptotic() {
        for &x in &[10.0, 20.0, 50.0, 100.0] {
            let v = erfcx(x);
            assert!(v.is_finite() && v > 0.0);
            close(x * PI.sqrt() * v, 1.0, 5e-2 / x);
        }
        close(erfcx(3.999), erfcx(4.001), 1e-3);
    }
}
