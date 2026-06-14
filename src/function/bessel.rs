//! Modified **Bessel functions of the first kind**, `I₀` and `I₁`, the stable ratio
//! `I₁/I₀`, and a stable `ln I₀`. These are the normalising constants and moments of
//! directional distributions (the von Mises density is `e^{κ cos θ} / (2π I₀(κ))`,
//! and its circular variance is `1 − I₁(κ)/I₀(κ)`). Implemented with the
//! Abramowitz & Stegun polynomial approximations (≈ 1e-7 relative accuracy).

use std::f64::consts::PI;

use super::gamma::ln_gamma;

/// `I₀` large-argument asymptotic polynomial in `t = 3.75/x` (A&S 9.8.2); the shared
/// `eˣ/√x` factor is applied by the caller.
fn i0_asymp(t: f64) -> f64 {
    0.398_942_28
        + t * (0.013_285_92
            + t * (0.002_253_19
                + t * (-0.001_575_65
                    + t * (0.009_162_81
                        + t * (-0.020_577_06
                            + t * (0.026_355_37 + t * (-0.016_476_33 + t * 0.003_923_77)))))))
}

/// `I₁` large-argument asymptotic polynomial in `t = 3.75/x` (A&S 9.8.4).
fn i1_asymp(t: f64) -> f64 {
    0.398_942_28
        + t * (-0.039_880_24
            + t * (-0.003_620_18
                + t * (0.001_638_01
                    + t * (-0.010_315_55
                        + t * (0.022_829_67
                            + t * (-0.028_953_12 + t * (0.017_876_54 + t * (-0.004_200_59))))))))
}

/// The modified Bessel function `I₀(x)` (even in `x`).
pub fn i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let t = (x / 3.75) * (x / 3.75);
        1.0 + t
            * (3.515_622_9
                + t * (3.089_942_4
                    + t * (1.206_749_2 + t * (0.265_973_2 + t * (0.036_076_8 + t * 0.004_581_3)))))
    } else {
        (ax.exp() / ax.sqrt()) * i0_asymp(3.75 / ax)
    }
}

/// The modified Bessel function `I₁(x)` (odd in `x`).
pub fn i1(x: f64) -> f64 {
    let ax = x.abs();
    let result = if ax < 3.75 {
        let t = (x / 3.75) * (x / 3.75);
        ax * (0.5
            + t * (0.878_905_94
                + t * (0.514_988_69
                    + t * (0.150_849_34
                        + t * (0.026_587_33 + t * (0.003_015_32 + t * 0.000_324_11))))))
    } else {
        (ax.exp() / ax.sqrt()) * i1_asymp(3.75 / ax)
    };
    if x < 0.0 { -result } else { result }
}

/// The ratio `I₁(x)/I₀(x) ∈ [0, 1)` for `x ≥ 0` — the von Mises mean resultant
/// length. Stable for large `x`: the shared `eˣ/√x` growth factor cancels, so it
/// never overflows (unlike `i1(x)/i0(x)` computed separately).
pub fn i1_over_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        i1(ax) / i0(ax)
    } else {
        let t = 3.75 / ax;
        i1_asymp(t) / i0_asymp(t)
    }
}

/// `ln I₀(x)` for `x ≥ 0`, stable for large `x` where `I₀` itself overflows: uses
/// the asymptotic `ln I₀(x) ≈ x − ½ln(2πx) + ln(1 + 1/8x + 9/128x² + …)`.
pub fn ln_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 15.0 {
        i0(ax).ln()
    } else {
        let r = 1.0 / ax;
        let series = 1.0 + r * (0.125 + r * (0.070_312_5 + r * 0.073_242_19));
        ax - 0.5 * (2.0 * PI * ax).ln() + series.ln()
    }
}

/// Stable `ln(eᵃ + eᵇ)`.
fn ln_add_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    hi + (lo - hi).exp().ln_1p()
}

/// `ln I_ν(x)` — the **modified Bessel function of the first kind of real order**
/// `ν ≥ 0`, argument `x ≥ 0`, in log space.
///
/// Evaluated from the ascending series `I_ν(x) = Σ_m (x/2)^{2m+ν} / (m! Γ(ν+m+1))`
/// accumulated by **log-sum-exp** over its terms, so it never overflows and stays
/// accurate for *any* order and argument (unlike the `eˣ/√x` asymptotic, which fails
/// once `x` is no longer ≫ `ν²`). The term count grows with the series peak
/// `m* ≈ (√(x²+ν²) − ν)/2`, which is modest for realistic concentrations.
///
/// This is the general-order normaliser behind a hypersphere von Mises–Fisher density,
/// `C_d(κ) = κ^{d/2−1} / ((2π)^{d/2} I_{d/2−1}(κ))` (`ν = d/2 − 1`); for `ν = 1/2`
/// (the 2-sphere) it agrees with the closed form `I_{1/2}(x) = √(2/πx)·sinh x`.
///
/// Returns `0` for `x = 0, ν = 0` and `−∞` for `x = 0, ν > 0`.
#[must_use]
pub fn ln_iv(nu: f64, x: f64) -> f64 {
    debug_assert!(nu >= 0.0 && x >= 0.0, "ln_iv needs ν ≥ 0, x ≥ 0");
    if x == 0.0 {
        return if nu == 0.0 { 0.0 } else { f64::NEG_INFINITY };
    }
    let ln_half_x = (0.5 * x).ln();
    let ln_x2_4 = 2.0 * ln_half_x; // ln(x²/4) — the term-ratio numerator
    // Accumulate ln Σ aₘ with a₀ = 1 (ln a₀ = 0) and aₘ/aₘ₋₁ = (x²/4)/(m(ν+m)).
    let m_peak = 0.5 * ((x * x + nu * nu).sqrt() - nu);
    let mut ln_term = 0.0;
    let mut ln_sum = 0.0;
    let mut m: f64 = 1.0;
    loop {
        ln_term += ln_x2_4 - m.ln() - (nu + m).ln();
        ln_sum = ln_add_exp(ln_sum, ln_term);
        if m > m_peak && ln_term < ln_sum - 38.0 {
            break; // past the peak and contributing below f64 precision
        }
        m += 1.0;
        if m > 1.0e7 {
            break; // safety backstop (never reached for realistic κ)
        }
    }
    nu.mul_add(ln_half_x, ln_sum) - ln_gamma(nu + 1.0)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn i0_known_values() {
        close(i0(0.0), 1.0, 1e-12);
        close(i0(1.0), 1.266_065_877_75, 1e-7);
        close(i0(2.0), 2.279_585_302_34, 1e-7);
        close(i0(5.0), 27.239_871_823_6, 1e-6);
        close(i0(-3.0), i0(3.0), 1e-12);
    }

    #[test]
    fn i1_known_values() {
        close(i1(0.0), 0.0, 1e-12);
        close(i1(1.0), 0.565_159_103_99, 1e-7);
        close(i1(2.0), 1.590_636_854_64, 1e-7);
        close(i1(5.0), 24.335_642_142_1, 1e-6);
        close(i1(-2.0), -i1(2.0), 1e-12);
    }

    #[test]
    fn i1_over_i0_matches_and_is_stable() {
        for &x in &[0.5, 2.0, 3.0, 6.0] {
            close(i1_over_i0(x), i1(x) / i0(x), 1e-9);
        }
        let r = i1_over_i0(1000.0);
        assert!(r.is_finite() && r > 0.999 && r < 1.0);
    }

    #[test]
    fn ln_i0_matches_log_of_i0_and_is_stable() {
        for &x in &[0.5, 2.0, 10.0, 14.9] {
            close(ln_i0(x), i0(x).ln(), 1e-7);
        }
        let v = ln_i0(500.0);
        assert!(v.is_finite());
        close(v, 500.0 - 0.5 * (2.0 * PI * 500.0).ln(), 1e-2);
    }

    #[test]
    fn ln_iv_matches_integer_orders() {
        // The references (i0/i1 A&S polynomials) are themselves only ~1e-7 accurate, so
        // that bounds the agreement — ln_iv's series is the more accurate of the two.
        for &x in &[0.3, 1.0, 4.0, 20.0, 100.0] {
            close(ln_iv(0.0, x), ln_i0(x), 1e-6);
            close(ln_iv(1.0, x), i1(x).ln(), 1e-6);
        }
    }

    #[test]
    fn ln_iv_matches_half_integer_closed_forms() {
        // I_{1/2}(x) = √(2/πx)·sinh x ; I_{3/2}(x) = √(2/πx)·(cosh x − sinh x / x).
        for &x in &[0.25, 1.0, 3.0, 10.0, 60.0] {
            let pref = 0.5 * (2.0 / (PI * x)).ln();
            close(ln_iv(0.5, x), pref + x.sinh().ln(), 1e-7);
            close(ln_iv(1.5, x), pref + (x.cosh() - x.sinh() / x).ln(), 1e-7);
        }
    }

    #[test]
    fn ln_iv_is_stable_in_high_dimension() {
        // ν = d/2 − 1 for a 256-sphere with moderate κ: the eˣ/√x asymptotic would be
        // useless (x ≪ ν²), but the log-space series stays finite and ordered.
        let v = ln_iv(127.0, 50.0);
        assert!(v.is_finite());
        // Monotone in κ at fixed order, and in order at fixed κ (I_ν decreasing in ν).
        assert!(ln_iv(127.0, 60.0) > ln_iv(127.0, 50.0));
        assert!(ln_iv(127.0, 50.0) < ln_iv(63.0, 50.0));
    }

    #[test]
    fn ln_iv_handles_zero_argument() {
        assert_eq!(ln_iv(0.0, 0.0), 0.0);
        assert_eq!(ln_iv(2.5, 0.0), f64::NEG_INFINITY);
    }
}
