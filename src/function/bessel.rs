//! Modified **Bessel functions of the first kind**, `I₀` and `I₁`, the stable ratio
//! `I₁/I₀`, and a stable `ln I₀`. These are the normalising constants and moments of
//! directional distributions (the von Mises density is `e^{κ cos θ} / (2π I₀(κ))`,
//! and its circular variance is `1 − I₁(κ)/I₀(κ)`). Implemented with the
//! Abramowitz & Stegun polynomial approximations (≈ 1e-7 relative accuracy).

use std::f64::consts::PI;

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
        close(i0(-3.0), i0(3.0), 1e-12); // even
    }

    #[test]
    fn i1_known_values() {
        close(i1(0.0), 0.0, 1e-12);
        close(i1(1.0), 0.565_159_103_99, 1e-7);
        close(i1(2.0), 1.590_636_854_64, 1e-7);
        close(i1(5.0), 24.335_642_142_1, 1e-6);
        close(i1(-2.0), -i1(2.0), 1e-12); // odd
    }

    #[test]
    fn i1_over_i0_matches_and_is_stable() {
        for &x in &[0.5, 2.0, 3.0, 6.0] {
            close(i1_over_i0(x), i1(x) / i0(x), 1e-9);
        }
        // Large argument: the separate ratio overflows (∞/∞ = NaN), this stays valid
        // and → 1⁻.
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
}
