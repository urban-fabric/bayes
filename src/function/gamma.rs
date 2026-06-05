//! The gamma function family: `ln_gamma` / `gamma` (Lanczos), the regularized
//! lower/upper incomplete gamma `gamma_lr` / `gamma_ur` (used by the Poisson, Gamma,
//! and chi-squared CDFs), and `digamma`.

use std::f64::consts::PI;

/// Lanczos `g` parameter and coefficients (g = 7, n = 9) — ~15-digit accuracy.
const G: f64 = 7.0;
const LANCZOS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

/// Natural log of the gamma function, `ln Γ(x)`, for `x > 0` (and, via reflection,
/// `x < 0` away from the non-positive integers).
pub fn ln_gamma(x: f64) -> f64 {
    if x < 0.5 {
        // Reflection: Γ(x)Γ(1-x) = π / sin(πx).
        (PI / (PI * x).sin()).ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = LANCZOS[0];
        let t = x + G + 0.5;
        for (i, &c) in LANCZOS.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// The gamma function `Γ(x)`.
pub fn gamma(x: f64) -> f64 {
    if x < 0.5 {
        PI / ((PI * x).sin() * gamma(1.0 - x))
    } else {
        let x = x - 1.0;
        let mut a = LANCZOS[0];
        let t = x + G + 0.5;
        for (i, &c) in LANCZOS.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        (2.0 * PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * a
    }
}

const MAX_ITER: usize = 200;
const EPS: f64 = 1e-15;

/// Regularized **lower** incomplete gamma `P(a, x) = γ(a, x) / Γ(a)`, `a, x ≥ 0`.
pub fn gamma_lr(a: f64, x: f64) -> f64 {
    if a <= 0.0 || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        // Series expansion (converges fast for x < a+1).
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut del = sum;
        for _ in 0..MAX_ITER {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * EPS {
                break;
            }
        }
        (sum.ln() - x + a * x.ln() - ln_gamma(a)).exp()
    } else {
        // Upper continued fraction → P = 1 - Q.
        1.0 - gamma_ur(a, x)
    }
}

/// Regularized **upper** incomplete gamma `Q(a, x) = Γ(a, x) / Γ(a) = 1 - P(a, x)`.
pub fn gamma_ur(a: f64, x: f64) -> f64 {
    if a <= 0.0 || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        return 1.0 - gamma_lr(a, x);
    }
    // Lentz's continued fraction for the upper incomplete gamma.
    let tiny = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..MAX_ITER {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < tiny {
            d = tiny;
        }
        c = b + an / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

/// The digamma function `ψ(x) = d/dx ln Γ(x)`.
pub fn digamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    // Recurrence up to x ≥ 10, then the Bernoulli asymptotic series.
    while x < 10.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result + x.ln() - 0.5 * inv - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))
}

/// The trigamma function `ψ₁(x) = d²/dx² ln Γ(x)` — the derivative of [`digamma`].
/// Used in exponential-family / Dirichlet parameter estimation and several
/// distribution entropies.
pub fn trigamma(x: f64) -> f64 {
    // Poles at the non-positive integers.
    if x <= 0.0 && x == x.floor() {
        return f64::INFINITY;
    }
    if x < 0.5 {
        // Reflection: ψ₁(x) = π²/sin²(πx) − ψ₁(1 − x).
        let s = (PI * x).sin();
        return PI * PI / (s * s) - trigamma(1.0 - x);
    }
    // Recurrence ψ₁(x) = ψ₁(x+1) + 1/x²; push x to ≥ 10, then the asymptotic series
    // 1/x + 1/(2x²) + 1/(6x³) − 1/(30x⁵) + 1/(42x⁷) − 1/(30x⁹).
    let mut x = x;
    let mut result = 0.0;
    while x < 10.0 {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    let r = 1.0 / x;
    let r2 = r * r;
    result + r + 0.5 * r2 + r2 * r * (1.0 / 6.0 - r2 * (1.0 / 30.0 - r2 * (1.0 / 42.0 - r2 / 30.0)))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn gamma_matches_factorials() {
        close(gamma(1.0), 1.0, 1e-12);
        close(gamma(5.0), 24.0, 1e-10);
        close(gamma(6.0), 120.0, 1e-10);
        close(gamma(0.5), std::f64::consts::PI.sqrt(), 1e-12); // Γ(1/2) = √π
    }

    #[test]
    fn ln_gamma_matches_gamma() {
        for &x in &[0.3, 0.7, 1.5, 3.2, 10.0, 50.0] {
            close(ln_gamma(x).exp(), gamma(x), 1e-9);
        }
    }

    #[test]
    fn regularized_incomplete_gamma_bounds_and_complement() {
        for &(a, x) in &[(1.0, 0.5), (2.5, 3.0), (5.0, 2.0), (10.0, 12.0)] {
            let p = gamma_lr(a, x);
            let q = gamma_ur(a, x);
            assert!((0.0..=1.0).contains(&p), "P out of range: {p}");
            close(p + q, 1.0, 1e-10);
        }
        // P(1, x) = 1 - e^{-x} (the exponential CDF).
        close(gamma_lr(1.0, 2.0), 1.0 - (-2.0f64).exp(), 1e-10);
    }

    #[test]
    fn digamma_known_values() {
        close(digamma(1.0), -0.577_215_664_901_532_9, 1e-9); // ψ(1) = -γ
        close(digamma(2.0), 1.0 - 0.577_215_664_901_532_9, 1e-9);
    }

    #[test]
    fn trigamma_known_values() {
        let pi2 = std::f64::consts::PI * std::f64::consts::PI;
        close(trigamma(1.0), pi2 / 6.0, 1e-10); // ψ₁(1) = π²/6
        close(trigamma(2.0), pi2 / 6.0 - 1.0, 1e-10); // ψ₁(2) = π²/6 − 1
        close(trigamma(0.5), pi2 / 2.0, 1e-10); // ψ₁(½) = π²/2
        // Recurrence ψ₁(x) − ψ₁(x+1) = 1/x².
        close(trigamma(3.3) - trigamma(4.3), 1.0 / (3.3 * 3.3), 1e-10);
    }
}
