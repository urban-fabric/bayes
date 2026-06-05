//! The beta function family: `ln_beta` / `beta`, and the regularized incomplete
//! beta `beta_reg` `Iₓ(a, b)` (the CDF kernel for the Beta, Student's-t, and F
//! distributions — Phase 1). Continued fraction from Numerical Recipes.

use super::gamma::ln_gamma;

/// `ln B(a, b) = ln Γ(a) + ln Γ(b) - ln Γ(a+b)`.
pub fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// The beta function `B(a, b)`.
pub fn beta(a: f64, b: f64) -> f64 {
    ln_beta(a, b).exp()
}

/// The regularized incomplete beta `Iₓ(a, b)`, `x ∈ [0, 1]`, `a, b > 0`.
pub fn beta_reg(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let front = (a * x.ln() + b * (1.0 - x).ln() - ln_beta(a, b)).exp();
    // Use the continued fraction; reflect for faster convergence when x is large.
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_cf(a, b, x) / a
    } else {
        1.0 - front * beta_cf(b, a, 1.0 - x) / b
    }
}

const MAX_ITER: usize = 300;
const EPS: f64 = 1e-15;

/// Lentz continued fraction for the incomplete beta (`betacf` in Numerical Recipes).
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let tiny = 1e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..MAX_ITER {
        let m = m as f64;
        let m2 = 2.0 * m;
        // Even step.
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        h *= d * c;
        // Odd step.
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
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
    h
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn beta_known_values() {
        // B(1,1) = 1, B(2,3) = 1/12.
        close(beta(1.0, 1.0), 1.0, 1e-12);
        close(beta(2.0, 3.0), 1.0 / 12.0, 1e-10);
    }

    #[test]
    fn beta_reg_endpoints_and_symmetry() {
        assert_eq!(beta_reg(2.0, 3.0, 0.0), 0.0);
        assert_eq!(beta_reg(2.0, 3.0, 1.0), 1.0);
        // I_x(a,b) = 1 - I_{1-x}(b,a).
        for &(a, b, x) in &[(2.0, 3.0, 0.4), (0.5, 0.5, 0.3), (5.0, 2.0, 0.7)] {
            close(beta_reg(a, b, x), 1.0 - beta_reg(b, a, 1.0 - x), 1e-10);
        }
        // I_x(1,1) = x (uniform CDF).
        close(beta_reg(1.0, 1.0, 0.37), 0.37, 1e-12);
    }
}
