//! Multivariate special functions used by matrix and simplex distributions.

use std::f64::consts::PI;

use super::gamma::ln_gamma;

/// Natural log of the multivariate gamma function
/// `Γ_p(a) = π^{p(p-1)/4} ∏_{j=1}^p Γ(a + (1-j)/2)`.
///
/// Returns `NaN` unless `p > 0` and `a > (p-1)/2`.
pub fn ln_multivariate_gamma(p: usize, a: f64) -> f64 {
    if p == 0 || !a.is_finite() || a <= (p as f64 - 1.0) / 2.0 {
        return f64::NAN;
    }
    let pf = p as f64;
    let log_pi = pf * (pf - 1.0) * PI.ln() / 4.0;
    log_pi
        + (1..=p)
            .map(|j| ln_gamma(a + (1.0 - j as f64) / 2.0))
            .sum::<f64>()
}

/// Multivariate gamma function `Γ_p(a)`.
pub fn multivariate_gamma(p: usize, a: f64) -> f64 {
    ln_multivariate_gamma(p, a).exp()
}

/// Natural log of the multivariate beta function
/// `B(α) = ∏ Γ(αᵢ) / Γ(Σ αᵢ)`.
///
/// Returns `NaN` for an empty vector, non-finite parameters, or any `αᵢ <= 0`.
pub fn ln_multivariate_beta(alpha: &[f64]) -> f64 {
    if alpha.is_empty() || alpha.iter().any(|&a| !a.is_finite() || a <= 0.0) {
        return f64::NAN;
    }
    let sum = alpha.iter().sum::<f64>();
    alpha.iter().map(|&a| ln_gamma(a)).sum::<f64>() - ln_gamma(sum)
}

/// Multivariate beta function `B(α)`.
pub fn multivariate_beta(alpha: &[f64]) -> f64 {
    ln_multivariate_beta(alpha).exp()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn multivariate_gamma_reduces_to_gamma_at_dimension_one() {
        close(ln_multivariate_gamma(1, 3.5), ln_gamma(3.5), 1e-12);
    }

    #[test]
    fn multivariate_beta_matches_dirichlet_normalizer() {
        close(multivariate_beta(&[1.0, 1.0, 1.0]), 0.5, 1e-12);
        assert!(ln_multivariate_beta(&[1.0, 0.0]).is_nan());
    }
}
