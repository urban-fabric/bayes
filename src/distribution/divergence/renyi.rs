//! **Rényi divergence** of order `α` — the one-parameter family interpolating
//! KL (`α→1`), Hellinger (`α=½`), and χ² (`α=2`).

use super::KlDivergence;
use super::kl_term;
use crate::distribution::{Bernoulli, Categorical, LogNormal, Normal};

/// The **Rényi divergence** of order `α`,
/// `D_α(p ‖ q) = 1/(α−1) · ln Σ pᵢ^α qᵢ^{1−α}` — the one-parameter family that
/// interpolates the others: `α → 1` is KL (returned exactly at `α = 1`),
/// `α = ½` is `−2 ln BC = −2 ln(1 − H²)` (Hellinger), and `α = 2` is
/// `ln(1 + χ²)`. Monotone non-decreasing in `α`. Implemented on the
/// finite-support discretes where it is an exact finite sum.
pub trait RenyiDivergence {
    /// `D_α(self ‖ other)`, in nats. `α > 0`; `α = 1` returns the KL limit.
    fn renyi_divergence(&self, other: &Self, alpha: f64) -> f64;
}

impl RenyiDivergence for Bernoulli {
    fn renyi_divergence(&self, other: &Self, alpha: f64) -> f64 {
        let (p, q) = (self.p(), other.p());
        if (alpha - 1.0).abs() < 1e-12 {
            return kl_term(p, q) + kl_term(1.0 - p, 1.0 - q);
        }
        let s = p.powf(alpha) * q.powf(1.0 - alpha)
            + (1.0 - p).powf(alpha) * (1.0 - q).powf(1.0 - alpha);
        s.ln() / (alpha - 1.0)
    }
}

impl RenyiDivergence for Categorical {
    /// Categories present in only one distribution are treated as `0` in the other.
    fn renyi_divergence(&self, other: &Self, alpha: f64) -> f64 {
        let (pa, pb) = (self.probabilities(), other.probabilities());
        let n = pa.len().max(pb.len());
        let at = |i: usize, v: &[f64]| v.get(i).copied().unwrap_or(0.0);
        if (alpha - 1.0).abs() < 1e-12 {
            return (0..n).map(|i| kl_term(at(i, pa), at(i, pb))).sum();
        }
        let s: f64 = (0..n)
            .map(|i| {
                let (a, b) = (at(i, pa), at(i, pb));
                if a > 0.0 {
                    a.powf(alpha) * b.powf(1.0 - alpha)
                } else {
                    0.0
                }
            })
            .sum();
        s.ln() / (alpha - 1.0)
    }
}

/// The closed-form Rényi divergence `D_α(N(μ₀,σ₀) ‖ N(μ₁,σ₁))`, shared by the
/// [`Normal`] and (via the log reparametrization) [`LogNormal`] impls.
///
/// `D_α = α(μ₀−μ₁)²/(2σ_α²) + [(1−α)ln σ₀ + α ln σ₁ − ln σ_α]/(α−1)` with
/// `σ_α² = α σ₁² + (1−α) σ₀²` (from `∫ pᵅq¹⁻ᵅ`, a Gaussian integral). Finite only
/// when `σ_α² > 0`: always for `α ∈ (0,1)`, and otherwise when the heavier-tailed
/// power stays integrable. `α → 1` is the KL limit (passed through directly).
fn normal_renyi(m0: f64, s0: f64, m1: f64, s1: f64, alpha: f64) -> f64 {
    let var_alpha = alpha * s1 * s1 + (1.0 - alpha) * s0 * s0;
    if var_alpha <= 0.0 {
        return f64::INFINITY;
    }
    let mean_term = alpha * (m0 - m1).powi(2) / (2.0 * var_alpha);
    let log_term =
        ((1.0 - alpha) * s0.ln() + alpha * s1.ln() - 0.5 * var_alpha.ln()) / (alpha - 1.0);
    mean_term + log_term
}

impl RenyiDivergence for Normal {
    /// The closed-form Gaussian Rényi divergence
    /// `D_α = α(μ₀−μ₁)²/(2σ_α²) + [(1−α)ln σ₀ + α ln σ₁ − ln σ_α]/(α−1)`,
    /// `σ_α² = α σ₁² + (1−α) σ₀²`; `α = 1` returns the KL limit, and `+∞` when the
    /// order-`α` integral diverges (`σ_α² ≤ 0`).
    fn renyi_divergence(&self, other: &Self, alpha: f64) -> f64 {
        if (alpha - 1.0).abs() < 1e-12 {
            return self.kl_divergence(other);
        }
        let (m0, s0) = self.params();
        let (m1, s1) = other.params();
        normal_renyi(m0, s0, m1, s1, alpha)
    }
}

impl RenyiDivergence for LogNormal {
    /// A log-normal is a normal in log-space and Rényi divergence is invariant
    /// under that monotone reparametrization, so this is the [`Normal`] formula
    /// on `(μ, σ)`.
    fn renyi_divergence(&self, other: &Self, alpha: f64) -> f64 {
        if (alpha - 1.0).abs() < 1e-12 {
            return self.kl_divergence(other);
        }
        normal_renyi(self.mu(), self.sigma(), other.mu(), other.sigma(), alpha)
    }
}
