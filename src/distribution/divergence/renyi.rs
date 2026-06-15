//! **Rényi divergence** of order `α` — the one-parameter family interpolating
//! KL (`α→1`), Hellinger (`α=½`), and χ² (`α=2`).

use super::KlDivergence;
use super::families::{gamma_renyi, gaussian_renyi};
use super::kl_term;
use crate::distribution::{
    Bernoulli, Binomial, Categorical, ChiSquared, DiscreteUniform, Gamma, HalfNormal, InverseGamma,
    LogNormal, NegativeBinomial, Normal, Uniform, VonMises, Wald,
};
use crate::function::bessel::ln_i0;

/// Tolerance for snapping `α` to the KL limit (`α = 1`).
const ALPHA_KL: f64 = 1e-12;

/// The **Rényi divergence** of order `α`,
/// `D_α(p ‖ q) = 1/(α−1) · ln Σ pᵢ^α qᵢ^{1−α}` — the one-parameter family that
/// interpolates the others: `α → 1` is KL (returned exactly at `α = 1`),
/// `α = ½` is `−2 ln BC = −2 ln(1 − H²)` (Hellinger), and `α = 2` is
/// `ln(1 + χ²)`. Monotone non-decreasing in `α`. Implemented in closed form
/// wherever `∫ pᵅ q¹⁻ᵅ` is analytic — the Gaussian, gamma, directional,
/// inverse-Gaussian, and uniform families, plus the finite-support discretes.
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

impl RenyiDivergence for Normal {
    /// The closed-form Gaussian Rényi divergence
    /// `D_α = α(μ₀−μ₁)²/(2σ_α²) + [(1−α)ln σ₀ + α ln σ₁ − ln σ_α]/(α−1)`,
    /// `σ_α² = α σ₁² + (1−α) σ₀²`; `α = 1` returns the KL limit, and `+∞` when the
    /// order-`α` integral diverges (`σ_α² ≤ 0`).
    fn renyi_divergence(&self, other: &Self, alpha: f64) -> f64 {
        if (alpha - 1.0).abs() < ALPHA_KL {
            return self.kl_divergence(other);
        }
        let (m0, s0) = self.params();
        let (m1, s1) = other.params();
        gaussian_renyi(m0, s0, m1, s1, alpha)
    }
}

impl RenyiDivergence for LogNormal {
    /// A log-normal is a normal in log-space and Rényi divergence is invariant
    /// under that monotone reparametrization, so this is the [`Normal`] formula
    /// on `(μ, σ)`.
    fn renyi_divergence(&self, other: &Self, alpha: f64) -> f64 {
        if (alpha - 1.0).abs() < ALPHA_KL {
            return self.kl_divergence(other);
        }
        gaussian_renyi(self.mu(), self.sigma(), other.mu(), other.sigma(), alpha)
    }
}

impl RenyiDivergence for HalfNormal {
    fn renyi_divergence(&self, other: &Self, alpha: f64) -> f64 {
        if (alpha - 1.0).abs() < ALPHA_KL {
            return self.kl_divergence(other);
        }
        gaussian_renyi(0.0, self.scale(), 0.0, other.scale(), alpha)
    }
}

impl RenyiDivergence for Gamma {
    fn renyi_divergence(&self, other: &Self, alpha: f64) -> f64 {
        if (alpha - 1.0).abs() < ALPHA_KL {
            return self.kl_divergence(other);
        }
        gamma_renyi(self.shape(), self.rate(), other.shape(), other.rate(), alpha)
    }
}

impl RenyiDivergence for ChiSquared {
    fn renyi_divergence(&self, other: &Self, alpha: f64) -> f64 {
        if (alpha - 1.0).abs() < ALPHA_KL {
            return self.kl_divergence(other);
        }
        gamma_renyi(0.5 * self.freedom(), 0.5, 0.5 * other.freedom(), 0.5, alpha)
    }
}

impl RenyiDivergence for InverseGamma {
    fn renyi_divergence(&self, other: &Self, alpha: f64) -> f64 {
        if (alpha - 1.0).abs() < ALPHA_KL {
            return self.kl_divergence(other);
        }
        gamma_renyi(self.shape(), self.scale(), other.shape(), other.scale(), alpha)
    }
}

impl RenyiDivergence for VonMises {
    /// `D_α = [ln I₀(R_α) − α ln I₀(κ₁) − (1−α) ln I₀(κ₂)] / (α−1)`, with
    /// `R_α = |α κ₁∠μ₁ + (1−α) κ₂∠μ₂|`. Always finite (compact support).
    fn renyi_divergence(&self, other: &Self, alpha: f64) -> f64 {
        if (alpha - 1.0).abs() < ALPHA_KL {
            return self.kl_divergence(other);
        }
        let (mu1, k1) = (self.mu(), self.kappa());
        let (mu2, k2) = (other.mu(), other.kappa());
        let r_alpha = ((alpha * k1).powi(2)
            + ((1.0 - alpha) * k2).powi(2)
            + 2.0 * alpha * (1.0 - alpha) * k1 * k2 * (mu1 - mu2).cos())
        .sqrt();
        (ln_i0(r_alpha) - alpha * ln_i0(k1) - (1.0 - alpha) * ln_i0(k2)) / (alpha - 1.0)
    }
}

impl RenyiDivergence for Wald {
    /// Inverse-Gaussian Rényi divergence via `∫ x^{−3/2} e^{−Ax−B/x}`. `+∞` when
    /// the order-`α` integral diverges (`α λ₁ + (1−α) λ₂ ≤ 0` or the `x`-coefficient
    /// turns non-positive).
    fn renyi_divergence(&self, other: &Self, alpha: f64) -> f64 {
        if (alpha - 1.0).abs() < ALPHA_KL {
            return self.kl_divergence(other);
        }
        let (m1, l1) = (self.mu(), self.shape());
        let (m2, l2) = (other.mu(), other.shape());
        let la = alpha * l1 + (1.0 - alpha) * l2;
        let aa = alpha * l1 / (m1 * m1) + (1.0 - alpha) * l2 / (m2 * m2);
        if la <= 0.0 || aa <= 0.0 {
            return f64::INFINITY;
        }
        let ln_int = 0.5 * (alpha * l1.ln() + (1.0 - alpha) * l2.ln())
            + alpha * l1 / m1
            + (1.0 - alpha) * l2 / m2
            - 0.5 * la.ln()
            - (aa * la).sqrt();
        ln_int / (alpha - 1.0)
    }
}

impl RenyiDivergence for Uniform {
    /// `D_α = ln(|overlap| · w₁^{−α} w₂^{−(1−α)}) / (α−1)`. For `α > 1`, `+∞` unless
    /// `[a₁,b₁] ⊆ [a₂,b₂]`; disjoint supports give `+∞` at every order.
    fn renyi_divergence(&self, other: &Self, alpha: f64) -> f64 {
        if (alpha - 1.0).abs() < ALPHA_KL {
            return self.kl_divergence(other);
        }
        let (a1, b1) = (self.lo(), self.hi());
        let (a2, b2) = (other.lo(), other.hi());
        let contained = a1 >= a2 && b1 <= b2;
        if alpha > 1.0 && !contained {
            return f64::INFINITY;
        }
        let overlap = (b1.min(b2) - a1.max(a2)).max(0.0);
        if overlap <= 0.0 {
            return f64::INFINITY;
        }
        let s = overlap * (b1 - a1).powf(-alpha) * (b2 - a2).powf(-(1.0 - alpha));
        s.ln() / (alpha - 1.0)
    }
}

impl RenyiDivergence for DiscreteUniform {
    fn renyi_divergence(&self, other: &Self, alpha: f64) -> f64 {
        if (alpha - 1.0).abs() < ALPHA_KL {
            return self.kl_divergence(other);
        }
        let (a1, b1) = (self.lo(), self.hi());
        let (a2, b2) = (other.lo(), other.hi());
        let contained = a1 >= a2 && b1 <= b2;
        if alpha > 1.0 && !contained {
            return f64::INFINITY;
        }
        let overlap = (b1.min(b2) - a1.max(a2) + 1).max(0) as f64;
        if overlap <= 0.0 {
            return f64::INFINITY;
        }
        let n1 = (b1 - a1 + 1) as f64;
        let n2 = (b2 - a2 + 1) as f64;
        let s = overlap * n1.powf(-alpha) * n2.powf(-(1.0 - alpha));
        s.ln() / (alpha - 1.0)
    }
}

impl RenyiDivergence for Binomial {
    /// `n ×` the per-trial Bernoulli Rényi divergence, for matched trial counts;
    /// differing counts are treated as out-of-family (`+∞`).
    fn renyi_divergence(&self, other: &Self, alpha: f64) -> f64 {
        if (alpha - 1.0).abs() < ALPHA_KL {
            return self.kl_divergence(other);
        }
        if self.trials() != other.trials() {
            return f64::INFINITY;
        }
        let (p, q) = (self.p(), other.p());
        let s = p.powf(alpha) * q.powf(1.0 - alpha)
            + (1.0 - p).powf(alpha) * (1.0 - q).powf(1.0 - alpha);
        self.trials() as f64 * s.ln() / (alpha - 1.0)
    }
}

impl RenyiDivergence for NegativeBinomial {
    /// `D_α = r·[α ln p₁ + (1−α) ln p₂ − ln(1 − w)]/(α−1)`,
    /// `w = (1−p₁)^α (1−p₂)^{1−α}`, for matched successes `r`. `+∞` when `w ≥ 1`
    /// (the order-`α` series diverges) or `r` differs.
    fn renyi_divergence(&self, other: &Self, alpha: f64) -> f64 {
        if (alpha - 1.0).abs() < ALPHA_KL {
            return self.kl_divergence(other);
        }
        if self.successes() != other.successes() {
            return f64::INFINITY;
        }
        let r = self.successes();
        let (p1, p2) = (self.p(), other.p());
        let w = (1.0 - p1).powf(alpha) * (1.0 - p2).powf(1.0 - alpha);
        if w >= 1.0 {
            return f64::INFINITY;
        }
        r * (alpha * p1.ln() + (1.0 - alpha) * p2.ln() - (1.0 - w).ln()) / (alpha - 1.0)
    }
}
