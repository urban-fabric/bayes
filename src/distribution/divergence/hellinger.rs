//! **Squared Hellinger distance** `H²(p, q) = 1 − BC`, a bounded symmetric
//! divergence whose square root is a metric. Closed-form per family via the
//! Bhattacharyya coefficient `BC = ∫√(pq)`.

use super::families::{gamma_bc, gaussian_bc};
use crate::distribution::{
    Bernoulli, Beta, Binomial, ChiSquared, DiscreteUniform, Exponential, Gamma, HalfNormal,
    InverseGamma, LogNormal, NegativeBinomial, Normal, Poisson, Uniform, VonMises, Wald,
};
use crate::function::bessel::ln_i0;
use crate::function::beta::ln_beta;

/// `H² = 1 − BC` clamped to `[0, ∞)` — the `max` only absorbs `O(ε)` rounding,
/// since `BC ≤ 1` is proved (`bhattacharyya_le_one`).
fn hellinger_sq_from_bc(bc: f64) -> f64 {
    (1.0 - bc).max(0.0)
}

/// The **squared Hellinger distance** `H²(p, q) = ½ ∫ (√p − √q)² = 1 − BC`, where
/// `BC = ∫ √(pq)` is the Bhattacharyya coefficient. A *bounded* divergence:
/// `0 ≤ H² ≤ 1`, symmetric, zero iff `p = q`, and `√H²` is a metric. Closed-form
/// per family (no Monte-Carlo).
///
/// Proved in `riemann` (`RiemannLean/Bayes/Divergence.lean`): `hellingerSq_nonneg`,
/// `hellingerSq_symm`, `hellingerSq_le_one`, and the comparisons
/// `hellingerSq_le_tvDist` (`H² ≤ TV`) and `hellingerSq_le_half_klDiv` (`H² ≤ ½ KL`).
pub trait HellingerDistance {
    /// The squared Hellinger distance `H² ∈ [0, 1]`.
    fn hellinger_sq(&self, other: &Self) -> f64;
    /// The Hellinger distance `H = √(H²) ∈ [0, 1]` — a metric.
    fn hellinger(&self, other: &Self) -> f64 {
        self.hellinger_sq(other).sqrt()
    }
}

impl HellingerDistance for Normal {
    fn hellinger_sq(&self, other: &Self) -> f64 {
        let (m1, s1) = self.params();
        let (m2, s2) = other.params();
        hellinger_sq_from_bc(gaussian_bc(m1, s1, m2, s2))
    }
}

impl HellingerDistance for Exponential {
    fn hellinger_sq(&self, other: &Self) -> f64 {
        let (l1, l2) = (self.rate(), other.rate());
        hellinger_sq_from_bc(2.0 * (l1 * l2).sqrt() / (l1 + l2))
    }
}

impl HellingerDistance for Gamma {
    fn hellinger_sq(&self, other: &Self) -> f64 {
        hellinger_sq_from_bc(gamma_bc(
            self.shape(),
            self.rate(),
            other.shape(),
            other.rate(),
        ))
    }
}

impl HellingerDistance for Beta {
    fn hellinger_sq(&self, other: &Self) -> f64 {
        let (a1, b1) = (self.alpha(), self.beta());
        let (a2, b2) = (other.alpha(), other.beta());
        let ln_bc =
            ln_beta(0.5 * (a1 + a2), 0.5 * (b1 + b2)) - 0.5 * (ln_beta(a1, b1) + ln_beta(a2, b2));
        hellinger_sq_from_bc(ln_bc.exp())
    }
}

impl HellingerDistance for Poisson {
    fn hellinger_sq(&self, other: &Self) -> f64 {
        let (l1, l2) = (self.lambda(), other.lambda());
        hellinger_sq_from_bc((-0.5 * (l1.sqrt() - l2.sqrt()).powi(2)).exp())
    }
}

impl HellingerDistance for Bernoulli {
    fn hellinger_sq(&self, other: &Self) -> f64 {
        let (p, q) = (self.p(), other.p());
        hellinger_sq_from_bc((p * q).sqrt() + ((1.0 - p) * (1.0 - q)).sqrt())
    }
}

impl HellingerDistance for LogNormal {
    fn hellinger_sq(&self, other: &Self) -> f64 {
        // H² is invariant under the monotone `exp` map, so this is the Gaussian
        // affinity on (μ, σ).
        hellinger_sq_from_bc(gaussian_bc(
            self.mu(),
            self.sigma(),
            other.mu(),
            other.sigma(),
        ))
    }
}

impl HellingerDistance for HalfNormal {
    fn hellinger_sq(&self, other: &Self) -> f64 {
        hellinger_sq_from_bc(gaussian_bc(0.0, self.scale(), 0.0, other.scale()))
    }
}

impl HellingerDistance for ChiSquared {
    fn hellinger_sq(&self, other: &Self) -> f64 {
        hellinger_sq_from_bc(gamma_bc(
            0.5 * self.freedom(),
            0.5,
            0.5 * other.freedom(),
            0.5,
        ))
    }
}

impl HellingerDistance for InverseGamma {
    fn hellinger_sq(&self, other: &Self) -> f64 {
        hellinger_sq_from_bc(gamma_bc(
            self.shape(),
            self.scale(),
            other.shape(),
            other.scale(),
        ))
    }
}

impl HellingerDistance for Wald {
    fn hellinger_sq(&self, other: &Self) -> f64 {
        // BC = (λ₁λ₂)^¼ √(2/(λ₁+λ₂)) · exp[½(λ₁/μ₁+λ₂/μ₂) − ½√((λ₁/μ₁²+λ₂/μ₂²)(λ₁+λ₂))].
        let (m1, l1) = (self.mu(), self.shape());
        let (m2, l2) = (other.mu(), other.shape());
        let s = ((l1 / (m1 * m1) + l2 / (m2 * m2)) * (l1 + l2)).sqrt();
        let ln_bc =
            0.25 * (l1 * l2).ln() + 0.5 * (2.0 / (l1 + l2)).ln() + 0.5 * (l1 / m1 + l2 / m2)
                - 0.5 * s;
        hellinger_sq_from_bc(ln_bc.exp())
    }
}

impl HellingerDistance for VonMises {
    fn hellinger_sq(&self, other: &Self) -> f64 {
        // BC = I₀(R) / √(I₀(κ₁)I₀(κ₂)), R = ½√(κ₁²+κ₂²+2κ₁κ₂cos Δμ).
        let (mu1, k1) = (self.mu(), self.kappa());
        let (mu2, k2) = (other.mu(), other.kappa());
        let r = 0.5 * (k1 * k1 + k2 * k2 + 2.0 * k1 * k2 * (mu1 - mu2).cos()).sqrt();
        let ln_bc = ln_i0(r) - 0.5 * (ln_i0(k1) + ln_i0(k2));
        hellinger_sq_from_bc(ln_bc.exp())
    }
}

impl HellingerDistance for Uniform {
    fn hellinger_sq(&self, other: &Self) -> f64 {
        // BC = |overlap| / √((b₁−a₁)(b₂−a₂)).
        let (a1, b1) = (self.lo(), self.hi());
        let (a2, b2) = (other.lo(), other.hi());
        let overlap = (b1.min(b2) - a1.max(a2)).max(0.0);
        hellinger_sq_from_bc(overlap / ((b1 - a1) * (b2 - a2)).sqrt())
    }
}

impl HellingerDistance for DiscreteUniform {
    fn hellinger_sq(&self, other: &Self) -> f64 {
        // BC = |overlap| / √(n₁ n₂).
        let (a1, b1) = (self.lo(), self.hi());
        let (a2, b2) = (other.lo(), other.hi());
        let n1 = (b1 - a1 + 1) as f64;
        let n2 = (b2 - a2 + 1) as f64;
        let overlap = (b1.min(b2) - a1.max(a2) + 1).max(0) as f64;
        hellinger_sq_from_bc(overlap / (n1 * n2).sqrt())
    }
}

impl HellingerDistance for Binomial {
    fn hellinger_sq(&self, other: &Self) -> f64 {
        // BC factorizes over the n iid Bernoulli trials: BC = BC_Bernoulli^n.
        // Differing trial counts are treated as out-of-family (H² = 1).
        if self.trials() != other.trials() {
            return 1.0;
        }
        let (p, q) = (self.p(), other.p());
        let bc = (p * q).sqrt() + ((1.0 - p) * (1.0 - q)).sqrt();
        hellinger_sq_from_bc(bc.powf(self.trials() as f64))
    }
}

impl HellingerDistance for NegativeBinomial {
    fn hellinger_sq(&self, other: &Self) -> f64 {
        // BC = [√(p₁p₂) / (1 − √((1−p₁)(1−p₂)))]^r, for matched successes `r`.
        if self.successes() != other.successes() {
            return 1.0;
        }
        let (p1, p2) = (self.p(), other.p());
        let bc = (p1 * p2).sqrt() / (1.0 - ((1.0 - p1) * (1.0 - p2)).sqrt());
        hellinger_sq_from_bc(bc.powf(self.successes()))
    }
}
