//! **χ² (Pearson) divergence** `χ²(p ‖ q) = ∫(p − q)²/q`, the f-divergence for
//! `f(t) = (t − 1)²` and an upper bound on KL.

use super::families::{gamma_chi_squared, gaussian_chi_squared};
use crate::{
    Bernoulli, Binomial, Categorical, ChiSquared, DiscreteUniform, Exponential, Gamma, HalfNormal,
    InverseGamma, LogNormal, NegativeBinomial, Normal, Poisson, Uniform, VonMises, Wald,
};
use euclid::Real;

/// The **χ² (Pearson) divergence** `χ²(p ‖ q) = ∫ (p − q)²/q = ∫ p²/q − 1`, in
/// `[0, ∞]`. The `f`-divergence for `f(t) = (t − 1)²`, and an upper bound on KL:
/// `D_KL(p ‖ q) ≤ χ²(p ‖ q)`. Implemented where it is a closed form (and, for
/// continuous families, where the integral converges — `χ²` can be `+∞` when the
/// second distribution has the lighter tail).
///
/// Proved in `riemann` (`RiemannLean/Bayes/Divergence.lean`): `chiSqDiv_nonneg`,
/// `klDiv_le_chiSqDiv`, and the umbrella `fDiv_nonneg` (of which both KL and χ²
/// are instances).
pub trait ChiSquaredDivergence {
    /// `χ²(self ‖ other) ∈ [0, ∞]` (`f64::INFINITY` when the integral diverges).
    fn chi_squared_divergence(&self, other: &Self) -> f64;
}

impl ChiSquaredDivergence for Bernoulli {
    fn chi_squared_divergence(&self, other: &Self) -> f64 {
        let (p, q) = (self.p(), other.p());
        // (p−q)²/q + (p−q)²/(1−q) = (p−q)² / (q(1−q)).
        (p - q).powi(2) / (q * (1.0 - q))
    }
}

impl ChiSquaredDivergence for Poisson {
    fn chi_squared_divergence(&self, other: &Self) -> f64 {
        let (l1, l2) = (self.lambda(), other.lambda());
        ((l1 - l2).powi(2) / l2).exp() - 1.0
    }
}

impl ChiSquaredDivergence for Exponential {
    fn chi_squared_divergence(&self, other: &Self) -> f64 {
        let (l1, l2) = (self.rate(), other.rate());
        // ∫ λ1²e^{−2λ1x}/(λ2 e^{−λ2x}) − 1 converges only when 2λ1 > λ2.
        if 2.0 * l1 > l2 {
            l1 * l1 / (l2 * (2.0 * l1 - l2)) - 1.0
        } else {
            f64::INFINITY
        }
    }
}

impl ChiSquaredDivergence for Categorical {
    /// Categories present in only one distribution are treated as `0` in the other;
    /// a category with `q = 0` but `p > 0` makes the divergence `+∞`.
    fn chi_squared_divergence(&self, other: &Self) -> f64 {
        let (pa, pb) = (self.probabilities(), other.probabilities());
        let n = pa.len().max(pb.len());
        (0..n)
            .map(|i| {
                let a = pa.get(i).copied().unwrap_or(0.0);
                let b = pb.get(i).copied().unwrap_or(0.0);
                if b > 0.0 {
                    (a - b).powi(2) / b
                } else if a > 0.0 {
                    f64::INFINITY
                } else {
                    0.0
                }
            })
            .sum()
    }
}

impl ChiSquaredDivergence for Normal {
    fn chi_squared_divergence(&self, other: &Self) -> f64 {
        let (m1, s1) = self.params();
        let (m2, s2) = other.params();
        gaussian_chi_squared(m1, s1, m2, s2)
    }
}

impl ChiSquaredDivergence for LogNormal {
    fn chi_squared_divergence(&self, other: &Self) -> f64 {
        // χ² is invariant under the monotone `exp` map.
        gaussian_chi_squared(self.mu(), self.sigma(), other.mu(), other.sigma())
    }
}

impl ChiSquaredDivergence for HalfNormal {
    fn chi_squared_divergence(&self, other: &Self) -> f64 {
        gaussian_chi_squared(0.0, self.scale(), 0.0, other.scale())
    }
}

impl ChiSquaredDivergence for Gamma {
    fn chi_squared_divergence(&self, other: &Self) -> f64 {
        gamma_chi_squared(self.shape(), self.rate(), other.shape(), other.rate())
    }
}

impl ChiSquaredDivergence for ChiSquared {
    fn chi_squared_divergence(&self, other: &Self) -> f64 {
        gamma_chi_squared(0.5 * self.freedom(), 0.5, 0.5 * other.freedom(), 0.5)
    }
}

impl ChiSquaredDivergence for InverseGamma {
    fn chi_squared_divergence(&self, other: &Self) -> f64 {
        gamma_chi_squared(self.shape(), self.scale(), other.shape(), other.scale())
    }
}

impl ChiSquaredDivergence for Wald {
    /// Inverse-Gaussian χ² (the `α = 2` member of the Rényi family). Finite only
    /// when `2λ₁ > λ₂` and `2λ₁/μ₁² > λ₂/μ₂²`; `+∞` otherwise.
    fn chi_squared_divergence(&self, other: &Self) -> f64 {
        let (m1, l1) = (self.mu(), self.shape());
        let (m2, l2) = (other.mu(), other.shape());
        let la = 2.0 * l1 - l2;
        let aa = 2.0 * l1 / (m1 * m1) - l2 / (m2 * m2);
        if la <= 0.0 || aa <= 0.0 {
            return f64::INFINITY;
        }
        let ln_int =
            l1.ln() - 0.5 * l2.ln() + 2.0 * l1 / m1 - l2 / m2 - 0.5 * la.ln() - (aa * la).sqrt();
        ln_int.exp() - 1.0
    }
}

impl ChiSquaredDivergence for VonMises {
    /// `χ² = I₀(R)·I₀(κ₂)/I₀(κ₁)² − 1`, `R = |2κ₁∠μ₁ − κ₂∠μ₂|`. Always finite.
    fn chi_squared_divergence(&self, other: &Self) -> f64 {
        let (mu1, k1) = (self.mu(), self.kappa());
        let (mu2, k2) = (other.mu(), other.kappa());
        let r = (4.0 * k1 * k1 + k2 * k2 - 4.0 * k1 * k2 * (mu1 - mu2).cos()).sqrt();
        (r.ln_bessel_i0() + k2.ln_bessel_i0() - 2.0 * k1.ln_bessel_i0()).exp() - 1.0
    }
}

impl ChiSquaredDivergence for Uniform {
    fn chi_squared_divergence(&self, other: &Self) -> f64 {
        // Finite only when `[a₁,b₁] ⊆ [a₂,b₂]`; then `(b₂−a₂)/(b₁−a₁) − 1`.
        let (a1, b1) = (self.lo(), self.hi());
        let (a2, b2) = (other.lo(), other.hi());
        if a1 < a2 || b1 > b2 {
            return f64::INFINITY;
        }
        (b2 - a2) / (b1 - a1) - 1.0
    }
}

impl ChiSquaredDivergence for DiscreteUniform {
    fn chi_squared_divergence(&self, other: &Self) -> f64 {
        let (a1, b1) = (self.lo(), self.hi());
        let (a2, b2) = (other.lo(), other.hi());
        if a1 < a2 || b1 > b2 {
            return f64::INFINITY;
        }
        let n1 = (b1 - a1 + 1) as f64;
        let n2 = (b2 - a2 + 1) as f64;
        n2 / n1 - 1.0
    }
}

impl ChiSquaredDivergence for Binomial {
    /// `(1 + χ²_Bernoulli)^n − 1` for matched trial counts; differing counts are
    /// treated as out-of-family (`+∞`).
    fn chi_squared_divergence(&self, other: &Self) -> f64 {
        if self.trials() != other.trials() {
            return f64::INFINITY;
        }
        let (p, q) = (self.p(), other.p());
        let chi_bern = (p - q).powi(2) / (q * (1.0 - q));
        (1.0 + chi_bern).powf(self.trials() as f64) - 1.0
    }
}

impl ChiSquaredDivergence for NegativeBinomial {
    /// `p₁^{2r} p₂^{−r} (1 − (1−p₁)²/(1−p₂))^{−r} − 1`, for matched successes `r`.
    /// Finite only when `(1−p₁)² < 1−p₂`; `+∞` otherwise (or when `r` differs).
    fn chi_squared_divergence(&self, other: &Self) -> f64 {
        if self.successes() != other.successes() {
            return f64::INFINITY;
        }
        let r = self.successes();
        let (p1, p2) = (self.p(), other.p());
        let z = (1.0 - p1).powi(2) / (1.0 - p2);
        if z >= 1.0 {
            return f64::INFINITY;
        }
        (r * (2.0 * p1.ln() - p2.ln() - (1.0 - z).ln())).exp() - 1.0
    }
}
