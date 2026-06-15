//! **Kullback–Leibler divergence** and its free symmetrizations.
//!
//! [`KlDivergence`] carries the per-family closed-form `D_KL(p ‖ q)`; [`Divergence`]
//! is a blanket trait giving every KL family the reverse direction, the Jeffreys
//! divergence, and cross-entropy for free.

use super::{discrete_kl, kl_term};
use crate::distribution::{
    Bernoulli, Beta, Categorical, Cauchy, Dirichlet, Exponential, Gamma, Geometric, Laplace,
    LogNormal, Moments, Multinomial, Normal, Pareto, Poisson, Rayleigh, Weibull,
};
use crate::function::beta::ln_beta;
use crate::function::gamma::{digamma, gamma, ln_gamma};

/// Euler–Mascheroni constant `γ`, for the Weibull cross-entropy term.
const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// Closed-form Kullback–Leibler divergence `D_KL(self ‖ other)`, in nats.
///
/// `D_KL ≥ 0` for any same-family pair (**Gibbs' inequality**), with equality iff
/// the distributions are identical — proved in `riemann`
/// (`RiemannLean/Bayes/Divergence.lean`: `klDiv_nonneg`, `klDiv_self`).
pub trait KlDivergence<Rhs = Self> {
    fn kl_divergence(&self, other: &Rhs) -> f64;
}

impl KlDivergence for Normal {
    fn kl_divergence(&self, other: &Self) -> f64 {
        let (m1, s1) = self.params();
        let (m2, s2) = other.params();
        (s2 / s1).ln() + (s1 * s1 + (m1 - m2) * (m1 - m2)) / (2.0 * s2 * s2) - 0.5
    }
}

impl KlDivergence for Exponential {
    fn kl_divergence(&self, other: &Self) -> f64 {
        let (l1, l2) = (self.rate(), other.rate());
        (l1 / l2).ln() + l2 / l1 - 1.0
    }
}

impl KlDivergence for Gamma {
    fn kl_divergence(&self, other: &Self) -> f64 {
        let (a1, b1) = (self.shape(), self.rate());
        let (a2, b2) = (other.shape(), other.rate());
        (a1 - a2) * digamma(a1) - ln_gamma(a1)
            + ln_gamma(a2)
            + a2 * (b1.ln() - b2.ln())
            + a1 * (b2 - b1) / b1
    }
}

impl KlDivergence for Beta {
    fn kl_divergence(&self, other: &Self) -> f64 {
        let (a1, b1) = (self.alpha(), self.beta());
        let (a2, b2) = (other.alpha(), other.beta());
        ln_beta(a2, b2) - ln_beta(a1, b1)
            + (a1 - a2) * digamma(a1)
            + (b1 - b2) * digamma(b1)
            + (a2 - a1 + b2 - b1) * digamma(a1 + b1)
    }
}

impl KlDivergence for Bernoulli {
    fn kl_divergence(&self, other: &Self) -> f64 {
        let (p, q) = (self.p(), other.p());
        kl_term(p, q) + kl_term(1.0 - p, 1.0 - q)
    }
}

impl KlDivergence for Poisson {
    fn kl_divergence(&self, other: &Self) -> f64 {
        let (l1, l2) = (self.lambda(), other.lambda());
        l1 * (l1 / l2).ln() - l1 + l2
    }
}

impl KlDivergence for LogNormal {
    fn kl_divergence(&self, other: &Self) -> f64 {
        // A log-normal is a normal in log-space, and KL is invariant under that
        // monotone reparametrization, so this is the `Normal` formula on (μ, σ).
        let (m1, s1) = (self.mu(), self.sigma());
        let (m2, s2) = (other.mu(), other.sigma());
        (s2 / s1).ln() + (s1 * s1 + (m1 - m2) * (m1 - m2)) / (2.0 * s2 * s2) - 0.5
    }
}

impl KlDivergence for Laplace {
    fn kl_divergence(&self, other: &Self) -> f64 {
        let (m1, b1) = (self.location(), self.scale());
        let (m2, b2) = (other.location(), other.scale());
        let d = (m1 - m2).abs();
        (b2 / b1).ln() + (b1 * (-d / b1).exp() + d) / b2 - 1.0
    }
}

impl KlDivergence for Geometric {
    fn kl_divergence(&self, other: &Self) -> f64 {
        // Support {1, 2, …}, pmf p(1−p)^{k−1}; `E[k−1] = (1−p)/p`.
        let (p1, p2) = (self.p(), other.p());
        (p1 / p2).ln() + (1.0 - p1) / p1 * ((1.0 - p1) / (1.0 - p2)).ln()
    }
}

impl KlDivergence for Categorical {
    fn kl_divergence(&self, other: &Self) -> f64 {
        discrete_kl(self.probabilities(), other.probabilities())
    }
}

impl KlDivergence for Multinomial {
    fn kl_divergence(&self, other: &Self) -> f64 {
        // With the same trial count, the joint KL is `n ×` the per-trial
        // (categorical) KL; different `n` means different support.
        if self.trials() != other.trials() {
            return f64::INFINITY;
        }
        self.trials() as f64 * discrete_kl(self.probabilities(), other.probabilities())
    }
}

impl KlDivergence for Dirichlet {
    fn kl_divergence(&self, other: &Self) -> f64 {
        let (a, b) = (self.alpha(), other.alpha());
        if a.len() != b.len() {
            return f64::INFINITY;
        }
        let a0: f64 = a.iter().sum();
        let b0: f64 = b.iter().sum();
        // ln[B(β)/B(α)] = [ln Γ(α₀) − Σ ln Γ(αᵢ)] + [Σ ln Γ(βᵢ) − ln Γ(β₀)].
        let mut kl = ln_gamma(a0) - ln_gamma(b0);
        let psi_a0 = digamma(a0);
        for (&ai, &bi) in a.iter().zip(b) {
            kl += ln_gamma(bi) - ln_gamma(ai) + (ai - bi) * (digamma(ai) - psi_a0);
        }
        kl
    }
}

impl KlDivergence for Cauchy {
    fn kl_divergence(&self, other: &Self) -> f64 {
        // The (symmetric) closed form `ln[((γ₁+γ₂)² + (x₀₁−x₀₂)²) / (4γ₁γ₂)]`.
        let (m1, s1) = (self.location(), self.scale());
        let (m2, s2) = (other.location(), other.scale());
        (((s1 + s2).powi(2) + (m1 - m2).powi(2)) / (4.0 * s1 * s2)).ln()
    }
}

impl KlDivergence for Rayleigh {
    fn kl_divergence(&self, other: &Self) -> f64 {
        // `2 ln(σ₂/σ₁) + (σ₁/σ₂)² − 1`, from `E₁[x²] = 2σ₁²`.
        let (s1, s2) = (self.scale(), other.scale());
        2.0 * (s2 / s1).ln() + (s1 / s2).powi(2) - 1.0
    }
}

impl KlDivergence for Weibull {
    fn kl_divergence(&self, other: &Self) -> f64 {
        // Bauckhage's closed form (shape `k`, scale `λ`); reduces to the
        // exponential KL at `k₁ = k₂ = 1` and to `0` on equal parameters.
        let (k1, l1) = (self.shape(), self.scale());
        let (k2, l2) = (other.shape(), other.scale());
        (k1.ln() - k1 * l1.ln()) - (k2.ln() - k2 * l2.ln())
            + (k1 - k2) * (l1.ln() - EULER_GAMMA / k1)
            + (l1 / l2).powf(k2) * gamma(1.0 + k2 / k1)
            - 1.0
    }
}

impl KlDivergence for Pareto {
    fn kl_divergence(&self, other: &Self) -> f64 {
        // `ln(α₁/α₂) + α₂ ln(xₘ₁/xₘ₂) + α₂/α₁ − 1`, finite only when the support
        // `[xₘ₁, ∞)` sits inside `[xₘ₂, ∞)` (i.e. `xₘ₁ ≥ xₘ₂`).
        let (xm1, a1) = (self.scale(), self.shape());
        let (xm2, a2) = (other.scale(), other.shape());
        if xm1 < xm2 {
            return f64::INFINITY;
        }
        (a1 / a2).ln() + a2 * (xm1 / xm2).ln() + a2 / a1 - 1.0
    }
}

/// Symmetrizations of [`KlDivergence`] available to **every** KL family for free:
/// the reverse direction and the **Jeffreys** divergence.
///
/// A blanket `impl` covers every `T: KlDivergence`, so adding these never touches
/// the per-family formulas. Jeffreys is symmetric and nonnegative — the latter
/// proved in `riemann` (`RiemannLean/Bayes/Divergence.lean`: `jeffreys_symm`,
/// `jeffreys_nonneg`), inherited directly from Gibbs.
///
/// (This is the *Jeffreys divergence* `D_KL(p ‖ q) + D_KL(q ‖ p)`; not to be
/// confused with [`statistics::proportion_jeffreys`](crate::proportion_jeffreys),
/// which is the Jeffreys-prior confidence interval.)
pub trait Divergence: KlDivergence + Sized {
    /// `D_KL(other ‖ self)` — the divergence in the reverse direction. KL is
    /// asymmetric, so this generally differs from [`kl_divergence`](KlDivergence::kl_divergence).
    fn reverse_kl(&self, other: &Self) -> f64 {
        other.kl_divergence(self)
    }

    /// **Jeffreys divergence** `J(self, other) = D_KL(self ‖ other) + D_KL(other ‖ self)`,
    /// the symmetrized KL. Symmetric (`a.jeffreys(b) == b.jeffreys(a)`) and `≥ 0`.
    fn jeffreys(&self, other: &Self) -> f64 {
        self.kl_divergence(other) + other.kl_divergence(self)
    }

    /// Cross-entropy `H(self, other) = H(self) + D_KL(self ‖ other)`, in nats —
    /// the expected surprise of coding `self`-distributed data under `other`.
    /// Derived from [`entropy`](Moments::entropy) and
    /// [`kl_divergence`](KlDivergence::kl_divergence); `None` when `self`'s
    /// entropy is unavailable. Because the KL term is proved nonnegative (Gibbs),
    /// `H(self, other) ≥ H(self)`, with equality iff the distributions agree.
    fn cross_entropy(&self, other: &Self) -> Option<f64>
    where
        Self: Moments,
    {
        Some(self.entropy()? + self.kl_divergence(other))
    }
}

impl<T: KlDivergence> Divergence for T {}
