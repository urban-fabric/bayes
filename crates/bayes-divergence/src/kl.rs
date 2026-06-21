//! **Kullback–Leibler divergence** and its free symmetrizations.
//!
//! [`KlDivergence`] carries the per-family closed-form `D_KL(p ‖ q)`; [`Divergence`]
//! is a blanket trait giving every KL family the reverse direction, the Jeffreys
//! divergence, and cross-entropy for free.

use super::families::{EULER_GAMMA, gamma_kl, gaussian_kl, weibull_kl};
use super::{discrete_kl, kl_term};
use crate::{
    Bernoulli, Beta, Binomial, Categorical, Cauchy, ChiSquared, DiscreteUniform, Exponential,
    Frechet, Gamma, Geometric, Gumbel, HalfNormal, InverseGamma, Laplace, LogNormal, Moments,
    NegativeBinomial, Normal, Pareto, Poisson, Rayleigh, Uniform, VonMises, Wald, Weibull,
};
use euclid::LazyReal;

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
        gaussian_kl(m1, s1, m2, s2)
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
        gamma_kl(self.shape(), self.rate(), other.shape(), other.rate())
    }
}

impl KlDivergence for Beta {
    fn kl_divergence(&self, other: &Self) -> f64 {
        let (a1, b1) = (self.alpha(), self.beta());
        let (a2, b2) = (other.alpha(), other.beta());
        a2.ln_beta(b2) - a1.ln_beta(b1)
            + (a1 - a2) * a1.digamma()
            + (b1 - b2) * b1.digamma()
            + (a2 - a1 + b2 - b1) * (a1 + b1).digamma()
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
        gaussian_kl(self.mu(), self.sigma(), other.mu(), other.sigma())
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
        weibull_kl(self.shape(), self.scale(), other.shape(), other.scale())
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

impl KlDivergence for ChiSquared {
    fn kl_divergence(&self, other: &Self) -> f64 {
        // χ²(k) is Gamma(shape k/2, rate ½).
        gamma_kl(0.5 * self.freedom(), 0.5, 0.5 * other.freedom(), 0.5)
    }
}

impl KlDivergence for InverseGamma {
    fn kl_divergence(&self, other: &Self) -> f64 {
        // KL is invariant under the `x ↦ 1/x` bijection, so an inverse-gamma KL
        // equals the gamma KL on the same (shape, rate ≡ scale) parameters.
        gamma_kl(self.shape(), self.scale(), other.shape(), other.scale())
    }
}

impl KlDivergence for HalfNormal {
    fn kl_divergence(&self, other: &Self) -> f64 {
        // Folding to x ≥ 0 cancels, leaving the zero-mean Gaussian KL on the scales.
        gaussian_kl(0.0, self.scale(), 0.0, other.scale())
    }
}

impl KlDivergence for Frechet {
    fn kl_divergence(&self, other: &Self) -> f64 {
        // X ~ Fréchet(α, s) ⟺ 1/X ~ Weibull(shape α, scale 1/s); KL is invariant
        // under that bijection.
        weibull_kl(
            self.shape(),
            1.0 / self.scale(),
            other.shape(),
            1.0 / other.scale(),
        )
    }
}

impl KlDivergence for Gumbel {
    fn kl_divergence(&self, other: &Self) -> f64 {
        // Closed form on (location μ, scale β); reduces to 0 on equal parameters.
        let (mu1, b1) = (self.location(), self.scale());
        let (mu2, b2) = (other.location(), other.scale());
        (b2 / b1).ln()
            + EULER_GAMMA * (b1 / b2 - 1.0)
            + (mu1 - mu2) / b2
            + ((mu2 - mu1) / b2).exp() * (1.0 + b1 / b2).gamma()
            - 1.0
    }
}

impl KlDivergence for Wald {
    fn kl_divergence(&self, other: &Self) -> f64 {
        // Inverse-Gaussian (mean μ, shape λ) closed form.
        let (m1, l1) = (self.mu(), self.shape());
        let (m2, l2) = (other.mu(), other.shape());
        0.5 * (l1 / l2).ln()
            + l2 * (m1 - 2.0 * m2) / (2.0 * m2 * m2)
            + 0.5 * l2 * (1.0 / m1 + 1.0 / l1)
            - 0.5
    }
}

impl KlDivergence for VonMises {
    fn kl_divergence(&self, other: &Self) -> f64 {
        // `ln(I₀(κ₂)/I₀(κ₁)) + A(κ₁)(κ₁ − κ₂ cos Δμ)`, A = I₁/I₀ the mean
        // resultant length.
        let (mu1, k1) = (self.mu(), self.kappa());
        let (mu2, k2) = (other.mu(), other.kappa());
        k2.ln_bessel_i0() - k1.ln_bessel_i0()
            + k1.bessel_i1_over_i0() * (k1 - k2 * (mu1 - mu2).cos())
    }
}

impl KlDivergence for Uniform {
    fn kl_divergence(&self, other: &Self) -> f64 {
        // Finite only when the support `[a₁,b₁]` sits inside `[a₂,b₂]`; then
        // `ln((b₂−a₂)/(b₁−a₁))`.
        let (a1, b1) = (self.lo(), self.hi());
        let (a2, b2) = (other.lo(), other.hi());
        if a1 < a2 || b1 > b2 {
            return f64::INFINITY;
        }
        ((b2 - a2) / (b1 - a1)).ln()
    }
}

impl KlDivergence for Binomial {
    fn kl_divergence(&self, other: &Self) -> f64 {
        // n iid Bernoulli trials: KL is `n ×` the per-trial Bernoulli KL — the
        // binomial coefficient cancels in the log-ratio. Differing trial counts
        // are treated as out-of-family (`+∞`).
        if self.trials() != other.trials() {
            return f64::INFINITY;
        }
        let (p, q) = (self.p(), other.p());
        self.trials() as f64 * (kl_term(p, q) + kl_term(1.0 - p, 1.0 - q))
    }
}

impl KlDivergence for NegativeBinomial {
    fn kl_divergence(&self, other: &Self) -> f64 {
        // Closed form for a matched number of successes `r` (the `Γ` factors then
        // cancel in the log-ratio); differing `r` is treated as out-of-family.
        if self.successes() != other.successes() {
            return f64::INFINITY;
        }
        let r = self.successes();
        let (p1, p2) = (self.p(), other.p());
        r * (p1 / p2).ln() + r * (1.0 - p1) / p1 * ((1.0 - p1) / (1.0 - p2)).ln()
    }
}

impl KlDivergence for DiscreteUniform {
    fn kl_divergence(&self, other: &Self) -> f64 {
        // Finite only when the support `{a₁..b₁}` sits inside `{a₂..b₂}`; then
        // `ln(n₂/n₁)`.
        let (a1, b1) = (self.lo(), self.hi());
        let (a2, b2) = (other.lo(), other.hi());
        if a1 < a2 || b1 > b2 {
            return f64::INFINITY;
        }
        let n1 = (b1 - a1 + 1) as f64;
        let n2 = (b2 - a2 + 1) as f64;
        (n2 / n1).ln()
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
/// confused with `bayes_statistics::proportion_jeffreys`,
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
    /// [`kl_divergence`](KlDivergence::kl_divergence); returns `NaN` when `self`'s
    /// entropy is unavailable. Because the KL term is proved nonnegative (Gibbs),
    /// `H(self, other) ≥ H(self)`, with equality iff the distributions agree.
    fn cross_entropy(&self, other: &Self) -> f64
    where
        Self: Moments,
    {
        self.entropy() + self.kl_divergence(other)
    }
}

impl<T: KlDivergence> Divergence for T {}
