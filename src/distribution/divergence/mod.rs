//! **Closed-form divergences** between distributions of the same family.
//!
//! The primitive is the **Kullback–Leibler divergence** `D_KL(p ‖ q) = ∫ p ln(p/q)`
//! in **nats** — an analytic formula per family (no Monte-Carlo estimate), so it
//! stays on the closed-form side of the library's boundary. Cross-entropy follows
//! as `H(p, q) = H(p) + D_KL(p ‖ q)`.
//!
//! The full divergence layer, built on top of KL:
//! - [`Divergence`] — **reverse** KL and the symmetric **Jeffreys** divergence
//!   `J = D_KL(p ‖ q) + D_KL(q ‖ p)`, closed-form for every KL family for free.
//! - [`JensenShannon`] — the symmetric, **bounded** divergence `0 ≤ JS ≤ ln 2`.
//! - [`HellingerDistance`] — the bounded metric `H² = 1 − ∫√(pq) ∈ [0, 1]`,
//!   closed-form for all six families.
//! - [`TotalVariation`] — `TV = ½∫|p − q| ∈ [0, 1]`.
//! - [`ChiSquaredDivergence`] — `χ² = ∫(p − q)²/q`, with `D_KL ≤ χ²`.
//! - [`RenyiDivergence`] — the order-`α` family interpolating KL (`α→1`),
//!   Hellinger (`α=½`), and χ² (`α=2`).
//! - [`mutual_information`] — `I(X;Y) = KL(p_{XY} ‖ p_X ⊗ p_Y) ≥ 0`.
//!
//! Jensen–Shannon and total variation are exact only where the relevant
//! mixture/sum stays in-family — the finite-support discretes ([`Bernoulli`],
//! [`Categorical`]); Rényi adds the Gaussian closed form ([`Normal`],
//! [`LogNormal`]); KL, Jeffreys, and Hellinger are closed-form for every family.
//!
//! **Every guarantee is machine-checked** in the `riemann` Lean development
//! (`RiemannLean/Bayes/Divergence.lean`): Gibbs `KL ≥ 0`; `0 ≤ JS ≤ ln 2`;
//! `0 ≤ H² ≤ 1`; `H² ≤ TV`; `H² ≤ ½ KL`; the loose Pinsker `TV² ≤ KL`;
//! `KL ≤ χ²`; the f-divergence umbrella `D_f ≥ 0`; the **data-processing
//! inequality** (KL — and every f-divergence — only decreases under a stochastic
//! channel); **joint convexity** of every f-divergence; the entropy identities
//! (`H(p,q) = H(p) + KL`, max-entropy `H ≤ log n`); and `I(X;Y) ≥ 0`.
//!
//! ## The f-divergence umbrella
//!
//! The per-family closed forms above each instantiate one general object: the
//! **f-divergence** `D_f(p ‖ q) = Σ qᵢ f(pᵢ/qᵢ)` for a convex generator `f` with
//! `f(1) = 0`. The [`fdiv`] submodule exposes it generically over probability
//! vectors — one [`f_divergence`] function with pluggable [`FGenerator`]s
//! ([`Kl`], [`ReverseKl`], [`ChiSquaredGen`], [`SquaredHellinger`],
//! [`TotalVariationGen`], [`Jeffreys`], [`Alpha`]) — plus the [`Channel`]
//! (Markov kernel) carrying the **data-processing inequality**. The three
//! universal guarantees (nonnegativity, data-processing monotonicity, joint
//! convexity) are the `fDiv_*` theorems in `RiemannLean/Bayes/Divergence.lean`.
//!
//! ## Information measures over probability vectors
//!
//! The [`info`] submodule rounds out the discrete information-theory toolkit on
//! the same `&[f64]` representation: [`shannon_entropy`], [`cross_entropy`],
//! [`jensen_shannon`], and [`bhattacharyya_coefficient`] — entropy/cross-entropy,
//! the symmetric bounded JS divergence, and the Hellinger affinity, each cited to
//! its `Divergence.lean` theorem (`entropy_le_log_card`,
//! `crossEntropy_eq_entropy_add_klDiv`, `jsDiv_le_log2`, `bhattacharyya_le_one`).

mod fdiv;
mod info;
pub use fdiv::{
    Alpha, Channel, ChiSquared as ChiSquaredGen, Custom, FGenerator, Jeffreys, Kl, ReverseKl,
    SquaredHellinger, TotalVariation as TotalVariationGen, f_divergence,
};
pub use info::{bhattacharyya_coefficient, cross_entropy, jensen_shannon, shannon_entropy};

use crate::distribution::{
    Bernoulli, Beta, Categorical, Cauchy, Dirichlet, Exponential, Gamma, Geometric, Laplace,
    LogNormal, Moments, Multinomial, Normal, Pareto, Poisson, Rayleigh, Weibull,
};
use crate::function::beta::ln_beta;
use crate::function::gamma::{digamma, gamma, ln_gamma};

/// Euler–Mascheroni constant `γ`, for the Weibull cross-entropy term.
const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// One term `x · ln(x/y)` of a discrete KL / Jensen–Shannon sum, with the limit
/// convention `0 · ln(0/·) = 0`.
fn kl_term(x: f64, y: f64) -> f64 {
    if x > 0.0 { x * (x / y).ln() } else { 0.0 }
}

/// `Σ pᵢ ln(pᵢ/qᵢ)` over two probability vectors, built on [`kl_term`]. `+∞`
/// when the supports disagree (differing lengths, or some `pᵢ > 0 = qᵢ`).
fn discrete_kl(p: &[f64], q: &[f64]) -> f64 {
    if p.len() != q.len() {
        return f64::INFINITY;
    }
    p.iter().zip(q).map(|(&pi, &qi)| kl_term(pi, qi)).sum()
}

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

/// The **Jensen–Shannon divergence** `JS(p, q) = ½ D_KL(p ‖ m) + ½ D_KL(q ‖ m)`,
/// `m = (p+q)/2`, in nats. Unlike KL it is **symmetric** and **bounded**:
/// `0 ≤ JS ≤ ln 2` (its square root is a metric).
///
/// It is implemented only where the mixture `m` stays in the same family and the
/// sum is therefore a closed form — the finite-support discretes. The bounds are
/// proved in `riemann` (`RiemannLean/Bayes/Divergence.lean`: `jsDiv_nonneg`,
/// `jsDiv_le_log2`).
pub trait JensenShannon {
    /// `JS(self ‖ other)`, in nats, in `[0, ln 2]`.
    fn js_divergence(&self, other: &Self) -> f64;
}

impl JensenShannon for Bernoulli {
    fn js_divergence(&self, other: &Self) -> f64 {
        let (p, q) = (self.p(), other.p());
        let m1 = 0.5 * (p + q); // mixture P(success)
        let m0 = 0.5 * ((1.0 - p) + (1.0 - q)); // mixture P(failure)
        0.5 * (kl_term(p, m1) + kl_term(1.0 - p, m0))
            + 0.5 * (kl_term(q, m1) + kl_term(1.0 - q, m0))
    }
}

impl JensenShannon for Categorical {
    /// Categories present in only one distribution (unequal lengths) are treated
    /// as having probability `0` in the other.
    fn js_divergence(&self, other: &Self) -> f64 {
        let (pa, pb) = (self.probabilities(), other.probabilities());
        let n = pa.len().max(pb.len());
        (0..n)
            .map(|i| {
                let a = pa.get(i).copied().unwrap_or(0.0);
                let b = pb.get(i).copied().unwrap_or(0.0);
                let m = 0.5 * (a + b);
                0.5 * (kl_term(a, m) + kl_term(b, m))
            })
            .sum()
    }
}

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
        let v = s1 * s1 + s2 * s2;
        let bc = (2.0 * s1 * s2 / v).sqrt() * (-0.25 * (m1 - m2) * (m1 - m2) / v).exp();
        hellinger_sq_from_bc(bc)
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
        let (a1, b1) = (self.shape(), self.rate());
        let (a2, b2) = (other.shape(), other.rate());
        let a = 0.5 * (a1 + a2);
        let b = 0.5 * (b1 + b2);
        let ln_bc = ln_gamma(a) + 0.5 * (a1 * b1.ln() + a2 * b2.ln())
            - 0.5 * (ln_gamma(a1) + ln_gamma(a2))
            - a * b.ln();
        hellinger_sq_from_bc(ln_bc.exp())
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

/// The **total-variation distance** `TV(p, q) = ½ Σ |pᵢ − qᵢ|` (`= sup_A |P(A) − Q(A)|`),
/// in `[0, 1]`. Symmetric and a metric. Implemented only where it is a closed-form
/// finite sum — the finite-support discretes ([`Bernoulli`], [`Categorical`]).
///
/// Proved in `riemann` (`RiemannLean/Bayes/Divergence.lean`): `tvDist_nonneg`,
/// `tvDist_symm`, `tvDist_le_one`, and **Pinsker's inequality** `klDiv_pinsker`
/// (`TV² ≤ KL`), which controls TV by the KL divergence.
pub trait TotalVariation {
    /// The total-variation distance `TV ∈ [0, 1]`.
    fn total_variation(&self, other: &Self) -> f64;
}

impl TotalVariation for Bernoulli {
    fn total_variation(&self, other: &Self) -> f64 {
        // ½(|p−q| + |(1−p)−(1−q)|) = |p − q|.
        (self.p() - other.p()).abs()
    }
}

impl TotalVariation for Categorical {
    /// Categories present in only one distribution are treated as `0` in the other.
    fn total_variation(&self, other: &Self) -> f64 {
        let (pa, pb) = (self.probabilities(), other.probabilities());
        let n = pa.len().max(pb.len());
        0.5 * (0..n)
            .map(|i| (pa.get(i).copied().unwrap_or(0.0) - pb.get(i).copied().unwrap_or(0.0)).abs())
            .sum::<f64>()
    }
}

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

/// **Mutual information** `I(X;Y) = Σ_{i,k} J[i][k]·ln( J[i][k] / (p_X[i]·p_Y[k]) )`,
/// in nats, for a joint distribution `joint` (row `i` = outcome of `X`, column `k`
/// = outcome of `Y`; entries should sum to 1). Equivalently `KL(p_{XY} ‖ p_X ⊗ p_Y)`.
///
/// `I(X;Y) ≥ 0`, with equality iff `X` and `Y` are independent — proved in `riemann`
/// (`RiemannLean/Bayes/Divergence.lean`: `mutualInfo_nonneg`). The `0·ln 0 = 0`
/// convention is applied to empty cells.
pub fn mutual_information(joint: &[Vec<f64>]) -> f64 {
    if joint.is_empty() {
        return 0.0;
    }
    let cols = joint.iter().map(Vec::len).max().unwrap_or(0);
    let px: Vec<f64> = joint.iter().map(|row| row.iter().sum()).collect();
    let mut py = vec![0.0; cols];
    for row in joint {
        for (k, &v) in row.iter().enumerate() {
            py[k] += v;
        }
    }
    joint
        .iter()
        .enumerate()
        .flat_map(|(i, row)| row.iter().enumerate().map(move |(k, &j)| (i, k, j)))
        .map(|(i, k, j)| kl_term(j, px[i] * py[k]))
        .sum()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::distribution::{Continuous, Discrete};

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    /// Numerical Bhattacharyya coefficient `∫ √(f·g)` by the trapezoid rule, for
    /// validating the closed-form Hellinger affinities.
    fn bc_integral(f: impl Fn(f64) -> f64, g: impl Fn(f64) -> f64, lo: f64, hi: f64) -> f64 {
        let n = 100_000usize;
        let h = (hi - lo) / n as f64;
        let mut s = 0.0;
        for k in 0..=n {
            let x = lo + k as f64 * h;
            let w = if k == 0 || k == n { 0.5 } else { 1.0 };
            s += w * (f(x) * g(x)).max(0.0).sqrt();
        }
        s * h
    }

    #[test]
    fn self_divergence_is_zero() {
        close(
            Normal::new(1.0, 2.0)
                .unwrap()
                .kl_divergence(&Normal::new(1.0, 2.0).unwrap()),
            0.0,
            1e-12,
        );
        close(
            Exponential::new(3.0)
                .unwrap()
                .kl_divergence(&Exponential::new(3.0).unwrap()),
            0.0,
            1e-12,
        );
        close(
            Gamma::new(2.0, 1.5)
                .unwrap()
                .kl_divergence(&Gamma::new(2.0, 1.5).unwrap()),
            0.0,
            1e-10,
        );
        close(
            Beta::new(2.0, 3.0)
                .unwrap()
                .kl_divergence(&Beta::new(2.0, 3.0).unwrap()),
            0.0,
            1e-10,
        );
        close(
            Poisson::new(4.0)
                .unwrap()
                .kl_divergence(&Poisson::new(4.0).unwrap()),
            0.0,
            1e-12,
        );
    }

    #[test]
    fn kl_is_non_negative_and_asymmetric() {
        let p = Normal::new(0.0, 1.0).unwrap();
        let q = Normal::new(2.0, 3.0).unwrap();
        assert!(p.kl_divergence(&q) > 0.0);
        assert!(q.kl_divergence(&p) > 0.0);
        assert_ne!(p.kl_divergence(&q), q.kl_divergence(&p));
    }

    #[test]
    fn normal_kl_known_value() {
        close(
            Normal::new(0.0, 1.0)
                .unwrap()
                .kl_divergence(&Normal::new(1.0, 1.0).unwrap()),
            0.5,
            1e-12,
        );
    }

    #[test]
    fn bernoulli_kl_matches_definition() {
        let kl = Bernoulli::new(0.3)
            .unwrap()
            .kl_divergence(&Bernoulli::new(0.5).unwrap());
        let expect = 0.3 * (0.3f64 / 0.5).ln() + 0.7 * (0.7f64 / 0.5).ln();
        close(kl, expect, 1e-12);
    }

    #[test]
    fn reverse_kl_is_the_swapped_direction() {
        let p = Normal::new(0.0, 1.0).unwrap();
        let q = Normal::new(2.0, 3.0).unwrap();
        close(p.reverse_kl(&q), q.kl_divergence(&p), 1e-12);
        // KL is asymmetric, so reverse ≠ forward here.
        assert_ne!(p.reverse_kl(&q), p.kl_divergence(&q));
    }

    #[test]
    fn jeffreys_is_symmetric_sum_of_both_directions_and_nonnegative() {
        for (p, q) in [
            (
                Normal::new(0.0, 1.0).unwrap(),
                Normal::new(2.0, 3.0).unwrap(),
            ),
            (
                Normal::new(-1.0, 2.0).unwrap(),
                Normal::new(1.0, 0.5).unwrap(),
            ),
        ] {
            let j = p.jeffreys(&q);
            close(j, p.kl_divergence(&q) + q.kl_divergence(&p), 1e-12);
            close(j, q.jeffreys(&p), 1e-12); // symmetric
            assert!(j > 0.0); // > 0 for distinct, ≥ 0 in general (Gibbs)
        }
        // Identical distributions: Jeffreys = 0.
        let b = Beta::new(2.0, 3.0).unwrap();
        close(b.jeffreys(&Beta::new(2.0, 3.0).unwrap()), 0.0, 1e-10);
    }

    #[test]
    fn bernoulli_js_self_is_zero_symmetric_and_within_bounds() {
        let a = Bernoulli::new(0.3).unwrap();
        let b = Bernoulli::new(0.9).unwrap();
        close(a.js_divergence(&a), 0.0, 1e-12);
        close(a.js_divergence(&b), b.js_divergence(&a), 1e-12); // symmetric
        let js = a.js_divergence(&b);
        assert!(js > 0.0 && js <= std::f64::consts::LN_2 + 1e-12); // 0 ≤ JS ≤ ln 2
    }

    #[test]
    fn bernoulli_js_matches_hand_expansion() {
        let (p, q) = (0.3, 0.5);
        let js = Bernoulli::new(p)
            .unwrap()
            .js_divergence(&Bernoulli::new(q).unwrap());
        let m1 = 0.5 * (p + q);
        let m0 = 0.5 * ((1.0 - p) + (1.0 - q));
        let expect = 0.5 * (p * (p / m1).ln() + (1.0 - p) * ((1.0 - p) / m0).ln())
            + 0.5 * (q * (q / m1).ln() + (1.0 - q) * ((1.0 - q) / m0).ln());
        close(js, expect, 1e-12);
    }

    #[test]
    fn maximally_separated_bernoullis_saturate_the_log2_bound() {
        // JS(δ₀, δ₁) = ln 2, the upper bound (disjoint supports).
        let js = Bernoulli::new(0.0)
            .unwrap()
            .js_divergence(&Bernoulli::new(1.0).unwrap());
        close(js, std::f64::consts::LN_2, 1e-12);
    }

    #[test]
    fn categorical_js_self_zero_symmetric_and_bounded() {
        let a = Categorical::new(&[0.2, 0.3, 0.5]).unwrap();
        let b = Categorical::new(&[0.6, 0.1, 0.3]).unwrap();
        close(a.js_divergence(&a), 0.0, 1e-12);
        close(a.js_divergence(&b), b.js_divergence(&a), 1e-12);
        let js = a.js_divergence(&b);
        assert!(js > 0.0 && js <= std::f64::consts::LN_2 + 1e-12);
    }

    #[test]
    fn categorical_js_matches_bernoulli_for_two_categories() {
        // A 2-category Categorical is a Bernoulli; the two paths must agree.
        let cat = Categorical::new(&[0.7, 0.3])
            .unwrap()
            .js_divergence(&Categorical::new(&[0.4, 0.6]).unwrap());
        // Bernoulli p() is P(success); category 0 here plays "success".
        let bern = Bernoulli::new(0.7)
            .unwrap()
            .js_divergence(&Bernoulli::new(0.4).unwrap());
        close(cat, bern, 1e-12);
    }

    #[test]
    fn hellinger_self_is_zero_and_symmetric() {
        macro_rules! check {
            ($a:expr, $b:expr) => {{
                close($a.hellinger_sq(&$a), 0.0, 1e-12);
                close($a.hellinger_sq(&$b), $b.hellinger_sq(&$a), 1e-12);
                let h2 = $a.hellinger_sq(&$b);
                assert!((0.0..=1.0).contains(&h2), "H² out of [0,1]: {h2}");
            }};
        }
        check!(
            Normal::new(0.0, 1.0).unwrap(),
            Normal::new(1.5, 2.0).unwrap()
        );
        check!(
            Exponential::new(2.0).unwrap(),
            Exponential::new(0.5).unwrap()
        );
        check!(Gamma::new(2.0, 1.5).unwrap(), Gamma::new(3.0, 0.7).unwrap());
        check!(Beta::new(2.0, 3.0).unwrap(), Beta::new(4.0, 2.0).unwrap());
        check!(Poisson::new(3.0).unwrap(), Poisson::new(7.0).unwrap());
        check!(Bernoulli::new(0.3).unwrap(), Bernoulli::new(0.8).unwrap());
    }

    #[test]
    fn bernoulli_hellinger_matches_closed_form() {
        let (p, q) = (0.3, 0.8);
        let h2 = Bernoulli::new(p)
            .unwrap()
            .hellinger_sq(&Bernoulli::new(q).unwrap());
        let bc = (p * q).sqrt() + ((1.0 - p) * (1.0 - q)).sqrt();
        close(h2, 1.0 - bc, 1e-12);
    }

    #[test]
    fn poisson_hellinger_matches_pmf_sum() {
        let (a, b) = (Poisson::new(3.0).unwrap(), Poisson::new(7.0).unwrap());
        let bc: f64 = (0..200).map(|k| (a.pmf(k) * b.pmf(k)).sqrt()).sum();
        close(a.hellinger_sq(&b), 1.0 - bc, 1e-9);
    }

    #[test]
    fn continuous_hellinger_matches_numeric_integral() {
        let n1 = Normal::new(0.0, 1.0).unwrap();
        let n2 = Normal::new(1.5, 2.0).unwrap();
        close(
            n1.hellinger_sq(&n2),
            1.0 - bc_integral(|x| n1.pdf(x), |x| n2.pdf(x), -40.0, 40.0),
            2e-3,
        );
        let e1 = Exponential::new(2.0).unwrap();
        let e2 = Exponential::new(0.5).unwrap();
        close(
            e1.hellinger_sq(&e2),
            1.0 - bc_integral(|x| e1.pdf(x), |x| e2.pdf(x), 0.0, 80.0),
            2e-3,
        );
        let g1 = Gamma::new(2.0, 1.5).unwrap();
        let g2 = Gamma::new(3.0, 0.7).unwrap();
        close(
            g1.hellinger_sq(&g2),
            1.0 - bc_integral(|x| g1.pdf(x), |x| g2.pdf(x), 0.0, 80.0),
            2e-3,
        );
        let b1 = Beta::new(2.0, 3.0).unwrap();
        let b2 = Beta::new(4.0, 2.0).unwrap();
        close(
            b1.hellinger_sq(&b2),
            1.0 - bc_integral(|x| b1.pdf(x), |x| b2.pdf(x), 0.0, 1.0),
            2e-3,
        );
    }

    #[test]
    fn total_variation_bernoulli_categorical_symmetric_bounded() {
        let a = Bernoulli::new(0.3).unwrap();
        let b = Bernoulli::new(0.8).unwrap();
        close(a.total_variation(&b), 0.5, 1e-12); // |0.3 − 0.8|
        close(a.total_variation(&a), 0.0, 1e-12);
        close(a.total_variation(&b), b.total_variation(&a), 1e-12);

        let c = Categorical::new(&[0.2, 0.3, 0.5]).unwrap();
        let d = Categorical::new(&[0.5, 0.1, 0.4]).unwrap();
        let tv = c.total_variation(&d);
        close(tv, 0.5 * (0.3 + 0.2 + 0.1), 1e-12);
        assert!((0.0..=1.0).contains(&tv));
        // 2-category Categorical agrees with Bernoulli.
        let c2 = Categorical::new(&[0.7, 0.3]).unwrap();
        let d2 = Categorical::new(&[0.4, 0.6]).unwrap();
        close(
            c2.total_variation(&d2),
            Bernoulli::new(0.7)
                .unwrap()
                .total_variation(&Bernoulli::new(0.4).unwrap()),
            1e-12,
        );
    }

    #[test]
    fn pinsker_and_hellinger_comparisons_hold() {
        // The proved inequalities, checked numerically on Bernoulli pairs:
        // TV² ≤ KL (Pinsker), H² ≤ TV, and H² ≤ ½ KL.
        for (p, q) in [(0.3, 0.8), (0.1, 0.2), (0.5, 0.95)] {
            let a = Bernoulli::new(p).unwrap();
            let b = Bernoulli::new(q).unwrap();
            let tv = a.total_variation(&b);
            let kl = a.kl_divergence(&b);
            let h2 = a.hellinger_sq(&b);
            assert!(tv * tv <= kl + 1e-12, "Pinsker TV²≤KL failed: {p},{q}");
            assert!(h2 <= tv + 1e-12, "H²≤TV failed: {p},{q}");
            assert!(h2 <= 0.5 * kl + 1e-12, "H²≤½KL failed: {p},{q}");
        }
    }

    #[test]
    fn chi_squared_self_zero_and_closed_forms() {
        // Self-divergence is zero.
        close(
            Bernoulli::new(0.3)
                .unwrap()
                .chi_squared_divergence(&Bernoulli::new(0.3).unwrap()),
            0.0,
            1e-12,
        );
        close(
            Poisson::new(4.0)
                .unwrap()
                .chi_squared_divergence(&Poisson::new(4.0).unwrap()),
            0.0,
            1e-12,
        );
        // Bernoulli closed form vs the two-term definition.
        let (p, q) = (0.3, 0.8);
        let chi = Bernoulli::new(p)
            .unwrap()
            .chi_squared_divergence(&Bernoulli::new(q).unwrap());
        let def = (p - q).powi(2) / q + ((1.0 - p) - (1.0 - q)).powi(2) / (1.0 - q);
        close(chi, def, 1e-12);
    }

    #[test]
    fn poisson_chi_squared_matches_pmf_sum() {
        let (a, b) = (Poisson::new(3.0).unwrap(), Poisson::new(5.0).unwrap());
        // χ² = Σ pₖ²/qₖ − 1.
        let s: f64 = (0..400)
            .map(|k| {
                let (pk, qk) = (a.pmf(k), b.pmf(k));
                if qk > 0.0 { pk * pk / qk } else { 0.0 }
            })
            .sum();
        close(a.chi_squared_divergence(&b), s - 1.0, 1e-9);
    }

    #[test]
    fn categorical_chi_squared_matches_definition_and_handles_zero_support() {
        let c = Categorical::new(&[0.2, 0.3, 0.5]).unwrap();
        let d = Categorical::new(&[0.5, 0.1, 0.4]).unwrap();
        let def: f64 = [(0.2_f64, 0.5_f64), (0.3, 0.1), (0.5, 0.4)]
            .iter()
            .map(|(a, b)| (a - b).powi(2) / b)
            .sum();
        close(c.chi_squared_divergence(&d), def, 1e-12);
        // q has a zero where p does not → +∞.
        let p = Categorical::new(&[0.5, 0.5]).unwrap();
        let q = Categorical::new(&[1.0, 0.0]).unwrap();
        assert!(p.chi_squared_divergence(&q).is_infinite());
    }

    #[test]
    fn exponential_chi_squared_converges_only_for_light_enough_denominator() {
        // Convergent case 2λ₁ > λ₂: validate against the numeric integral of p²/q.
        let a = Exponential::new(2.0).unwrap();
        let b = Exponential::new(1.0).unwrap();
        let integral = {
            let n = 200_000usize;
            let (lo, hi) = (0.0, 100.0);
            let h = (hi - lo) / n as f64;
            (0..=n)
                .map(|k| {
                    let x = lo + k as f64 * h;
                    let w = if k == 0 || k == n { 0.5 } else { 1.0 };
                    w * a.pdf(x) * a.pdf(x) / b.pdf(x)
                })
                .sum::<f64>()
                * h
        };
        close(a.chi_squared_divergence(&b), integral - 1.0, 2e-3);
        // Divergent case 2λ₁ ≤ λ₂: heavier-tailed numerator → +∞.
        let c = Exponential::new(1.0).unwrap();
        let d = Exponential::new(3.0).unwrap();
        assert!(c.chi_squared_divergence(&d).is_infinite());
    }

    #[test]
    fn kl_le_chi_squared_holds() {
        // The proved bound KL ≤ χ², checked on Bernoulli and Poisson pairs.
        for (p, q) in [(0.3, 0.8), (0.1, 0.2), (0.6, 0.55)] {
            let a = Bernoulli::new(p).unwrap();
            let b = Bernoulli::new(q).unwrap();
            assert!(
                a.kl_divergence(&b) <= a.chi_squared_divergence(&b) + 1e-12,
                "KL≤χ² (Bernoulli) failed: {p},{q}"
            );
        }
        let a = Poisson::new(3.0).unwrap();
        let b = Poisson::new(4.0).unwrap();
        assert!(a.kl_divergence(&b) <= a.chi_squared_divergence(&b) + 1e-12);
    }

    #[test]
    fn renyi_self_zero_and_limits_tie_to_the_other_divergences() {
        let a = Bernoulli::new(0.3).unwrap();
        let b = Bernoulli::new(0.8).unwrap();
        // Self-divergence is zero at any order.
        for alpha in [0.5, 1.0, 2.0, 3.0] {
            close(a.renyi_divergence(&a, alpha), 0.0, 1e-12);
        }
        // α = 1 is exactly KL; just below/above α = 1 approaches it.
        close(a.renyi_divergence(&b, 1.0), a.kl_divergence(&b), 1e-12);
        close(
            a.renyi_divergence(&b, 1.0 + 1e-6),
            a.kl_divergence(&b),
            1e-4,
        );
        // α = ½ is −2·ln(1 − H²) (Hellinger).
        close(
            a.renyi_divergence(&b, 0.5),
            -2.0 * (1.0 - a.hellinger_sq(&b)).ln(),
            1e-12,
        );
        // α = 2 is ln(1 + χ²).
        close(
            a.renyi_divergence(&b, 2.0),
            (1.0 + a.chi_squared_divergence(&b)).ln(),
            1e-12,
        );
        // Monotone non-decreasing in α.
        assert!(a.renyi_divergence(&b, 0.5) <= a.renyi_divergence(&b, 2.0) + 1e-12);
    }

    #[test]
    fn categorical_renyi_matches_bernoulli_for_two_categories() {
        for alpha in [0.5, 1.0, 2.0] {
            let cat = Categorical::new(&[0.7, 0.3])
                .unwrap()
                .renyi_divergence(&Categorical::new(&[0.4, 0.6]).unwrap(), alpha);
            let bern = Bernoulli::new(0.7)
                .unwrap()
                .renyi_divergence(&Bernoulli::new(0.4).unwrap(), alpha);
            close(cat, bern, 1e-12);
        }
    }

    #[test]
    fn gaussian_renyi_matches_its_limits_and_the_integral() {
        let p = Normal::new(0.0, 1.0).unwrap();
        let q = Normal::new(1.5, 2.0).unwrap();
        // Self-divergence is zero at every order.
        for alpha in [0.3, 0.5, 1.0, 2.0, 3.0] {
            close(p.renyi_divergence(&p, alpha), 0.0, 1e-12);
        }
        // α = 1 is exactly KL.
        close(p.renyi_divergence(&q, 1.0), p.kl_divergence(&q), 1e-12);
        // α = ½ is −2·ln(1 − H²) (Hellinger), reusing the closed-form affinity.
        close(
            p.renyi_divergence(&q, 0.5),
            -2.0 * (1.0 - p.hellinger_sq(&q)).ln(),
            1e-12,
        );
        // Direct check against D_α = ln(∫ pᵅ q¹⁻ᵅ)/(α−1) by numeric integration.
        let renyi_integral = |alpha: f64| {
            let n = 200_000usize;
            let (lo, hi) = (-60.0, 60.0);
            let h = (hi - lo) / n as f64;
            let s: f64 = (0..=n)
                .map(|k| {
                    let x = lo + k as f64 * h;
                    let w = if k == 0 || k == n { 0.5 } else { 1.0 };
                    w * p.pdf(x).powf(alpha) * q.pdf(x).powf(1.0 - alpha)
                })
                .sum::<f64>()
                * h;
            s.ln() / (alpha - 1.0)
        };
        for alpha in [0.3, 0.7, 2.0] {
            close(p.renyi_divergence(&q, alpha), renyi_integral(alpha), 2e-3);
        }
        // Monotone non-decreasing in α.
        assert!(p.renyi_divergence(&q, 0.5) <= p.renyi_divergence(&q, 2.0) + 1e-12);
        // Diverges when the order-α power is not integrable (σ_α² ≤ 0):
        // α = 2, σ₀ = 3, σ₁ = 1 ⇒ 2·1 − 1·9 < 0.
        assert!(
            Normal::new(0.0, 3.0)
                .unwrap()
                .renyi_divergence(&Normal::new(0.0, 1.0).unwrap(), 2.0)
                .is_infinite()
        );
        // LogNormal reuses the Normal formula on (μ, σ).
        for alpha in [0.4, 2.0] {
            close(
                LogNormal::new(0.0, 1.0)
                    .unwrap()
                    .renyi_divergence(&LogNormal::new(1.5, 2.0).unwrap(), alpha),
                p.renyi_divergence(&q, alpha),
                1e-12,
            );
        }
    }

    #[test]
    fn mutual_information_zero_iff_independent_and_nonneg() {
        // Independent joint p(i,k) = pX(i)·pY(k): I = 0.
        let indep = vec![vec![0.25, 0.25], vec![0.25, 0.25]];
        close(mutual_information(&indep), 0.0, 1e-12);
        // Perfectly correlated (diagonal) uniform: I = ln 2 (= marginal entropy).
        let correlated = vec![vec![0.5, 0.0], vec![0.0, 0.5]];
        close(
            mutual_information(&correlated),
            std::f64::consts::LN_2,
            1e-12,
        );
        // A generic joint: I ≥ 0, and matches KL(joint ‖ pX⊗pY) by hand.
        let j = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
        let mi = mutual_information(&j);
        let (px, py) = ([0.3, 0.7], [0.4, 0.6]);
        let expect: f64 = [(0, 0, 0.1), (0, 1, 0.2), (1, 0, 0.3), (1, 1, 0.4)]
            .iter()
            .map(|&(i, k, v): &(usize, usize, f64)| v * (v / (px[i] * py[k])).ln())
            .sum();
        close(mi, expect, 1e-12);
        assert!(mi > 0.0);
    }

    // --- broadened KL coverage beyond the base six families ---

    #[test]
    fn lognormal_kl_equals_underlying_normal_kl() {
        // KL is invariant under the log reparametrization, so LN(0,1)‖LN(1,1)
        // matches N(0,1)‖N(1,1) = 0.5.
        close(
            LogNormal::new(0.0, 1.0)
                .unwrap()
                .kl_divergence(&LogNormal::new(1.0, 1.0).unwrap()),
            0.5,
            1e-12,
        );
    }

    #[test]
    fn laplace_and_geometric_kl_known_values() {
        // Laplace, same location: ln(b₂/b₁) + b₁/b₂ − 1.
        close(
            Laplace::new(0.0, 1.0)
                .unwrap()
                .kl_divergence(&Laplace::new(0.0, 2.0).unwrap()),
            2.0_f64.ln() - 0.5,
            1e-12,
        );
        // Geometric: ln(p₁/p₂) + ((1−p₁)/p₁)·ln((1−p₁)/(1−p₂)) = ln(4/3).
        close(
            Geometric::new(0.5)
                .unwrap()
                .kl_divergence(&Geometric::new(0.25).unwrap()),
            (4.0_f64 / 3.0).ln(),
            1e-12,
        );
    }

    #[test]
    fn categorical_multinomial_dirichlet_kl() {
        let p = Categorical::new(&[0.5, 0.5]).unwrap();
        let q = Categorical::new(&[0.25, 0.75]).unwrap();
        close(p.kl_divergence(&q), 0.5 * (4.0_f64 / 3.0).ln(), 1e-12);
        close(p.kl_divergence(&p), 0.0, 1e-12);
        assert!(
            p.kl_divergence(&Categorical::new(&[0.3, 0.3, 0.4]).unwrap())
                .is_infinite()
        );

        // Multinomial KL is `n ×` the categorical KL for matched trial counts.
        let m = Multinomial::new(10, &[0.5, 0.5]).unwrap();
        let mq = Multinomial::new(10, &[0.25, 0.75]).unwrap();
        close(m.kl_divergence(&mq), 10.0 * p.kl_divergence(&q), 1e-12);
        assert!(
            m.kl_divergence(&Multinomial::new(7, &[0.5, 0.5]).unwrap())
                .is_infinite()
        );

        // Dirichlet: nonneg, self-zero, asymmetric.
        let da = Dirichlet::new(vec![2.0, 3.0, 4.0]).unwrap();
        let db = Dirichlet::new(vec![1.0, 1.0, 1.0]).unwrap();
        close(da.kl_divergence(&da), 0.0, 1e-12);
        assert!(da.kl_divergence(&db) > 0.0);
        assert_ne!(da.kl_divergence(&db), db.kl_divergence(&da));
    }

    #[test]
    fn cauchy_rayleigh_weibull_pareto_kl() {
        // Cauchy: ln[((γ₁+γ₂)² + Δ²)/(4γ₁γ₂)]; here ln(9/8).
        let c = Cauchy::new(0.0, 1.0).unwrap();
        close(
            c.kl_divergence(&Cauchy::new(0.0, 2.0).unwrap()),
            (9.0_f64 / 8.0).ln(),
            1e-12,
        );
        close(c.kl_divergence(&c), 0.0, 1e-12);

        // Rayleigh: 2 ln(σ₂/σ₁) + (σ₁/σ₂)² − 1.
        close(
            Rayleigh::new(1.0)
                .unwrap()
                .kl_divergence(&Rayleigh::new(2.0).unwrap()),
            2.0 * 2.0_f64.ln() - 0.75,
            1e-12,
        );

        // Weibull(scale, shape=1) is Exponential(rate 1/scale); the KLs agree.
        close(
            Weibull::new(2.0, 1.0)
                .unwrap()
                .kl_divergence(&Weibull::new(3.0, 1.0).unwrap()),
            Exponential::new(0.5)
                .unwrap()
                .kl_divergence(&Exponential::new(1.0 / 3.0).unwrap()),
            1e-12,
        );

        // Pareto: finite only when xₘ₁ ≥ xₘ₂.
        let p = Pareto::new(1.0, 3.0).unwrap();
        close(
            p.kl_divergence(&Pareto::new(1.0, 2.0).unwrap()),
            1.5_f64.ln() + 2.0 / 3.0 - 1.0,
            1e-12,
        );
        assert!(
            Pareto::new(2.0, 2.0)
                .unwrap()
                .kl_divergence(&Pareto::new(3.0, 2.0).unwrap())
                .is_infinite()
        );
    }

    #[test]
    fn cross_entropy_decomposes_and_handles_missing_entropy() {
        // H(p, q) = H(p) + D_KL(p ‖ q), equal to the closed form for normals.
        let p = Normal::new(0.0, 1.0).unwrap();
        let q = Normal::new(1.0, 1.0).unwrap();
        close(
            p.cross_entropy(&q).unwrap(),
            p.entropy().unwrap() + p.kl_divergence(&q),
            1e-12,
        );
        close(
            p.cross_entropy(&q).unwrap(),
            1.0 + 0.5 * std::f64::consts::TAU.ln(),
            1e-12,
        );
        // Gibbs equality case: H(p, p) = H(p); nonneg KL only raises it.
        close(p.cross_entropy(&p).unwrap(), p.entropy().unwrap(), 1e-12);
        assert!(p.cross_entropy(&q).unwrap() > p.entropy().unwrap());
        // No analytic entropy (Poisson) → honest `None`.
        assert!(
            Poisson::new(2.0)
                .unwrap()
                .cross_entropy(&Poisson::new(3.0).unwrap())
                .is_none()
        );
    }
}
