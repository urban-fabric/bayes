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
//!   closed-form wherever the affinity `∫√(pq)` is analytic.
//! - [`TotalVariation`] — `TV = ½∫|p − q| ∈ [0, 1]`.
//! - [`ChiSquaredDivergence`] — `χ² = ∫(p − q)²/q`, with `D_KL ≤ χ²`.
//! - [`RenyiDivergence`] — the order-`α` family interpolating KL (`α→1`),
//!   Hellinger (`α=½`), and χ² (`α=2`).
//! - [`mutual_information`] / [`joint_entropy`] / [`conditional_entropy`] /
//!   [`variation_of_information`] / [`normalized_mutual_information`] — the full information
//!   diagram over a joint table: `I(X;Y) ≥ 0`, `H(X,Y)`, `H(Y|X) ≥ 0`, the VI metric, and
//!   the `[0,1]`-normalized MI.
//!
//! KL (and its free Jeffreys symmetrization) has the widest coverage. Hellinger,
//! Rényi, and χ² are closed-form across the Gaussian
//! ([`Normal`](crate::Normal), [`HalfNormal`](crate::HalfNormal),
//! [`LogNormal`](crate::LogNormal)), gamma
//! ([`Gamma`](crate::Gamma), [`ChiSquared`](crate::ChiSquared),
//! [`InverseGamma`](crate::InverseGamma)), directional
//! ([`VonMises`](crate::VonMises)), inverse-Gaussian ([`Wald`](crate::Wald)),
//! uniform, and finite-support discrete ([`Binomial`](crate::Binomial),
//! [`NegativeBinomial`](crate::NegativeBinomial), …) families where the relevant
//! integral or sum collapses. Jensen–Shannon and total variation stay restricted
//! to the finite-support discretes ([`Bernoulli`](crate::Bernoulli),
//! [`Categorical`](crate::Categorical)), where the mixture/sum stays in-family.
//! Several heavier-tailed families ([`Gumbel`](crate::Gumbel),
//! [`Frechet`](crate::Frechet)) expose KL only — their other divergences have no
//! general closed form.
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
//! the symmetric bounded JS divergence, and the Hellinger affinity — plus the
//! **maximum-entropy** surface [`max_entropy`], [`normalized_entropy`] (the
//! `H/ln n ∈ [0, 1]` equitability index), and the [`is_entropy_admissible`]
//! certifier. Each is cited to its `Divergence.lean` theorem (`entropy_le_log_card`,
//! `entropy_nonneg`, `crossEntropy_eq_entropy_add_klDiv`, `jsDiv_le_log2`,
//! `bhattacharyya_le_one`).
#![cfg_attr(test, allow(clippy::expect_used, clippy::panic, clippy::unwrap_used))]

// Bring the concrete families + trait spine into crate-root scope so closed-form
// divergence impls (and intra-doc links) reference them as `crate::Normal`, etc.
// Private: not re-exported — divergence's own public surface stays its trait set.
#[allow(unused_imports)]
use bayes_distribution::*;
use euclid::Real;

mod chi_squared;
mod fdiv;
mod hellinger;
mod info;
mod jensen_shannon;
mod kl;
mod mutual_information;
mod renyi;
mod total_variation;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests_families;

pub use chi_squared::ChiSquaredDivergence;
pub use fdiv::{
    Alpha, Channel, ChiSquared as ChiSquaredGen, Custom, FGenerator, Jeffreys, Kl, RealFGenerator,
    ReverseKl, SquaredHellinger, TotalVariation as TotalVariationGen, f_divergence,
    f_divergence_real,
};
pub use hellinger::HellingerDistance;
pub use info::{
    bhattacharyya_coefficient, bhattacharyya_coefficient_real, cross_entropy, cross_entropy_real,
    is_entropy_admissible, jensen_shannon, jensen_shannon_real, max_entropy, max_entropy_real,
    normalized_entropy, normalized_entropy_real, shannon_entropy, shannon_entropy_real,
};
pub use jensen_shannon::JensenShannon;
pub use kl::{Divergence, KlDivergence};
pub use mutual_information::{
    conditional_entropy, conditional_entropy_real, joint_entropy, joint_entropy_real,
    mutual_information, mutual_information_real, normalized_mutual_information,
    normalized_mutual_information_real, variation_of_information, variation_of_information_real,
};
pub use renyi::RenyiDivergence;
pub use total_variation::TotalVariation;

/// One term `x · ln(x/y)` of a discrete KL / Jensen–Shannon sum, with the limit
/// convention `0 · ln(0/·) = 0`. Shared across the divergence submodules.
fn kl_term<R: Real>(x: R, y: R) -> R {
    let ratio = x.clone() / y;
    x.xlogy(ratio)
}

/// `Σ pᵢ ln(pᵢ/qᵢ)` over two probability vectors, built on [`kl_term`]. `+∞`
/// when the supports disagree (differing lengths, or some `pᵢ > 0 = qᵢ`).
fn discrete_kl<R: Real>(p: &[R], q: &[R]) -> R {
    if p.len() != q.len() {
        return R::infinity();
    }
    p.iter()
        .cloned()
        .zip(q.iter().cloned())
        .map(|(pi, qi)| kl_term(pi, qi))
        .fold(R::zero(), |acc, x| acc + x)
}
