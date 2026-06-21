//! Shared parametric closed-form divergences, reused across distribution families
//! related by reparametrization.
//!
//! - **Gaussian** (mean `m`, std `s`): [`Normal`](crate::Normal),
//!   [`HalfNormal`](crate::HalfNormal) (with `m = 0`), and — in log-space —
//!   [`LogNormal`](crate::LogNormal).
//! - **Gamma** (shape `a`, rate `b`): [`Gamma`](crate::Gamma),
//!   [`ChiSquared`](crate::ChiSquared) (`a = k/2, b = ½`), and — via the `x ↦ 1/x`
//!   bijection, under which every f-divergence is invariant —
//!   [`InverseGamma`](crate::InverseGamma) (shape `α`, rate = scale `β`).
//! - **Weibull** (shape `k`, scale `λ`): [`Weibull`](crate::Weibull) and — via
//!   `x ↦ 1/x` — [`Frechet`](crate::Frechet) (scale `1/s`).
//!
//! Each helper takes `self`'s parameters first, then `other`'s, matching the
//! `D(self ‖ other)` convention. The `*_bc` helpers return the raw Bhattacharyya
//! coefficient `BC = ∫√(pq)`; the caller forms `H² = 1 − BC`.

use euclid::Real;

/// Euler–Mascheroni constant `γ`.
pub(super) const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

// ---------------------------------------------------------------------------
// Gaussian family, parameterized by (mean m, std s).
// ---------------------------------------------------------------------------

/// `D_KL(N(m0,s0) ‖ N(m1,s1))`.
pub(super) fn gaussian_kl(m0: f64, s0: f64, m1: f64, s1: f64) -> f64 {
    (s1 / s0).ln() + (s0 * s0 + (m0 - m1) * (m0 - m1)) / (2.0 * s1 * s1) - 0.5
}

/// Bhattacharyya coefficient `∫√(pq)` for two Gaussians.
pub(super) fn gaussian_bc(m0: f64, s0: f64, m1: f64, s1: f64) -> f64 {
    let v = s0 * s0 + s1 * s1;
    (2.0 * s0 * s1 / v).sqrt() * (-0.25 * (m0 - m1) * (m0 - m1) / v).exp()
}

/// `χ²(N(m0,s0) ‖ N(m1,s1)) = ∫ p²/q − 1`. Finite only when the numerator stays
/// integrable, i.e. `s0² < 2 s1²`; `+∞` otherwise.
pub(super) fn gaussian_chi_squared(m0: f64, s0: f64, m1: f64, s1: f64) -> f64 {
    let denom = 2.0 * s1 * s1 - s0 * s0;
    if denom <= 0.0 {
        return f64::INFINITY;
    }
    s1 * s1 / (s0 * denom.sqrt()) * ((m0 - m1) * (m0 - m1) / denom).exp() - 1.0
}

/// `D_α(N(m0,s0) ‖ N(m1,s1))` for `α ≠ 1`. `σ_α² = α s1² + (1−α) s0²`; `+∞` when
/// the order-`α` integral diverges (`σ_α² ≤ 0`).
pub(super) fn gaussian_renyi(m0: f64, s0: f64, m1: f64, s1: f64, alpha: f64) -> f64 {
    let var_alpha = alpha * s1 * s1 + (1.0 - alpha) * s0 * s0;
    if var_alpha <= 0.0 {
        return f64::INFINITY;
    }
    let mean_term = alpha * (m0 - m1).powi(2) / (2.0 * var_alpha);
    let log_term =
        ((1.0 - alpha) * s0.ln() + alpha * s1.ln() - 0.5 * var_alpha.ln()) / (alpha - 1.0);
    mean_term + log_term
}

// ---------------------------------------------------------------------------
// Gamma family, parameterized by (shape a, rate b).
// ---------------------------------------------------------------------------

/// `D_KL(Gamma(a1,b1) ‖ Gamma(a2,b2))`.
pub(super) fn gamma_kl(a1: f64, b1: f64, a2: f64, b2: f64) -> f64 {
    (a1 - a2) * a1.digamma() - a1.ln_gamma()
        + a2.ln_gamma()
        + a2 * (b1.ln() - b2.ln())
        + a1 * (b2 - b1) / b1
}

/// Bhattacharyya coefficient `∫√(pq)` for two Gammas.
pub(super) fn gamma_bc(a1: f64, b1: f64, a2: f64, b2: f64) -> f64 {
    let a = 0.5 * (a1 + a2);
    let b = 0.5 * (b1 + b2);
    let ln_bc = a.ln_gamma() + 0.5 * (a1 * b1.ln() + a2 * b2.ln())
        - 0.5 * (a1.ln_gamma() + a2.ln_gamma())
        - a * b.ln();
    ln_bc.exp()
}

/// `χ²(Gamma(a1,b1) ‖ Gamma(a2,b2)) = ∫ p²/q − 1`. Finite only when `2a1 > a2`
/// and `2b1 > b2`; `+∞` otherwise.
pub(super) fn gamma_chi_squared(a1: f64, b1: f64, a2: f64, b2: f64) -> f64 {
    let a = 2.0 * a1 - a2;
    let b = 2.0 * b1 - b2;
    if a <= 0.0 || b <= 0.0 {
        return f64::INFINITY;
    }
    let ln_term = 2.0 * a1 * b1.ln() - 2.0 * a1.ln_gamma() + a2.ln_gamma() - a2 * b2.ln()
        + a.ln_gamma()
        - a * b.ln();
    ln_term.exp() - 1.0
}

/// `D_α(Gamma(a1,b1) ‖ Gamma(a2,b2))` for `α ≠ 1`. `a_α = α a1 + (1−α) a2`,
/// `b_α = α b1 + (1−α) b2`; `+∞` when either is non-positive (integral diverges).
pub(super) fn gamma_renyi(a1: f64, b1: f64, a2: f64, b2: f64, alpha: f64) -> f64 {
    let a_alpha = alpha * a1 + (1.0 - alpha) * a2;
    let b_alpha = alpha * b1 + (1.0 - alpha) * b2;
    if a_alpha <= 0.0 || b_alpha <= 0.0 {
        return f64::INFINITY;
    }
    let num = alpha * (a1 * b1.ln() - a1.ln_gamma())
        + (1.0 - alpha) * (a2 * b2.ln() - a2.ln_gamma())
        + a_alpha.ln_gamma()
        - a_alpha * b_alpha.ln();
    num / (alpha - 1.0)
}

// ---------------------------------------------------------------------------
// Weibull family, parameterized by (shape k, scale λ).
// ---------------------------------------------------------------------------

/// `D_KL(Weibull(k1,l1) ‖ Weibull(k2,l2))` — Bauckhage's closed form.
pub(super) fn weibull_kl(k1: f64, l1: f64, k2: f64, l2: f64) -> f64 {
    (k1.ln() - k1 * l1.ln()) - (k2.ln() - k2 * l2.ln())
        + (k1 - k2) * (l1.ln() - EULER_GAMMA / k1)
        + (l1 / l2).powf(k2) * (1.0 + k2 / k1).gamma()
        - 1.0
}
