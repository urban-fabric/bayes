//! **Estimation-theory diagnostics** — the Cramér–Rao efficiency surface, backed by the
//! machine-checked estimation core in riemann `RiemannLean/Bayes/CramerRao.lean` (all on
//! the pinned 3-axiom basis):
//!
//! - `cramer_rao` — for any unbiased estimator `T`, `Var(T)·I(θ) ≥ 1`;
//! - `cramer_rao_attained` — equality (efficiency) holds iff the score is affine in `T`;
//! - `fisher_eq_var_score` / `score_mean_zero` — the Fisher information `I(θ)` is the
//!   variance of the (mean-zero) score `U = ∂/∂θ log p`.
//!
//! These functions take the Fisher information as an input — like
//! [`super::confidence::concentration`] takes a tail/MGF bound — so no per-distribution
//! score machinery is needed; [`fisher_information_from_scores`] supplies it empirically.

/// **Observed Fisher information** from a sample of **score** values
/// `U(xᵢ) = ∂/∂θ log p(xᵢ; θ)`: `I = E[U²]`. This equals the variance of the score
/// because the score has mean zero (riemann `score_mean_zero`), and that variance *is*
/// the Fisher information (riemann `fisher_eq_var_score`). Returns `NaN` on an
/// empty/non-finite sample.
pub fn fisher_information_from_scores(scores: &[f64]) -> f64 {
    if scores.is_empty() || scores.iter().any(|u| !u.is_finite()) {
        return f64::NAN;
    }
    scores.iter().map(|u| u * u).sum::<f64>() / scores.len() as f64
}

/// **The Cramér–Rao lower bound** on the variance of any unbiased estimator: `1/I(θ)`.
/// No unbiased estimator can have variance below this floor (riemann `cramer_rao`:
/// `Var(T)·I ≥ 1`). Returns `NaN` for `fisher_information ≤ 0`.
pub fn cramer_rao_lower_bound(fisher_information: f64) -> f64 {
    if fisher_information <= 0.0 {
        return f64::NAN;
    }
    1.0 / fisher_information
}

/// **Relative efficiency** of an unbiased estimator: `1/(Var(T)·I(θ)) = CRB / Var(T)`,
/// lying in `(0, 1]`. It is `1` exactly when the estimator attains the Cramér–Rao bound
/// — i.e. is *efficient* — which happens iff its score is affine in `T` (riemann
/// `cramer_rao_attained`). A value `> 1` is impossible for a genuine unbiased estimator
/// and flags a variance below the CR floor (see [`is_cr_admissible`]). `NaN` on invalid
/// input.
pub fn relative_efficiency(estimator_variance: f64, fisher_information: f64) -> f64 {
    if estimator_variance <= 0.0 || fisher_information <= 0.0 {
        return f64::NAN;
    }
    1.0 / (estimator_variance * fisher_information)
}

/// **Cramér–Rao admissibility check**: whether a reported unbiased-estimator variance
/// respects the CR floor `Var(T)·I(θ) ≥ 1` (riemann `cramer_rao`). A `false` means the
/// claimed variance is *below* the information-theoretic minimum — impossible for an
/// unbiased estimator, so it flags an over-optimistic variance or a bug. A small
/// relative tolerance absorbs floating-point slack at the efficient boundary.
pub fn is_cr_admissible(estimator_variance: f64, fisher_information: f64) -> bool {
    estimator_variance > 0.0
        && fisher_information > 0.0
        && estimator_variance * fisher_information >= 1.0 - 1e-9
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn fisher_information_is_the_mean_square_score() {
        // E[U²] for U ∈ {−1, 1, −1, 1} is 1; for {−2, 2} it is 4.
        close(
            fisher_information_from_scores(&[-1.0, 1.0, -1.0, 1.0]),
            1.0,
            1e-12,
        );
        close(fisher_information_from_scores(&[-2.0, 2.0]), 4.0, 1e-12);
        assert!(fisher_information_from_scores(&[]).is_nan());
    }

    #[test]
    fn crb_is_inverse_fisher() {
        close(cramer_rao_lower_bound(4.0), 0.25, 1e-12);
        // a Normal(μ, σ²) mean has I = n/σ²; with n/σ² = 4 the floor is 1/4.
        assert!(cramer_rao_lower_bound(0.0).is_nan());
    }

    #[test]
    fn efficiency_is_one_at_the_bound_and_flags_violations() {
        // var = CRB = 1/I ⇒ efficiency 1 (efficient estimator).
        close(relative_efficiency(0.25, 4.0), 1.0, 1e-12);
        // twice the CRB ⇒ efficiency 1/2.
        close(relative_efficiency(0.5, 4.0), 0.5, 1e-12);
        // an admissible estimator has efficiency ≤ 1; a sub-CRB variance gives > 1.
        assert!(relative_efficiency(1.0, 4.0) <= 1.0);
        assert!(relative_efficiency(0.1, 4.0) > 1.0);
    }

    #[test]
    fn admissibility_rejects_sub_crb_variances() {
        assert!(is_cr_admissible(0.25, 4.0)); // exactly at the bound
        assert!(is_cr_admissible(1.0, 4.0)); // comfortably above
        assert!(!is_cr_admissible(0.1, 4.0)); // 0.4 < 1: impossible for unbiased T
        assert!(!is_cr_admissible(0.25, 0.0)); // invalid Fisher information
    }
}
