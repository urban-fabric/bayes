//! **Estimation-theory diagnostics** — the Cramér–Rao efficiency surface, backed by the
//! machine-checked estimation core in riemann `RiemannLean/Bayes/CramerRao.lean` (all on
//! the pinned 3-axiom basis):
//!
//! - `cramer_rao` — for any unbiased estimator `T`, `Var(T)·I(θ) ≥ 1`;
//! - `cramer_rao_attained` — equality (efficiency) holds iff the score is affine in `T`;
//! - `fisher_eq_var_score` / `score_mean_zero` — the Fisher information `I(θ)` is the
//!   variance of the (mean-zero) score `U = ∂/∂θ log p`;
//! - `variance_ge_two_proj` (riemann `Bayes/Inequalities`) — the Bhattacharyya / higher-order
//!   Cramér–Rao bound: a second orthogonal score direction tightens the variance floor;
//! - `rao_blackwell` / `condExp_orthogonality` (riemann `Bayes/RaoBlackwell.lean`) —
//!   conditioning an estimator on a sufficient statistic never increases its variance;
//! - `lehmann_scheffe_unique` (same module) — an unbiased function of a complete sufficient
//!   statistic is the *unique* UMVU estimator ([`is_umvu`]).
//! - `sum_sq_decomp_chains` (riemann `Markov/Diagnostics.lean`) — the finite-sample
//!   within/between/total sum-of-squares partition behind grouped variance diagnostics.
//!
//! The Cramér–Rao functions take the Fisher information as an input — like
//! [`super::confidence::concentration`] takes a tail/MGF bound — so no per-distribution
//! score machinery is needed; [`fisher_information_from_scores`] supplies it empirically.
//! [`rao_blackwell`] / [`conditional_variance`] act directly on grouped sample data.

/// A grouped finite-sample variance partition using the population (`/ n`) convention.
///
/// The fields satisfy `total_ss = within_ss + between_ss` and therefore
/// `total_variance = within_variance + between_variance`, up to floating-point
/// roundoff. This is the direct data-analysis form of riemann
/// `Markov/Diagnostics.sum_sq_decomp_chains`: within-group spread plus group-mean
/// spread recovers total spread about the grand mean.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VarianceDecomposition {
    /// Number of observations.
    pub count: usize,
    /// Number of distinct groups.
    pub groups: usize,
    /// Mean over all observations.
    pub grand_mean: f64,
    /// `Σᵢ (xᵢ - grand_mean)²`.
    pub total_ss: f64,
    /// `Σ_g Σ_{i∈g} (xᵢ - mean_g)²`.
    pub within_ss: f64,
    /// `Σ_g n_g (mean_g - grand_mean)²`.
    pub between_ss: f64,
    /// `total_ss / count`.
    pub total_variance: f64,
    /// `within_ss / count`.
    pub within_variance: f64,
    /// `between_ss / count`.
    pub between_variance: f64,
}

impl VarianceDecomposition {
    /// Fraction of total variation explained by group means, `between_ss / total_ss`.
    ///
    /// This is the same one-way ANOVA effect-size ratio usually called η².
    /// Returns `NaN` for a zero-total-variance decomposition.
    pub fn explained_fraction(self) -> f64 {
        if self.total_ss <= 0.0 {
            f64::NAN
        } else {
            self.between_ss / self.total_ss
        }
    }
}

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

/// **The Bhattacharyya (higher-order Cramér–Rao) variance bound** — the two-projection
/// generalization of [`cramer_rao_lower_bound`]. Given two **uncorrelated** score directions
/// `U, V` (`Cov(U, V) = 0`) with positive variances, the variance of any estimator `X` is
/// bounded below by what it explains along *both*:
/// `Var(X) ≥ Cov(X,U)²/Var(U) + Cov(X,V)²/Var(V)` — Bessel's inequality for two orthogonal
/// projections. The single term `Cov(X,U)²/Var(U)` is the Cauchy–Schwarz / plain Cramér–Rao
/// floor (for an unbiased estimator `Cov(T, U) = 1` and `Var(U) = I`, recovering `1/I`); the
/// second orthogonal direction `V` *raises* the floor — the Bhattacharyya improvement over
/// first-order Cramér–Rao. Returns the variance lower bound. Backed by riemann
/// `Bayes/Inequalities.variance_ge_two_proj`. Returns `NaN` for a non-positive `var_u`/`var_v`.
pub fn bhattacharyya_lower_bound(cov_u: f64, var_u: f64, cov_v: f64, var_v: f64) -> f64 {
    if var_u <= 0.0 || var_v <= 0.0 {
        return f64::NAN;
    }
    cov_u * cov_u / var_u + cov_v * cov_v / var_v
}

/// **Bhattacharyya admissibility check**: whether a reported estimator variance respects the
/// two-projection floor [`bhattacharyya_lower_bound`]. A `false` means the claimed variance
/// is below what the two orthogonal score directions already explain — impossible, so it
/// flags an over-optimistic variance or a bug. The sharper companion to [`is_cr_admissible`]
/// (which uses a single direction). Backed by riemann `Bayes/Inequalities.variance_ge_two_proj`.
pub fn is_bhattacharyya_admissible(
    estimator_variance: f64,
    cov_u: f64,
    var_u: f64,
    cov_v: f64,
    var_v: f64,
) -> bool {
    let floor = bhattacharyya_lower_bound(cov_u, var_u, cov_v, var_v);
    floor.is_finite() && estimator_variance >= floor * (1.0 - 1e-9)
}

/// **Rao–Blackwellization** — improve an estimator by conditioning on a sufficient statistic.
/// Given `values[i]` (the raw estimator `T` at sample `i`) and `groups[i]` (the value of the
/// sufficient statistic there — samples sharing a group lie on the same fiber), returns the
/// **conditional-mean estimator** `E[T | group]`: each value replaced by the mean of `T` over
/// its group. The Rao–Blackwellized estimator has the **same mean** and **never larger variance**
/// than the original (riemann `rao_blackwell`: `Var(E[T|g]) ≤ Var(T)`, via the
/// conditional-expectation orthogonality `condExp_orthogonality`) — the variance drops by exactly
/// the within-group ([`conditional_variance`]) term. Returns an empty vector on length mismatch
/// or empty input.
pub fn rao_blackwell(values: &[f64], groups: &[usize]) -> Vec<f64> {
    use std::collections::HashMap;
    if values.is_empty() || values.len() != groups.len() {
        return Vec::new();
    }
    let mut acc: HashMap<usize, (f64, usize)> = HashMap::new();
    for (&v, &g) in values.iter().zip(groups) {
        let entry = acc.entry(g).or_insert((0.0, 0));
        entry.0 += v;
        entry.1 += 1;
    }
    groups
        .iter()
        .map(|&g| {
            let (sum, count) = acc[&g];
            sum / count as f64
        })
        .collect()
}

/// **Conditional (within-group) variance** `E[Var(T | group)] = E[(T − E[T|group])²]` — the
/// mean squared deviation of each value from its group mean, i.e. the variance Rao–Blackwellization
/// removes. With the between-group variance `Var(E[T|group])` (the population variance of
/// [`rao_blackwell`]'s output) it gives the **law of total variance**
/// `Var(T) = E[Var(T|g)] + Var(E[T|g])` (riemann `condExp_orthogonality` / `rao_blackwell`),
/// the ANOVA / random-effects decomposition into unexplained (within) and explained (between)
/// parts. Uses the population (`÷ n`) convention so the decomposition is exact. Returns `NaN`
/// on length mismatch or empty input.
pub fn conditional_variance(values: &[f64], groups: &[usize]) -> f64 {
    let conditional_mean = rao_blackwell(values, groups);
    if conditional_mean.is_empty() {
        return f64::NAN;
    }
    values
        .iter()
        .zip(&conditional_mean)
        .map(|(&v, &m)| (v - m) * (v - m))
        .sum::<f64>()
        / values.len() as f64
}

/// **Grouped variance decomposition** — split finite-sample spread into within-group and
/// between-group components:
/// `Σᵢ(xᵢ − μ)² = Σ_g Σ_{i∈g}(xᵢ − μ_g)² + Σ_g n_g(μ_g − μ)²`.
///
/// The return value uses population (`/ n`) scaling, so the variance partition mirrors
/// the sum-of-squares identity exactly. This generalizes [`conditional_variance`] by
/// returning the full within/between/total partition and the explained fraction
/// [`VarianceDecomposition::explained_fraction`]. Backed by riemann
/// `Markov/Diagnostics.sum_sq_decomp` / `sum_sq_decomp_chains`. Returns `None` on
/// empty input, length mismatch, or non-finite values.
pub fn variance_decomposition(values: &[f64], groups: &[usize]) -> Option<VarianceDecomposition> {
    use std::collections::HashMap;

    if values.is_empty() || values.len() != groups.len() || values.iter().any(|x| !x.is_finite()) {
        return None;
    }

    let count = values.len();
    let grand_mean = values.iter().sum::<f64>() / count as f64;
    let mut group_acc: HashMap<usize, (f64, usize)> = HashMap::new();
    for (&value, &group) in values.iter().zip(groups) {
        let entry = group_acc.entry(group).or_insert((0.0, 0));
        entry.0 += value;
        entry.1 += 1;
    }

    let group_means: HashMap<usize, f64> = group_acc
        .iter()
        .map(|(&group, &(sum, n))| (group, sum / n as f64))
        .collect();

    let within_ss = values
        .iter()
        .zip(groups)
        .map(|(&value, &group)| {
            let centered = value - group_means[&group];
            centered * centered
        })
        .sum::<f64>();

    let between_ss = group_acc
        .iter()
        .map(|(&group, &(_, n))| {
            let centered = group_means[&group] - grand_mean;
            n as f64 * centered * centered
        })
        .sum::<f64>();

    let total_ss = within_ss + between_ss;
    let denom = count as f64;
    Some(VarianceDecomposition {
        count,
        groups: group_acc.len(),
        grand_mean,
        total_ss,
        within_ss,
        between_ss,
        total_variance: total_ss / denom,
        within_variance: within_ss / denom,
        between_variance: between_ss / denom,
    })
}

/// **UMVU certification (Lehmann–Scheffé)** — whether an unbiased estimator is the *unique*
/// uniformly-minimum-variance unbiased estimator. The condition is structural: the estimator
/// must be a **function of the sufficient statistic** — constant on each fiber of `groups`,
/// equivalently left unchanged by [`rao_blackwell`] (so conditioning cannot improve it). Under
/// the standard preconditions (`groups` a *complete* sufficient statistic and `values` unbiased)
/// this certifies both optimality and **uniqueness**: by riemann `rao_blackwell` the optimum is a
/// function of the statistic, and by `lehmann_scheffe_unique` there is exactly *one* such unbiased
/// estimator — so any other unbiased function of the statistic must equal this one. A `false`
/// means the estimator can be strictly improved by Rao–Blackwellization (it is not yet the UMVU).
/// Returns `false` on length mismatch or empty input.
pub fn is_umvu(values: &[f64], groups: &[usize]) -> bool {
    let conditional_mean = rao_blackwell(values, groups);
    !conditional_mean.is_empty()
        && values
            .iter()
            .zip(&conditional_mean)
            .all(|(&v, &m)| (v - m).abs() <= 1e-9 * v.abs().max(1.0))
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

    #[test]
    fn bhattacharyya_generalizes_cramer_rao_and_tightens_it() {
        // One direction (cov_v = 0) recovers the plain CRB: Cov(T,U)=1, Var(U)=I=4 ⇒ 1/4.
        close(
            bhattacharyya_lower_bound(1.0, 4.0, 0.0, 1.0),
            cramer_rao_lower_bound(4.0),
            1e-12,
        );
        // A second orthogonal direction strictly raises the floor: 1/4 + 1/2 = 3/4.
        let two = bhattacharyya_lower_bound(1.0, 4.0, 1.0, 2.0);
        close(two, 0.75, 1e-12);
        assert!(two > bhattacharyya_lower_bound(1.0, 4.0, 0.0, 1.0));
        assert!(bhattacharyya_lower_bound(1.0, 0.0, 1.0, 2.0).is_nan());
    }

    #[test]
    fn bhattacharyya_admissibility_flags_sub_floor_variances() {
        // floor = 3/4: admissible at/above, rejected below.
        assert!(is_bhattacharyya_admissible(0.75, 1.0, 4.0, 1.0, 2.0));
        assert!(is_bhattacharyya_admissible(1.0, 1.0, 4.0, 1.0, 2.0));
        assert!(!is_bhattacharyya_admissible(0.5, 1.0, 4.0, 1.0, 2.0));
        // an invalid direction makes the floor non-finite ⇒ never admissible.
        assert!(!is_bhattacharyya_admissible(10.0, 1.0, 0.0, 1.0, 2.0));
    }

    #[test]
    fn rao_blackwell_reduces_variance_and_preserves_the_mean() {
        use crate::statistics::{mean, population_variance};
        let values = [1.0, 3.0, 10.0, 12.0];
        let groups = [0usize, 0, 1, 1];
        let rb = rao_blackwell(&values, &groups);
        // each value → its group mean: {1,3}→2, {10,12}→11.
        assert_eq!(rb, vec![2.0, 2.0, 11.0, 11.0]);
        // the conditional-mean estimator preserves the overall mean...
        close(mean(&rb), mean(&values), 1e-12);
        // ...and never has larger variance (Rao–Blackwell).
        assert!(population_variance(&rb) <= population_variance(&values) + 1e-12);
        // singleton groups change nothing; one group ⇒ everything is the grand mean.
        assert_eq!(rao_blackwell(&values, &[0, 1, 2, 3]), values.to_vec());
        assert_eq!(rao_blackwell(&values, &[5, 5, 5, 5]), vec![6.5; 4]);
        // degenerate inputs.
        assert!(rao_blackwell(&[], &[]).is_empty());
        assert!(rao_blackwell(&[1.0, 2.0], &[0]).is_empty());
    }

    #[test]
    fn conditional_variance_satisfies_the_law_of_total_variance() {
        use crate::statistics::population_variance;
        let values = [1.0, 3.0, 10.0, 12.0];
        let groups = [0usize, 0, 1, 1];
        let within = conditional_variance(&values, &groups);
        let between = population_variance(&rao_blackwell(&values, &groups));
        // Var(T) = E[Var(T|g)] + Var(E[T|g]) — exact (population convention).
        close(within + between, population_variance(&values), 1e-12);
        // within is the mean of the per-group variances: both groups have var 1.
        close(within, 1.0, 1e-12);
        // one group per point ⇒ no within variance; one group ⇒ all variance is within.
        close(conditional_variance(&values, &[0, 1, 2, 3]), 0.0, 1e-12);
        close(
            conditional_variance(&values, &[7, 7, 7, 7]),
            population_variance(&values),
            1e-12,
        );
        assert!(conditional_variance(&[1.0], &[0, 1]).is_nan());
    }

    #[test]
    fn variance_decomposition_splits_within_and_between_spread() {
        let values = [1.0, 3.0, 10.0, 12.0];
        let groups = [0usize, 0, 1, 1];
        let split = variance_decomposition(&values, &groups).unwrap();

        close(split.grand_mean, 6.5, 1e-12);
        close(split.within_ss, 4.0, 1e-12);
        close(split.between_ss, 81.0, 1e-12);
        close(split.total_ss, split.within_ss + split.between_ss, 1e-12);
        close(
            split.total_variance,
            split.within_variance + split.between_variance,
            1e-12,
        );
        close(
            split.within_variance,
            conditional_variance(&values, &groups),
            1e-12,
        );
        close(split.explained_fraction(), 81.0 / 85.0, 1e-12);
        assert_eq!(split.count, 4);
        assert_eq!(split.groups, 2);
    }

    #[test]
    fn variance_decomposition_handles_extreme_groupings() {
        let values = [1.0, 3.0, 10.0, 12.0];

        let singleton = variance_decomposition(&values, &[0, 1, 2, 3]).unwrap();
        close(singleton.within_ss, 0.0, 1e-12);
        close(singleton.between_ss, singleton.total_ss, 1e-12);
        close(singleton.explained_fraction(), 1.0, 1e-12);

        let one_group = variance_decomposition(&values, &[7, 7, 7, 7]).unwrap();
        close(one_group.between_ss, 0.0, 1e-12);
        close(one_group.within_ss, one_group.total_ss, 1e-12);
        close(one_group.explained_fraction(), 0.0, 1e-12);

        assert!(variance_decomposition(&[], &[]).is_none());
        assert!(variance_decomposition(&[1.0], &[0, 1]).is_none());
        assert!(variance_decomposition(&[f64::NAN], &[0]).is_none());
    }

    #[test]
    fn is_umvu_certifies_a_function_of_the_statistic() {
        let values = [1.0, 3.0, 10.0, 12.0];
        let groups = [0usize, 0, 1, 1];
        // The raw estimator varies within each group ⇒ improvable, not yet UMVU.
        assert!(!is_umvu(&values, &groups));
        // Its Rao–Blackwellization is constant on the fibers ⇒ the (unique) UMVU; and
        // Rao–Blackwellizing it again is a no-op (fixed point).
        let rb = rao_blackwell(&values, &groups);
        assert!(is_umvu(&rb, &groups));
        assert_eq!(rao_blackwell(&rb, &groups), rb);
        // A fully-informative statistic (singleton fibers) makes every estimator UMVU.
        assert!(is_umvu(&values, &[0, 1, 2, 3]));
        // Under one group, only a constant estimator is a function of the statistic.
        assert!(is_umvu(&[5.0, 5.0, 5.0], &[0, 0, 0]));
        assert!(!is_umvu(&[5.0, 6.0, 5.0], &[0, 0, 0]));
        // degenerate inputs are not certified.
        assert!(!is_umvu(&[], &[]));
        assert!(!is_umvu(&[1.0, 2.0], &[0]));
    }
}
