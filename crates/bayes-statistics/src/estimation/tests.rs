use super::*;
use bayes_distribution_ext::assert_close;

#[test]
fn fisher_information_is_the_mean_square_score() {
    // E[U²] for U ∈ {−1, 1, −1, 1} is 1; for {−2, 2} it is 4.
    assert_close!(
        fisher_information_from_scores(&[-1.0, 1.0, -1.0, 1.0]),
        1.0,
        1e-12,
    );
    assert_close!(fisher_information_from_scores(&[-2.0, 2.0]), 4.0, 1e-12);
    assert!(fisher_information_from_scores(&[]).is_nan());
}

#[test]
fn crb_is_inverse_fisher() {
    assert_close!(cramer_rao_lower_bound(4.0), 0.25, 1e-12);
    // a Normal(μ, σ²) mean has I = n/σ²; with n/σ² = 4 the floor is 1/4.
    assert!(cramer_rao_lower_bound(0.0).is_nan());
}

#[test]
fn efficiency_is_one_at_the_bound_and_flags_violations() {
    // var = CRB = 1/I ⇒ efficiency 1 (efficient estimator).
    assert_close!(relative_efficiency(0.25, 4.0), 1.0, 1e-12);
    // twice the CRB ⇒ efficiency 1/2.
    assert_close!(relative_efficiency(0.5, 4.0), 0.5, 1e-12);
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
    assert_close!(
        bhattacharyya_lower_bound(1.0, 4.0, 0.0, 1.0),
        cramer_rao_lower_bound(4.0),
        1e-12,
    );
    // A second orthogonal direction strictly raises the floor: 1/4 + 1/2 = 3/4.
    let two = bhattacharyya_lower_bound(1.0, 4.0, 1.0, 2.0);
    assert_close!(two, 0.75, 1e-12);
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
    use crate::{mean, population_variance};
    let values = [1.0, 3.0, 10.0, 12.0];
    let groups = [0usize, 0, 1, 1];
    let rb = rao_blackwell(&values, &groups);
    // each value → its group mean: {1,3}→2, {10,12}→11.
    assert_eq!(rb, vec![2.0, 2.0, 11.0, 11.0]);
    // the conditional-mean estimator preserves the overall mean...
    assert_close!(mean(&rb), mean(&values), 1e-12);
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
    use crate::population_variance;
    let values = [1.0, 3.0, 10.0, 12.0];
    let groups = [0usize, 0, 1, 1];
    let within = conditional_variance(&values, &groups);
    let between = population_variance(&rao_blackwell(&values, &groups));
    // Var(T) = E[Var(T|g)] + Var(E[T|g]) — exact (population convention).
    assert_close!(within + between, population_variance(&values), 1e-12);
    // within is the mean of the per-group variances: both groups have var 1.
    assert_close!(within, 1.0, 1e-12);
    // one group per point ⇒ no within variance; one group ⇒ all variance is within.
    assert_close!(conditional_variance(&values, &[0, 1, 2, 3]), 0.0, 1e-12);
    assert_close!(
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

    assert_close!(split.grand_mean, 6.5, 1e-12);
    assert_close!(split.within_ss, 4.0, 1e-12);
    assert_close!(split.between_ss, 81.0, 1e-12);
    assert_close!(split.total_ss, split.within_ss + split.between_ss, 1e-12);
    assert_close!(
        split.total_variance,
        split.within_variance + split.between_variance,
        1e-12,
    );
    assert_close!(
        split.within_variance,
        conditional_variance(&values, &groups),
        1e-12,
    );
    assert_close!(split.explained_fraction(), 81.0 / 85.0, 1e-12);
    assert_eq!(split.count, 4);
    assert_eq!(split.groups, 2);
}

#[test]
fn variance_decomposition_handles_extreme_groupings() {
    let values = [1.0, 3.0, 10.0, 12.0];

    let singleton = variance_decomposition(&values, &[0, 1, 2, 3]).unwrap();
    assert_close!(singleton.within_ss, 0.0, 1e-12);
    assert_close!(singleton.between_ss, singleton.total_ss, 1e-12);
    assert_close!(singleton.explained_fraction(), 1.0, 1e-12);

    let one_group = variance_decomposition(&values, &[7, 7, 7, 7]).unwrap();
    assert_close!(one_group.between_ss, 0.0, 1e-12);
    assert_close!(one_group.within_ss, one_group.total_ss, 1e-12);
    assert_close!(one_group.explained_fraction(), 0.0, 1e-12);

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
