#![allow(clippy::unwrap_used, clippy::panic)]

use bayes::function::{ln_multivariate_beta, ln_multivariate_gamma};
use bayes::statistics::hypothesis::{binomial_test, mcnemar, paired_t};
use bayes::{
    cohen_d, covariance_matrix, mean_difference_welch_interval, odds_ratio,
    proportion_clopper_pearson, trimmed_mean, weighted_mean,
};

#[test]
fn new_closed_form_statistics_are_public() {
    assert!(cohen_d(&[3.0, 4.0, 5.0], &[1.0, 2.0, 3.0]) > 1.0);
    assert_eq!(odds_ratio(10, 5, 4, 8), 4.0);
    assert_eq!(weighted_mean(&[1.0, 3.0], &[1.0, 3.0]), 2.5);
    assert_eq!(trimmed_mean(&[1.0, 2.0, 3.0, 100.0], 0.25), 2.5);

    let cov = covariance_matrix(&[vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]]);
    assert_eq!(cov.rows(), 2);
    assert!(cov.get(0, 1) > 0.0);

    let cp = proportion_clopper_pearson(0, 10, 0.95);
    assert_eq!(cp.lower, 0.0);
    assert!(
        mean_difference_welch_interval(&[10.0, 11.0, 12.0], &[1.0, 2.0, 3.0], 0.95).lower > 0.0
    );

    assert!(binomial_test(9, 10, 0.5).p_value < 0.05);
    assert!(mcnemar(12, 2).p_value < 0.05);
    assert_eq!(paired_t(&[2.0, 3.0, 4.0], &[1.0, 2.0, 3.0]).p_value, 0.0);

    assert!(ln_multivariate_gamma(2, 3.0).is_finite());
    assert!(ln_multivariate_beta(&[1.0, 1.0, 1.0]).is_finite());
}
