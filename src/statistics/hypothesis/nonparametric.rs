//! Rank-based non-parametric tests: **Mann–Whitney U** (independent samples) and
//! **Wilcoxon signed-rank** (paired samples).

use super::{TestResult, binomial_two_sided_p_value, nan_result, z_two_sided};
use crate::statistics::ranks;

/// **Mann–Whitney U** test (rank-sum) for whether two samples come from the same
/// distribution, using the normal approximation (suitable for moderate `n`).
pub fn mann_whitney_u(a: &[f64], b: &[f64]) -> TestResult {
    let (na, nb) = (a.len(), b.len());
    if na == 0 || nb == 0 {
        return nan_result();
    }
    // Rank the pooled samples; U from the rank sum of `a`.
    let mut pooled = Vec::with_capacity(na + nb);
    pooled.extend_from_slice(a);
    pooled.extend_from_slice(b);
    let r = ranks(&pooled);
    let rank_sum_a: f64 = r[..na].iter().sum();
    let (na_f, nb_f) = (na as f64, nb as f64);
    let u_a = rank_sum_a - na_f * (na_f + 1.0) / 2.0;
    let u = u_a.min(na_f * nb_f - u_a);
    // Normal approximation.
    let mu = na_f * nb_f / 2.0;
    let sigma = (na_f * nb_f * (na_f + nb_f + 1.0) / 12.0).sqrt();
    let z = (u - mu) / sigma;
    TestResult {
        statistic: u,
        p_value: z_two_sided(z),
    }
}

/// **Wilcoxon signed-rank** test for paired samples `a`, `b` (the non-parametric
/// analogue of the paired t-test). Zero differences are dropped; the p-value uses
/// the normal approximation. `a` and `b` must have equal length.
pub fn wilcoxon_signed_rank(a: &[f64], b: &[f64]) -> TestResult {
    if a.len() != b.len() || a.is_empty() {
        return nan_result();
    }
    let diffs: Vec<f64> = a
        .iter()
        .zip(b)
        .map(|(x, y)| x - y)
        .filter(|d| *d != 0.0)
        .collect();
    let n = diffs.len();
    if n == 0 {
        return nan_result();
    }
    let abs: Vec<f64> = diffs.iter().map(|d| d.abs()).collect();
    let r = ranks(&abs);
    let w_plus: f64 = r
        .iter()
        .zip(&diffs)
        .filter(|(_, d)| **d > 0.0)
        .map(|(rk, _)| rk)
        .sum();
    let w_minus: f64 = r
        .iter()
        .zip(&diffs)
        .filter(|(_, d)| **d < 0.0)
        .map(|(rk, _)| rk)
        .sum();
    let w = w_plus.min(w_minus);
    let nf = n as f64;
    let mu = nf * (nf + 1.0) / 4.0;
    let sigma = (nf * (nf + 1.0) * (2.0 * nf + 1.0) / 24.0).sqrt();
    let z = (w - mu) / sigma;
    TestResult {
        statistic: w,
        p_value: z_two_sided(z),
    }
}

/// Exact two-sided sign test for paired samples, dropping zero differences.
pub fn sign_test(a: &[f64], b: &[f64]) -> TestResult {
    if a.len() != b.len() || a.is_empty() {
        return nan_result();
    }
    let mut positive = 0u64;
    let mut nonzero = 0u64;
    for (&x, &y) in a.iter().zip(b) {
        let d = x - y;
        if d > 0.0 {
            positive += 1;
            nonzero += 1;
        } else if d < 0.0 {
            nonzero += 1;
        }
    }
    if nonzero == 0 {
        return nan_result();
    }
    TestResult {
        statistic: positive as f64,
        p_value: binomial_two_sided_p_value(positive, nonzero, 0.5),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn mann_whitney_separates_shifted_samples() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [11.0, 12.0, 13.0, 14.0, 15.0];
        assert!(mann_whitney_u(&a, &b).p_value < 0.05);
    }

    #[test]
    fn wilcoxon_detects_consistent_difference() {
        let a = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0];
        let b: Vec<f64> = a.iter().map(|x| x - 2.0).collect();
        assert!(wilcoxon_signed_rank(&a, &b).p_value < 0.05);
        // Identical samples ⇒ all-zero diffs ⇒ NaN (no signed ranks).
        assert!(wilcoxon_signed_rank(&a, &a).p_value.is_nan());
    }

    #[test]
    fn sign_test_uses_exact_binomial_tail() {
        let a = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
        let b = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0];
        assert!(sign_test(&a, &b).p_value < 0.05);
        assert!(sign_test(&a, &a).p_value.is_nan());
    }
}
