//! Parametric mean tests: one-/two-sample **z** and **t**, **Welch's t**, and
//! **one-way ANOVA**.

use super::{TestResult, nan_result, t_two_sided, z_two_sided};
use crate::distribution::{ContinuousCdf, FisherSnedecor};
use crate::statistics::{mean, variance};

/// One-sample **z-test** that the mean of `data` equals `mu0`, with a *known*
/// population standard deviation `sigma`.
pub fn one_sample_z(data: &[f64], mu0: f64, sigma: f64) -> TestResult {
    let n = data.len() as f64;
    let z = (mean(data) - mu0) / (sigma / n.sqrt());
    TestResult {
        statistic: z,
        p_value: z_two_sided(z),
    }
}

/// One-sample **t-test** that the mean of `data` equals `mu0` (population standard
/// deviation unknown, estimated from the sample).
pub fn one_sample_t(data: &[f64], mu0: f64) -> TestResult {
    let n = data.len() as f64;
    let s = variance(data).sqrt();
    let t = (mean(data) - mu0) / (s / n.sqrt());
    TestResult {
        statistic: t,
        p_value: t_two_sided(t, n - 1.0),
    }
}

/// Two-sample **pooled t-test** for equal means, assuming equal variances.
pub fn two_sample_t(a: &[f64], b: &[f64]) -> TestResult {
    let (na, nb) = (a.len() as f64, b.len() as f64);
    let (va, vb) = (variance(a), variance(b));
    let sp2 = ((na - 1.0) * va + (nb - 1.0) * vb) / (na + nb - 2.0);
    let t = (mean(a) - mean(b)) / (sp2 * (1.0 / na + 1.0 / nb)).sqrt();
    TestResult {
        statistic: t,
        p_value: t_two_sided(t, na + nb - 2.0),
    }
}

/// **Welch's t-test** for equal means with *unequal* variances (the
/// Welch–Satterthwaite degrees of freedom).
pub fn welch_t(a: &[f64], b: &[f64]) -> TestResult {
    let (na, nb) = (a.len() as f64, b.len() as f64);
    let (va, vb) = (variance(a), variance(b));
    let (sea, seb) = (va / na, vb / nb);
    let t = (mean(a) - mean(b)) / (sea + seb).sqrt();
    let df = (sea + seb).powi(2) / (sea * sea / (na - 1.0) + seb * seb / (nb - 1.0));
    TestResult {
        statistic: t,
        p_value: t_two_sided(t, df),
    }
}

/// **One-way ANOVA** F-test for equality of the means of `groups`. `F = MSB/MSW`
/// with `df = (k−1, N−k)`; the p-value is the upper tail of the F distribution.
pub fn one_way_anova(groups: &[&[f64]]) -> TestResult {
    let k = groups.len();
    let total_n: usize = groups.iter().map(|g| g.len()).sum();
    if k < 2 || total_n <= k {
        return nan_result();
    }
    let grand_mean = groups.iter().flat_map(|g| g.iter()).sum::<f64>() / total_n as f64;
    let mut ssb = 0.0;
    let mut ssw = 0.0;
    for g in groups {
        if g.is_empty() {
            continue;
        }
        let gm = g.iter().sum::<f64>() / g.len() as f64;
        ssb += g.len() as f64 * (gm - grand_mean).powi(2);
        ssw += g.iter().map(|&x| (x - gm).powi(2)).sum::<f64>();
    }
    let df1 = (k - 1) as f64;
    let df2 = (total_n - k) as f64;
    let f = (ssb / df1) / (ssw / df2);
    let p = match FisherSnedecor::new(df1, df2) {
        Ok(d) => d.sf(f),
        Err(_) => f64::NAN,
    };
    TestResult {
        statistic: f,
        p_value: p,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn z_test_detects_a_clear_shift() {
        // Data centered at 5, testing μ = 0 with σ = 1 ⇒ huge z, tiny p.
        let data: Vec<f64> = (0..30).map(|_| 5.0).collect();
        let res = one_sample_z(&data, 0.0, 1.0);
        assert!(res.statistic > 10.0);
        assert!(res.p_value < 1e-6);
    }

    #[test]
    fn t_test_does_not_reject_when_consistent() {
        // Symmetric data with mean 0 ⇒ large p (fail to reject μ = 0).
        let data = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let res = one_sample_t(&data, 0.0);
        assert!(res.p_value > 0.5);
    }

    #[test]
    fn welch_separates_distinct_means() {
        let a = [10.0, 11.0, 9.0, 10.5, 9.5];
        let b = [20.0, 21.0, 19.0, 20.5, 19.5];
        let res = welch_t(&a, &b);
        assert!(res.p_value < 0.01);
    }

    #[test]
    fn anova_separates_group_means() {
        let g1 = [1.0, 2.0, 3.0, 2.0, 1.0];
        let g2 = [1.5, 2.5, 1.0, 2.0, 3.0];
        let g3 = [2.0, 1.0, 3.0, 2.5, 1.5];
        assert!(one_way_anova(&[&g1, &g2, &g3]).p_value > 0.05);
        let g3_hi = [20.0, 21.0, 19.0, 22.0, 20.5];
        assert!(one_way_anova(&[&g1, &g2, &g3_hi]).p_value < 0.01);
    }
}
