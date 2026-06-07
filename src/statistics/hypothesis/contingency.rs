//! Contingency-table tests: **χ² goodness-of-fit**, **χ² independence**, **Fisher's
//! exact**, the **exact binomial** test, and **McNemar's** test.

use super::{TestResult, binomial_two_sided_p_value, chi2_upper, nan_result};
use crate::function::factorial::ln_binomial;

fn valid_table(table: &[Vec<f64>]) -> Option<(usize, usize)> {
    let r = table.len();
    let c = table.first().map_or(0, Vec::len);
    if r < 2
        || c < 2
        || table.iter().any(|row| row.len() != c)
        || table.iter().flatten().any(|x| !x.is_finite() || *x < 0.0)
    {
        return None;
    }
    Some((r, c))
}

/// **Chi-squared goodness-of-fit** test: observed counts vs expected counts.
/// `df = k - 1`. Returns the upper-tail p-value.
pub fn chi_squared_goodness_of_fit(observed: &[f64], expected: &[f64]) -> TestResult {
    if observed.len() != expected.len()
        || observed.len() < 2
        || observed.iter().any(|x| !x.is_finite() || *x < 0.0)
        || expected.iter().any(|x| !x.is_finite() || *x <= 0.0)
    {
        return nan_result();
    }
    let chi2: f64 = observed
        .iter()
        .zip(expected)
        .map(|(&o, &e)| (o - e) * (o - e) / e)
        .sum();
    TestResult {
        statistic: chi2,
        p_value: chi2_upper(chi2, observed.len() as f64 - 1.0),
    }
}

/// **Chi-squared test of independence** on a contingency `table` (rows × cols of
/// counts). Expected counts come from the row/column margins; `df = (r-1)(c-1)`.
pub fn chi_squared_independence(table: &[Vec<f64>]) -> TestResult {
    let Some((r, c)) = valid_table(table) else {
        return nan_result();
    };
    let row_sums: Vec<f64> = table.iter().map(|row| row.iter().sum()).collect();
    let col_sums: Vec<f64> = (0..c)
        .map(|j| table.iter().map(|row| row[j]).sum())
        .collect();
    let total: f64 = row_sums.iter().sum();
    if total <= 0.0 {
        return nan_result();
    }
    let mut chi2 = 0.0;
    for i in 0..r {
        for j in 0..c {
            let e = row_sums[i] * col_sums[j] / total;
            if e > 0.0 {
                chi2 += (table[i][j] - e) * (table[i][j] - e) / e;
            }
        }
    }
    let df = (r as f64 - 1.0) * (c as f64 - 1.0);
    TestResult {
        statistic: chi2,
        p_value: chi2_upper(chi2, df),
    }
}

/// **Fisher's exact test** for a 2×2 contingency table `[[a, b], [c, d]]`. The
/// statistic is the sample odds ratio `ad/bc`; the two-sided p-value sums the
/// hypergeometric probabilities of all tables (with the same margins) no more likely
/// than the observed one. Exact and closed-form — appropriate for small counts.
pub fn fishers_exact(a: u64, b: u64, c: u64, d: u64) -> TestResult {
    let r1 = a + b;
    let r2 = c + d;
    let c1 = a + c;
    let n = r1 + r2;
    if n == 0 {
        return nan_result();
    }
    let log_norm = ln_binomial(n, c1);
    let ln_p = |k: u64| ln_binomial(r1, k) + ln_binomial(r2, c1 - k) - log_norm;
    let p_obs = ln_p(a).exp();
    let k_min = c1.saturating_sub(r2);
    let k_max = r1.min(c1);
    let mut p_value = 0.0;
    for k in k_min..=k_max {
        let pk = ln_p(k).exp();
        if pk <= p_obs * (1.0 + 1e-7) {
            p_value += pk;
        }
    }
    let odds = if b == 0 || c == 0 {
        f64::INFINITY
    } else {
        (a as f64 * d as f64) / (b as f64 * c as f64)
    };
    TestResult {
        statistic: odds,
        p_value: p_value.min(1.0),
    }
}

/// Exact two-sided binomial test for `successes ~ Binomial(n, p0)`.
pub fn binomial_test(successes: u64, n: u64, p0: f64) -> TestResult {
    if successes > n || p0.is_nan() || !(0.0..=1.0).contains(&p0) {
        return nan_result();
    }
    TestResult {
        statistic: successes as f64,
        p_value: binomial_two_sided_p_value(successes, n, p0),
    }
}

/// McNemar's test for paired binary outcomes. Inputs are the discordant counts:
/// `b` for `A=1,B=0` and `c` for `A=0,B=1`.
pub fn mcnemar(b: u64, c: u64) -> TestResult {
    let n = b + c;
    if n == 0 {
        return nan_result();
    }
    let diff = b.abs_diff(c) as f64;
    let statistic = (diff - 1.0).max(0.0).powi(2) / n as f64;
    TestResult {
        statistic,
        p_value: binomial_two_sided_p_value(b, n, 0.5),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn chi_squared_fair_die() {
        let observed = [10.0, 11.0, 9.0, 10.0, 12.0, 8.0];
        let expected = [10.0; 6];
        let res = chi_squared_goodness_of_fit(&observed, &expected);
        assert!(res.p_value > 0.5);
        let loaded = [2.0, 2.0, 2.0, 2.0, 2.0, 50.0];
        assert!(chi_squared_goodness_of_fit(&loaded, &[10.0; 6]).p_value < 1e-6);
    }

    #[test]
    fn independence_of_a_balanced_table() {
        let table = vec![vec![10.0, 20.0], vec![20.0, 40.0]];
        let res = chi_squared_independence(&table);
        assert!(res.statistic.abs() < 1e-9);
        assert!(res.p_value > 0.99);
    }

    #[test]
    fn fishers_exact_known_values() {
        close(fishers_exact(3, 1, 1, 3).p_value, 34.0 / 70.0, 1e-9);
        assert!(fishers_exact(10, 0, 0, 10).p_value < 0.01);
        close(fishers_exact(5, 5, 5, 5).p_value, 1.0, 1e-9);
    }

    #[test]
    fn malformed_tables_return_nan() {
        assert!(
            chi_squared_independence(&[vec![1.0, 2.0], vec![3.0]])
                .p_value
                .is_nan()
        );
        assert!(
            chi_squared_independence(&[vec![1.0, -2.0], vec![3.0, 4.0]])
                .p_value
                .is_nan()
        );
        assert!(
            chi_squared_goodness_of_fit(&[1.0, 2.0], &[1.0, 0.0])
                .p_value
                .is_nan()
        );
    }

    #[test]
    fn exact_binomial_and_mcnemar_tests() {
        assert!(binomial_test(9, 10, 0.5).p_value < 0.05);
        assert!(binomial_test(5, 10, 0.5).p_value > 0.9);
        assert!(mcnemar(12, 2).p_value < 0.05);
        assert!(mcnemar(5, 5).p_value > 0.9);
    }
}
