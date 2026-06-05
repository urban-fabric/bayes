//! Distribution-fit tests: **Kolmogorov–Smirnov** (one/two-sample) and the
//! **Jarque–Bera** normality test.

use super::{TestResult, chi2_upper, nan_result};
use crate::statistics::{Ecdf, kurtosis, skewness};

/// Asymptotic Kolmogorov distribution survival
/// `Q(λ) = 2 Σ_{k≥1} (-1)^{k-1} e^{-2k²λ²}` — the limiting p-value for a scaled KS
/// statistic. Closed-form (a rapidly converging series).
fn kolmogorov_sf(lambda: f64) -> f64 {
    if lambda <= 0.0 {
        return 1.0;
    }
    let mut sum = 0.0;
    for k in 1..=100i32 {
        let term = (-2.0 * (k * k) as f64 * lambda * lambda).exp();
        sum += if k % 2 == 1 { term } else { -term };
        if term < 1e-12 {
            break;
        }
    }
    (2.0 * sum).clamp(0.0, 1.0)
}

/// One-sample **Kolmogorov–Smirnov** goodness-of-fit test: the largest gap between
/// the empirical CDF of `data` and a reference `cdf`. The p-value uses the
/// asymptotic Kolmogorov distribution (Stephens's small-sample correction).
pub fn ks_one_sample(data: &[f64], cdf: impl Fn(f64) -> f64) -> TestResult {
    let ecdf = Ecdf::new(data);
    let sorted = ecdf.sorted();
    let n = sorted.len();
    if n == 0 {
        return nan_result();
    }
    let nf = n as f64;
    let mut d: f64 = 0.0;
    for (i, &x) in sorted.iter().enumerate() {
        let f = cdf(x);
        let d_plus = (i as f64 + 1.0) / nf - f;
        let d_minus = f - i as f64 / nf;
        d = d.max(d_plus).max(d_minus);
    }
    let en = nf.sqrt();
    let lambda = (en + 0.12 + 0.11 / en) * d;
    TestResult {
        statistic: d,
        p_value: kolmogorov_sf(lambda),
    }
}

/// Two-sample **Kolmogorov–Smirnov** test: do `a` and `b` come from the same
/// distribution? The statistic is `sup |F̂_a − F̂_b|`, evaluated over the pooled
/// support; the p-value uses the effective size `n₁n₂/(n₁+n₂)`.
pub fn ks_two_sample(a: &[f64], b: &[f64]) -> TestResult {
    if a.is_empty() || b.is_empty() {
        return nan_result();
    }
    let (ea, eb) = (Ecdf::new(a), Ecdf::new(b));
    let mut points: Vec<f64> = a.iter().chain(b).copied().filter(|x| !x.is_nan()).collect();
    points.sort_by(f64::total_cmp);
    let mut d: f64 = 0.0;
    for &x in &points {
        d = d.max((ea.eval(x) - eb.eval(x)).abs());
    }
    let (n1, n2) = (a.len() as f64, b.len() as f64);
    let en = (n1 * n2 / (n1 + n2)).sqrt();
    let lambda = (en + 0.12 + 0.11 / en) * d;
    TestResult {
        statistic: d,
        p_value: kolmogorov_sf(lambda),
    }
}

/// **Jarque–Bera** normality test: `JB = n/6·(S² + K²/4)` from the sample skewness
/// `S` and excess kurtosis `K`, compared to a χ² with 2 degrees of freedom.
pub fn jarque_bera(data: &[f64]) -> TestResult {
    if data.len() < 2 {
        return nan_result();
    }
    let n = data.len() as f64;
    let s = skewness(data);
    let k = kurtosis(data); // excess kurtosis
    let jb = n / 6.0 * (s * s + k * k / 4.0);
    TestResult {
        statistic: jb,
        p_value: chi2_upper(jb, 2.0),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn jarque_bera_separates_symmetric_from_skewed() {
        let sym: Vec<f64> = (-10..=10).map(|i| i as f64).collect();
        let skew: Vec<f64> = (1..=20).map(|i| (i * i) as f64).collect();
        assert!(jarque_bera(&sym).p_value > 0.05);
        assert!(jarque_bera(&skew).p_value < jarque_bera(&sym).p_value);
    }

    #[test]
    fn ks_one_sample_matches_uniform() {
        let data: Vec<f64> = (1..10).map(|i| i as f64 / 10.0).collect(); // 0.1..0.9
        // Against the true Uniform(0,1) CDF: small D, large p.
        let good = ks_one_sample(&data, |x| x.clamp(0.0, 1.0));
        assert!(good.p_value > 0.2);
        // Against a badly mismatched CDF: larger D, smaller p.
        let bad = ks_one_sample(&data, |x| (x - 5.0).clamp(0.0, 1.0));
        assert!(bad.statistic > good.statistic);
        assert!(bad.p_value < good.p_value);
    }

    #[test]
    fn ks_two_sample_detects_shift() {
        let a: Vec<f64> = (0..20).map(|i| i as f64 / 20.0).collect();
        let same: Vec<f64> = (0..20).map(|i| i as f64 / 20.0 + 0.001).collect();
        let shifted: Vec<f64> = a.iter().map(|x| x + 5.0).collect();
        assert!(ks_two_sample(&a, &same).p_value > 0.2);
        assert!(ks_two_sample(&a, &shifted).p_value < 0.01);
    }
}
