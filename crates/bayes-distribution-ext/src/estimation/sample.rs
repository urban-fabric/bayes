//! The sample summaries every closed-form fit is built from.
//!
//! Each constructor here does the validation once, so the estimators can be the
//! arithmetic they actually are. The rule throughout: **return `None` rather
//! than a parameter the data does not support**, because every caller is about
//! to hand the result to a constructor that will reject it anyway, with a
//! message about a parameter rather than about the sample.

/// Mean, variance and range of a sample, in one pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Summary {
    /// The sample size, as a float because every formula divides by it.
    pub n: f64,
    pub mean: f64,
    /// The **population** variance, dividing by `n` rather than `n − 1`.
    ///
    /// This is the right choice for method-of-moments fitting even though it is
    /// the biased estimator of `σ²`: the moment equations are written in terms
    /// of the distribution's central moments, and `Σ(x − x̄)²/n` is the sample
    /// analogue of `E[(X − μ)²]`. Switching to `n − 1` would make the resulting
    /// parameter estimates inconsistent with the equations they solve.
    pub variance: f64,
    pub min: f64,
    pub max: f64,
}

impl Summary {
    /// `None` for fewer than two points, or if anything is not finite.
    ///
    /// Non-finite input is rejected up front because arithmetic would otherwise
    /// carry it straight into a parameter: one `NaN` in the sample makes the
    /// mean `NaN`, which makes every fitted parameter `NaN`, which a
    /// constructor rejects while naming the parameter rather than the cause.
    pub fn new(data: &[f64]) -> Option<Self> {
        if data.len() < 2 || data.iter().any(|x| !x.is_finite()) {
            return None;
        }
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let variance = data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n;
        let min = data.iter().copied().fold(f64::INFINITY, f64::min);
        let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Some(Self {
            n,
            mean,
            variance,
            min,
            max,
        })
    }

    /// The same summaries, but the **spread must be real**.
    ///
    /// Deliberately not `variance > 0`. Summing `n` copies of a value and
    /// dividing by `n` need not return that value exactly, so a genuinely
    /// constant sample comes back with a variance of about `1e-32` — positive,
    /// and nothing but rounding noise. A `Weibull` fitted to it gets a shape of
    /// `1e16`.
    ///
    /// So the bar is the noise a naive sum of `n` terms can manufacture,
    /// `n · ε · |mean|`, which is the same "is this scale below one ulp at this
    /// location" question the conformance suite asks of a distribution's own
    /// parameters. At the origin there is no such obstruction and any positive
    /// variance counts.
    pub fn with_spread(data: &[f64]) -> Option<Self> {
        let s = Self::new(data)?;
        let noise = s.n * f64::EPSILON * s.mean.abs();
        (s.variance > 0.0 && s.std_dev() > noise).then_some(s)
    }

    /// The summaries of `ln x`, for the families that are a location-scale
    /// family *after* taking logs — log-normal, Weibull, Frechet.
    ///
    /// `None` unless every observation is strictly positive, since otherwise
    /// the log is not defined and the family has no mass there anyway.
    pub fn of_logs(data: &[f64]) -> Option<Self> {
        if data.iter().any(|&x| x <= 0.0) {
            return None;
        }
        let logs: Vec<f64> = data.iter().map(|x| x.ln()).collect();
        Self::with_spread(&logs)
    }

    /// `√variance`.
    pub fn std_dev(&self) -> f64 {
        self.variance.sqrt()
    }

    /// The raw second moment `Σx²/n = variance + mean²`, for the families
    /// parameterized about a fixed origin rather than about their own mean.
    pub fn second_moment(&self) -> f64 {
        self.variance + self.mean * self.mean
    }
}

/// The **excess kurtosis** of a sample, `m₄/m₂² − 3`.
///
/// Separate from [`Summary`] because it needs a second pass and only one
/// estimator wants it. Carries a health warning: the sampling error of a fourth
/// moment is large, so a parameter derived from this is far noisier than one
/// derived from a mean or a variance.
pub fn excess_kurtosis(data: &[f64]) -> Option<f64> {
    let s = Summary::with_spread(data)?;
    let m4 = data.iter().map(|&x| (x - s.mean).powi(4)).sum::<f64>() / s.n;
    let value = m4 / (s.variance * s.variance) - 3.0;
    value.is_finite().then_some(value)
}

/// A quantile of `data` by linear interpolation, for the families that have no
/// moments to match.
///
/// Sorts a copy, so it costs `O(n log n)` where the rest of this module costs
/// `O(n)`. Only [`Cauchy`](crate::Cauchy) needs it — and it needs it precisely
/// because its mean and variance do not exist, which makes an order statistic
/// the only consistent thing available.
pub fn quantile(data: &[f64], p: f64) -> Option<f64> {
    if data.is_empty() || data.iter().any(|x| !x.is_finite()) || !(0.0..=1.0).contains(&p) {
        return None;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let position = p * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let weight = position - lower as f64;
    Some(sorted[lower] * (1.0 - weight) + sorted[upper] * weight)
}

/// Counts of the integers `0..=max` appearing in `data`, for
/// [`Categorical`](crate::Categorical).
///
/// `None` unless every observation is a non-negative integer, which is what
/// that family's support is.
pub fn category_counts(data: &[f64]) -> Option<Vec<f64>> {
    if data.is_empty() {
        return None;
    }
    let mut max = 0_usize;
    for &x in data {
        if !x.is_finite() || x < 0.0 || x.fract() != 0.0 || x > 1e6 {
            return None;
        }
        max = max.max(x as usize);
    }
    let mut counts = vec![0.0; max + 1];
    for &x in data {
        counts[x as usize] += 1.0;
    }
    Some(counts)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn summaries_are_what_they_say() {
        let s = Summary::new(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(s.n, 4.0);
        assert_eq!(s.mean, 2.5);
        // Population variance: (2.25 + 0.25 + 0.25 + 2.25)/4 = 1.25.
        assert_eq!(s.variance, 1.25);
        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 4.0);
        assert_eq!(s.second_moment(), 1.25 + 6.25);
    }

    #[test]
    fn degenerate_samples_are_refused_rather_than_fitted() {
        assert!(Summary::new(&[]).is_none());
        assert!(Summary::new(&[1.0]).is_none(), "one point has no variance");
        assert!(Summary::new(&[1.0, f64::NAN]).is_none());
        assert!(Summary::new(&[1.0, f64::INFINITY]).is_none());
        // A constant sample has a mean but no scale.
        assert!(Summary::new(&[2.0, 2.0, 2.0]).is_some());
        assert!(Summary::with_spread(&[2.0, 2.0, 2.0]).is_none());
        // Logs need positive data.
        assert!(Summary::of_logs(&[1.0, 2.0, 0.0]).is_none());
        assert!(Summary::of_logs(&[1.0, 2.0, -3.0]).is_none());
        assert!(Summary::of_logs(&[1.0, 2.0, 3.0]).is_some());
    }

    #[test]
    fn log_summaries_are_the_summaries_of_the_logs() {
        let data = [
            1.0,
            std::f64::consts::E,
            std::f64::consts::E * std::f64::consts::E,
        ];
        let s = Summary::of_logs(&data).unwrap();
        // ln of the sample is 0, 1, 2.
        assert!((s.mean - 1.0).abs() < 1e-12);
        assert!((s.variance - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn quantiles_interpolate_and_hit_the_ends() {
        let data = [4.0, 1.0, 3.0, 2.0];
        assert_eq!(quantile(&data, 0.0).unwrap(), 1.0);
        assert_eq!(quantile(&data, 1.0).unwrap(), 4.0);
        assert_eq!(quantile(&data, 0.5).unwrap(), 2.5);
        assert!(quantile(&data, 1.5).is_none());
        assert!(quantile(&[], 0.5).is_none());
    }

    #[test]
    fn excess_kurtosis_is_zero_for_a_normal_ish_sample_and_refuses_constants() {
        assert!(excess_kurtosis(&[1.0, 1.0, 1.0]).is_none());
        // A two-point symmetric sample has m4/m2² = 1, so excess kurtosis −2 —
        // the minimum possible, which is a useful sanity anchor.
        let k = excess_kurtosis(&[-1.0, 1.0]).unwrap();
        assert!((k + 2.0).abs() < 1e-12, "got {k}");
    }

    #[test]
    fn category_counts_require_non_negative_integers() {
        assert_eq!(
            category_counts(&[0.0, 2.0, 2.0]).unwrap(),
            vec![1.0, 0.0, 2.0]
        );
        assert!(category_counts(&[0.0, 1.5]).is_none());
        assert!(category_counts(&[-1.0]).is_none());
        assert!(category_counts(&[]).is_none());
    }
}
