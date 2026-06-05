//! Resampling methods: the **bootstrap** (resample with replacement) and the
//! **jackknife** (leave-one-out) — distribution-free ways to assess the
//! variability of a statistic without assuming a parametric model.

use rand::{Rng, RngExt};

use super::order::quantile;

/// The **bootstrap distribution** of `statistic`: draw `resamples` resamples of
/// `data` (same size, sampled with replacement) and apply `statistic` to each.
/// The spread of the returned values estimates the statistic's sampling
/// variability; feed it to [`percentile_interval`] for a confidence interval.
pub fn bootstrap<R, F>(data: &[f64], resamples: usize, statistic: F, rng: &mut R) -> Vec<f64>
where
    R: Rng + ?Sized,
    F: Fn(&[f64]) -> f64,
{
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }
    let mut buf = vec![0.0; n];
    (0..resamples)
        .map(|_| {
            for slot in buf.iter_mut() {
                *slot = data[rng.random_range(0..n)];
            }
            statistic(&buf)
        })
        .collect()
}

/// The **jackknife** leave-one-out estimates of `statistic`: for each `i`, apply
/// `statistic` to `data` with element `i` removed. Returns one value per point.
pub fn jackknife<F>(data: &[f64], statistic: F) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let n = data.len();
    (0..n)
        .map(|i| {
            let mut without = Vec::with_capacity(n - 1);
            without.extend_from_slice(&data[..i]);
            without.extend_from_slice(&data[i + 1..]);
            statistic(&without)
        })
        .collect()
}

/// A two-sided **percentile confidence interval** at confidence `level` (e.g.
/// `0.95`) from a set of resampled statistics.
pub fn percentile_interval(samples: &[f64], level: f64) -> (f64, f64) {
    let alpha = (1.0 - level.clamp(0.0, 1.0)) / 2.0;
    (quantile(samples, alpha), quantile(samples, 1.0 - alpha))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::super::summary::mean;
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn bootstrap_mean_centers_on_the_sample_mean() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let data: Vec<f64> = (0..200).map(|i| i as f64).collect();
        let boot = bootstrap(&data, 2000, mean, &mut rng);
        // The bootstrap distribution of the mean is centered on the sample mean.
        assert!((mean(&boot) - mean(&data)).abs() < 2.0);
        let (lo, hi) = percentile_interval(&boot, 0.95);
        assert!(lo < mean(&data) && mean(&data) < hi);
    }

    #[test]
    fn jackknife_leaves_one_out() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let loo = jackknife(&data, mean);
        assert_eq!(loo.len(), 4);
        // Dropping element 0 (value 1) ⇒ mean of {2,3,4} = 3.
        assert!((loo[0] - 3.0).abs() < 1e-12);
    }
}
