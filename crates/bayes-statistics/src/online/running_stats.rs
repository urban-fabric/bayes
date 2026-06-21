//! [`RunningStats`] — a single-pass, mergeable accumulator for mean, variance,
//! skewness, kurtosis, and extrema, via Welford's running-moment recurrences
//! (Terriberry's extension to the third and fourth moments).

/// A running accumulator of summary statistics over a stream of `f64` values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunningStats {
    n: u64,
    mean: f64,
    m2: f64,
    m3: f64,
    m4: f64,
    min: f64,
    max: f64,
}

impl Default for RunningStats {
    fn default() -> Self {
        Self::new()
    }
}

impl RunningStats {
    pub fn new() -> Self {
        Self {
            n: 0,
            mean: 0.0,
            m2: 0.0,
            m3: 0.0,
            m4: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Absorb one observation.
    pub fn push(&mut self, x: f64) {
        let n1 = self.n as f64;
        self.n += 1;
        let n = self.n as f64;
        let delta = x - self.mean;
        let delta_n = delta / n;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * n1;
        self.mean += delta_n;
        self.m4 += term1 * delta_n2 * (n * n - 3.0 * n + 3.0) + 6.0 * delta_n2 * self.m2
            - 4.0 * delta_n * self.m3;
        self.m3 += term1 * delta_n * (n - 2.0) - 3.0 * delta_n * self.m2;
        self.m2 += term1;
        self.min = self.min.min(x);
        self.max = self.max.max(x);
    }

    /// Combine another accumulator into this one (for parallel/streamed reduction),
    /// via Chan et al.'s parallel update. The result equals a single pass over the
    /// pooled data, so `merge` is associative and commutative with
    /// [`new`](Self::new) as identity. The mean and variance (`M2`) merge is proved
    /// correct in `riemann` (`RiemannLean/Bayes/Streaming.lean`: `merge_mean`,
    /// `merge_m2`); the `m3`/`m4` (skewness/kurtosis) merge is Terriberry's
    /// higher-order extension, validated against the single pass by test.
    pub fn merge(&mut self, other: &Self) {
        if other.n == 0 {
            return;
        }
        if self.n == 0 {
            *self = *other;
            return;
        }
        let (na, nb) = (self.n as f64, other.n as f64);
        let n = na + nb;
        let delta = other.mean - self.mean;
        let delta2 = delta * delta;
        let delta3 = delta2 * delta;
        let delta4 = delta2 * delta2;
        let mean = self.mean + delta * nb / n;
        let m2 = self.m2 + other.m2 + delta2 * na * nb / n;
        let m3 = self.m3
            + other.m3
            + delta3 * na * nb * (na - nb) / (n * n)
            + 3.0 * delta * (na * other.m2 - nb * self.m2) / n;
        let m4 = self.m4
            + other.m4
            + delta4 * na * nb * (na * na - na * nb + nb * nb) / (n * n * n)
            + 6.0 * delta2 * (na * na * other.m2 + nb * nb * self.m2) / (n * n)
            + 4.0 * delta * (na * other.m3 - nb * self.m3) / n;
        *self = Self {
            n: self.n + other.n,
            mean,
            m2,
            m3,
            m4,
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        };
    }

    pub fn count(&self) -> u64 {
        self.n
    }
    pub fn mean(&self) -> f64 {
        if self.n == 0 { f64::NAN } else { self.mean }
    }
    /// Unbiased sample variance (`n-1`).
    pub fn variance(&self) -> f64 {
        if self.n < 2 {
            f64::NAN
        } else {
            self.m2 / (self.n as f64 - 1.0)
        }
    }
    /// Population (biased) variance (`n`).
    pub fn population_variance(&self) -> f64 {
        if self.n == 0 {
            f64::NAN
        } else {
            self.m2 / self.n as f64
        }
    }
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
    /// Population skewness (third standardized moment).
    pub fn skewness(&self) -> f64 {
        if self.n < 2 || self.m2 <= 0.0 {
            return f64::NAN;
        }
        (self.n as f64).sqrt() * self.m3 / self.m2.powf(1.5)
    }
    /// Excess kurtosis (fourth standardized moment − 3).
    pub fn kurtosis(&self) -> f64 {
        if self.n < 2 || self.m2 <= 0.0 {
            return f64::NAN;
        }
        self.n as f64 * self.m4 / (self.m2 * self.m2) - 3.0
    }
    pub fn min(&self) -> f64 {
        if self.n == 0 { f64::NAN } else { self.min }
    }
    pub fn max(&self) -> f64 {
        if self.n == 0 { f64::NAN } else { self.max }
    }
}

impl Extend<f64> for RunningStats {
    fn extend<I: IntoIterator<Item = f64>>(&mut self, iter: I) {
        for x in iter {
            self.push(x);
        }
    }
}

impl FromIterator<f64> for RunningStats {
    fn from_iter<I: IntoIterator<Item = f64>>(iter: I) -> Self {
        let mut s = Self::new();
        s.extend(iter);
        s
    }
}

impl std::iter::Sum for RunningStats {
    /// Reduce a collection of partial accumulators into one by [`merge`](Self::merge)
    /// — the map-reduce path: build a `RunningStats` per shard, then `.sum()` them.
    /// Because `merge` equals a single pass over the pooled data (`riemann`'s
    /// `merge_mean`/`merge_m2`), the reduction is order-independent and agrees with
    /// accumulating the whole stream at once.
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::new(), |mut acc, s| {
            acc.merge(&s);
            acc
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::summary::{kurtosis, mean, skewness, variance};

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn matches_the_batch_statistics() {
        let data: Vec<f64> = (0..500)
            .map(|i| (i as f64 * 0.31).sin() * 10.0 + 3.0)
            .collect();
        let rs: RunningStats = data.iter().copied().collect();
        close(rs.mean(), mean(&data), 1e-9);
        close(rs.variance(), variance(&data), 1e-7);
        close(rs.skewness(), skewness(&data), 1e-6);
        close(rs.kurtosis(), kurtosis(&data), 1e-6);
        assert_eq!(rs.count(), 500);
    }

    #[test]
    fn merge_equals_one_pass() {
        let data: Vec<f64> = (0..400).map(|i| i as f64).collect();
        let whole: RunningStats = data.iter().copied().collect();
        let mut a: RunningStats = data[..137].iter().copied().collect();
        let b: RunningStats = data[137..].iter().copied().collect();
        a.merge(&b);
        close(a.mean(), whole.mean(), 1e-9);
        close(a.variance(), whole.variance(), 1e-7);
        close(a.skewness(), whole.skewness(), 1e-6);
        close(a.kurtosis(), whole.kurtosis(), 1e-6);
    }

    #[test]
    fn sum_of_shards_equals_one_pass() {
        // Map-reduce: shard the data, accumulate each, then `.sum()` the parts.
        let data: Vec<f64> = (0..400)
            .map(|i| (i as f64 * 0.17).cos() * 5.0 - 1.0)
            .collect();
        let whole: RunningStats = data.iter().copied().collect();
        let shards = [&data[..91], &data[91..210], &data[210..210], &data[210..]];
        let merged: RunningStats = shards
            .iter()
            .map(|s| s.iter().copied().collect::<RunningStats>())
            .sum();
        close(merged.mean(), whole.mean(), 1e-9);
        close(merged.variance(), whole.variance(), 1e-7); // merge_mean / merge_m2
        close(merged.skewness(), whole.skewness(), 1e-6);
        close(merged.kurtosis(), whole.kurtosis(), 1e-6);
        assert_eq!(merged.count(), 400);
    }

    #[test]
    fn extrema_and_empty() {
        let mut rs = RunningStats::new();
        assert!(rs.mean().is_nan());
        rs.extend([3.0, -1.0, 7.0, 2.0]);
        assert_eq!(rs.min(), -1.0);
        assert_eq!(rs.max(), 7.0);
    }
}
