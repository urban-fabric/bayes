//! [`RunningStats`] — a single-pass, mergeable accumulator for mean, variance,
//! skewness, kurtosis, and extrema, via Welford's running-moment recurrences
//! (Terriberry's extension to the third and fourth moments).

use euclid::{Compare, Real};

/// A running accumulator of summary statistics over a stream of [`Real`] values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunningStats<R: Real = f64> {
    n: u64,
    mean: R,
    m2: R,
    m3: R,
    m4: R,
    min: R,
    max: R,
}

impl<R> Default for RunningStats<R>
where
    R: Real + Compare,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<R> RunningStats<R>
where
    R: Real + Compare,
{
    pub fn new() -> Self {
        Self {
            n: 0,
            mean: R::zero(),
            m2: R::zero(),
            m3: R::zero(),
            m4: R::zero(),
            min: R::infinity(),
            max: R::neg_infinity(),
        }
    }

    /// Absorb one observation.
    pub fn push(&mut self, x: R) {
        let n1 = R::scalar(self.n as f64);
        self.n += 1;
        let n = R::scalar(self.n as f64);
        let delta = x.clone() - self.mean.clone();
        let delta_n = delta.clone() / n.clone();
        let delta_n2 = delta_n.clone() * delta_n.clone();
        let term1 = delta.clone() * delta_n.clone() * n1;
        self.mean = self.mean.clone() + delta_n.clone();
        self.m4 = self.m4.clone()
            + term1.clone()
                * delta_n2.clone()
                * (n.clone() * n.clone() - R::scalar(3.0) * n.clone() + R::scalar(3.0))
            + R::scalar(6.0) * delta_n2 * self.m2.clone()
            - R::scalar(4.0) * delta_n.clone() * self.m3.clone();
        self.m3 = self.m3.clone() + term1.clone() * delta_n.clone() * (n - R::scalar(2.0))
            - R::scalar(3.0) * delta_n.clone() * self.m2.clone();
        self.m2 = self.m2.clone() + term1;
        let new_min = x.lt(&self.min);
        let new_max = x.gt(&self.max);
        self.min = R::select(&new_min, x.clone(), self.min.clone());
        self.max = R::select(&new_max, x, self.max.clone());
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
            *self = other.clone();
            return;
        }
        let (na, nb) = (R::scalar(self.n as f64), R::scalar(other.n as f64));
        let n = na.clone() + nb.clone();
        let delta = other.mean.clone() - self.mean.clone();
        let delta2 = delta.clone() * delta.clone();
        let delta3 = delta2.clone() * delta.clone();
        let delta4 = delta2.clone() * delta2.clone();
        let mean = self.mean.clone() + delta.clone() * nb.clone() / n.clone();
        let m2 = self.m2.clone()
            + other.m2.clone()
            + delta2.clone() * na.clone() * nb.clone() / n.clone();
        let m3 = self.m3.clone()
            + other.m3.clone()
            + delta3 * na.clone() * nb.clone() * (na.clone() - nb.clone())
                / (n.clone() * n.clone())
            + R::scalar(3.0)
                * delta.clone()
                * (na.clone() * other.m2.clone() - nb.clone() * self.m2.clone())
                / n.clone();
        let m4 = self.m4.clone()
            + other.m4.clone()
            + delta4
                * na.clone()
                * nb.clone()
                * (na.clone() * na.clone() - na.clone() * nb.clone() + nb.clone() * nb.clone())
                / (n.clone() * n.clone() * n.clone())
            + R::scalar(6.0)
                * delta2
                * (na.clone() * na.clone() * other.m2.clone()
                    + nb.clone() * nb.clone() * self.m2.clone())
                / (n.clone() * n.clone())
            + R::scalar(4.0) * delta * (na.clone() * other.m3.clone() - nb * self.m3.clone()) / n;
        let other_min = other.min.lt(&self.min);
        let other_max = other.max.gt(&self.max);
        *self = Self {
            n: self.n + other.n,
            mean,
            m2,
            m3,
            m4,
            min: R::select(&other_min, other.min.clone(), self.min.clone()),
            max: R::select(&other_max, other.max.clone(), self.max.clone()),
        };
    }

    pub fn count(&self) -> u64 {
        self.n
    }
    pub fn mean(&self) -> R {
        if self.n == 0 {
            R::nan()
        } else {
            self.mean.clone()
        }
    }
    /// Unbiased sample variance (`n-1`).
    pub fn variance(&self) -> R {
        if self.n < 2 {
            R::nan()
        } else {
            self.m2.clone() / R::scalar(self.n as f64 - 1.0)
        }
    }
    /// Population (biased) variance (`n`).
    pub fn population_variance(&self) -> R {
        if self.n == 0 {
            R::nan()
        } else {
            self.m2.clone() / R::scalar(self.n as f64)
        }
    }
    pub fn std_dev(&self) -> R {
        self.variance().sqrt()
    }
    /// Population skewness (third standardized moment).
    pub fn skewness(&self) -> R {
        if self.n < 2 {
            return R::nan();
        }
        let positive = self.m2.gt(&R::zero());
        let value = R::scalar(self.n as f64).sqrt() * self.m3.clone()
            / self.m2.clone().powf(R::scalar(1.5));
        R::select(&positive, value, R::nan())
    }
    /// Excess kurtosis (fourth standardized moment − 3).
    pub fn kurtosis(&self) -> R {
        if self.n < 2 {
            return R::nan();
        }
        let positive = self.m2.gt(&R::zero());
        let value = R::scalar(self.n as f64) * self.m4.clone()
            / (self.m2.clone() * self.m2.clone())
            - R::scalar(3.0);
        R::select(&positive, value, R::nan())
    }
    pub fn min(&self) -> R {
        if self.n == 0 {
            R::nan()
        } else {
            self.min.clone()
        }
    }
    pub fn max(&self) -> R {
        if self.n == 0 {
            R::nan()
        } else {
            self.max.clone()
        }
    }
}

impl<R> Extend<R> for RunningStats<R>
where
    R: Real + Compare,
{
    fn extend<I: IntoIterator<Item = R>>(&mut self, iter: I) {
        for x in iter {
            self.push(x);
        }
    }
}

impl<R> FromIterator<R> for RunningStats<R>
where
    R: Real + Compare,
{
    fn from_iter<I: IntoIterator<Item = R>>(iter: I) -> Self {
        let mut s = Self::new();
        s.extend(iter);
        s
    }
}

impl<R> std::iter::Sum for RunningStats<R>
where
    R: Real + Compare,
{
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
        let mut rs: RunningStats = RunningStats::new();
        assert!(rs.mean().is_nan());
        rs.extend([3.0, -1.0, 7.0, 2.0]);
        assert_eq!(rs.min(), -1.0);
        assert_eq!(rs.max(), 7.0);
    }
}
