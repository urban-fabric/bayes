//! A **t-digest** — a mergeable, bounded-memory sketch for streaming quantiles
//! (Dunning's merging variant). It keeps a small set of weighted *centroids* whose
//! sizes are limited by the `k1` scale function `k(q) = δ/2π · asin(2q−1)`, which
//! permits large centroids in the middle of the distribution but forces fine
//! resolution in the tails — so extreme quantiles stay accurate.
//!
//! It is approximate, but with a *bounded* error controlled by the compression
//! `δ`, and it merges directly (two digests combine into one without iterative
//! fitting), which is why it belongs in the direct/streaming foundation rather than
//! the inference layer.
//!
//! Construction/update lives here; the `quantile`/`cdf` queries are in `query`.

use std::f64::consts::PI;

mod query;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Centroid {
    mean: f64,
    weight: f64,
}

/// A streaming-quantile sketch. Build with [`new`](Self::new) (or
/// [`with_compression`](Self::with_compression)), feed values with
/// [`push`](Self::push), combine with [`merge`](Self::merge), and query
/// [`quantile`](Self::quantile) / [`cdf`](Self::cdf).
#[derive(Clone, Debug, PartialEq)]
pub struct TDigest {
    compression: f64,
    centroids: Vec<Centroid>,
    buffer: Vec<f64>,
    count: u64,
    min: f64,
    max: f64,
}

impl Default for TDigest {
    fn default() -> Self {
        Self::new()
    }
}

impl TDigest {
    /// A digest with the default compression (`δ = 100`).
    pub fn new() -> Self {
        Self::with_compression(100.0)
    }

    /// A digest with an explicit compression `δ` (larger ⇒ more centroids ⇒ more
    /// accurate, more memory). Values below 20 are clamped up.
    pub fn with_compression(compression: f64) -> Self {
        Self {
            compression: compression.max(20.0),
            centroids: Vec::new(),
            buffer: Vec::new(),
            count: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Absorb one observation (NaNs are ignored).
    pub fn push(&mut self, x: f64) {
        if x.is_nan() {
            return;
        }
        self.buffer.push(x);
        self.count += 1;
        self.min = self.min.min(x);
        self.max = self.max.max(x);
        if self.buffer.len() as f64 >= 10.0 * self.compression {
            self.compress();
        }
    }

    /// Combine another digest into this one (parallel/streamed reduction).
    pub fn merge(&mut self, other: &Self) {
        if other.count == 0 {
            return;
        }
        self.centroids.extend_from_slice(&other.centroids);
        self.buffer.extend_from_slice(&other.buffer);
        self.count += other.count;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
        self.compress();
    }

    pub fn count(&self) -> u64 {
        self.count
    }
    pub fn min(&self) -> f64 {
        if self.count == 0 { f64::NAN } else { self.min }
    }
    pub fn max(&self) -> f64 {
        if self.count == 0 { f64::NAN } else { self.max }
    }

    /// The `k1` scale function `k(q) = δ/2π · asin(2q − 1)`.
    fn scale(&self, q: f64) -> f64 {
        self.compression / (2.0 * PI) * (2.0 * q.clamp(0.0, 1.0) - 1.0).asin()
    }

    /// Fold the buffer into the centroids and re-compress under the scale bound.
    fn compress(&mut self) {
        if self.buffer.is_empty() && self.centroids.len() <= 1 {
            return;
        }
        let mut all = std::mem::take(&mut self.centroids);
        all.extend(self.buffer.drain(..).map(|x| Centroid {
            mean: x,
            weight: 1.0,
        }));
        if all.is_empty() {
            return;
        }
        all.sort_by(|a, b| a.mean.total_cmp(&b.mean));
        let w_total: f64 = all.iter().map(|c| c.weight).sum();

        let mut merged: Vec<Centroid> = Vec::with_capacity(all.len());
        let mut w_so_far = 0.0;
        let mut cur = all[0];
        for c in all.into_iter().skip(1) {
            let q0 = w_so_far / w_total;
            let q2 = (w_so_far + cur.weight + c.weight) / w_total;
            // Merge while the combined centroid still fits within one scale unit.
            if self.scale(q2) - self.scale(q0) <= 1.0 {
                let nw = cur.weight + c.weight;
                cur.mean += c.weight * (c.mean - cur.mean) / nw;
                cur.weight = nw;
            } else {
                w_so_far += cur.weight;
                merged.push(cur);
                cur = c;
            }
        }
        merged.push(cur);
        self.centroids = merged;
    }
}

impl Extend<f64> for TDigest {
    fn extend<I: IntoIterator<Item = f64>>(&mut self, iter: I) {
        for x in iter {
            self.push(x);
        }
    }
}

impl FromIterator<f64> for TDigest {
    fn from_iter<I: IntoIterator<Item = f64>>(iter: I) -> Self {
        let mut d = Self::new();
        d.extend(iter);
        d
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn uniform_quantiles_are_accurate() {
        // 0, 1/n, …, 1: the p-quantile of Uniform(0,1) is ≈ p.
        let n = 10_000;
        let d: TDigest = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        for &p in &[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
            assert!(
                (d.quantile(p) - p).abs() < 0.02,
                "q({p}) = {}",
                d.quantile(p)
            );
        }
        assert!(d.quantile(0.0) <= 0.01);
        assert!(d.quantile(1.0) >= 0.99);
    }

    #[test]
    fn quantile_and_cdf_are_inverse() {
        let n = 5_000;
        let d: TDigest = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        for &p in &[0.2, 0.5, 0.8] {
            let x = d.quantile(p);
            assert!((d.cdf(x) - p).abs() < 0.03, "cdf(q({p})) = {}", d.cdf(x));
        }
    }

    #[test]
    fn merge_matches_combined_stream() {
        let n = 8_000;
        let all: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.001).sin() + i as f64 / n as f64)
            .collect();
        let whole: TDigest = all.iter().copied().collect();
        let mut a: TDigest = all[..3000].iter().copied().collect();
        let b: TDigest = all[3000..].iter().copied().collect();
        a.merge(&b);
        assert_eq!(a.count(), whole.count());
        for &p in &[0.1, 0.5, 0.9] {
            assert!((a.quantile(p) - whole.quantile(p)).abs() < 0.02);
        }
    }

    #[test]
    fn empty_and_singleton() {
        let empty = TDigest::new();
        assert!(empty.quantile(0.5).is_nan());
        let mut one = TDigest::new();
        one.push(42.0);
        assert_eq!(one.quantile(0.5), 42.0);
        assert_eq!(one.count(), 1);
    }
}
