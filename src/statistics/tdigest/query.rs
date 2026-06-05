//! [`TDigest`] queries: the `quantile` (inverse CDF) and `cdf` estimates, with
//! linear interpolation between centroid centers and the global min/max anchoring
//! the tails. Each query operates on the compressed centroids — if anything is still
//! buffered it compresses a clone first.

use super::TDigest;

impl TDigest {
    /// The estimated `p`-quantile (`p ∈ [0, 1]`). `NaN` for an empty digest.
    pub fn quantile(&self, p: f64) -> f64 {
        // Fast path: with nothing buffered, the centroids are already current.
        if self.buffer.is_empty() {
            return self.quantile_compressed(p);
        }
        let mut snap = self.clone();
        snap.compress();
        snap.quantile_compressed(p)
    }

    fn quantile_compressed(&self, p: f64) -> f64 {
        let cs = &self.centroids;
        if cs.is_empty() {
            return f64::NAN;
        }
        if cs.len() == 1 {
            return cs[0].mean;
        }
        let w_total: f64 = cs.iter().map(|c| c.weight).sum();
        let target = p.clamp(0.0, 1.0) * w_total;

        // Centroid "centers" in cumulative-weight space.
        let mut centers = Vec::with_capacity(cs.len());
        let mut cum = 0.0;
        for c in cs {
            centers.push(cum + c.weight / 2.0);
            cum += c.weight;
        }

        // Left tail: interpolate from the global min to the first centroid.
        if target < centers[0] {
            if centers[0] <= 0.0 {
                return cs[0].mean;
            }
            let t = target / centers[0];
            return self.min + t * (cs[0].mean - self.min);
        }
        // Right tail: interpolate from the last centroid to the global max.
        let last = cs.len() - 1;
        if target > centers[last] {
            let denom = w_total - centers[last];
            if denom <= 0.0 {
                return cs[last].mean;
            }
            let t = (target - centers[last]) / denom;
            return cs[last].mean + t * (self.max - cs[last].mean);
        }
        // Interior: linear interpolation between adjacent centroid centers.
        for i in 1..cs.len() {
            if target <= centers[i] {
                let t = (target - centers[i - 1]) / (centers[i] - centers[i - 1]);
                return cs[i - 1].mean + t * (cs[i].mean - cs[i - 1].mean);
            }
        }
        cs[last].mean
    }

    /// The estimated CDF `P(X ≤ x)` (the inverse of [`quantile`](Self::quantile)).
    pub fn cdf(&self, x: f64) -> f64 {
        if self.buffer.is_empty() {
            return self.cdf_compressed(x);
        }
        let mut snap = self.clone();
        snap.compress();
        snap.cdf_compressed(x)
    }

    fn cdf_compressed(&self, x: f64) -> f64 {
        let cs = &self.centroids;
        if cs.is_empty() {
            return f64::NAN;
        }
        if x < self.min {
            return 0.0;
        }
        if x >= self.max {
            return 1.0;
        }
        let w_total: f64 = cs.iter().map(|c| c.weight).sum();
        let mut cum = 0.0;
        let mut centers = Vec::with_capacity(cs.len());
        for c in cs {
            centers.push(cum + c.weight / 2.0);
            cum += c.weight;
        }
        if x < cs[0].mean {
            let t = (x - self.min) / (cs[0].mean - self.min);
            return (t * centers[0] / w_total).clamp(0.0, 1.0);
        }
        let last = cs.len() - 1;
        if x >= cs[last].mean {
            let t = (x - cs[last].mean) / (self.max - cs[last].mean);
            return ((centers[last] + t * (w_total - centers[last])) / w_total).clamp(0.0, 1.0);
        }
        for i in 1..cs.len() {
            if x <= cs[i].mean {
                let t = (x - cs[i - 1].mean) / (cs[i].mean - cs[i - 1].mean);
                return ((centers[i - 1] + t * (centers[i] - centers[i - 1])) / w_total)
                    .clamp(0.0, 1.0);
            }
        }
        1.0
    }
}
