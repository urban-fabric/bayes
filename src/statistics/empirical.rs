//! **Empirical distribution summaries** over a sample: the empirical CDF
//! ([`Ecdf`]) and the [`histogram`] (counts, with an optional density
//! normalisation). These are the non-parametric, data-side counterparts of the
//! analytic CDFs — the substrate for goodness-of-fit tests (e.g. Kolmogorov–
//! Smirnov) and exploratory density estimates.

/// The **empirical cumulative distribution function** of a sample:
/// `F̂(x) = (#{xᵢ ≤ x}) / n`, a right-continuous step function. NaNs are dropped on
/// construction.
#[derive(Clone, Debug, PartialEq)]
pub struct Ecdf {
    sorted: Vec<f64>,
}

impl Ecdf {
    /// Build from a sample (copied and sorted; NaNs removed).
    pub fn new(data: &[f64]) -> Self {
        let mut sorted: Vec<f64> = data.iter().copied().filter(|x| !x.is_nan()).collect();
        sorted.sort_by(f64::total_cmp);
        Self { sorted }
    }

    /// `F̂(x)` — the fraction of (non-NaN) samples `≤ x`. `NaN` for an empty sample.
    pub fn eval(&self, x: f64) -> f64 {
        if self.sorted.is_empty() {
            return f64::NAN;
        }
        let count = self.sorted.partition_point(|&v| v <= x);
        count as f64 / self.sorted.len() as f64
    }

    /// The empirical `p`-quantile (the smallest sample value with `F̂ ≥ p`).
    pub fn quantile(&self, p: f64) -> f64 {
        if self.sorted.is_empty() {
            return f64::NAN;
        }
        let n = self.sorted.len();
        let idx = (p.clamp(0.0, 1.0) * n as f64).ceil() as usize;
        let i = idx.saturating_sub(1).min(n - 1);
        self.sorted[i]
    }

    /// The sorted sample values backing the ECDF.
    pub fn sorted(&self) -> &[f64] {
        &self.sorted
    }

    pub fn len(&self) -> usize {
        self.sorted.len()
    }
    pub fn is_empty(&self) -> bool {
        self.sorted.is_empty()
    }
}

/// A histogram over equal-width bins: `edges` has `counts.len() + 1` entries, and
/// bin `i` is the half-open interval `[edges[i], edges[i+1])` (the last bin is
/// closed on the right so the maximum is included).
#[derive(Clone, Debug, PartialEq)]
pub struct Histogram {
    pub edges: Vec<f64>,
    pub counts: Vec<u64>,
}

impl Histogram {
    /// Bin counts normalised to a probability **density** (so `Σ densityᵢ · widthᵢ`
    /// = 1): `countᵢ / (n · widthᵢ)`.
    pub fn density(&self) -> Vec<f64> {
        let n: u64 = self.counts.iter().sum();
        if n == 0 {
            return vec![0.0; self.counts.len()];
        }
        let nf = n as f64;
        self.counts
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let width = self.edges[i + 1] - self.edges[i];
                if width > 0.0 {
                    c as f64 / (nf * width)
                } else {
                    0.0
                }
            })
            .collect()
    }
}

/// Build an equal-width [`Histogram`] with `bins` bins spanning the data range.
/// Returns `None` for empty data, `bins == 0`, or a degenerate (zero-width) range.
pub fn histogram(data: &[f64], bins: usize) -> Option<Histogram> {
    if bins == 0 {
        return None;
    }
    let clean: Vec<f64> = data.iter().copied().filter(|x| !x.is_nan()).collect();
    if clean.is_empty() {
        return None;
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &x in &clean {
        lo = lo.min(x);
        hi = hi.max(x);
    }
    if hi <= lo {
        return None; // degenerate (or non-finite) range
    }
    let width = (hi - lo) / bins as f64;
    let edges: Vec<f64> = (0..=bins).map(|i| lo + i as f64 * width).collect();
    let mut counts = vec![0u64; bins];
    for &x in &clean {
        // Last bin is closed on the right so the maximum lands in bin `bins-1`.
        let mut b = ((x - lo) / width) as usize;
        if b >= bins {
            b = bins - 1;
        }
        counts[b] += 1;
    }
    Some(Histogram { edges, counts })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn ecdf_steps_and_quantiles() {
        let e = Ecdf::new(&[3.0, 1.0, 2.0, 4.0]);
        close(e.eval(0.0), 0.0, 1e-12);
        close(e.eval(2.0), 0.5, 1e-12); // two of four ≤ 2
        close(e.eval(4.0), 1.0, 1e-12);
        close(e.eval(100.0), 1.0, 1e-12);
        assert_eq!(e.quantile(0.5), 2.0);
        assert_eq!(e.quantile(1.0), 4.0);
        assert_eq!(e.quantile(0.0), 1.0);
    }

    #[test]
    fn ecdf_handles_empty_and_nan() {
        assert!(Ecdf::new(&[]).eval(0.0).is_nan());
        let e = Ecdf::new(&[1.0, f64::NAN, 3.0]);
        assert_eq!(e.len(), 2); // NaN dropped
    }

    #[test]
    fn histogram_counts_and_density() {
        let data = [0.0, 0.5, 1.0, 1.5, 2.0]; // range [0, 2], 2 bins of width 1
        let h = histogram(&data, 2).unwrap();
        assert_eq!(h.edges, vec![0.0, 1.0, 2.0]);
        // [0,1): 0.0, 0.5 → 2 ; [1,2]: 1.0, 1.5, 2.0 → 3
        assert_eq!(h.counts, vec![2, 3]);
        let d = h.density();
        // Σ densityᵢ · widthᵢ = (2/5)·1 + (3/5)·1 = 1.
        close(d[0] * 1.0 + d[1] * 1.0, 1.0, 1e-12);
    }

    #[test]
    fn histogram_rejects_degenerate_inputs() {
        assert!(histogram(&[], 4).is_none());
        assert!(histogram(&[1.0, 2.0], 0).is_none());
        assert!(histogram(&[5.0, 5.0, 5.0], 3).is_none()); // zero-width range
    }
}
