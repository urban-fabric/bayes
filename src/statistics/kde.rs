//! **Kernel density estimation** — a smooth, non-parametric estimate of a sample's
//! probability density. Uses a Gaussian kernel with a Silverman rule-of-thumb
//! bandwidth (overridable).

use std::f64::consts::TAU;

use super::order::iqr;
use super::summary::std_dev;

/// A Gaussian kernel density estimate over a sample.
#[derive(Clone, Debug)]
pub struct KernelDensity {
    samples: Vec<f64>,
    bandwidth: f64,
}

impl KernelDensity {
    /// Fit a KDE to `data` with Silverman's rule-of-thumb bandwidth
    /// `0.9·min(σ̂, IQR/1.349)·n^{-1/5}`. `None` if `data` has fewer than 2 points or
    /// the bandwidth degenerates to zero (all values identical).
    pub fn new(data: &[f64]) -> Option<Self> {
        if data.len() < 2 {
            return None;
        }
        let n = data.len() as f64;
        let spread = {
            let sd = std_dev(data);
            let iqr_scaled = iqr(data) / 1.349;
            if iqr_scaled > 0.0 {
                sd.min(iqr_scaled)
            } else {
                sd
            }
        };
        let bandwidth = 0.9 * spread * n.powf(-0.2);
        Self::with_bandwidth(data, bandwidth)
    }

    /// Fit a KDE with an explicit `bandwidth > 0`.
    pub fn with_bandwidth(data: &[f64], bandwidth: f64) -> Option<Self> {
        if data.is_empty() || !bandwidth.is_finite() || bandwidth <= 0.0 {
            return None;
        }
        Some(Self {
            samples: data.to_vec(),
            bandwidth,
        })
    }

    pub fn bandwidth(&self) -> f64 {
        self.bandwidth
    }

    /// The estimated density at `x`: `1/(n h) Σ φ((x − xᵢ)/h)` with `φ` the
    /// standard-normal density.
    pub fn density(&self, x: f64) -> f64 {
        let h = self.bandwidth;
        let norm = 1.0 / (self.samples.len() as f64 * h * TAU.sqrt());
        norm * self
            .samples
            .iter()
            .map(|&xi| {
                let u = (x - xi) / h;
                (-0.5 * u * u).exp()
            })
            .sum::<f64>()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn density_is_non_negative_and_integrates_to_about_one() {
        let data: Vec<f64> = (0..200).map(|i| (i as f64) * 0.05 - 5.0).collect();
        let kde = KernelDensity::new(&data).unwrap();
        let (lo, hi, steps) = (-15.0, 15.0, 6000);
        let dx = (hi - lo) / steps as f64;
        let mut area = 0.0;
        for k in 0..steps {
            let x = lo + (k as f64 + 0.5) * dx;
            let d = kde.density(x);
            assert!(d >= 0.0);
            area += d * dx;
        }
        assert!((area - 1.0).abs() < 0.02, "∫ kde ≈ 1, got {area}");
    }

    #[test]
    fn peaks_near_a_cluster() {
        let data = [-0.1, 0.0, 0.05, -0.05, 0.1, 0.02];
        let kde = KernelDensity::new(&data).unwrap();
        assert!(kde.density(0.0) > kde.density(5.0));
    }

    #[test]
    fn degenerate_inputs_return_none() {
        assert!(KernelDensity::new(&[1.0]).is_none());
        assert!(KernelDensity::new(&[2.0, 2.0, 2.0]).is_none());
    }
}
