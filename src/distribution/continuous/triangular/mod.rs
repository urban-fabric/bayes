use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::distribution::{Continuous, ContinuousCdf, Moments, Support};
use crate::error::BayesError;

/// The triangular distribution on `[lo, hi]` with mode `mode ∈ [lo, hi]` — a
/// simple bounded distribution with a single peak.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Triangular {
    lo: f64,
    hi: f64,
    mode: f64,
}

impl Triangular {
    pub fn new(lo: f64, hi: f64, mode: f64) -> Result<Self, BayesError> {
        for (name, v) in [("lo", lo), ("hi", hi), ("mode", mode)] {
            if !v.is_finite() {
                return Err(BayesError::NonFinite { name, value: v });
            }
        }
        if lo >= hi {
            return Err(BayesError::EmptyRange { lo, hi });
        }
        if mode < lo || mode > hi {
            return Err(BayesError::Malformed {
                what: "mode must lie in [lo, hi]",
            });
        }
        Ok(Self { lo, hi, mode })
    }

    pub fn lo(&self) -> f64 {
        self.lo
    }
    pub fn hi(&self) -> f64 {
        self.hi
    }
    pub fn mode_param(&self) -> f64 {
        self.mode
    }
}

impl Distribution<f64> for Triangular {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.inverse_cdf(rng.random::<f64>())
    }
}

impl Continuous for Triangular {
    fn pdf(&self, x: f64) -> f64 {
        let (a, b, c) = (self.lo, self.hi, self.mode);
        if x < a || x > b {
            0.0
        } else if x < c {
            2.0 * (x - a) / ((b - a) * (c - a))
        } else if x > c {
            2.0 * (b - x) / ((b - a) * (b - c))
        } else {
            2.0 / (b - a) // peak at the mode
        }
    }
}

impl Support for Triangular {
    fn min(&self) -> f64 {
        self.lo
    }
    fn max(&self) -> f64 {
        self.hi
    }
}

impl ContinuousCdf for Triangular {
    fn cdf(&self, x: f64) -> f64 {
        let (a, b, c) = (self.lo, self.hi, self.mode);
        if x <= a {
            0.0
        } else if x < c {
            (x - a) * (x - a) / ((b - a) * (c - a))
        } else if x < b {
            1.0 - (b - x) * (b - x) / ((b - a) * (b - c))
        } else {
            1.0
        }
    }
    fn inverse_cdf(&self, p: f64) -> f64 {
        let (a, b, c) = (self.lo, self.hi, self.mode);
        let p = p.clamp(0.0, 1.0);
        let split = (c - a) / (b - a);
        if p < split {
            a + ((b - a) * (c - a) * p).sqrt()
        } else {
            b - ((b - a) * (b - c) * (1.0 - p)).sqrt()
        }
    }
}

impl Moments for Triangular {
    fn mean(&self) -> Option<f64> {
        Some((self.lo + self.hi + self.mode) / 3.0)
    }
    fn variance(&self) -> Option<f64> {
        let (a, b, c) = (self.lo, self.hi, self.mode);
        Some((a * a + b * b + c * c - a * b - a * c - b * c) / 18.0)
    }
    fn mode(&self) -> Option<f64> {
        Some(self.mode)
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
    fn pdf_cdf_quantile_and_moments() {
        let t = Triangular::new(0.0, 4.0, 1.0).unwrap();
        close(t.pdf(1.0), 2.0 / 4.0, 1e-12); // peak 2/(b-a)
        close(t.cdf(4.0), 1.0, 1e-12);
        close(t.cdf(0.0), 0.0, 1e-12);
        for &p in &[0.05, 0.25, 0.5, 0.75, 0.95] {
            close(t.cdf(t.inverse_cdf(p)), p, 1e-10);
        }
        close(t.mean().unwrap(), 5.0 / 3.0, 1e-12);
    }
}
