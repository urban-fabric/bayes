use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::{Continuous, ContinuousCdf, Moments, Support};
use bayes_core_ext::BayesError;
use bayes_kernels as kernels;
use euclid::Real;

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
        kernels::triangular_pdf(x, self.lo, self.hi, self.mode)
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        self.pdf(x).ln()
    }
}

impl<T: Real> Support<T> for Triangular {
    fn min(&self) -> T {
        T::scalar(self.lo)
    }
    fn max(&self) -> T {
        T::scalar(self.hi)
    }
}

impl<T: Real> ContinuousCdf<T> for Triangular {
    fn cdf(&self, x: T) -> T {
        kernels::triangular_cdf(
            x,
            T::scalar(self.lo),
            T::scalar(self.hi),
            T::scalar(self.mode),
        )
    }
    fn inverse_cdf(&self, p: T) -> T {
        kernels::triangular_quantile(
            p.clamp(T::zero(), T::one()),
            T::scalar(self.lo),
            T::scalar(self.hi),
            T::scalar(self.mode),
        )
    }
}

impl Moments for Triangular {
    fn mean(&self) -> f64 {
        (self.lo + self.hi + self.mode) / 3.0
    }
    fn variance(&self) -> f64 {
        let (a, b, c) = (self.lo, self.hi, self.mode);
        (a * a + b * b + c * c - a * b - a * c - b * c) / 18.0
    }
    fn mode(&self) -> f64 {
        self.mode
    }
    /// `H = ½ + ln((b − a)/2)` — independent of the mode `c`.
    fn entropy(&self) -> f64 {
        0.5 + ((self.hi - self.lo) / 2.0).ln()
    }
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn skewness(&self) -> f64 {
        f64::NAN
    }

    fn kurtosis(&self) -> f64 {
        f64::NAN
    }

    fn median(&self) -> f64 {
        f64::NAN
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
        close(t.pdf(1.0), 2.0 / 4.0, 1e-12);
        close(t.cdf(4.0), 1.0, 1e-12);
        close(t.cdf(0.0), 0.0, 1e-12);
        for &p in &[0.05, 0.25, 0.5, 0.75, 0.95] {
            close(t.cdf(t.inverse_cdf(p)), p, 1e-10);
        }
        close(t.mean(), 5.0 / 3.0, 1e-12);
    }
}
