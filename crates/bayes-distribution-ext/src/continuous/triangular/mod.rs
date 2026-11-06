use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::estimation::sample::Summary;
use crate::{Continuous, ContinuousCdf, Moments, Support};
use bayes_core_ext::BayesError;
use bayes_kernels as kernels;
use euclid_numeric::Real;

/// The triangular distribution on `[lo, hi]` with mode `mode ∈ [lo, hi]` — a
/// simple bounded distribution with a single peak.
///
/// # Examples
///
/// ```
/// use bayes_distribution_ext::{Continuous, Moments, Triangular};
///
/// // The density is a tent, and its peak height is 2/(hi - lo) wherever the
/// // mode happens to sit — the area has to come to 1 either way.
/// let t = Triangular::new(0.0, 1.0, 0.5)?;
/// assert!((t.pdf(0.5_f64) - 2.0).abs() < 1e-12);
/// assert!((Triangular::new(0.0, 1.0, 0.9)?.pdf(0.9_f64) - 2.0).abs() < 1e-12);
///
/// // The mean is the average of the three parameters.
/// assert!((t.mean() - 0.5).abs() < 1e-12);
/// assert!((Triangular::new(0.0, 1.0, 0.9)?.mean() - 1.9 / 3.0).abs() < 1e-12);
/// # Ok::<(), bayes_core_ext::BayesError>(())
/// ```
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

/// The support from the sample's range, and the mode from its mean.
///
/// A triangular distribution has compact support, so the data bounds it:
/// `lo = min` and `hi = max`. The mean is `(lo + hi + mode)/3`, which inverts
/// to `mode = 3x̄ − lo − hi`.
///
/// That expression can land outside `[lo, hi]` on a small or lopsided sample,
/// so it is clamped — the one place in this crate where a fitted parameter is
/// forced into range instead of refused. The reason is specific to this family:
/// `lo` and `hi` are order statistics and already known to be consistent, so a
/// mode outside them reflects noise in `x̄` rather than evidence against the
/// family. Refusing would throw away a fit whose other two parameters are fine.
impl crate::Estimate for Triangular {
    fn from_sample(data: &[f64]) -> Option<Self> {
        let s = Summary::new(data)?;
        if s.max <= s.min {
            return None;
        }
        let mode = (3.0 * s.mean - s.min - s.max).clamp(s.min, s.max);
        Triangular::new(s.min, s.max, mode).ok()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use bayes_core_ext::assert_close;

    #[test]
    fn pdf_cdf_quantile_and_moments() {
        let t = Triangular::new(0.0, 4.0, 1.0).unwrap();
        assert_close!(t.pdf(1.0), 2.0 / 4.0, 1e-12);
        assert_close!(t.cdf(4.0), 1.0, 1e-12);
        assert_close!(t.cdf(0.0), 0.0, 1e-12);
        for &p in &[0.05, 0.25, 0.5, 0.75, 0.95] {
            assert_close!(t.cdf(t.inverse_cdf(p)), p, 1e-10);
        }
        assert_close!(t.mean(), 5.0 / 3.0, 1e-12);
    }
}
