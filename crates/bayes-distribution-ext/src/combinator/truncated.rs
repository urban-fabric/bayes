use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::{Continuous, ContinuousCdf, Support};
use bayes_core_ext::BayesError;
use euclid::{Boolean, Real};

/// A continuous distribution **truncated** to the interval `[lo, hi]` and
/// renormalized: density `f(x) / (F(hi) − F(lo))` on the interval, zero outside.
/// The base must provide a CDF and quantile (`ContinuousCdf`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Truncated<D> {
    base: D,
    lo: f64,
    hi: f64,
    cdf_lo: f64,
    /// Normalizing mass `F(hi) − F(lo)`.
    z: f64,
}

impl<D: ContinuousCdf> Truncated<D> {
    /// Truncate `base` to `[lo, hi]`. Errors if `lo ≥ hi` or the interval carries
    /// no probability mass under `base`.
    pub fn new(base: D, lo: f64, hi: f64) -> Result<Self, BayesError> {
        if !lo.is_finite() || !hi.is_finite() || lo >= hi {
            return Err(BayesError::EmptyRange { lo, hi });
        }
        let cdf_lo = base.cdf(lo);
        let z = base.cdf(hi) - cdf_lo;
        if z <= 0.0 {
            return Err(BayesError::Malformed {
                what: "truncation interval has zero probability mass",
            });
        }
        Ok(Self {
            base,
            lo,
            hi,
            cdf_lo,
            z,
        })
    }

    pub fn bounds(&self) -> (f64, f64) {
        (self.lo, self.hi)
    }
    pub fn base(&self) -> &D {
        &self.base
    }
}

impl<T, D> Continuous<T> for Truncated<D>
where
    T: Real,
    D: Continuous<T>,
{
    fn pdf(&self, x: T) -> T {
        let outside = x.lt(&T::scalar(self.lo)).or(&x.gt(&T::scalar(self.hi)));
        T::select(&outside, T::zero(), self.base.pdf(x) / T::scalar(self.z))
    }
    fn ln_pdf(&self, x: T) -> T {
        let outside = x.lt(&T::scalar(self.lo)).or(&x.gt(&T::scalar(self.hi)));
        T::select(
            &outside,
            T::neg_infinity(),
            self.base.ln_pdf(x) - T::scalar(self.z).ln(),
        )
    }
}

impl<T: Real, D> Support<T> for Truncated<D> {
    fn min(&self) -> T {
        T::scalar(self.lo)
    }
    fn max(&self) -> T {
        T::scalar(self.hi)
    }
}

impl<T, D> ContinuousCdf<T> for Truncated<D>
where
    T: Real,
    D: ContinuousCdf<T>,
{
    fn cdf(&self, x: T) -> T {
        let below = x.le(&T::scalar(self.lo));
        let above = x.ge(&T::scalar(self.hi));
        let interior = (self.base.cdf(x) - T::scalar(self.cdf_lo)) / T::scalar(self.z);
        T::select(&below, T::zero(), T::select(&above, T::one(), interior))
    }
    fn inverse_cdf(&self, p: T) -> T {
        self.base
            .inverse_cdf(T::scalar(self.cdf_lo) + p.clamp(T::zero(), T::one()) * T::scalar(self.z))
    }
}

impl<D: ContinuousCdf> Distribution<f64> for Truncated<D> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.inverse_cdf(rng.random::<f64>())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::Normal;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn density_renormalizes_and_is_zero_outside() {
        let t = Truncated::new(Normal::new(0.0, 1.0).unwrap(), -1.0, 1.0).unwrap();
        let base = Normal::new(0.0, 1.0).unwrap();
        let z = base.cdf(1.0) - base.cdf(-1.0);
        close(t.pdf(0.0), base.pdf(0.0) / z, 1e-12);
        assert_eq!(t.pdf(2.0), 0.0);
        assert_eq!(t.pdf(-3.0), 0.0);
    }

    #[test]
    fn cdf_quantile_and_bounded_samples() {
        let t = Truncated::new(Normal::new(0.0, 2.0).unwrap(), -1.0, 3.0).unwrap();
        close(t.cdf(-1.0), 0.0, 1e-12);
        close(t.cdf(3.0), 1.0, 1e-12);
        for &p in &[0.1, 0.5, 0.9] {
            close(t.cdf(t.inverse_cdf(p)), p, 1e-9);
        }
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        for _ in 0..2000 {
            let x = t.sample(&mut rng);
            assert!((-1.0..=3.0).contains(&x), "sample {x} out of [lo,hi]");
        }
    }
}
