use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::distribution::{Continuous, ContinuousCdf, Support};
use crate::error::BayesError;

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

impl<D: Continuous + ContinuousCdf> Continuous for Truncated<D> {
    fn pdf(&self, x: f64) -> f64 {
        if x < self.lo || x > self.hi {
            0.0
        } else {
            self.base.pdf(x) / self.z
        }
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        if x < self.lo || x > self.hi {
            f64::NEG_INFINITY
        } else {
            self.base.ln_pdf(x) - self.z.ln()
        }
    }
}

impl<D: ContinuousCdf> Support for Truncated<D> {
    fn min(&self) -> f64 {
        self.lo
    }
    fn max(&self) -> f64 {
        self.hi
    }
}

impl<D: ContinuousCdf> ContinuousCdf for Truncated<D> {
    fn cdf(&self, x: f64) -> f64 {
        if x <= self.lo {
            0.0
        } else if x >= self.hi {
            1.0
        } else {
            (self.base.cdf(x) - self.cdf_lo) / self.z
        }
    }
    fn inverse_cdf(&self, p: f64) -> f64 {
        // Map p ∈ [0,1] back to the base's CDF range [F(lo), F(hi)].
        self.base
            .inverse_cdf(self.cdf_lo + p.clamp(0.0, 1.0) * self.z)
    }
}

impl<D: ContinuousCdf> Distribution<f64> for Truncated<D> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        // Inverse-transform within the truncated range — always inside [lo, hi].
        self.inverse_cdf(rng.random::<f64>())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::distribution::Normal;
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
