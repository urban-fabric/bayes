use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::distribution::{Continuous, ContinuousCdf, Moments, Support};
use crate::error::BayesError;
use crate::function::gamma::gamma;

/// The Fréchet (type-II generalized extreme value) distribution with shape
/// `α > 0` and scale `s > 0`, support `(0, ∞)`. CDF `F(x) = exp(−(x/s)^{−α})`; the
/// distribution of block maxima with heavy (polynomial) right tails.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frechet {
    shape: f64,
    scale: f64,
}

impl Frechet {
    pub fn new(shape: f64, scale: f64) -> Result<Self, BayesError> {
        for (name, v) in [("shape", shape), ("scale", scale)] {
            if !v.is_finite() {
                return Err(BayesError::NonFinite { name, value: v });
            }
            if v <= 0.0 {
                return Err(BayesError::NotPositive { name, value: v });
            }
        }
        Ok(Self { shape, scale })
    }

    pub fn shape(&self) -> f64 {
        self.shape
    }
    pub fn scale(&self) -> f64 {
        self.scale
    }
}

impl Distribution<f64> for Frechet {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let u: f64 = rng.random();
        self.scale * (-u.ln()).powf(-1.0 / self.shape)
    }
}

impl Continuous for Frechet {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        self.ln_pdf(x).exp()
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let zinv = self.scale / x;
        (self.shape / self.scale).ln() + (self.shape + 1.0) * zinv.ln() - zinv.powf(self.shape)
    }
}

impl Support for Frechet {
    fn min(&self) -> f64 {
        0.0
    }
    fn max(&self) -> f64 {
        f64::INFINITY
    }
}

impl ContinuousCdf for Frechet {
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            (-(self.scale / x).powf(self.shape)).exp()
        }
    }
    fn inverse_cdf(&self, p: f64) -> f64 {
        let p = p.clamp(0.0, 1.0);
        if p <= 0.0 {
            return 0.0;
        }
        self.scale * (-p.ln()).powf(-1.0 / self.shape)
    }
    fn ln_cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            f64::NEG_INFINITY
        } else {
            -(self.scale / x).powf(self.shape)
        }
    }
}

impl Moments for Frechet {
    fn mean(&self) -> Option<f64> {
        (self.shape > 1.0).then(|| self.scale * gamma(1.0 - 1.0 / self.shape))
    }
    fn variance(&self) -> Option<f64> {
        if self.shape > 2.0 {
            let g1 = gamma(1.0 - 1.0 / self.shape);
            let g2 = gamma(1.0 - 2.0 / self.shape);
            Some(self.scale * self.scale * (g2 - g1 * g1))
        } else {
            None
        }
    }
    fn median(&self) -> Option<f64> {
        Some(self.scale * std::f64::consts::LN_2.powf(-1.0 / self.shape))
    }
    fn mode(&self) -> Option<f64> {
        Some(self.scale * (self.shape / (self.shape + 1.0)).powf(1.0 / self.shape))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn cdf_pdf_quantile_and_moments() {
        let d = Frechet::new(3.0, 2.0).unwrap();
        for &p in &[0.1, 0.5, 0.9] {
            close(d.cdf(d.inverse_cdf(p)), p, 1e-9);
        }
        close(
            d.median().unwrap(),
            2.0 * std::f64::consts::LN_2.powf(-1.0 / 3.0),
            1e-12,
        );
        assert!(Frechet::new(0.8, 1.0).unwrap().mean().is_none());
    }

    #[test]
    fn sampled_median_matches() {
        let mut rng = ChaCha8Rng::seed_from_u64(17);
        let d = Frechet::new(4.0, 1.5).unwrap();
        let n = 40_000;
        let mut xs: Vec<f64> = (0..n).map(|_| d.sample(&mut rng)).collect();
        xs.sort_by(f64::total_cmp);
        close(xs[n / 2], d.median().unwrap(), 0.03);
    }
}
