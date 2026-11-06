use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::estimation::sample::Summary;
use crate::{Continuous, ContinuousDensityKernel};
use crate::{ContinuousCdf, Moments, Support};
use bayes_core_ext::{BayesError, validate};
use bayes_kernels as kernels;
use euclid_numeric::{Compare, Real};

/// The Fréchet (type-II generalized extreme value) distribution with shape
/// `α > 0` and scale `s > 0`, support `(0, ∞)`. CDF `F(x) = exp(−(x/s)^{−α})`; the
/// distribution of block maxima with heavy (polynomial) right tails.
///
/// # Examples
///
/// ```
/// use bayes_distribution_ext::{ContinuousCdf, Frechet, Moments};
///
/// // The scale is the distribution's own 1/e quantile, whatever the shape:
/// // F(scale) = exp(-1).
/// let f = Frechet::new(3.0, 2.0)?;
/// assert!((f.cdf(2.0_f64) - (-1.0_f64).exp()).abs() < 1e-12);
///
/// // The shape is a tail index: moments exist only below it, so a shape of 3
/// // has a mean and a variance but no skewness.
/// assert!(f.mean().is_finite());
/// assert!(Frechet::new(0.5, 1.0)?.mean().is_nan());
/// # Ok::<(), bayes_core_ext::BayesError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frechet {
    shape: f64,
    scale: f64,
}

impl Frechet {
    pub fn new(shape: f64, scale: f64) -> Result<Self, BayesError> {
        validate::positive(&[("shape", shape), ("scale", scale)])?;
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
        kernels::frechet_cdf(x, self.scale, self.shape)
    }
    fn inverse_cdf(&self, p: f64) -> f64 {
        kernels::frechet_quantile(p.clamp(0.0, 1.0), self.scale, self.shape)
    }
    fn ln_cdf(&self, x: f64) -> f64 {
        kernels::frechet_ln_cdf(x, self.scale, self.shape)
    }
    /// `ln(1 − e^{−t})` with `t = (x/s)^{−α}`, through `ln_1m_exp` so the
    /// upper tail does not vanish into the subtraction.
    fn ln_sf(&self, x: f64) -> f64 {
        kernels::frechet_ln_sf(x, self.scale, self.shape)
    }
}

impl Moments for Frechet {
    fn mean(&self) -> f64 {
        if self.shape > 1.0 {
            self.scale * (1.0 - 1.0 / self.shape).gamma()
        } else {
            f64::NAN
        }
    }
    fn variance(&self) -> f64 {
        if self.shape > 2.0 {
            let g1 = (1.0 - 1.0 / self.shape).gamma();
            let g2 = (1.0 - 2.0 / self.shape).gamma();
            self.scale * self.scale * (g2 - g1 * g1)
        } else {
            f64::NAN
        }
    }
    fn median(&self) -> f64 {
        self.scale * std::f64::consts::LN_2.powf(-1.0 / self.shape)
    }
    fn mode(&self) -> f64 {
        self.scale * (self.shape / (self.shape + 1.0)).powf(1.0 / self.shape)
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

    fn entropy(&self) -> f64 {
        f64::NAN
    }
}

// Density kernel.
impl<T: Real> ContinuousDensityKernel<T> for Frechet {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::frechet_log_pdf_on_support(x, T::scalar(self.scale()), T::scalar(self.shape()))
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

// Continuous distribution.
impl<T> Continuous<T> for Frechet
where
    T: Real + Compare,
{
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        let zero = T::zero();
        let valid = x.gt(&zero);
        T::select(&valid, self.ln_pdf_kernel(x), T::neg_infinity())
    }
}

/// Moments of `ln x`, where a Frechet becomes a location-scale family.
///
/// If `X` is Frechet with shape `α` and scale `s`, then `(X/s)^{−α}` is a
/// standard exponential, so `ln X = ln s − (ln U)/α` for `U ~ Exp(1)`. The log
/// of a standard exponential has mean `−γ` and variance `π²/6` — both constants
/// — which pins the two parameters with no iteration at all:
///
/// ```text
///     α̂ = π / (√6 · sd(ln x)),      ln ŝ = mean(ln x) − γ/α̂
/// ```
///
/// Note which summary feeds which parameter: on the log scale the shape is the
/// reciprocal of a standard deviation, so it comes out of the sample *spread*,
/// and the scale out of the sample *centre*. Matching ordinary moments instead
/// would need `Γ(1 − 1/α)` inverted, which has no closed form.
impl crate::Estimate for Frechet {
    fn from_sample(data: &[f64]) -> Option<Self> {
        let s = Summary::of_logs(data)?;
        let shape = std::f64::consts::PI / (6.0_f64.sqrt() * s.std_dev());
        let scale = (s.mean - std::f64::consts::EULER_GAMMA / shape).exp();
        Frechet::new(shape, scale).ok()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use bayes_core_ext::assert_close;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn cdf_pdf_quantile_and_moments() {
        let d = Frechet::new(3.0, 2.0).unwrap();
        for &p in &[0.1, 0.5, 0.9] {
            assert_close!(d.cdf(d.inverse_cdf(p)), p, 1e-9);
        }
        assert_close!(
            d.median(),
            2.0 * std::f64::consts::LN_2.powf(-1.0 / 3.0),
            1e-12,
        );
        assert!(Frechet::new(0.8, 1.0).unwrap().mean().is_nan());
    }

    #[test]
    fn sampled_median_matches() {
        let mut rng = ChaCha8Rng::seed_from_u64(17);
        let d = Frechet::new(4.0, 1.5).unwrap();
        let n = 40_000;
        let mut xs: Vec<f64> = (0..n).map(|_| d.sample(&mut rng)).collect();
        xs.sort_by(f64::total_cmp);
        assert_close!(xs[n / 2], d.median(), 0.03);
    }
}
