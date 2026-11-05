use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::estimation::sample::Summary;
use crate::{Continuous, ContinuousDensityKernel};
use crate::{ContinuousCdf, Moments, Support};
use bayes_core_ext::{BayesError, validate};
use bayes_kernels as kernels;
use euclid_numeric::{Boolean, Compare, Real};

/// The Weibull distribution with `scale λ > 0` and `shape k > 0`, support `[0, ∞)`.
/// Density `f(x) = (k/λ)(x/λ)^{k-1} exp(-(x/λ)^k)`.
///
/// # Examples
///
/// ```
/// use bayes_distribution_ext::{Continuous, ContinuousCdf, Exponential, Weibull};
///
/// // Note the argument order: the **scale** comes first.
/// //
/// // The scale is always the 63rd percentile, whatever the shape, because
/// // F(scale) = 1 - 1/e identically.
/// let w = Weibull::new(10.0, 2.0)?;
/// assert!((w.cdf(10.0_f64) - (1.0 - (-1.0_f64).exp())).abs() < 1e-12);
///
/// // Shape 1 is the exponential — a constant hazard. Above 1 the hazard rises
/// // with age, which is what makes this the workhorse of reliability analysis.
/// assert!((Weibull::new(2.0, 1.0)?.pdf(1.0_f64) - Exponential::new(0.5)?.pdf(1.0_f64)).abs() < 1e-12);
/// assert!(w.pdf(5.0_f64) > w.pdf(20.0_f64));
/// # Ok::<(), bayes_core_ext::BayesError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Weibull {
    scale: f64,
    shape: f64,
}

impl Weibull {
    pub fn new(scale: f64, shape: f64) -> Result<Self, BayesError> {
        validate::positive(&[("scale", scale), ("shape", shape)])?;
        Ok(Self { scale, shape })
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }
    pub fn shape(&self) -> f64 {
        self.shape
    }
}

impl Distribution<f64> for Weibull {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let u = 1.0 - rng.random::<f64>();
        self.scale * (-u.ln()).powf(1.0 / self.shape)
    }
}

impl<T: Real> Support<T> for Weibull {
    fn min(&self) -> T {
        T::zero()
    }
    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Weibull {
    fn cdf(&self, x: T) -> T {
        kernels::weibull_cdf(x, T::scalar(self.scale), T::scalar(self.shape))
    }
    fn inverse_cdf(&self, p: T) -> T {
        kernels::weibull_quantile(
            p.clamp(T::zero(), T::one()),
            T::scalar(self.scale),
            T::scalar(self.shape),
        )
    }

    /// `−(x/λ)^k`, exact at every representable argument.
    fn ln_sf(&self, x: T) -> T {
        kernels::weibull_ln_sf(x, T::scalar(self.scale()), T::scalar(self.shape()))
    }
    /// `λ(−ln p)^{1/k}` — the reliability question "what age has survival `p`?",
    /// answerable for an arbitrarily small `p`.
    fn isf(&self, p: T) -> T {
        kernels::weibull_isf(p, T::scalar(self.scale()), T::scalar(self.shape()))
    }
}

impl Moments for Weibull {
    fn mean(&self) -> f64 {
        self.scale * (1.0 + 1.0 / self.shape).gamma()
    }
    fn variance(&self) -> f64 {
        let g1 = (1.0 + 1.0 / self.shape).gamma();
        let g2 = (1.0 + 2.0 / self.shape).gamma();
        self.scale * self.scale * (g2 - g1 * g1)
    }
    fn median(&self) -> f64 {
        self.scale * std::f64::consts::LN_2.powf(1.0 / self.shape)
    }
    fn mode(&self) -> f64 {
        if self.shape > 1.0 {
            self.scale * ((self.shape - 1.0) / self.shape).powf(1.0 / self.shape)
        } else {
            0.0
        }
    }
    /// `H = γ(1 − 1/k) + ln(λ/k) + 1`, with `γ` the Euler–Mascheroni constant.
    fn entropy(&self) -> f64 {
        use std::f64::consts::EULER_GAMMA;
        let (k, l) = (self.shape, self.scale);
        EULER_GAMMA * (1.0 - 1.0 / k) + (l / k).ln() + 1.0
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
}

// Density kernel.
impl<T: Real> ContinuousDensityKernel<T> for Weibull {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::weibull_log_pdf_on_support(x, T::scalar(self.scale()), T::scalar(self.shape()))
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

// Continuous distribution.
impl<T> Continuous<T> for Weibull
where
    T: Real + Compare,
{
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        let zero = T::zero();
        // `is_finite` belongs in the support test rather than beside it: at an infinite
        // argument the kernel evaluates a log term against a linear one, which is
        // `inf - inf = NaN`, so `pdf(inf)` came back `NaN` when the density there is 0.
        let valid = x.ge(&zero).and(&x.is_finite());
        T::select(&valid, self.ln_pdf_kernel(x), T::neg_infinity())
    }
}

/// Moments of `ln x` — the same device as [`Frechet`](crate::Frechet), with the
/// sign of the exponent flipped.
///
/// `(X/s)^k` is a standard exponential, so `ln X = ln s + (ln U)/k` and
///
/// ```text
///     k̂ = π / (√6 · sd(ln x)),      ln ŝ = mean(ln x) + γ/k̂
/// ```
///
/// The textbook Weibull MLE needs Newton iterations on the shape; this needs
/// none, and it is the estimator behind a Weibull probability plot.
///
/// Mind the constructor's argument order: [`Weibull::new`] takes the **scale
/// first**, then the shape.
impl crate::Estimate for Weibull {
    fn from_sample(data: &[f64]) -> Option<Self> {
        let s = Summary::of_logs(data)?;
        let shape = std::f64::consts::PI / (6.0_f64.sqrt() * s.std_dev());
        let scale = (s.mean + std::f64::consts::EULER_GAMMA / shape).exp();
        Weibull::new(scale, shape).ok()
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
    fn weibull_reduces_to_exponential_at_shape_one() {
        let w = Weibull::new(2.0, 1.0).unwrap();
        assert_close!(w.cdf(2.0), 1.0 - (-1.0f64).exp(), 1e-12);
        assert_close!(w.mean(), 2.0, 1e-10);
    }

    #[test]
    fn cdf_quantile_and_sampled_mean() {
        let w = Weibull::new(1.5, 2.0).unwrap();
        for &p in &[0.1, 0.5, 0.9] {
            assert_close!(w.cdf(w.inverse_cdf(p)), p, 1e-12);
        }
        let mut rng = ChaCha8Rng::seed_from_u64(8);
        let n = 60_000;
        let s: f64 = (0..n).map(|_| w.sample(&mut rng)).sum();
        assert_close!(s / n as f64, w.mean(), 0.02);
    }
}
