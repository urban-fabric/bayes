use std::f64::consts::{EULER_GAMMA, PI};

use crate::{Continuous, ContinuousDensityKernel};
use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::estimation::sample::Summary;
use crate::{ContinuousCdf, Moments, Support};
use bayes_core_ext::{BayesError, validate};
use bayes_kernels as kernels;
use euclid_numeric::Real;

/// The Gumbel (Type-I extreme value, maximum) distribution with location `μ` and
/// scale `β > 0`, support `(−∞, ∞)`. Models the distribution of maxima.
///
/// # Examples
///
/// ```
/// use bayes_distribution_ext::{ContinuousCdf, Gumbel, Moments};
///
/// // The limit law for a maximum, after centring: the standard Gumbel's CDF at
/// // zero is exp(-1).
/// let g = Gumbel::new(0.0, 1.0)?;
/// assert!((g.cdf(0.0_f64) - (-1.0_f64).exp()).abs() < 1e-12);
///
/// // Its mean is the Euler-Mascheroni constant, and its skewness is a fixed
/// // positive number — the shape never changes, only the location and scale.
/// assert!((g.mean() - std::f64::consts::EULER_GAMMA).abs() < 1e-12);
/// assert!((g.skewness() - Gumbel::new(5.0, 3.0)?.skewness()).abs() < 1e-12);
/// # Ok::<(), bayes_core_ext::BayesError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gumbel {
    location: f64,
    scale: f64,
}

impl Gumbel {
    pub fn new(location: f64, scale: f64) -> Result<Self, BayesError> {
        if !location.is_finite() {
            return Err(BayesError::NonFinite {
                name: "location",
                value: location,
            });
        }
        validate::positive(&[("scale", scale)])?;
        Ok(Self { location, scale })
    }

    pub fn location(&self) -> f64 {
        self.location
    }
    pub fn scale(&self) -> f64 {
        self.scale
    }
}

impl Distribution<f64> for Gumbel {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.inverse_cdf(rng.random::<f64>())
    }
}

impl<T: Real> Support<T> for Gumbel {
    fn min(&self) -> T {
        T::neg_infinity()
    }
    fn max(&self) -> T {
        T::infinity()
    }
}

impl<T: Real> ContinuousCdf<T> for Gumbel {
    fn cdf(&self, x: T) -> T {
        kernels::gumbel_cdf(x, T::scalar(self.location), T::scalar(self.scale))
    }
    fn inverse_cdf(&self, p: T) -> T {
        // No clamp. `location − scale·ln(−ln p)` already yields `∓∞` at `p = 0` and `p = 1`,
        // which are the support bounds. Clamping to `[MIN_POSITIVE, 1 − EPSILON]` instead
        // returned a *finite* answer there — `Q(0) = −6.56` for the standard Gumbel, which is
        // the 0.14th percentile, not the infimum — and silently extrapolated out-of-range `p`
        // onto those same two values.
        kernels::gumbel_quantile(p, T::scalar(self.location), T::scalar(self.scale))
    }

    /// `ln(1 − e^{−t})` with `t = e^{−z}`, through `ln_1m_exp` so the upper
    /// tail's `−z` asymptote survives.
    fn ln_sf(&self, x: T) -> T {
        kernels::gumbel_ln_sf(x, T::scalar(self.location()), T::scalar(self.scale()))
    }
}

impl Moments for Gumbel {
    fn mean(&self) -> f64 {
        self.location + self.scale * EULER_GAMMA
    }
    fn variance(&self) -> f64 {
        PI * PI / 6.0 * self.scale * self.scale
    }
    fn skewness(&self) -> f64 {
        1.139_547_099_404_649
    }
    fn median(&self) -> f64 {
        self.location - self.scale * std::f64::consts::LN_2.ln()
    }
    fn mode(&self) -> f64 {
        self.location
    }
    fn entropy(&self) -> f64 {
        self.scale.ln() + EULER_GAMMA + 1.0
    }
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Excess kurtosis is the scale-free constant `12/5`.
    fn kurtosis(&self) -> f64 {
        12.0 / 5.0
    }
}

// Density kernel.
impl<T: Real> ContinuousDensityKernel<T> for Gumbel {
    fn ln_pdf_kernel(&self, x: T) -> T {
        kernels::gumbel_log_pdf(x, T::scalar(self.location()), T::scalar(self.scale()))
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

// Continuous distribution.
impl<T: Real> Continuous<T> for Gumbel {
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        // The support is all of R, but an infinite argument still drives the kernel to
        // `inf - inf = NaN`. The density at an infinite argument is 0.
        let finite = x.is_finite();
        T::select(&finite, self.ln_pdf_kernel(x), T::neg_infinity())
    }
}

/// Method of moments: `Var = π²β²/6` gives the scale, then `E = μ + γβ` gives
/// the location.
impl crate::Estimate for Gumbel {
    fn from_sample(data: &[f64]) -> Option<Self> {
        let s = Summary::with_spread(data)?;
        let scale = s.std_dev() * 6.0_f64.sqrt() / PI;
        Gumbel::new(s.mean - EULER_GAMMA * scale, scale).ok()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use bayes_core_ext::assert_close;

    #[test]
    fn cdf_quantile_and_moments() {
        let g = Gumbel::new(1.0, 2.0).unwrap();
        for &p in &[0.1, 0.5, 0.9] {
            assert_close!(g.cdf(g.inverse_cdf(p)), p, 1e-10);
        }
        assert_close!(g.mean(), 1.0 + 2.0 * EULER_GAMMA, 1e-12);
        assert_close!(g.mode(), 1.0, 1e-12);
        assert_close!(g.cdf(1.0), (-1.0f64).exp(), 1e-12);
    }
}
