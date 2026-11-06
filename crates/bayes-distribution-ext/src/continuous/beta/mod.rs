use rand::Rng;
use rand::distr::Distribution;

use super::gamma::sample_standard_gamma;
use crate::estimation::sample::Summary;
use crate::{Continuous, ContinuousDensityKernel};
use crate::{ContinuousCdf, Moments, Support};
use bayes_core_ext::{BayesError, validate};
use bayes_kernels as kernels;
use euclid_numeric::{Boolean, Compare, Real};

/// The beta distribution on `(0, 1)` with shape parameters `α, β > 0`. Density
/// `f(x) = x^{α-1}(1-x)^{β-1} / B(α, β)`.
///
/// # Examples
///
/// ```
/// use bayes_distribution_ext::{Beta, Continuous, Moments};
///
/// // Beta(1, 1) is the uniform distribution on (0, 1): a flat density of 1.
/// assert!((Beta::new(1.0, 1.0)?.pdf(0.3_f64) - 1.0).abs() < 1e-12);
///
/// // Raising `beta` pulls the mass toward 0. The mean is alpha/(alpha + beta).
/// let skewed = Beta::new(2.0, 5.0)?;
/// assert!((skewed.mean() - 2.0 / 7.0).abs() < 1e-12);
/// assert!(skewed.pdf(0.2_f64) > skewed.pdf(0.8_f64));
///
/// // Both shapes below 1 turns the density inside out, into a U.
/// let u_shaped = Beta::new(0.5, 0.5)?;
/// assert!(u_shaped.pdf(0.05_f64) > u_shaped.pdf(0.5_f64));
/// # Ok::<(), bayes_core_ext::BayesError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Beta {
    alpha: f64,
    beta: f64,
}

impl Beta {
    pub fn new(alpha: f64, beta: f64) -> Result<Self, BayesError> {
        validate::positive(&[("alpha", alpha), ("beta", beta)])?;
        Ok(Self { alpha, beta })
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }
    pub fn beta(&self) -> f64 {
        self.beta
    }
}

impl Distribution<f64> for Beta {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let ga = sample_standard_gamma(rng, self.alpha);
        let gb = sample_standard_gamma(rng, self.beta);
        ga / (ga + gb)
    }
}

impl<T: Real> Support<T> for Beta {
    fn min(&self) -> T {
        T::zero()
    }
    fn max(&self) -> T {
        T::one()
    }
}

impl<T: Real> ContinuousCdf<T> for Beta {
    fn cdf(&self, x: T) -> T {
        kernels::beta_cdf_on_unit(
            x.clamp(T::zero(), T::one()),
            T::scalar(self.alpha),
            T::scalar(self.beta),
        )
    }
    fn inverse_cdf(&self, p: T) -> T {
        kernels::inverse_cdf_bisect(|x| self.cdf(x), T::zero(), T::one(), p)
    }
}

impl Moments for Beta {
    fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }
    fn variance(&self) -> f64 {
        let s = self.alpha + self.beta;
        self.alpha * self.beta / (s * s * (s + 1.0))
    }
    fn skewness(&self) -> f64 {
        let s = self.alpha + self.beta;

        2.0 * (self.beta - self.alpha) * (s + 1.0).sqrt()
            / ((s + 2.0) * (self.alpha * self.beta).sqrt())
    }
    fn mode(&self) -> f64 {
        if self.alpha > 1.0 && self.beta > 1.0 {
            (self.alpha - 1.0) / (self.alpha + self.beta - 2.0)
        } else {
            f64::NAN
        }
    }
    /// `H = ln B(α,β) − (α−1)ψ(α) − (β−1)ψ(β) + (α+β−2)ψ(α+β)`.
    fn entropy(&self) -> f64 {
        let (a, b) = (self.alpha, self.beta);

        a.ln_beta(b) - (a - 1.0) * a.digamma() - (b - 1.0) * b.digamma()
            + (a + b - 2.0) * (a + b).digamma()
    }
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// `6[(α−β)²(α+β+1) − αβ(α+β+2)] / [αβ(α+β+2)(α+β+3)]`.
    fn kurtosis(&self) -> f64 {
        let s = self.alpha + self.beta;
        let d = self.alpha - self.beta;
        let prod = self.alpha * self.beta;
        6.0 * (d * d * (s + 1.0) - prod * (s + 2.0)) / (prod * (s + 2.0) * (s + 3.0))
    }

    fn median(&self) -> f64 {
        f64::NAN
    }
}

// Density kernel.
impl<T: Real> ContinuousDensityKernel<T> for Beta {
    fn ln_pdf_kernel(&self, x: T) -> T {
        let alpha = T::scalar(self.alpha());
        let beta = T::scalar(self.beta());
        kernels::beta_log_pdf_on_unit(x, alpha.clone(), beta.clone(), alpha.ln_beta(beta))
    }

    fn pdf_kernel(&self, x: T) -> T {
        self.ln_pdf_kernel(x).exp()
    }
}

// Continuous distribution.
impl<T> Continuous<T> for Beta
where
    T: Real + Compare,
{
    fn pdf(&self, x: T) -> T {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: T) -> T {
        let zero = T::zero();
        let one = T::one();
        // The **closed** interval, matching `Support`. The kernel is written with `xlogy`, so it
        // carries the correct endpoint limits itself (`+inf` for shape < 1, `ln β` at shape = 1,
        // `-inf` above); masking the endpoints off instead reported a density of zero there for
        // every shape, which made `Beta(1, 1)` differ from `Uniform(0, 1)` at both ends.
        let valid = x.ge(&zero).and(&x.le(&one));
        T::select(&valid, self.ln_pdf_kernel(x), T::neg_infinity())
    }
}

/// Method of moments on `(0, 1)`, **delegated** to
/// [`ExponentialFamily::fit`](crate::ExponentialFamily::fit).
///
/// The arithmetic — `t = m(1−m)/v − 1`, then `α = m·t` and `β = (1−m)·t` — is
/// already written once in this crate, as `Beta`'s
/// [`from_sufficient`](crate::ExponentialFamily::from_sufficient). A second
/// copy here would be free to drift away from it, so this adds only what that
/// one structurally cannot see, and then hands over.
///
/// What it adds is the **support**. A sufficient statistic carries a mean and a
/// variance and nothing else, so `from_sufficient` cannot distinguish
/// `[0.0, 0.5]` from a sample drawn inside the open interval: it sees
/// `m = 0.25` and `v = 0.0625`, finds `t = 2`, and returns `Beta(0.5, 1.5)` — a
/// fit to data containing a point where that density is zero. Only the sample
/// itself shows the violation.
///
/// Note that the other precondition is unreachable rather than merely unmet:
/// for data strictly inside `(0, 1)` the variance is always below `m(1−m)`,
/// since that is the supremum attained only by mass at the two endpoints. So a
/// bimodal sample does not fail to fit — it fits a U-shaped beta with both
/// shapes below 1, which is the correct answer.
impl crate::Estimate for Beta {
    fn from_sample(data: &[f64]) -> Option<Self> {
        let s = Summary::with_spread(data)?;
        if s.min <= 0.0 || s.max >= 1.0 {
            return None;
        }
        <Self as crate::ExponentialFamily>::fit(data)
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
    fn uniform_is_beta_one_one() {
        let b = Beta::new(1.0, 1.0).unwrap();
        assert_close!(b.cdf(0.37), 0.37, 1e-10);
        assert_close!(b.pdf(0.5), 1.0, 1e-10);
    }

    #[test]
    fn moments_and_quantile() {
        let b = Beta::new(2.0, 5.0).unwrap();
        assert_close!(b.mean(), 2.0 / 7.0, 1e-12);
        for &p in &[0.1, 0.5, 0.9] {
            assert_close!(b.cdf(b.inverse_cdf(p)), p, 1e-8);
        }
    }

    #[test]
    fn sampled_mean_matches() {
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let b = Beta::new(2.0, 5.0).unwrap();
        let n = 60_000;
        let s: f64 = (0..n).map(|_| b.sample(&mut rng)).sum();
        assert_close!(s / n as f64, b.mean(), 0.01);
    }
}
