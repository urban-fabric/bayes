use rand::Rng;
use rand::distr::Distribution;

use super::gamma::sample_standard_gamma;
use crate::{Continuous, ContinuousDensityKernel};
use crate::{ContinuousCdf, Moments, Support};
use bayes_core_ext::BayesError;
use bayes_kernels as kernels;
use euclid::{Boolean, Compare, Real};

/// The beta distribution on `(0, 1)` with shape parameters `α, β > 0`. Density
/// `f(x) = x^{α-1}(1-x)^{β-1} / B(α, β)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Beta {
    alpha: f64,
    beta: f64,
}

impl Beta {
    pub fn new(alpha: f64, beta: f64) -> Result<Self, BayesError> {
        for (name, v) in [("alpha", alpha), ("beta", beta)] {
            if !v.is_finite() {
                return Err(BayesError::NonFinite { name, value: v });
            }
            if v <= 0.0 {
                return Err(BayesError::NotPositive { name, value: v });
            }
        }
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
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn uniform_is_beta_one_one() {
        let b = Beta::new(1.0, 1.0).unwrap();
        close(b.cdf(0.37), 0.37, 1e-10);
        close(b.pdf(0.5), 1.0, 1e-10);
    }

    #[test]
    fn moments_and_quantile() {
        let b = Beta::new(2.0, 5.0).unwrap();
        close(b.mean(), 2.0 / 7.0, 1e-12);
        for &p in &[0.1, 0.5, 0.9] {
            close(b.cdf(b.inverse_cdf(p)), p, 1e-8);
        }
    }

    #[test]
    fn sampled_mean_matches() {
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let b = Beta::new(2.0, 5.0).unwrap();
        let n = 60_000;
        let s: f64 = (0..n).map(|_| b.sample(&mut rng)).sum();
        close(s / n as f64, b.mean(), 0.01);
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
        let valid = x.gt(&zero).and(&x.lt(&one));
        T::select(&valid, self.ln_pdf_kernel(x), T::neg_infinity())
    }
}
