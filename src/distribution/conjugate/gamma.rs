//! Gamma-prior conjugate models: **Gamma–Poisson** and **Gamma–Exponential**.

use super::Conjugate;
use crate::distribution::{Gamma, NegativeBinomial};

/// **Gamma–Poisson.** A `Gamma(shape α, rate β)` prior on the Poisson rate;
/// posterior `Gamma(α + Σk, β + n)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GammaPoisson {
    prior: Gamma,
}

impl GammaPoisson {
    pub fn new(prior: Gamma) -> Self {
        Self { prior }
    }
    pub fn prior(&self) -> Gamma {
        self.prior
    }
    /// The posterior-predictive: `NegativeBinomial(r = α′, p = β′/(β′+1))`.
    pub fn posterior_predictive(&self, data: &[u64]) -> NegativeBinomial {
        let post = self.posterior(data);
        let (shape, rate) = (post.shape(), post.rate());
        NegativeBinomial::new(shape, rate / (rate + 1.0)).unwrap_or_else(|_| unreachable!())
    }
}

impl Conjugate for GammaPoisson {
    type Datum = u64;
    type Posterior = Gamma;
    fn posterior(&self, data: &[u64]) -> Gamma {
        let sum: f64 = data.iter().map(|&k| k as f64).sum();
        let n = data.len() as f64;
        Gamma::new(self.prior.shape() + sum, self.prior.rate() + n).unwrap_or(self.prior)
    }
}

/// **Gamma–Exponential.** A `Gamma(shape α, rate β)` prior on the exponential rate
/// `λ`; posterior `Gamma(α + n, β + Σx)`. (The posterior-predictive is a Lomax /
/// Pareto-II — available as [`Lomax`](crate::Lomax), but not auto-constructed here.)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GammaExponential {
    prior: Gamma,
}

impl GammaExponential {
    pub fn new(prior: Gamma) -> Self {
        Self { prior }
    }
    pub fn prior(&self) -> Gamma {
        self.prior
    }
}

impl Conjugate for GammaExponential {
    type Datum = f64;
    type Posterior = Gamma;
    fn posterior(&self, data: &[f64]) -> Gamma {
        let sum: f64 = data.iter().sum();
        let n = data.len() as f64;
        Gamma::new(self.prior.shape() + n, self.prior.rate() + sum).unwrap_or(self.prior)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::distribution::{Discrete, Moments};

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn gamma_poisson_posterior_and_predictive() {
        let model = GammaPoisson::new(Gamma::new(2.0, 1.0).unwrap());
        let data = [3u64, 5, 2, 4];
        let post = model.posterior(&data);
        close(post.shape(), 16.0, 1e-12);
        close(post.rate(), 5.0, 1e-12);
        let pred = model.posterior_predictive(&data);
        close(pred.mean().unwrap(), 16.0 / 5.0, 1e-12);
        assert!(pred.pmf(3) > 0.0);
    }

    #[test]
    fn gamma_exponential_posterior() {
        let model = GammaExponential::new(Gamma::new(1.0, 1.0).unwrap());
        let data = [0.5, 1.5, 2.0];
        let post = model.posterior(&data);
        close(post.shape(), 4.0, 1e-12);
        close(post.rate(), 5.0, 1e-12);
    }
}
