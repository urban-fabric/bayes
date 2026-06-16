//! Gamma-prior conjugate models: **Gamma–Poisson** and **Gamma–Exponential**.

use super::{Conjugate, Pair};
use crate::distribution::{Gamma, Lomax, NegativeBinomial};
use crate::function::{ln_factorial, ln_gamma};

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

    /// Closed-form log marginal likelihood for Poisson counts with the rate
    /// integrated out under a Gamma(shape, rate) prior:
    ///
    /// `α ln β - ln Γ(α) - Σ ln(kᵢ!) + ln Γ(α+Σkᵢ) - (α+Σkᵢ) ln(β+n)`.
    pub fn ln_marginal_likelihood(&self, data: &[u64]) -> f64 {
        let sum = data.iter().sum::<u64>() as f64;
        let n = data.len() as f64;
        let shape = self.prior.shape();
        let rate = self.prior.rate();
        shape * rate.ln() - ln_gamma(shape) - data.iter().map(|&k| ln_factorial(k)).sum::<f64>()
            + ln_gamma(shape + sum)
            - (shape + sum) * (rate + n).ln()
    }
}

impl Conjugate for GammaPoisson {
    type Datum = u64;
    type Posterior = Gamma;
    type Predictive = NegativeBinomial;
    type Natural = Pair;

    /// One Poisson count: `T(k) = (k, 1)` — adds `k` to the shape, `1` to the rate.
    fn statistic(&self, &k: &u64) -> Pair {
        Pair::new(k as f64, 1.0)
    }
    fn prior_natural(&self) -> Pair {
        Pair::new(self.prior.shape(), self.prior.rate())
    }
    fn posterior_from_natural(&self, p: Pair) -> Gamma {
        Gamma::new(p.a, p.b).unwrap_or(self.prior)
    }
    /// The posterior-predictive: `NegativeBinomial(r = α′, p = β′/(β′+1))`.
    fn posterior_predictive(&self, data: &[u64]) -> NegativeBinomial {
        let post = self.posterior(data);
        let (shape, rate) = (post.shape(), post.rate());
        NegativeBinomial::new(shape, rate / (rate + 1.0)).unwrap_or_else(|_| unreachable!())
    }
}

/// **Gamma–Exponential.** A `Gamma(shape α, rate β)` prior on the exponential rate
/// `λ`; posterior `Gamma(α + n, β + Σx)`. The posterior-predictive is a Lomax /
/// Pareto-II [`Lomax(α′, β′)`](crate::Lomax).
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

    /// Closed-form log marginal likelihood for exponential observations with
    /// the rate integrated out under a Gamma(shape, rate) prior:
    ///
    /// `α ln β - ln Γ(α) + ln Γ(α+n) - (α+n) ln(β+Σxᵢ)`.
    ///
    /// Returns `NaN` for negative or non-finite observations.
    pub fn ln_marginal_likelihood(&self, data: &[f64]) -> f64 {
        if data.iter().any(|&x| !x.is_finite() || x < 0.0) {
            return f64::NAN;
        }
        let sum = data.iter().sum::<f64>();
        let n = data.len() as f64;
        let shape = self.prior.shape();
        let rate = self.prior.rate();
        shape * rate.ln() - ln_gamma(shape) + ln_gamma(shape + n) - (shape + n) * (rate + sum).ln()
    }
}

impl Conjugate for GammaExponential {
    type Datum = f64;
    type Posterior = Gamma;
    type Predictive = Lomax;
    type Natural = Pair;

    /// One interarrival time: `T(x) = (1, x)` — adds `1` to the shape (a count),
    /// `x` to the rate.
    fn statistic(&self, &x: &f64) -> Pair {
        Pair::new(1.0, x)
    }
    fn prior_natural(&self) -> Pair {
        Pair::new(self.prior.shape(), self.prior.rate())
    }
    fn posterior_from_natural(&self, p: Pair) -> Gamma {
        Gamma::new(p.a, p.b).unwrap_or(self.prior)
    }
    /// The posterior-predictive: `Lomax(shape α′, scale β′)` from the posterior
    /// `Gamma(α′, β′)` — the rate marginalized out of the exponential likelihood.
    fn posterior_predictive(&self, data: &[f64]) -> Lomax {
        let post = self.posterior(data);
        Lomax::new(post.shape(), post.rate()).unwrap_or_else(|_| unreachable!())
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
        let evidence =
            2.0 * 1.0_f64.ln() - ln_gamma(2.0) - data.iter().map(|&k| ln_factorial(k)).sum::<f64>()
                + ln_gamma(16.0)
                - 16.0 * 5.0_f64.ln();
        close(model.ln_marginal_likelihood(&data), evidence, 1e-12);
    }

    #[test]
    fn gamma_exponential_posterior_and_lomax_predictive() {
        let model = GammaExponential::new(Gamma::new(1.0, 1.0).unwrap());
        let data = [0.5, 1.5, 2.0];
        let post = model.posterior(&data);
        close(post.shape(), 4.0, 1e-12);
        close(post.rate(), 5.0, 1e-12);
        // Predictive is Lomax(α′, β′) = Lomax(4, 5); mean = β′/(α′−1) = 5/3.
        let pred = model.posterior_predictive(&data);
        close(pred.shape(), 4.0, 1e-12);
        close(pred.scale(), 5.0, 1e-12);
        close(pred.mean().unwrap(), 5.0 / 3.0, 1e-12);
        close(
            model.ln_marginal_likelihood(&data),
            1.0 * 1.0_f64.ln() - ln_gamma(1.0) + ln_gamma(4.0) - 4.0 * 5.0_f64.ln(),
            1e-12,
        );
        assert!(model.ln_marginal_likelihood(&[-1.0]).is_nan());
    }
}
