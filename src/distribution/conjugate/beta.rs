//! Beta-prior conjugate models: **Beta–Bernoulli** and **Beta–Binomial**.

use super::Conjugate;
use crate::distribution::{Bernoulli, Beta, BetaBinomial};

/// **Beta–Bernoulli.** A `Beta(α, β)` prior on a Bernoulli success probability;
/// posterior `Beta(α + #successes, β + #failures)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BetaBernoulli {
    prior: Beta,
}

impl BetaBernoulli {
    pub fn new(prior: Beta) -> Self {
        Self { prior }
    }
    pub fn prior(&self) -> Beta {
        self.prior
    }
    /// The posterior-predictive for one future trial: `Bernoulli(α′/(α′+β′))`.
    pub fn posterior_predictive(&self, data: &[bool]) -> Bernoulli {
        let post = self.posterior(data);
        Bernoulli::from_probability_saturating(post.alpha() / (post.alpha() + post.beta()))
    }
}

impl Conjugate for BetaBernoulli {
    type Datum = bool;
    type Posterior = Beta;
    fn posterior(&self, data: &[bool]) -> Beta {
        let successes = data.iter().filter(|&&b| b).count() as f64;
        let failures = data.len() as f64 - successes;
        Beta::new(self.prior.alpha() + successes, self.prior.beta() + failures)
            .unwrap_or(self.prior)
    }
}

/// **Beta–Binomial.** A `Beta(α, β)` prior on the success probability of a binomial
/// with a fixed number of `trials` per observation; posterior
/// `Beta(α + Σ successes, β + Σ failures)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BetaBinomialModel {
    prior: Beta,
    trials: u64,
}

impl BetaBinomialModel {
    pub fn new(prior: Beta, trials: u64) -> Self {
        Self { prior, trials }
    }
    pub fn prior(&self) -> Beta {
        self.prior
    }
    /// The posterior-predictive: a [`BetaBinomial`] over `trials` future draws.
    pub fn posterior_predictive(&self, data: &[u64]) -> BetaBinomial {
        BetaBinomial::from_beta(self.trials, self.posterior(data))
    }
}

impl Conjugate for BetaBinomialModel {
    type Datum = u64;
    type Posterior = Beta;
    fn posterior(&self, data: &[u64]) -> Beta {
        let successes: f64 = data.iter().map(|&k| k.min(self.trials) as f64).sum();
        let failures = (data.len() as u64 * self.trials) as f64 - successes;
        Beta::new(self.prior.alpha() + successes, self.prior.beta() + failures)
            .unwrap_or(self.prior)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::distribution::Moments;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn beta_bernoulli_counts() {
        let model = BetaBernoulli::new(Beta::new(2.0, 2.0).unwrap());
        let data = [true, true, true, false]; // 3 success, 1 failure
        let post = model.posterior(&data);
        close(post.alpha(), 5.0, 1e-12);
        close(post.beta(), 3.0, 1e-12);
        // Predictive p = 5/8.
        close(model.posterior_predictive(&data).p(), 5.0 / 8.0, 1e-12);
    }

    #[test]
    fn beta_binomial_predictive_is_overdispersed() {
        let model = BetaBinomialModel::new(Beta::new(1.0, 1.0).unwrap(), 10);
        let data = [7u64, 8, 6, 9]; // Σ success = 30 of 40
        let post = model.posterior(&data);
        close(post.alpha(), 31.0, 1e-12); // 1 + 30
        close(post.beta(), 11.0, 1e-12); // 1 + 10
        let pred = model.posterior_predictive(&data);
        // Predictive mean = trials · α′/(α′+β′).
        close(pred.mean().unwrap(), 10.0 * 31.0 / 42.0, 1e-12);
    }
}
