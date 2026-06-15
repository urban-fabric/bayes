//! Beta-prior conjugate models: **Beta–Bernoulli** and **Beta–Binomial**.

use super::{Conjugate, Pair};
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
}

impl Conjugate for BetaBernoulli {
    type Datum = bool;
    type Posterior = Beta;
    type Predictive = Bernoulli;
    type Natural = Pair;

    /// One success or one failure: `T(true) = (1, 0)`, `T(false) = (0, 1)`.
    fn statistic(&self, &b: &bool) -> Pair {
        if b {
            Pair::new(1.0, 0.0)
        } else {
            Pair::new(0.0, 1.0)
        }
    }
    fn prior_natural(&self) -> Pair {
        Pair::new(self.prior.alpha(), self.prior.beta())
    }
    fn posterior_from_natural(&self, p: Pair) -> Beta {
        Beta::new(p.a, p.b).unwrap_or(self.prior)
    }
    /// The posterior-predictive for one future trial: `Bernoulli(α′/(α′+β′))`.
    fn posterior_predictive(&self, data: &[bool]) -> Bernoulli {
        let post = self.posterior(data);
        Bernoulli::from_probability_saturating(post.alpha() / (post.alpha() + post.beta()))
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
}

impl Conjugate for BetaBinomialModel {
    type Datum = u64;
    type Posterior = Beta;
    type Predictive = BetaBinomial;
    type Natural = Pair;

    /// `k` successes out of `m` trials: `T(k) = (k, m − k)` (the riemann
    /// `binomStat m`), with `k` clamped to the trial count.
    fn statistic(&self, &k: &u64) -> Pair {
        let successes = k.min(self.trials) as f64;
        Pair::new(successes, self.trials as f64 - successes)
    }
    fn prior_natural(&self) -> Pair {
        Pair::new(self.prior.alpha(), self.prior.beta())
    }
    fn posterior_from_natural(&self, p: Pair) -> Beta {
        Beta::new(p.a, p.b).unwrap_or(self.prior)
    }
    /// The posterior-predictive: a [`BetaBinomial`] over `trials` future draws.
    fn posterior_predictive(&self, data: &[u64]) -> BetaBinomial {
        BetaBinomial::from_beta(self.trials, self.posterior(data))
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
        let data = [true, true, true, false];
        let post = model.posterior(&data);
        close(post.alpha(), 5.0, 1e-12);
        close(post.beta(), 3.0, 1e-12);
        close(model.posterior_predictive(&data).p(), 5.0 / 8.0, 1e-12);
    }

    #[test]
    fn beta_binomial_predictive_is_overdispersed() {
        let model = BetaBinomialModel::new(Beta::new(1.0, 1.0).unwrap(), 10);
        let data = [7u64, 8, 6, 9];
        let post = model.posterior(&data);
        close(post.alpha(), 31.0, 1e-12);
        close(post.beta(), 11.0, 1e-12);
        let pred = model.posterior_predictive(&data);
        close(pred.mean().unwrap(), 10.0 * 31.0 / 42.0, 1e-12);
    }
}
