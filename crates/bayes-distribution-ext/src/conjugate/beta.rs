//! Beta-prior conjugate models: **Beta–Bernoulli** and **Beta–Binomial**.

use super::{Conjugate, Pair};
use crate::{Bernoulli, Beta, BetaBinomial};
use euclid::{LazyNatural, LazyReal};

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

    /// Closed-form prior/posterior predictive log probability for one future
    /// Bernoulli outcome under this Beta state.
    pub fn ln_predictive(&self, outcome: bool) -> f64 {
        ln_beta_bernoulli_predictive(self.prior.alpha(), self.prior.beta(), outcome)
    }

    /// Closed-form posterior-predictive log probability for one future
    /// Bernoulli outcome after observing `data`.
    pub fn ln_posterior_predictive(&self, data: &[bool], outcome: bool) -> f64 {
        Self::new(self.posterior(data)).ln_predictive(outcome)
    }

    /// Closed-form log probability of a future Bernoulli batch after observing
    /// `data`.
    pub fn ln_posterior_predictive_likelihood(&self, data: &[bool], future: &[bool]) -> f64 {
        Self::new(self.posterior(data)).ln_marginal_likelihood(future)
    }

    /// Sequential predictive factorization `Σᵢ ln p(xᵢ | x₁,…,xᵢ₋₁)`.
    ///
    /// In exact arithmetic this equals [`ln_marginal_likelihood`](Self::ln_marginal_likelihood).
    pub fn ln_sequential_predictive(&self, data: &[bool]) -> f64 {
        let mut alpha = self.prior.alpha();
        let mut beta = self.prior.beta();
        let mut ln_prob = 0.0;
        for &outcome in data {
            ln_prob += ln_beta_bernoulli_predictive(alpha, beta, outcome);
            if outcome {
                alpha += 1.0;
            } else {
                beta += 1.0;
            }
        }
        ln_prob
    }

    /// Closed-form log marginal likelihood `ln p(data | α, β)` with the
    /// Bernoulli probability integrated out:
    /// `ln B(α+s, β+f) - ln B(α, β)`.
    pub fn ln_marginal_likelihood(&self, data: &[bool]) -> f64 {
        let successes = data.iter().filter(|&&b| b).count() as f64;
        let failures = data.len() as f64 - successes;
        (self.prior.alpha() + successes).ln_beta(self.prior.beta() + failures)
            - self.prior.alpha().ln_beta(self.prior.beta())
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
    pub fn trials(&self) -> u64 {
        self.trials
    }

    /// Closed-form prior/posterior predictive log probability for one future
    /// fixed-trial binomial observation under this Beta state.
    pub fn ln_predictive(&self, successes: u64) -> f64 {
        ln_beta_binomial_predictive(
            self.prior.alpha(),
            self.prior.beta(),
            self.trials,
            successes,
        )
    }

    /// Closed-form posterior-predictive log probability for one future
    /// fixed-trial binomial observation after observing `data`.
    pub fn ln_posterior_predictive(&self, data: &[u64], successes: u64) -> f64 {
        if data.iter().any(|&k| k > self.trials) {
            return f64::NEG_INFINITY;
        }
        Self::new(self.posterior(data), self.trials).ln_predictive(successes)
    }

    /// Closed-form log probability of a future fixed-trial binomial batch after
    /// observing `data`.
    pub fn ln_posterior_predictive_likelihood(&self, data: &[u64], future: &[u64]) -> f64 {
        if data.iter().any(|&k| k > self.trials) {
            return f64::NEG_INFINITY;
        }
        Self::new(self.posterior(data), self.trials).ln_marginal_likelihood(future)
    }

    /// Sequential predictive factorization `Σᵢ ln p(kᵢ | k₁,…,kᵢ₋₁)`.
    ///
    /// In exact arithmetic this equals [`ln_marginal_likelihood`](Self::ln_marginal_likelihood).
    pub fn ln_sequential_predictive(&self, data: &[u64]) -> f64 {
        let mut alpha = self.prior.alpha();
        let mut beta = self.prior.beta();
        let mut ln_prob = 0.0;
        for &successes in data {
            if successes > self.trials {
                return f64::NEG_INFINITY;
            }
            ln_prob += ln_beta_binomial_predictive(alpha, beta, self.trials, successes);
            alpha += successes as f64;
            beta += (self.trials - successes) as f64;
        }
        ln_prob
    }

    /// Closed-form log marginal likelihood for fixed-trial binomial observations
    /// with the success probability integrated out:
    /// `Σ ln C(m,kᵢ) + ln B(α+Σkᵢ, β+Σ(m-kᵢ)) - ln B(α, β)`.
    ///
    /// Returns `-∞` if any observed success count exceeds the model's trial count.
    pub fn ln_marginal_likelihood(&self, data: &[u64]) -> f64 {
        if data.iter().any(|&k| k > self.trials) {
            return f64::NEG_INFINITY;
        }
        let successes = data.iter().sum::<u64>() as f64;
        let failures = self.trials as f64 * data.len() as f64 - successes;
        let coeff = data
            .iter()
            .map(|&k| self.trials.ln_binomial(&k))
            .sum::<f64>();
        coeff + (self.prior.alpha() + successes).ln_beta(self.prior.beta() + failures)
            - self.prior.alpha().ln_beta(self.prior.beta())
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

fn ln_beta_bernoulli_predictive(alpha: f64, beta: f64, outcome: bool) -> f64 {
    let total = alpha + beta;
    if outcome {
        (alpha / total).ln()
    } else {
        (beta / total).ln()
    }
}

fn ln_beta_binomial_predictive(alpha: f64, beta: f64, trials: u64, successes: u64) -> f64 {
    if successes > trials {
        return f64::NEG_INFINITY;
    }
    let successes_u = successes;
    let successes = successes_u as f64;
    let trials_f = trials as f64;
    trials.ln_binomial(&successes_u) + (successes + alpha).ln_beta(trials_f - successes + beta)
        - alpha.ln_beta(beta)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::{Discrete, Moments};

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
        close(model.ln_predictive(true).exp(), 0.5, 1e-12);
        close(
            model.ln_posterior_predictive(&data, true).exp(),
            5.0 / 8.0,
            1e-12,
        );
        close(
            model.ln_marginal_likelihood(&data),
            5.0_f64.ln_beta(3.0) - 2.0_f64.ln_beta(2.0),
            1e-12,
        );
        close(
            model.ln_sequential_predictive(&data),
            model.ln_marginal_likelihood(&data),
            1e-12,
        );
        close(
            model.ln_posterior_predictive_likelihood(&data, &[true, false]),
            BetaBernoulli::new(post).ln_marginal_likelihood(&[true, false]),
            1e-12,
        );
    }

    #[test]
    fn beta_binomial_predictive_is_overdispersed() {
        let model = BetaBinomialModel::new(Beta::new(1.0, 1.0).unwrap(), 10);
        let data = [7u64, 8, 6, 9];
        let post = model.posterior(&data);
        close(post.alpha(), 31.0, 1e-12);
        close(post.beta(), 11.0, 1e-12);
        assert_eq!(model.trials(), 10);
        let pred = model.posterior_predictive(&data);
        close(pred.mean(), 10.0 * 31.0 / 42.0, 1e-12);
        close(
            model.ln_predictive(7).exp(),
            BetaBinomial::from_beta(10, model.prior()).pmf(7_u64),
            1e-12,
        );
        close(
            model.ln_posterior_predictive(&data, 7).exp(),
            pred.pmf(7_u64),
            1e-12,
        );
        let evidence = (10_u64.ln_binomial(&7)
            + 10_u64.ln_binomial(&8)
            + 10_u64.ln_binomial(&6)
            + 10_u64.ln_binomial(&9))
            + 31.0_f64.ln_beta(11.0)
            - 1.0_f64.ln_beta(1.0);
        close(model.ln_marginal_likelihood(&data), evidence, 1e-12);
        close(
            model.ln_sequential_predictive(&data),
            model.ln_marginal_likelihood(&data),
            1e-12,
        );
        close(
            model.ln_posterior_predictive_likelihood(&data, &[7, 6]),
            BetaBinomialModel::new(post, 10).ln_marginal_likelihood(&[7, 6]),
            1e-12,
        );
        assert_eq!(model.ln_marginal_likelihood(&[11]), f64::NEG_INFINITY);
        assert_eq!(model.ln_predictive(11), f64::NEG_INFINITY);
        assert_eq!(model.ln_posterior_predictive(&[11], 1), f64::NEG_INFINITY);
    }
}
