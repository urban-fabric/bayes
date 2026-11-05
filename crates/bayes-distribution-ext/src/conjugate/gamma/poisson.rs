use super::*;
use crate::NegativeBinomial;
use bayes_core_ext::validate;

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

    /// Closed-form prior/posterior predictive log probability for one future
    /// Poisson count over unit exposure under this Gamma state.
    pub fn ln_predictive(&self, count: u64) -> f64 {
        ln_gamma_poisson_predictive(self.prior.shape(), self.prior.rate(), count, 1.0)
    }

    /// Closed-form prior/posterior predictive log probability for one future
    /// Poisson count over a supplied exposure interval under this Gamma state.
    ///
    /// Returns `NaN` for malformed exposures, matching
    /// [`ln_marginal_likelihood_with_exposure`](Self::ln_marginal_likelihood_with_exposure).
    pub fn ln_predictive_with_exposure(&self, count: u64, exposure: f64) -> f64 {
        ln_gamma_poisson_predictive(self.prior.shape(), self.prior.rate(), count, exposure)
    }

    /// Closed-form posterior-predictive log probability for one future Poisson
    /// count over unit exposure after observing `data`.
    pub fn ln_posterior_predictive(&self, data: &[u64], count: u64) -> f64 {
        Self::new(self.posterior(data)).ln_predictive(count)
    }

    /// Closed-form posterior-predictive log probability for one future Poisson
    /// count over `future_exposure` after observing exposure-offset data.
    pub fn ln_posterior_predictive_with_exposure(
        &self,
        counts: &[u64],
        exposures: &[f64],
        future_count: u64,
        future_exposure: f64,
    ) -> f64 {
        let Ok(post) = self.posterior_with_exposure(counts, exposures) else {
            return f64::NAN;
        };
        Self::new(post).ln_predictive_with_exposure(future_count, future_exposure)
    }

    /// Closed-form log probability of a future unit-exposure Poisson count batch
    /// after observing `data`.
    pub fn ln_posterior_predictive_likelihood(&self, data: &[u64], future: &[u64]) -> f64 {
        Self::new(self.posterior(data)).ln_marginal_likelihood(future)
    }

    /// Closed-form log probability of a future exposure-offset Poisson batch
    /// after observing exposure-offset data.
    pub fn ln_posterior_predictive_likelihood_with_exposure(
        &self,
        counts: &[u64],
        exposures: &[f64],
        future_counts: &[u64],
        future_exposures: &[f64],
    ) -> f64 {
        let Ok(post) = self.posterior_with_exposure(counts, exposures) else {
            return f64::NAN;
        };
        Self::new(post).ln_marginal_likelihood_with_exposure(future_counts, future_exposures)
    }

    /// Sequential predictive factorization over unit-exposure Poisson counts.
    ///
    /// In exact arithmetic this equals [`ln_marginal_likelihood`](Self::ln_marginal_likelihood).
    pub fn ln_sequential_predictive(&self, data: &[u64]) -> f64 {
        let mut shape = self.prior.shape();
        let mut rate = self.prior.rate();
        let mut ln_prob = 0.0;
        for &count in data {
            ln_prob += ln_gamma_poisson_predictive(shape, rate, count, 1.0);
            shape += count as f64;
            rate += 1.0;
        }
        ln_prob
    }

    /// Sequential predictive factorization over exposure-offset Poisson counts.
    ///
    /// In exact arithmetic this equals
    /// [`ln_marginal_likelihood_with_exposure`](Self::ln_marginal_likelihood_with_exposure).
    pub fn ln_sequential_predictive_with_exposure(&self, counts: &[u64], exposures: &[f64]) -> f64 {
        if counts.len() != exposures.len() {
            return f64::NAN;
        }
        let mut shape = self.prior.shape();
        let mut rate = self.prior.rate();
        let mut ln_prob = 0.0;
        for (&count, &exposure) in counts.iter().zip(exposures) {
            let term = ln_gamma_poisson_predictive(shape, rate, count, exposure);
            if term.is_nan() {
                return f64::NAN;
            }
            ln_prob += term;
            shape += count as f64;
            rate += exposure;
        }
        ln_prob
    }

    /// Closed-form posterior with per-observation Poisson exposure offsets:
    /// `y_i ~ Poisson(λ e_i)`.
    ///
    /// The sufficient statistic is `(Σ y_i, Σ e_i)`, so the posterior is
    /// `Gamma(α + Σy, β + Σe)`. Exposures must be finite and non-negative; a
    /// positive count at zero exposure is malformed because it has zero
    /// likelihood under the model.
    pub fn posterior_with_exposure(
        &self,
        counts: &[u64],
        exposures: &[f64],
    ) -> Result<Gamma, BayesError> {
        let (sum_counts, total_exposure, _) = poisson_exposure_stats(counts, exposures)?;
        Gamma::new(
            self.prior.shape() + sum_counts,
            self.prior.rate() + total_exposure,
        )
    }

    /// Posterior-predictive count distribution for a future exposure interval.
    ///
    /// If `λ | data ~ Gamma(α′, β′)`, then a future count over exposure `e > 0`
    /// is `NegativeBinomial(r = α′, p = β′/(β′+e))`.
    pub fn posterior_predictive_with_exposure(
        &self,
        counts: &[u64],
        exposures: &[f64],
        future_exposure: f64,
    ) -> Result<NegativeBinomial, BayesError> {
        validate::positive(&[("future_exposure", future_exposure)])?;
        let post = self.posterior_with_exposure(counts, exposures)?;
        NegativeBinomial::new(post.shape(), post.rate() / (post.rate() + future_exposure))
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
        shape * rate.ln() - shape.ln_gamma() - data.iter().map(|&k| k.ln_factorial()).sum::<f64>()
            + (shape + sum).ln_gamma()
            - (shape + sum) * (rate + n).ln()
    }

    /// Closed-form log marginal likelihood for Poisson counts with exposure
    /// offsets and the shared rate integrated out.
    ///
    /// `ln p(y | e) = α ln β - ln Γ(α) + ln Γ(α+Σy) - (α+Σy) ln(β+Σe)
    /// + Σ yᵢ ln eᵢ - Σ ln(yᵢ!)`.
    pub fn ln_marginal_likelihood_with_exposure(&self, counts: &[u64], exposures: &[f64]) -> f64 {
        let Ok((sum_counts, total_exposure, exposure_coeff)) =
            poisson_exposure_stats(counts, exposures)
        else {
            return f64::NAN;
        };
        let shape = self.prior.shape();
        let rate = self.prior.rate();
        shape * rate.ln() - shape.ln_gamma() + (shape + sum_counts).ln_gamma()
            - (shape + sum_counts) * (rate + total_exposure).ln()
            + exposure_coeff
            - counts.iter().map(|&k| k.ln_factorial()).sum::<f64>()
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
