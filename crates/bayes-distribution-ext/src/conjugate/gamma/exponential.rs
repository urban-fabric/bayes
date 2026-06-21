use super::*;

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

    /// Closed-form prior/posterior predictive log density for one future
    /// exponential waiting time under this Gamma state.
    ///
    /// Returns `NaN` for negative or non-finite observations, matching
    /// [`ln_marginal_likelihood`](Self::ln_marginal_likelihood).
    pub fn ln_predictive(&self, x: f64) -> f64 {
        ln_gamma_exponential_predictive(self.prior.shape(), self.prior.rate(), x)
    }

    /// Closed-form prior/posterior predictive log likelihood for aggregate
    /// right-censored/exposure data under this Gamma state.
    pub fn ln_predictive_with_exposure(&self, failures: u64, total_exposure: f64) -> f64 {
        ln_gamma_exponential_exposure_evidence(
            self.prior.shape(),
            self.prior.rate(),
            failures,
            total_exposure,
        )
    }

    /// Closed-form posterior-predictive log density for one future waiting time
    /// after observing exact waiting times `data`.
    pub fn ln_posterior_predictive(&self, data: &[f64], x: f64) -> f64 {
        if !exponential_data_are_valid(data) {
            return f64::NAN;
        }
        Self::new(self.posterior(data)).ln_predictive(x)
    }

    /// Closed-form posterior-predictive log likelihood for future aggregate
    /// right-censored/exposure data after observing aggregate exposure data.
    pub fn ln_posterior_predictive_with_exposure(
        &self,
        failures: u64,
        total_exposure: f64,
        future_failures: u64,
        future_exposure: f64,
    ) -> f64 {
        let Ok(post) = self.posterior_with_exposure(failures, total_exposure) else {
            return f64::NAN;
        };
        Self::new(post).ln_predictive_with_exposure(future_failures, future_exposure)
    }

    /// Closed-form log probability of a future exact waiting-time batch after
    /// observing `data`.
    pub fn ln_posterior_predictive_likelihood(&self, data: &[f64], future: &[f64]) -> f64 {
        if !exponential_data_are_valid(data) {
            return f64::NAN;
        }
        Self::new(self.posterior(data)).ln_marginal_likelihood(future)
    }

    /// Sequential predictive factorization over exact exponential waiting times.
    ///
    /// In exact arithmetic this equals [`ln_marginal_likelihood`](Self::ln_marginal_likelihood).
    pub fn ln_sequential_predictive(&self, data: &[f64]) -> f64 {
        let mut shape = self.prior.shape();
        let mut rate = self.prior.rate();
        let mut ln_prob = 0.0;
        for &x in data {
            let term = ln_gamma_exponential_predictive(shape, rate, x);
            if term.is_nan() {
                return f64::NAN;
            }
            ln_prob += term;
            shape += 1.0;
            rate += x;
        }
        ln_prob
    }

    /// Closed-form posterior from right-censored/exposure data.
    ///
    /// For an exponential rate `λ`, failures contribute `λ^d` and all observed
    /// time-at-risk contributes `exp(-λT)`, so the posterior is
    /// `Gamma(α + d, β + T)`.
    pub fn posterior_with_exposure(
        &self,
        failures: u64,
        total_exposure: f64,
    ) -> Result<Gamma, BayesError> {
        validate_exposure(total_exposure, "total_exposure")?;
        Gamma::new(
            self.prior.shape() + failures as f64,
            self.prior.rate() + total_exposure,
        )
    }

    /// Posterior-predictive future waiting-time distribution after observing
    /// `failures` over `total_exposure`.
    pub fn posterior_predictive_with_exposure(
        &self,
        failures: u64,
        total_exposure: f64,
    ) -> Result<Lomax, BayesError> {
        let post = self.posterior_with_exposure(failures, total_exposure)?;
        Lomax::new(post.shape(), post.rate())
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
        shape * rate.ln() - shape.ln_gamma() + (shape + n).ln_gamma()
            - (shape + n) * (rate + sum).ln()
    }

    /// Closed-form log marginal likelihood for `failures` events observed over
    /// `total_exposure` time-at-risk, with right-censoring integrated through
    /// the exposure total.
    pub fn ln_marginal_likelihood_with_exposure(&self, failures: u64, total_exposure: f64) -> f64 {
        if validate_exposure(total_exposure, "total_exposure").is_err() {
            return f64::NAN;
        }
        let shape = self.prior.shape();
        let rate = self.prior.rate();
        let failures = failures as f64;
        shape * rate.ln() - shape.ln_gamma() + (shape + failures).ln_gamma()
            - (shape + failures) * (rate + total_exposure).ln()
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
