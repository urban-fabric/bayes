//! Gamma-prior conjugate models: **Gamma–Poisson** and **Gamma–Exponential**.

use super::{Conjugate, Pair};
use crate::distribution::{Gamma, Lomax, NegativeBinomial};
use crate::error::BayesError;
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
        if !future_exposure.is_finite() {
            return Err(BayesError::NonFinite {
                name: "future_exposure",
                value: future_exposure,
            });
        }
        if future_exposure <= 0.0 {
            return Err(BayesError::NotPositive {
                name: "future_exposure",
                value: future_exposure,
            });
        }
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
        shape * rate.ln() - ln_gamma(shape) - data.iter().map(|&k| ln_factorial(k)).sum::<f64>()
            + ln_gamma(shape + sum)
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
        shape * rate.ln() - ln_gamma(shape) + ln_gamma(shape + sum_counts)
            - (shape + sum_counts) * (rate + total_exposure).ln()
            + exposure_coeff
            - counts.iter().map(|&k| ln_factorial(k)).sum::<f64>()
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
        shape * rate.ln() - ln_gamma(shape) + ln_gamma(shape + n) - (shape + n) * (rate + sum).ln()
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
        shape * rate.ln() - ln_gamma(shape) + ln_gamma(shape + failures)
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

fn validate_exposure(value: f64, name: &'static str) -> Result<(), BayesError> {
    if !value.is_finite() {
        return Err(BayesError::NonFinite { name, value });
    }
    if value < 0.0 {
        return Err(BayesError::Malformed {
            what: "exposures must be non-negative",
        });
    }
    Ok(())
}

fn exponential_data_are_valid(data: &[f64]) -> bool {
    data.iter().all(|&x| x.is_finite() && x >= 0.0)
}

fn ln_gamma_poisson_predictive(shape: f64, rate: f64, count: u64, exposure: f64) -> f64 {
    if validate_exposure(exposure, "exposure").is_err() {
        return f64::NAN;
    }
    if exposure == 0.0 {
        return if count == 0 { 0.0 } else { f64::NAN };
    }
    let count_f = count as f64;
    ln_gamma(shape + count_f) - ln_gamma(shape) - ln_factorial(count)
        + shape * rate.ln()
        + count_f * exposure.ln()
        - (shape + count_f) * (rate + exposure).ln()
}

fn ln_gamma_exponential_predictive(shape: f64, rate: f64, x: f64) -> f64 {
    if !x.is_finite() || x < 0.0 {
        return f64::NAN;
    }
    shape.ln() + shape * rate.ln() - (shape + 1.0) * (rate + x).ln()
}

fn ln_gamma_exponential_exposure_evidence(
    shape: f64,
    rate: f64,
    failures: u64,
    total_exposure: f64,
) -> f64 {
    if validate_exposure(total_exposure, "total_exposure").is_err() {
        return f64::NAN;
    }
    let failures = failures as f64;
    shape * rate.ln() - ln_gamma(shape) + ln_gamma(shape + failures)
        - (shape + failures) * (rate + total_exposure).ln()
}

fn poisson_exposure_stats(
    counts: &[u64],
    exposures: &[f64],
) -> Result<(f64, f64, f64), BayesError> {
    if counts.len() != exposures.len() {
        return Err(BayesError::Malformed {
            what: "counts and exposures must have the same length",
        });
    }
    let mut sum_counts = 0.0;
    let mut total_exposure = 0.0;
    let mut exposure_coeff = 0.0;
    for (&count, &exposure) in counts.iter().zip(exposures) {
        validate_exposure(exposure, "exposure")?;
        if count > 0 && exposure == 0.0 {
            return Err(BayesError::Malformed {
                what: "positive counts require positive exposure",
            });
        }
        sum_counts += count as f64;
        total_exposure += exposure;
        if count > 0 {
            exposure_coeff += count as f64 * exposure.ln();
        }
    }
    Ok((sum_counts, total_exposure, exposure_coeff))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::distribution::{Continuous, Discrete, Moments};

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
        close(
            model.ln_predictive(3).exp(),
            model.posterior_predictive(&[]).pmf(3),
            1e-12,
        );
        close(
            model.ln_posterior_predictive(&data, 3).exp(),
            pred.pmf(3),
            1e-12,
        );
        let evidence =
            2.0 * 1.0_f64.ln() - ln_gamma(2.0) - data.iter().map(|&k| ln_factorial(k)).sum::<f64>()
                + ln_gamma(16.0)
                - 16.0 * 5.0_f64.ln();
        close(model.ln_marginal_likelihood(&data), evidence, 1e-12);
        close(
            model.ln_sequential_predictive(&data),
            model.ln_marginal_likelihood(&data),
            1e-12,
        );
        close(
            model.ln_posterior_predictive_likelihood(&data, &[3, 0]),
            GammaPoisson::new(post).ln_marginal_likelihood(&[3, 0]),
            1e-12,
        );
    }

    #[test]
    fn gamma_poisson_exposure_offsets_are_closed_form() {
        let model = GammaPoisson::new(Gamma::new(2.0, 1.0).unwrap());
        let counts = [3u64, 2];
        let exposures = [0.5, 1.5];
        let post = model.posterior_with_exposure(&counts, &exposures).unwrap();
        close(post.shape(), 7.0, 1e-12);
        close(post.rate(), 3.0, 1e-12);

        let pred = model
            .posterior_predictive_with_exposure(&counts, &exposures, 2.0)
            .unwrap();
        close(pred.mean().unwrap(), 14.0 / 3.0, 1e-12);
        close(
            model.ln_predictive_with_exposure(3, 0.5).exp(),
            model
                .posterior_predictive_with_exposure(&[], &[], 0.5)
                .unwrap()
                .pmf(3),
            1e-12,
        );
        close(
            model
                .ln_posterior_predictive_with_exposure(&counts, &exposures, 1, 2.0)
                .exp(),
            pred.pmf(1),
            1e-12,
        );

        let evidence = ln_gamma(7.0) - ln_gamma(2.0) - 7.0 * 3.0_f64.ln()
            + 3.0 * 0.5_f64.ln()
            + 2.0 * 1.5_f64.ln()
            - ln_factorial(3)
            - ln_factorial(2);
        close(
            model.ln_marginal_likelihood_with_exposure(&counts, &exposures),
            evidence,
            1e-12,
        );
        close(
            model.ln_sequential_predictive_with_exposure(&counts, &exposures),
            model.ln_marginal_likelihood_with_exposure(&counts, &exposures),
            1e-12,
        );
        close(
            model.ln_posterior_predictive_likelihood_with_exposure(
                &counts,
                &exposures,
                &[1],
                &[2.0],
            ),
            GammaPoisson::new(post).ln_marginal_likelihood_with_exposure(&[1], &[2.0]),
            1e-12,
        );
        assert!(model.posterior_with_exposure(&[1], &[0.0]).is_err());
        assert!(model.posterior_with_exposure(&[0], &[-1.0]).is_err());
        assert!(
            model
                .posterior_predictive_with_exposure(&counts, &exposures, 0.0)
                .is_err()
        );
        assert!(model.ln_predictive_with_exposure(1, 0.0).is_nan());
        assert!(
            model
                .ln_sequential_predictive_with_exposure(&[1], &[0.0])
                .is_nan()
        );
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
            model.ln_predictive(0.5).exp(),
            model.posterior_predictive(&[]).pdf(0.5),
            1e-12,
        );
        close(
            model.ln_posterior_predictive(&data, 1.0).exp(),
            pred.pdf(1.0),
            1e-12,
        );
        close(
            model.ln_marginal_likelihood(&data),
            1.0 * 1.0_f64.ln() - ln_gamma(1.0) + ln_gamma(4.0) - 4.0 * 5.0_f64.ln(),
            1e-12,
        );
        close(
            model.ln_sequential_predictive(&data),
            model.ln_marginal_likelihood(&data),
            1e-12,
        );
        close(
            model.ln_posterior_predictive_likelihood(&data, &[0.25, 0.75]),
            GammaExponential::new(post).ln_marginal_likelihood(&[0.25, 0.75]),
            1e-12,
        );
        assert!(model.ln_marginal_likelihood(&[-1.0]).is_nan());
        assert!(model.ln_predictive(-1.0).is_nan());
        assert!(model.ln_posterior_predictive(&[-1.0], 1.0).is_nan());
    }

    #[test]
    fn gamma_exponential_censored_exposure_is_closed_form() {
        let model = GammaExponential::new(Gamma::new(1.0, 1.0).unwrap());
        let post = model.posterior_with_exposure(2, 4.0).unwrap();
        close(post.shape(), 3.0, 1e-12);
        close(post.rate(), 5.0, 1e-12);

        let pred = model.posterior_predictive_with_exposure(2, 4.0).unwrap();
        close(pred.shape(), 3.0, 1e-12);
        close(pred.scale(), 5.0, 1e-12);
        close(pred.mean().unwrap(), 2.5, 1e-12);
        close(
            model.ln_posterior_predictive_with_exposure(2, 4.0, 1, 2.0),
            GammaExponential::new(post).ln_marginal_likelihood_with_exposure(1, 2.0),
            1e-12,
        );

        let evidence = ln_gamma(3.0) - 3.0 * 5.0_f64.ln();
        close(
            model.ln_marginal_likelihood_with_exposure(2, 4.0),
            evidence,
            1e-12,
        );
        close(model.ln_predictive_with_exposure(2, 4.0), evidence, 1e-12);
        close(
            model.ln_marginal_likelihood_with_exposure(0, 0.0),
            0.0,
            1e-12,
        );
        assert!(model.posterior_with_exposure(1, f64::NAN).is_err());
        assert!(model.ln_marginal_likelihood_with_exposure(1, -1.0).is_nan());
    }
}
