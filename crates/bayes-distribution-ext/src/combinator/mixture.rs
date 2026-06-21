use rand::distr::Distribution;
use rand::{Rng, RngExt};

use crate::{Continuous, Discrete, Moments};
use bayes_core_ext::BayesError;

fn log_sum_exp(xs: &[f64]) -> Option<f64> {
    let (&first, rest) = xs.split_first()?;
    let max = rest.iter().fold(first, |acc, &x| acc.max(x));
    let shifted_sum = xs.iter().map(|&x| (x - max).exp()).sum::<f64>();
    Some(max + shifted_sum.ln())
}

fn log_softmax(xs: &[f64]) -> Option<Vec<f64>> {
    let log_z = log_sum_exp(xs)?;
    Some(xs.iter().map(|&x| x - log_z).collect())
}

/// A finite **mixture distribution**: a weighted blend of `components`, all of the
/// same type `D`. Sampling picks a component by weight then samples it; the density
/// is the weighted sum of the components' densities.
#[derive(Clone, Debug, PartialEq)]
pub struct Mixture<D> {
    weights: Vec<f64>,
    cumulative: Vec<f64>,
    components: Vec<D>,
}

impl<D> Mixture<D> {
    /// Build from non-negative `weights` (normalized internally) and the same
    /// number of `components`.
    pub fn new(weights: Vec<f64>, components: Vec<D>) -> Result<Self, BayesError> {
        if components.is_empty() || weights.len() != components.len() {
            return Err(BayesError::Malformed {
                what: "mixture needs matching, non-empty weights and components",
            });
        }
        let mut sum = 0.0;
        for &w in &weights {
            if !w.is_finite() || w < 0.0 {
                return Err(BayesError::Malformed {
                    what: "mixture weights must be finite and non-negative",
                });
            }
            sum += w;
        }
        if sum <= 0.0 {
            return Err(BayesError::NotPositive {
                name: "weight sum",
                value: sum,
            });
        }
        let weights: Vec<f64> = weights.iter().map(|&w| w / sum).collect();
        let mut cumulative = Vec::with_capacity(weights.len());
        let mut acc = 0.0;
        for &w in &weights {
            acc += w;
            cumulative.push(acc);
        }
        Ok(Self {
            weights,
            cumulative,
            components,
        })
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }
    pub fn components(&self) -> &[D] {
        &self.components
    }

    fn log_weighted_terms_by<F>(&self, mut log_component_likelihood: F) -> Vec<f64>
    where
        F: FnMut(&D) -> f64,
    {
        self.weights
            .iter()
            .zip(&self.components)
            .map(|(&w, component)| {
                if w == 0.0 {
                    f64::NEG_INFINITY
                } else {
                    w.ln() + log_component_likelihood(component)
                }
            })
            .collect()
    }

    /// Posterior log responsibilities for the mixture component indicator, from
    /// arbitrary per-component log likelihoods.
    ///
    /// The returned vector is `log P(Z = i | observation) =
    /// log(wᵢ likelihoodᵢ) - log Σⱼ wⱼ likelihoodⱼ`, computed with stable
    /// log-softmax. This is the finite mixture analogue of riemann's
    /// `Markov/LogSumExp.softmax_isDistribution`: when the result is non-empty,
    /// exponentiating it gives a probability vector. Returns an empty vector when
    /// the observation has zero likelihood under every positive-weight component
    /// (or if any log likelihood is `NaN`).
    pub fn ln_responsibilities_by<F>(&self, log_component_likelihood: F) -> Vec<f64>
    where
        F: FnMut(&D) -> f64,
    {
        let terms = self.log_weighted_terms_by(log_component_likelihood);
        // The "empty on zero total likelihood / NaN" contract is this consumer's host
        // edge policy: an all-`−∞` (or NaN) term set normalizes to `NaN`, which we drop.
        // A valid result may still contain `−∞` (a zero-weight component), so reject
        // only `NaN`.
        log_softmax(&terms)
            .filter(|lp| lp.iter().all(|x| !x.is_nan()))
            .unwrap_or_default()
    }

    /// Posterior responsibilities for the mixture component indicator, from
    /// arbitrary per-component log likelihoods.
    ///
    /// See [`ln_responsibilities_by`](Self::ln_responsibilities_by) for the log
    /// form. This exponentiates the stable log-softmax result.
    pub fn responsibilities_by<F>(&self, log_component_likelihood: F) -> Vec<f64>
    where
        F: FnMut(&D) -> f64,
    {
        self.ln_responsibilities_by(log_component_likelihood)
            .into_iter()
            .map(f64::exp)
            .collect()
    }
}

impl<D: Moments> Mixture<D> {
    /// The **law-of-total-variance decomposition** `(within, between)` of the
    /// mixture variance: the *within-component* part `E[Var(X|Z)] = Σ wᵢ Var(Xᵢ)`
    /// (the weight-averaged component variance) and the *between-component* part
    /// `Var(E[X|Z]) = Σ wᵢ (μᵢ − μ)²` (the spread of component means about the
    /// mixture mean `μ`). Their sum is [`variance`](Moments::variance).
    ///
    /// The identity `variance = within + between` and its nonnegativity are proved
    /// in `riemann` (`RiemannLean/Bayes/Mixture.lean`: `law_of_total_variance`,
    /// `mixtureVar_nonneg`). Returns `NaN` components if any component lacks a
    /// mean or variance.
    pub fn variance_decomposition(&self) -> (f64, f64) {
        let mut mean = 0.0;
        for (&w, c) in self.weights.iter().zip(&self.components) {
            mean += w * c.mean();
        }
        let mut within = 0.0;
        let mut between = 0.0;
        for (&w, c) in self.weights.iter().zip(&self.components) {
            within += w * c.variance();
            between += w * (c.mean() - mean).powi(2);
        }
        (within, between)
    }
}

impl<D, T> Distribution<T> for Mixture<D>
where
    D: Distribution<T>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        let u: f64 = rng.random();
        let i = self
            .cumulative
            .iter()
            .position(|&c| u <= c)
            .unwrap_or(self.components.len() - 1);
        self.components[i].sample(rng)
    }
}

impl<D> Continuous for Mixture<D>
where
    D: Continuous,
{
    fn pdf(&self, x: f64) -> f64 {
        self.weights
            .iter()
            .zip(&self.components)
            .map(|(&w, c)| w * c.pdf(x))
            .sum()
    }
    /// `ln Σ wᵢ fᵢ(x)` via log-sum-exp over `ln wᵢ + ln fᵢ(x)` — stable when the
    /// component log-densities are very negative (the naive `pdf().ln()` would
    /// underflow each term to `0` before summing).
    fn ln_pdf(&self, x: f64) -> f64 {
        let terms = self.log_weighted_terms_by(|c| c.ln_pdf(x));
        log_sum_exp(&terms).unwrap_or(f64::NEG_INFINITY)
    }
}

impl<D> Mixture<D>
where
    D: Continuous,
{
    /// Component log responsibilities for a continuous observation `x`.
    ///
    /// Entry `i` is `log(wᵢ fᵢ(x) / Σⱼ wⱼ fⱼ(x))`, computed in log space.
    pub fn ln_pdf_responsibilities(&self, x: f64) -> Vec<f64> {
        self.ln_responsibilities_by(|c| c.ln_pdf(x))
    }

    /// Component responsibilities for a continuous observation `x`.
    pub fn pdf_responsibilities(&self, x: f64) -> Vec<f64> {
        self.responsibilities_by(|c| c.ln_pdf(x))
    }
}

impl<D> Discrete<u64> for Mixture<D>
where
    D: Discrete<u64>,
{
    fn pmf(&self, k: u64) -> f64 {
        self.weights
            .iter()
            .zip(&self.components)
            .map(|(&w, c)| w * c.pmf(k))
            .sum()
    }
    /// `ln Σ wᵢ Pᵢ(k)` via log-sum-exp (see [`Continuous::ln_pdf`] for the mixture).
    fn ln_pmf(&self, k: u64) -> f64 {
        let terms = self.log_weighted_terms_by(|c| c.ln_pmf(k));
        log_sum_exp(&terms).unwrap_or(f64::NEG_INFINITY)
    }
}

impl<D> Mixture<D>
where
    D: Discrete<u64>,
{
    /// Component log responsibilities for a discrete observation `k`.
    ///
    /// Entry `i` is `log(wᵢ pᵢ(k) / Σⱼ wⱼ pⱼ(k))`, computed in log space.
    pub fn ln_pmf_responsibilities(&self, k: u64) -> Vec<f64> {
        self.ln_responsibilities_by(|c| c.ln_pmf(k))
    }

    /// Component responsibilities for a discrete observation `k`.
    pub fn pmf_responsibilities(&self, k: u64) -> Vec<f64> {
        self.responsibilities_by(|c| c.ln_pmf(k))
    }
}

impl<D> Moments for Mixture<D>
where
    D: Moments,
{
    /// Mixture mean `Σ wᵢ μᵢ`, propagating `NaN` from undefined component means.
    fn mean(&self) -> f64 {
        let mut m = 0.0;
        for (&w, c) in self.weights.iter().zip(&self.components) {
            m += w * c.mean();
        }
        m
    }
    /// Mixture variance by the **law of total variance**, in the one-pass form
    /// `Σ wᵢ(σᵢ² + μᵢ²) − (Σ wᵢ μᵢ)²` — equal to the within + between split
    /// ([`variance_decomposition`](Mixture::variance_decomposition)) and always
    /// nonnegative, proved in `riemann` (`RiemannLean/Bayes/Mixture.lean`:
    /// `law_of_total_variance`, `mixtureVar_nonneg`).
    fn variance(&self) -> f64 {
        let mut e_x2 = 0.0;
        for (&w, c) in self.weights.iter().zip(&self.components) {
            let mi = c.mean();
            let vi = c.variance();
            e_x2 += w * (vi + mi * mi);
        }
        let m = self.mean();
        e_x2 - m * m
    }
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn skewness(&self) -> f64 {
        f64::NAN
    }

    fn kurtosis(&self) -> f64 {
        f64::NAN
    }

    fn entropy(&self) -> f64 {
        f64::NAN
    }

    fn median(&self) -> f64 {
        f64::NAN
    }

    fn mode(&self) -> f64 {
        f64::NAN
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests;
