//! Dirichlet-prior conjugate models: **Dirichlet–Categorical** and
//! **Dirichlet–Multinomial**.

use super::{Conjugate, Counts};
use crate::distribution::{
    Categorical, Dirichlet, DirichletMultinomialDistribution, MultivariateMoments,
};
use crate::function::{ln_factorial, ln_multivariate_beta};
use ramanujan::DenseMatrix;

/// **Dirichlet–Categorical.** A `Dirichlet(α)` prior on category probabilities;
/// posterior `Dirichlet(α + counts)` from observed category labels in `{0,…,K−1}`
/// (labels outside the support are ignored).
#[derive(Clone, Debug, PartialEq)]
pub struct DirichletCategorical {
    prior: Dirichlet,
}

impl DirichletCategorical {
    pub fn new(prior: Dirichlet) -> Self {
        Self { prior }
    }
    pub fn prior(&self) -> &Dirichlet {
        &self.prior
    }

    /// Closed-form log marginal likelihood for categorical labels with the
    /// category probabilities integrated out:
    /// `ln B(α + counts) - ln B(α)`.
    ///
    /// Returns `-∞` if any label is outside the prior's category support.
    pub fn ln_marginal_likelihood(&self, data: &[u64]) -> f64 {
        let k = self.prior.alpha().len();
        let Some(counts) = categorical_counts(data, k) else {
            return f64::NEG_INFINITY;
        };
        let posterior_alpha: Vec<f64> = self
            .prior
            .alpha()
            .iter()
            .zip(counts)
            .map(|(&a, c)| a + c)
            .collect();
        ln_multivariate_beta(&posterior_alpha) - ln_multivariate_beta(self.prior.alpha())
    }

    /// Closed-form prior/posterior predictive log probability for one future
    /// category label under this Dirichlet state.
    pub fn ln_predictive(&self, label: u64) -> f64 {
        let label = label as usize;
        let alpha = self.prior.alpha();
        if label >= alpha.len() {
            return f64::NEG_INFINITY;
        }
        (alpha[label] / self.prior.concentration()).ln()
    }

    /// Closed-form posterior-predictive log probability for one future category
    /// label after observing `data`.
    pub fn ln_posterior_predictive(&self, data: &[u64], label: u64) -> f64 {
        if categorical_counts(data, self.prior.alpha().len()).is_none() {
            return f64::NEG_INFINITY;
        }
        Self::new(self.posterior(data)).ln_predictive(label)
    }

    /// Closed-form log probability of a future category-label batch after
    /// observing `data`.
    pub fn ln_posterior_predictive_likelihood(&self, data: &[u64], future: &[u64]) -> f64 {
        if categorical_counts(data, self.prior.alpha().len()).is_none() {
            return f64::NEG_INFINITY;
        }
        Self::new(self.posterior(data)).ln_marginal_likelihood(future)
    }

    /// Sequential predictive factorization `Σᵢ ln p(cᵢ | c₁,…,cᵢ₋₁)`.
    ///
    /// In exact arithmetic this equals [`ln_marginal_likelihood`](Self::ln_marginal_likelihood).
    pub fn ln_sequential_predictive(&self, data: &[u64]) -> f64 {
        let mut alpha = self.prior.alpha().to_vec();
        let mut total: f64 = alpha.iter().sum();
        let mut ln_prob = 0.0;
        for &label in data {
            let label = label as usize;
            if label >= alpha.len() {
                return f64::NEG_INFINITY;
            }
            ln_prob += (alpha[label] / total).ln();
            alpha[label] += 1.0;
            total += 1.0;
        }
        ln_prob
    }

    /// Posterior-predictive category probabilities `E[p | data]`.
    pub fn posterior_predictive_probabilities(&self, data: &[u64]) -> Vec<f64> {
        self.posterior(data).mean_vector()
    }

    /// Mean count vector for `future_trials` future categorical observations
    /// with the category probabilities integrated out.
    pub fn posterior_predictive_count_mean(&self, data: &[u64], future_trials: u64) -> Vec<f64> {
        self.posterior(data).predictive_count_mean(future_trials)
    }

    /// Full Dirichlet-multinomial posterior-predictive count distribution for
    /// `future_trials` future categorical observations.
    pub fn posterior_predictive_counts(
        &self,
        data: &[u64],
        future_trials: u64,
    ) -> DirichletMultinomialDistribution {
        DirichletMultinomialDistribution::from_dirichlet(future_trials, self.posterior(data))
    }

    /// Dirichlet-multinomial covariance for `future_trials` future categorical
    /// observations with the category probabilities integrated out.
    pub fn posterior_predictive_count_covariance(
        &self,
        data: &[u64],
        future_trials: u64,
    ) -> DenseMatrix {
        self.posterior(data)
            .predictive_count_covariance(future_trials)
    }
}

impl Conjugate for DirichletCategorical {
    type Datum = u64;
    type Posterior = Dirichlet;
    type Predictive = Categorical;
    type Natural = Counts;

    /// A one-hot count for category `c` — the riemann `catStat`. Labels outside
    /// the prior's `K` categories contribute the zero increment (ignored).
    fn statistic(&self, &c: &u64) -> Counts {
        let k = c as usize;
        if k < self.prior.alpha().len() {
            let mut v = vec![0.0; k + 1];
            v[k] = 1.0;
            Counts(v)
        } else {
            Counts(Vec::new())
        }
    }
    fn prior_natural(&self) -> Counts {
        Counts(self.prior.alpha().to_vec())
    }
    fn posterior_from_natural(&self, c: Counts) -> Dirichlet {
        Dirichlet::new(c.0).unwrap_or_else(|_| self.prior.clone())
    }
    /// The posterior-predictive for one future draw: `Categorical(α′/Σα′)`.
    fn posterior_predictive(&self, data: &[u64]) -> Categorical {
        let post = self.posterior(data);
        Categorical::new(post.alpha()).unwrap_or_else(|_| unreachable!())
    }
}

/// **Dirichlet–Multinomial.** A `Dirichlet(α)` prior on category probabilities;
/// posterior `Dirichlet(α + Σ counts)` from observed per-trial **count vectors**.
#[derive(Clone, Debug, PartialEq)]
pub struct DirichletMultinomial {
    prior: Dirichlet,
}

impl DirichletMultinomial {
    pub fn new(prior: Dirichlet) -> Self {
        Self { prior }
    }
    pub fn prior(&self) -> &Dirichlet {
        &self.prior
    }

    /// Closed-form log marginal likelihood for multinomial count vectors with
    /// the category probabilities integrated out:
    ///
    /// `Σ [ln(nᵢ!) - Σₖ ln(cᵢₖ!)] + ln B(α + Σᵢ cᵢ) - ln B(α)`.
    ///
    /// Count vectors may be shorter than the prior support (missing categories
    /// are zeros). Returns `-∞` if a vector has positive counts beyond the prior
    /// support.
    pub fn ln_marginal_likelihood(&self, data: &[Vec<u64>]) -> f64 {
        let k = self.prior.alpha().len();
        let mut coeff = 0.0;
        let mut counts = vec![0.0; k];
        for row in data {
            let Some(row_coeff) = multinomial_count_log_coefficient(row, k) else {
                return f64::NEG_INFINITY;
            };
            coeff += row_coeff;
            for (j, &count) in row.iter().take(k).enumerate() {
                counts[j] += count as f64;
            }
        }
        let posterior_alpha: Vec<f64> = self
            .prior
            .alpha()
            .iter()
            .zip(counts)
            .map(|(&a, c)| a + c)
            .collect();
        coeff + ln_multivariate_beta(&posterior_alpha) - ln_multivariate_beta(self.prior.alpha())
    }

    /// Closed-form prior/posterior predictive log probability for one future
    /// multinomial count vector under this Dirichlet state.
    pub fn ln_predictive(&self, counts: &[u64]) -> f64 {
        let k = self.prior.alpha().len();
        let Some(coeff) = multinomial_count_log_coefficient(counts, k) else {
            return f64::NEG_INFINITY;
        };
        let posterior_alpha: Vec<f64> = self
            .prior
            .alpha()
            .iter()
            .enumerate()
            .map(|(j, &a)| a + counts.get(j).copied().unwrap_or(0) as f64)
            .collect();
        coeff + ln_multivariate_beta(&posterior_alpha) - ln_multivariate_beta(self.prior.alpha())
    }

    /// Closed-form posterior-predictive log probability for one future
    /// multinomial count vector after observing `data`.
    pub fn ln_posterior_predictive(&self, data: &[Vec<u64>], counts: &[u64]) -> f64 {
        if multinomial_count_totals(data, self.prior.alpha().len()).is_none() {
            return f64::NEG_INFINITY;
        }
        Self::new(self.posterior(data)).ln_predictive(counts)
    }

    /// Closed-form log probability of a future batch of multinomial count
    /// vectors after observing `data`.
    pub fn ln_posterior_predictive_likelihood(
        &self,
        data: &[Vec<u64>],
        future: &[Vec<u64>],
    ) -> f64 {
        if multinomial_count_totals(data, self.prior.alpha().len()).is_none() {
            return f64::NEG_INFINITY;
        }
        Self::new(self.posterior(data)).ln_marginal_likelihood(future)
    }

    /// Sequential predictive factorization over count-vector observations.
    ///
    /// In exact arithmetic this equals [`ln_marginal_likelihood`](Self::ln_marginal_likelihood).
    pub fn ln_sequential_predictive(&self, data: &[Vec<u64>]) -> f64 {
        let k = self.prior.alpha().len();
        let mut alpha = self.prior.alpha().to_vec();
        let mut ln_prob = 0.0;
        for row in data {
            let Some(coeff) = multinomial_count_log_coefficient(row, k) else {
                return f64::NEG_INFINITY;
            };
            let posterior_alpha: Vec<f64> = alpha
                .iter()
                .enumerate()
                .map(|(j, &a)| a + row.get(j).copied().unwrap_or(0) as f64)
                .collect();
            ln_prob +=
                coeff + ln_multivariate_beta(&posterior_alpha) - ln_multivariate_beta(&alpha);
            alpha = posterior_alpha;
        }
        ln_prob
    }

    /// Posterior-predictive category probabilities `E[p | data]`.
    pub fn posterior_predictive_probabilities(&self, data: &[Vec<u64>]) -> Vec<f64> {
        self.posterior(data).mean_vector()
    }

    /// Mean count vector for a future multinomial observation with
    /// `future_trials`, integrating out category probabilities.
    pub fn posterior_predictive_count_mean(
        &self,
        data: &[Vec<u64>],
        future_trials: u64,
    ) -> Vec<f64> {
        self.posterior(data).predictive_count_mean(future_trials)
    }

    /// Full Dirichlet-multinomial posterior-predictive count distribution for a
    /// future multinomial observation with `future_trials`.
    pub fn posterior_predictive_counts(
        &self,
        data: &[Vec<u64>],
        future_trials: u64,
    ) -> DirichletMultinomialDistribution {
        DirichletMultinomialDistribution::from_dirichlet(future_trials, self.posterior(data))
    }

    /// Dirichlet-multinomial covariance for a future multinomial observation
    /// with `future_trials`, integrating out category probabilities.
    pub fn posterior_predictive_count_covariance(
        &self,
        data: &[Vec<u64>],
        future_trials: u64,
    ) -> DenseMatrix {
        self.posterior(data)
            .predictive_count_covariance(future_trials)
    }
}

impl Conjugate for DirichletMultinomial {
    type Datum = Vec<u64>;
    type Posterior = Dirichlet;
    type Predictive = Categorical;
    type Natural = Counts;

    /// A per-trial count vector, truncated to the prior's `K` categories (counts
    /// beyond `K` are ignored).
    fn statistic(&self, counts: &Vec<u64>) -> Counts {
        Counts(
            counts
                .iter()
                .take(self.prior.alpha().len())
                .map(|&c| c as f64)
                .collect(),
        )
    }
    fn prior_natural(&self) -> Counts {
        Counts(self.prior.alpha().to_vec())
    }
    fn posterior_from_natural(&self, c: Counts) -> Dirichlet {
        Dirichlet::new(c.0).unwrap_or_else(|_| self.prior.clone())
    }
    /// The posterior-predictive category probabilities for one future trial:
    /// `Categorical(α′/Σα′)`.
    fn posterior_predictive(&self, data: &[Vec<u64>]) -> Categorical {
        let post = self.posterior(data);
        Categorical::new(post.alpha()).unwrap_or_else(|_| unreachable!())
    }
}

fn categorical_counts(data: &[u64], k: usize) -> Option<Vec<f64>> {
    let mut counts = vec![0.0; k];
    for &label in data {
        let label = label as usize;
        if label >= k {
            return None;
        }
        counts[label] += 1.0;
    }
    Some(counts)
}

fn multinomial_count_log_coefficient(row: &[u64], k: usize) -> Option<f64> {
    if row.iter().skip(k).any(|&c| c > 0) {
        return None;
    }
    let n = row.iter().sum::<u64>();
    Some(
        ln_factorial(n)
            - row
                .iter()
                .take(k)
                .map(|&count| ln_factorial(count))
                .sum::<f64>(),
    )
}

fn multinomial_count_totals(data: &[Vec<u64>], k: usize) -> Option<Vec<f64>> {
    let mut counts = vec![0.0; k];
    for row in data {
        multinomial_count_log_coefficient(row, k)?;
        for (j, &count) in row.iter().take(k).enumerate() {
            counts[j] += count as f64;
        }
    }
    Some(counts)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn dirichlet_categorical_counts_and_predictive() {
        let model = DirichletCategorical::new(Dirichlet::new(vec![1.0, 1.0, 1.0]).unwrap());
        let data = [0u64, 0, 1, 2, 0];
        let post = model.posterior(&data);
        assert_eq!(post.alpha(), &[4.0, 2.0, 2.0]);
        let pred = model.posterior_predictive(&data);
        close(pred.probabilities()[0], 0.5, 1e-12);
        close(pred.probabilities()[1], 0.25, 1e-12);
        let probs = model.posterior_predictive_probabilities(&data);
        close(probs[0], 0.5, 1e-12);
        close(model.ln_predictive(0).exp(), 1.0 / 3.0, 1e-12);
        close(model.ln_posterior_predictive(&data, 0).exp(), 0.5, 1e-12);
        let mean = model.posterior_predictive_count_mean(&data, 4);
        close(mean[0], 2.0, 1e-12);
        let counts = model.posterior_predictive_counts(&data, 4);
        close(counts.mean_vector()[0], 2.0, 1e-12);
        assert!(counts.pmf(&[2, 1, 1]) > 0.0);
        let cov = model.posterior_predictive_count_covariance(&data, 4);
        close(cov.get(0, 1), -4.0 * 0.5 * 0.25 * 12.0 / 9.0, 1e-12);
        close(
            model.ln_marginal_likelihood(&data),
            ln_multivariate_beta(&[4.0, 2.0, 2.0]) - ln_multivariate_beta(&[1.0, 1.0, 1.0]),
            1e-12,
        );
        close(
            model.ln_sequential_predictive(&data),
            model.ln_marginal_likelihood(&data),
            1e-12,
        );
        close(
            model.ln_posterior_predictive_likelihood(&data, &[0, 1]),
            DirichletCategorical::new(post).ln_marginal_likelihood(&[0, 1]),
            1e-12,
        );
        assert_eq!(model.ln_marginal_likelihood(&[3]), f64::NEG_INFINITY);
        assert_eq!(model.ln_predictive(3), f64::NEG_INFINITY);
        assert_eq!(model.ln_posterior_predictive(&[3], 0), f64::NEG_INFINITY);
    }

    #[test]
    fn dirichlet_multinomial_aggregates_count_vectors() {
        let model = DirichletMultinomial::new(Dirichlet::new(vec![1.0, 1.0]).unwrap());
        let data = [vec![3u64, 1], vec![2, 2]];
        let post = model.posterior(&data);
        assert_eq!(post.alpha(), &[6.0, 4.0]);
        let pred = model.posterior_predictive(&data);
        close(pred.probabilities()[0], 0.6, 1e-12);
        close(pred.probabilities()[1], 0.4, 1e-12);
        close(model.ln_predictive(&[1, 0]).exp(), 0.5, 1e-12);
        close(
            model.ln_posterior_predictive(&data, &[1, 0]).exp(),
            0.6,
            1e-12,
        );
        let probs = model.posterior_predictive_probabilities(&data);
        close(probs[0], 0.6, 1e-12);
        let mean = model.posterior_predictive_count_mean(&data, 5);
        close(mean[0], 3.0, 1e-12);
        let counts = model.posterior_predictive_counts(&data, 5);
        close(counts.mean_vector()[0], 3.0, 1e-12);
        assert!(counts.pmf(&[3, 2]) > 0.0);
        let cov = model.posterior_predictive_count_covariance(&data, 5);
        close(cov.get(0, 0), 5.0 * 0.6 * 0.4 * 15.0 / 11.0, 1e-12);
        let coeff = ln_factorial(4) - ln_factorial(3) - ln_factorial(1) + ln_factorial(4)
            - ln_factorial(2)
            - ln_factorial(2);
        close(
            model.ln_marginal_likelihood(&data),
            coeff + ln_multivariate_beta(&[6.0, 4.0]) - ln_multivariate_beta(&[1.0, 1.0]),
            1e-12,
        );
        close(
            model.ln_sequential_predictive(&data),
            model.ln_marginal_likelihood(&data),
            1e-12,
        );
        close(
            model.ln_posterior_predictive_likelihood(&data, &[vec![3, 2]]),
            DirichletMultinomial::new(post).ln_marginal_likelihood(&[vec![3, 2]]),
            1e-12,
        );
        assert_eq!(
            model.ln_marginal_likelihood(&[vec![0, 0, 1]]),
            f64::NEG_INFINITY
        );
        assert_eq!(model.ln_predictive(&[0, 0, 1]), f64::NEG_INFINITY);
    }
}
