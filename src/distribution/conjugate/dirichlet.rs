//! Dirichlet-prior conjugate models: **Dirichlet–Categorical** and
//! **Dirichlet–Multinomial**.

use super::{Conjugate, Counts};
use crate::distribution::{Categorical, Dirichlet};
use crate::function::{ln_factorial, ln_multivariate_beta};

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
        let mut counts = vec![0.0; k];
        for &label in data {
            let label = label as usize;
            if label >= k {
                return f64::NEG_INFINITY;
            }
            counts[label] += 1.0;
        }
        let posterior_alpha: Vec<f64> = self
            .prior
            .alpha()
            .iter()
            .zip(counts)
            .map(|(&a, c)| a + c)
            .collect();
        ln_multivariate_beta(&posterior_alpha) - ln_multivariate_beta(self.prior.alpha())
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
        let mut counts = vec![0.0; k];
        let mut coeff = 0.0;
        for row in data {
            if row.iter().skip(k).any(|&c| c > 0) {
                return f64::NEG_INFINITY;
            }
            let n = row.iter().sum::<u64>();
            coeff += ln_factorial(n);
            for (j, &count) in row.iter().take(k).enumerate() {
                coeff -= ln_factorial(count);
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
        close(
            model.ln_marginal_likelihood(&data),
            ln_multivariate_beta(&[4.0, 2.0, 2.0]) - ln_multivariate_beta(&[1.0, 1.0, 1.0]),
            1e-12,
        );
        assert_eq!(model.ln_marginal_likelihood(&[3]), f64::NEG_INFINITY);
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
        let coeff = ln_factorial(4) - ln_factorial(3) - ln_factorial(1) + ln_factorial(4)
            - ln_factorial(2)
            - ln_factorial(2);
        close(
            model.ln_marginal_likelihood(&data),
            coeff + ln_multivariate_beta(&[6.0, 4.0]) - ln_multivariate_beta(&[1.0, 1.0]),
            1e-12,
        );
        assert_eq!(
            model.ln_marginal_likelihood(&[vec![0, 0, 1]]),
            f64::NEG_INFINITY
        );
    }
}
