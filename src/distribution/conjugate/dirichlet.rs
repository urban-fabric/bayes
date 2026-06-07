//! Dirichlet-prior conjugate models: **Dirichlet–Categorical** and
//! **Dirichlet–Multinomial**.

use super::Conjugate;
use crate::distribution::{Categorical, Dirichlet};

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
    /// The posterior-predictive for one future draw: `Categorical(α′/Σα′)`.
    pub fn posterior_predictive(&self, data: &[u64]) -> Categorical {
        let post = self.posterior(data);
        Categorical::new(post.alpha()).unwrap_or_else(|_| unreachable!())
    }
}

impl Conjugate for DirichletCategorical {
    type Datum = u64;
    type Posterior = Dirichlet;
    fn posterior(&self, data: &[u64]) -> Dirichlet {
        let mut alpha = self.prior.alpha().to_vec();
        for &k in data {
            if let Some(a) = alpha.get_mut(k as usize) {
                *a += 1.0;
            }
        }
        Dirichlet::new(alpha).unwrap_or_else(|_| self.prior.clone())
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
    /// The posterior-predictive category probabilities for one future trial:
    /// `Categorical(α′/Σα′)`.
    pub fn posterior_predictive(&self, data: &[Vec<u64>]) -> Categorical {
        let post = self.posterior(data);
        Categorical::new(post.alpha()).unwrap_or_else(|_| unreachable!())
    }
}

impl Conjugate for DirichletMultinomial {
    type Datum = Vec<u64>;
    type Posterior = Dirichlet;
    fn posterior(&self, data: &[Vec<u64>]) -> Dirichlet {
        let mut alpha = self.prior.alpha().to_vec();
        for counts in data {
            for (i, &c) in counts.iter().enumerate() {
                if let Some(a) = alpha.get_mut(i) {
                    *a += c as f64;
                }
            }
        }
        Dirichlet::new(alpha).unwrap_or_else(|_| self.prior.clone())
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
    }
}
