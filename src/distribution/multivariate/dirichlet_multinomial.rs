use rand::Rng;
use rand::distr::Distribution;

use ramanujan::DenseMatrix;

use super::{Dirichlet, Multinomial, MultivariateMoments};
use crate::error::BayesError;
use crate::function::{ln_factorial, ln_multivariate_beta};

/// The Dirichlet-multinomial distribution over count vectors.
///
/// It is the closed-form marginal of `C | p ~ Multinomial(n, p)` with
/// `p ~ Dirichlet(α)`, and the finite-count posterior predictive distribution
/// for Dirichlet-categorical/multinomial conjugate models.
#[derive(Clone, Debug, PartialEq)]
pub struct DirichletMultinomialDistribution {
    trials: u64,
    dirichlet: Dirichlet,
}

impl DirichletMultinomialDistribution {
    pub fn new(trials: u64, alpha: Vec<f64>) -> Result<Self, BayesError> {
        Ok(Self {
            trials,
            dirichlet: Dirichlet::new(alpha)?,
        })
    }

    pub fn from_dirichlet(trials: u64, dirichlet: Dirichlet) -> Self {
        Self { trials, dirichlet }
    }

    pub fn trials(&self) -> u64 {
        self.trials
    }

    pub fn alpha(&self) -> &[f64] {
        self.dirichlet.alpha()
    }

    pub fn concentration(&self) -> f64 {
        self.dirichlet.concentration()
    }

    pub fn categories(&self) -> usize {
        self.alpha().len()
    }

    /// Component variances of the count vector.
    pub fn variance_vector(&self) -> Vec<f64> {
        let n = self.trials as f64;
        let a0 = self.concentration();
        let inflation = (n + a0) / (1.0 + a0);
        self.dirichlet
            .mean_vector()
            .into_iter()
            .map(|p| n * p * (1.0 - p) * inflation)
            .collect()
    }

    /// Covariance entry for category counts.
    ///
    /// Returns `NaN` for out-of-range indices.
    pub fn covariance_entry(&self, i: usize, j: usize) -> f64 {
        if i >= self.categories() || j >= self.categories() {
            return f64::NAN;
        }
        self.covariance().get(i, j)
    }

    /// `ln P(counts)` for a category-count vector summing to `trials`.
    ///
    /// Returns `-∞` if the counts have the wrong length or total.
    pub fn ln_pmf(&self, counts: &[u64]) -> f64 {
        if counts.len() != self.categories() || counts.iter().sum::<u64>() != self.trials {
            return f64::NEG_INFINITY;
        }
        let alpha_post: Vec<f64> = self
            .alpha()
            .iter()
            .zip(counts)
            .map(|(&a, &c)| a + c as f64)
            .collect();
        ln_factorial(self.trials) - counts.iter().map(|&c| ln_factorial(c)).sum::<f64>()
            + ln_multivariate_beta(&alpha_post)
            - ln_multivariate_beta(self.alpha())
    }

    pub fn pmf(&self, counts: &[u64]) -> f64 {
        self.ln_pmf(counts).exp()
    }
}

impl Distribution<Vec<u64>> for DirichletMultinomialDistribution {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<u64> {
        let probabilities = self.dirichlet.sample(rng);
        Multinomial::new(self.trials, &probabilities)
            .unwrap_or_else(|_| unreachable!())
            .sample(rng)
    }
}

impl MultivariateMoments for DirichletMultinomialDistribution {
    fn mean_vector(&self) -> Vec<f64> {
        self.dirichlet.predictive_count_mean(self.trials)
    }

    fn covariance(&self) -> DenseMatrix {
        self.dirichlet.predictive_count_covariance(self.trials)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::distribution::{BetaBinomial, Discrete};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn two_category_form_matches_beta_binomial() {
        let dm = DirichletMultinomialDistribution::new(5, vec![2.0, 3.0]).unwrap();
        let bb = BetaBinomial::new(5, 2.0, 3.0).unwrap();
        close(dm.pmf(&[2, 3]), bb.pmf(2), 1e-12);
        close(dm.ln_pmf(&[2, 3]), bb.ln_pmf(2), 1e-12);
        assert_eq!(dm.pmf(&[2, 2]), 0.0);
        assert_eq!(dm.pmf(&[2, 2, 1]), 0.0);
    }

    #[test]
    fn moments_match_dirichlet_multinomial_closed_forms() {
        let dm = DirichletMultinomialDistribution::new(4, vec![2.0, 3.0, 5.0]).unwrap();
        let mean = dm.mean_vector();
        close(mean[0], 0.8, 1e-12);
        close(mean[2], 2.0, 1e-12);
        let inflation = 14.0 / 11.0;
        close(dm.variance_vector()[0], 4.0 * 0.2 * 0.8 * inflation, 1e-12);
        close(
            dm.covariance_entry(0, 2),
            -4.0 * 0.2 * 0.5 * inflation,
            1e-12,
        );
        assert!(dm.covariance_entry(0, 3).is_nan());
    }

    #[test]
    fn sampled_counts_have_the_right_total() {
        let dm = DirichletMultinomialDistribution::new(8, vec![2.0, 3.0, 5.0]).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(31);
        for _ in 0..100 {
            let counts = dm.sample(&mut rng);
            assert_eq!(counts.len(), 3);
            assert_eq!(counts.iter().sum::<u64>(), 8);
        }
    }
}
