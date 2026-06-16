use rand::Rng;
use rand::distr::Distribution;

use ramanujan::DenseMatrix;

use super::{MultivariateContinuous, MultivariateMoments};
use crate::distribution::continuous::Beta;
use crate::distribution::continuous::gamma::sample_standard_gamma;
use crate::error::BayesError;
use crate::function::gamma::{digamma, ln_gamma};

/// The Dirichlet distribution over the `k`-simplex (vectors of non-negative
/// components summing to 1), with concentration parameters `α₁, …, α_k > 0`.
#[derive(Clone, Debug, PartialEq)]
pub struct Dirichlet {
    alpha: Vec<f64>,
}

impl Dirichlet {
    pub fn new(alpha: Vec<f64>) -> Result<Self, BayesError> {
        if alpha.len() < 2 {
            return Err(BayesError::Malformed {
                what: "Dirichlet needs at least two components",
            });
        }
        for &a in &alpha {
            if !a.is_finite() || a <= 0.0 {
                return Err(BayesError::NotPositive {
                    name: "alpha",
                    value: a,
                });
            }
        }
        Ok(Self { alpha })
    }

    pub fn alpha(&self) -> &[f64] {
        &self.alpha
    }

    /// Total concentration `α₀ = Σ αᵢ`.
    pub fn concentration(&self) -> f64 {
        self.alpha.iter().sum()
    }

    /// Alias for [`concentration`](Self::concentration).
    pub fn alpha_sum(&self) -> f64 {
        self.concentration()
    }

    /// Marginal distribution of component `P_i`.
    ///
    /// If `P ~ Dirichlet(α)`, then `P_i ~ Beta(α_i, α₀ - α_i)`.
    pub fn component_marginal(&self, index: usize) -> Result<Beta, BayesError> {
        let Some(&alpha_i) = self.alpha.get(index) else {
            return Err(BayesError::Malformed {
                what: "Dirichlet component index out of range",
            });
        };
        Beta::new(alpha_i, self.concentration() - alpha_i)
    }

    /// Marginal Beta distributions of all components.
    pub fn component_marginals(&self) -> Vec<Beta> {
        let a0 = self.concentration();
        self.alpha
            .iter()
            .map(|&a| Beta::new(a, a0 - a).unwrap_or_else(|_| unreachable!()))
            .collect()
    }

    /// Marginal distribution of the sum of a proper non-empty subset:
    /// `Σ_{i∈S} P_i ~ Beta(Σ_{i∈S} α_i, α₀ - Σ_{i∈S} α_i)`.
    pub fn sum_marginal(&self, indices: &[usize]) -> Result<Beta, BayesError> {
        let selected = self.subset_alpha_sum(indices)?;
        Beta::new(selected, self.concentration() - selected)
    }

    /// Lump categories by a partition of the original coordinates.
    ///
    /// If `P ~ Dirichlet(α)`, then grouped sums over a disjoint exhaustive
    /// partition are again Dirichlet with concentrations equal to group sums.
    pub fn lump(&self, groups: &[Vec<usize>]) -> Result<Self, BayesError> {
        if groups.len() < 2 {
            return Err(BayesError::Malformed {
                what: "Dirichlet lumping needs at least two groups",
            });
        }
        let mut seen = vec![false; self.alpha.len()];
        let mut alpha = Vec::with_capacity(groups.len());
        for group in groups {
            if group.is_empty() {
                return Err(BayesError::Malformed {
                    what: "Dirichlet lumping groups must be non-empty",
                });
            }
            let mut sum = 0.0;
            for &idx in group {
                if idx >= self.alpha.len() || seen[idx] {
                    return Err(BayesError::Malformed {
                        what: "Dirichlet lumping groups must form a disjoint partition",
                    });
                }
                seen[idx] = true;
                sum += self.alpha[idx];
            }
            alpha.push(sum);
        }
        if seen.iter().any(|&hit| !hit) {
            return Err(BayesError::Malformed {
                what: "Dirichlet lumping groups must cover every component",
            });
        }
        Self::new(alpha)
    }

    /// Conditional law of the normalized selected coordinates given their sum.
    ///
    /// For `|S| >= 2`, `(P_i / Σ_{j∈S} P_j)_{i∈S} ~ Dirichlet((α_i)_{i∈S})`
    /// and is independent of the subset sum.
    pub fn conditional_on_subset(&self, indices: &[usize]) -> Result<Self, BayesError> {
        if indices.len() < 2 {
            return Err(BayesError::Malformed {
                what: "Dirichlet conditional subset needs at least two components",
            });
        }
        let mut seen = vec![false; self.alpha.len()];
        let mut alpha = Vec::with_capacity(indices.len());
        for &idx in indices {
            if idx >= self.alpha.len() || seen[idx] {
                return Err(BayesError::Malformed {
                    what: "Dirichlet subset indices must be distinct and in range",
                });
            }
            seen[idx] = true;
            alpha.push(self.alpha[idx]);
        }
        Self::new(alpha)
    }

    /// Component variances of a Dirichlet random probability vector.
    pub fn variance_vector(&self) -> Vec<f64> {
        let a0 = self.concentration();
        let denom = a0 * a0 * (a0 + 1.0);
        self.alpha.iter().map(|&a| a * (a0 - a) / denom).collect()
    }

    /// Covariance entry `Cov[P_i, P_j]`.
    ///
    /// Returns `NaN` for out-of-range indices.
    pub fn covariance_entry(&self, i: usize, j: usize) -> f64 {
        if i >= self.alpha.len() || j >= self.alpha.len() {
            return f64::NAN;
        }
        self.covariance().get(i, j)
    }

    /// `E[ln P_i] = ψ(α_i) - ψ(α₀)` for each component.
    pub fn expected_log_probabilities(&self) -> Vec<f64> {
        let a0 = self.concentration();
        self.alpha
            .iter()
            .map(|&a| digamma(a) - digamma(a0))
            .collect()
    }

    /// Mean count vector for a future Dirichlet-multinomial draw with `trials`.
    pub fn predictive_count_mean(&self, trials: u64) -> Vec<f64> {
        let n = trials as f64;
        self.mean_vector().into_iter().map(|p| n * p).collect()
    }

    /// Covariance matrix for a future Dirichlet-multinomial count vector with
    /// `trials`, after integrating out the category probabilities.
    pub fn predictive_count_covariance(&self, trials: u64) -> DenseMatrix {
        let n = trials as f64;
        let a0 = self.concentration();
        let p = self.mean_vector();
        let inflation = (n + a0) / (1.0 + a0);
        DenseMatrix::from_fn(self.alpha.len(), self.alpha.len(), |i, j| {
            if i == j {
                n * p[i] * (1.0 - p[i]) * inflation
            } else {
                -n * p[i] * p[j] * inflation
            }
        })
    }

    fn subset_alpha_sum(&self, indices: &[usize]) -> Result<f64, BayesError> {
        if indices.is_empty() {
            return Err(BayesError::Malformed {
                what: "Dirichlet subset must be non-empty",
            });
        }
        let mut seen = vec![false; self.alpha.len()];
        let mut sum = 0.0;
        for &idx in indices {
            if idx >= self.alpha.len() || seen[idx] {
                return Err(BayesError::Malformed {
                    what: "Dirichlet subset indices must be distinct and in range",
                });
            }
            seen[idx] = true;
            sum += self.alpha[idx];
        }
        if seen.iter().all(|&hit| hit) {
            return Err(BayesError::Malformed {
                what: "Dirichlet subset must be a proper subset",
            });
        }
        Ok(sum)
    }
}

impl Distribution<Vec<f64>> for Dirichlet {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<f64> {
        let y: Vec<f64> = self
            .alpha
            .iter()
            .map(|&a| sample_standard_gamma(rng, a))
            .collect();
        let sum: f64 = y.iter().sum();
        y.iter().map(|&yi| yi / sum).collect()
    }
}

impl MultivariateContinuous for Dirichlet {
    fn dim(&self) -> usize {
        self.alpha.len()
    }
    fn ln_pdf(&self, x: &[f64]) -> f64 {
        if x.len() != self.alpha.len()
            || x.iter().any(|&xi| !xi.is_finite() || xi <= 0.0)
            || (x.iter().sum::<f64>() - 1.0).abs() > 1e-10
        {
            return f64::NEG_INFINITY;
        }
        let ln_norm =
            ln_gamma(self.alpha_sum()) - self.alpha.iter().map(|&a| ln_gamma(a)).sum::<f64>();
        ln_norm
            + self
                .alpha
                .iter()
                .zip(x)
                .map(|(&a, &xi)| (a - 1.0) * xi.ln())
                .sum::<f64>()
    }
}

impl MultivariateMoments for Dirichlet {
    fn mean_vector(&self) -> Vec<f64> {
        let a0 = self.concentration();
        self.alpha.iter().map(|&a| a / a0).collect()
    }
    fn covariance(&self) -> DenseMatrix {
        let a0 = self.concentration();
        let denom = a0 * a0 * (a0 + 1.0);
        let k = self.alpha.len();
        DenseMatrix::from_fn(k, k, |i, j| {
            if i == j {
                self.alpha[i] * (a0 - self.alpha[i]) / denom
            } else {
                -self.alpha[i] * self.alpha[j] / denom
            }
        })
    }
    /// `H = ln B(α) + (α₀ − K)ψ(α₀) − Σ(αᵢ − 1)ψ(αᵢ)`, with
    /// `ln B(α) = Σ ln Γ(αᵢ) − ln Γ(α₀)`.
    fn entropy(&self) -> Option<f64> {
        let a0 = self.concentration();
        let k = self.alpha.len() as f64;
        let ln_b = self.alpha.iter().map(|&a| ln_gamma(a)).sum::<f64>() - ln_gamma(a0);
        let sum_terms: f64 = self.alpha.iter().map(|&a| (a - 1.0) * digamma(a)).sum();
        Some(ln_b + (a0 - k) * digamma(a0) - sum_terms)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn mean_and_samples_lie_on_the_simplex() {
        let d = Dirichlet::new(vec![1.0, 2.0, 3.0]).unwrap();
        let m = d.mean_vector();
        close(m[0], 1.0 / 6.0, 1e-12);
        close(m.iter().sum::<f64>(), 1.0, 1e-12);

        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let mut avg = [0.0; 3];
        let n = 40_000;
        for _ in 0..n {
            let x = d.sample(&mut rng);
            close(x.iter().sum::<f64>(), 1.0, 1e-12);
            for i in 0..3 {
                avg[i] += x[i] / n as f64;
            }
        }
        for i in 0..3 {
            close(avg[i], m[i], 0.02);
        }
    }

    #[test]
    fn uniform_dirichlet_has_constant_density() {
        let d = Dirichlet::new(vec![1.0, 1.0, 1.0]).unwrap();
        close(d.pdf(&[0.2, 0.3, 0.5]), 2.0, 1e-10);
        close(d.pdf(&[0.6, 0.1, 0.3]), 2.0, 1e-10);
    }

    #[test]
    fn density_rejects_off_simplex_points() {
        let d = Dirichlet::new(vec![1.0, 1.0, 1.0]).unwrap();
        assert_eq!(d.pdf(&[0.2, 0.3, 0.6]), 0.0);
        assert_eq!(d.pdf(&[0.2, f64::NAN, 0.8]), 0.0);
    }

    #[test]
    fn closed_form_log_probability_and_predictive_count_moments() {
        let d = Dirichlet::new(vec![2.0, 3.0, 5.0]).unwrap();
        close(d.concentration(), 10.0, 1e-12);
        let elog = d.expected_log_probabilities();
        close(elog[0], digamma(2.0) - digamma(10.0), 1e-12);
        close(d.variance_vector()[1], 3.0 * 7.0 / (100.0 * 11.0), 1e-12);
        close(d.covariance_entry(0, 2), -2.0 * 5.0 / (100.0 * 11.0), 1e-12);

        let mean = d.predictive_count_mean(4);
        close(mean[0], 0.8, 1e-12);
        close(mean[2], 2.0, 1e-12);
        let cov = d.predictive_count_covariance(4);
        let inflation = 14.0 / 11.0;
        close(cov.get(0, 0), 4.0 * 0.2 * 0.8 * inflation, 1e-12);
        close(cov.get(0, 2), -4.0 * 0.2 * 0.5 * inflation, 1e-12);
    }

    #[test]
    fn simplex_marginals_lumping_and_conditioning_follow_dirichlet_algebra() {
        let d = Dirichlet::new(vec![2.0, 3.0, 5.0, 7.0]).unwrap();

        let component = d.component_marginal(2).unwrap();
        close(component.alpha(), 5.0, 1e-12);
        close(component.beta(), 12.0, 1e-12);
        close(d.component_marginals()[0].beta(), 15.0, 1e-12);

        let selected_sum = d.sum_marginal(&[0, 2]).unwrap();
        close(selected_sum.alpha(), 7.0, 1e-12);
        close(selected_sum.beta(), 10.0, 1e-12);

        let lumped = d.lump(&[vec![0, 2], vec![1], vec![3]]).unwrap();
        assert_eq!(lumped.alpha(), &[7.0, 3.0, 7.0]);
        close(lumped.mean_vector()[0], 7.0 / 17.0, 1e-12);

        let conditional = d.conditional_on_subset(&[3, 1, 0]).unwrap();
        assert_eq!(conditional.alpha(), &[7.0, 3.0, 2.0]);
        close(conditional.mean_vector()[0], 7.0 / 12.0, 1e-12);

        assert!(d.component_marginal(4).is_err());
        assert!(d.sum_marginal(&[]).is_err());
        assert!(d.sum_marginal(&[0, 0]).is_err());
        assert!(d.sum_marginal(&[0, 1, 2, 3]).is_err());
        assert!(d.lump(&[vec![0, 1], vec![1, 2], vec![3]]).is_err());
        assert!(d.lump(&[vec![0, 1], vec![2]]).is_err());
        assert!(d.conditional_on_subset(&[1]).is_err());
    }
}
