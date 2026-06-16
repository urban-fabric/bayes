//! Multivariate Gaussian conjugacy: **Normal–Inverse-Wishart**.

use std::f64::consts::TAU;

use ramanujan::DenseMatrix;

use crate::distribution::multivariate::matrix;
use crate::distribution::{InverseWishart, MultivariateNormal, MultivariateStudentT};
use crate::error::BayesError;
use crate::function::ln_multivariate_gamma;

/// Normal–Inverse-Wishart prior for a multivariate Normal with unknown mean and
/// covariance:
///
/// `Σ ~ InvWishart(ν, Ψ)` and `μ | Σ ~ N(μ₀, Σ/κ)`.
///
/// The posterior after finite observations is another NIW, and the posterior
/// predictive is a multivariate Student-t. All updates are direct matrix algebra.
#[derive(Clone, Debug, PartialEq)]
pub struct NormalInverseWishart {
    mean: Vec<f64>,
    kappa: f64,
    freedom: f64,
    scale: DenseMatrix,
}

impl NormalInverseWishart {
    /// Build a Normal–Inverse-Wishart prior. Requires `κ > 0`, `ν > d - 1`,
    /// finite `mean`, and an SPD `scale` matrix matching the mean dimension.
    pub fn new(
        mean: Vec<f64>,
        kappa: f64,
        freedom: f64,
        scale: DenseMatrix,
    ) -> Result<Self, BayesError> {
        let d = mean.len();
        if d == 0 || scale.rows() != d || scale.cols() != d {
            return Err(BayesError::Malformed {
                what: "scale must be square and match the mean",
            });
        }
        if mean.iter().any(|x| !x.is_finite()) {
            return Err(BayesError::Malformed {
                what: "mean entries must be finite",
            });
        }
        if !kappa.is_finite() {
            return Err(BayesError::NonFinite {
                name: "kappa",
                value: kappa,
            });
        }
        if kappa <= 0.0 {
            return Err(BayesError::NotPositive {
                name: "kappa",
                value: kappa,
            });
        }
        InverseWishart::new(freedom, scale.clone())?;
        Ok(Self {
            mean,
            kappa,
            freedom,
            scale,
        })
    }

    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    pub fn kappa(&self) -> f64 {
        self.kappa
    }

    pub fn freedom(&self) -> f64 {
        self.freedom
    }

    pub fn scale_matrix(&self) -> &DenseMatrix {
        &self.scale
    }

    pub fn dimension(&self) -> usize {
        self.mean.len()
    }

    /// The inverse-Wishart marginal prior/posterior over the covariance matrix.
    pub fn covariance_marginal(&self) -> InverseWishart {
        InverseWishart::new(self.freedom, self.scale.clone()).unwrap_or_else(|_| unreachable!())
    }

    /// Conditional distribution of the mean `μ | Σ`.
    pub fn mean_conditional(
        &self,
        covariance: &DenseMatrix,
    ) -> Result<MultivariateNormal, BayesError> {
        if covariance.rows() != self.dimension() || covariance.cols() != self.dimension() {
            return Err(BayesError::Malformed {
                what: "covariance must match NIW dimension",
            });
        }
        MultivariateNormal::new(
            self.mean.clone(),
            matrix::scaled(covariance, 1.0 / self.kappa),
        )
    }

    /// Closed-form posterior NIW after observing rows `x_i`.
    pub fn posterior(&self, data: &[Vec<f64>]) -> Result<Self, BayesError> {
        let d = self.dimension();
        if data
            .iter()
            .any(|x| x.len() != d || x.iter().any(|v| !v.is_finite()))
        {
            return Err(BayesError::Malformed {
                what: "observations must be finite and match NIW dimension",
            });
        }
        if data.is_empty() {
            return Ok(self.clone());
        }

        let n = data.len() as f64;
        let xbar = mean_vector(data, d);
        let scatter = scatter_matrix(data, &xbar);
        let kappa_n = self.kappa + n;
        let freedom_n = self.freedom + n;
        let mean_n: Vec<f64> = (0..d)
            .map(|i| (self.kappa * self.mean[i] + n * xbar[i]) / kappa_n)
            .collect();
        let delta: Vec<f64> = (0..d).map(|i| xbar[i] - self.mean[i]).collect();
        let between = matrix::scaled(&matrix::outer(&delta, &delta), self.kappa * n / kappa_n);
        let scale_n =
            matrix::symmetrize(&matrix::add(&matrix::add(&self.scale, &scatter), &between));
        Self::new(mean_n, kappa_n, freedom_n, scale_n)
    }

    /// Prior/posterior predictive multivariate Student-t for one future
    /// observation.
    pub fn predictive(&self) -> MultivariateStudentT {
        let d = self.dimension();
        let freedom = self.freedom - d as f64 + 1.0;
        let scale = matrix::scaled(&self.scale, (self.kappa + 1.0) / (self.kappa * freedom));
        MultivariateStudentT::new(freedom, self.mean.clone(), scale)
            .unwrap_or_else(|_| unreachable!())
    }

    /// Posterior predictive after observing `data`.
    pub fn posterior_predictive(
        &self,
        data: &[Vec<f64>],
    ) -> Result<MultivariateStudentT, BayesError> {
        Ok(self.posterior(data)?.predictive())
    }

    /// Closed-form log marginal likelihood with the unknown mean/covariance
    /// integrated out.
    pub fn ln_marginal_likelihood(&self, data: &[Vec<f64>]) -> f64 {
        let Ok(post) = self.posterior(data) else {
            return f64::NEG_INFINITY;
        };
        let d = self.dimension();
        let n = data.len() as f64;
        let Ok(ln_det_0) = matrix::ln_det_spd(&self.scale) else {
            return f64::NAN;
        };
        let Ok(ln_det_n) = matrix::ln_det_spd(&post.scale) else {
            return f64::NAN;
        };
        -0.5 * n * d as f64 * TAU.ln()
            + 0.5 * d as f64 * (self.kappa.ln() - post.kappa.ln())
            + 0.5 * self.freedom * ln_det_0
            - 0.5 * post.freedom * ln_det_n
            + ln_multivariate_gamma(d, post.freedom / 2.0)
            - ln_multivariate_gamma(d, self.freedom / 2.0)
    }
}

fn mean_vector(data: &[Vec<f64>], d: usize) -> Vec<f64> {
    let n = data.len() as f64;
    (0..d)
        .map(|j| data.iter().map(|x| x[j]).sum::<f64>() / n)
        .collect()
}

fn scatter_matrix(data: &[Vec<f64>], mean: &[f64]) -> DenseMatrix {
    let d = mean.len();
    DenseMatrix::from_fn(d, d, |i, j| {
        data.iter()
            .map(|x| (x[i] - mean[i]) * (x[j] - mean[j]))
            .sum()
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::distribution::MultivariateContinuous;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn posterior_matches_the_niw_closed_form() {
        let prior =
            NormalInverseWishart::new(vec![0.0, 0.0], 1.0, 4.0, DenseMatrix::identity(2)).unwrap();
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let post = prior.posterior(&data).unwrap();

        close(post.kappa(), 3.0, 1e-12);
        close(post.freedom(), 6.0, 1e-12);
        close(post.mean()[0], 4.0 / 3.0, 1e-12);
        close(post.mean()[1], 2.0, 1e-12);
        close(post.scale_matrix().get(0, 0), 17.0 / 3.0, 1e-12);
        close(post.scale_matrix().get(0, 1), 6.0, 1e-12);
        close(post.scale_matrix().get(1, 1), 9.0, 1e-12);

        let pred = post.predictive();
        close(pred.freedom(), 5.0, 1e-12);
        close(pred.location()[0], 4.0 / 3.0, 1e-12);
        close(pred.scale_matrix().get(0, 0), 68.0 / 45.0, 1e-12);
        close(pred.scale_matrix().get(0, 1), 1.6, 1e-12);
        close(pred.scale_matrix().get(1, 1), 2.4, 1e-12);
        assert!(pred.ln_pdf(&[1.5, 2.5]).is_finite());

        close(prior.ln_marginal_likelihood(&[]), 0.0, 1e-12);
        assert!(prior.ln_marginal_likelihood(&data).is_finite());
    }

    #[test]
    fn rejects_malformed_data_or_parameters() {
        assert!(
            NormalInverseWishart::new(vec![0.0, 0.0], 0.0, 4.0, DenseMatrix::identity(2)).is_err()
        );
        let prior =
            NormalInverseWishart::new(vec![0.0, 0.0], 1.0, 4.0, DenseMatrix::identity(2)).unwrap();
        assert!(prior.posterior(&[vec![1.0]]).is_err());
    }
}
