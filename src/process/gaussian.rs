//! **Gaussian processes** — distributions over functions. A GP is defined by a
//! constant mean and a covariance [`Kernel`]; you can sample a realization at any
//! set of inputs (the GP *prior*) or condition on observations to get the
//! closed-form *posterior* predictive (GP regression).
//!
//! Inputs are points in `ℝ^d` (each a `Vec<f64>`); the GP draws a value per input
//! by sampling the multivariate normal `N(m·1, K)` via the Cholesky factor.
#![allow(clippy::needless_range_loop)] // matrix/vector index arithmetic reads clearer here

use rand::Rng;

use super::kernel::Kernel;
use crate::distribution::normal::standard_normal_polar;
use crate::linalg::{Matrix, cholesky, cholesky_solve, solve_lower};

const DEFAULT_JITTER: f64 = 1e-9;

fn points_are_rectangular_and_finite(points: &[Vec<f64>]) -> bool {
    let Some(dim) = points.first().map(Vec::len) else {
        return true;
    };
    dim > 0
        && points
            .iter()
            .all(|p| p.len() == dim && p.iter().all(|x| x.is_finite()))
}

fn point_sets_compatible(a: &[Vec<f64>], b: &[Vec<f64>]) -> bool {
    points_are_rectangular_and_finite(a)
        && points_are_rectangular_and_finite(b)
        && match (a.first(), b.first()) {
            (Some(x), Some(y)) => x.len() == y.len(),
            _ => true,
        }
}

/// A Gaussian process with a constant mean and a covariance kernel.
#[derive(Clone, Copy, Debug)]
pub struct GaussianProcess<K: Kernel> {
    kernel: K,
    mean: f64,
    jitter: f64,
}

impl<K: Kernel> GaussianProcess<K> {
    /// A zero-mean GP with covariance `kernel`.
    pub fn new(kernel: K) -> Self {
        Self {
            kernel,
            mean: 0.0,
            jitter: DEFAULT_JITTER,
        }
    }

    /// Set the (constant) mean function.
    pub fn with_mean(mut self, mean: f64) -> Self {
        self.mean = mean;
        self
    }

    /// Set the diagonal jitter added for numerical positive-definiteness.
    pub fn with_jitter(mut self, jitter: f64) -> Self {
        self.jitter = jitter;
        self
    }

    /// The cross-covariance matrix `K[i][j] = kernel(xs[i], ys[j])`.
    fn gram(&self, xs: &[Vec<f64>], ys: &[Vec<f64>]) -> Matrix {
        xs.iter()
            .map(|x| ys.iter().map(|y| self.kernel.covariance(x, y)).collect())
            .collect()
    }

    /// Sample a realization of the process at `inputs`, drawn from the GP **prior**
    /// `N(mean, K)`. Returns one value per input point.
    pub fn sample_prior<R: Rng + ?Sized>(
        &self,
        inputs: &[Vec<f64>],
        rng: &mut R,
    ) -> Option<Vec<f64>> {
        if inputs.is_empty() {
            return Some(Vec::new());
        }
        if !self.mean.is_finite() || !self.jitter.is_finite() || self.jitter < 0.0 {
            return None;
        }
        if !points_are_rectangular_and_finite(inputs) {
            return None;
        }
        let k = self.gram(inputs, inputs);
        let l = cholesky(&k, self.jitter)?;
        let z: Vec<f64> = (0..inputs.len())
            .map(|_| standard_normal_polar(rng))
            .collect();
        // f = mean·1 + L z.
        Some(
            (0..inputs.len())
                .map(|i| self.mean + (0..=i).map(|j| l[i][j] * z[j]).sum::<f64>())
                .collect(),
        )
    }

    /// Condition on observations `(train_x, train_y)` with i.i.d. observation-noise
    /// variance `noise`, returning the **posterior predictive** at `test_x`. `None`
    /// if the (noisy) training covariance is not positive-definite.
    pub fn posterior(
        &self,
        train_x: &[Vec<f64>],
        train_y: &[f64],
        noise: f64,
        test_x: &[Vec<f64>],
    ) -> Option<GpPosterior> {
        if train_x.len() != train_y.len()
            || !self.mean.is_finite()
            || !self.jitter.is_finite()
            || self.jitter < 0.0
            || !noise.is_finite()
            || noise < 0.0
            || train_y.iter().any(|y| !y.is_finite())
            || !point_sets_compatible(train_x, test_x)
        {
            return None;
        }
        let n = train_x.len();
        // Ky = K_XX + noise·I.
        let mut ky = self.gram(train_x, train_x);
        for (i, row) in ky.iter_mut().enumerate() {
            row[i] += noise;
        }
        let l = cholesky(&ky, self.jitter)?;

        // α = Ky⁻¹ (y − mean).
        let y_centered: Vec<f64> = train_y.iter().map(|&y| y - self.mean).collect();
        let alpha = cholesky_solve(&l, &y_centered);

        // Cross covariance columns ks[j] = cov(train, test_j), length n.
        let ks: Vec<Vec<f64>> = test_x
            .iter()
            .map(|t| {
                train_x
                    .iter()
                    .map(|x| self.kernel.covariance(x, t))
                    .collect()
            })
            .collect();
        let kss = self.gram(test_x, test_x);

        // Mean: m + ksⱼᵀ α.   Covariance: Kss − VᵀV with V = L \ ks.
        let m = test_x.len();
        let v: Vec<Vec<f64>> = ks.iter().map(|col| solve_lower(&l, col)).collect();
        let mean: Vec<f64> = (0..m)
            .map(|j| self.mean + (0..n).map(|k| ks[j][k] * alpha[k]).sum::<f64>())
            .collect();
        let covariance: Matrix = (0..m)
            .map(|i| {
                (0..m)
                    .map(|j| kss[i][j] - (0..n).map(|k| v[i][k] * v[j][k]).sum::<f64>())
                    .collect()
            })
            .collect();

        Some(GpPosterior {
            mean,
            covariance,
            jitter: self.jitter,
        })
    }
}

/// The posterior predictive of a [`GaussianProcess`] at a set of test inputs: a
/// multivariate normal `N(mean, covariance)`.
#[derive(Clone, Debug)]
pub struct GpPosterior {
    pub mean: Vec<f64>,
    pub covariance: Matrix,
    jitter: f64,
}

impl GpPosterior {
    /// The marginal predictive variance at each test point (the covariance
    /// diagonal).
    pub fn variance(&self) -> Vec<f64> {
        (0..self.mean.len())
            .map(|i| self.covariance[i][i])
            .collect()
    }

    /// The marginal predictive standard deviation at each test point.
    pub fn std_dev(&self) -> Vec<f64> {
        self.variance().iter().map(|v| v.max(0.0).sqrt()).collect()
    }

    /// Draw a joint sample from the posterior `N(mean, covariance)`.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<Vec<f64>> {
        if self.mean.is_empty() {
            return Some(Vec::new());
        }
        let l = cholesky(&self.covariance, self.jitter)?;
        let z: Vec<f64> = (0..self.mean.len())
            .map(|_| standard_normal_polar(rng))
            .collect();
        Some(
            (0..self.mean.len())
                .map(|i| self.mean[i] + (0..=i).map(|j| l[i][j] * z[j]).sum::<f64>())
                .collect(),
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::process::kernel::Rbf;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn points(xs: &[f64]) -> Vec<Vec<f64>> {
        xs.iter().map(|&x| vec![x]).collect()
    }
    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn prior_sample_has_the_right_length() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let gp = GaussianProcess::new(Rbf::new(1.0, 1.0));
        let f = gp
            .sample_prior(&points(&[0.0, 1.0, 2.0, 3.0]), &mut rng)
            .unwrap();
        assert_eq!(f.len(), 4);
        assert!(f.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn posterior_interpolates_noiseless_observations() {
        // With ~zero noise, the GP posterior mean passes through the training data.
        let gp = GaussianProcess::new(Rbf::new(1.0, 1.0));
        let tx = points(&[-2.0, 0.0, 2.0]);
        let ty = [1.0, 0.5, -1.0];
        let post = gp.posterior(&tx, &ty, 1e-10, &tx).unwrap();
        for i in 0..3 {
            close(post.mean[i], ty[i], 1e-4);
        }
        // Predictive variance at training points ≈ 0.
        for v in post.variance() {
            assert!(v.abs() < 1e-3, "variance at observed point: {v}");
        }
    }

    #[test]
    fn posterior_reverts_to_prior_far_from_data() {
        // Far from any training point, an RBF GP predicts the prior mean (0) with
        // ≈ prior variance (1).
        let gp = GaussianProcess::new(Rbf::new(1.0, 0.5));
        let post = gp
            .posterior(&points(&[0.0]), &[3.0], 1e-6, &points(&[100.0]))
            .unwrap();
        close(post.mean[0], 0.0, 1e-6);
        close(post.variance()[0], 1.0, 1e-6);
    }

    #[test]
    fn malformed_inputs_return_none() {
        let gp = GaussianProcess::new(Rbf::new(1.0, 0.5));
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        assert!(
            gp.sample_prior(&[vec![0.0], vec![1.0, 2.0]], &mut rng)
                .is_none()
        );
        assert!(
            gp.posterior(&points(&[0.0, 1.0]), &[1.0], 1e-6, &points(&[2.0]))
                .is_none()
        );
        assert!(
            gp.posterior(&points(&[0.0]), &[1.0], -1.0, &points(&[2.0]))
                .is_none()
        );
    }
}
