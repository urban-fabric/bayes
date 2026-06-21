//! Numeric contracts for vector-valued distributions (`f64` host interface).
//!
//! The *lazy/backend-generic* multivariate contracts that referenced the container
//! tier (`LazyMatrix`/`LazyVector`) have moved to the backend: the concrete
//! tensor-coupled multivariate densities live in `maxwell::dist`.

/// A continuous distribution over vectors: the joint density `f(x)` for `x in R^d`.
pub trait MultivariateContinuous {
    /// The dimension `d`.
    fn dim(&self) -> usize;
    /// The joint density at `x`.
    fn pdf(&self, x: &[f64]) -> f64 {
        self.ln_pdf(x).exp()
    }
    /// `ln f(x)`; implementors should override with a stable closed form.
    fn ln_pdf(&self, x: &[f64]) -> f64;
}

/// Vector/matrix moments of a multivariate distribution.
pub trait MultivariateMoments {
    /// The covariance representation used by the concrete backend.
    type Covariance;

    /// The mean vector `E[X]`.
    fn mean_vector(&self) -> Vec<f64>;
    /// The covariance matrix `Cov[X]`.
    fn covariance(&self) -> Self::Covariance;
    /// Differential / Shannon entropy, in nats. `None` when not available in closed form.
    fn entropy(&self) -> Option<f64> {
        None
    }
}
