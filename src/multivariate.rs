//! Backend-neutral contracts for vector-valued distributions.

use euclid::array::{LazyMatrix, LazyVector};
use euclid::numeric::LazyReal;

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

/// Lazy moments for backend-generic vector-valued distributions.
pub trait LazyMultivariateMoments {
    type Scalar: LazyReal;
    type Vector: LazyVector<Scalar = Self::Scalar>;
    type Covariance: LazyMatrix<Scalar = Self::Scalar, Vector = Self::Vector>;

    fn mean(&self) -> Self::Vector;
    fn covariance_lazy(&self) -> Self::Covariance;
}

/// Lazy log-density contract over a backend vector type.
pub trait LazyMultivariateContinuous {
    type Scalar: LazyReal;
    type Vector: LazyVector<Scalar = Self::Scalar>;

    fn dim(&self) -> usize;
    fn log_pdf(&self, x: Self::Vector) -> Self::Scalar;
}
