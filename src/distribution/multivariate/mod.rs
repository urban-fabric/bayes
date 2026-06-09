//! **Multivariate** distributions — over vectors rather than scalars. They share
//! a small spine analogous to the scalar one: a density/mass over `&[f64]` and
//! vector/matrix moments. Sampling is via `rand::distr::Distribution<Vec<_>>`.
//!
//! These build on `ramanujan`'s dense linear algebra (the Cholesky factorization
//! and its solves) for the density and sampling paths, and report their covariance
//! as a `ramanujan::DenseMatrix`.

mod dirichlet;
mod multinomial;
mod normal;

pub use dirichlet::Dirichlet;
pub use multinomial::Multinomial;
pub use normal::MultivariateNormal;
use ramanujan::DenseMatrix;

/// A continuous distribution over vectors: the joint density `f(x)` for `x ∈ ℝ^d`.
pub trait MultivariateContinuous {
    /// The dimension `d`.
    fn dim(&self) -> usize;
    /// The joint density at `x`.
    fn pdf(&self, x: &[f64]) -> f64 {
        self.ln_pdf(x).exp()
    }
    /// `ln f(x)` — override with the stable closed form.
    fn ln_pdf(&self, x: &[f64]) -> f64;
}

/// Vector/matrix moments of a multivariate distribution.
pub trait MultivariateMoments {
    /// The mean vector `E[X]`.
    fn mean_vector(&self) -> Vec<f64>;
    /// The covariance matrix `Cov[X]`.
    fn covariance(&self) -> DenseMatrix;
}
