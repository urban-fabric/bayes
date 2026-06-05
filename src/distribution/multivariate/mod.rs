//! **Multivariate** distributions — over vectors rather than scalars. They share
//! a small spine analogous to the scalar one: a density/mass over `&[f64]` and
//! vector/matrix moments. Sampling is via `rand::distr::Distribution<Vec<_>>`.
//!
//! These reuse the crate's hand-rolled Cholesky (`crate::linalg`), so the
//! multivariate core needs **no external linear-algebra dependency**.

mod dirichlet;
mod multinomial;
mod normal;

pub use dirichlet::Dirichlet;
pub use multinomial::Multinomial;
pub use normal::MultivariateNormal;

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
    /// The covariance matrix `Cov[X]` (row-major).
    fn covariance(&self) -> Vec<Vec<f64>>;
}
