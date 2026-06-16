//! **Multivariate** distributions — over vectors rather than scalars. They share
//! a small spine analogous to the scalar one: a density/mass over `&[f64]` and
//! vector/matrix moments. Sampling is via `rand::distr::Distribution<Vec<_>>`.
//!
//! These build on `ramanujan`'s dense linear algebra (the Cholesky factorization
//! and its solves) for the density and sampling paths, and report their covariance
//! as a `ramanujan::DenseMatrix`.

mod dirichlet;
mod dirichlet_multinomial;
pub(crate) mod matrix;
mod multinomial;
mod normal;
mod orientation;
mod spherical_uniform;
mod student_t;
mod von_mises_fisher;
mod watson;
mod wishart;

pub use dirichlet::Dirichlet;
pub use dirichlet_multinomial::DirichletMultinomialDistribution;
pub use multinomial::Multinomial;
pub use normal::MultivariateNormal;
pub use orientation::{BinghamPotential, BinghamRankOne, BinghamSpectrum, KentPotential};
use ramanujan::DenseMatrix;
pub use spherical_uniform::SphericalUniform;
pub use student_t::MultivariateStudentT;
pub use von_mises_fisher::VonMisesFisher;
pub use watson::WatsonS2;
pub use wishart::{InverseWishart, Wishart};

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
    /// Differential / Shannon entropy, in nats — the multivariate analogue of
    /// [`Moments::entropy`](crate::distribution::Moments::entropy). `None` when
    /// not available in closed form.
    fn entropy(&self) -> Option<f64> {
        None
    }
}
