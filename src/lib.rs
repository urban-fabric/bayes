//! # bayes
//!
//! Efficient, general-purpose **probability & statistics for Rust**: probability
//! distributions on a decomposed trait spine (density/mass, CDF/quantile, support,
//! moments, sampling), backed by a special-functions module. A standalone numerics
//! library with no domain dependency. See `ARCHITECTURE.md` for the current module
//! boundaries and scope rules.
#![cfg_attr(test, allow(clippy::expect_used, clippy::panic, clippy::unwrap_used))]

pub mod distribution;
mod error;
pub mod function;
pub(crate) mod linalg;
pub mod process;
pub mod statistics;

// The distribution types and the trait spine, re-exported at the crate root.
pub use distribution::{
    Affine, Bernoulli, Beta, BetaBernoulli, BetaBinomial, BetaBinomialModel, Bijector, Binomial,
    Categorical, Cauchy, ChiSquared, Conjugate, Continuous, ContinuousCdf, CountSum, Dirichlet,
    DirichletCategorical, DirichletMultinomial, Discrete, DiscreteCdf, DiscreteUniform, Estimate,
    Exp, Exponential, ExponentialFamily, FisherSnedecor, Frechet, Gamma, GammaExponential,
    GammaPoisson, Geometric, Gumbel, HalfNormal, Hypergeometric, InverseGamma, KlDivergence,
    Laplace, LogNormal, Logistic, Lomax, MeanVar, Mixture, Moments, Multinomial,
    MultivariateContinuous, MultivariateMoments, MultivariateNormal, NegativeBinomial, Normal,
    NormalInverseGamma, NormalKnownVariance, Pareto, Poisson, Rayleigh, Sigmoid, StudentT,
    SufficientStatistic, Support, Transformed, Trials, Triangular, Truncated, Uniform, Uniform2D,
    VonMises, Wald, Weibull,
};
pub use error::{BayesError, StatsError};

/// Sample statistics over data (`data.mean()`, `data.median()`, …) plus rank
/// correlation, linear regression, resampling, KDE, streaming stats, and tests.
pub use statistics::{
    ConfidenceInterval, Ecdf, Histogram, KernelDensity, LinearRegression, RunningCovariance,
    RunningStats, SampleStats, TDigest, TestResult, fit_least_squares, histogram, mean_t_interval,
    mean_z_interval, proportion_wald, proportion_wilson,
};

/// Probabilistic processes: Gaussian processes, random walks, Poisson processes,
/// Markov chains.
pub use process::{
    GaussianProcess, GaussianRandomWalk, GpPosterior, Kernel, Linear, MarkovChain, Matern,
    MaternNu, Periodic, PoissonProcess, Rbf, inhomogeneous_arrivals, wiener_path,
};

/// The sampling contract every distribution implements (`rng.sample(&dist)`).
pub use rand::distr::Distribution;
