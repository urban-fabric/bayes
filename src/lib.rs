//! # bayes
//!
//! Efficient, general-purpose **probability & statistics for Rust**: probability
//! distributions on a decomposed trait spine (density/mass, CDF/quantile, support,
//! moments, sampling), special functions, sample statistics, stochastic processes,
//! and closed-form Bayesian updates. A standalone numerics library with no domain
//! dependency. See `ARCHITECTURE.md` for the current module boundaries and scope
//! rules.
#![cfg_attr(test, allow(clippy::expect_used, clippy::panic, clippy::unwrap_used))]

pub mod distribution;
mod error;
pub mod function;
pub mod process;
pub mod statistics;

pub use distribution::{
    Affine, Bernoulli, Beta, BetaBernoulli, BetaBinomial, BetaBinomialModel, Bijector, Binomial,
    Categorical, Cauchy, ChiSquared, ChiSquaredDivergence, Conjugate, Constrained, Constraint,
    Continuous, ContinuousCdf, ContinuousCdfExt, ContinuousDensity, ContinuousDistributionExt,
    ContinuousFn, CountSum, Dirichlet, DirichletCategorical, DirichletMultinomial, Discrete,
    DiscreteCdf, DiscreteDistributionExt, DiscreteFn, DiscreteMass, DiscreteUniform, Divergence,
    Estimate, Exp, Exponential, ExponentialFamily, FisherSnedecor, Frechet, Gamma,
    GammaExponential, GammaPoisson, Geometric, Gumbel, HalfNormal, HellingerDistance,
    Hypergeometric, InverseGamma, JensenShannon, KlDivergence, Laplace, LogNormal, Logistic, Lomax,
    MeanVar, Mixture, Moments, Multinomial, MultivariateContinuous, MultivariateMoments,
    MultivariateNormal, NaturalParams, NegativeBinomial, Normal, NormalInverseGamma,
    NormalKnownVariance, Pareto, Poisson, Rayleigh, RenyiDivergence, Sigmoid, StudentT,
    SufficientStatistic, Support, TotalVariation, Transformed, Trials, Triangular, Truncated,
    Uniform, Uniform2D, VonMises, VonMisesFisher, Wald, Weibull, mutual_information,
};
pub use error::{BayesError, StatsError};

/// Sample statistics over data (`data.mean()`, `data.median()`, …), including
/// robust/weighted summaries, effect sizes, OLS diagnostics, confidence intervals,
/// resampling, KDE, streaming stats, and test result types.
pub use statistics::{
    ConfidenceInterval, Ecdf, Histogram, KernelDensity, LinearRegression, RunningCovariance,
    RunningStats, SampleStats, TDigest, TestResult, coefficient_of_variation, cohen_d,
    correlation_matrix, covariance_matrix, effective_sample_size, eta_squared, fit_least_squares,
    gini_mean_difference, glass_delta, hedges_g, histogram, hotelling_t_squared,
    mahalanobis_squared, mean_abs_deviation, mean_difference_welch_interval, mean_t_interval,
    mean_vector, mean_z_interval, odds_ratio, omega_squared, poisson_rate_interval,
    pooled_covariance_matrix, pooled_std_dev, proportion_agresti_coull, proportion_clopper_pearson,
    proportion_difference_wald, proportion_jeffreys, proportion_wald, proportion_wilson,
    risk_difference, risk_ratio, std_dev_chi_square_interval, trimmed_mean, tukey_fences,
    tukey_outlier_flags, variance_chi_square_interval, weight_sum, weighted_correlation,
    weighted_covariance, weighted_mean, weighted_population_variance, weighted_quantile,
    weighted_std_dev, weighted_variance, winsorized_mean, z_scores,
};

/// Probabilistic processes: Gaussian processes, random walks, Poisson processes,
/// Markov chains.
pub use process::{
    GaussianProcess, GaussianRandomWalk, GpPosterior, Kernel, Linear, MarkovChain, Matern,
    MaternNu, Periodic, PoissonProcess, Rbf, inhomogeneous_arrivals, wiener_path,
};

/// The sampling contract every distribution implements (`rng.sample(&dist)`).
pub use rand::distr::Distribution;
