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
pub mod gp;
pub mod lazy;
pub mod multivariate;
pub mod primitives;
pub mod process;
pub mod statistics;

pub use primitives::{
    complex_normal_log_prob_with_normalizer, complex_residual_norm_sqr, normal_log_kernel,
    normal_log_prob, normal_log_prob_with_constants, normal_log_prob_with_normalizer,
};

pub use distribution::{
    Affine, Alpha, Bernoulli, Beta, BetaBernoulli, BetaBinomial, BetaBinomialModel, Bijector,
    Binomial, Categorical, Cauchy, Channel, ChiSquared, ChiSquaredDivergence, ChiSquaredGen,
    Conjugate, Constrained, Constraint, Continuous, ContinuousCdf, ContinuousCdfExt,
    ContinuousDensity, ContinuousDistributionExt, ContinuousFn, CountSum, Custom, Discrete,
    DiscreteCdf, DiscreteDistributionExt, DiscreteFn, DiscreteMass, DiscreteUniform, Divergence,
    Estimate, Exp, Exponential, ExponentialFamily, FGenerator, FisherSnedecor, Frechet, Gamma,
    GammaExponential, GammaPoisson, Geometric, Gumbel, HalfNormal, HellingerDistance,
    Hypergeometric, InverseGamma, Jeffreys, JensenShannon, Kl, KlDivergence, Laplace, LogNormal,
    Logistic, Lomax, MeanVar, Mixture, Moments, NaturalParams, NegativeBinomial, Normal,
    OrderStatistic, PROBABILITY_TOLERANCE, Pareto, Poisson, Rayleigh, RenyiDivergence, ReverseKl,
    Sigmoid, SquaredHellinger, StudentT, SufficientStatistic, Support, TotalVariation,
    TotalVariationGen, Transformed, Trials, Triangular, Truncated, Uniform, Uniform2D, VonMises,
    Wald, Weibull, bhattacharyya_coefficient, brier_score, conditional_entropy, cross_entropy,
    f_divergence, is_entropy_admissible, is_probability_vector,
    is_probability_vector_with_tolerance, is_row_stochastic, is_row_stochastic_with_tolerance,
    jensen_shannon, joint_entropy, log_loss, log_score, max_entropy, mutual_information,
    normalize_probability_rows, normalize_probability_weights, normalized_entropy,
    normalized_mutual_information, probability_sum, ranked_probability_score, shannon_entropy,
    spherical_score, validate_probability_vector, validate_row_stochastic,
    validate_row_stochastic_with_tolerance, variation_of_information,
};
pub use error::{BayesError, StatsError};
pub use multivariate::{MultivariateContinuous, MultivariateMoments};

/// Sample statistics over data (`data.mean()`, `data.median()`, …), including
/// robust/weighted summaries, effect sizes, OLS diagnostics, confidence intervals,
/// resampling, KDE, streaming stats, and test result types.
pub use statistics::{
    ConfidenceInterval, Ecdf, Histogram, KernelDensity, LinearRegression, RunningCovariance,
    RunningStats, SampleStats, TDigest, TestResult, VarianceDecomposition,
    coefficient_of_variation, cohen_d, dot, effective_sample_size, eta_squared,
    gini_mean_difference, glass_delta, hedges_g, histogram, is_effective_sample_size_admissible,
    mcdiarmid_bound, mean_abs_deviation, mean_difference_welch_interval, mean_t_interval,
    mean_vector, mean_z_interval, norm2, normalized_log_weights, normalized_weights, odds_ratio,
    omega_squared, poisson_rate_interval, pooled_std_dev, proportion_agresti_coull,
    proportion_clopper_pearson, proportion_difference_wald, proportion_jeffreys, proportion_wald,
    proportion_wilson, risk_difference, risk_ratio, sample_shape, std_dev_chi_square_interval,
    trimmed_mean, tukey_fences, tukey_outlier_flags, variance_chi_square_interval,
    variance_decomposition, weight_sum, weighted_correlation, weighted_covariance, weighted_mean,
    weighted_population_variance, weighted_quantile, weighted_std_dev, weighted_variance,
    winsorized_mean, z_scores,
};

/// Probabilistic processes: random walks, Poisson processes, and the covariance
/// kernels used by Gaussian processes. (`GaussianProcess` and `MarkovChain` are
/// matrix-coupled and live in the sibling crate `ito`.)
pub use process::{
    GaussianRandomWalk, Kernel, Linear, Matern, MaternNu, Periodic, PoissonProcess, Rbf,
    inhomogeneous_arrivals, wiener_path,
};

/// The sampling contract every distribution implements (`rng.sample(&dist)`).
pub use rand::distr::Distribution;
