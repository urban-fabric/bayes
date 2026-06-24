//! # bayes
//!
//! Efficient, general-purpose **probability & statistics for Rust**: probability
//! distributions on a decomposed trait spine (density/mass, CDF/quantile, support,
//! moments, sampling), sample statistics, stochastic processes, and closed-form
//! Bayesian updates. A standalone numerics library with no domain
//! dependency.
//!
//! This crate is a thin **facade** over the `bayes` workspace: each module below
//! re-exports a focused member crate, preserving the original `bayes::…` public
//! API. The implementations live in the members — `bayes-core` /
//! `bayes-core-ext` (the trait spine), `bayes-distribution-ext` (full eager
//! families), `bayes-divergence`, `bayes-kernels`, `bayes-statistics`, and
//! `bayes-process`.
#![cfg_attr(test, allow(clippy::expect_used, clippy::panic, clippy::unwrap_used))]

/// The full eager distribution surface and closed-form divergences.
pub mod distribution {
    pub use bayes_distribution_ext::*;
    pub use bayes_divergence::*;
}

/// Euclid-generic closed-form probability density kernels.
pub mod kernels {
    pub use bayes_kernels::*;
}

/// Euclid-generic lazy distribution structs, divergence traits, and kernels.
pub mod lazy {
    pub use crate::kernels::*;
    pub use bayes_core_ext::{
        brier_score_real, log_loss_real, log_score_real, normalize_probability_rows_real,
        normalize_probability_weights_real, probability_sum_real, ranked_probability_score_real,
        spherical_score_real,
    };
    pub use bayes_distribution::*;
    pub use bayes_divergence::*;
}

/// Normal / complex-normal log-probability primitives.
pub mod primitives {
    pub use bayes_kernels::primitives::*;
}

/// Sample statistics over data.
pub mod statistics {
    pub use bayes_statistics::*;
}

/// Probabilistic processes: random walks, Poisson processes, and GP covariance kernels.
pub mod process {
    pub use bayes_process::*;
}

/// The Gaussian-process kernel contract and marginal-likelihood objective.
pub mod gp {
    pub use bayes_process::gp::*;
}

pub use primitives::{
    complex_normal_log_prob_with_normalizer, complex_residual_norm_sqr, normal_log_kernel,
    normal_log_prob, normal_log_prob_with_constants, normal_log_prob_with_normalizer,
};

pub use bayes_core_ext::{BayesError, StatsError};
pub use distribution::{
    Affine, Alpha, Bernoulli, Beta, BetaBernoulli, BetaBinomial, BetaBinomialModel, Bijector,
    Binomial, Boundary, Categorical, Cauchy, Channel, CharacteristicFunction, ChiSquared,
    ChiSquaredDivergence, ChiSquaredGen, Conjugate, Constrained, Constraint, Continuous,
    ContinuousCdf, ContinuousCdfExt, ContinuousDensity, ContinuousDensityKernel,
    ContinuousDistributionExt, ContinuousFn, CountSum, Custom, Discrete, DiscreteCdf,
    DiscreteDistributionExt, DiscreteFn, DiscreteMass, DiscreteMassKernel, DiscreteUniform,
    Divergence, Estimate, Exp, Exponential, ExponentialFamily, FGenerator, FisherInformation,
    FisherSnedecor, Frechet, Gamma, GammaExponential, GammaPoisson, GeneralizedEntropy, Geometric,
    Gumbel, HalfNormal, HellingerDistance, Hypergeometric, IntegerConstraint, IntegerDiscrete,
    IntegerDiscreteCdf, IntegerMassKernel, InverseGamma, Jeffreys, JensenShannon, Kl, KlDivergence,
    LMoments, Laplace, LogNormal, Logistic, Lomax, MeanVar, Mixture, MomentGenerating, Moments,
    NaturalConstraint, NaturalParams, NegativeBinomial, Normal, NormalInverseGamma,
    NormalInverseGammaModel, NormalKnownVariance, OrderStatistic, PROBABILITY_TOLERANCE, Pareto,
    Poisson, ProbabilityGenerating, Rayleigh, RealConstraint, RenyiDivergence, ReverseKl, Sigmoid,
    SquaredHellinger, StudentT, SufficientStatistic, Support, TotalVariation, TotalVariationGen,
    Transformed, Trials, Triangular, Truncated, Uniform, VonMises, Wald, Weibull,
    bhattacharyya_coefficient, bhattacharyya_coefficient_real, brier_score, brier_score_real,
    conditional_entropy, conditional_entropy_real, cross_entropy, cross_entropy_real, f_divergence,
    f_divergence_real, is_entropy_admissible, is_probability_vector,
    is_probability_vector_with_tolerance, is_row_stochastic, is_row_stochastic_with_tolerance,
    jensen_shannon, jensen_shannon_real, joint_entropy, joint_entropy_real, log_loss,
    log_loss_real, log_score, log_score_real, max_entropy, max_entropy_real, mutual_information,
    mutual_information_real, normalize_probability_rows, normalize_probability_rows_real,
    normalize_probability_weights, normalize_probability_weights_real, normalized_entropy,
    normalized_entropy_real, normalized_mutual_information, normalized_mutual_information_real,
    probability_sum, probability_sum_real, ranked_probability_score, ranked_probability_score_real,
    shannon_entropy, shannon_entropy_real, spherical_score, spherical_score_real,
    validate_probability_vector, validate_row_stochastic, validate_row_stochastic_with_tolerance,
    variation_of_information, variation_of_information_real,
};

/// Sample statistics over data (`data.mean()`, `data.median()`, …), including
/// robust/weighted summaries, effect sizes, OLS diagnostics, confidence intervals,
/// resampling, KDE, streaming stats, and test result types.
pub use statistics::{
    ConfidenceInterval, Ecdf, Histogram, KernelDensity, LinearRegression, RunningCovariance,
    RunningStats, SampleStats, TDigest, TestResult, VarianceDecomposition,
    coefficient_of_variation, coefficient_of_variation_real, cohen_d, cohen_d_real,
    correlation_real, covariance_real, dot, dot_real, effective_sample_size,
    effective_sample_size_real, eta_squared, eta_squared_real, geometric_mean_real,
    gini_mean_difference, gini_mean_difference_real, glass_delta, glass_delta_real,
    harmonic_mean_real, hedges_g, hedges_g_real, histogram, is_effective_sample_size_admissible,
    kurtosis_real, mcdiarmid_bound, mean_abs_deviation, mean_abs_deviation_real,
    mean_difference_welch_interval, mean_real, mean_t_interval, mean_vector, mean_vector_real,
    mean_z_interval, norm2, norm2_real, normalized_log_weights, normalized_weights,
    normalized_weights_real, odds_ratio, odds_ratio_real, omega_squared, omega_squared_real,
    poisson_rate_interval, pooled_std_dev, pooled_std_dev_real, population_variance_real,
    proportion_agresti_coull, proportion_clopper_pearson, proportion_difference_wald,
    proportion_jeffreys, proportion_wald, proportion_wilson, risk_difference, risk_difference_real,
    risk_ratio, risk_ratio_real, sample_shape, skewness_real, std_dev_chi_square_interval,
    std_dev_real, trimmed_mean, tukey_fences, tukey_outlier_flags, variance_chi_square_interval,
    variance_decomposition, variance_real, weight_sum, weight_sum_real, weighted_correlation,
    weighted_correlation_real, weighted_covariance, weighted_covariance_real, weighted_mean,
    weighted_mean_real, weighted_population_variance, weighted_population_variance_real,
    weighted_quantile, weighted_std_dev, weighted_std_dev_real, weighted_variance,
    weighted_variance_real, winsorized_mean, z_scores, z_scores_real,
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

/// Numeric method traits from `euclid`; bring these into scope for methods such as
/// `x.ln_gamma()`, `x.ln_beta(y)`, `n.ln_factorial()`, and `n.ln_binomial(&k)`.
pub use euclid::{Natural, Real};
