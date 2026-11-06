//! # bayes-distribution-ext
//!
//! The concrete **distribution families** built on the [`bayes-core-ext`](bayes_core_ext)
//! trait spine: continuous and discrete named families, distribution
//! [`combinator`]s (truncation, mixtures, order statistics), bijective
//! [`transformed`] distributions, [`exponential_family`] fitting via sufficient
//! statistics, and the [`conjugate`] Bayesian updates.
//!
//! The spine traits ([`Continuous`], [`Discrete`], [`ContinuousCdf`],
//! [`DiscreteCdf`], [`Support`], [`Moments`], [`Estimate`], …) are re-exported
//! from `bayes-core-ext` so generic code can `use bayes_distribution_ext::*` and get
//! both the contract and the families. Divergences between families live in the
//! sibling crate `bayes-divergence`; sampling stays on [`rand::distr::Distribution`].
#![cfg_attr(test, allow(clippy::expect_used, clippy::panic, clippy::unwrap_used))]

pub mod characteristic_function;
pub mod combinator;
pub mod conjugate;
pub mod continuous;
pub mod discrete;
pub mod estimation;
pub mod exponential_family;
pub mod functional;
pub mod generalized_entropy;
#[cfg(test)]
mod kernel_impls;
pub mod l_moments;
pub mod moment_generating;
pub mod probability_generating;
pub mod transformed;

// Re-export the whole trait spine + foundation modules so in-crate `crate::<Spine>`
// paths resolve and downstream `use bayes_distribution_ext::*` gets the full contract.
pub use bayes_core_ext::*;
pub use bayes_core_ext::{assert_close, assert_close_scaled};

pub use characteristic_function::CharacteristicFunction;
pub use combinator::{Mixture, OrderStatistic, Truncated};
pub use conjugate::{
    BetaBernoulli, BetaBinomialModel, Conjugate, GammaExponential, GammaPoisson, NaturalParams,
    NormalInverseGamma, NormalInverseGammaModel, NormalKnownVariance,
};
pub use continuous::{
    Beta, Cauchy, ChiSquared, Exponential, FisherSnedecor, Frechet, Gamma, Gumbel, HalfNormal,
    InverseGamma, Laplace, LogNormal, Logistic, Lomax, NoncentralChiSquared, NoncentralF,
    NoncentralT, Normal, Pareto, Rayleigh, StudentT, Triangular, Uniform, VonMises, Wald, Weibull,
};
pub use discrete::{
    Bernoulli, BetaBinomial, Binomial, Categorical, DiscreteUniform, Geometric, Hypergeometric,
    NegativeBinomial, Poisson,
};
pub use estimation::FisherInformation;
pub use exponential_family::{CountSum, ExponentialFamily, MeanVar, SufficientStatistic, Trials};
pub use functional::{
    ContinuousCdfExt, ContinuousDensity, ContinuousDistributionExt, ContinuousFn,
    DiscreteDistributionExt, DiscreteFn, DiscreteMass,
};
pub use generalized_entropy::GeneralizedEntropy;
pub use l_moments::LMoments;
pub use moment_generating::MomentGenerating;
pub use probability_generating::ProbabilityGenerating;
pub use transformed::{
    Affine, Bijector, Compose, Exp, Identity, Inverse, Sigmoid, Softplus, Transformed,
};

pub(crate) use continuous::normal;
