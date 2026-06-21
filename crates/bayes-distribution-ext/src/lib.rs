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

pub mod combinator;
pub mod conjugate;
pub mod continuous;
pub mod discrete;
pub mod exponential_family;
pub mod functional;
#[cfg(test)]
mod kernel_impls;
pub mod transformed;

// Re-export the whole trait spine + foundation modules so in-crate `crate::<Spine>`
// paths resolve and downstream `use bayes_distribution_ext::*` gets the full contract.
pub use bayes_core_ext::*;

pub use combinator::{Mixture, OrderStatistic, Truncated};
pub use conjugate::{
    BetaBernoulli, BetaBinomialModel, Conjugate, GammaExponential, GammaPoisson, NaturalParams,
};
pub use continuous::{
    Beta, Cauchy, ChiSquared, Exponential, FisherSnedecor, Frechet, Gamma, Gumbel, HalfNormal,
    InverseGamma, Laplace, LogNormal, Logistic, Lomax, Normal, Pareto, Rayleigh, StudentT,
    Triangular, Uniform, VonMises, Wald, Weibull,
};
pub use discrete::{
    Bernoulli, BetaBinomial, Binomial, Categorical, DiscreteUniform, Geometric, Hypergeometric,
    NegativeBinomial, Poisson,
};
pub use exponential_family::{CountSum, ExponentialFamily, MeanVar, SufficientStatistic, Trials};
pub use functional::{
    ContinuousCdfExt, ContinuousDensity, ContinuousDistributionExt, ContinuousFn,
    DiscreteDistributionExt, DiscreteFn, DiscreteMass,
};
pub use transformed::{Affine, Bijector, Exp, Sigmoid, Transformed};

pub(crate) use continuous::normal;
