//! Lazy-only concrete distribution families.
//!
//! This crate owns the parameter structs and lazy density/mass implementations
//! for univariate probability distributions. It also provides closure-backed
//! functional adapters. Eager host sampling, combinators, conjugate models,
//! fitting, and richer statistical utilities live in `bayes-distribution-ext`.
#![cfg_attr(test, allow(clippy::expect_used, clippy::panic, clippy::unwrap_used))]

mod cdf;
pub mod combinator;
pub mod conjugate;
pub mod constraint;
pub mod continuous;
pub mod discrete;
pub mod functional;
mod sample;

pub use bayes_core::{
    BayesError, Boundary, Constrained, Constraint, Continuous, ContinuousCdf,
    ContinuousDensityKernel, Discrete, DiscreteCdf, DiscreteMassKernel, IntegerConstraint,
    IntegerDiscrete, IntegerDiscreteCdf, IntegerMassKernel, Moments, NaturalConstraint,
    RealConstraint, Support,
};
pub use combinator::OrderStatistic;
pub use conjugate::{Conjugate, Counts, NaturalParams, NigNatural, Pair};
pub use continuous::{
    Beta, Cauchy, ChiSquared, Exponential, FisherSnedecor, Frechet, Gamma, Gumbel, HalfNormal,
    InverseGamma, Laplace, LogNormal, Logistic, Lomax, Normal, Pareto, Rayleigh, StudentT, Uniform,
    VonMises, Wald, Weibull,
};
pub use discrete::{
    Bernoulli, BetaBinomial, Binomial, Categorical, DiscreteUniform, Geometric, Hypergeometric,
    NegativeBinomial, Poisson,
};
pub use functional::{
    ContinuousCdfExt, ContinuousDensity, ContinuousDistributionExt, ContinuousFn,
    DiscreteDistributionExt, DiscreteFn, DiscreteMass,
};
pub use sample::{SampleTensorWithKey, SampleWithKey};
