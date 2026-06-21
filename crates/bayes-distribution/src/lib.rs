//! Lazy-only concrete distribution families.
//!
//! This crate owns the parameter structs and lazy density/mass implementations
//! for univariate probability distributions. Eager validation, CDFs, sampling,
//! combinators, conjugate models, fitting, and probability-vector utilities live
//! in `bayes-distribution-ext`.
#![cfg_attr(test, allow(clippy::expect_used, clippy::panic, clippy::unwrap_used))]

pub mod continuous;
pub mod discrete;
mod kernel_impls;

pub use bayes_core::{
    Continuous, ContinuousCdf, ContinuousDensityKernel, Discrete, DiscreteCdf, DiscreteMassKernel,
    IntegerDiscrete, IntegerDiscreteCdf, IntegerMassKernel, Moments, Support,
};
pub use continuous::{
    Beta, Cauchy, ChiSquared, Exponential, FisherSnedecor, Frechet, Gamma, Gumbel, HalfNormal,
    InverseGamma, Laplace, LogNormal, Logistic, Lomax, Normal, Pareto, Rayleigh, StudentT, Uniform,
    VonMises, Wald, Weibull,
};
pub use discrete::{
    Bernoulli, BetaBinomial, Binomial, DiscreteUniform, Geometric, Hypergeometric,
    NegativeBinomial, Poisson,
};
