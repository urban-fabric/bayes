//! Chi-squared divergence kernel view.
//!
//! Implementations live in [`crate::families`].

pub use crate::families::{
    bernoulli::chi_squared_bernoulli, binomial::chi_squared_binomial,
    discrete_uniform::chi_squared_discrete_uniform_on_support,
    exponential::chi_squared_exponential_on_support, gamma::chi_squared_gamma_on_support,
    negative_binomial::chi_squared_negative_binomial_on_support,
    normal::chi_squared_normal_on_support, poisson::chi_squared_poisson,
    uniform::chi_squared_uniform_on_support, von_mises::chi_squared_von_mises,
    wald::chi_squared_wald_on_support,
};
