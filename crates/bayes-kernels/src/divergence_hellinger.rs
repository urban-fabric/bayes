//! Squared Hellinger distance kernel view.
//!
//! Implementations live in [`crate::families`].

pub use crate::families::{
    bernoulli::hellinger_sq_bernoulli, beta::hellinger_sq_beta, binomial::hellinger_sq_binomial,
    discrete_uniform::hellinger_sq_discrete_uniform, exponential::hellinger_sq_exponential,
    gamma::hellinger_sq_gamma, negative_binomial::hellinger_sq_negative_binomial,
    normal::hellinger_sq_normal, poisson::hellinger_sq_poisson, uniform::hellinger_sq_uniform,
    von_mises::hellinger_sq_von_mises, wald::hellinger_sq_wald,
};
