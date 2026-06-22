//! Conjugate posterior-predictive and marginal-likelihood kernel view.
//!
//! Implementations live in [`crate::families`].

pub use crate::families::{
    beta::{
        beta_bernoulli_log_marginal_likelihood, beta_bernoulli_log_predictive,
        beta_binomial_log_predictive,
    },
    gamma::{
        gamma_exponential_log_marginal_likelihood, gamma_exponential_log_predictive,
        gamma_poisson_log_marginal_likelihood, gamma_poisson_log_predictive,
    },
    normal::{
        normal_inverse_gamma_log_marginal_likelihood, normal_known_variance_log_marginal_likelihood,
    },
};
