//! Entropy kernel view.
//!
//! Implementations live in [`crate::families`].

pub use crate::families::{
    beta::beta_entropy, cauchy::cauchy_entropy, chi_squared::chi_squared_entropy,
    discrete_uniform::discrete_uniform_entropy, exponential::exponential_entropy,
    gamma::gamma_entropy, gumbel::gumbel_entropy, half_normal::half_normal_entropy,
    inverse_gamma::inverse_gamma_entropy, laplace::laplace_entropy, logistic::logistic_entropy,
    lognormal::lognormal_entropy, normal::normal_entropy, pareto::pareto_entropy,
    rayleigh::rayleigh_entropy, student_t::students_t_entropy, triangular::triangular_entropy,
    uniform::uniform_entropy, von_mises::von_mises_entropy, weibull::weibull_entropy,
};
