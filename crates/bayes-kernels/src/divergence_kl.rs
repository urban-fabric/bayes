//! Kullback-Leibler divergence kernel view.
//!
//! Implementations live in [`crate::families`].

pub use crate::families::{
    bernoulli::kl_bernoulli_on_unit, beta::kl_beta, binomial::kl_binomial, cauchy::kl_cauchy,
    discrete_uniform::kl_discrete_uniform_on_support, exponential::kl_exponential, gamma::kl_gamma,
    geometric::kl_geometric, gumbel::kl_gumbel, laplace::kl_laplace,
    negative_binomial::kl_negative_binomial, normal::kl_normal, pareto::kl_pareto_on_support,
    poisson::kl_poisson, rayleigh::kl_rayleigh, uniform::kl_uniform_on_support,
    von_mises::kl_von_mises, wald::kl_wald, weibull::kl_weibull,
};
