//! Renyi divergence and total-variation kernel view.
//!
//! Implementations live in [`crate::families`].

pub use crate::families::{
    bernoulli::{renyi_bernoulli, total_variation_bernoulli},
    binomial::renyi_binomial,
    discrete_uniform::renyi_discrete_uniform_on_support,
    gamma::renyi_gamma_on_support,
    negative_binomial::renyi_negative_binomial_on_support,
    normal::renyi_normal_on_support,
    uniform::renyi_uniform_on_support,
    von_mises::renyi_von_mises,
    wald::renyi_wald_on_support,
};
