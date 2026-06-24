//! Discrete distribution parameter types for the lazy layer.

mod bernoulli;
mod beta_binomial;
mod binomial;
mod categorical;
mod discrete_uniform;
mod geometric;
mod hypergeometric;
mod negative_binomial;
mod poisson;

pub use bernoulli::Bernoulli;
pub use beta_binomial::BetaBinomial;
pub use binomial::Binomial;
pub use categorical::Categorical;
pub use discrete_uniform::DiscreteUniform;
pub use geometric::Geometric;
pub use hypergeometric::Hypergeometric;
pub use negative_binomial::NegativeBinomial;
pub use poisson::Poisson;
