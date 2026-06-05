//! **Discrete** probability distributions — those with a probability *mass*
//! ([`Discrete`](crate::distribution::Discrete)) on a countable support.

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
