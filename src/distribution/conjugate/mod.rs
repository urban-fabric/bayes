//! **Closed-form conjugate-prior updates.**
//!
//! For a handful of prior/likelihood pairs the Bayesian posterior is in the *same
//! family* as the prior, with parameters given by a direct formula — no MCMC, no
//! optimisation. Each model here wraps a prior distribution and exposes:
//!
//! * [`posterior`](Conjugate::posterior) — the updated same-family distribution
//!   after observing data; and
//! * `posterior_predictive` — the distribution of a *future* observation with the
//!   parameter integrated out (often a compound distribution such as the
//!   [`BetaBinomial`](crate::BetaBinomial) or [`NegativeBinomial`](crate::NegativeBinomial)).
//!
//! This is the one genuinely *Bayesian* capability the library exposes, and it sits
//! squarely on the closed-form side of the boundary: a posterior is just another
//! distribution definition. Iterative inference (MCMC / variational) is **not** here
//! — it belongs to a higher layer that consumes these as priors and likelihoods.
//!
//! The models are grouped by prior family (`beta`, `gamma`, `normal`, `dirichlet`).
//!
//! ```
//! use bayes::{Beta, BetaBernoulli, Conjugate, Moments};
//! let model = BetaBernoulli::new(Beta::new(1.0, 1.0).unwrap()); // uniform prior
//! let data = [true, true, false, true, true];                   // 4 of 5 successes
//! let post = model.posterior(&data);                            // Beta(5, 2)
//! assert!((post.mean().unwrap() - 5.0 / 7.0).abs() < 1e-12);
//! ```

mod beta;
mod dirichlet;
mod gamma;
mod normal;

pub use beta::{BetaBernoulli, BetaBinomialModel};
pub use dirichlet::{DirichletCategorical, DirichletMultinomial};
pub use gamma::{GammaExponential, GammaPoisson};
pub use normal::{NormalInverseGamma, NormalKnownVariance};

/// A conjugate prior/likelihood pair whose posterior is computable in closed form.
pub trait Conjugate {
    /// The observation type of the likelihood.
    type Datum;
    /// The posterior distribution type (the same family as the prior).
    type Posterior;

    /// The closed-form posterior after observing `data`.
    fn posterior(&self, data: &[Self::Datum]) -> Self::Posterior;
}
