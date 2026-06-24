//! **Closed-form conjugate-prior updates.**
//!
//! For a handful of prior/likelihood pairs the Bayesian posterior is in the *same
//! family* as the prior, with parameters given by a direct formula — no MCMC, no
//! optimisation. Each model wraps a prior distribution and exposes a closed-form
//! [`posterior`](Conjugate::posterior) — the updated same-family distribution after
//! observing data — and a
//! [`posterior_predictive`](Conjugate::posterior_predictive) — the distribution of a
//! *future* observation with the parameter integrated out (often a compound
//! distribution such as the [`BetaBinomial`](crate::BetaBinomial),
//! [`NegativeBinomial`](crate::NegativeBinomial), or [`Lomax`](crate::Lomax)).
//! Family wrappers also expose exact log marginal likelihoods (`ln p(data)`)
//! where the likelihood parameter is analytically integrated out.
//!
//! This is the one genuinely *Bayesian* capability the library exposes, and it sits
//! squarely on the closed-form side of the boundary: a posterior is just another
//! distribution definition. Iterative inference (MCMC / variational) is **not** here
//! — it belongs to a higher layer that consumes these as priors and likelihoods.
//!
//! The scalar prior families live here: `beta`, `gamma`, and the univariate
//! Gaussian conjugates in `normal` (Normal–Normal with known variance, and
//! Normal–Inverse-Gamma for unknown mean *and* variance — both observing scalar
//! data). The genuinely matrix-coupled conjugates (`dirichlet`,
//! `multivariate_normal`) — the ones whose posteriors are multivariate-Normal /
//! Wishart / Dirichlet, all carrying `ramanujan` matrices — live in the sibling
//! crate `ito`, reusing the [`Conjugate`] trait and the [`natural`]
//! hyperparameter monoid defined here.
//!
//! ```
//! use bayes_distribution_ext::{Beta, BetaBernoulli, Conjugate, Moments};
//! let model = BetaBernoulli::new(Beta::new(1.0, 1.0).unwrap()); // uniform prior
//! let data = [true, true, false, true, true];                   // 4 of 5 successes
//! let post = model.posterior(&data);                            // Beta(5, 2)
//! assert!((post.mean() - 5.0 / 7.0).abs() < 1e-12);
//! ```

mod beta;
mod gamma;
mod normal;

pub use bayes_distribution::conjugate::{Conjugate, Counts, NaturalParams, NigNatural, Pair};
pub use beta::{BetaBernoulli, BetaBinomialModel};
pub use gamma::{GammaExponential, GammaPoisson};
pub use normal::{NormalInverseGamma, NormalInverseGammaModel, NormalKnownVariance};

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    //! The two corollaries every family inherits from the riemann umbrella law
    //! (`Bayes/ExponentialFamily`): **online = batch** (`posterior_online_eq_batch`)
    //! and **order-independence** (`posterior_perm`). Exercised here on the
    //! Beta–Bernoulli family, whose natural coordinates are trivial. The
    //! matrix-coupled families (Normal–Inverse-Gamma, Dirichlet–Categorical) carry
    //! their analogous corollary checks in the sibling crate `ito`.

    use super::*;
    use crate::Beta;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    /// Fold `data` into the prior one datum at a time, continuing from a
    /// running natural accumulator — the streaming update. By
    /// `posterior_online_eq_batch` this must equal the batch posterior.
    fn streamed<C: Conjugate>(model: &C, first: &[C::Datum], rest: &[C::Datum]) -> C::Natural {
        let mut acc = model.posterior_natural(first);
        for x in rest {
            acc = acc.combine(&model.statistic(x));
        }
        acc
    }

    #[test]
    fn beta_bernoulli_online_equals_batch_and_order_independent() {
        let model = BetaBernoulli::new(Beta::new(1.0, 1.0).unwrap());
        let d1 = [true, false, true];
        let d2 = [true, true, false, false];
        let all: Vec<bool> = d1.iter().chain(d2.iter()).copied().collect();

        let batch = model.posterior(&all);
        let online = model.posterior_from_natural(streamed(&model, &d1, &d2));
        close(batch.alpha(), online.alpha(), 1e-12);
        close(batch.beta(), online.beta(), 1e-12);

        // A permutation of the same multiset (4 successes, 3 failures).
        let shuffled = [false, false, true, true, true, true, false];
        let permuted = model.posterior(&shuffled);
        close(permuted.alpha(), batch.alpha(), 1e-12);
        close(permuted.beta(), batch.beta(), 1e-12);
    }
}
