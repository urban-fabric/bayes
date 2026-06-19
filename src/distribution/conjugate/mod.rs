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
//! The scalar prior families (`beta`, `gamma`) live here; the matrix-coupled
//! conjugates (`normal`, `dirichlet`, `multivariate_normal`) — the ones whose
//! posteriors are multivariate-Normal / Wishart / Dirichlet, all carrying
//! `ramanujan` matrices — live in the sibling crate `ito`, reusing the
//! [`Conjugate`] trait and the [`natural`] hyperparameter monoid defined here.
//!
//! ```
//! use bayes::{Beta, BetaBernoulli, Conjugate, Moments};
//! let model = BetaBernoulli::new(Beta::new(1.0, 1.0).unwrap()); // uniform prior
//! let data = [true, true, false, true, true];                   // 4 of 5 successes
//! let post = model.posterior(&data);                            // Beta(5, 2)
//! assert!((post.mean().unwrap() - 5.0 / 7.0).abs() < 1e-12);
//! ```

mod beta;
mod gamma;
mod natural;

pub use beta::{BetaBernoulli, BetaBinomialModel};
pub use gamma::{GammaExponential, GammaPoisson};
pub use natural::{Counts, NaturalParams, NigNatural, Pair};

/// A conjugate prior/likelihood pair whose posterior is computable in closed
/// form by the **one** algebraic move every conjugate family shares: add the
/// data's sufficient statistic to the prior's natural hyperparameters,
///
/// ```text
/// posterior = η + Σ T(xᵢ)
/// ```
///
/// in a commutative additive monoid of hyperparameters ([`NaturalParams`]).
/// That umbrella law — and its two corollaries, **online updating equals batch**
/// and **order-independence** — is machine-checked once, generically over the
/// monoid, in the `riemann` Lean development
/// (`RiemannLean/Bayes/ExponentialFamily.lean`: `posterior`,
/// `posterior_online_eq_batch`, `posterior_perm`); the named families specialize
/// it in `Bayes/ConjugateFamilies.lean`.
///
/// So an implementor supplies only the family-specific pieces — the per-datum
/// statistic [`statistic`](Self::statistic), the prior's natural coordinates
/// [`prior_natural`](Self::prior_natural), the map back to a distribution
/// [`posterior_from_natural`](Self::posterior_from_natural), and the family's
/// [`posterior_predictive`](Self::posterior_predictive) closed form — and inherits
/// the proved-correct [`posterior`](Self::posterior) accumulation (and its streaming
/// accumulator [`posterior_natural`](Self::posterior_natural)) for free.
pub trait Conjugate {
    /// The observation type of the likelihood.
    type Datum;
    /// The posterior distribution type (the same family as the prior).
    type Posterior;
    /// The **posterior-predictive** type — the law of a *future* observation with
    /// the parameter integrated out against the posterior. Usually a compound
    /// distribution (Beta–Binomial, Negative-Binomial, Student-t, Lomax, …) in a
    /// *different* family from the prior, which is why it is its own associated
    /// type rather than [`Posterior`](Self::Posterior).
    type Predictive;
    /// The natural-hyperparameter monoid this family accumulates in.
    type Natural: NaturalParams;

    /// `T(x)` — the per-datum sufficient statistic as a natural-hyperparameter
    /// increment. Takes `&self` because some statistics carry a model constant
    /// (the binomial trial count, the known likelihood variance) — matching
    /// riemann's parameterized `binomStat m`.
    fn statistic(&self, x: &Self::Datum) -> Self::Natural;

    /// `η` — the prior expressed in natural hyperparameters (the fold seed).
    fn prior_natural(&self) -> Self::Natural;

    /// Recover the posterior distribution from accumulated natural
    /// hyperparameters `η + Σ T(xᵢ)`.
    fn posterior_from_natural(&self, natural: Self::Natural) -> Self::Posterior;

    /// `η + Σ T(xᵢ)` — the posterior's natural hyperparameters, the accumulator
    /// behind a streaming update. Folds each datum's statistic into the prior;
    /// by riemann's `posterior_online_eq_batch` / `posterior_perm` the result is
    /// independent of how `data` is split or ordered.
    fn posterior_natural(&self, data: &[Self::Datum]) -> Self::Natural {
        let mut acc = self.prior_natural();
        for x in data {
            acc = acc.combine(&self.statistic(x));
        }
        acc
    }

    /// The closed-form posterior after observing `data`.
    fn posterior(&self, data: &[Self::Datum]) -> Self::Posterior {
        self.posterior_from_natural(self.posterior_natural(data))
    }

    /// The closed-form **posterior-predictive** after observing `data`: the
    /// distribution of the next observation with the parameter marginalized out
    /// against [`posterior`](Self::posterior). Each conjugate family supplies its
    /// own compound form; the update half of conjugacy is generic
    /// ([`posterior`](Self::posterior)), the predictive half is family-specific.
    fn posterior_predictive(&self, data: &[Self::Datum]) -> Self::Predictive;
}

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
    use crate::distribution::Beta;

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
