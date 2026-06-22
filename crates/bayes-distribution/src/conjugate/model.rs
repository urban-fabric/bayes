use super::NaturalParams;

/// A conjugate prior/likelihood pair whose posterior is computable in closed
/// form by adding each datum's sufficient statistic to the prior's natural
/// hyperparameters.
///
/// Implementors supply the family-specific statistic, prior coordinates, map
/// back to a posterior distribution, and posterior-predictive form. The default
/// [`posterior`](Self::posterior) and [`posterior_natural`](Self::posterior_natural)
/// implementations provide the shared batch/streaming accumulation.
pub trait Conjugate {
    /// The observation type of the likelihood.
    type Datum;
    /// The posterior distribution type, usually the same family as the prior.
    type Posterior;
    /// The posterior-predictive distribution type.
    type Predictive;
    /// The natural-hyperparameter monoid this family accumulates in.
    type Natural: NaturalParams;

    /// Per-datum sufficient statistic as a natural-hyperparameter increment.
    fn statistic(&self, x: &Self::Datum) -> Self::Natural;

    /// The prior expressed in natural hyperparameters.
    fn prior_natural(&self) -> Self::Natural;

    /// Recover the posterior distribution from accumulated natural parameters.
    fn posterior_from_natural(&self, natural: Self::Natural) -> Self::Posterior;

    /// Accumulate prior natural parameters plus every datum's statistic.
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

    /// The closed-form posterior-predictive after observing `data`.
    fn posterior_predictive(&self, data: &[Self::Datum]) -> Self::Predictive;
}
