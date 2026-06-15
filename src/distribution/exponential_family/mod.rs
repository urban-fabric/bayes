//! **Exponential-family fitting via sufficient statistics.**
//!
//! A [`SufficientStatistic`] is everything a sample tells you about an
//! exponential-family parameter, packed into a small fixed-size summary that you
//! can accumulate one observation at a time and **merge** across batches (it forms
//! a monoid — the same algebraic shape as [`RunningStats`](crate::RunningStats)).
//! An [`ExponentialFamily`] distribution then recovers its parameters from that
//! summary by a **closed-form** maximum-likelihood or method-of-moments formula.
//!
//! This generalises the per-distribution [`Estimate`](crate::Estimate) trait: where
//! `Estimate::from_sample` is a one-shot batch fit, sufficient statistics fit the
//! *same* closed forms while supporting single-pass streaming and parallel merge —
//! and they are the substrate the [`conjugate`](crate::distribution::conjugate)
//! updates build on. Everything here is a direct formula; nothing iterates.
//!
//! The summary types live in `statistic`; the per-distribution fits in `family`.

mod family;
mod statistic;

pub use statistic::{CountSum, MeanVar, Trials};

/// A sufficient statistic: built from observations, mergeable, and the input to a
/// closed-form fit. Implementors are small `Copy` summaries.
///
/// [`merge`](Self::merge) is a commutative-monoid `+` — the same algebraic shape
/// as [`NaturalParams::combine`](crate::distribution::NaturalParams::combine) in
/// the conjugate layer. Both accumulate independent-of-order, streaming-equals-
/// batch; the difference is only what comes out the end — here a closed-form MLE
/// ([`ExponentialFamily::from_sufficient`]), there a Bayesian posterior.
pub trait SufficientStatistic: Copy {
    /// The observation type the statistic is computed from.
    type Datum: Copy;

    /// The empty summary (identity for [`merge`](Self::merge)).
    fn empty() -> Self;

    /// The summary of a single observation.
    fn observe(x: Self::Datum) -> Self;

    /// Combine two summaries (associative; `empty()` is the identity).
    fn merge(&self, other: &Self) -> Self;

    /// Fold a whole sample into one summary.
    fn from_sample(data: &[Self::Datum]) -> Self {
        let mut acc = Self::empty();
        for &x in data {
            acc = acc.merge(&Self::observe(x));
        }
        acc
    }
}

/// An exponential-family distribution that can be recovered in closed form from
/// its [`SufficientStatistic`].
pub trait ExponentialFamily: Sized {
    /// The sufficient statistic for this family.
    type Stat: SufficientStatistic;

    /// Closed-form parameter estimate from accumulated statistics. `None` when the
    /// summary is too small or degenerate (e.g. zero variance) to fit.
    fn from_sufficient(stat: &Self::Stat) -> Option<Self>;

    /// Fit directly from a sample: fold into a sufficient statistic, then estimate.
    fn fit(data: &[<Self::Stat as SufficientStatistic>::Datum]) -> Option<Self> {
        Self::from_sufficient(&Self::Stat::from_sample(data))
    }
}
