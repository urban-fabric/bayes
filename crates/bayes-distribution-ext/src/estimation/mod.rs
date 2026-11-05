//! **Fitting and the information it carries.**
//!
//! Two halves that answer opposite questions about the same sample:
//!
//! | module | question |
//! |--------|----------|
//! | [`sample`] | what are the parameters? — the summaries every closed-form fit is built from |
//! | [`fisher`] | how well could they possibly be known? — Fisher information and the Cramér–Rao floor |
//!
//! The estimators themselves live with their families, next to the constructors
//! whose preconditions they have to respect, as
//! [`Estimate`](bayes_core_ext::Estimate) implementations. What lives here is
//! only what they share.
//!
//! # What "closed form" rules out
//!
//! Every [`Estimate`](bayes_core_ext::Estimate) in this crate is a **single
//! pass plus arithmetic** — a method of moments, a maximum likelihood estimator
//! that happens to be explicit, a log-scale moment match, or an order statistic. None of them
//! iterates to a convergence criterion. That is a deliberate boundary: a
//! Weibull or Frechet shape is fit here by matching moments of `ln x`, for
//! which the answer is a formula, rather than by Newton steps on the
//! likelihood, which would be a solver.
//!
//! The cost is honesty about preconditions. A method of moments can ask for a
//! parameter its own moment equations put out of range — a negative-binomial
//! fit needs the sample variance to exceed the mean, and an inverse-gamma fit
//! needs a finite one. Every such estimator returns `None` rather than clamping
//! into range, because a clamped parameter is a fit that silently reports
//! something the data did not say.
//!
//! # The three families with no `Estimate` at all
//!
//! [`Binomial`](crate::Binomial) and [`BetaBinomial`](crate::BetaBinomial) need
//! a trial count, and [`Hypergeometric`](crate::Hypergeometric) needs a
//! population and a draw size. Those are **structural**: properties of the
//! experiment, not of the data. A sample of counts cannot recover them, so
//! these families have no `from_sample`.
//!
//! The temptation is to fit them anyway — method of moments gives
//! `n̂ = m²/(m − v)` for a binomial — but that is negative for every
//! overdispersed sample, which is most real ones, and the overdispersed case is
//! precisely when someone reaches for a beta-binomial. Supply the structural
//! parameter and fit the rest.
//!
//! # Examples
//!
//! ```
//! use bayes_distribution_ext::{Estimate, LogNormal, Moments, NegativeBinomial, Poisson};
//!
//! // A fit is a formula over one pass of the data.
//! let durations = [1.2, 0.8, 2.4, 1.0, 3.1, 0.6, 1.7, 2.0];
//! let fitted = LogNormal::from_sample(&durations).expect("positive data fits");
//! assert!(fitted.median() > 0.0);
//!
//! // Where the family cannot represent the sample, the answer is `None`
//! // rather than a clamped parameter. These counts are underdispersed, and a
//! // negative binomial is always overdispersed:
//! let counts = [4.0, 5.0, 4.0, 5.0, 4.0, 5.0];
//! assert!(NegativeBinomial::from_sample(&counts).is_none());
//! // The same sample is exactly what a Poisson is for.
//! assert!(Poisson::from_sample(&counts).is_some());
//!
//! // And a log-normal declines data it has no mass on at all.
//! assert!(LogNormal::from_sample(&[1.0, 2.0, 0.0]).is_none());
//! ```

pub mod fisher;
pub mod sample;

pub use fisher::FisherInformation;
