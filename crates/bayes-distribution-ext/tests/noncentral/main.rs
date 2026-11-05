//! **Non-central families**: χ², F and t away from their null hypotheses.
//!
//! These three are what a test statistic follows when the null is *false*, which
//! makes them the distributions a power calculation needs. They are also the
//! three hardest things in the crate to check, because none has a closed-form
//! CDF — each is an infinite series, and a series is wrong in ways a formula is
//! not: a shape shifted by one, a factor of two in the mixture weights, a
//! truncation that stops before reaching the terms that carry the mass.
//!
//! The modules are organized by **what the answer is checked against**, because
//! that is what determines how much a passing check is worth:
//!
//! | module | checks against | strength |
//! |--------|----------------|----------|
//! | [`reduction`] | the central families, and each other | an identity, to `1e-12` |
//! | [`definition`] | samples drawn from the defining construction | `5σ` of sampling error |
//! | [`shape`] | itself — monotone, differentiable, invertible | internal consistency |
//! | [`power`] | a simulated t-test, and the approximation it retires | `6σ` of 20k trials |
//! | [`extremes`] | the normal limit, and the constructors' contract | derived from the skewness |
//!
//! The load-bearing one is [`reduction`]. At zero non-centrality each family
//! *is* its central counterpart, so the comparison is an identity rather than a
//! limit and can be asserted near machine precision — one relation that pins
//! the normalizer, the degrees-of-freedom bookkeeping and the series truncation
//! all at once, and that nothing else catches so cheaply.
//!
//! # The upper few ulps of a CDF are off limits here
//!
//! A probability reaches 1 to within an ulp long before its argument runs out of
//! range: `NC-t(10, 2.5)` has `1 − cdf(x) = 2.2e-16` by `x = 100` and stays
//! there. The quantile for any `p` inside that plateau is not a recoverable
//! number, and a bisection asked for one cannot bracket it — it widens until its
//! trip count runs out and returns `2^64`. So no check here integrates `Q(u)`
//! over the unit interval, which is exactly what asks for those `p`. See
//! [`definition`] for what replaced that route and why it is the better test
//! anyway, and [`shape`] for the range where the round trip does hold.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

mod definition;
mod extremes;
mod power;
mod reduction;
mod shape;
