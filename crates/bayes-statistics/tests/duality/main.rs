//! **Duality**: a test and an interval are the same statement twice.
//!
//! A two-sided test at level `α` rejects exactly when the `1 − α` confidence
//! interval excludes the null value. That is not an approximation or a rule of
//! thumb — it is what makes the two reports interchangeable, and it couples two
//! functions that share no code: one builds a statistic and looks it up in a
//! reference distribution, the other inverts that lookup to find bounds. They
//! can only agree if the quantile, the standard error and the degrees of
//! freedom match in both.
//!
//! The existing relationship suite checks each test against *its own*
//! reference distribution, which is a within-function statement. This suite
//! checks functions against each other, which is where a disagreement about
//! `n` versus `n − 1`, or a two-sided factor of two, actually shows up.
//!
//! | module | asks |
//! |--------|------|
//! | [`intervals`] | does each test reject exactly where its interval excludes? |
//! | [`identities`] | do the algebraic identities between tests hold exactly? |
//! | [`proportions`] | do the proportion intervals deliver their nominal coverage? |
//!
//! The identities in [`identities`] are of the same kind as the dualities: a
//! two-sample `t²` is the one-way ANOVA `F` on the same two groups; a paired
//! `t` is a one-sample `t` on the differences; Welch is Student when the two
//! samples have equal size and equal variance. Each relates two independently
//! written functions.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

mod identities;
mod intervals;
mod proportions;
