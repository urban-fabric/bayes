//! **Exact samplers for distributions that do not sample easily.**
//!
//! The families in `bayes-distribution-ext` each know how to draw from
//! themselves, because each has a closed-form inverse CDF or a known algorithm.
//! This crate is for everything else: an unnormalized posterior, a density you
//! wrote down five minutes ago, a categorical over fifty thousand outcomes, a
//! truncated or reweighted version of something standard.
//!
//! Everything here is **exact**. There is no burn-in, no autocorrelation, and no
//! convergence diagnostic, because none of these methods is a Markov chain: a
//! draw that survives is distributed precisely as the target, and consecutive
//! draws are independent. That is the property that makes them worth reaching
//! for before MCMC, and the reason the boundary in this workspace sits where it
//! does — sampling *from a density you can evaluate* is this crate's job;
//! exploring a posterior you cannot is an inference layer's.
//!
//! ## Choosing
//!
//! | you have | use | cost per draw |
//! |----------|-----|---------------|
//! | finite outcomes with known weights | [`AliasTable`] | O(1), two coin flips |
//! | a density and a known envelope | [`RejectionSampler`] | O(1/acceptance) |
//! | a **log-concave** density and its derivative | [`AdaptiveRejection`] | tends to 1 proposal |
//! | an invertible CDF | nothing here — use `inverse_cdf(u)` directly | O(1) |
//!
//! The ordering is deliberate. If the CDF inverts, invert it; nothing beats one
//! uniform and one closed form. If it does not but the density is log-concave,
//! [`AdaptiveRejection`] needs no tuning at all and converges to near-perfect
//! acceptance by itself. Only when neither holds do you have to construct an
//! envelope by hand, and then [`RejectionSampler`] will at least tell you when
//! you got it wrong.
//!
//! ## The one mistake that does not announce itself
//!
//! A rejection envelope that dips below the target does not panic, does not
//! return an error, and does not make the output look wrong. It silently
//! under-represents the region where the envelope failed. Both samplers here
//! therefore track it — [`RejectionSampler::envelope_was_breached`] and
//! [`AdaptiveRejection::hull_was_breached`] — and both should be checked, not
//! assumed.

mod adaptive;
mod alias;
mod quasi;
mod rejection;

pub use adaptive::AdaptiveRejection;
pub use alias::AliasTable;
pub use quasi::{Halton, Sobol};
pub use rejection::{Draw, RejectionSampler, RngCoreShim};
