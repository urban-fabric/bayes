//! **Fitting**: every `Estimate` implementation, against data it generated.
//!
//! A closed-form estimator is a formula, and a formula is wrong in quiet ways —
//! a swapped constructor argument, a variance where a standard deviation
//! belongs, `n` where `n − 1` belongs, a parameterization that counts failures
//! where the sample counts trials. None of those produce an error or a `NaN`.
//! They produce a plausible number that is not the right one.
//!
//! So the load-bearing suite here is **round-trip recovery**: sample from a
//! family whose parameters are known, fit it, and ask whether the parameters
//! come back. That closes the loop through the sampler, the estimator and the
//! constructor at once, and it is the only check that can see a formula which
//! is self-consistent but mis-derived.
//!
//! | module | covers |
//! |--------|--------|
//! | [`harness`] | the recovery check, and why its tolerance is the fits' own scatter |
//! | [`continuous`] | 22 continuous families, plus Student's t separately |
//! | [`discrete`] | the 6 fittable discrete families |
//! | [`refusal`] | what every estimator must *decline* to fit |
//!
//! [`refusal`] is the other half and not an afterthought. A method of moments
//! can be handed a sample whose moments no member of the family possesses — an
//! underdispersed negative binomial, a beta sample more spread out than any
//! beta — and the useful behaviour there is `None`, not a clamped parameter
//! that reports something the data did not say.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

mod continuous;
mod discrete;
mod harness;
mod refusal;
