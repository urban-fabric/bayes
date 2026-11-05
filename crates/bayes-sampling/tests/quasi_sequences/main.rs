//! **Quasi-random sequences**: are they low-discrepancy, or just uniform?
//!
//! Sobol and Halton were the largest untested surface in this crate, and they
//! are the kind of code that fails quietly. A wrong Sobol direction number, or
//! a radical inverse with the digits assembled in the wrong order, still
//! produces points that fill `[0,1)` and still passes any test that only looks
//! at the marginal distribution. What it loses is the *structure* — and the
//! structure is the entire reason to use these instead of an RNG.
//!
//! So each sequence is held to the property that defines it rather than to
//! uniformity:
//!
//! | module | asks |
//! |--------|------|
//! | [`structure`] | Halton against the radical inverse; Sobol against its net property |
//! | [`contracts`] | the module's own documented claims, and what happens at the edges |
//!
//! The sharpest check is Sobol's **net property**: among the first `2^m`
//! points, each dimension must hit each of the `2^m` equal subintervals
//! *exactly once*. That is what makes a `(t,s)`-sequence in base 2 what it is,
//! it is destroyed by a single wrong direction number, and it is invisible to
//! any marginal test — a clumped sequence and a stratified one have the same
//! histogram at coarse resolution.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

mod contracts;
mod structure;

/// The van der Corput radical inverse of `n` in `base`, written out
/// independently of the library.
///
/// Digits of `n` in `base`, reflected about the radix point. This is the
/// definition, and it shares no code with the implementation under test — which
/// is the point, since a radical inverse that assembles its digits in the wrong
/// order is still a bijection onto a dense subset of `[0,1)`.
pub(crate) fn radical_inverse(mut n: u64, base: u64) -> f64 {
    let (mut place, mut total) = (1.0 / base as f64, 0.0);
    while n > 0 {
        total += place * (n % base) as f64;
        n /= base;
        place /= base as f64;
    }
    total
}

/// The first 32 primes, as Halton's per-dimension bases.
pub(crate) const PRIMES: [u64; 32] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131,
];
