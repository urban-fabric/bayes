//! Low-discrepancy (quasi-random) sequences.
//!
//! Monte Carlo integration with pseudorandom points converges at `O(n^{-1/2})`,
//! and that rate is a property of the *randomness*, not of the integrand: random
//! points clump, leaving gaps that contribute nothing and clusters that
//! contribute redundantly. A low-discrepancy sequence fills the unit cube
//! deliberately instead, and buys `O(n^{-1}(log n)^d)` for smooth integrands —
//! for a 1000-point integral in low dimension that is roughly the accuracy of
//! a million random ones.
//!
//! The catch is that the points are **not random**. Consecutive values are
//! strongly dependent by construction, so they are correct for *integration* and
//! wrong for anything that needs independence: a bootstrap, a permutation test,
//! a simulation whose answer depends on the draws being i.i.d. If you need a
//! confidence interval on the estimate itself, randomize the sequence (scramble
//! or Cranley–Patterson rotate) and average over several independent
//! randomizations.
//!
//! | sequence | good for | watch out for |
//! |----------|----------|---------------|
//! | [`Halton`] | any dimension, incremental use | correlation between high dimensions |
//! | [`Sobol`] | up to the tabulated dimension, powers of two | needs the direction numbers |
//!
//! Both are **deterministic**: the `n`-th point is a pure function of `n`, so a
//! result reproduces without storing a seed, and two processes can split the
//! sequence by index with no coordination.

mod halton;
mod sobol;

pub use halton::Halton;
pub use sobol::Sobol;
