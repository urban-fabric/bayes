//! The non-central χ², t and F families.
//!
//! These are what a **power calculation** actually needs. The central families
//! describe a statistic when the null hypothesis is true; the non-central ones
//! describe it when the null is false, with the non-centrality parameter
//! measuring how false. Without them, power has to be approximated through the
//! normal — which is what the `sample_size` example did, and what cost it a
//! point of accuracy at every sample size.
//!
//! ## Why these live here and not in `bayes-kernels`
//!
//! Every other density in the workspace is carrier-generic, written once in
//! `bayes-kernels` against `euclid_numeric::Real`. These three are not, and the
//! reason is structural rather than incidental: each is a **Poisson-weighted
//! mixture of central families**,
//!
//! ```text
//!     f(x; k, λ) = Σ_{j=0}^∞  e^{-λ/2}(λ/2)^j / j!  ·  f_central(x; k + 2j)
//! ```
//!
//! and the series has to be truncated where the terms stop mattering. That
//! truncation point depends on `λ` — the terms peak near `j ≈ λ/2` — so deciding
//! when to stop requires a host `bool` comparing a running term against a
//! tolerance. A carrier-generic version would need a fixed trip count sized for
//! the largest `λ` anyone might pass, which would be both wasteful at small `λ`
//! and still wrong at large.
//!
//! The consequence is worth stating plainly: these families do not run on a
//! deferred carrier. If you need a non-central density inside a compiled model,
//! the honest options are a fixed-trip-count approximation with the trip count
//! chosen for your `λ` range, or the normal approximation.
//!
//! ## Verification
//!
//! Each reduces **exactly** to its central counterpart at zero non-centrality,
//! which is the strongest available check and the one the test suite leans on:
//! `NoncentralChiSquared(k, 0) == ChiSquared(k)` is an identity, not an
//! approximation, so it can be asserted near machine precision.

mod chi_squared;
mod f;
mod series;
mod t;

pub use chi_squared::NoncentralChiSquared;
pub use f::NoncentralF;
pub use t::NoncentralT;
