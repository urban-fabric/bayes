//! Explicit-key generic samplers for the lazy distribution structs.
//!
//! The `rand::Distribution` impls in `bayes-distribution-ext` remain the eager
//! host sampling surface. This module is the backend-generic companion:
//! immutable keys, pure draws, and optional shaped samples through euclid's
//! random sampler contract.
//!
//! The split is by **what a family needs to draw**, which is the same axis that
//! decides whether a backend can vectorize it:
//!
//! | module | needs |
//! |--------|-------|
//! | [`contract`] | the two traits themselves |
//! | [`location_scale`] | one uniform or normal variate; a closed-form transform |
//! | [`gamma_family`] | `GammaSampler::standard_gamma`, since the quantile has no closed form |
//! | [`count_family`] | `CountSampler`, for the same reason on the integer side |

mod contract;
mod count_family;
mod gamma_family;
mod location_scale;

pub use contract::{SampleTensorWithKey, SampleWithKey};
