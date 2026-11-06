//! The Gaussian conjugate models.
//!
//! | module | unknown |
//! |--------|---------|
//! | [`known_variance`] | the mean, with the variance given |
//! | [`inverse_gamma`] | the mean *and* the variance |

mod inverse_gamma;
mod known_variance;

pub use inverse_gamma::{NormalInverseGamma, NormalInverseGammaModel};
pub use known_variance::NormalKnownVariance;
