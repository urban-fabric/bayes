//! Continuous distribution parameter types for the lazy layer.

mod bounded;
mod circular;
mod gamma_family;
mod positive_support;
mod symmetric;

pub use bounded::{Beta, Uniform};
pub use circular::VonMises;
pub use gamma_family::{ChiSquared, Exponential, FisherSnedecor, Gamma, InverseGamma};
pub use positive_support::{
    Frechet, HalfNormal, LogNormal, Lomax, Pareto, Rayleigh, Wald, Weibull,
};
pub use symmetric::{Cauchy, Gumbel, Laplace, Logistic, Normal, StudentT};

use bayes_error::BayesError;

fn finite(name: &'static str, value: f64) -> Result<f64, BayesError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(BayesError::NonFinite { name, value })
    }
}

fn positive(name: &'static str, value: f64) -> Result<f64, BayesError> {
    finite(name, value)?;
    if value > 0.0 {
        Ok(value)
    } else {
        Err(BayesError::NotPositive { name, value })
    }
}
