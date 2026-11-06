//! Summary statistics over `&[f64]`, and their carrier-generic `_real` twins.
//!
//! | module | holds |
//! |--------|-------|
//! | [`location`] | means, and the shared two-pass mean/`M2` accumulator |
//! | [`spread`] | variance, standard deviation, and the robust spreads |
//! | [`shape`] | standardized moments, z-scores, covariance, correlation, trimmed means |

mod location;
mod shape;
mod spread;

pub use location::{
    geometric_mean, geometric_mean_real, harmonic_mean, harmonic_mean_real, mean, mean_real,
};
pub use shape::{
    correlation, correlation_real, covariance, covariance_real, kurtosis, kurtosis_real, skewness,
    skewness_real, trimmed_mean, winsorized_mean, z_scores, z_scores_real,
};
pub use spread::{
    coefficient_of_variation, coefficient_of_variation_real, gini_mean_difference,
    gini_mean_difference_real, mean_abs_deviation, mean_abs_deviation_real, population_variance,
    population_variance_real, std_dev, std_dev_real, variance, variance_real,
};
