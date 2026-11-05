//! Weighted reductions, and their carrier-generic `_real` twins.
//!
//! | module | holds |
//! |--------|-------|
//! | [`normalize`] | weight sums, normalization (linear and log-domain), ESS |
//! | [`moments`] | weighted mean, variance, covariance, correlation |
//! | [`order`] | the weighted quantile |

mod moments;
mod normalize;
mod order;

pub use moments::{
    weighted_correlation, weighted_correlation_real, weighted_covariance, weighted_covariance_real,
    weighted_mean, weighted_mean_real, weighted_population_variance,
    weighted_population_variance_real, weighted_std_dev, weighted_std_dev_real, weighted_variance,
    weighted_variance_real,
};
pub use normalize::{
    effective_sample_size, effective_sample_size_real, is_effective_sample_size_admissible,
    normalized_log_weights, normalized_weights, normalized_weights_real, weight_sum,
    weight_sum_real,
};
pub use order::weighted_quantile;
