//! **Sample statistics** over data (`&[f64]`):
//!
//! * `summary` / `order` — mean, variance, skewness, kurtosis, covariance,
//!   correlation; robust summaries, z-scores, quantiles, median, IQR, MAD, Tukey fences.
//! * `correlation` — rank correlation (Spearman ρ, Kendall τ-b) and Cramér's V.
//! * `effect` — standardized mean effects, ANOVA effect sizes, odds/risk effects.
//! * `weighted` — weighted mean/variance/covariance/correlation and quantiles.
//! * `multivariate` — sample mean/covariance/correlation matrices and Mahalanobis
//!   / Hotelling statistics.
//! * `regression` — ordinary least squares (simple and multiple) with diagnostics.
//! * `resampling` — bootstrap and jackknife with percentile intervals.
//! * [`hypothesis`] — z / t / Welch / F / χ² / Mann–Whitney / KS / Jarque–Bera /
//!   ANOVA / Wilcoxon / Fisher's exact / binomial / McNemar, each returning a
//!   [`TestResult`].
//! * `confidence` — closed-form intervals for means, proportions, rates, variances,
//!   standard deviations, and two-sample differences.
//! * `empirical` — empirical CDF ([`Ecdf`]) and [`histogram`]; `kde` — Gaussian KDE.
//! * streaming accumulators — [`RunningStats`], [`RunningCovariance`], and the
//!   [`TDigest`] quantile sketch (all single-pass and mergeable).
//!
//! Summary/order stats are available both as free functions and as the
//! [`SampleStats`] extension trait on `[f64]`, so `bayes::statistics::mean(data)`
//! and `data.mean()` both work. Degenerate inputs yield `f64::NAN`, never panic.

mod confidence;
mod correlation;
mod effect;
mod empirical;
pub mod hypothesis;
mod kde;
mod multivariate;
mod online;
mod order;
mod regression;
mod resampling;
mod summary;
mod tdigest;
mod weighted;

pub use confidence::{
    ConfidenceInterval, mean_difference_welch_interval, mean_t_interval, mean_z_interval,
    poisson_rate_interval, proportion_agresti_coull, proportion_clopper_pearson,
    proportion_difference_wald, proportion_jeffreys, proportion_wald, proportion_wilson,
    std_dev_chi_square_interval, variance_chi_square_interval,
};
pub use correlation::{cramers_v, kendall_tau, ranks, spearman};
pub use effect::{
    cohen_d, eta_squared, glass_delta, hedges_g, odds_ratio, omega_squared, pooled_std_dev,
    risk_difference, risk_ratio,
};
pub use empirical::{Ecdf, Histogram, histogram};
pub use hypothesis::TestResult;
pub use kde::KernelDensity;
pub use multivariate::{
    correlation_matrix, covariance_matrix, hotelling_t_squared, mahalanobis_squared, mean_vector,
    pooled_covariance_matrix,
};
pub use online::{RunningCovariance, RunningStats};
pub use order::{
    five_number_summary, iqr, median, median_abs_deviation, percentile, quantile, tukey_fences,
    tukey_outlier_flags,
};
pub use regression::{LinearRegression, fit_least_squares};
pub use resampling::{bootstrap, jackknife, percentile_interval};
pub use summary::{
    coefficient_of_variation, correlation, covariance, geometric_mean, gini_mean_difference,
    harmonic_mean, kurtosis, mean, mean_abs_deviation, population_variance, skewness, std_dev,
    trimmed_mean, variance, winsorized_mean, z_scores,
};
pub use tdigest::TDigest;
pub use weighted::{
    effective_sample_size, weight_sum, weighted_correlation, weighted_covariance, weighted_mean,
    weighted_population_variance, weighted_quantile, weighted_std_dev, weighted_variance,
};

/// Summary + order statistics as methods on a slice of samples (`data.mean()`,
/// `data.median()`, …) — the ergonomic mirror of the free functions.
pub trait SampleStats {
    fn mean(&self) -> f64;
    /// Unbiased sample variance (Bessel's `n-1`).
    fn variance(&self) -> f64;
    fn std_dev(&self) -> f64;
    fn skewness(&self) -> f64;
    /// Excess kurtosis.
    fn kurtosis(&self) -> f64;
    fn geometric_mean(&self) -> f64;
    fn harmonic_mean(&self) -> f64;
    fn coefficient_of_variation(&self) -> f64;
    fn mean_abs_deviation(&self) -> f64;
    fn trimmed_mean(&self, proportion: f64) -> f64;
    fn winsorized_mean(&self, proportion: f64) -> f64;
    fn gini_mean_difference(&self) -> f64;
    fn quantile(&self, tau: f64) -> f64;
    fn median(&self) -> f64;
    fn iqr(&self) -> f64;
    fn median_abs_deviation(&self) -> f64;
}

impl SampleStats for [f64] {
    fn mean(&self) -> f64 {
        mean(self)
    }
    fn variance(&self) -> f64 {
        variance(self)
    }
    fn std_dev(&self) -> f64 {
        std_dev(self)
    }
    fn skewness(&self) -> f64 {
        skewness(self)
    }
    fn kurtosis(&self) -> f64 {
        kurtosis(self)
    }
    fn geometric_mean(&self) -> f64 {
        geometric_mean(self)
    }
    fn harmonic_mean(&self) -> f64 {
        harmonic_mean(self)
    }
    fn coefficient_of_variation(&self) -> f64 {
        coefficient_of_variation(self)
    }
    fn mean_abs_deviation(&self) -> f64 {
        mean_abs_deviation(self)
    }
    fn trimmed_mean(&self, proportion: f64) -> f64 {
        trimmed_mean(self, proportion)
    }
    fn winsorized_mean(&self, proportion: f64) -> f64 {
        winsorized_mean(self, proportion)
    }
    fn gini_mean_difference(&self) -> f64 {
        gini_mean_difference(self)
    }
    fn quantile(&self, tau: f64) -> f64 {
        quantile(self, tau)
    }
    fn median(&self) -> f64 {
        median(self)
    }
    fn iqr(&self) -> f64 {
        iqr(self)
    }
    fn median_abs_deviation(&self) -> f64 {
        median_abs_deviation(self)
    }
}
