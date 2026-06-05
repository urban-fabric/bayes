//! **Sample statistics** over data (`&[f64]`):
//!
//! * `summary` / `order` — mean, variance, skewness, kurtosis, covariance,
//!   correlation; quantiles, median, IQR, MAD.
//! * `correlation` — rank correlation (Spearman ρ, Kendall τ-b) and Cramér's V.
//! * `regression` — ordinary least squares (simple and multiple).
//! * `resampling` — bootstrap and jackknife with percentile intervals.
//! * [`hypothesis`] — z / t / Welch / χ² / Mann–Whitney / KS / Jarque–Bera / ANOVA /
//!   Wilcoxon / Fisher's exact, each returning a [`TestResult`].
//! * `confidence` — closed-form intervals (t/z means, Wald/Wilson proportions).
//! * `empirical` — empirical CDF ([`Ecdf`]) and [`histogram`]; `kde` — Gaussian KDE.
//! * streaming accumulators — [`RunningStats`], [`RunningCovariance`], and the
//!   [`TDigest`] quantile sketch (all single-pass and mergeable).
//!
//! Summary/order stats are available both as free functions and as the
//! [`SampleStats`] extension trait on `[f64]`, so `bayes::statistics::mean(data)`
//! and `data.mean()` both work. Degenerate inputs yield `f64::NAN`, never panic.

mod confidence;
mod correlation;
mod empirical;
pub mod hypothesis;
mod kde;
mod online;
mod order;
mod regression;
mod resampling;
mod summary;
mod tdigest;

pub use confidence::{
    ConfidenceInterval, mean_t_interval, mean_z_interval, proportion_wald, proportion_wilson,
};
pub use correlation::{cramers_v, kendall_tau, ranks, spearman};
pub use empirical::{Ecdf, Histogram, histogram};
pub use hypothesis::TestResult;
pub use kde::KernelDensity;
pub use online::{RunningCovariance, RunningStats};
pub use order::{five_number_summary, iqr, median, median_abs_deviation, percentile, quantile};
pub use regression::{LinearRegression, fit_least_squares};
pub use resampling::{bootstrap, jackknife, percentile_interval};
pub use summary::{
    correlation, covariance, geometric_mean, harmonic_mean, kurtosis, mean, population_variance,
    skewness, std_dev, variance,
};
pub use tdigest::TDigest;

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
