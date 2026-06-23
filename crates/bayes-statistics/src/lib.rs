//! **Sample statistics** over data (`&[f64]`):
//!
//! * `summary` / `order` — mean, variance, skewness, kurtosis, covariance,
//!   correlation; robust summaries, z-scores, quantiles, median, IQR, MAD, Tukey fences.
//! * `correlation` — rank correlation (Spearman ρ, Kendall τ-b) and Cramér's V.
//! * `effect` — standardized mean effects, ANOVA effect sizes, odds/risk effects.
//! * `weighted` — weight normalization (raw or log-domain), ESS diagnostics, weighted
//!   mean/variance/covariance/correlation, and quantiles.
//! * `regression` — ordinary least squares (simple linear) with diagnostics; the
//!   matrix-design multiple regression lives in the sibling crate `ito`.
//! * `resampling` — bootstrap and jackknife with percentile intervals.
//! * [`hypothesis`] — z / t / Welch / F / χ² / Mann–Whitney / KS / Jarque–Bera /
//!   ANOVA / Wilcoxon / Fisher's exact / binomial / McNemar, each returning a
//!   [`TestResult`].
//! * `confidence` — closed-form intervals for means, proportions, rates, variances,
//!   standard deviations, and two-sample differences, plus a distribution-free
//!   (Chebyshev/Hoeffding/McDiarmid/Chernoff/Bernstein/Paley–Zygmund/Popoviciu)
//!   concentration tier.
//! * `estimation` — Cramér–Rao (and Bhattacharyya higher-order) efficiency diagnostics,
//!   observed Fisher information, Rao–Blackwellization / grouped variance decomposition, and
//!   the Lehmann–Scheffé UMVU certifier.
//! * `inequalities` — proof-backed moment certifiers: Cauchy–Schwarz consistency and
//!   the Jensen gap.
//! * `empirical` — empirical CDF ([`Ecdf`]) and [`histogram`]; `kde` — Gaussian KDE.
//! * `multivariate` — finite rectangular sample validation, vector means, small vector
//!   helpers, and paired-observation covariance/correlation.
//! * streaming accumulators — [`RunningStats`], [`RunningCovariance`], and the [`TDigest`]
//!   quantile sketch (single-pass and mergeable).
//!
//! Summary/order stats are available both as free functions and as the
//! [`SampleStats`] extension trait on `[f64]`, so `bayes::statistics::mean(data)`
//! and `data.mean()` both work. Degenerate inputs yield `f64::NAN`, never panic.

mod confidence;
mod correlation;
mod effect;
mod empirical;
mod estimation;
pub mod hypothesis;
mod inequalities;
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
    ConfidenceInterval, bernstein_bound, cantelli_bound, chebyshev_bound, chebyshev_sample_size,
    chernoff_bound, hoeffding_bound, is_variance_admissible, markov_bound, max_variance_for_range,
    mcdiarmid_bound, mean_chebyshev_interval, mean_chebyshev_interval_bounded,
    mean_difference_welch_interval, mean_t_interval, mean_z_interval, paley_zygmund_bound,
    poisson_rate_interval, proportion_agresti_coull, proportion_clopper_pearson,
    proportion_difference_wald, proportion_jeffreys, proportion_wald, proportion_wilson,
    std_dev_chi_square_interval, variance_chi_square_interval,
};
pub use correlation::{cramers_v, kendall_tau, ranks, spearman};
pub use effect::{
    cohen_d, cohen_d_real, eta_squared, eta_squared_real, glass_delta, glass_delta_real, hedges_g,
    hedges_g_real, odds_ratio, odds_ratio_real, omega_squared, omega_squared_real, pooled_std_dev,
    pooled_std_dev_real, risk_difference, risk_difference_real, risk_ratio, risk_ratio_real,
};
pub use empirical::{Ecdf, Histogram, histogram};
pub use estimation::{
    VarianceDecomposition, bhattacharyya_lower_bound, conditional_variance, cramer_rao_lower_bound,
    fisher_information_from_scores, is_bhattacharyya_admissible, is_cr_admissible, is_umvu,
    rao_blackwell, relative_efficiency, variance_decomposition,
};
pub use hypothesis::TestResult;
pub use inequalities::{cauchy_schwarz_moment_holds, jensen_gap_lower_bound};
pub use kde::KernelDensity;
pub use multivariate::{
    RunningCovariance, dot, dot_real, mean_vector, mean_vector_real, norm2, norm2_real,
    sample_shape,
};
pub use online::RunningStats;
pub use order::{
    five_number_summary, iqr, median, median_abs_deviation, percentile, quantile, tukey_fences,
    tukey_outlier_flags,
};
pub use regression::LinearRegression;
pub use resampling::{bootstrap, jackknife, percentile_interval};
pub use summary::{
    coefficient_of_variation, coefficient_of_variation_real, correlation, correlation_real,
    covariance, covariance_real, geometric_mean, geometric_mean_real, gini_mean_difference,
    gini_mean_difference_real, harmonic_mean, harmonic_mean_real, kurtosis, kurtosis_real, mean,
    mean_abs_deviation,
    mean_abs_deviation_real, mean_real, population_variance, population_variance_real, skewness,
    skewness_real, std_dev, std_dev_real, trimmed_mean, variance, variance_real, winsorized_mean,
    z_scores, z_scores_real,
};
pub use tdigest::TDigest;
pub use weighted::{
    effective_sample_size, effective_sample_size_real, is_effective_sample_size_admissible,
    normalized_log_weights, normalized_weights, normalized_weights_real, weight_sum,
    weight_sum_real, weighted_correlation, weighted_correlation_real, weighted_covariance,
    weighted_covariance_real, weighted_mean, weighted_mean_real, weighted_population_variance,
    weighted_population_variance_real, weighted_quantile, weighted_std_dev, weighted_std_dev_real,
    weighted_variance, weighted_variance_real,
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
