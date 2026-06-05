//! **Streaming / online statistics** — single-pass, mergeable accumulators that
//! summarise a stream without holding the data:
//!
//! * [`RunningStats`] — mean, variance, skewness, kurtosis, and extrema (Welford /
//!   Terriberry running moments).
//! * [`RunningCovariance`] — covariance, Pearson correlation, and OLS slope/intercept
//!   over paired observations (Welford co-moment).
//!
//! Both `merge` two accumulators exactly, so a stream can be split across workers and
//! recombined.

mod running_covariance;
mod running_stats;

pub use running_covariance::RunningCovariance;
pub use running_stats::RunningStats;
