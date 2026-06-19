//! **Streaming / online statistics** — single-pass, mergeable accumulators that
//! summarise a stream without holding the data:
//!
//! * [`RunningStats`] — mean, variance, skewness, kurtosis, and extrema (Welford /
//!   Terriberry running moments).
//!
//! `merge`s two accumulators exactly, so a stream can be split across workers and
//! recombined. The paired-observation `RunningCovariance` (covariance / correlation /
//! OLS slope) lives in the matrix-coupled sibling crate `ito`.

mod running_stats;

pub use running_stats::RunningStats;
