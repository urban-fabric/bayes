//! The **statistics half of the relationship network**: every statistic and test
//! checked against the distribution or the other statistic that determines it.
//!
//! The principle is the one the whole suite rests on: two implementations that
//! share no code cannot agree while both are wrong. A `StudentT` survival
//! function and a hand-written `(x̄ − μ₀)/(s/√n)` have nothing in common, so when
//! `one_sample_t` matches both, all three are probably right.
//!
//! | module | asks |
//! |--------|------|
//! | [`tests_vs_distributions`] | does each test match the distribution it looks up? |
//! | [`exact_tests`] | does each exact test match brute-force enumeration? |
//! | [`statistics_vs_statistics`] | do algebraically-linked statistics agree? |
//! | [`intervals_and_bounds`] | are intervals the right quantiles, and do bounds hold? |
//! | [`weighted`] | do weighted reductions match literal repetition? |

mod exact_tests;
mod intervals_and_bounds;
mod statistics_vs_statistics;
mod tests_vs_distributions;
mod weighted;
