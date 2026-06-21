//! **Closed-form confidence intervals** — the frequentist companions to the point
//! estimates in [`summary`](super::summary). Each returns a [`ConfidenceInterval`]
//! at a requested coverage `level` (e.g. `0.95`), computed from an analytic
//! quantile — no resampling (for that, see [`resampling`](super::resampling)).
//!
//! Grouped by target: `mean` (Student-t, known-σ normal, Welch difference),
//! `proportion` (Wald, Wilson, Agresti–Coull, Clopper–Pearson, Jeffreys, and the
//! two-sample difference), and `chi_squared` (Poisson rate, variance, std dev).
//! The shared [`ConfidenceInterval`] type and quantile helpers live here.
//!
//! The `concentration` group is the **distribution-free, proof-backed** companion:
//! Markov/Chebyshev/Cantelli/Hoeffding/McDiarmid tail bounds, the Chernoff-method and
//! variance-aware Bernstein exponential tails, the Paley–Zygmund anti-concentration (lower) bound, a
//! Chebyshev mean interval and sample-size calculator, and the Popoviciu variance cap.
//! Unlike the parametric intervals above, these assume only finite variance (or bounded
//! support) and carry a *guaranteed* bound — each backed by a machine-checked theorem in
//! riemann `RiemannLean/Bayes/Inequalities.lean`.

use euclid::Real;
use std::f64::consts::SQRT_2;

mod chi_squared;
mod concentration;
mod mean;
mod proportion;

pub use chi_squared::{
    poisson_rate_interval, std_dev_chi_square_interval, variance_chi_square_interval,
};
pub use concentration::{
    bernstein_bound, cantelli_bound, chebyshev_bound, chebyshev_sample_size, chernoff_bound,
    hoeffding_bound, is_variance_admissible, markov_bound, max_variance_for_range, mcdiarmid_bound,
    mean_chebyshev_interval, mean_chebyshev_interval_bounded, paley_zygmund_bound,
};
pub use mean::{mean_difference_welch_interval, mean_t_interval, mean_z_interval};
pub use proportion::{
    proportion_agresti_coull, proportion_clopper_pearson, proportion_difference_wald,
    proportion_jeffreys, proportion_wald, proportion_wilson,
};

/// A confidence interval `[lower, upper]` at coverage `level ∈ (0, 1)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConfidenceInterval {
    pub lower: f64,
    pub upper: f64,
    pub level: f64,
}

impl ConfidenceInterval {
    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }
    /// Whether `x` lies within the interval (inclusive).
    pub fn contains(&self, x: f64) -> bool {
        x >= self.lower && x <= self.upper
    }
}

/// Standard-normal quantile `Φ⁻¹(q)` via the inverse error function.
fn z_quantile(q: f64) -> f64 {
    SQRT_2 * (2.0 * q - 1.0).erf_inv()
}

fn invalid(level: f64) -> ConfidenceInterval {
    ConfidenceInterval {
        lower: f64::NAN,
        upper: f64::NAN,
        level,
    }
}

fn valid_level(level: f64) -> bool {
    (0.0..1.0).contains(&level)
}

fn alpha(level: f64) -> f64 {
    1.0 - level
}
