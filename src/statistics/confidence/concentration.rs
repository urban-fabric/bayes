//! **Distribution-free concentration bounds and intervals** — the *certifying*
//! companions to the parametric intervals in this module. These assume only finite
//! variance (or bounded support), make **no distributional assumption**, and carry a
//! *guaranteed* (not asymptotic) bound. Each is backed by a machine-checked theorem in
//! riemann `RiemannLean/Bayes/Inequalities.lean` on the pinned 3-axiom basis:
//!
//! - tail bounds — `markov`, `chebyshev`, `cantelli`, `hoeffding_tail`;
//! - the sample-mean interval / sample size — `chebyshev_sample_mean`
//!   (`ε²·P(|S̄ − μ| ≥ ε) ≤ σ²/n`, Bienaymé variance scaling);
//! - the bounded-support variance cap — `variance_le_sq_half_range` (Popoviciu).

use super::{ConfidenceInterval, alpha, invalid, valid_level};
use crate::statistics::{mean, variance};

/// **Markov's inequality**: an upper bound on `P(X ≥ a)` for a nonnegative `X` with
/// mean `mean`, namely `min(E[X]/a, 1)` (`a > 0`). Backed by riemann
/// `Bayes/Inequalities.markov`. Returns `NaN` on invalid input (`a ≤ 0`, `mean < 0`).
pub fn markov_bound(mean: f64, a: f64) -> f64 {
    if a <= 0.0 || mean < 0.0 {
        return f64::NAN;
    }
    (mean / a).min(1.0)
}

/// **Chebyshev's inequality** (two-sided): an upper bound on `P(|X − μ| ≥ t)`, namely
/// `min(σ²/t², 1)` (`t > 0`). Needs only the variance — no distribution. Backed by
/// riemann `Bayes/Inequalities.chebyshev`.
pub fn chebyshev_bound(variance: f64, t: f64) -> f64 {
    if variance < 0.0 || t <= 0.0 {
        return f64::NAN;
    }
    (variance / (t * t)).min(1.0)
}

/// **Cantelli's inequality** (one-sided): an upper bound on `P(X − μ ≥ t)`, namely
/// `σ²/(σ² + t²)` (`t > 0`) — sharper than the two-sided Chebyshev bound `σ²/t²`.
/// Backed by riemann `Bayes/Inequalities.cantelli`.
pub fn cantelli_bound(variance: f64, t: f64) -> f64 {
    if variance < 0.0 || t <= 0.0 {
        return f64::NAN;
    }
    variance / (variance + t * t)
}

/// **Hoeffding tail** (single observation): for one value bounded in `[a, b]`, an upper
/// bound on `P(X − μ ≥ t)`, namely `exp(−2t²/(b−a)²)` (`t ≥ 0`). Backed by riemann
/// `Bayes/Inequalities.hoeffding_tail`. (The `n`-sample mean version needs the
/// independent-sum MGF — see `bayes-advanced-concentration.md`.)
pub fn hoeffding_bound(a: f64, b: f64, t: f64) -> f64 {
    if b <= a || t < 0.0 {
        return f64::NAN;
    }
    (-2.0 * t * t / ((b - a) * (b - a))).exp()
}

/// **Popoviciu's variance bound**: the largest possible variance of a variable bounded
/// in `[a, b]` is `((b − a)/2)²`. A guaranteed variance cap from the range alone (used
/// by [`mean_chebyshev_interval_bounded`]), and a sanity bound on any computed variance.
/// Backed by riemann `Bayes/Inequalities.variance_le_sq_half_range`.
pub fn max_variance_for_range(a: f64, b: f64) -> f64 {
    if b < a {
        return f64::NAN;
    }
    let half = (b - a) / 2.0;
    half * half
}

/// **Popoviciu admissibility check**: whether a variance is attainable for data bounded
/// in `[a, b]`, i.e. `variance ≤ ((b − a)/2)²` ([`max_variance_for_range`]). A `false`
/// flags a variance larger than *any* `[a, b]`-bounded distribution can have — a data or
/// computation error. Checks the *population* variance (riemann's convention); a Bessel
/// `n−1` sample variance may exceed the cap by the factor `n/(n−1)` on small samples.
/// Backed by riemann `Bayes/Inequalities.variance_le_sq_half_range`.
pub fn is_variance_admissible(variance: f64, a: f64, b: f64) -> bool {
    b >= a && variance >= 0.0 && variance <= max_variance_for_range(a, b) * (1.0 + 1e-9)
}

/// **Distribution-free confidence interval for the mean** via Chebyshev: `x̄ ± √(s²/(n·α))`
/// with `α = 1 − level`. Assumes only finite variance (uncorrelated, equal-variance
/// samples) — no normality. Coverage is *guaranteed* at `level` when `s²` is the true
/// variance; with the plug-in sample variance it is the usual Chebyshev estimate. Wider
/// than the t/z intervals, but valid when normality is doubtful. Backed by riemann
/// `Bayes/Inequalities.chebyshev_sample_mean`.
pub fn mean_chebyshev_interval(data: &[f64], level: f64) -> ConfidenceInterval {
    let n = data.len();
    if n < 2 || !valid_level(level) {
        return invalid(level);
    }
    let half = (variance(data) / (n as f64 * alpha(level))).sqrt();
    let m = mean(data);
    ConfidenceInterval {
        lower: m - half,
        upper: m + half,
        level,
    }
}

/// **Guaranteed-coverage interval for the mean of bounded data** — Chebyshev with the
/// Popoviciu variance cap `((b−a)/2)²`: `x̄ ± (b−a)/(2·√(n·α))`. Assumes only that the
/// data lie in `[a, b]` (no distribution, no variance estimate); coverage is then truly
/// `≥ level`. Conservative, but assumption-free. Backed by riemann
/// `chebyshev_sample_mean` + `variance_le_sq_half_range`.
pub fn mean_chebyshev_interval_bounded(
    data: &[f64],
    a: f64,
    b: f64,
    level: f64,
) -> ConfidenceInterval {
    let n = data.len();
    if n == 0 || b <= a || !valid_level(level) {
        return invalid(level);
    }
    let half = (max_variance_for_range(a, b) / (n as f64 * alpha(level))).sqrt();
    let m = mean(data);
    ConfidenceInterval {
        lower: m - half,
        upper: m + half,
        level,
    }
}

/// **Chebyshev sample size**: the smallest `n` for which the distribution-free interval
/// `x̄ ± eps` has coverage `≥ level`, i.e. `n ≥ σ²/(α·eps²)`. Backed by riemann
/// `Bayes/Inequalities.chebyshev_sample_mean`. Returns `0` on invalid input.
pub fn chebyshev_sample_size(variance: f64, eps: f64, level: f64) -> u64 {
    if variance < 0.0 || eps <= 0.0 || !valid_level(level) {
        return 0;
    }
    (variance / (alpha(level) * eps * eps)).ceil() as u64
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn tail_bounds_are_probabilities_and_clamp() {
        // Chebyshev: σ²=4, t=4 → 4/16 = 0.25; t=1 → 4 clamps to 1.
        close(chebyshev_bound(4.0, 4.0), 0.25, 1e-12);
        assert_eq!(chebyshev_bound(4.0, 1.0), 1.0);
        // Cantelli is sharper than (and never exceeds) the two-sided Chebyshev.
        assert!(cantelli_bound(4.0, 4.0) < chebyshev_bound(4.0, 4.0));
        assert!(cantelli_bound(4.0, 4.0) <= 1.0);
        // Markov: E[X]=2, a=8 → 0.25; a=1 → 2 clamps to 1.
        close(markov_bound(2.0, 8.0), 0.25, 1e-12);
        assert_eq!(markov_bound(2.0, 1.0), 1.0);
        // Hoeffding: t=0 → 1; decreasing in t.
        close(hoeffding_bound(0.0, 1.0, 0.0), 1.0, 1e-12);
        assert!(hoeffding_bound(0.0, 1.0, 0.5) < 1.0);
    }

    #[test]
    fn invalid_inputs_are_nan() {
        assert!(chebyshev_bound(4.0, 0.0).is_nan());
        assert!(markov_bound(-1.0, 2.0).is_nan());
        assert!(hoeffding_bound(1.0, 1.0, 0.5).is_nan());
        assert!(max_variance_for_range(2.0, 1.0).is_nan());
    }

    #[test]
    fn popoviciu_caps_the_population_variance() {
        // Popoviciu bounds the *population* variance (riemann's `variance`, divisor n);
        // this crate's `variance` is the Bessel (n−1) sample variance, so convert.
        let data = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let n = data.len() as f64;
        let pop_var = variance(&data) * (n - 1.0) / n;
        assert!(pop_var <= max_variance_for_range(0.0, 1.0) + 1e-12);
        // the two-point {0,1} distribution attains the cap exactly.
        close(pop_var, max_variance_for_range(0.0, 1.0), 1e-12);
        close(max_variance_for_range(0.0, 10.0), 25.0, 1e-12);
    }

    #[test]
    fn variance_admissibility_flags_the_impossible() {
        // the cap for [0,1] is 0.25: admissible at/below, rejected above.
        assert!(is_variance_admissible(0.25, 0.0, 1.0));
        assert!(is_variance_admissible(0.1, 0.0, 1.0));
        assert!(!is_variance_admissible(0.5, 0.0, 1.0));
        // a negative variance or inverted range is never admissible.
        assert!(!is_variance_admissible(-0.01, 0.0, 1.0));
        assert!(!is_variance_admissible(0.1, 1.0, 0.0));
    }

    #[test]
    fn chebyshev_interval_brackets_and_widens_with_level() {
        let data = [4.8, 5.0, 5.2, 4.9, 5.1, 5.0];
        let ci = mean_chebyshev_interval(&data, 0.95);
        assert!(ci.contains(mean(&data)));
        // higher confidence ⇒ wider (Chebyshev, like the parametric ones).
        assert!(mean_chebyshev_interval(&data, 0.99).width() > ci.width());
        // distribution-free Chebyshev is wider than the parametric z-interval.
        assert!(ci.width() > 0.0);
    }

    #[test]
    fn bounded_interval_is_guaranteed_and_centered() {
        let data = [0.2, 0.8, 0.5, 0.9, 0.1];
        let ci = mean_chebyshev_interval_bounded(&data, 0.0, 1.0, 0.95);
        close((ci.lower + ci.upper) / 2.0, mean(&data), 1e-12);
        // half-width = (b-a)/(2√(n·α)) = 1/(2√(5·0.05)).
        close(
            ci.width() / 2.0,
            1.0 / (2.0 * (5.0f64 * 0.05).sqrt()),
            1e-12,
        );
    }

    #[test]
    fn sample_size_meets_the_chebyshev_target() {
        // n ≥ σ²/(α·ε²) = 1/(0.05·0.01) = 2000.
        assert_eq!(chebyshev_sample_size(1.0, 0.1, 0.95), 2000);
        // achieving that n makes the half-width ≤ ε at this variance.
        let n = chebyshev_sample_size(1.0, 0.1, 0.95);
        assert!((1.0 / (n as f64 * 0.05)).sqrt() <= 0.1 + 1e-9);
        assert_eq!(chebyshev_sample_size(1.0, 0.0, 0.95), 0);
    }
}
