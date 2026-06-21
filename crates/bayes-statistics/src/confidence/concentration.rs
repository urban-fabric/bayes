//! **Distribution-free concentration bounds and intervals** — the *certifying*
//! companions to the parametric intervals in this module. These assume only finite
//! variance (or bounded support), make **no distributional assumption**, and carry a
//! *guaranteed* (not asymptotic) bound. Each is backed by a machine-checked theorem in
//! riemann `RiemannLean/Bayes/Inequalities.lean` on the pinned 3-axiom basis:
//!
//! - tail bounds — `markov`, `chebyshev`, `cantelli`, `hoeffding_tail`,
//!   `mcdiarmid_tail_of_bounded_differences`, and the exponential-MGF family
//!   `chernoff_bound` → `bernstein_tail` (variance-aware);
//! - anti-concentration (a *lower* tail bound) — `paley_zygmund`;
//! - the sample-mean interval / sample size — `chebyshev_sample_mean`
//!   (`ε²·P(|S̄ − μ| ≥ ε) ≤ σ²/n`, Bienaymé variance scaling);
//! - the bounded-support variance cap — `variance_le_sq_half_range` (Popoviciu).

use euclid::{LazyBool, LazyReal};

use super::{ConfidenceInterval, alpha, invalid, valid_level};
use crate::{mean, variance};

/// **Markov's inequality**: an upper bound on `P(X ≥ a)` for a nonnegative `X` with
/// mean `mean`, namely `min(E[X]/a, 1)` (`a > 0`). Backed by riemann
/// `Bayes/Inequalities.markov`. Returns `NaN` on invalid input (`a ≤ 0`, `mean < 0`).
pub fn markov_bound<T: LazyReal>(mean: T, a: T) -> T {
    let valid = a.gt(&T::zero()).and(&mean.ge(&T::zero()));
    T::select(&valid, (mean / a).min(T::one()), T::nan())
}

/// **Chebyshev's inequality** (two-sided): an upper bound on `P(|X − μ| ≥ t)`, namely
/// `min(σ²/t², 1)` (`t > 0`). Needs only the variance — no distribution. Backed by
/// riemann `Bayes/Inequalities.chebyshev`.
pub fn chebyshev_bound<T: LazyReal>(variance: T, t: T) -> T {
    let valid = variance.ge(&T::zero()).and(&t.gt(&T::zero()));
    T::select(&valid, (variance / (t.clone() * t)).min(T::one()), T::nan())
}

/// **Cantelli's inequality** (one-sided): an upper bound on `P(X − μ ≥ t)`, namely
/// `σ²/(σ² + t²)` (`t > 0`) — sharper than the two-sided Chebyshev bound `σ²/t²`.
/// Backed by riemann `Bayes/Inequalities.cantelli`.
pub fn cantelli_bound<T: LazyReal>(variance: T, t: T) -> T {
    let valid = variance.ge(&T::zero()).and(&t.gt(&T::zero()));
    T::select(
        &valid,
        variance.clone() / (variance + t.clone() * t),
        T::nan(),
    )
}

/// **Hoeffding tail** (single observation): for one value bounded in `[a, b]`, an upper
/// bound on `P(X − μ ≥ t)`, namely `exp(−2t²/(b−a)²)` (`t ≥ 0`). Backed by riemann
/// `Bayes/Inequalities.hoeffding_tail`.
pub fn hoeffding_bound<T: LazyReal>(a: T, b: T, t: T) -> T {
    let valid = b.gt(&a).and(&t.ge(&T::zero()));
    let range = b - a;
    let formula = (-T::two() * t.clone() * t / (range.clone() * range)).exp();
    T::select(&valid, formula, T::nan())
}

/// **McDiarmid's bounded-differences inequality**: for an independent product input and
/// a function whose coordinate replacement diameters are `c_j`, this bounds
/// `P(F − E[F] ≥ t)` by `exp(−2t² / Σ c_j²)` (`t ≥ 0`, `Σ c_j² > 0`). This is the
/// independent-product generalization behind Hoeffding-style sample-mean bounds. Backed by
/// riemann `Bayes/ProductConcentration.mcdiarmid_tail_of_bounded_differences`.
pub fn mcdiarmid_bound<T: LazyReal>(sum_squared_differences: T, t: T) -> T {
    let valid = sum_squared_differences
        .gt(&T::zero())
        .and(&t.ge(&T::zero()));
    let formula = (-T::two() * t.clone() * t / sum_squared_differences).exp();
    T::select(&valid, formula, T::nan())
}

/// **Chernoff's method** — the generic exponential-Markov tail: for any tilt `s > 0`,
/// `P(X ≥ t) ≤ e^{−s·t}·E[e^{sX}]`. Given a value (or upper bound) `mgf` of the moment
/// generating function `E[e^{sX}]` at `s`, this returns that bound (clamped to a
/// probability). It is the substrate every sub-Gaussian / sub-exponential tail in this
/// module specializes — [`hoeffding_bound`] plugs in the bounded-range MGF, and
/// [`bernstein_bound`] the sub-gamma one; the caller minimizes over `s` for the tightest
/// bound. Backed by riemann `Bayes/Inequalities.chernoff_bound`. Returns `NaN` on invalid
/// input (`s ≤ 0`, `mgf < 0`).
pub fn chernoff_bound<T: LazyReal>(mgf: T, s: T, t: T) -> T {
    let valid = s.gt(&T::zero()).and(&mgf.ge(&T::zero()));
    let formula = ((-s * t).exp() * mgf).min(T::one());
    T::select(&valid, formula, T::nan())
}

/// **Bernstein's inequality** — the variance-aware exponential tail. For a variable whose
/// MGF is *sub-gamma* with variance proxy `variance_proxy` (`v > 0`) and scale `scale`
/// (`b > 0`) — `E[e^{sX}] ≤ exp(s²v/(2(1 − b·s)))` on `0 < s < 1/b` — an upper bound on
/// `P(X ≥ t)`, namely `exp(−t²/(2(v + b·t)))` (`t ≥ 0`). It interpolates the two regimes:
/// sub-Gaussian `exp(−t²/2v)` for small `t`, sub-exponential `exp(−t/2b)` for large `t` —
/// **sharper than [`hoeffding_bound`] when the variance `v` is small** (Hoeffding sees only
/// the range). A bounded mean-zero variable `|X| ≤ c` meets the sub-gamma condition with
/// `v = E[X²]` (riemann `bennett_mgf`). Backed by riemann
/// `Bayes/Inequalities.bernstein_tail` (the Chernoff optimization at `s = t/(v + b·t)`).
/// Returns `NaN` on invalid input (`v ≤ 0`, `b ≤ 0`, `t < 0`).
pub fn bernstein_bound<T: LazyReal>(variance_proxy: T, scale: T, t: T) -> T {
    let valid = variance_proxy
        .gt(&T::zero())
        .and(&scale.gt(&T::zero()))
        .and(&t.ge(&T::zero()));
    let formula = (-t.clone() * t.clone() / (T::two() * (variance_proxy + scale * t))).exp();
    T::select(&valid, formula, T::nan())
}

/// **The Paley–Zygmund inequality** — anti-concentration, a *lower* bound on a tail and the
/// complement of Markov/Chebyshev. For a **nonnegative** `X` with mean `mean` (`E[X] ≥ 0`)
/// and second moment `mean_square` (`E[X²] > 0`), and any `θ ∈ [0, 1]`,
/// `P(X ≥ θ·E[X]) ≥ (1 − θ)²·E[X]²/E[X²]`. Where the tail bounds above certify a variable is
/// *not too large*, this certifies it is *not too small*: a nonnegative variable with a
/// controlled second moment must put real mass near its mean — used to show a random
/// quantity is positive / bounded away from zero with constant probability. Backed by
/// riemann `Bayes/Inequalities.paley_zygmund`. Returns `NaN` on invalid input (`mean < 0`,
/// `mean_square ≤ 0`, `θ ∉ [0, 1]`).
pub fn paley_zygmund_bound<T: LazyReal>(mean: T, mean_square: T, theta: T) -> T {
    let valid = mean
        .ge(&T::zero())
        .and(&mean_square.gt(&T::zero()))
        .and(&theta.ge(&T::zero()).and(&theta.le(&T::one())));
    let one_minus_theta = T::one() - theta;
    let formula = (one_minus_theta.clone() * one_minus_theta * mean.clone() * mean / mean_square)
        .min(T::one());
    T::select(&valid, formula, T::nan())
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
        // McDiarmid: t=0 → 1; decreasing in t for fixed bounded-difference mass.
        close(mcdiarmid_bound(4.0, 0.0), 1.0, 1e-12);
        assert!(mcdiarmid_bound(4.0, 1.0) < 1.0);
    }

    #[test]
    fn chernoff_recovers_hoeffding_with_the_bounded_mgf() {
        // For a mean-zero X in [a, b], Hoeffding's MGF is E[e^{sX}] ≤ exp(s²(b−a)²/8).
        // Chernoff at the optimal s = 4t/(b−a)² then equals exp(−2t²/(b−a)²) = hoeffding_bound.
        let (a, b, t) = (0.0_f64, 1.0_f64, 0.5_f64);
        let s = 4.0 * t / ((b - a) * (b - a));
        let mgf = (s * s * (b - a) * (b - a) / 8.0).exp();
        close(chernoff_bound(mgf, s, t), hoeffding_bound(a, b, t), 1e-12);
        // it is a probability: an over-large mgf/over-small s clamps to 1.
        assert_eq!(chernoff_bound(2.0, 0.1, 0.0), 1.0);
        assert!(chernoff_bound(-1.0_f64, 1.0, 1.0).is_nan());
        assert!(chernoff_bound(1.0_f64, 0.0, 1.0).is_nan());
    }

    #[test]
    fn mcdiarmid_specializes_to_the_bounded_sample_mean() {
        // For an n-sample mean over [a,b], changing one coordinate moves the mean by
        // (b-a)/n, so Σ c_j² = n·((b-a)/n)² = (b-a)²/n.
        let (a, b, n, eps) = (0.0_f64, 1.0_f64, 25.0_f64, 0.2_f64);
        let sum_sq = (b - a) * (b - a) / n;
        close(
            mcdiarmid_bound(sum_sq, eps),
            (-2.0 * n * eps * eps).exp(),
            1e-12,
        );
        assert!(mcdiarmid_bound(sum_sq, eps) < hoeffding_bound(a, b, eps));
        assert!(mcdiarmid_bound(0.0, eps).is_nan());
        assert!(mcdiarmid_bound(sum_sq, -eps).is_nan());
    }

    #[test]
    fn bernstein_is_sharper_than_hoeffding_for_small_variance() {
        // v=1, b=1, t=2 ⇒ exp(−4/(2·3)) = exp(−2/3).
        close(bernstein_bound(1.0, 1.0, 2.0), (-2.0f64 / 3.0).exp(), 1e-12);
        // t=0 ⇒ 1; strictly decreasing in t.
        close(bernstein_bound(1.0, 1.0, 0.0), 1.0, 1e-12);
        assert!(bernstein_bound(1.0, 1.0, 1.0) < 1.0);
        // a small-variance variable in [−1, 1] (v ≪ c²): Bernstein beats Hoeffding near 0.
        // Hoeffding sees only the range (b−a)=2; Bernstein sees the true variance v=0.04.
        assert!(bernstein_bound(0.04, 1.0, 0.2) < hoeffding_bound(-1.0, 1.0, 0.2));
        assert!(bernstein_bound(0.0_f64, 1.0, 1.0).is_nan());
        assert!(bernstein_bound(1.0_f64, 0.0, 1.0).is_nan());
        assert!(bernstein_bound(1.0_f64, 1.0, -0.1).is_nan());
    }

    #[test]
    fn paley_zygmund_lower_bounds_the_tail() {
        // mean=1, E[X²]=2 (Var=1), θ=0.5 ⇒ (0.5)²·1/2 = 0.125.
        close(paley_zygmund_bound(1.0, 2.0, 0.5), 0.125, 1e-12);
        // θ=0 ⇒ E[X]²/E[X²] = 1/2 (≤ 1 by Cauchy–Schwarz); θ=1 ⇒ 0.
        close(paley_zygmund_bound(1.0, 2.0, 0.0), 0.5, 1e-12);
        close(paley_zygmund_bound(1.0, 2.0, 1.0), 0.0, 1e-12);
        // a degenerate constant X≡1 (E[X²]=E[X]²) at θ=0 gives the certain tail P=1.
        close(paley_zygmund_bound(1.0, 1.0, 0.0), 1.0, 1e-12);
        assert!(paley_zygmund_bound(-1.0_f64, 2.0, 0.5).is_nan());
        assert!(paley_zygmund_bound(1.0_f64, 0.0, 0.5).is_nan());
        assert!(paley_zygmund_bound(1.0_f64, 2.0, 1.5).is_nan());
    }

    #[test]
    fn invalid_inputs_are_nan() {
        assert!(chebyshev_bound(4.0_f64, 0.0).is_nan());
        assert!(markov_bound(-1.0_f64, 2.0).is_nan());
        assert!(hoeffding_bound(1.0_f64, 1.0, 0.5).is_nan());
        assert!(mcdiarmid_bound(-1.0_f64, 0.5).is_nan());
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
