//! The shared Poisson-mixture machinery.
//!
//! All three non-central families are `Σ_j w_j · (central term)` with the same
//! weights `w_j = e^{-λ/2}(λ/2)^j / j!`. The only differences are the central
//! term and how the degrees of freedom shift with `j`, so the summation loop
//! lives here once.
//!
//! Because the loop walks outward from the mode one index at a time, a term that
//! knows how to step from its neighbour can replace a special-function call per
//! term with a single multiply — which is the difference between a χ² CDF that
//! costs one incomplete gamma and one that costs several thousand.
//! [`MixtureTerm`] is where a family declares that it can; the default
//! recomputes from scratch.

/// Terms below this fraction of the running total stop contributing.
const TOLERANCE: f64 = 1e-14;
/// A hard cap, so a pathological `λ` cannot spin forever. At `λ = 10^4` the
/// weights peak near `j = 5000`, which this still covers.
const MAX_TERMS: usize = 20_000;

/// The central term of a Poisson mixture, as a function of the index `j`.
///
/// A plain `FnMut(usize) -> f64` is one of these, which is all the F family
/// needs. A family whose central term satisfies a recurrence in `j` implements
/// [`MixtureTerm::up`] and [`MixtureTerm::down`] instead and pays one multiply
/// per term.
pub(super) trait MixtureTerm {
    /// The term at `j`, computed from scratch.
    ///
    /// This also **re-seeds** any recurrence state onto `j`: the summation loop
    /// calls it once before each outward walk, so an implementation may assume
    /// its state is reset here and nowhere else.
    fn at(&mut self, j: usize) -> f64;

    /// The term at `j + 1`, given its value at `j`.
    fn up(&mut self, _current: f64, j: usize) -> f64 {
        self.at(j + 1)
    }

    /// The term at `j - 1`, given its value at `j`.
    fn down(&mut self, _current: f64, j: usize) -> f64 {
        self.at(j - 1)
    }
}

impl<F: FnMut(usize) -> f64> MixtureTerm for F {
    fn at(&mut self, j: usize) -> f64 {
        self(j)
    }
}

/// `Σ_j w_j · term(j)` over the Poisson weights for non-centrality `lambda`,
/// **renormalized by the weight actually summed**.
///
/// The series is truncated where the remaining weight cannot move the answer, so
/// the weights retained total `1 − ε` rather than 1. Dividing by that total is
/// not cosmetic: in the far upper tail of a CDF every retained term is exactly
/// `1.0`, so the unnormalized sum returns `1 − ε` and *never reaches 1 however
/// large `x` gets*. A quantile found by inverting such a CDF then cannot be
/// bracketed for any `p` within `ε` of 1 — the bisection widens until its trip
/// count runs out and returns `2^64` instead of the support bound.
///
/// Renormalizing replaces the omitted terms with the average of the retained
/// ones, which is exact when the terms are constant (the tail case above) and
/// wrong by less than the `1e-14` truncation threshold everywhere else.
pub(super) fn poisson_mixture(lambda: f64, term: impl MixtureTerm) -> f64 {
    let (total, weight) = poisson_mixture_raw(lambda, term);
    if weight > 0.0 { total / weight } else { total }
}

/// The summation itself, returning the weight it accumulated alongside the
/// total so that [`poisson_mixture`] can renormalize and the tests can assert
/// the weights really do sum to one.
///
/// The weights are built by **recurrence** (`w_{j+1} = w_j · (λ/2) / (j+1)`)
/// rather than from `exp(ln w_j)` per term: the recurrence is one multiply
/// instead of a `lgamma` and an `exp`, and it cannot overflow on the way to a
/// weight that is itself representable.
///
/// Summation starts at the **mode** `j ≈ λ/2` and walks outward in both
/// directions. Starting at `j = 0` and walking up would spend thousands of
/// iterations on terms that underflow to zero before reaching the ones that
/// carry the mass — for `λ = 400` the first non-negligible weight is around
/// `j = 150`.
pub(super) fn poisson_mixture_raw(lambda: f64, mut term: impl MixtureTerm) -> (f64, f64) {
    let half = lambda / 2.0;
    if !(half.is_finite() && half >= 0.0) {
        return (f64::NAN, f64::NAN);
    }
    if half == 0.0 {
        // Degenerate mixture: all the weight is on j = 0, so the non-central
        // family *is* the central one. Returning the single term keeps that an
        // exact identity rather than a series that happens to converge to it.
        return (term.at(0), 1.0);
    }

    let mode = half.floor() as usize;
    // ln w_mode, computed once; everything else is a recurrence from here.
    let ln_w_mode = -half + mode as f64 * half.ln() - ln_factorial(mode);
    let w_mode = ln_w_mode.exp();

    // Walk up from the mode. `at` seeds the term's recurrence state onto `mode`.
    let mut t = term.at(mode);
    let mut total = w_mode * t;
    let mut weight_sum = w_mode;

    let mut w = w_mode;
    for j in (mode + 1)..MAX_TERMS {
        w *= half / j as f64;
        t = term.up(t, j - 1);
        if w <= 0.0 {
            break;
        }
        total += w * t;
        weight_sum += w;
        // Stop when the remaining weight cannot move the answer. The weights
        // sum to one, so `1 - weight_sum` bounds everything still unvisited.
        if w < TOLERANCE * weight_sum.max(1e-300) && (1.0 - weight_sum) < TOLERANCE {
            break;
        }
    }

    // And down toward zero, from a freshly seeded state: the walk above left the
    // recurrence at the far end of the series, not at the mode.
    if mode > 0 {
        let mut t = term.at(mode);
        let mut w = w_mode;
        for j in (0..mode).rev() {
            w *= (j + 1) as f64 / half;
            t = term.down(t, j + 1);
            if w <= 0.0 {
                break;
            }
            total += w * t;
            weight_sum += w;
            if w < TOLERANCE * weight_sum.max(1e-300) {
                break;
            }
        }
    }

    (total, weight_sum)
}

/// `ln(n!)` via the gamma function, for the one weight computed directly.
fn ln_factorial(n: usize) -> f64 {
    use euclid_numeric::Real;
    (n as f64 + 1.0).ln_gamma()
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    /// The weights are a probability distribution, so they must sum to 1
    /// whatever the non-centrality.
    ///
    /// This has to read the **raw** sum: [`poisson_mixture`] divides by the
    /// weight it accumulated, which would make `term(j) = 1` return 1 by
    /// construction and leave this test asserting `1/1 == 1`.
    #[test]
    fn the_poisson_weights_sum_to_one() {
        for lambda in [0.0, 1e-6, 0.5, 1.0, 10.0, 100.0, 1000.0, 1e4] {
            let (_, weight) = poisson_mixture_raw(lambda, |_: usize| 1.0);
            assert!(
                (weight - 1.0).abs() < 1e-10,
                "lambda = {lambda}: weights sum to {weight}"
            );
        }
    }

    /// And their mean is `λ/2`, which pins the recurrence in both directions:
    /// an error walking down from the mode would show up here and not above.
    #[test]
    fn the_poisson_weights_have_mean_half_lambda() {
        for lambda in [0.5, 2.0, 20.0, 500.0] {
            let mean = poisson_mixture(lambda, |j| j as f64);
            assert!(
                (mean - lambda / 2.0).abs() < 1e-8 * (lambda / 2.0).max(1.0),
                "lambda = {lambda}: weighted mean of j is {mean}, expected {}",
                lambda / 2.0
            );
        }
    }

    #[test]
    fn zero_non_centrality_takes_only_the_first_term() {
        let mut calls = Vec::new();
        let total = poisson_mixture(0.0, |j| {
            calls.push(j);
            if j == 0 { 7.0 } else { 0.0 }
        });
        assert_eq!(total, 7.0);
        assert_eq!(
            calls,
            vec![0],
            "a zero-lambda mixture should evaluate one term"
        );
    }
}
