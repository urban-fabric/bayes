//! **Proof-backed moment inequalities** — certifiers and bounds on raw moments, mirroring
//! riemann `RiemannLean/Bayes/Inequalities.lean` (3-axiom basis). The tail/concentration
//! bounds and the variance cap live in [`super::confidence::concentration`]; this module
//! collects the moment-*relationship* facts: Cauchy–Schwarz and the Jensen gap.

use euclid::Real;

/// **Cauchy–Schwarz moment consistency**: whether three (computed) moments satisfy the
/// inequality that *every* joint distribution obeys, `E[XY]² ≤ E[X²]·E[Y²]`. A `false`
/// means the moments are mutually inconsistent — no random pair `(X, Y)` can produce them
/// — so it flags a numerical error or a bad moment estimate. Backed by riemann
/// `Bayes/Inequalities.expectation_mul_sq_le`. A small relative tolerance absorbs
/// floating-point slack at the equality (perfectly-correlated) boundary.
pub fn cauchy_schwarz_moment_holds(e_xy: f64, e_x2: f64, e_y2: f64) -> bool {
    e_x2 >= 0.0 && e_y2 >= 0.0 && e_xy * e_xy <= e_x2 * e_y2 * (1.0 + 1e-9)
}

/// **Jensen-gap lower bound**: for an `m`-strongly-convex `φ` (`φ'' ≥ m`), the convex
/// transform gap is at least `(m/2)·Var(X)`: `E[φ(X)] − φ(E[X]) ≥ (m/2)·Var(X)`. So
/// plugging the mean into a convex `φ` *understates* `E[φ(X)]` by at least this much — a
/// guaranteed bias floor for plug-in / delta-method estimates. Backed by riemann
/// `Bayes/Inequalities.jensen_gap_lower` (`m = 0` recovers plain Jensen, gap `≥ 0`).
/// Returns `NaN` on a negative variance.
pub fn jensen_gap_lower_bound<T: Real>(variance: T, convexity_modulus: T) -> T {
    let valid = variance.ge(&T::zero());
    T::select(&valid, convexity_modulus / T::two() * variance, T::nan())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn cauchy_schwarz_accepts_consistent_and_rejects_impossible_moments() {
        // E[XY]²=4 ≤ E[X²]E[Y²]=6: consistent.
        assert!(cauchy_schwarz_moment_holds(2.0, 2.0, 3.0));
        // equality (perfect correlation): 4 ≤ 4.
        assert!(cauchy_schwarz_moment_holds(2.0, 2.0, 2.0));
        // E[XY]²=9 > E[X²]E[Y²]=1: impossible for any (X, Y).
        assert!(!cauchy_schwarz_moment_holds(3.0, 1.0, 1.0));
        // negative second moments are never valid.
        assert!(!cauchy_schwarz_moment_holds(0.0, -1.0, 1.0));
    }

    #[test]
    fn jensen_gap_scales_with_variance_and_modulus() {
        // strongly-convex φ (m=2), Var=4 ⇒ gap ≥ 4.
        close(jensen_gap_lower_bound(4.0, 2.0), 4.0, 1e-12);
        // m=0 recovers plain Jensen: only gap ≥ 0.
        close(jensen_gap_lower_bound(4.0, 0.0), 0.0, 1e-12);
        assert!(jensen_gap_lower_bound(-1.0_f64, 2.0).is_nan());
    }
}
