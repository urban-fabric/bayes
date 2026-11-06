//! **Numerical stability**: cancellation and equivariance.
//!
//! A variance computed as `Σx²/n − x̄²` is algebraically right and numerically
//! useless: on `[10⁹+1, 10⁹+2, 10⁹+3]` the two terms agree to 18 digits and
//! their difference is noise. The true answer is exactly 1, so the test is
//! unambiguous — and this is the single most common defect in hand-rolled
//! statistics code.
//!
//! The equivariance checks are the other half: a mean must scale and shift with
//! the data, a variance must scale by the square and not shift at all, and a
//! standardized moment must be scale-free at *every* scale `f64` can hold. That
//! last one is how `skewness` was found returning `NaN` at `1e150`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use bayes_statistics::*;

use bayes_utils::Failures;

/// The variance of a tightly clustered sample sitting far from the origin.
///
/// This is the cancellation test. For `[k+1, k+2, k+3]` the sample variance is
/// exactly 1 and the population variance exactly 2/3, no matter how large `k` is,
/// right up to the point where `k+1` and `k+2` stop being distinct doubles.
#[test]
fn variance_survives_a_large_offset_without_cancelling() {
    let mut f = Failures::default();
    // At k = 1e15 the spacing of doubles is 0.125, so the integers are still
    // exact; beyond that the input itself stops being representable.
    for k in [0.0, 1.0, 1e3, 1e6, 1e9, 1e12, 1e15] {
        let data = [k + 1.0, k + 2.0, k + 3.0];
        f.close(
            || format!("variance of [k+1,k+2,k+3] at k={k:e}"),
            variance(&data),
            1.0,
            1e-9,
        );
        f.close(
            || format!("population_variance at k={k:e}"),
            population_variance(&data),
            2.0 / 3.0,
            1e-9,
        );
        f.close(|| format!("std_dev at k={k:e}"), std_dev(&data), 1.0, 1e-9);
        f.close(|| format!("mean at k={k:e}"), mean(&data), k + 2.0, 1e-15);

        // The streaming accumulator has to survive it too — Welford exists for
        // exactly this reason, so it would be the first thing to notice.
        let mut rs = RunningStats::new();
        for &x in &data {
            rs.push(x);
        }
        f.close(
            || format!("RunningStats variance at k={k:e}"),
            rs.variance(),
            1.0,
            1e-9,
        );

        // And covariance of the sample with itself is its variance.
        f.close(
            || format!("covariance(x,x) at k={k:e}"),
            covariance(&data, &data),
            1.0,
            1e-9,
        );
        // Correlation with itself is 1 regardless of offset.
        f.close(
            || format!("correlation(x,x) at k={k:e}"),
            correlation(&data, &data),
            1.0,
            1e-9,
        );
    }
    f.finish("offset variance");
}

/// A scale change must move variance by exactly its square, at any scale.
#[test]
fn reductions_are_equivariant_under_scaling_and_shifting() {
    let mut f = Failures::default();
    let base = [1.0, 2.5, 3.0, 4.75, 6.0];
    let v0 = variance(&base);
    let m0 = mean(&base);

    for scale in [1e-150, 1e-6, 0.5, 3.0, 1e6, 1e150] {
        let scaled: Vec<f64> = base.iter().map(|x| x * scale).collect();
        f.close(
            || format!("mean scales by {scale:e}"),
            mean(&scaled),
            m0 * scale,
            1e-12,
        );
        // variance scales by the square; at 1e150 that overflows, so only check
        // where the answer is representable.
        let expected = v0 * scale * scale;
        if expected.is_finite() && expected > 0.0 {
            f.close(
                || format!("variance scales by {scale:e}²"),
                variance(&scaled),
                expected,
                1e-11,
            );
        }
        // Correlation and skewness are scale-free.
        f.close(
            || format!("skewness is scale-free at {scale:e}"),
            skewness(&scaled),
            skewness(&base),
            1e-9,
        );
    }

    for shift in [-1e12, -5.0, 0.0, 5.0, 1e12] {
        let shifted: Vec<f64> = base.iter().map(|x| x + shift).collect();
        f.close(
            || format!("mean shifts by {shift:e}"),
            mean(&shifted),
            m0 + shift,
            1e-12,
        );

        // The *input* is what limits this, not the reduction. One ulp at 1e12 is
        // 1.2e-4, so `1e12 + 2.5` is only representable to within 6e-5 and the
        // shifted sample is no longer the shifted sample. The variance of what
        // was actually stored differs from `v0` by about `2·sd·ulp`, so that is
        // the bound to hold the library to — anything tighter is testing `f64`.
        let ulp = shift.abs().max(1.0) * f64::EPSILON;
        let bound = (2.0 * v0.sqrt() * ulp).max(1e-9 * v0);
        f.check((variance(&shifted) - v0).abs() <= bound, || {
            format!(
                "variance at shift {shift:e}: {} vs {v0} (diff {:.3e} > representability \
                 bound {bound:.3e})",
                variance(&shifted),
                (variance(&shifted) - v0).abs()
            )
        });
        f.check(
            (iqr(&shifted) - iqr(&base)).abs() <= 4.0 * ulp + 1e-9,
            || {
                format!(
                    "iqr at shift {shift:e}: {} vs {}",
                    iqr(&shifted),
                    iqr(&base)
                )
            },
        );
    }
    f.finish("equivariance");
}

// ---------------------------------------------------------------------------
// Degenerate inputs: the contract is NaN, never a panic
// ---------------------------------------------------------------------------
