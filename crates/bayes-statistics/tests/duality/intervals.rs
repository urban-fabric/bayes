//! A test and its interval, checked against each other at the endpoint.
//!
//! Each test here pairs one hypothesis test with the interval that reports the
//! same conclusion, and evaluates the test at the interval's own endpoints.
//! That is the one place where the two must agree to the last digit: the
//! p-value there is exactly `α`. Anywhere strictly inside or outside, the
//! agreement is only an inequality, and an inequality hides a quantile that is
//! slightly wrong.

use bayes_statistics::hypothesis::{one_sample_t, one_sample_z, welch_t};
use bayes_statistics::{mean_difference_welch_interval, mean_t_interval, mean_z_interval};
use bayes_utils::Failures;

/// Samples spanning sizes, spreads and locations, so a duality that holds by
/// luck at one configuration has to hold at twenty.
fn samples() -> Vec<(&'static str, Vec<f64>)> {
    vec![
        ("tiny", vec![1.0, 2.0, 4.0]),
        ("small spread", vec![10.0, 10.1, 9.9, 10.05, 9.95]),
        (
            "wide spread",
            vec![-50.0, 3.0, 17.0, -8.0, 42.0, 0.5, -21.0],
        ),
        (
            "offset far from zero",
            vec![1e6, 1e6 + 1.0, 1e6 - 2.0, 1e6 + 0.5, 1e6 + 3.0],
        ),
        (
            "twenty points",
            (0..20).map(|i| f64::from(i) * 0.7 - 3.0).collect(),
        ),
        ("two points", vec![4.0, 6.0]),
    ]
}

/// A one-sample `t` test rejects at `α` exactly when the `1 − α` `t` interval
/// excludes the null.
///
/// Checked by scanning the null value across and just inside each interval
/// endpoint. The endpoint itself is the interesting place: there the p-value
/// must be `α` to within rounding, so a factor of two on the wrong side of the
/// two-sided correction, or a `z` where a `t` belongs, separates the two
/// functions immediately.
#[test]
fn the_one_sample_t_test_and_its_interval_are_dual() {
    let mut f = Failures::default();
    for (name, data) in samples() {
        for level in [0.80, 0.90, 0.95, 0.99] {
            let alpha = 1.0 - level;
            let interval = mean_t_interval(&data, level);
            if !interval.lower.is_finite() || !interval.upper.is_finite() {
                continue;
            }
            // At an endpoint the p-value is exactly alpha.
            for endpoint in [interval.lower, interval.upper] {
                let p = one_sample_t(&data, endpoint).p_value;
                f.check((p - alpha).abs() < 1e-7 * alpha.max(1e-3), || {
                    format!(
                        "{name} at level {level}: the null at the interval endpoint \
                         {endpoint:.12} has p = {p:.12}, expected alpha = {alpha}"
                    )
                });
            }
            // Inside the interval the test does not reject; outside it does.
            let width = interval.upper - interval.lower;
            for fraction in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9] {
                let inside = interval.lower + width * fraction + width * 1e-6;
                let p = one_sample_t(&data, inside).p_value;
                f.check(p > alpha, || {
                    format!(
                        "{name} at level {level}: the null {inside:.12} is inside the interval \
                         but p = {p:.12} rejects at alpha = {alpha}"
                    )
                });
            }
            for outside in [
                interval.lower - width * 0.01,
                interval.lower - width,
                interval.upper + width * 0.01,
                interval.upper + width,
            ] {
                let p = one_sample_t(&data, outside).p_value;
                f.check(p < alpha, || {
                    format!(
                        "{name} at level {level}: the null {outside:.12} is outside the interval \
                         but p = {p:.12} does not reject at alpha = {alpha}"
                    )
                });
            }
        }
    }
    f.finish("one-sample t duality");
}

/// The same duality for the `z` test and the `z` interval, at a known `sigma`.
///
/// Worth checking separately because the two differ only in which reference
/// distribution they invert, and a `t`/`z` mix-up is invisible in a single
/// function — it produces a slightly wrong number, not a wrong shape.
#[test]
fn the_one_sample_z_test_and_its_interval_are_dual() {
    let mut f = Failures::default();
    for (name, data) in samples() {
        for sigma in [0.5_f64, 2.0, 25.0] {
            for level in [0.90, 0.95, 0.99] {
                let alpha = 1.0 - level;
                let interval = mean_z_interval(&data, sigma, level);
                if !interval.lower.is_finite() {
                    continue;
                }
                for endpoint in [interval.lower, interval.upper] {
                    let p = one_sample_z(&data, endpoint, sigma).p_value;
                    f.check((p - alpha).abs() < 1e-7 * alpha.max(1e-3), || {
                        format!(
                            "{name}, sigma {sigma}, level {level}: endpoint {endpoint:.12} has \
                             p = {p:.12}, expected {alpha}"
                        )
                    });
                }
            }
        }
    }
    f.finish("one-sample z duality");
}

/// The Welch difference-of-means interval is dual to the Welch test.
///
/// This is the strictest of the three: Welch's degrees of freedom are a
/// non-integer function of both samples' variances and sizes, so the interval
/// and the test must compute the same Satterthwaite value. A difference there
/// is a few percent in the p-value and invisible against any single reference.
#[test]
fn the_welch_test_and_its_difference_interval_are_dual() {
    let mut f = Failures::default();
    let pairs: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        (
            "equal size, unequal spread",
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![2.0, 6.0, -3.0, 11.0, 0.0],
        ),
        (
            "unequal size",
            vec![10.0, 11.0, 9.0],
            (0..15).map(|i| f64::from(i) * 0.4 + 8.0).collect(),
        ),
        (
            "far from zero",
            vec![1e5, 1e5 + 2.0, 1e5 - 1.0, 1e5 + 4.0],
            vec![1e5 + 10.0, 1e5 + 7.0, 1e5 + 13.0, 1e5 + 9.0],
        ),
    ];
    for (name, a, b) in pairs {
        for level in [0.90, 0.95, 0.99] {
            let alpha = 1.0 - level;
            let interval = mean_difference_welch_interval(&a, &b, level);
            if !interval.lower.is_finite() {
                continue;
            }
            // The interval is centred on the observed difference, where the
            // test's p-value is its largest.
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            let observed = mean(&a) - mean(&b);
            let centre = 0.5 * (interval.lower + interval.upper);
            f.check(
                (centre - observed).abs() < 1e-9 * observed.abs().max(1.0),
                || {
                    format!(
                        "{name}: interval centre {centre:.12} vs observed difference {observed:.12}"
                    )
                },
            );
            // Shifting both samples apart by exactly the upper endpoint makes
            // the test's p-value equal alpha.
            let shifted: Vec<f64> = b.iter().map(|x| x + interval.upper).collect();
            let p = welch_t(&a, &shifted).p_value;
            f.check((p - alpha).abs() < 1e-6 * alpha.max(1e-3), || {
                format!(
                    "{name} at level {level}: shifting by the upper endpoint gives p = {p:.12}, \
                     expected alpha = {alpha}"
                )
            });
        }
    }
    f.finish("Welch duality");
}
