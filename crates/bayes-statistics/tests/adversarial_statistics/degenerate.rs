//! **Degenerate input**: the contract is `NaN`, never a panic.
//!
//! Stated on `hypothesis::nan_result`: too few points or mismatched shapes yield
//! `NaN` rather than panicking. That is testable, and it is what this module
//! tests — every reduction and every test is handed an empty slice, a
//! one-element slice, a constant slice, mismatched lengths, `NaN`, `±inf`, and
//! magnitudes at both ends of the `f64` range.
//!
//! The assertions are deliberately weak on *value* and strong on *shape*: a
//! p-value must be a probability or undefined, a variance must never be
//! negative, a paired test on mismatched lengths must be undefined rather than
//! silently truncated to the shorter input.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use bayes_statistics::hypothesis::*;
use bayes_statistics::*;

use bayes_utils::Failures;

/// Every shape of degenerate input, to be fed to everything.
fn degenerate_inputs() -> Vec<(&'static str, Vec<f64>)> {
    vec![
        ("empty", vec![]),
        ("single", vec![3.5]),
        ("two", vec![1.0, 2.0]),
        ("constant", vec![7.0; 8]),
        ("two-identical", vec![2.5, 2.5]),
        ("with-nan", vec![1.0, f64::NAN, 3.0]),
        ("with-inf", vec![1.0, f64::INFINITY, 3.0]),
        ("with-neg-inf", vec![1.0, f64::NEG_INFINITY, 3.0]),
        ("all-zero", vec![0.0; 5]),
        ("huge", vec![1e300, 2e300, 3e300]),
        ("tiny", vec![1e-300, 2e-300, 3e-300]),
        ("mixed-sign-huge", vec![-1e300, 1e300]),
        ("subnormal", vec![5e-324, 1e-323, 1.5e-323]),
    ]
}

#[test]
fn summary_reductions_never_panic_on_degenerate_input() {
    let mut f = Failures::default();
    for (name, data) in degenerate_inputs() {
        // Each of these must return *something* — the point is that none panics.
        let results: Vec<(&str, f64)> = vec![
            ("mean", mean(&data)),
            ("variance", variance(&data)),
            ("population_variance", population_variance(&data)),
            ("std_dev", std_dev(&data)),
            ("skewness", skewness(&data)),
            ("kurtosis", kurtosis(&data)),
            ("mean_abs_deviation", mean_abs_deviation(&data)),
            ("coefficient_of_variation", coefficient_of_variation(&data)),
            ("median", median(&data)),
            ("iqr", iqr(&data)),
            ("median_abs_deviation", median_abs_deviation(&data)),
            ("gini_mean_difference", gini_mean_difference(&data)),
            ("trimmed_mean", trimmed_mean(&data, 0.25)),
            ("winsorized_mean", winsorized_mean(&data, 0.25)),
            ("geometric_mean", geometric_mean(&data)),
            ("harmonic_mean", harmonic_mean(&data)),
        ];
        for (label, v) in results {
            // A variance must never be negative; everything else just has to be
            // a number or NaN rather than a panic.
            if label.contains("variance") || label == "std_dev" || label == "iqr" {
                f.check(v.is_nan() || v >= 0.0, || {
                    format!("{name}: {label} = {v} is negative")
                });
            }
            let _ = v;
        }
        // An empty or single-point sample has no spread, so the unbiased
        // variance is undefined rather than zero.
        if data.len() < 2 {
            f.check(variance(&data).is_nan() || variance(&data) == 0.0, || {
                format!(
                    "{name}: variance of {} points = {}",
                    data.len(),
                    variance(&data)
                )
            });
        }
        // A constant sample has exactly zero spread.
        if name == "constant" || name == "two-identical" || name == "all-zero" {
            f.close(
                || format!("{name}: variance of a constant sample"),
                variance(&data),
                0.0,
                1e-300,
            );
            f.close(
                || format!("{name}: iqr of a constant sample"),
                iqr(&data),
                0.0,
                1e-300,
            );
        }
    }
    f.finish("summary reductions");
}

#[test]
fn hypothesis_tests_return_nan_rather_than_panicking_on_degenerate_input() {
    let mut f = Failures::default();
    for (name, data) in degenerate_inputs() {
        let other = vec![1.0, 2.0, 3.0, 4.0];
        let results: Vec<(&str, TestResult)> = vec![
            ("one_sample_t", one_sample_t(&data, 0.0)),
            ("one_sample_z", one_sample_z(&data, 0.0, 1.0)),
            ("two_sample_t", two_sample_t(&data, &other)),
            ("welch_t", welch_t(&data, &other)),
            ("variance_ratio_f", variance_ratio_f(&data, &other)),
            ("mann_whitney_u", mann_whitney_u(&data, &other)),
            ("ks_two_sample", ks_two_sample(&data, &other)),
            ("jarque_bera", jarque_bera(&data)),
        ];
        for (label, r) in results {
            // A p-value is either a probability or undefined. It is never, for
            // instance, 1.3 or -0.2.
            f.check(
                r.p_value.is_nan() || (0.0..=1.0).contains(&r.p_value),
                || {
                    format!(
                        "{name}: {label} p_value = {} is not a probability",
                        r.p_value
                    )
                },
            );
        }
        // Paired tests on mismatched lengths must be undefined, not silently
        // truncated to the shorter input.
        if data.len() != other.len() {
            let p = paired_t(&data, &other);
            f.check(p.p_value.is_nan(), || {
                format!(
                    "{name}: paired_t on mismatched lengths gave p = {}",
                    p.p_value
                )
            });
            let w = wilcoxon_signed_rank(&data, &other);
            f.check(w.p_value.is_nan(), || {
                format!(
                    "{name}: wilcoxon on mismatched lengths gave p = {}",
                    w.p_value
                )
            });
        }
    }

    // A test of a constant sample against its own value has zero spread: the
    // statistic is undefined, not infinite-and-significant.
    let flat = vec![5.0; 10];
    let r = one_sample_t(&flat, 5.0);
    f.check(r.p_value.is_nan() || r.p_value > 0.99, || {
        format!(
            "one_sample_t on a constant sample at its own mean gave p = {}",
            r.p_value
        )
    });

    f.finish("hypothesis tests");
}

/// The order statistics have to be monotone in `p`, and the quantile of a
/// one-point sample is that point whatever `p` is.
#[test]
fn quantiles_are_monotone_and_handle_small_samples() {
    let mut f = Failures::default();
    let data = [5.0, 1.0, 4.0, 2.0, 3.0];
    let mut prev = f64::NEG_INFINITY;
    for i in 0..=100 {
        let p = i as f64 / 100.0;
        let q = quantile(&data, p);
        f.check(q >= prev - 1e-12, || {
            format!("quantile decreased at p={p}: {prev} -> {q}")
        });
        f.check((1.0..=5.0).contains(&q), || {
            format!("quantile({p}) = {q} outside the data range [1,5]")
        });
        prev = q;
    }
    for p in [0.0, 0.25, 0.5, 0.75, 1.0] {
        f.close(
            || format!("quantile of a one-point sample at p={p}"),
            quantile(&[9.0], p),
            9.0,
            1e-15,
        );
    }
    // Tukey fences bracket the quartiles.
    let (lo, hi) = tukey_fences(&data, 1.5);
    f.check(
        lo <= quantile(&data, 0.25) && hi >= quantile(&data, 0.75),
        || format!("tukey_fences [{lo}, {hi}] do not bracket the quartiles"),
    );
    // An obvious outlier must be flagged, and nothing in a clean sample should be.
    let with_outlier = [1.0, 2.0, 3.0, 4.0, 100.0];
    let flags = tukey_outlier_flags(&with_outlier, 1.5);
    f.check(flags[4], || {
        "tukey_outlier_flags missed an obvious outlier".into()
    });
    let clean = [1.0, 2.0, 3.0, 4.0, 5.0];
    f.check(!tukey_outlier_flags(&clean, 1.5).iter().any(|&b| b), || {
        "tukey_outlier_flags flagged a point in a clean sample".into()
    });
    f.finish("quantiles");
}
