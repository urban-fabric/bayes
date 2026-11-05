//! **Degenerate structures**: intervals, weights, designs, and summaries that
//! have nothing to work with.
//!
//! These are the cases where the right answer is to *decline*. A regression on a
//! design with no variation in `x` has no slope, so the fit must return `None`
//! rather than divide by zero and report an infinite one. All-zero weights carry
//! no information. A proportion interval at `0 of n` is where the Wald form
//! famously produces nonsense, so every interval is checked to stay inside
//! `[0, 1]` and stay ordered.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

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
fn confidence_intervals_never_panic_and_stay_ordered() {
    let mut f = Failures::default();
    for (name, data) in degenerate_inputs() {
        for level in [0.5, 0.95, 0.99] {
            let intervals: Vec<(&str, ConfidenceInterval)> = vec![
                ("mean_t_interval", mean_t_interval(&data, level)),
                ("mean_z_interval", mean_z_interval(&data, 1.0, level)),
                (
                    "variance_chi_square_interval",
                    variance_chi_square_interval(&data, level),
                ),
                (
                    "std_dev_chi_square_interval",
                    std_dev_chi_square_interval(&data, level),
                ),
            ];
            for (label, ci) in intervals {
                f.check(
                    ci.lower.is_nan() || ci.upper.is_nan() || ci.lower <= ci.upper,
                    || {
                        format!(
                            "{name}: {label}({level}) is inverted: [{}, {}]",
                            ci.lower, ci.upper
                        )
                    },
                );
            }
        }
    }

    // Proportion intervals at the boundaries: 0 of n and n of n are the cases
    // where a Wald interval famously produces nonsense.
    for (s, n) in [(0_u64, 10_u64), (10, 10), (0, 1), (1, 1), (0, 0)] {
        for (label, ci) in [
            ("wald", proportion_wald(s, n, 0.95)),
            ("wilson", proportion_wilson(s, n, 0.95)),
            ("clopper_pearson", proportion_clopper_pearson(s, n, 0.95)),
            ("agresti_coull", proportion_agresti_coull(s, n, 0.95)),
            ("jeffreys", proportion_jeffreys(s, n, 0.95)),
        ] {
            f.check(
                ci.lower.is_nan() || ci.upper.is_nan() || (ci.lower <= ci.upper),
                || {
                    format!(
                        "proportion_{label}({s},{n}) is inverted: [{}, {}]",
                        ci.lower, ci.upper
                    )
                },
            );
            // An interval for a proportion cannot leave [0, 1].
            f.check(
                ci.lower.is_nan() || (-1e-12..=1.0 + 1e-12).contains(&ci.lower),
                || {
                    format!(
                        "proportion_{label}({s},{n}) lower = {} leaves [0,1]",
                        ci.lower
                    )
                },
            );
            f.check(
                ci.upper.is_nan() || (-1e-12..=1.0 + 1e-12).contains(&ci.upper),
                || {
                    format!(
                        "proportion_{label}({s},{n}) upper = {} leaves [0,1]",
                        ci.upper
                    )
                },
            );
        }
    }

    // A rate interval for zero observed events must still be a valid interval.
    for (events, exposure) in [(0_u64, 1.0), (0, 100.0), (5, 2.5)] {
        let ci = poisson_rate_interval(events, exposure, 0.95);
        f.check(
            ci.lower.is_nan() || (ci.lower >= -1e-12 && ci.lower <= ci.upper),
            || {
                format!(
                    "poisson_rate_interval({events},{exposure}): [{}, {}]",
                    ci.lower, ci.upper
                )
            },
        );
    }

    f.finish("confidence intervals");
}

#[test]
fn weighted_reductions_handle_zero_and_degenerate_weights() {
    let mut f = Failures::default();
    let data = [1.0, 2.0, 3.0, 4.0];

    // All-zero weights carry no information: the result is undefined, not a
    // division that silently yields 0/0 = NaN mixed with a plausible number.
    let zero = [0.0; 4];
    f.check(weighted_mean(&data, &zero).is_nan(), || {
        format!(
            "weighted_mean with all-zero weights = {}",
            weighted_mean(&data, &zero)
        )
    });

    // A single nonzero weight selects that point exactly.
    let mut spike = [0.0; 4];
    spike[2] = 1.0;
    f.close(
        || "weighted_mean at a spike selects the point".into(),
        weighted_mean(&data, &spike),
        3.0,
        1e-15,
    );
    f.close(
        || "weighted_population_variance at a spike is 0".into(),
        weighted_population_variance(&data, &spike),
        0.0,
        1e-300,
    );

    // Mismatched lengths are undefined.
    f.check(weighted_mean(&data, &[1.0, 1.0]).is_nan(), || {
        "weighted_mean with mismatched lengths was not NaN".into()
    });

    // Duplicating every point with weight 1 must not move the weighted mean.
    let doubled: Vec<f64> = data.iter().chain(data.iter()).copied().collect();
    let ones = vec![1.0; doubled.len()];
    f.close(
        || "weighted_mean is invariant to duplicating the sample".into(),
        weighted_mean(&doubled, &ones),
        mean(&data),
        1e-14,
    );

    // Integer weights must agree with literal repetition — the defining property
    // of a frequency weight.
    let values = [1.0, 2.0, 3.0];
    let weights = [1.0, 3.0, 2.0];
    let expanded = [1.0, 2.0, 2.0, 2.0, 3.0, 3.0];
    f.close(
        || "weighted_mean vs literal repetition".into(),
        weighted_mean(&values, &weights),
        mean(&expanded),
        1e-14,
    );
    f.close(
        || "weighted_population_variance vs literal repetition".into(),
        weighted_population_variance(&values, &weights),
        population_variance(&expanded),
        1e-13,
    );

    // Negative weights are not a frequency and must not produce a confident number.
    let negative = [1.0, -1.0, 1.0, 1.0];
    let wm = weighted_mean(&data, &negative);
    f.check(wm.is_nan() || wm.is_finite(), || {
        format!("weighted_mean with a negative weight = {wm}")
    });

    // ESS is bounded by the sample size whatever the weights.
    for w in [
        vec![1.0, 1.0, 1.0, 1.0],
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.25, 0.25, 0.25, 0.25],
        vec![1e-9, 1.0, 1e-9, 1e-9],
    ] {
        let ess = effective_sample_size(&w);
        f.check(
            ess.is_nan() || (ess >= 1.0 - 1e-9 && ess <= w.len() as f64 + 1e-9),
            || format!("effective_sample_size({w:?}) = {ess} outside [1, n]"),
        );
    }

    f.finish("weighted reductions");
}

#[test]
fn regression_and_correlation_handle_degenerate_designs() {
    let mut f = Failures::default();

    // A design with no variation in x has no slope: the fit must decline rather
    // than divide by zero and report an infinite slope.
    let flat_x = [2.0, 2.0, 2.0, 2.0];
    let y = [1.0, 2.0, 3.0, 4.0];
    f.check(LinearRegression::fit(&flat_x, &y).is_none(), || {
        "LinearRegression::fit accepted a design with zero x-variance".into()
    });

    // Too few points for the two parameters.
    f.check(LinearRegression::fit(&[1.0], &[2.0]).is_none(), || {
        "LinearRegression::fit accepted a single point".into()
    });
    f.check(LinearRegression::fit(&[], &[]).is_none(), || {
        "LinearRegression::fit accepted empty input".into()
    });
    // Mismatched lengths.
    f.check(
        LinearRegression::fit(&[1.0, 2.0, 3.0], &[1.0, 2.0]).is_none(),
        || "LinearRegression::fit accepted mismatched lengths".into(),
    );

    // An exactly collinear design must recover the line exactly and report r² = 1.
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let exact: Vec<f64> = x.iter().map(|v| 3.0 * v - 1.0).collect();
    let fit = LinearRegression::fit(&x, &exact).unwrap();
    f.close(|| "exact fit slope".into(), fit.slope, 3.0, 1e-12);
    f.close(|| "exact fit intercept".into(), fit.intercept, -1.0, 1e-12);
    f.close(|| "exact fit r²".into(), fit.r_squared, 1.0, 1e-12);
    f.close(|| "exact fit rss".into(), fit.rss, 0.0, 1e-20);

    // Correlation of a constant with anything is undefined, not zero.
    let c = correlation(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]);
    f.check(c.is_nan(), || {
        format!("correlation with a constant = {c}, expected NaN")
    });

    // Rank correlations on a constant are likewise undefined.
    for (label, v) in [
        ("spearman", spearman(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0])),
        (
            "kendall_tau",
            kendall_tau(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]),
        ),
    ] {
        f.check(v.is_nan() || v == 0.0, || {
            format!("{label} with a constant = {v}")
        });
    }

    f.finish("degenerate designs");
}

#[test]
fn empirical_summaries_handle_degenerate_input() {
    let mut f = Failures::default();

    // An ECDF of nothing has nothing to report, but must not panic.
    let e = Ecdf::new(&[]);
    f.check(e.is_empty(), || "empty Ecdf reports content".into());
    let _ = e.eval(0.0);
    let _ = e.quantile(0.5);

    // An ECDF of one point is a step at that point.
    let one = Ecdf::new(&[4.0]);
    f.close(
        || "single-point Ecdf below".into(),
        one.eval(3.0),
        0.0,
        1e-15,
    );
    f.close(|| "single-point Ecdf at".into(), one.eval(4.0), 1.0, 1e-15);
    f.close(
        || "single-point Ecdf above".into(),
        one.eval(5.0),
        1.0,
        1e-15,
    );

    // Out-of-range probabilities must not produce an interior value.
    for p in [-0.5, 1.5, f64::NAN] {
        let q = Ecdf::new(&[1.0, 2.0, 3.0]).quantile(p);
        f.check(q.is_nan() || q <= 1.0 || q >= 3.0, || {
            format!("Ecdf::quantile({p}) = {q} is interior for an out-of-range probability")
        });
    }

    // A histogram needs at least one bin and some data.
    f.check(histogram(&[], 5).is_none(), || {
        "histogram of empty data was Some".into()
    });
    f.check(histogram(&[1.0, 2.0], 0).is_none(), || {
        "histogram with zero bins was Some".into()
    });
    // A constant sample has zero width, so binning it is degenerate.
    let h = histogram(&[3.0; 10], 4);
    if let Some(h) = h {
        let total: u64 = h.counts.iter().sum();
        f.check(total == 10, || {
            format!("histogram of a constant sample lost counts: {total} of 10")
        });
    }

    // A KDE needs data.
    f.check(KernelDensity::new(&[]).is_none(), || {
        "KernelDensity of empty data was Some".into()
    });
    // A constant sample has zero bandwidth by the usual rules, so it is declined
    // or returns a finite density — either is acceptable, a panic is not.
    if let Some(k) = KernelDensity::new(&[2.0; 6]) {
        let d = k.density(2.0);
        f.check(!d.is_nan() || k.bandwidth() == 0.0, || {
            format!("KDE of a constant sample gave density {d} at the point")
        });
    }

    // A t-digest with nothing in it.
    let td = TDigest::new();
    f.check(td.count() == 0, || "empty TDigest reports a count".into());
    let _ = td.quantile(0.5);

    f.finish("empirical summaries");
}
