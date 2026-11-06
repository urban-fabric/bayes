//! Weighted reductions against their unweighted forms.
//!
//! The defining property of a frequency weight is that an integer weight of `k`
//! is the same as repeating the observation `k` times. That makes the whole
//! weighted surface checkable against the unweighted one: equal weights must
//! reproduce the plain reduction exactly, integer weights must reproduce literal
//! repetition, and scaling every weight by a constant must change nothing.
//!
//! Note *which* normalization each carries. `weighted_covariance` is the
//! unbiased form, so at equal weights it lands on the **sample** variance, while
//! `weighted_population_variance` lands on the **population** one. Getting that
//! backwards is a silent `n/(n−1)` error, which is what these checks exist to
//! catch.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use bayes_statistics::*;
use bayes_utils::Failures;

const SAMPLE: [f64; 12] = [4.1, 5.2, 3.8, 6.0, 4.7, 5.5, 4.2, 5.9, 3.3, 6.4, 4.9, 5.1];

#[test]
fn weighted_statistics_reduce_to_unweighted_at_equal_weights() {
    let mut f = Failures::default();
    let w = vec![1.0; SAMPLE.len()];

    f.close(
        || "weighted_mean at equal weights vs mean".into(),
        weighted_mean(&SAMPLE, &w),
        mean(&SAMPLE),
        1e-13,
    );
    f.close(
        || "weighted_population_variance at equal weights".into(),
        weighted_population_variance(&SAMPLE, &w),
        population_variance(&SAMPLE),
        1e-11,
    );
    // The unbiased form, so this reduces to the *sample* variance.
    f.close(
        || "weighted_covariance(x,x) at equal weights vs sample variance".into(),
        weighted_covariance(&SAMPLE, &SAMPLE, &w),
        variance(&SAMPLE),
        1e-11,
    );
    f.close(
        || "weighted_correlation of x with itself".into(),
        weighted_correlation(&SAMPLE, &SAMPLE, &w),
        1.0,
        1e-12,
    );
    f.close(
        || "weighted_std_dev is the root of weighted_variance".into(),
        weighted_std_dev(&SAMPLE, &w),
        weighted_variance(&SAMPLE, &w).sqrt(),
        1e-13,
    );

    // Scaling every weight by the same factor cannot move a weighted mean.
    let scaled: Vec<f64> = w.iter().map(|x| x * 7.5).collect();
    f.close(
        || "weighted_mean is scale-invariant in the weights".into(),
        weighted_mean(&SAMPLE, &scaled),
        weighted_mean(&SAMPLE, &w),
        1e-13,
    );

    // `weighted_quantile` does **not** reduce to `quantile`, and that is
    // deliberate: the two document different conventions. `quantile`
    // interpolates linearly between order statistics; `weighted_quantile` takes
    // "the first ordered value whose cumulative weight reaches p·Sum(w)", which is
    // an order statistic with no interpolation. At equal weights that is
    // `sorted[ceil(p·n) - 1]`, and it differs from the interpolated value by up
    // to one gap between adjacent points.
    //
    // Worth knowing at a call site: swapping one for the other silently shifts
    // every quantile but the extremes.
    let mut sorted = SAMPLE.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in the sample"));
    let n = SAMPLE.len() as f64;
    for p in [0.1, 0.25, 0.5, 0.75, 0.9] {
        let rank = ((p * n).ceil() as usize).max(1) - 1;
        f.close(
            || format!("weighted_quantile({p}) at equal weights vs order statistic {rank}"),
            weighted_quantile(&SAMPLE, &w, p),
            sorted[rank],
            1e-12,
        );
        // And it always lands *on* a data point, which the interpolated form
        // does not.
        f.check(sorted.contains(&weighted_quantile(&SAMPLE, &w, p)), || {
            format!("weighted_quantile({p}) returned a value not in the sample")
        });
    }

    f.finish("weighted reduces to unweighted");
}

/// An integer weight of `k` must equal repeating the observation `k` times.
/// This is the definition of a frequency weight, so it is the load-bearing check.
#[test]
fn integer_weights_equal_literal_repetition() {
    let mut f = Failures::default();
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

    // Duplicating the whole sample at weight 1 must not move anything.
    let doubled: Vec<f64> = SAMPLE.iter().chain(SAMPLE.iter()).copied().collect();
    let ones = vec![1.0; doubled.len()];
    f.close(
        || "weighted_mean is invariant to duplicating the sample".into(),
        weighted_mean(&doubled, &ones),
        mean(&SAMPLE),
        1e-14,
    );
    f.close(
        || "weighted_population_variance is invariant to duplication".into(),
        weighted_population_variance(&doubled, &ones),
        population_variance(&SAMPLE),
        1e-12,
    );

    f.finish("integer weights vs repetition");
}

/// The effective sample size is the diagnostic that says whether a set of
/// weights is usable at all, so its two extremes are exact.
#[test]
fn effective_sample_size_spans_one_to_n() {
    let mut f = Failures::default();
    let n = SAMPLE.len();

    // Equal weights carry the most information possible: exactly `n`.
    f.close(
        || "effective_sample_size at equal weights = n".into(),
        effective_sample_size(&vec![1.0; n]),
        n as f64,
        1e-11,
    );
    // All the mass on one point carries the least: exactly 1.
    let mut spike = vec![0.0; n];
    spike[3] = 1.0;
    f.close(
        || "effective_sample_size at a spike = 1".into(),
        effective_sample_size(&spike),
        1.0,
        1e-11,
    );
    // And it is scale-free in the weights.
    f.close(
        || "effective_sample_size is scale-free".into(),
        effective_sample_size(&vec![0.25; n]),
        effective_sample_size(&vec![1.0; n]),
        1e-11,
    );

    f.finish("effective sample size");
}

/// Normalizing in the log domain must agree with the linear form wherever the
/// linear form does not underflow — and the log form is the only one that
/// survives where it does.
#[test]
fn log_domain_normalization_agrees_with_the_linear_form() {
    let mut f = Failures::default();

    let raw = [2.0, 5.0, 3.0];
    let norm = normalized_weights(&raw);
    f.close(
        || "normalized_weights sum to one".into(),
        norm.iter().sum::<f64>(),
        1.0,
        1e-14,
    );
    let log_norm = normalized_log_weights(&raw.map(f64::ln));
    for (i, (&a, &b)) in norm.iter().zip(log_norm.iter()).enumerate() {
        f.close(
            || format!("normalized_log_weights[{i}] vs normalized_weights"),
            b,
            a,
            1e-12,
        );
    }

    // The case the log domain exists for: importance weights spanning hundreds
    // of orders of magnitude. Exponentiating these first would underflow every
    // one but the largest to zero and lose the whole tail.
    let log_weights = [-900.0, -901.0, -903.0, -1200.0];
    let from_logs = normalized_log_weights(&log_weights);
    f.close(
        || "log-domain weights still sum to one at extreme magnitudes".into(),
        from_logs.iter().sum::<f64>(),
        1.0,
        1e-12,
    );
    // The ratio of the first two is exp(1), which the normalization preserves.
    f.check(
        (from_logs[0] / from_logs[1] - 1.0_f64.exp()).abs() < 1e-9,
        || {
            format!(
                "the ratio of adjacent log-weights was lost: {} vs {}",
                from_logs[0] / from_logs[1],
                1.0_f64.exp()
            )
        },
    );
    // And the far-smaller weight stays representable rather than flushed to zero.
    f.check(from_logs[3] > 0.0, || {
        "a weight 300 log-units down underflowed to zero".into()
    });

    f.finish("log-domain normalization");
}
