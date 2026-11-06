//! Statistics against each other, where one is algebraically determined by
//! others.
//!
//! Welford's one-pass accumulator against the two-pass batch computation;
//! regression slope against `cov/var`; `r²` against the squared correlation;
//! Hedges' *g* against Cohen's *d* times its small-sample correction; a variance
//! decomposition that must actually decompose. These catch the class of defect
//! where two routes to the same number have drifted apart.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use bayes_distribution_ext::{ContinuousCdf, *};
use bayes_statistics::hypothesis::*;
use bayes_statistics::*;

use bayes_utils::Failures;

const SAMPLE: [f64; 12] = [4.1, 5.2, 3.8, 6.0, 4.7, 5.5, 4.2, 5.9, 3.3, 6.4, 4.9, 5.1];
const OTHER: [f64; 10] = [6.2, 7.1, 5.8, 6.9, 7.4, 6.1, 5.5, 7.8, 6.6, 7.0];

/// Welford's one-pass update must agree with the two-pass batch computation.
#[test]
fn streaming_accumulators_agree_with_batch_statistics() {
    let mut f = Failures::default();

    let mut rs = RunningStats::new();
    for &x in &SAMPLE {
        rs.push(x);
    }
    f.close(
        || "RunningStats mean vs batch mean".into(),
        rs.mean(),
        mean(&SAMPLE),
        1e-13,
    );
    f.close(
        || "RunningStats variance vs batch variance".into(),
        rs.variance(),
        variance(&SAMPLE),
        1e-11,
    );
    f.close(
        || "RunningStats population_variance vs batch".into(),
        rs.population_variance(),
        population_variance(&SAMPLE),
        1e-11,
    );
    f.close(
        || "RunningStats std_dev vs batch".into(),
        rs.std_dev(),
        std_dev(&SAMPLE),
        1e-11,
    );
    f.close(
        || "RunningStats skewness vs batch".into(),
        rs.skewness(),
        skewness(&SAMPLE),
        1e-9,
    );
    f.close(
        || "RunningStats kurtosis vs batch".into(),
        rs.kurtosis(),
        kurtosis(&SAMPLE),
        1e-9,
    );
    f.check(rs.count() == SAMPLE.len() as u64, || {
        "RunningStats count".into()
    });

    // Merging two halves must equal accumulating the whole.
    let (left, right) = SAMPLE.split_at(5);
    let mut a = RunningStats::new();
    for &x in left {
        a.push(x);
    }
    let mut b = RunningStats::new();
    for &x in right {
        b.push(x);
    }
    a.merge(&b);
    f.close(
        || "merged RunningStats mean vs batch".into(),
        a.mean(),
        mean(&SAMPLE),
        1e-13,
    );
    f.close(
        || "merged RunningStats variance vs batch".into(),
        a.variance(),
        variance(&SAMPLE),
        1e-11,
    );

    // A t-digest is approximate by design, but must bracket the exact quantiles.
    let mut td = TDigest::new();
    let mut many: Vec<f64> = Vec::new();
    for i in 0..5_000 {
        let x = (i as f64) * 0.001;
        td.push(x);
        many.push(x);
    }
    for p in [0.1, 0.25, 0.5, 0.75, 0.9] {
        let approx = td.quantile(p);
        let exact = quantile(&many, p);
        f.check((approx - exact).abs() < 0.05, || {
            format!("TDigest quantile({p}) = {approx} vs exact {exact}")
        });
    }

    f.finish("streaming vs batch");
}

/// Correlation, covariance, and regression are three views of the same fit.
#[test]
fn regression_correlation_and_covariance_are_mutually_consistent() {
    let mut f = Failures::default();
    let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y = [2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1];

    let r = correlation(&x, &y);
    f.close(
        || "correlation vs cov/(sx·sy)".into(),
        r,
        covariance(&x, &y) / (std_dev(&x) * std_dev(&y)),
        1e-12,
    );

    let fit = LinearRegression::fit(&x, &y).unwrap();
    // Slope is cov/var(x); intercept passes through the means.
    f.close(
        || "regression slope vs cov/var(x)".into(),
        fit.slope,
        covariance(&x, &y) / variance(&x),
        1e-11,
    );
    f.close(
        || "regression intercept vs ȳ − slope·x̄".into(),
        fit.intercept,
        mean(&y) - fit.slope * mean(&x),
        1e-11,
    );
    // For a simple regression r² is exactly the squared correlation.
    f.close(
        || "regression r² vs correlation²".into(),
        fit.r_squared,
        r * r,
        1e-11,
    );
    // And the residual sum of squares is what it says it is.
    let rss: f64 = x
        .iter()
        .zip(&y)
        .map(|(&xi, &yi)| (yi - (fit.intercept + fit.slope * xi)).powi(2))
        .sum();
    f.close(
        || "regression rss vs Σ residual²".into(),
        fit.rss,
        rss,
        1e-10,
    );
    // `rmse` is documented as √(RSS/n) while `mse` is RSS/(n−2), so these are
    // deliberately *not* each other's square root — the field that is √mse is
    // `residual_std_error`. Worth knowing: a reader who assumes `rmse == √mse`
    // is off by √(n/(n−2)).
    let n_f = x.len() as f64;
    f.close(
        || "regression rmse vs √(RSS/n)".into(),
        fit.rmse,
        (fit.rss / n_f).sqrt(),
        1e-12,
    );
    f.close(
        || "regression mse vs RSS/(n−2)".into(),
        fit.mse,
        fit.rss / (n_f - 2.0),
        1e-12,
    );
    f.close(
        || "regression residual_std_error vs √mse".into(),
        fit.residual_std_error,
        fit.mse.sqrt(),
        1e-12,
    );

    // Pearson's correlation test is the same correlation looked up in a t.
    let ct = pearson_correlation_test(&x, &y);
    let n = x.len() as f64;
    let t_stat = r * ((n - 2.0) / (1.0 - r * r)).sqrt();
    f.close(
        || "pearson_correlation_test statistic".into(),
        ct.statistic,
        t_stat,
        1e-9,
    );
    let t = StudentT::new(n - 2.0).unwrap();
    f.close(
        || "pearson_correlation_test p vs 2·StudentT(n−2).sf".into(),
        ct.p_value,
        2.0 * t.sf(ct.statistic.abs()),
        1e-9,
    );

    f.finish("regression/correlation");
}

/// A variance decomposition must actually decompose: total = within + between.
#[test]
fn the_variance_decomposition_adds_up_and_matches_the_anova_f() {
    let mut f = Failures::default();
    let g1 = [4.0, 5.0, 6.0, 5.5];
    let g2 = [7.0, 8.0, 7.5, 9.0];
    let g3 = [2.0, 3.5, 3.0, 2.5];
    let groups: Vec<&[f64]> = vec![&g1, &g2, &g3];

    let flat: Vec<f64> = groups.iter().flat_map(|g| g.iter().copied()).collect();
    let labels: Vec<usize> = groups
        .iter()
        .enumerate()
        .flat_map(|(i, g)| std::iter::repeat_n(i, g.len()))
        .collect();
    let vd = variance_decomposition(&flat, &labels).unwrap();

    f.close(
        || "total_ss = within_ss + between_ss".into(),
        vd.total_ss,
        vd.within_ss + vd.between_ss,
        1e-10,
    );
    f.close(
        || "grand_mean vs mean of the pooled data".into(),
        vd.grand_mean,
        mean(&flat),
        1e-12,
    );
    f.check(vd.groups == 3 && vd.count == 12, || {
        "decomposition counts".into()
    });

    // The one-way ANOVA F is (between/(k−1)) / (within/(n−k)).
    let anova = one_way_anova(&groups);
    let (k, n) = (3.0, 12.0);
    let expected_f = (vd.between_ss / (k - 1.0)) / (vd.within_ss / (n - k));
    f.close(
        || "one_way_anova F vs decomposition ratio".into(),
        anova.statistic,
        expected_f,
        1e-9,
    );
    let fd = FisherSnedecor::new(k - 1.0, n - k).unwrap();
    f.close(
        || "one_way_anova p vs F(k−1, n−k).sf".into(),
        anova.p_value,
        fd.sf(anova.statistic),
        1e-9,
    );

    // eta² is the between-group share of the total.
    f.close(
        || "eta_squared vs between/total".into(),
        eta_squared(&groups),
        vd.between_ss / vd.total_ss,
        1e-10,
    );

    f.finish("variance decomposition");
}

/// Effect sizes are algebraically linked to each other and to the summaries.
#[test]
fn effect_sizes_agree_with_their_definitions() {
    let mut f = Failures::default();
    let d = cohen_d(&SAMPLE, &OTHER);
    f.close(
        || "cohen_d vs (x̄₁−x̄₂)/s_pooled".into(),
        d,
        (mean(&SAMPLE) - mean(&OTHER)) / pooled_std_dev(&SAMPLE, &OTHER),
        1e-11,
    );

    // Hedges' g applies the small-sample correction factor to Cohen's d.
    let n = (SAMPLE.len() + OTHER.len()) as f64;
    let correction = 1.0 - 3.0 / (4.0 * n - 9.0);
    f.close(
        || "hedges_g vs cohen_d·(1 − 3/(4N−9))".into(),
        hedges_g(&SAMPLE, &OTHER),
        d * correction,
        1e-9,
    );

    // Glass's delta standardizes by the control group's sd alone.
    f.close(
        || "glass_delta vs (x̄₁−x̄₂)/s₂".into(),
        glass_delta(&SAMPLE, &OTHER),
        (mean(&SAMPLE) - mean(&OTHER)) / std_dev(&OTHER),
        1e-11,
    );

    // Risk and odds measures on a 2x2 table.
    let (a, b, c, dd) = (10_u64, 90_u64, 20_u64, 80_u64);
    let (r1, r2) = (a as f64 / (a + b) as f64, c as f64 / (c + dd) as f64);
    f.close(
        || "risk_ratio vs r₁/r₂".into(),
        risk_ratio(a, b, c, dd),
        r1 / r2,
        1e-12,
    );
    f.close(
        || "risk_difference vs r₁−r₂".into(),
        risk_difference(a, b, c, dd),
        r1 - r2,
        1e-12,
    );
    f.close(
        || "odds_ratio vs ad/bc".into(),
        odds_ratio(a, b, c, dd),
        (a * dd) as f64 / (b * c) as f64,
        1e-12,
    );

    f.finish("effect sizes");
}
