//! Each classical test against the distribution it looks its p-value up in.
//!
//! Almost every test in the library is "compute a statistic, then look it up in
//! a distribution". That makes the two independently checkable against each
//! other: `one_sample_t` must report exactly `2·StudentT(n−1).sf(|t|)`, and the
//! `t` it reports must be exactly `(x̄ − μ₀)/(s/√n)` as computed from the summary
//! functions. If all three agree, a single wrong constant would have to be wrong
//! in three places at once and in a mutually consistent way.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use bayes_distribution_ext::{ContinuousCdf, *};
use bayes_statistics::hypothesis::*;
use bayes_statistics::*;

use bayes_utils::Failures;

const SAMPLE: [f64; 12] = [4.1, 5.2, 3.8, 6.0, 4.7, 5.5, 4.2, 5.9, 3.3, 6.4, 4.9, 5.1];
const OTHER: [f64; 10] = [6.2, 7.1, 5.8, 6.9, 7.4, 6.1, 5.5, 7.8, 6.6, 7.0];

#[test]
fn the_one_sample_t_test_is_its_statistic_looked_up_in_a_student_t() {
    let mut f = Failures::default();
    let n = SAMPLE.len() as f64;
    for mu0 in [4.0, 5.0, 5.0909, 6.0] {
        let r = one_sample_t(&SAMPLE, mu0);
        // The statistic, from the summary functions rather than from the test.
        let expected_t = (mean(&SAMPLE) - mu0) / (std_dev(&SAMPLE) / n.sqrt());
        f.close(
            || format!("one_sample_t(μ₀={mu0}) statistic vs (x̄−μ₀)/(s/√n)"),
            r.statistic,
            expected_t,
            1e-12,
        );
        // The p-value, from the distribution rather than from the test.
        let t = StudentT::new(n - 1.0).unwrap();
        let two_sided = 2.0 * t.sf(r.statistic.abs());
        f.close(
            || {
                format!(
                    "one_sample_t(μ₀={mu0}) p vs 2·StudentT({}).sf(|t|)",
                    n - 1.0
                )
            },
            r.p_value,
            two_sided,
            1e-10,
        );
    }
    f.finish("one-sample t");
}

#[test]
fn the_one_sample_z_test_is_its_statistic_looked_up_in_a_normal() {
    let mut f = Failures::default();
    let n = SAMPLE.len() as f64;
    let sigma = 1.0;
    for mu0 in [4.5, 5.0, 5.5] {
        let r = one_sample_z(&SAMPLE, mu0, sigma);
        let expected_z = (mean(&SAMPLE) - mu0) / (sigma / n.sqrt());
        f.close(
            || format!("one_sample_z(μ₀={mu0}) statistic"),
            r.statistic,
            expected_z,
            1e-12,
        );
        let z = Normal::new(0.0, 1.0).unwrap();
        f.close(
            || format!("one_sample_z(μ₀={mu0}) p vs 2·Φ̄(|z|)"),
            r.p_value,
            2.0 * z.sf(r.statistic.abs()),
            1e-10,
        );
    }
    f.finish("one-sample z");
}

#[test]
fn welch_and_two_sample_t_agree_with_their_reference_distributions() {
    let mut f = Failures::default();

    let r = two_sample_t(&SAMPLE, &OTHER);
    let (n1, n2) = (SAMPLE.len() as f64, OTHER.len() as f64);
    let sp = pooled_std_dev(&SAMPLE, &OTHER);
    let expected = (mean(&SAMPLE) - mean(&OTHER)) / (sp * (1.0 / n1 + 1.0 / n2).sqrt());
    f.close(
        || "two_sample_t statistic vs pooled formula".into(),
        r.statistic,
        expected,
        1e-11,
    );
    let t = StudentT::new(n1 + n2 - 2.0).unwrap();
    f.close(
        || "two_sample_t p vs 2·StudentT(n1+n2-2).sf".into(),
        r.p_value,
        2.0 * t.sf(r.statistic.abs()),
        1e-10,
    );

    // Welch uses the Satterthwaite degrees of freedom; recompute them here.
    let w = welch_t(&SAMPLE, &OTHER);
    let (v1, v2) = (variance(&SAMPLE) / n1, variance(&OTHER) / n2);
    let expected_w = (mean(&SAMPLE) - mean(&OTHER)) / (v1 + v2).sqrt();
    f.close(
        || "welch_t statistic".into(),
        w.statistic,
        expected_w,
        1e-11,
    );
    let df = (v1 + v2).powi(2) / (v1 * v1 / (n1 - 1.0) + v2 * v2 / (n2 - 1.0));
    let tw = StudentT::new(df).unwrap();
    f.close(
        || "welch_t p vs 2·StudentT(Satterthwaite df).sf".into(),
        w.p_value,
        2.0 * tw.sf(w.statistic.abs()),
        1e-9,
    );

    f.finish("two-sample t / Welch");
}

#[test]
fn the_variance_ratio_test_is_an_f_distribution_lookup() {
    let mut f = Failures::default();
    let r = variance_ratio_f(&SAMPLE, &OTHER);
    f.close(
        || "variance_ratio_f statistic vs s₁²/s₂²".into(),
        r.statistic,
        variance(&SAMPLE) / variance(&OTHER),
        1e-12,
    );
    let (d1, d2) = (SAMPLE.len() as f64 - 1.0, OTHER.len() as f64 - 1.0);
    let fd = FisherSnedecor::new(d1, d2).unwrap();
    // Two-sided: twice the smaller tail.
    let tail = fd.sf(r.statistic).min(fd.cdf(r.statistic));
    f.close(
        || "variance_ratio_f p vs 2·min tail of F".into(),
        r.p_value,
        2.0 * tail,
        1e-9,
    );
    f.finish("variance ratio F");
}

#[test]
fn chi_squared_tests_are_chi_squared_lookups() {
    let mut f = Failures::default();

    let observed = [18.0, 22.0, 31.0, 29.0];
    let expected = [25.0, 25.0, 25.0, 25.0];
    let r = chi_squared_goodness_of_fit(&observed, &expected);
    // The statistic is Σ(O−E)²/E.
    let stat: f64 = observed
        .iter()
        .zip(&expected)
        .map(|(o, e)| (o - e).powi(2) / e)
        .sum();
    f.close(
        || "chi_squared_goodness_of_fit statistic".into(),
        r.statistic,
        stat,
        1e-12,
    );
    let chi = ChiSquared::new(observed.len() as f64 - 1.0).unwrap();
    f.close(
        || "chi_squared_goodness_of_fit p vs ChiSquared(k−1).sf".into(),
        r.p_value,
        chi.sf(stat),
        1e-10,
    );

    // Independence on a 2x3 table: df = (rows−1)(cols−1).
    let table = vec![vec![10.0, 20.0, 30.0], vec![15.0, 25.0, 20.0]];
    let ri = chi_squared_independence(&table);
    let chi_i = ChiSquared::new(2.0).unwrap();
    f.close(
        || "chi_squared_independence p vs ChiSquared(2).sf".into(),
        ri.p_value,
        chi_i.sf(ri.statistic),
        1e-10,
    );

    // Jarque–Bera is asymptotically ChiSquared(2).
    let jb = jarque_bera(&SAMPLE);
    f.close(
        || "jarque_bera p vs ChiSquared(2).sf".into(),
        jb.p_value,
        chi_i.sf(jb.statistic),
        1e-10,
    );
    // And its statistic is n/6·(S² + (K²)/4) with S skewness, K excess kurtosis.
    let n = SAMPLE.len() as f64;
    let expected_jb = n / 6.0 * (skewness(&SAMPLE).powi(2) + kurtosis(&SAMPLE).powi(2) / 4.0);
    f.close(
        || "jarque_bera statistic vs n/6(S²+K²/4)".into(),
        jb.statistic,
        expected_jb,
        1e-9,
    );

    f.finish("chi-squared tests");
}
