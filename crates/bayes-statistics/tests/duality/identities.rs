//! One test written two ways, and the two must coincide.
//!
//! These are algebraic identities between tests rather than between a test and
//! an interval: `t² = F` on two groups, a paired `t` as a one-sample `t` on
//! differences, Welch collapsing onto Student when the variances match. Each
//! pairs two functions that share no code and reach the same number by
//! different routes, so the identity pins down the denominator each one uses.

use bayes_statistics::hypothesis::{one_sample_t, one_way_anova, paired_t, two_sample_t, welch_t};
use bayes_utils::Failures;

/// A two-sample `t` statistic squared is the one-way ANOVA `F` on the same two
/// groups, and the p-values agree exactly.
///
/// `F(1, n−2) = t(n−2)²` is the identity, and the two functions reach it by
/// entirely different routes: one pools two variances into a standard error,
/// the other decomposes a total sum of squares into between and within. A
/// disagreement about the pooled denominator shows up here and nowhere else.
#[test]
fn a_two_sample_t_squared_is_the_anova_f() {
    let mut f = Failures::default();
    let pairs: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        (
            "equal size",
            vec![1.0, 2.0, 3.0, 4.0],
            vec![3.0, 4.0, 5.0, 6.0],
        ),
        (
            "unequal size",
            vec![2.0, 4.0, 6.0],
            vec![1.0, 1.5, 2.5, 3.0, 5.0, 7.0],
        ),
        ("identical groups", vec![5.0, 6.0, 7.0], vec![5.0, 6.0, 7.0]),
        (
            "well separated",
            vec![0.0, 0.1, -0.1, 0.05],
            vec![100.0, 100.2, 99.8, 100.1],
        ),
    ];
    for (name, a, b) in pairs {
        let t = two_sample_t(&a, &b);
        let anova = one_way_anova(&[&a, &b]);
        if !t.statistic.is_finite() || !anova.statistic.is_finite() {
            continue;
        }
        f.check(
            (t.statistic * t.statistic - anova.statistic).abs()
                < 1e-9 * anova.statistic.abs().max(1.0),
            || {
                format!(
                    "{name}: t² = {:.12} but the ANOVA F = {:.12}",
                    t.statistic * t.statistic,
                    anova.statistic
                )
            },
        );
        f.check(
            (t.p_value - anova.p_value).abs() < 1e-9 * anova.p_value.max(1e-12),
            || {
                format!(
                    "{name}: the two-sided t p-value {:.12} should equal the ANOVA p-value {:.12}",
                    t.p_value, anova.p_value
                )
            },
        );
    }
    f.finish("t² = F");
}

/// A paired `t` test is a one-sample `t` test on the differences.
///
/// Trivially true by definition, which is exactly why it is worth asserting:
/// the failure mode is a paired test that quietly computes something else —
/// pooling the two samples, or dividing by the wrong `n` — and the result is
/// still a plausible p-value.
#[test]
fn a_paired_t_test_is_a_one_sample_t_on_the_differences() {
    let mut f = Failures::default();
    let pairs: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        ("small", vec![1.0, 2.0, 3.0, 4.0], vec![1.5, 1.8, 3.6, 3.9]),
        (
            "correlated",
            (0..12).map(f64::from).collect(),
            (0..12).map(|i| f64::from(i) + 0.3).collect(),
        ),
        (
            "offset",
            vec![1e4, 1e4 + 1.0, 1e4 + 2.0, 1e4 + 3.0],
            vec![1e4 + 5.0, 1e4 + 4.0, 1e4 + 8.0, 1e4 + 6.0],
        ),
    ];
    for (name, a, b) in pairs {
        let differences: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x - y).collect();
        let paired = paired_t(&a, &b);
        let direct = one_sample_t(&differences, 0.0);
        f.check(
            (paired.statistic - direct.statistic).abs() < 1e-12 * direct.statistic.abs().max(1.0),
            || {
                format!(
                    "{name}: paired statistic {:.12} vs one-sample on differences {:.12}",
                    paired.statistic, direct.statistic
                )
            },
        );
        f.check((paired.p_value - direct.p_value).abs() < 1e-12, || {
            format!(
                "{name}: paired p {:.12} vs one-sample p {:.12}",
                paired.p_value, direct.p_value
            )
        });
    }
    f.finish("paired = one-sample on differences");
}

/// Welch and Student agree exactly when the two samples have equal size and
/// equal variance, and not otherwise.
///
/// Both halves matter. Agreement in the special case says the two share a
/// standard error there; *disagreement* elsewhere says Welch is actually doing
/// the Satterthwaite correction rather than silently falling back to the
/// pooled form.
#[test]
fn welch_reduces_to_student_only_when_the_variances_match() {
    let mut f = Failures::default();
    // Equal size, equal variance: a pure translation of the same shape.
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let b: Vec<f64> = a.iter().map(|x| x + 7.0).collect();
    let (student, welch) = (two_sample_t(&a, &b), welch_t(&a, &b));
    f.check(
        (student.statistic - welch.statistic).abs() < 1e-12 * student.statistic.abs(),
        || {
            format!(
                "at equal size and variance the statistics should match: {:.12} vs {:.12}",
                student.statistic, welch.statistic
            )
        },
    );
    f.check((student.p_value - welch.p_value).abs() < 1e-12, || {
        format!(
            "at equal size and variance the p-values should match: {:.12} vs {:.12}",
            student.p_value, welch.p_value
        )
    });

    // Badly unequal variances: Welch must *not* agree, and must be the more
    // conservative of the two.
    let tight = vec![0.0, 0.01, -0.01, 0.005, -0.005];
    let loose = vec![5.0, -5.0, 10.0, -10.0, 2.0];
    let (student, welch) = (two_sample_t(&tight, &loose), welch_t(&tight, &loose));
    f.check((student.p_value - welch.p_value).abs() > 1e-6, || {
        format!(
            "with variances three orders of magnitude apart Welch should differ from Student, \
             but both gave {:.12}",
            student.p_value
        )
    });
    f.check(welch.p_value > student.p_value, || {
        format!(
            "Welch should be the more conservative test here: Welch p = {:.12}, Student p = \
             {:.12}",
            welch.p_value, student.p_value
        )
    });
    f.finish("Welch vs Student");
}
