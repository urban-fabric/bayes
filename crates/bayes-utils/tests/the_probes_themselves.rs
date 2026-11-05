//! **The probes themselves**: does the test machinery work?
//!
//! Everything in this crate is dev-only, which is exactly why it needs testing.
//! A quadrature rule that is quietly inaccurate makes every "moments against
//! integration" check in the workspace weaker than it looks; a numerical
//! derivative with the wrong coefficients makes every `pdf == d(cdf)/dx` check
//! agree for the wrong reason. And if [`Failures`] ever stopped panicking,
//! **every suite in the repository would pass vacuously** — several hundred
//! tests reporting success while asserting nothing.
//!
//! So the first test here is that the harness can fail. The rest pin the
//! numerics against closed forms that need no library to evaluate: polynomials,
//! where Simpson's rule and a five-point difference are *exact* rather than
//! approximate, so the comparison is to the last bit.

#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use bayes_utils::{
    Failures, chi_squared_gof_pvalue, derivative, integrate_unit, ks_pvalue, ks_statistic,
    mean_tolerance, rel_error, second_derivative, simpson, variance_tolerance,
};

/// [`Failures`] accumulates and then panics. If it did not, nothing would.
///
/// Both halves matter. It must **not** abort on the first failure — that is the
/// whole reason the repository uses it instead of bare asserts, since a sweep
/// reporting one failure out of six hundred tells you far less than one
/// reporting all twelve. And it must panic at `finish`, or the suites are
/// decorative.
#[test]
fn the_failure_harness_accumulates_and_then_panics() {
    // Counting without aborting.
    let mut f = Failures::default();
    f.check(false, || "first".into());
    f.check(true, || "second".into());
    f.check(false, || "third".into());
    f.close(|| "fourth".into(), 1.0, 2.0, 1e-9);
    assert_eq!(
        f.failures(),
        3,
        "three of four checks failed and all three should have been recorded"
    );

    // And `finish` panics, naming the count.
    let panicked = std::panic::catch_unwind(|| {
        let mut f = Failures::default();
        f.check(false, || "a failure".into());
        f.finish("deliberate");
    });
    assert!(
        panicked.is_err(),
        "Failures::finish must panic when a check failed — otherwise every suite in the \
         workspace passes vacuously"
    );

    // A clean run does not panic.
    let clean = std::panic::catch_unwind(|| {
        let mut f = Failures::default();
        f.check(true, || "fine".into());
        f.close(|| "also fine".into(), 1.0, 1.0 + 1e-15, 1e-9);
        f.finish("clean");
    });
    assert!(clean.is_ok(), "a run with no failures must not panic");
}

/// `close` compares against an absolute tolerance, and a `NaN` is never close
/// to anything.
///
/// The `NaN` half is load bearing: `NaN` comparisons are all false, so a check
/// written as `!(diff > tol)` would treat every `NaN` as a pass and silently
/// excuse exactly the failures most worth catching.
#[test]
fn closeness_refuses_a_nan() {
    let mut f = Failures::default();
    f.close(|| "nan vs number".into(), f64::NAN, 1.0, 1e9);
    f.close(|| "number vs nan".into(), 1.0, f64::NAN, 1e9);
    f.close(|| "nan vs nan".into(), f64::NAN, f64::NAN, 1e9);
    f.close(|| "inf vs number".into(), f64::INFINITY, 1.0, 1e9);
    assert_eq!(
        f.failures(),
        4,
        "a NaN or an infinity must never satisfy a closeness check, however wide the tolerance"
    );
}

/// Simpson's rule is **exact** for polynomials up to degree three.
///
/// Not approximately — the rule is built from a quadratic interpolant whose
/// cubic error term cancels over a pair of panels. So this is a comparison to
/// machine precision against an antiderivative evaluated by hand, which pins
/// the coefficients rather than merely checking the rule is "about right".
#[test]
fn simpson_is_exact_on_cubics() {
    let mut f = Failures::default();
    // (name, f, antiderivative)
    /// A named integrand and its antiderivative.
    type Pair = (&'static str, fn(f64) -> f64, fn(f64) -> f64);
    let cases: Vec<Pair> = vec![
        ("1", |_| 1.0, |x| x),
        ("x", |x| x, |x| x * x / 2.0),
        ("x^2", |x| x * x, |x| x * x * x / 3.0),
        ("x^3", |x| x * x * x, |x| x * x * x * x / 4.0),
        (
            "2x^3 - 5x + 1",
            |x| 2.0 * x * x * x - 5.0 * x + 1.0,
            |x| 0.5 * x * x * x * x - 2.5 * x * x + x,
        ),
    ];
    for (name, g, antiderivative) in cases {
        for (a, b) in [(0.0, 1.0), (-2.0, 3.0), (1.5, 1.5), (-1.0, -0.25)] {
            let truth = antiderivative(b) - antiderivative(a);
            for n in [2_usize, 4, 10, 100] {
                f.close(
                    || format!("simpson {name} on [{a}, {b}] with n={n}"),
                    simpson(g, a, b, n),
                    truth,
                    1e-12 * truth.abs().max(1.0),
                );
            }
        }
    }
    f.finish("Simpson exactness");
}

/// Simpson's error falls as `h⁴`, and it skips non-finite samples rather than
/// returning `NaN`.
///
/// The convergence order is what justifies the point counts chosen throughout
/// the workspace. The skipping behaviour is what lets a quantile with an
/// integrable endpoint singularity be integrated at all — and it is checked
/// here because it is also a hazard: a rule that silently drops samples can
/// drop *meaningful* ones.
#[test]
fn simpson_converges_at_fourth_order_and_tolerates_a_singular_endpoint() {
    // `exp` is not a polynomial, so the error term is live.
    let truth = std::f64::consts::E - 1.0;
    let coarse = (simpson(f64::exp, 0.0, 1.0, 8) - truth).abs();
    let fine = (simpson(f64::exp, 0.0, 1.0, 16) - truth).abs();
    let ratio = coarse / fine;
    assert!(
        (12.0..20.0).contains(&ratio),
        "halving the step should cut the error by about 16 for a fourth-order rule; got {ratio:.3}"
    );

    // An integrable singularity at the left endpoint: ∫₀¹ x^(-1/2) dx = 2.
    let value = simpson(|x| x.powf(-0.5), 0.0, 1.0, 200_000);
    assert!(
        value.is_finite(),
        "a non-finite sample at the endpoint must be skipped, not propagated: got {value}"
    );
    assert!(
        (value - 2.0).abs() < 0.01,
        "∫₀¹ x^(-1/2) = 2, got {value} — the singular endpoint should contribute ~nothing"
    );
}

/// `integrate_unit` is `∫₀¹ g(u) du`, including where `g` blows up at an
/// endpoint.
///
/// The `tanh` substitution is what makes `E[X] = ∫₀¹ Q(u) du` checkable for a
/// heavy-tailed family, so these are the integrals that matter: a constant, a
/// linear ramp, and a `u^{-1/2}` singularity whose antiderivative is known.
#[test]
fn integrate_unit_handles_the_endpoints() {
    let mut f = Failures::default();
    for (name, g, truth) in [
        ("1", (|_: f64| 1.0) as fn(f64) -> f64, 1.0),
        ("u", |u: f64| u, 0.5),
        ("u^2", |u: f64| u * u, 1.0 / 3.0),
        ("u^(-1/2)", |u: f64| u.powf(-0.5), 2.0),
        ("(1-u)^(-1/2)", |u: f64| (1.0 - u).powf(-0.5), 2.0),
        ("-ln u", |u: f64| -u.ln(), 1.0),
    ] {
        f.close(
            || format!("integrate_unit {name}"),
            integrate_unit(g, 20_000),
            truth,
            1e-6 * truth.abs(),
        );
    }
    f.finish("integrate_unit");
}

/// The five-point difference is **exact** for polynomials up to degree four,
/// and the second difference up to degree five.
///
/// That is what `O(h⁴)` means for these stencils: the leading error term is the
/// fifth (respectively sixth) derivative, which a low-degree polynomial does
/// not have. So the check is to machine precision against a hand-differentiated
/// polynomial — which pins the `(-1, 8, -8, 1)/12h` and
/// `(-1, 16, -30, 16, -1)/12h²` coefficients exactly.
#[test]
fn the_finite_differences_are_exact_on_low_degree_polynomials() {
    let mut f = Failures::default();
    let quartic = |x: f64| 3.0 * x.powi(4) - 2.0 * x.powi(3) + x * x - 5.0 * x + 7.0;
    let first = |x: f64| 12.0 * x.powi(3) - 6.0 * x * x + 2.0 * x - 5.0;
    let second = |x: f64| 36.0 * x * x - 12.0 * x + 2.0;
    for x in [-2.0_f64, -0.5, 0.0, 0.25, 1.0, 3.0] {
        for h in [1e-1, 1e-2, 1e-3] {
            let scale = first(x).abs().max(1.0);
            f.close(
                || format!("derivative of a quartic at {x} with h={h}"),
                derivative(quartic, x, h),
                first(x),
                1e-9 * scale,
            );
            let scale2 = second(x).abs().max(1.0);
            f.close(
                || format!("second derivative of a quartic at {x} with h={h}"),
                second_derivative(quartic, x, h),
                second(x),
                1e-7 * scale2,
            );
        }
    }
    f.finish("finite differences");
}

/// `rel_error` floors its denominator at 1, so a comparison against zero does
/// not divide by it.
#[test]
fn relative_error_is_defined_against_zero() {
    assert_eq!(rel_error(0.0, 0.0), 0.0);
    assert_eq!(rel_error(1e-9, 0.0), 1e-9, "the denominator floors at 1");
    assert!((rel_error(1.0, 2.0) - 0.5).abs() < 1e-15);
    assert!((rel_error(102.0, 100.0) - 0.02).abs() < 1e-15);
    assert!(rel_error(f64::NAN, 1.0).is_nan());
}

/// The Kolmogorov–Smirnov probe separates a correct null from a wrong one.
///
/// Tested in both directions, because only one of them is informative on its
/// own: a statistic that was always tiny would pass the first check and fail
/// the second, and one that was always huge the reverse.
#[test]
fn the_kolmogorov_smirnov_probe_discriminates() {
    // A deterministic "sample" placed exactly at the uniform quantiles has the
    // smallest achievable statistic, so a correct null gives a large p-value.
    let n = 200;
    let aligned: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();
    let d_good = ks_statistic(&aligned, |x| x.clamp(0.0, 1.0));
    assert!(
        d_good < 2.0 / n as f64,
        "a sample at the uniform quantiles should have a tiny KS statistic, got {d_good}"
    );
    assert!(
        ks_pvalue(d_good, n) > 0.5,
        "a correct null should not be rejected: p = {}",
        ks_pvalue(d_good, n)
    );

    // The same sample against a badly wrong null: everything shifted.
    let d_bad = ks_statistic(&aligned, |x| (x / 4.0).clamp(0.0, 1.0));
    assert!(
        d_bad > 0.5,
        "a null four times too wide should be obvious, got {d_bad}"
    );
    assert!(
        ks_pvalue(d_bad, n) < 1e-6,
        "a wrong null should be rejected decisively: p = {}",
        ks_pvalue(d_bad, n)
    );

    // Degenerate input is refused rather than scored.
    assert!(ks_statistic(&[], |x| x).is_nan());
    assert!(ks_pvalue(0.1, 0).is_nan());
    assert!(ks_pvalue(f64::NAN, 10).is_nan());
    // And the p-value is monotone decreasing in the statistic.
    let mut previous = 2.0;
    for step in 0..20 {
        let p = ks_pvalue(f64::from(step) * 0.05, 100);
        assert!(
            p <= previous + 1e-12,
            "ks_pvalue rose at d = {}: {previous} -> {p}",
            f64::from(step) * 0.05
        );
        previous = p;
    }
}

/// The chi-squared goodness-of-fit probe separates agreement from
/// disagreement, and pools thin cells rather than dividing by them.
#[test]
fn the_chi_squared_goodness_of_fit_probe_discriminates() {
    let expected = vec![25.0, 25.0, 25.0, 25.0];
    // Exact agreement: p should be large.
    let p_good = chi_squared_gof_pvalue(&expected, &expected, 5.0);
    assert!(
        p_good > 0.99,
        "identical observed and expected should give p near 1, got {p_good}"
    );
    // Gross disagreement: p should be tiny.
    let p_bad = chi_squared_gof_pvalue(&[100.0, 0.0, 0.0, 0.0], &expected, 5.0);
    assert!(
        p_bad < 1e-10,
        "all mass in one of four equally-likely cells should be rejected, got {p_bad}"
    );
    // A cell below `min_expected` must be pooled, not divided by: a zero
    // expectation would otherwise produce an infinite contribution.
    let thin = chi_squared_gof_pvalue(&[50.0, 50.0, 0.0], &[50.0, 50.0, 0.0], 5.0);
    assert!(
        thin.is_finite(),
        "a cell with zero expectation must be pooled away, got {thin}"
    );
    // Mismatched lengths are refused.
    assert!(chi_squared_gof_pvalue(&[1.0], &[1.0, 2.0], 5.0).is_nan());
    assert!(chi_squared_gof_pvalue(&[], &[], 5.0).is_nan());
}

/// The derived sampling tolerances are the formulas they claim, and absent
/// where the moment they need does not exist.
///
/// `variance_tolerance` returning `None` for a non-finite kurtosis is the point
/// of the whole design: a family with no fourth moment has no usable bound on
/// its sample variance, and saying so beats inventing one.
#[test]
fn the_sampling_tolerances_are_derived_not_invented() {
    // `sigmas * sqrt(variance / n)`.
    assert!((mean_tolerance(4.0, 100, 5.0) - 5.0 * 0.2).abs() < 1e-15);
    assert!((mean_tolerance(1.0, 10_000, 3.0) - 0.03).abs() < 1e-15);
    // Four times the sample size halves the bound.
    let wide = mean_tolerance(2.0, 1_000, 4.0);
    let narrow = mean_tolerance(2.0, 4_000, 4.0);
    assert!((wide / narrow - 2.0).abs() < 1e-12);

    // `sigmas * variance * sqrt((kurtosis + 2) / n)`.
    let bound = variance_tolerance(3.0, 1.0, 100, 4.0).expect("finite kurtosis has a bound");
    assert!((bound - 4.0 * 3.0 * (3.0_f64 / 100.0).sqrt()).abs() < 1e-12);
    // A normal has excess kurtosis 0, so the factor is sqrt(2/n).
    let normal = variance_tolerance(1.0, 0.0, 200, 1.0).expect("a normal has a fourth moment");
    assert!((normal - (2.0_f64 / 200.0).sqrt()).abs() < 1e-12);

    // No fourth moment, no bound.
    assert!(variance_tolerance(1.0, f64::NAN, 100, 4.0).is_none());
    assert!(variance_tolerance(1.0, f64::INFINITY, 100, 4.0).is_none());
    assert!(variance_tolerance(f64::NAN, 1.0, 100, 4.0).is_none());
}
