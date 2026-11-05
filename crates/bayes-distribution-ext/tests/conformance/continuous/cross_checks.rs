//! The independent cross-checks: two code paths for the same quantity.
//!
//! `pdf` against a Richardson-extrapolated central difference of `cdf`, and the
//! analytic moments and entropy against integration in **quantile space**
//! (`E[X] = ∫₀¹ Q(u) du`, `H = −∫₀¹ ln f(Q(u)) du`) under the
//! endpoint-clustering substitution.
//!
//! Quantile space is deliberate rather than incidental: several families here
//! have integrable density singularities — `Beta(½,½)` at both ends,
//! `Gamma(10⁻³,1)` at zero — that no fixed-grid quadrature in `x` can survive
//! but which are mild in `u`. Getting that wrong produced four false bug reports
//! before the substitution went in.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

// The glob brings every family constructor into scope for the validation sweep.
use bayes_utils::families::{
    ContCase, continuous_cases, quantile_unrepresentable_at, quantiles_are_unrepresentable,
};
use bayes_utils::{Failures, integrate_unit, rel_error as rel};

// ---------------------------------------------------------------------------
// Shape and range invariants
// ---------------------------------------------------------------------------

/// `pdf` against a Richardson-extrapolated central difference of `cdf`.
///
/// Richardson is what makes this usable as a *tight* check: the two step sizes
/// also tell us whether the difference is trustworthy at all, so a density with
/// enormous local curvature is skipped rather than reported as a defect.
#[test]
fn pdf_is_the_derivative_of_the_cdf() {
    let mut f = Failures::default();
    for ContCase { name, dist } in continuous_cases() {
        if quantiles_are_unrepresentable(&*dist) {
            continue;
        }
        for &p in &[0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9] {
            let x = dist.inverse_cdf(p);
            if !x.is_finite() {
                continue;
            }
            // Local scale from the surrounding quantile spread.
            let scale = (dist.inverse_cdf((p + 0.05).min(1.0))
                - dist.inverse_cdf((p - 0.05).max(0.0)))
            .abs();
            if !scale.is_finite() || scale == 0.0 {
                continue;
            }
            let h = scale * 1e-4;
            let d1 = (dist.cdf(x + h) - dist.cdf(x - h)) / (2.0 * h);
            let d2 = (dist.cdf(x + h / 2.0) - dist.cdf(x - h / 2.0)) / h;
            if !d1.is_finite() || !d2.is_finite() || d2 == 0.0 {
                continue;
            }
            // Disagreement between step sizes means the difference itself is
            // unreliable here; that is a limit of the probe, not a defect.
            if (d1 - d2).abs() / d2.abs() > 1e-3 {
                continue;
            }
            let numeric = (4.0 * d2 - d1) / 3.0;
            let analytic = dist.pdf(x);
            f.check(
                rel(numeric, analytic) < 1e-5 * analytic.abs().max(1.0),
                || {
                    format!(
                        "{name}: at p={p} x={x:.6e}: d/dx cdf = {numeric:.12e} but \
                     pdf = {analytic:.12e} (rel {:.3e})",
                        rel(numeric, analytic)
                    )
                },
            );
        }
    }
    f.finish("pdf = d(cdf)/dx");
}

/// `E[X] = ∫₀¹ Q(u) du` and `Var(X) = ∫₀¹ (Q(u) − μ)² du`, integrated with the
/// endpoint-clustering substitution above.
#[test]
fn analytic_mean_and_variance_match_quantile_space_integration() {
    let mut f = Failures::default();
    for ContCase { name, dist } in continuous_cases() {
        let (mean, var) = (dist.mean(), dist.variance());
        if !mean.is_finite() || !var.is_finite() || quantiles_are_unrepresentable(&*dist) {
            continue;
        }
        let q = |u: f64| dist.inverse_cdf(u);
        let sd = var.sqrt();

        // A mean is compared on the distribution's own scale. Judging the mean
        // of `Normal(0, 1e10)` against an absolute tolerance would flag an
        // error of 1e-6 as a defect when it is 1e-16 of a standard deviation.
        let m1 = integrate_unit(q, 20_000);
        f.check((m1 - mean).abs() <= 1e-6 * sd.max(mean.abs()), || {
            format!("{name}: ∫Q(u)du = {m1:.9e}, analytic mean = {mean:.9e}")
        });

        // The quadrature error of a variance integral is governed by the fourth
        // moment, so a family whose excess kurtosis is huge or undefined has a
        // tail this probe cannot resolve. That is a limit of the probe.
        let kurt = dist.kurtosis();
        if kurt.is_finite() && kurt <= 50.0 {
            let m2 = integrate_unit(|u| (q(u) - mean).powi(2), 20_000);
            f.check((m2 - var).abs() <= 1e-4 * var.abs(), || {
                format!("{name}: ∫(Q(u)-μ)²du = {m2:.9e}, analytic variance = {var:.9e}")
            });
        }
    }
    f.finish("moments vs quantile-space integration");
}

/// Differential entropy as `H = -∫₀¹ ln f(Q(u)) du`.
#[test]
fn analytic_entropy_matches_quantile_space_integration() {
    let mut f = Failures::default();
    for ContCase { name, dist } in continuous_cases() {
        let h = dist.entropy();
        // The integral runs over the whole quantile range, so it is only
        // meaningful when that whole range is representable.
        if !h.is_finite()
            || quantiles_are_unrepresentable(&*dist)
            || quantile_unrepresentable_at(&*dist, 1e-3)
            || quantile_unrepresentable_at(&*dist, 1.0 - 1e-3)
        {
            continue;
        }
        let num = -integrate_unit(|u| dist.ln_pdf(dist.inverse_cdf(u)), 20_000);
        f.check(rel(num, h) < 1e-5, || {
            format!("{name}: -∫ln f(Q(u))du = {num:.9e}, analytic entropy = {h:.9e}")
        });
    }
    f.finish("entropy vs quantile-space integration");
}
