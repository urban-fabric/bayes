//! Shape and range invariants every continuous family must satisfy.
//!
//! The cheap checks, and the ones most likely to catch a parameterization slip:
//! a CDF confined to `[0,1]` and non-decreasing, saturating at the support
//! bounds and staying flat outside them; a non-negative density that agrees with
//! its own logarithm; and the survival and log forms agreeing with the linear
//! ones wherever both carry information.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

// The glob brings every family constructor into scope for the validation sweep.
use bayes_utils::families::{ContCase, continuous_cases};
use bayes_utils::{Failures, rel_error as rel};

// ---------------------------------------------------------------------------
// Shape and range invariants
// ---------------------------------------------------------------------------

#[test]
fn cdf_is_in_the_unit_interval_and_never_decreases() {
    let mut f = Failures::default();
    for ContCase { name, dist } in continuous_cases() {
        let (a, b) = (dist.inverse_cdf(1e-9), dist.inverse_cdf(1.0 - 1e-9));
        if !a.is_finite() || !b.is_finite() {
            continue;
        }
        let mut prev = f64::NEG_INFINITY;
        for i in 0..=200 {
            let x = a + (b - a) * i as f64 / 200.0;
            let c = dist.cdf(x);
            f.check(c.is_finite() && (-0.0..=1.0).contains(&c), || {
                format!("{name}: cdf({x:.6e}) = {c} outside [0,1]")
            });
            f.check(c >= prev - 1e-12, || {
                format!("{name}: cdf decreased at x={x:.6e}: {prev} -> {c}")
            });
            prev = c;
        }
    }
    f.finish("cdf range/monotonicity");
}

#[test]
fn cdf_saturates_at_and_outside_the_support_boundaries() {
    let mut f = Failures::default();
    for ContCase { name, dist } in continuous_cases() {
        let (lo, hi) = (dist.lo(), dist.hi());
        if lo.is_finite() {
            let c = dist.cdf(lo);
            f.check(c.abs() < 1e-9, || {
                format!("{name}: cdf(min={lo:.6e}) = {c}, expected 0")
            });
            let below = lo - lo.abs().max(1.0) * 1e-3;
            let cb = dist.cdf(below);
            f.check(cb.abs() < 1e-12, || {
                format!("{name}: cdf({below:.6e}) = {cb} below support min {lo:.6e}")
            });
        }
        if hi.is_finite() {
            let c = dist.cdf(hi);
            f.check((c - 1.0).abs() < 1e-9, || {
                format!("{name}: cdf(max={hi:.6e}) = {c}, expected 1")
            });
            let above = hi + hi.abs().max(1.0) * 1e-3;
            let ca = dist.cdf(above);
            f.check((ca - 1.0).abs() < 1e-12, || {
                format!("{name}: cdf({above:.6e}) = {ca} above support max {hi:.6e}")
            });
        }
    }
    f.finish("cdf support saturation");
}

#[test]
fn pdf_is_nonnegative_and_agrees_with_ln_pdf() {
    let mut f = Failures::default();
    for ContCase { name, dist } in continuous_cases() {
        let (a, b) = (dist.inverse_cdf(1e-6), dist.inverse_cdf(1.0 - 1e-6));
        if !a.is_finite() || !b.is_finite() {
            continue;
        }
        for i in 0..=100 {
            let x = a + (b - a) * i as f64 / 100.0;
            let p = dist.pdf(x);
            f.check(!p.is_nan() && p >= 0.0, || {
                format!("{name}: pdf({x:.6e}) = {p}, expected a nonnegative number")
            });
            let lp = dist.ln_pdf(x);
            if p > 0.0 && p.is_finite() {
                f.check(rel(lp, p.ln()) < 1e-9, || {
                    format!("{name}: ln_pdf({x:.6e}) = {lp} but pdf.ln() = {}", p.ln())
                });
            }
        }
    }
    f.finish("pdf/ln_pdf consistency");
}

#[test]
fn survival_is_the_cdf_complement() {
    let mut f = Failures::default();
    for ContCase { name, dist } in continuous_cases() {
        let (a, b) = (dist.inverse_cdf(1e-9), dist.inverse_cdf(1.0 - 1e-9));
        if !a.is_finite() || !b.is_finite() {
            continue;
        }
        for i in 0..=100 {
            let x = a + (b - a) * i as f64 / 100.0;
            let (c, s) = (dist.cdf(x), dist.sf(x));
            f.check((c + s - 1.0).abs() < 1e-9, || {
                format!("{name}: cdf + sf at {x:.6e} = {}, expected 1", c + s)
            });
        }
    }
    f.finish("survival complement");
}

#[test]
fn log_cdf_and_log_sf_agree_with_their_linear_forms() {
    let mut f = Failures::default();
    for ContCase { name, dist } in continuous_cases() {
        let (a, b) = (dist.inverse_cdf(1e-9), dist.inverse_cdf(1.0 - 1e-9));
        if !a.is_finite() || !b.is_finite() {
            continue;
        }
        for i in 0..=100 {
            let x = a + (b - a) * i as f64 / 100.0;
            let c = dist.cdf(x);
            if c > 1e-280 && c < 1.0 {
                let lc = dist.ln_cdf(x);
                f.check(rel(lc, c.ln()) < 1e-8, || {
                    format!("{name}: ln_cdf({x:.6e}) = {lc}, cdf.ln() = {}", c.ln())
                });
            }
            let s = dist.sf(x);
            if s > 1e-280 && s < 1.0 {
                let ls = dist.ln_sf(x);
                f.check(rel(ls, s.ln()) < 1e-8, || {
                    format!("{name}: ln_sf({x:.6e}) = {ls}, sf.ln() = {}", s.ln())
                });
            }
        }
    }
    f.finish("log cdf/sf consistency");
}

// ---------------------------------------------------------------------------
// Quantile invariants
// ---------------------------------------------------------------------------
