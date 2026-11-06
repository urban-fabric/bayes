//! Does anything produce a `NaN`, or an overflow, where a finite answer exists?
//!
//! The kernels are where every `exp`, `ln` and special function in the
//! workspace lives, so this is where an intermediate that overflows on its way
//! to a perfectly ordinary answer gets caught. The cases are chosen to be the
//! arguments and parameters that do it: a shape below `1e-3`, a scale above
//! `1e9`, an argument at the edge of `f64`.

use super::rel;
use super::table::cases;
use bayes_utils::Failures;

/// No in-support argument may produce a `NaN`, at any scale.
///
/// Including the extremes of `f64`. `wald_cdf` carries a comment about exactly
/// this: written as `√(λ/x)·(x/μ − 1)` it is `0 · ∞` at `x = ∞` and returned
/// `NaN` for `cdf(∞)` instead of 1.
#[test]
fn no_in_support_argument_produces_a_nan() {
    let mut f = Failures::default();
    let magnitudes = [
        1e-320_f64,
        1e-300,
        1e-100,
        1e-10,
        1e-3,
        1.0,
        1e3,
        1e10,
        1e100,
        1e300,
        f64::MAX,
    ];
    for item in cases() {
        let mut probes: Vec<f64> = Vec::new();
        for m in magnitudes {
            probes.push(m);
            if !item.lo.is_finite() {
                probes.push(-m);
            } else {
                probes.push(item.lo + m);
            }
        }
        probes.push(f64::INFINITY);
        if !item.lo.is_finite() {
            probes.push(f64::NEG_INFINITY);
        }
        for x in probes {
            let c = (item.cdf)(x);
            f.check(!c.is_nan(), || {
                format!("{}: cdf({x:.3e}) is NaN", item.name)
            });
            f.check((0.0..=1.0).contains(&c), || {
                format!("{}: cdf({x:.3e}) = {c} escapes [0, 1]", item.name)
            });
            let ls = (item.ln_sf)(x);
            f.check(!ls.is_nan(), || {
                format!("{}: ln_sf({x:.3e}) is NaN", item.name)
            });
            f.check(ls <= 1e-12, || {
                format!("{}: ln_sf({x:.3e}) = {ls} is positive", item.name)
            });
        }
    }
    f.finish("no NaN in support");
}

/// The CDF must saturate at the support bounds, exactly.
#[test]
fn the_cdf_saturates_at_the_support_bounds() {
    let mut f = Failures::default();
    for item in cases() {
        f.check((item.cdf)(f64::INFINITY) == 1.0, || {
            format!(
                "{}: cdf(inf) = {}, expected 1",
                item.name,
                (item.cdf)(f64::INFINITY)
            )
        });
        let below = if item.lo.is_finite() {
            item.lo - 1.0
        } else {
            f64::NEG_INFINITY
        };
        f.check((item.cdf)(below) == 0.0, || {
            format!(
                "{}: cdf({below:.3e}) = {}, expected 0 below the support",
                item.name,
                (item.cdf)(below)
            )
        });
        f.check((item.ln_sf)(below) == 0.0, || {
            format!(
                "{}: ln_sf({below:.3e}) = {}, expected 0 below the support",
                item.name,
                (item.ln_sf)(below)
            )
        });
    }
    f.finish("support saturation");
}

/// Parameters spanning twenty orders of magnitude must not break a formula
/// that is scale-free in principle.
///
/// Each of these families is a location-scale or scale family, so the answer at
/// a given *standardized* argument cannot depend on the scale at all. Checking
/// that directly is far sharper than checking for `NaN`: it catches an
/// intermediate that overflowed and was then clamped back into `[0, 1]`, which
/// looks healthy and is not.
#[test]
fn a_scale_family_is_invariant_to_its_scale() {
    use bayes_kernels as k;
    let mut f = Failures::default();
    let scales = [1e-12_f64, 1e-6, 1.0, 1e6, 1e12];
    for z in [0.1_f64, 1.0, 3.0, 10.0] {
        for s in scales {
            // Exponential: cdf depends on `rate * x` alone.
            f.close(
                || format!("exponential scale invariance at z={z}, rate 1/{s}"),
                k::exponential_cdf(z * s, 1.0 / s),
                k::exponential_cdf(z, 1.0),
                1e-12,
            );
            f.close(
                || format!("exponential ln_sf invariance at z={z}, rate 1/{s}"),
                k::exponential_ln_sf(z * s, 1.0 / s),
                k::exponential_ln_sf(z, 1.0),
                1e-12,
            );
            // Rayleigh, Weibull, half-normal: pure scale families.
            f.close(
                || format!("rayleigh scale invariance at z={z}, scale {s}"),
                k::rayleigh_ln_sf(z * s, s),
                k::rayleigh_ln_sf(z, 1.0),
                1e-12,
            );
            f.close(
                || format!("weibull scale invariance at z={z}, scale {s}"),
                k::weibull_ln_sf(z * s, s, 1.7),
                k::weibull_ln_sf(z, 1.0, 1.7),
                1e-12,
            );
            f.close(
                || format!("half_normal scale invariance at z={z}, scale {s}"),
                k::half_normal_ln_sf(z * s, s),
                k::half_normal_ln_sf(z, 1.0),
                1e-12,
            );
            // Normal: a location-scale family, so standardize both.
            f.close(
                || format!("normal scale invariance at z={z}, scale {s}"),
                k::normal_ln_sf(z * s, 0.0, s),
                k::normal_ln_sf(z, 0.0, 1.0),
                1e-12,
            );
            // Pareto is scale-free in the ratio `x / x_m`.
            f.close(
                || format!("pareto scale invariance at z={z}, minimum {s}"),
                k::pareto_ln_sf((z + 1.0) * s, s, 1.3),
                k::pareto_ln_sf(z + 1.0, 1.0, 1.3),
                1e-12,
            );
        }
    }
    f.finish("scale invariance");
}

/// A Wald whose `exp(2λ/μ)` factor overflows must still have a survival.
///
/// `2λ/μ = 800` puts that factor past `f64::MAX`, so the textbook form
/// `Φ(−a) − e^{2λ/μ}Φ(b)` evaluates to `NaN` — `inf · 0`. The identity
/// `b² − a² = 4λ/μ` removes it, and this pins both halves: the factor really
/// does overflow, and the survival really is finite and correct anyway.
///
/// Correctness is checked against the **CDF of a different parameterization**:
/// a Wald's survival at `x` has no elementary closed form to compare against,
/// but `cdf` and `ln_sf` are separate expressions here, and in the body they
/// must still complement.
#[test]
fn a_wald_survives_an_overflowing_exponential() {
    use bayes_kernels as k;
    let mut f = Failures::default();
    // The factor the textbook form needs, and cannot have.
    assert!(
        (2.0 * 400.0_f64 / 1.0).exp().is_infinite(),
        "this test is only meaningful if exp(2 lambda / mu) overflows"
    );

    for x in [0.5_f64, 0.9, 1.0, 1.1, 1.5, 2.0, 3.0, 8.0] {
        let ln_sf = k::wald_ln_sf(x, 1.0, 400.0);
        f.check(!ln_sf.is_nan(), || format!("wald(1,400) ln_sf({x}) is NaN"));
        f.check(ln_sf <= 1e-12, || {
            format!("wald(1,400) ln_sf({x}) = {ln_sf} is positive")
        });
    }
    // In the body, where `1 - cdf` still carries digits, the two must agree.
    for x in [0.85_f64, 0.95, 1.0, 1.05] {
        let direct = 1.0 - k::wald_cdf(x, 1.0, 400.0);
        if direct <= 1e-12 {
            continue;
        }
        f.check(
            rel(k::wald_ln_sf(x, 1.0, 400.0).exp(), direct) < 1e-6,
            || {
                format!(
                    "wald(1,400) at {x}: exp(ln_sf) = {:.10e}, 1 - cdf = {direct:.10e}",
                    k::wald_ln_sf(x, 1.0, 400.0).exp()
                )
            },
        );
    }
    // And far out it is still informative, where `1 - cdf` is exactly zero.
    let far: f64 = k::wald_ln_sf(3.0, 1.0, 400.0);
    assert!(
        far.is_finite() && far < -200.0,
        "the survival at x = 3 should be around e^-272, got ln_sf = {far}"
    );
    assert_eq!(
        1.0 - k::wald_cdf(3.0_f64, 1.0, 400.0),
        0.0,
        "which is the point: the subtraction has nothing left there"
    );
    f.finish("wald overflow");
}

/// Shapes near the edges of usefulness must not break the gamma and beta
/// families.
///
/// A shape of `1e-3` puts almost all of a gamma's mass within `1e-300` of zero;
/// a shape of `1e4` pushes it far from the origin and makes the series long.
/// Both are in the canonical sweep because both have broken something before.
#[test]
fn extreme_shapes_still_give_probabilities() {
    use bayes_kernels as k;
    let mut f = Failures::default();
    for shape in [1e-3_f64, 0.1, 1.0, 100.0, 1e4] {
        for x in [1e-300_f64, 1e-10, 0.5, 1.0, 100.0, 1e6] {
            let c = k::gamma_cdf(x, shape, 1.0);
            f.check(!c.is_nan() && (0.0..=1.0).contains(&c), || {
                format!("gamma_cdf({x:.1e}, shape {shape}) = {c}")
            });
            let ls = k::gamma_ln_sf(x, shape, 1.0);
            f.check(!ls.is_nan() && ls <= 1e-12, || {
                format!("gamma_ln_sf({x:.1e}, shape {shape}) = {ls}")
            });
            let chi = k::chi_squared_cdf(x, shape);
            f.check(!chi.is_nan() && (0.0..=1.0).contains(&chi), || {
                format!("chi_squared_cdf({x:.1e}, half-freedom {shape}) = {chi}")
            });
        }
    }
    for (a, b) in [
        (1e-3_f64, 1e-3_f64),
        (0.5, 0.5),
        (1.0, 1.0),
        (1e4, 1e4),
        (1e-3, 1e4),
    ] {
        for x in [0.0_f64, 1e-12, 0.25, 0.5, 0.75, 1.0 - 1e-12, 1.0] {
            let c = k::beta_cdf_on_unit(x, a, b);
            f.check(!c.is_nan() && (0.0..=1.0).contains(&c), || {
                format!("beta_cdf_on_unit({x}, {a}, {b}) = {c}")
            });
        }
        // The endpoints are exact for every shape pair.
        f.close(
            || format!("beta_cdf_on_unit(0, {a}, {b})"),
            k::beta_cdf_on_unit(0.0, a, b),
            0.0,
            0.0,
        );
        f.close(
            || format!("beta_cdf_on_unit(1, {a}, {b})"),
            k::beta_cdf_on_unit(1.0, a, b),
            1.0,
            0.0,
        );
    }
    f.finish("extreme shapes");
}
