//! Do the families that coincide actually coincide?
//!
//! Each check here is an **identity between separately written formulas**. A
//! gamma CDF is an incomplete gamma; an exponential CDF is `1 − e^{−λx}`.
//! Nothing in the source connects them, so `Gamma(1, λ) == Exponential(λ)`
//! holds only if both are right — which is what makes these the strongest
//! checks available to a crate with no distributions in it.

use super::rel;
use bayes_kernels as k;
use bayes_utils::Failures;

/// Arguments spanning several decades, since a reduction that holds in the
/// body can still fail where one side's series changes branch.
const POINTS: [f64; 9] = [1e-6, 1e-3, 0.1, 0.5, 1.0, 2.5, 10.0, 100.0, 1e4];

/// `Gamma(1, λ)` is `Exponential(λ)`, and `ChiSquared(2)` is `Exponential(½)`.
///
/// The density comparison passes `ln Γ(1) = 0` as the normalizer, which is the
/// whole content of `shape = 1` for a gamma.
///
/// Three routes to the same numbers: a lower regularized incomplete gamma, the
/// chi-squared's own specialization of it, and `1 − e^{−λx}`.
#[test]
fn the_gamma_family_collapses_to_the_exponential() {
    let mut f = Failures::default();
    for rate in [0.25_f64, 1.0, 7.5] {
        for x in POINTS {
            f.close(
                || format!("gamma_cdf(1, {rate}) vs exponential at {x}"),
                k::gamma_cdf(x, 1.0, rate),
                k::exponential_cdf(x, rate),
                1e-12,
            );
            // `gamma_ln_sf` goes through `gamma_ur`, which underflows near
            // `1e-308`, so it floors around `ln_sf = -709` where the
            // exponential's closed form keeps going. Documented on the kernel;
            // compared only where both carry information.
            let reference = k::exponential_ln_sf(x, rate);
            if reference > -700.0 {
                f.close(
                    || format!("gamma_ln_sf(1, {rate}) vs exponential at {x}"),
                    k::gamma_ln_sf(x, 1.0, rate),
                    reference,
                    1e-10,
                );
            }
            f.close(
                || format!("gamma_log_pdf(1, {rate}) vs exponential at {x}"),
                k::gamma_log_pdf_on_support(x, 1.0, rate, 0.0),
                k::exponential_log_pdf_on_support(x, rate),
                1e-12,
            );
        }
    }
    // Two degrees of freedom is the exponential of rate one half.
    //
    // Note the argument: these kernels take **half** the degrees of freedom,
    // so two degrees of freedom is `1.0`. Passing `2.0` here instead compares
    // against a chi-squared on four degrees of freedom, which is a different
    // distribution and looked like a factor-of-four error in the library.
    for x in POINTS {
        f.close(
            || format!("chi_squared_cdf(2 df) vs exponential(0.5) at {x}"),
            k::chi_squared_cdf(x, 1.0),
            k::exponential_cdf(x, 0.5),
            1e-12,
        );
        // Both are exact until `gamma_ur` underflows; past that the chi-squared
        // reports `-inf` where the exponential still has `-5000`. That floor is
        // a property of the incomplete gamma, documented on `gamma_ln_sf`, and
        // is still 292 orders of magnitude below where `ln(1 - cdf)` stopped.
        let reference = k::exponential_ln_sf(x, 0.5);
        if reference > -700.0 {
            f.close(
                || format!("chi_squared_ln_sf(2 df) vs exponential(0.5) at {x}"),
                k::chi_squared_ln_sf(x, 1.0),
                reference,
                1e-10,
            );
        }
    }
    f.finish("gamma reductions");
}

/// `Weibull(λ, 1)` is `Exponential(1/λ)`, and `Weibull(σ√2, 2)` is
/// `Rayleigh(σ)`.
///
/// The second is the one worth having: it relates a two-parameter shape family
/// to a one-parameter one through a `√2` in the scale, which is exactly the
/// sort of factor that goes missing.
#[test]
fn the_weibull_collapses_to_the_exponential_and_the_rayleigh() {
    let mut f = Failures::default();
    for scale in [0.5_f64, 1.0, 4.0] {
        for x in POINTS {
            f.close(
                || format!("weibull_cdf({scale}, 1) vs exponential at {x}"),
                k::weibull_cdf(x, scale, 1.0),
                k::exponential_cdf(x, 1.0 / scale),
                1e-12,
            );
            f.close(
                || format!("weibull_ln_sf({scale}, 1) vs exponential at {x}"),
                k::weibull_ln_sf(x, scale, 1.0),
                k::exponential_ln_sf(x, 1.0 / scale),
                1e-12,
            );
            // Rayleigh(sigma) == Weibull(sigma*sqrt(2), 2).
            f.close(
                || format!("weibull_ln_sf({scale}√2, 2) vs rayleigh({scale}) at {x}"),
                k::weibull_ln_sf(x, scale * std::f64::consts::SQRT_2, 2.0),
                k::rayleigh_ln_sf(x, scale),
                1e-12,
            );
            f.close(
                || format!("weibull_cdf({scale}√2, 2) vs rayleigh({scale}) at {x}"),
                k::weibull_cdf(x, scale * std::f64::consts::SQRT_2, 2.0),
                k::rayleigh_cdf(x, scale),
                1e-12,
            );
        }
    }
    f.finish("weibull reductions");
}

/// A Student's t with one degree of freedom is a standard Cauchy.
///
/// Completely different machinery on each side — a regularized incomplete beta
/// against an arctangent — which is why this catches a mis-parameterized `t`
/// that every self-consistency check would pass.
#[test]
fn a_student_t_with_one_degree_of_freedom_is_a_cauchy() {
    let mut f = Failures::default();
    for x in [-1e6_f64, -100.0, -3.0, -0.5, 0.0, 0.5, 3.0, 100.0, 1e6] {
        f.close(
            || format!("students_t_cdf(1) vs cauchy_cdf at {x}"),
            k::students_t_cdf(x, 0.0, 1.0, 1.0),
            k::cauchy_cdf(x, 0.0, 1.0),
            1e-11,
        );
        f.close(
            || format!("students_t_ln_sf(1) vs cauchy_ln_sf at {x}"),
            k::students_t_ln_sf(x, 0.0, 1.0, 1.0),
            k::cauchy_ln_sf(x, 0.0, 1.0),
            1e-9,
        );
    }
    // And the location-scale form is the general Cauchy.
    for x in [-20.0_f64, -1.0, 2.0, 50.0] {
        f.close(
            || format!("students_t_cdf(1, loc 2, scale 3) vs cauchy at {x}"),
            k::students_t_cdf(x, 2.0, 3.0, 1.0),
            k::cauchy_cdf(x, 2.0, 3.0),
            1e-11,
        );
    }
    f.finish("t(1) = Cauchy");
}

/// A log-normal is a normal on `ln x`, and a Lomax is a Pareto shifted to the
/// origin.
#[test]
fn the_shifted_and_transformed_families_match_their_sources() {
    let mut f = Failures::default();
    for (mu, sigma) in [(0.0_f64, 1.0_f64), (-1.5, 0.4), (2.0, 2.5)] {
        for x in POINTS {
            f.close(
                || format!("lognormal_cdf({mu},{sigma}) vs normal at ln {x}"),
                k::lognormal_cdf(x, mu, sigma),
                k::normal_cdf(x.ln(), mu, sigma),
                1e-12,
            );
            f.close(
                || format!("lognormal_ln_sf({mu},{sigma}) vs normal at ln {x}"),
                k::lognormal_ln_sf(x, mu, sigma),
                k::normal_ln_sf(x.ln(), mu, sigma),
                1e-12,
            );
        }
    }
    // Lomax(shape, scale) is Pareto(scale, shape) shifted left by `scale`:
    // `P(L > x) = P(Par > x + scale)`.
    for (shape, scale) in [(1.5_f64, 1.0_f64), (3.0, 2.0), (0.5, 4.0)] {
        for x in POINTS {
            f.close(
                || format!("lomax_ln_sf({shape},{scale}) vs shifted pareto at {x}"),
                k::lomax_ln_sf(x, shape, scale),
                k::pareto_ln_sf(x + scale, scale, shape),
                1e-12,
            );
        }
    }
    f.finish("transformed families");
}

/// A Frechet is the reciprocal of a Weibull: `P(F > x) = P(W < 1/x)` with the
/// scale inverted.
///
/// If `X` is Frechet with shape `α` and scale `s`, then `1/X` is Weibull with
/// shape `α` and scale `1/s`, so `F_Frechet(x) = 1 − F_Weibull(1/x)` — which
/// is to say the Frechet CDF equals the Weibull **survival** at the reciprocal
/// point.
#[test]
fn a_frechet_is_a_reciprocal_weibull() {
    let mut f = Failures::default();
    for (shape, scale) in [(1.0_f64, 1.0_f64), (2.5, 3.0), (0.7, 0.5)] {
        for x in [0.05_f64, 0.3, 1.0, 4.0, 50.0] {
            let frechet = k::frechet_cdf(x, scale, shape);
            let weibull_survival = k::weibull_ln_sf(1.0 / x, 1.0 / scale, shape).exp();
            f.close(
                || format!("frechet_cdf({shape},{scale}) vs weibull sf at 1/{x}"),
                frechet,
                weibull_survival,
                1e-11,
            );
        }
    }
    f.finish("Frechet = reciprocal Weibull");
}

/// The two regularized incomplete gammas are complements, and the regularized
/// incomplete beta is symmetric under swapping both its shapes and its
/// argument.
///
/// These are properties of the special functions the kernels are built on
/// rather than of the kernels themselves — but a family that reads the wrong
/// one of a complementary pair is a real failure mode, and this pins which is
/// which. `InverseGamma` genuinely needs the **lower** function for its
/// survival because its CDF is the upper one.
#[test]
fn the_special_functions_satisfy_their_defining_relations() {
    use euclid_numeric::Real;
    let mut f = Failures::default();
    for shape in [0.5_f64, 1.0, 3.0, 20.0] {
        for z in [1e-4_f64, 0.1, 1.0, 5.0, 50.0] {
            let total = z.gamma_lr(shape) + z.gamma_ur(shape);
            f.close(
                || format!("gamma_lr + gamma_ur at shape {shape}, z {z}"),
                total,
                1.0,
                1e-13,
            );
        }
    }
    for (a, b) in [(0.5_f64, 0.5_f64), (2.0, 3.0), (10.0, 0.5)] {
        for y in [1e-6_f64, 0.25, 0.5, 0.75, 1.0 - 1e-6] {
            let direct = y.beta_reg(a, b);
            let reflected = 1.0 - (1.0 - y).beta_reg(b, a);
            // The **reflected** side is the blunt one. It subtracts the answer
            // from 1, so it can only carry the answer down to one ulp of 1 — a
            // relative accuracy of `eps / direct`. That is exactly why
            // `fisher_snedecor_cdf_on_support` switches branch at the midpoint
            // instead of reflecting everywhere, and the bound states the cost
            // rather than hiding it: at `direct = 1.9e-7` the reflection is
            // good to about `1e-9`, and at `1.8e-61` it carries nothing at all.
            // `direct` is the reference, not `reflected` — the accurate side
            // belongs in the denominator. With them the other way round the
            // `beta_reg(10, 0.5)` case divides by a reflected value that has
            // underflowed to exactly zero, which reports `8e246` instead of the
            // true answer, which is that the reflection has lost everything.
            let bound = (4.0 * f64::EPSILON / direct.max(f64::MIN_POSITIVE)).max(1e-10);
            f.check(rel(reflected, direct) <= bound, || {
                format!(
                    "beta_reg({a},{b}) at {y} = {direct:.12e}, reflected = {reflected:.12e}, \
                     bound {bound:.2e}"
                )
            });
        }
    }
    f.finish("special function relations");
}
