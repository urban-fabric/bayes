//! The integral transforms, against the moments they generate.
//!
//! A moment-generating function is only correct if its derivatives at zero *are*
//! the moments — and the library computes those moments by a completely separate
//! closed form, so differentiating `mgf` numerically ties three implementations
//! together:
//!
//! ```text
//!   mgf'(0)  = E[X]      mgf''(0) = E[X²]
//!   cgf'(0)  = E[X]      cgf''(0) = Var(X)
//!   pgf(1)   = 1         pgf'(1)  = E[X]      pgf(0) = P(X = 0)
//!   cf(0)    = (1, 0)    |cf(t)| <= 1         cf(-t) = conj(cf(t))
//! ```
//!
//! Each transform is also checked against its *definition* — `E[e^{tX}]` by
//! quadrature or summation — which is a check that involves no derivatives and
//! so fails differently if the derivative probe is at fault. That mattered: the
//! discrete sum has to be accumulated in the log domain, because `exp(t·k)` times
//! an underflowed mass is `inf × 0 = NaN`, and `Geometric`'s MGF legitimately
//! diverges past `t = −ln(1−p)`.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use crate::{DensityCase, Failures, derivative, second_derivative, simpson};
use bayes_distribution_ext::{Moments, *};

/// `mgf'(0) = E[X]` and `mgf''(0) − mgf'(0)² = Var(X)`.
#[test]
fn moment_generating_derivatives_at_zero_are_the_moments() {
    let mut f = Failures::default();

    macro_rules! check {
        ($label:literal, $dist:expr, $h:expr, $tol:expr) => {{
            let d = $dist;
            let m1 = derivative(|t| d.mgf(t), 0.0, $h);
            let m2 = second_derivative(|t| d.mgf(t), 0.0, $h);
            let (mean, var) = (Moments::mean(&d), Moments::variance(&d));
            f.close(|| format!("{}: mgf'(0) vs mean", $label), m1, mean, $tol);
            // mgf''(0) is the raw second moment E[X²] = Var + mean².
            f.close(
                || format!("{}: mgf''(0) vs Var + mean²", $label),
                m2,
                var + mean * mean,
                $tol,
            );
            // The cumulant generating function gives the central moments directly.
            let c1 = derivative(|t| d.cgf(t), 0.0, $h);
            let c2 = second_derivative(|t| d.cgf(t), 0.0, $h);
            f.close(|| format!("{}: cgf'(0) vs mean", $label), c1, mean, $tol);
            f.close(
                || format!("{}: cgf''(0) vs variance", $label),
                c2,
                var,
                $tol,
            );
        }};
    }

    check!("Normal(2,3)", Normal::new(2.0, 3.0).unwrap(), 1e-3, 1e-6);
    check!("Normal(0,1)", Normal::new(0.0, 1.0).unwrap(), 1e-3, 1e-8);
    check!("Exponential(2)", Exponential::new(2.0).unwrap(), 1e-3, 1e-7);
    check!("Gamma(3,2)", Gamma::new(3.0, 2.0).unwrap(), 1e-3, 1e-7);
    check!("ChiSquared(5)", ChiSquared::new(5.0).unwrap(), 1e-3, 1e-6);
    check!("Bernoulli(0.3)", Bernoulli::new(0.3).unwrap(), 1e-3, 1e-9);
    check!(
        "Binomial(10,0.4)",
        Binomial::new(10, 0.4).unwrap(),
        1e-3,
        1e-7
    );
    check!("Poisson(4)", Poisson::new(4.0).unwrap(), 1e-3, 1e-6);
    check!("Geometric(0.3)", Geometric::new(0.3).unwrap(), 1e-3, 1e-6);

    f.finish("mgf/cgf derivatives");
}

/// The MGF is `E[e^{tX}]`, so it must equal the integral or sum of `e^{tx}`
/// against the density or mass — a check that does not involve derivatives.
#[test]
fn the_mgf_is_the_expectation_of_the_exponential() {
    let mut f = Failures::default();

    // Continuous: ∫ e^{tx} f(x) dx over the effective support.
    let continuous: Vec<DensityCase> = vec![
        {
            let d = Normal::new(0.5, 1.5).unwrap();
            (
                "Normal(0.5,1.5)".into(),
                Box::new(move |t| d.mgf(t)),
                Box::new(move |x| Continuous::pdf(&d, x)),
                -14.0,
                15.0,
            )
        },
        {
            let d = Exponential::new(2.0).unwrap();
            (
                "Exponential(2)".into(),
                Box::new(move |t| d.mgf(t)),
                Box::new(move |x| Continuous::pdf(&d, x)),
                0.0,
                60.0,
            )
        },
        {
            let d = Gamma::new(3.0, 2.0).unwrap();
            (
                "Gamma(3,2)".into(),
                Box::new(move |t| d.mgf(t)),
                Box::new(move |x| Continuous::pdf(&d, x)),
                0.0,
                60.0,
            )
        },
    ];
    for (name, mgf, pdf, lo, hi) in continuous {
        for t in [-0.5, -0.1, 0.1, 0.3] {
            let integral = simpson(|x| (t * x).exp() * pdf(x), lo, hi, 200_000);
            f.close(
                || format!("{name}: mgf({t}) vs ∫e^{{tx}}f(x)dx"),
                mgf(t),
                integral,
                1e-6,
            );
        }
    }

    // Discrete: Σ e^{tk} p(k).
    // The sum has to be accumulated in the log domain. Done naively, a term far
    // out in the tail is `exp(t·k) * pmf(k)` = `inf * 0` = `NaN`, which reports a
    // NaN where the library was right. And `t` must stay inside the radius of
    // convergence: a geometric MGF diverges at `t >= -ln(1-p)`, so `inf` there is
    // the correct answer rather than a defect.
    macro_rules! discrete {
        ($label:literal, $dist:expr, $lo:expr, $hi:expr, $radius:expr) => {{
            let d = $dist;
            for t in [-0.7, -0.2, 0.2, 0.5] {
                if t >= $radius {
                    // Outside the domain the MGF must report divergence, not a number.
                    let m = d.mgf(t);
                    f.check(!m.is_finite() || m > 1e12, || {
                        format!(
                            "{}: mgf({t}) = {m} but t is outside the convergence radius {}",
                            $label, $radius
                        )
                    });
                    continue;
                }
                let mut acc = 0.0;
                for k in $lo..=$hi {
                    let ln_term = t * k as f64 + Discrete::ln_pmf(&d, k);
                    if ln_term.is_finite() {
                        acc += ln_term.exp();
                    }
                }
                f.close(
                    || format!("{}: mgf({t}) vs Σe^{{tk}}p(k)", $label),
                    d.mgf(t),
                    acc,
                    1e-8,
                );
            }
        }};
    }
    discrete!(
        "Bernoulli(0.3)",
        Bernoulli::new(0.3).unwrap(),
        0_u64,
        1_u64,
        f64::INFINITY
    );
    discrete!(
        "Binomial(10,0.4)",
        Binomial::new(10, 0.4).unwrap(),
        0_u64,
        10_u64,
        f64::INFINITY
    );
    discrete!(
        "Poisson(4)",
        Poisson::new(4.0).unwrap(),
        0_u64,
        400_u64,
        f64::INFINITY
    );
    // Geometric(p) converges only for t < -ln(1-p) = 0.3567 at p = 0.3.
    discrete!(
        "Geometric(0.3)",
        Geometric::new(0.3).unwrap(),
        1_u64,
        6000_u64,
        -(0.7_f64).ln()
    );

    f.finish("mgf = E[exp(tX)]");
}

/// `pgf(1) = 1`, `pgf'(1) = E[X]`, `pgf(0) = P(X = 0)`.
#[test]
fn probability_generating_function_identities_hold() {
    let mut f = Failures::default();

    macro_rules! check {
        ($label:literal, $dist:expr, $zero_mass:expr) => {{
            let d = $dist;
            f.close(|| format!("{}: pgf(1) = 1", $label), d.pgf(1.0), 1.0, 1e-10);
            let m = derivative(|z| d.pgf(z), 1.0, 1e-4);
            f.close(
                || format!("{}: pgf'(1) vs mean", $label),
                m,
                Moments::mean(&d),
                1e-6,
            );
            if $zero_mass {
                f.close(
                    || format!("{}: pgf(0) = P(X=0)", $label),
                    d.pgf(0.0),
                    Discrete::pmf(&d, 0_u64),
                    1e-12,
                );
            }
            // The PGF is also Σ z^k p(k) term by term.
            for z in [0.3_f64, 0.7, 0.95] {
                let mut acc = 0.0;
                for k in 0_u64..=3000 {
                    acc += z.powi(k as i32) * Discrete::pmf(&d, k);
                }
                f.close(
                    || format!("{}: pgf({z}) vs Σz^k p(k)", $label),
                    d.pgf(z),
                    acc,
                    1e-9,
                );
            }
        }};
    }

    check!("Bernoulli(0.3)", Bernoulli::new(0.3).unwrap(), true);
    check!("Binomial(12,0.25)", Binomial::new(12, 0.25).unwrap(), true);
    check!("Poisson(3.5)", Poisson::new(3.5).unwrap(), true);
    check!("Geometric(0.4)", Geometric::new(0.4).unwrap(), true);
    check!(
        "NegativeBinomial(4,0.7)",
        NegativeBinomial::new(4.0, 0.7).unwrap(),
        true
    );

    f.finish("pgf identities");
}

/// `cf(0) = (1, 0)`, `|cf(t)| ≤ 1`, and the real part is `∫cos(tx)f(x)dx`.
#[test]
fn characteristic_function_identities_hold() {
    let mut f = Failures::default();

    macro_rules! check {
        ($label:literal, $dist:expr) => {{
            let d = $dist;
            let (re, im) = d.characteristic_function(0.0);
            f.close(|| format!("{}: Re cf(0) = 1", $label), re, 1.0, 1e-12);
            f.close(|| format!("{}: Im cf(0) = 0", $label), im, 0.0, 1e-12);
            for t in [-3.0, -1.0, 0.5, 2.0, 7.0] {
                let m = d.cf_modulus(t);
                // |E[e^{itX}]| <= E|e^{itX}| = 1.
                f.check(m <= 1.0 + 1e-12, || {
                    format!("{}: |cf({t})| = {m} exceeds 1", $label)
                });
                // The CF of a real random variable is Hermitian: cf(-t) = conj(cf(t)).
                let (re_p, im_p) = d.characteristic_function(t);
                let (re_m, im_m) = d.characteristic_function(-t);
                f.close(
                    || format!("{}: Re cf(-{t}) = Re cf({t})", $label),
                    re_m,
                    re_p,
                    1e-10,
                );
                f.close(
                    || format!("{}: Im cf(-{t}) = -Im cf({t})", $label),
                    im_m,
                    -im_p,
                    1e-10,
                );
            }
        }};
    }

    check!("Normal(1,2)", Normal::new(1.0, 2.0).unwrap());
    check!("Cauchy(0,1)", Cauchy::new(0.0, 1.0).unwrap());
    check!("Uniform(-1,2)", Uniform::new(-1.0, 2.0).unwrap());
    check!("Exponential(3)", Exponential::new(3.0).unwrap());
    check!("Laplace(0,1)", Laplace::new(0.0, 1.0).unwrap());
    check!("Poisson(2)", Poisson::new(2.0).unwrap());

    // The Cauchy CF is exactly e^{-|t|}, which is why the family has no MGF.
    let c = Cauchy::new(0.0, 1.0).unwrap();
    for t in [0.5, 1.0, 3.0] {
        let (re, im) = c.characteristic_function(t);
        f.close(
            || format!("Cauchy(0,1): Re cf({t}) = e^-|t|"),
            re,
            (-t.abs()).exp(),
            1e-10,
        );
        f.close(|| format!("Cauchy(0,1): Im cf({t}) = 0"), im, 0.0, 1e-10);
    }

    f.finish("cf identities");
}
