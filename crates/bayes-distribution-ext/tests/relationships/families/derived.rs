//! Families defined as a **ratio** of others, and closed forms worth naming.
//!
//! `F(1,ν)` is the square of a `t(ν)`, and `F(2a,2b)` is a rescaled odds-ratio
//! of a `Beta(a,b)` — two relations that between them constrain the F
//! distribution from both of its parents. The closed-form checks are the
//! remainder: a logistic CDF must be exactly the sigmoid, and the triangular
//! and uniform moments are exact rationals that can be written down.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use crate::{Failures, quantile_grid};
use bayes_distribution_ext::{ContinuousCdf, Moments, *};

const TOL: f64 = 1e-11;

#[test]
fn fisher_snedecor_relates_to_both_student_t_and_beta() {
    let mut f = Failures::default();

    // T ~ t(ν) implies T² ~ F(1, ν), so cdf_F(x) = 2·cdf_t(√x) − 1.
    for nu in [1.0, 3.0, 10.0, 50.0] {
        let fd = FisherSnedecor::new(1.0, nu).unwrap();
        let t = StudentT::new(nu).unwrap();
        for x in [0.05, 0.5, 1.0, 2.0, 10.0, 100.0] {
            f.close(
                || format!("F(1,{nu}) cdf = 2·t({nu}) cdf(√x) − 1 at {x}"),
                fd.cdf(x),
                2.0 * t.cdf(x.sqrt()) - 1.0,
                1e-9,
            );
        }
    }

    // X ~ Beta(a, b) implies (b/a)·X/(1−X) ~ F(2a, 2b), so
    // cdf_F(y) = cdf_Beta(a·y / (a·y + b)).
    for (a, b) in [(1.0, 1.0), (2.0, 3.0), (0.5, 4.0), (10.0, 10.0)] {
        let fd = FisherSnedecor::new(2.0 * a, 2.0 * b).unwrap();
        let beta = Beta::new(a, b).unwrap();
        for y in [0.01, 0.1, 1.0, 3.0, 25.0] {
            let x = a * y / (a * y + b);
            f.close(
                || {
                    format!(
                        "F({},{}) cdf({y}) = Beta({a},{b}) cdf({x:.6})",
                        2.0 * a,
                        2.0 * b
                    )
                },
                fd.cdf(y),
                beta.cdf(x),
                1e-9,
            );
        }
    }

    f.finish("FisherSnedecor relations");
}

#[test]
fn logistic_and_uniform_and_triangular_match_their_closed_forms() {
    let mut f = Failures::default();

    // The logistic CDF is the sigmoid; this pins the scale convention.
    for (mu, s) in [(0.0, 1.0), (-1.0, 0.3), (4.0, 2.0)] {
        let l = Logistic::new(mu, s).unwrap();
        for x in quantile_grid(|p| l.inverse_cdf(p)) {
            let sigmoid = 1.0 / (1.0 + (-(x - mu) / s).exp());
            f.close(
                || format!("Logistic({mu},{s}) cdf = sigmoid at {x:.6e}"),
                l.cdf(x),
                sigmoid,
                TOL,
            );
        }
        f.close(
            || format!("Logistic({mu},{s}) variance = s²π²/3"),
            l.variance(),
            s * s * std::f64::consts::PI.powi(2) / 3.0,
            TOL,
        );
    }

    // A symmetric triangular has mean and median at the midpoint, and the
    // variance of a triangular is (a²+b²+c²−ab−ac−bc)/18.
    let (a, b, c) = (0.0, 4.0, 2.0);
    let t = Triangular::new(a, b, c).unwrap();
    f.close(
        || "Triangular(0,4,2) mean".into(),
        t.mean(),
        (a + b + c) / 3.0,
        TOL,
    );
    f.close(
        || "Triangular(0,4,2) variance".into(),
        t.variance(),
        (a * a + b * b + c * c - a * b - a * c - b * c) / 18.0,
        TOL,
    );
    f.close(
        || "Triangular(0,4,2) cdf at the mode".into(),
        t.cdf(c),
        (c - a) / (b - a),
        TOL,
    );

    // Uniform's moments are exact rationals.
    let u = Uniform::new(-3.0, 5.0).unwrap();
    f.close(|| "Uniform(-3,5) mean".into(), u.mean(), 1.0, TOL);
    f.close(
        || "Uniform(-3,5) variance".into(),
        u.variance(),
        64.0 / 12.0,
        TOL,
    );
    f.close(
        || "Uniform(-3,5) entropy".into(),
        u.entropy(),
        8.0_f64.ln(),
        TOL,
    );

    f.finish("closed-form checks");
}
