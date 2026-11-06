//! Against the central families, and against each other.
//!
//! Both checks here are **identities**, not approximations, which is why they
//! carry tolerances near machine precision while everything else in this suite
//! is held to a sampling or truncation bound.

use bayes_distribution_ext::{
    ChiSquared, Continuous, ContinuousCdf, FisherSnedecor, Moments, NoncentralChiSquared,
    NoncentralF, NoncentralT, StudentT,
};
use bayes_utils::Failures;

/// At `λ = 0` the mixture collapses onto its `j = 0` term, so each non-central
/// family *is* its central counterpart — exactly, not in the limit.
#[test]
fn zero_non_centrality_reduces_to_the_central_family_exactly() {
    let mut f = Failures::default();

    for k in [1.0, 2.0, 3.0, 7.5, 40.0] {
        let nc = NoncentralChiSquared::new(k, 0.0).unwrap();
        let c = ChiSquared::new(k).unwrap();
        for p in [0.01, 0.1, 0.5, 0.9, 0.99] {
            let x = c.inverse_cdf(p);
            f.close(
                || format!("NC-chi2({k},0) cdf at {x:.6}"),
                nc.cdf(x),
                c.cdf(x),
                1e-12,
            );
            f.close(
                || format!("NC-chi2({k},0) pdf at {x:.6}"),
                nc.pdf(x),
                c.pdf(x),
                1e-11,
            );
        }
        f.close(
            || format!("NC-chi2({k},0) mean"),
            nc.mean(),
            c.mean(),
            1e-14,
        );
        f.close(
            || format!("NC-chi2({k},0) variance"),
            nc.variance(),
            c.variance(),
            1e-14,
        );
        f.close(
            || format!("NC-chi2({k},0) skewness"),
            nc.skewness(),
            c.skewness(),
            1e-12,
        );
    }

    for (d1, d2) in [(1.0, 1.0), (2.0, 5.0), (6.0, 9.0), (10.0, 20.0)] {
        let nc = NoncentralF::new(d1, d2, 0.0).unwrap();
        let c = FisherSnedecor::new(d1, d2).unwrap();
        for x in [0.05, 0.5, 1.0, 2.5, 10.0] {
            f.close(
                || format!("NC-F({d1},{d2},0) cdf at {x}"),
                nc.cdf(x),
                c.cdf(x),
                1e-11,
            );
            f.close(
                || format!("NC-F({d1},{d2},0) pdf at {x}"),
                nc.pdf(x),
                c.pdf(x),
                1e-10,
            );
        }
        if d2 > 2.0 {
            f.close(
                || format!("NC-F({d1},{d2},0) mean"),
                nc.mean(),
                c.mean(),
                1e-12,
            );
        }
    }

    for nu in [1.0, 2.0, 5.0, 30.0] {
        let nc = NoncentralT::new(nu, 0.0).unwrap();
        let c = StudentT::new(nu).unwrap();
        for x in [-4.0, -1.0, -0.25, 0.0, 0.25, 1.0, 4.0] {
            f.close(
                || format!("NC-t({nu},0) cdf at {x}"),
                nc.cdf(x),
                c.cdf(x),
                1e-10,
            );
            f.close(
                || format!("NC-t({nu},0) pdf at {x}"),
                nc.pdf(x),
                c.pdf(x),
                1e-7,
            );
        }
        if nu > 2.0 {
            f.close(
                || format!("NC-t({nu},0) variance"),
                nc.variance(),
                c.variance(),
                1e-10,
            );
        }
    }

    f.finish("exact reduction at zero non-centrality");
}

/// `T² ~ F(1, ν)` extends to the non-central case: a non-central t squared is a
/// non-central F with `λ = δ²`. This ties two of the three families together
/// through entirely separate implementations — Lenth's series on one side, a
/// Poisson mixture of incomplete betas on the other.
#[test]
fn a_squared_non_central_t_is_a_non_central_f() {
    let mut f = Failures::default();
    for (nu, delta) in [(10.0, 1.5), (20.0, 2.0), (5.0, 3.0)] {
        let t = NoncentralT::new(nu, delta).unwrap();
        let fd = NoncentralF::new(1.0, nu, delta * delta).unwrap();
        for x in [0.25_f64, 1.0, 4.0, 9.0, 25.0] {
            // P(T² <= x) = P(-√x <= T <= √x)
            let via_t = t.cdf(x.sqrt()) - t.cdf(-x.sqrt());
            f.close(
                || {
                    format!(
                        "NC-t({nu},{delta})² vs NC-F(1,{nu},{}) at x={x}",
                        delta * delta
                    )
                },
                fd.cdf(x),
                via_t,
                1e-8,
            );
        }
    }
    f.finish("NC-t squared = NC-F");
}
