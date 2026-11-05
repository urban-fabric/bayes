//! Where the series and the constructors are supposed to break, and where they
//! are not.

use bayes_distribution_ext::{
    ContinuousCdf, Moments, NoncentralChiSquared, NoncentralF, NoncentralT, Normal,
};
use bayes_utils::Failures;

/// At `λ = 5000` the Poisson weights peak near `j = 2500` and the first
/// non-negligible one is well past `j = 2000`, so a summation that began at
/// `j = 0` and gave up early would return nonsense here while looking perfect
/// at `λ = 2`. This is the test that justifies summing outward from the mode.
///
/// Every check is a CDF evaluation within two standard deviations of the mean,
/// which is the range where a CDF carries information.
#[test]
fn large_non_centrality_stays_accurate() {
    let mut f = Failures::default();
    let z = Normal::new(0.0, 1.0).unwrap();

    for l in [100.0, 500.0, 1000.0, 5000.0] {
        let d = NoncentralChiSquared::new(4.0, l).unwrap();
        f.check(d.cdf(0.0) == 0.0, || {
            format!("NC-chi2(4,{l}) cdf(0) is {}", d.cdf(0.0))
        });
        f.check(d.cdf(f64::INFINITY) == 1.0, || {
            format!("NC-chi2(4,{l}) cdf(inf) is {}", d.cdf(f64::INFINITY))
        });

        // A large non-centrality drives the family to the normal, and the
        // leading error in the standardized CDF is the Edgeworth term
        // `(γ/6)(1 − z²)φ(z)`, which is at most `0.4·γ/6` over `z`. The skewness
        // `γ = 2^{3/2}(k + 3λ)/(k + 2λ)^{3/2}` is about `2.1/√λ`, so this bound
        // tightens as `√λ` on its own — it is a consequence of the expansion,
        // not a number chosen to make the suite pass. The factor of four is
        // headroom for the next order, which falls off as `1/λ`.
        let tolerance = 4.0 * 0.4 * d.skewness() / 6.0;
        let (mean, sd) = (d.mean(), d.std_dev());
        for quantile in [-2.0, -1.0, 0.0, 1.0, 2.0] {
            f.close(
                || format!("NC-chi2(4,{l}) standardized cdf at z={quantile}"),
                d.cdf(mean + quantile * sd),
                z.cdf(quantile),
                tolerance,
            );
        }
    }
    f.finish("large non-centrality");
}

#[test]
fn constructors_reject_invalid_parameters() {
    let mut f = Failures::default();
    for (label, rejected) in [
        (
            "NC-chi2(0, 1)",
            NoncentralChiSquared::new(0.0, 1.0).is_err(),
        ),
        (
            "NC-chi2(-1, 1)",
            NoncentralChiSquared::new(-1.0, 1.0).is_err(),
        ),
        (
            "NC-chi2(3, -1)",
            NoncentralChiSquared::new(3.0, -1.0).is_err(),
        ),
        (
            "NC-chi2(3, NaN)",
            NoncentralChiSquared::new(3.0, f64::NAN).is_err(),
        ),
        (
            "NC-chi2(3, inf)",
            NoncentralChiSquared::new(3.0, f64::INFINITY).is_err(),
        ),
        ("NC-F(0, 1, 1)", NoncentralF::new(0.0, 1.0, 1.0).is_err()),
        ("NC-F(1, 0, 1)", NoncentralF::new(1.0, 0.0, 1.0).is_err()),
        ("NC-F(1, 1, -1)", NoncentralF::new(1.0, 1.0, -1.0).is_err()),
        ("NC-t(0, 1)", NoncentralT::new(0.0, 1.0).is_err()),
        ("NC-t(5, NaN)", NoncentralT::new(5.0, f64::NAN).is_err()),
        (
            "NC-t(5, inf)",
            NoncentralT::new(5.0, f64::INFINITY).is_err(),
        ),
    ] {
        f.check(rejected, || format!("{label} was accepted"));
    }
    // Zero non-centrality is legitimate: it is the central family.
    f.check(NoncentralChiSquared::new(3.0, 0.0).is_ok(), || {
        "zero non-centrality should be accepted".into()
    });
    // A negative non-centrality is fine for t: it shifts the other way.
    f.check(NoncentralT::new(5.0, -2.0).is_ok(), || {
        "a negative delta should be accepted".into()
    });
    f.finish("parameter validation");
}

/// A non-central t is the one family here whose argument can reach the edge of
/// `f64`, and `t²` overflows above `√f64::MAX ≈ 1.34e154`. That used to make
/// the ratio `t²/(t² + ν)` a `NaN` and the whole CDF with it.
///
/// Past the overflow the limit is returned directly, so the answer is exactly
/// `1` or `0`. Below it the answer comes from Lenth's series, which lands
/// **one ulp short of 1** and stays there: the series accumulates `Φ(−δ)` and a
/// sum that converges to `Φ(δ)`, and the two do not add to exactly 1 in `f64`.
///
/// That plateau is a known limit, asserted here rather than hidden. It is why
/// nothing in this suite inverts a non-central t CDF for a `p` within an ulp
/// of 1. The χ² and F families do not have it, because their mixture is
/// renormalized by the weight it actually summed — so every term being `1.0`
/// makes the total exactly `1.0`. The equivalent for this family is to sum the
/// *complement* through the reflected incomplete beta `I_{1−y}(ν/2, j+½)`, the
/// way `NoncentralF::cdf` already does above its midpoint.
#[test]
fn an_enormous_argument_saturates_instead_of_going_nan() {
    const PLATEAU: f64 = 1e-15;
    let mut f = Failures::default();
    for (nu, delta) in [(20.0, 1.5), (6.0, -3.0), (1.0, 0.0)] {
        let d = NoncentralT::new(nu, delta).unwrap();
        for x in [1e100, 1e154, 1e200, 1e300, f64::MAX] {
            let (upper, lower) = (d.cdf(x), d.cdf(-x));
            f.check(upper.is_finite() && lower.is_finite(), || {
                format!("NC-t({nu},{delta}) at {x:e} gave {upper} and {lower}")
            });
            f.check(1.0 - upper < PLATEAU, || {
                format!("NC-t({nu},{delta}) cdf({x:e}) is {upper}, expected 1")
            });
            f.check(lower < PLATEAU, || {
                format!("NC-t({nu},{delta}) cdf(-{x:e}) is {lower}, expected 0")
            });
            if x * x == f64::INFINITY {
                f.check(upper == 1.0 && lower == 0.0, || {
                    format!(
                        "NC-t({nu},{delta}) past the overflow of t² should saturate exactly, \
                         got {upper} and {lower}"
                    )
                });
            }
        }
    }
    f.finish("non-central t at the edge of f64");
}
