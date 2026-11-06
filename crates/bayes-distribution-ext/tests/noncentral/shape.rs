//! Against itself: monotone, differentiable, invertible.
//!
//! None of these compares to an outside answer, so none of them can catch a
//! family that is consistently wrong — that is [`super::reduction`]'s and
//! [`super::definition`]'s job. What they catch is the series misbehaving
//! somewhere its average behaviour looks fine: a term that goes negative, a
//! density that does not integrate to the distribution function it came from, a
//! quantile that ignores its argument.

use bayes_distribution_ext::{
    Continuous, ContinuousCdf, NoncentralChiSquared, NoncentralF, NoncentralT,
};
use bayes_utils::{Failures, derivative};

/// A named CDF, with the bracket chosen for that family's own tails.
type CdfCase = (String, Box<dyn Fn(f64) -> f64>, f64, f64);

/// Monotone, inside `[0, 1]`, and negligible outside `[lo, hi]`.
///
/// The brackets are **per case**, because the three families do not have
/// comparable tails and a shared one would assert something false about at
/// least one of them. `NC-t(6,-3)` is the extreme: a t with 6 degrees of
/// freedom has a lower tail going like `|x|^-6`, so `cdf(-20)` is still
/// `1.5e-4` and the bracket has to reach -600 before the mass outside it is
/// negligible. `NC-chi2(1,100)` is finished by 300.
///
/// Picking one convenient bracket for all five is how this test first "found a
/// bug" in a perfectly correct `NC-F`: at the original `hi` of 60, the `x^-6`
/// tail of an F with 12 denominator degrees of freedom is a genuine `8.7e-6`,
/// which the library reproduced to ten significant figures.
#[test]
fn the_cdfs_are_proper_distribution_functions() {
    const TAIL: f64 = 1e-8;
    let mut f = Failures::default();
    let cases: Vec<CdfCase> = vec![
        {
            let d = NoncentralChiSquared::new(3.0, 5.0).unwrap();
            (
                "NC-chi2(3,5)".into(),
                Box::new(move |x| d.cdf(x)),
                0.0,
                80.0,
            )
        },
        {
            let d = NoncentralChiSquared::new(1.0, 100.0).unwrap();
            (
                "NC-chi2(1,100)".into(),
                Box::new(move |x| d.cdf(x)),
                0.0,
                300.0,
            )
        },
        {
            let d = NoncentralF::new(4.0, 12.0, 8.0).unwrap();
            (
                "NC-F(4,12,8)".into(),
                Box::new(move |x| d.cdf(x)),
                0.0,
                300.0,
            )
        },
        {
            let d = NoncentralT::new(10.0, 2.5).unwrap();
            (
                "NC-t(10,2.5)".into(),
                Box::new(move |x| d.cdf(x)),
                -40.0,
                40.0,
            )
        },
        {
            let d = NoncentralT::new(6.0, -3.0).unwrap();
            (
                "NC-t(6,-3)".into(),
                Box::new(move |x| d.cdf(x)),
                -600.0,
                60.0,
            )
        },
    ];
    for (name, cdf, lo, hi) in cases {
        let mut prev = -1.0;
        for i in 0..=400 {
            let x = lo + (hi - lo) * i as f64 / 400.0;
            let c = cdf(x);
            f.check(c.is_finite() && (0.0..=1.0).contains(&c), || {
                format!("{name}: cdf({x:.4}) = {c} is not a probability")
            });
            f.check(c >= prev - 1e-10, || {
                format!("{name}: cdf decreased at {x:.4}: {prev} -> {c}")
            });
            prev = c;
        }
        // Both of these are real assertions about where the mass is. The low-end
        // one used to read `cdf(lo).abs() < 1e-9 || lo > f64::NEG_INFINITY`,
        // whose second clause is true for every finite `lo` — so it passed
        // unconditionally and checked nothing at all.
        f.check(cdf(lo) < TAIL, || {
            format!(
                "{name}: cdf at the low end is {}, expected below {TAIL}",
                cdf(lo)
            )
        });
        f.check(1.0 - cdf(hi) < TAIL, || {
            format!(
                "{name}: cdf at the high end is {}, expected within {TAIL} of 1",
                cdf(hi)
            )
        });
    }
    f.finish("cdf shape");
}

/// `pdf` against a central difference of `cdf` — two independent routes, since
/// the chi-squared and F densities are their own series while the t density
/// comes from a CDF recurrence.
#[test]
fn the_densities_are_the_derivatives_of_the_cdfs() {
    let mut f = Failures::default();

    for (k, l) in [(3.0, 2.0), (1.0, 10.0), (8.0, 40.0)] {
        let d = NoncentralChiSquared::new(k, l).unwrap();
        for p in [0.2, 0.4, 0.6, 0.8] {
            let x = d.inverse_cdf(p);
            let h = (d.inverse_cdf(p + 0.05) - d.inverse_cdf(p - 0.05)) * 1e-3;
            let numeric = derivative(|t| d.cdf(t), x, h);
            f.close(
                || format!("NC-chi2({k},{l}) pdf vs d(cdf) at p={p}"),
                d.pdf(x),
                numeric,
                1e-5,
            );
        }
    }

    for (d1, d2, l) in [(4.0, 12.0, 6.0), (2.0, 20.0, 1.0)] {
        let d = NoncentralF::new(d1, d2, l).unwrap();
        for p in [0.3, 0.5, 0.7] {
            let x = d.inverse_cdf(p);
            let h = (d.inverse_cdf(p + 0.05) - d.inverse_cdf(p - 0.05)) * 1e-3;
            let numeric = derivative(|t| d.cdf(t), x, h);
            f.close(
                || format!("NC-F({d1},{d2},{l}) pdf vs d(cdf) at p={p}"),
                d.pdf(x),
                numeric,
                1e-4,
            );
        }
    }

    for (nu, delta) in [(10.0, 2.0), (25.0, -1.5), (6.0, 3.0)] {
        let d = NoncentralT::new(nu, delta).unwrap();
        for p in [0.25, 0.5, 0.75] {
            let x = d.inverse_cdf(p);
            let h = (d.inverse_cdf(p + 0.05) - d.inverse_cdf(p - 0.05)) * 1e-3;
            let numeric = derivative(|t| d.cdf(t), x, h);
            f.close(
                || format!("NC-t({nu},{delta}) pdf vs d(cdf) at p={p}"),
                d.pdf(x),
                numeric,
                1e-4,
            );
        }
    }

    f.finish("pdf = d(cdf)/dx");
}

/// `cdf(inverse_cdf(p)) == p` across the range where the quantile is a
/// recoverable number.
///
/// The limits are `1e-9` and `1 − 1e-9` deliberately. Past roughly `1 − 1e-15`
/// a CDF is 1 to within an ulp, its quantile is not recoverable by inverting
/// it, and asking anyway returns `2^64` — see the module docs in
/// [`super`](super).
///
/// The bound is **relative to `p`**, because that is the form the error takes.
/// Bisection resolves `x` to a relative precision, and a tail going like
/// `|x|^-ν` turns that into `dp/p = ν·dx/x` — so the absolute error at
/// `p = 1e-9` is near `1e-17` while at `p = 0.5` it is near `1e-16`, and only a
/// relative bound describes both. A genuinely wrong inversion misses by orders
/// of magnitude, not by parts per million.
#[test]
fn the_quantiles_round_trip_through_the_cdf() {
    const RELATIVE: f64 = 1e-6;
    let mut f = Failures::default();
    let probabilities = [
        1e-9,
        1e-6,
        1e-3,
        0.01,
        0.25,
        0.5,
        0.75,
        0.99,
        1.0 - 1e-6,
        1.0 - 1e-9,
    ];

    for (k, l) in [(3.0, 2.0), (1.0, 100.0), (7.5, 40.0)] {
        let d = NoncentralChiSquared::new(k, l).unwrap();
        for p in probabilities {
            let x = d.inverse_cdf(p);
            f.close(
                || format!("NC-chi2({k},{l}) round trip at p={p}"),
                d.cdf(x),
                p,
                RELATIVE * p,
            );
        }
    }
    for (d1, d2, l) in [(4.0, 12.0, 8.0), (2.0, 20.0, 1.0)] {
        let d = NoncentralF::new(d1, d2, l).unwrap();
        for p in probabilities {
            let x = d.inverse_cdf(p);
            f.close(
                || format!("NC-F({d1},{d2},{l}) round trip at p={p}"),
                d.cdf(x),
                p,
                RELATIVE * p,
            );
        }
    }
    for (nu, delta) in [(10.0, 2.0), (6.0, -3.0)] {
        let d = NoncentralT::new(nu, delta).unwrap();
        for p in probabilities {
            let x = d.inverse_cdf(p);
            f.close(
                || format!("NC-t({nu},{delta}) round trip at p={p}"),
                d.cdf(x),
                p,
                RELATIVE * p,
            );
        }
    }

    f.finish("quantile round trip");
}
