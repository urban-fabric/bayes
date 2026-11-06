//! Does a tail probability round trip back to the point it came from?

use super::{bounded_above, rel};
use bayes_utils::Failures;
use bayes_utils::families::{ContCase, continuous_cases};

/// Tail probabilities spanning the whole representable range, not just the
/// first few orders of magnitude.
const TAIL_PROBABILITIES: [f64; 10] = [
    1e-2, 1e-4, 1e-8, 1e-16, 1e-30, 1e-60, 1e-100, 1e-180, 1e-250, 1e-300,
];

/// What a round trip through `1 − p` can possibly achieve.
///
/// The generic `isf(p) = inverse_cdf(1 − p)` default rounds `1 − p` to the
/// nearest `f64` below 1, an absolute error of up to `ε/2`. The survival of the
/// point that comes back is therefore `p ± ε/2`, a **relative** error of about
/// `ε/p` — which is 1 at `p = 2e-16` and worse below it.
///
/// So this is not a tuned tolerance: it is the arithmetic cost of the
/// subtraction, and a family that pays it cannot do better. The families that
/// avoid the subtraction entirely are held to `1e-9` instead, by
/// [`an_exact_inverse_survival_round_trips_at_every_scale`].
fn subtraction_floor(p: f64) -> f64 {
    (4.0 * f64::EPSILON / p).max(1e-9)
}

/// Families whose own quartiles are indistinguishable in `f64`.
///
/// `Normal(1e12, 1e-6)` is in the canonical sweep precisely to be this case:
/// one ulp at `1e12` is `1.2e-4`, over a hundred times the scale, so *every*
/// quantile rounds to the mean. There is no quantile to check, and reporting
/// the format's resolution as a library defect is the mistake the sweep's own
/// documentation warns about.
fn scale_below_one_ulp(case: &ContCase) -> bool {
    case.dist.inverse_cdf(0.25) == case.dist.inverse_cdf(0.75)
}

/// `sf(isf(p)) == p`, for every family, to the accuracy its `isf` route allows.
///
/// Each half is a different computation — `isf` is a closed form or a
/// reflection, `sf` comes from a log-survival built out of `erfcx`,
/// `ln_1m_exp` or a reflected incomplete beta — and the target `p` is known
/// exactly in advance rather than being whatever the library said last time.
/// That is what makes composing them informative where testing either alone is
/// not.
#[test]
fn a_tail_probability_round_trips_through_the_inverse_survival() {
    let mut f = Failures::default();
    for case in continuous_cases() {
        if bounded_above(&case.name) || scale_below_one_ulp(&case) {
            continue;
        }
        for p in TAIL_PROBABILITIES {
            let x = case.dist.isf(p);
            // An `isf` that cannot reach this `p` returns the support bound.
            // That honest refusal is caught by the survival suite, not here.
            if !x.is_finite() {
                continue;
            }
            let back = case.dist.sf(x);
            let bound = subtraction_floor(p);
            f.check(rel(back, p) <= bound, || {
                format!(
                    "{}: isf({p:.1e}) = {x:.8e}, whose sf is {back:.6e} — relative error \
                     {:.2e} against a subtraction floor of {bound:.2e}",
                    case.name,
                    rel(back, p)
                )
            });
        }
    }
    f.finish("tail round trip through isf");
}

/// The families that avoid `1 − p` must round trip to `1e-9` at **every**
/// scale, down to `p = 1e-300`.
///
/// Without this the test above is satisfied by doing nothing: its bound widens
/// as `p` shrinks, so a family could pay the subtraction cost forever and never
/// be noticed. This one names the families that should not pay it, and holds
/// them to a bound that does not move.
///
/// Two routes qualify. A **closed form** takes `p` as its own argument
/// (`−ln(p)/λ`, `σ√2·erfc_inv(p)`). A **reflection** uses the exact symmetry
/// `isf(p) = 2·location − Q(p)`, which is only as good as `Q` in the lower
/// tail — and that is the other half of what this checks, since `Q` had to stop
/// forming `1 − p` too.
#[test]
fn an_exact_inverse_survival_round_trips_at_every_scale() {
    use bayes_distribution_ext::{
        Cauchy, ContinuousCdf, Exponential, HalfNormal, Laplace, Logistic, Normal, Pareto,
        Rayleigh, Weibull,
    };
    /// A named family, its inverse survival, and its survival.
    type ExactCase = (
        &'static str,
        Box<dyn Fn(f64) -> f64>,
        Box<dyn Fn(f64) -> f64>,
    );
    let cases: Vec<ExactCase> = vec![
        {
            let d = Exponential::new(2.5).unwrap();
            (
                "Exponential(2.5)",
                Box::new(move |p| d.isf(p)),
                Box::new(move |x| d.sf(x)),
            )
        },
        {
            let d = Weibull::new(3.0, 1.7).unwrap();
            (
                "Weibull(3,1.7)",
                Box::new(move |p| d.isf(p)),
                Box::new(move |x| d.sf(x)),
            )
        },
        {
            let d = Pareto::new(2.0, 1.3).unwrap();
            (
                "Pareto(2,1.3)",
                Box::new(move |p| d.isf(p)),
                Box::new(move |x| d.sf(x)),
            )
        },
        {
            let d = Rayleigh::new(1.4).unwrap();
            (
                "Rayleigh(1.4)",
                Box::new(move |p| d.isf(p)),
                Box::new(move |x| d.sf(x)),
            )
        },
        {
            let d = HalfNormal::new(2.0).unwrap();
            (
                "HalfNormal(2)",
                Box::new(move |p| d.isf(p)),
                Box::new(move |x| d.sf(x)),
            )
        },
        {
            let d = Normal::new(1.0, 2.0).unwrap();
            (
                "Normal(1,2)",
                Box::new(move |p| d.isf(p)),
                Box::new(move |x| d.sf(x)),
            )
        },
        {
            let d = Cauchy::new(0.5, 1.5).unwrap();
            (
                "Cauchy(0.5,1.5)",
                Box::new(move |p| d.isf(p)),
                Box::new(move |x| d.sf(x)),
            )
        },
        {
            let d = Laplace::new(-1.0, 0.5).unwrap();
            (
                "Laplace(-1,0.5)",
                Box::new(move |p| d.isf(p)),
                Box::new(move |x| d.sf(x)),
            )
        },
        {
            let d = Logistic::new(2.0, 0.8).unwrap();
            (
                "Logistic(2,0.8)",
                Box::new(move |p| d.isf(p)),
                Box::new(move |x| d.sf(x)),
            )
        },
    ];

    let mut f = Failures::default();
    for (name, isf, sf) in cases {
        for p in TAIL_PROBABILITIES {
            let x = isf(p);
            f.check(x.is_finite(), || {
                format!("{name}: isf({p:.1e}) = {x}, which should be a finite point")
            });
            if !x.is_finite() {
                continue;
            }
            let back = sf(x);
            f.check(rel(back, p) < 1e-9, || {
                format!(
                    "{name}: isf({p:.1e}) = {x:.10e}, whose sf is {back:.10e} — relative \
                     error {:.2e}",
                    rel(back, p)
                )
            });
        }
    }
    f.finish("exact inverse survival at every scale");
}

/// `cdf(inverse_cdf(p)) == p` deep into the **lower** tail.
///
/// The lower tail is the recoverable one: `p = 1e-300` is an ordinary `f64`, so
/// a quantile has no excuse for losing it — unlike `1 − 1e-300`, which is
/// simply `1`. This is the check that caught six quantile kernels forming
/// `ln(1 − p)` where `ln_1p(−p)` was meant, each of which quantised its lower
/// tail into steps of `1.1e-16`.
#[test]
fn a_lower_tail_probability_round_trips_through_the_quantile() {
    let mut f = Failures::default();
    for case in continuous_cases() {
        if scale_below_one_ulp(&case) {
            continue;
        }
        for p in TAIL_PROBABILITIES {
            let x = case.dist.inverse_cdf(p);
            if !x.is_finite() || x <= case.dist.lo() {
                continue;
            }
            // A quantile that has underflowed into the subnormals is not a
            // recoverable number: `Beta(0.5, 0.5)` has `Q(1e-180) ≈ 2.5e-360`,
            // which is simply not an `f64`, and the library correctly reports
            // the smallest one it has.
            if x != 0.0 && x.abs() < 1e-300 {
                continue;
            }
            let back = case.dist.cdf(x);
            if back == 0.0 {
                continue;
            }
            f.check(rel(back, p) <= subtraction_floor(p), || {
                format!(
                    "{}: inverse_cdf({p:.1e}) = {x:.8e}, whose cdf is {back:.6e} — relative \
                     error {:.2e}",
                    case.name,
                    rel(back, p)
                )
            });
        }
    }
    f.finish("lower tail round trip through the quantile");
}

/// Both inverses must be monotone across the range they accept.
///
/// A quantile that has collapsed onto one value shows up here as a flat spot —
/// which is what `erf_inv(2p − 1)` produced for every `p` below `1e-16`, and
/// what the bisection kernel's own documentation records happening to
/// `Gamma(1e-3, 1)` before it searched an exponential coordinate.
#[test]
fn both_inverses_are_monotone_into_the_tail() {
    let mut f = Failures::default();
    for case in continuous_cases() {
        let mut previous = f64::NEG_INFINITY;
        for p in TAIL_PROBABILITIES.iter().rev() {
            let x = case.dist.inverse_cdf(*p);
            if !x.is_finite() {
                continue;
            }
            f.check(x >= previous, || {
                format!(
                    "{}: inverse_cdf fell as p rose to {p:.1e}: {previous:.6e} -> {x:.6e}",
                    case.name
                )
            });
            previous = x;
        }
        if bounded_above(&case.name) {
            continue;
        }
        let mut previous = f64::NEG_INFINITY;
        for p in TAIL_PROBABILITIES {
            let x = case.dist.isf(p);
            if !x.is_finite() {
                continue;
            }
            f.check(x >= previous, || {
                format!(
                    "{}: isf fell as p fell to {p:.1e}: {previous:.6e} -> {x:.6e}",
                    case.name
                )
            });
            previous = x;
        }
    }
    f.finish("inverse monotonicity into the tail");
}

/// A standard normal's tail against values that are not the library's own.
///
/// Everything else in this suite is self-consistency, which cannot catch a
/// family that is wrong the same way twice. These are independent: `Φ(−z)` at
/// integer sigma, to twelve significant figures.
#[test]
fn the_standard_normal_tail_matches_published_values() {
    use bayes_distribution_ext::{ContinuousCdf, Normal};
    let n = Normal::new(0.0, 1.0).unwrap();
    let reference = [
        (1.0, 1.586_552_539_315e-1),
        (2.0, 2.275_013_194_818e-2),
        (3.0, 1.349_898_031_630e-3),
        (4.0, 3.167_124_183_312e-5),
        (5.0, 2.866_515_718_792e-7),
        (6.0, 9.865_876_450_377e-10),
        (7.0, 1.279_812_543_886e-12),
        (8.0, 6.220_960_574_148e-16),
        (9.0, 1.128_588_405_6e-19),
        (10.0, 7.619_853_024_16e-24),
    ];
    let mut f = Failures::default();
    for (z, truth) in reference {
        f.check(rel(n.sf(z), truth) < 1e-9, || {
            format!("sf({z}) = {:.12e}, published {truth:.12e}", n.sf(z))
        });
        f.check(rel(n.cdf(-z), truth) < 1e-9, || {
            format!("cdf(-{z}) = {:.12e}, published {truth:.12e}", n.cdf(-z))
        });
        // And the quantile inverts it. This is where the 4% error lived.
        f.check(rel(n.isf(truth), z) < 1e-6, || {
            format!("isf({truth:.6e}) = {:.12}, expected {z}", n.isf(truth))
        });
    }
    f.finish("standard normal tail vs published values");
}

/// A Student's t quantile reaches `p = 1e-60`, and **refuses** past it.
///
/// Both halves of that sentence were bugs until this suite existed.
///
/// It used to stop at `t ≈ −3212`, returning that same value for every smaller
/// `p`, because `students_t_cdf` computed `½ + ½·sgn(z)·(1 − I_w)` — two
/// subtractions that quantised the lower tail into steps of `1.1e-16`, leaving
/// the bisection nothing to bracket. Reading `½·I_w` directly below the
/// location moved the floor out by forty orders of magnitude.
///
/// And past the new floor it returns `−inf`, not a number. The bisection's
/// bracket search doubles a finite number of times, so a quantile near `−1e60`
/// is beyond its reach — and the honest report of that is the support bound.
/// Before the guard added to `inverse_cdf_bisect` it returned the midpoint of
/// whatever bracket the doubling had stopped at, which is how
/// `InverseGamma(0.5, 1)` came to have a quantile of `1.8446744e19`.
#[test]
fn a_students_t_quantile_reaches_far_and_then_refuses() {
    use bayes_distribution_ext::{ContinuousCdf, StudentT};
    let t = StudentT::new(5.0).unwrap();

    // Well past the old floor, and the round trip holds there.
    let deep: f64 = t.inverse_cdf(1e-60);
    assert!(
        deep.is_finite() && deep < -1e11,
        "the quantile at 1e-60 should be around -1.6e12, got {deep}"
    );
    assert!(
        (t.cdf(deep) / 1e-60 - 1.0).abs() < 1e-6,
        "cdf(inverse_cdf(1e-60)) = {:.6e}, expected 1e-60",
        t.cdf(deep)
    );

    // The old floor is now far inside the resolvable range, which is the
    // regression guard: if the cdf reverts to its subtracting form, this value
    // collapses back to about -3212.
    let mid: f64 = t.inverse_cdf(1e-30);
    assert!(
        mid < -1e5,
        "the quantile at 1e-30 should be near -6e5, got {mid} — a value near -3212 means \
         students_t_cdf is subtracting again"
    );

    // And beyond the bracket search's reach, a refusal rather than a number.
    assert_eq!(
        t.inverse_cdf(1e-300_f64),
        f64::NEG_INFINITY,
        "an unreachable quantile must report the support bound"
    );
}
