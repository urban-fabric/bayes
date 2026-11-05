//! What happens at the boundary, past a convergence condition, and on `NaN`?
//!
//! A divergence has three distinct ways to be large, and they must not be
//! confused: a genuine infinity from disjoint supports, an infinity from a
//! convergence condition that fails, and a `NaN` from an input that was never
//! a number. The lazy families here deliberately do not validate their
//! parameters — that is their documented contract, since a generic comparison
//! produces a mask rather than a host boolean — so `NaN` parameters are
//! reachable and have to be handled downstream.

use bayes_distribution::{Bernoulli, Gamma, Normal, Uniform};
use bayes_divergence::{
    ChiSquaredDivergence, HellingerDistance, JensenShannon, KlDivergence, RenyiDivergence,
    TotalVariation,
};
use bayes_utils::Failures;

/// A `NaN` parameter yields `NaN`, never a confident number.
///
/// Three different wrong answers lived here. `hellinger_sq` returned **`0`** —
/// the clamp `(1 − BC).max(0)` implements IEEE `maxNum`, which *ignores* a
/// `NaN` operand, so "no information" became "identical distributions". `χ²`
/// and Rényi returned **`inf`**, because a convergence guard is a comparison
/// and every comparison against `NaN` is false, so `select(converges, value,
/// inf)` took the divergent branch for garbage input.
///
/// `0` is the worse of the two: an infinity at least signals that something
/// extreme happened.
#[test]
fn a_nan_parameter_never_produces_a_confident_answer() {
    let mut f = Failures::default();
    let good = Normal::new(0.0, 1.0);
    let cases = [
        ("NaN scale", Normal::new(0.0, f64::NAN)),
        ("NaN location", Normal::new(f64::NAN, 1.0)),
        ("both NaN", Normal::new(f64::NAN, f64::NAN)),
    ];
    for (what, bad) in cases {
        for (name, value) in [
            ("kl forward", good.kl_divergence(&bad)),
            ("kl reverse", bad.kl_divergence(&good)),
            ("hellinger_sq forward", good.hellinger_sq(&bad)),
            ("hellinger_sq reverse", bad.hellinger_sq(&good)),
            ("chi_squared forward", good.chi_squared_divergence(&bad)),
            ("chi_squared reverse", bad.chi_squared_divergence(&good)),
            ("renyi(2)", good.renyi_divergence(&bad, 2.0)),
            ("renyi(0.5)", good.renyi_divergence(&bad, 0.5)),
        ] {
            f.check(value.is_nan(), || {
                format!("Normal with a {what}: {name} returned {value}, expected NaN")
            });
        }
    }
    // A NaN *order* is equally not a number.
    for alpha in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let value = good.renyi_divergence(&good, alpha);
        f.check(value.is_nan(), || {
            format!("renyi with order {alpha} returned {value}, expected NaN")
        });
    }
    // And a gamma, to confirm this is not specific to the normal.
    let gamma_bad = Gamma::new(2.0, f64::NAN);
    let gamma_good = Gamma::new(2.0, 1.0);
    for (name, value) in [
        ("chi_squared", gamma_good.chi_squared_divergence(&gamma_bad)),
        ("renyi(2)", gamma_good.renyi_divergence(&gamma_bad, 2.0)),
        ("hellinger_sq", gamma_good.hellinger_sq(&gamma_bad)),
    ] {
        f.check(value.is_nan(), || {
            format!("Gamma with a NaN rate: {name} returned {value}, expected NaN")
        });
    }
    f.finish("NaN propagation");
}

/// A genuine infinity survives the `NaN` fix.
///
/// The fix had to distinguish "not a number" from "divergent", so this is the
/// other half: the cases that *should* be infinite still are. Three sources —
/// a chi-squared past its convergence condition, a KL between non-nested
/// supports, and a Bernoulli asserting something another rules out.
#[test]
fn the_genuine_infinities_are_still_infinite() {
    let mut f = Failures::default();

    // A normal chi-squared needs `2σ₁² > σ₀²`; just past it, it diverges.
    let p = Normal::new(0.0, 1.0);
    for s1 in [0.707_f64, 0.5, 0.1] {
        let value = p.chi_squared_divergence(&Normal::new(0.0, s1));
        f.check(value.is_infinite() && value > 0.0, || {
            format!("χ²(N(0,1) ‖ N(0,{s1})) = {value}, expected +inf past the condition")
        });
    }
    // And just inside it, a large finite number.
    let inside = p.chi_squared_divergence(&Normal::new(0.0, 0.7072));
    f.check(inside.is_finite() && inside > 1.0, || {
        format!("χ² just inside the condition is {inside}, expected large and finite")
    });

    // Non-nested uniform supports: KL is infinite in at least one direction.
    for (lo, hi) in [(0.5_f64, 1.5_f64), (1.0, 3.0), (-1.0, 0.5)] {
        let value = Uniform::new(0.0, 1.0).kl_divergence(&Uniform::new(lo, hi));
        f.check(value.is_infinite(), || {
            format!("KL(U(0,1) ‖ U({lo},{hi})) = {value}, expected +inf for non-nested support")
        });
    }
    // Nested the other way is finite: `ln 2` for a support twice as wide.
    let nested = Uniform::new(0.0, 1.0).kl_divergence(&Uniform::new(0.0, 2.0));
    f.check((nested - std::f64::consts::LN_2).abs() < 1e-12, || {
        format!("KL(U(0,1) ‖ U(0,2)) = {nested}, expected ln 2")
    });

    // A Bernoulli that rules out what another asserts.
    let ruled_out: f64 = Bernoulli::new(0.5).kl_divergence(&Bernoulli::new(0.0));
    f.check(ruled_out.is_infinite(), || {
        format!("KL(Bern(0.5) ‖ Bern(0)) = {ruled_out}, expected infinite")
    });
    f.finish("genuine infinities");
}

/// The degenerate Bernoulli agrees with itself.
///
/// `χ²(p ‖ p) = 0` by definition, but the formula is `(p−q)²/(q(1−q))` and at
/// `p = q = 0` both halves vanish: it returned `NaN`, a point mass reporting
/// infinite disagreement with itself. The bounded divergences had it right
/// already, which is what made the discrepancy visible.
#[test]
fn a_degenerate_bernoulli_agrees_with_itself() {
    let mut f = Failures::default();
    for p in [0.0_f64, 1.0] {
        let d = Bernoulli::new(p);
        for (name, value) in [
            ("KL", d.kl_divergence(&d)),
            ("H²", d.hellinger_sq(&d)),
            ("χ²", d.chi_squared_divergence(&d)),
            ("JS", d.js_divergence(&d)),
            ("TV", d.total_variation(&d)),
            ("R(2)", d.renyi_divergence(&d, 2.0)),
        ] {
            f.check(value == 0.0, || {
                format!("Bernoulli({p}): {name} against itself is {value}, expected exactly 0")
            });
        }
    }
    // And the fully-opposed pair is maximal in the bounded measures.
    let (zero, one) = (Bernoulli::new(0.0), Bernoulli::new(1.0));
    let opposed_h2: f64 = zero.hellinger_sq(&one);
    let opposed_tv: f64 = zero.total_variation(&one);
    f.check((opposed_h2 - 1.0).abs() < 1e-15, || {
        format!("H²(Bern(0) ‖ Bern(1)) = {opposed_h2}, expected 1")
    });
    f.check((opposed_tv - 1.0).abs() < 1e-15, || {
        format!("TV = {opposed_tv}, expected 1")
    });
    f.check(
        (zero.js_divergence(&one) - std::f64::consts::LN_2).abs() < 1e-12,
        || {
            format!(
                "JS = {}, expected ln 2 — its documented maximum",
                zero.js_divergence(&one)
            )
        },
    );
    f.finish("degenerate Bernoulli");
}

/// The bounded divergences stay inside their bounds, at every `p` pair
/// including the endpoints.
///
/// `0 ≤ JS ≤ ln 2` and `0 ≤ TV ≤ 1` are the documented ranges, and
/// `H² ≤ TV` relates two of them.
#[test]
fn the_bounded_divergences_respect_their_bounds() {
    let mut f = Failures::default();
    let grid = [
        0.0_f64,
        1e-12,
        0.01,
        0.25,
        0.5,
        0.75,
        0.99,
        1.0 - 1e-12,
        1.0,
    ];
    for &p0 in &grid {
        for &p1 in &grid {
            let (a, b) = (Bernoulli::new(p0), Bernoulli::new(p1));
            let (js, tv, h2) = (
                a.js_divergence(&b),
                a.total_variation(&b),
                a.hellinger_sq(&b),
            );
            f.check((0.0..=std::f64::consts::LN_2 + 1e-12).contains(&js), || {
                format!("JS({p0}, {p1}) = {js} escapes [0, ln 2]")
            });
            f.check((0.0..=1.0 + 1e-12).contains(&tv), || {
                format!("TV({p0}, {p1}) = {tv} escapes [0, 1]")
            });
            f.check((0.0..=1.0 + 1e-12).contains(&h2), || {
                format!("H²({p0}, {p1}) = {h2} escapes [0, 1]")
            });
            f.check(h2 <= tv + 1e-12, || {
                format!("H² = {h2} exceeds TV = {tv} for ({p0}, {p1})")
            });
            // Both are symmetric.
            f.check((js - b.js_divergence(&a)).abs() < 1e-15, || {
                format!("JS is not symmetric at ({p0}, {p1})")
            });
            f.check((tv - b.total_variation(&a)).abs() < 1e-15, || {
                format!("TV is not symmetric at ({p0}, {p1})")
            });
        }
    }
    f.finish("bounded divergences");
}
