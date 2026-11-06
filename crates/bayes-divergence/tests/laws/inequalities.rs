//! Do the documented bounds hold across every family and parameter pair?

use super::close;
use bayes_distribution::{
    Bernoulli, Binomial, ChiSquared, Gamma, HalfNormal, InverseGamma, LogNormal, NegativeBinomial,
    Normal, Uniform, VonMises, Wald,
};
use bayes_divergence::{
    ChiSquaredDivergence, Divergence, HellingerDistance, KlDivergence, RenyiDivergence,
};
use bayes_utils::Failures;

/// One ordered pair of same-family distributions, reduced to the five
/// quantities the laws relate.
struct Pair {
    name: String,
    kl: f64,
    reverse: f64,
    hellinger_sq: f64,
    chi_squared: f64,
    renyi_half: f64,
    renyi_two: f64,
}

fn pair<D>(name: String, a: &D, b: &D) -> Pair
where
    D: KlDivergence<Output = f64>
        + Divergence
        + HellingerDistance<Output = f64>
        + ChiSquaredDivergence<Output = f64>
        + RenyiDivergence<Output = f64>,
{
    Pair {
        name,
        kl: a.kl_divergence(b),
        reverse: b.kl_divergence(a),
        hellinger_sq: a.hellinger_sq(b),
        chi_squared: a.chi_squared_divergence(b),
        renyi_half: a.renyi_divergence(b, 0.5),
        renyi_two: a.renyi_divergence(b, 2.0),
    }
}

/// Every family that carries all five quantities, swept over parameter pairs
/// that include the self-comparison (where everything must be exactly zero)
/// and ratios wide enough to stress the formulas.
fn pairs() -> Vec<Pair> {
    let mut out = Vec::new();
    let scales = [0.5_f64, 1.0, 2.0];
    for &s0 in &scales {
        for &s1 in &scales {
            for &shift in &[0.0_f64, 1.0] {
                out.push(pair(
                    format!("Normal(0,{s0}) vs Normal({shift},{s1})"),
                    &Normal::new(0.0, s0),
                    &Normal::new(shift, s1),
                ));
            }
            out.push(pair(
                format!("Gamma(2,{s0}) vs Gamma(2,{s1})"),
                &Gamma::new(2.0, s0),
                &Gamma::new(2.0, s1),
            ));
            out.push(pair(
                format!("HalfNormal({s0}) vs HalfNormal({s1})"),
                &HalfNormal::new(s0),
                &HalfNormal::new(s1),
            ));
            out.push(pair(
                format!("LogNormal(0,{s0}) vs LogNormal(0,{s1})"),
                &LogNormal::new(0.0, s0),
                &LogNormal::new(0.0, s1),
            ));
            out.push(pair(
                format!("InverseGamma(3,{s0}) vs InverseGamma(3,{s1})"),
                &InverseGamma::new(3.0, s0),
                &InverseGamma::new(3.0, s1),
            ));
            out.push(pair(
                format!("Wald(1,{s0}) vs Wald(1,{s1})"),
                &Wald::new(1.0, s0),
                &Wald::new(1.0, s1),
            ));
            out.push(pair(
                format!("VonMises(0,{s0}) vs VonMises(0,{s1})"),
                &VonMises::new(0.0, s0),
                &VonMises::new(0.0, s1),
            ));
            out.push(pair(
                format!("ChiSquared({}) vs ChiSquared({})", s0 + 1.0, s1 + 1.0),
                &ChiSquared::new(s0 + 1.0),
                &ChiSquared::new(s1 + 1.0),
            ));
        }
    }
    for &p0 in &[0.1_f64, 0.3, 0.5, 0.9] {
        for &p1 in &[0.1_f64, 0.3, 0.5, 0.9] {
            out.push(pair(
                format!("Bernoulli({p0}) vs Bernoulli({p1})"),
                &Bernoulli::new(p0),
                &Bernoulli::new(p1),
            ));
            out.push(pair(
                format!("Binomial(10,{p0}) vs Binomial(10,{p1})"),
                &Binomial::new(10_u64, p0),
                &Binomial::new(10_u64, p1),
            ));
            out.push(pair(
                format!("NegativeBinomial(5,{p0}) vs NegativeBinomial(5,{p1})"),
                &NegativeBinomial::new(5.0, p0),
                &NegativeBinomial::new(5.0, p1),
            ));
        }
    }
    out.push(pair(
        "Uniform(0,1) vs Uniform(0,2)".into(),
        &Uniform::new(0.0, 1.0),
        &Uniform::new(0.0, 2.0),
    ));
    out
}

/// Gibbs, the Hellinger range, `H² ≤ ½KL`, and `KL ≤ χ²`.
///
/// Each of these relates one family's closed form to another's, so none of
/// them can be satisfied by a formula that is merely self-consistent.
#[test]
fn the_documented_bounds_hold() {
    let mut f = Failures::default();
    for p in pairs() {
        if p.kl.is_finite() {
            f.check(p.kl >= -1e-12, || {
                format!("{}: KL = {:.6e} is negative (Gibbs)", p.name, p.kl)
            });
        }
        if p.reverse.is_finite() {
            f.check(p.reverse >= -1e-12, || {
                format!("{}: reverse KL = {:.6e} is negative", p.name, p.reverse)
            });
        }
        if p.hellinger_sq.is_finite() {
            f.check((-1e-12..=1.0 + 1e-12).contains(&p.hellinger_sq), || {
                format!("{}: H² = {:.6e} escapes [0, 1]", p.name, p.hellinger_sq)
            });
        }
        if p.chi_squared.is_finite() {
            f.check(p.chi_squared >= -1e-12, || {
                format!("{}: χ² = {:.6e} is negative", p.name, p.chi_squared)
            });
        }
        if p.hellinger_sq.is_finite() && p.kl.is_finite() {
            f.check(p.hellinger_sq <= 0.5 * p.kl + 1e-9, || {
                format!(
                    "{}: H² = {:.6e} exceeds KL/2 = {:.6e}",
                    p.name,
                    p.hellinger_sq,
                    0.5 * p.kl
                )
            });
        }
        if p.kl.is_finite() && p.chi_squared.is_finite() {
            f.check(p.kl <= p.chi_squared + 1e-9, || {
                format!(
                    "{}: KL = {:.6e} exceeds χ² = {:.6e}",
                    p.name, p.kl, p.chi_squared
                )
            });
        }
    }
    f.finish("documented bounds");
}

/// The two exact cross-identities between Rényi, Hellinger and chi-squared.
///
/// `R_{1/2} = −2 ln(1 − H²)` and `R_2 = ln(1 + χ²)`. These are equalities
/// between three separately written closed forms per family, which makes them
/// the sharpest checks available — an inequality tolerates a wrong constant in
/// one direction, an identity does not.
#[test]
fn renyi_agrees_with_hellinger_and_chi_squared() {
    let mut f = Failures::default();
    for p in pairs() {
        if p.renyi_half.is_finite() && p.hellinger_sq.is_finite() && p.hellinger_sq < 1.0 {
            let expected = -2.0 * (1.0 - p.hellinger_sq).ln();
            f.check(close(p.renyi_half, expected, 1e-7, 1e-9), || {
                format!(
                    "{}: R(1/2) = {:.10e} but -2 ln(1 - H²) = {:.10e}",
                    p.name, p.renyi_half, expected
                )
            });
        }
        if p.renyi_two.is_finite() && p.chi_squared.is_finite() && p.chi_squared > -1.0 {
            let expected = (1.0 + p.chi_squared).ln();
            f.check(close(p.renyi_two, expected, 1e-7, 1e-9), || {
                format!(
                    "{}: R(2) = {:.10e} but ln(1 + χ²) = {:.10e}",
                    p.name, p.renyi_two, expected
                )
            });
        }
    }
    f.finish("Rényi cross-identities");
}

/// Rényi is nonnegative and non-decreasing in its order, and meets KL at 1.
///
/// The nonnegativity is not free: the `1/(α − 1)` factor amplifies the
/// logarithm's rounding error, and at `α = 1 ± 1e-7` that produced values
/// around `−2e-9` for two *identical* distributions. Clamping at zero is
/// correct because the analytic value cannot be negative.
#[test]
fn renyi_is_nonnegative_monotone_and_meets_kl_at_one() {
    let mut f = Failures::default();
    let orders = [0.1_f64, 0.25, 0.5, 0.9, 1.0, 1.5, 2.0, 5.0, 20.0];
    for (name, a, b) in [
        (
            "Normal(0,1) vs Normal(1,2)",
            Normal::new(0.0, 1.0),
            Normal::new(1.0, 2.0),
        ),
        (
            "Normal(0,1) vs Normal(0,1)",
            Normal::new(0.0, 1.0),
            Normal::new(0.0, 1.0),
        ),
        (
            "Normal(0,2) vs Normal(0,1)",
            Normal::new(0.0, 2.0),
            Normal::new(0.0, 1.0),
        ),
    ] {
        let mut previous = -1.0_f64;
        for alpha in orders {
            let value = a.renyi_divergence(&b, alpha);
            if !value.is_finite() {
                continue;
            }
            f.check(value >= 0.0, || {
                format!("{name}: R({alpha}) = {value:.6e} is negative")
            });
            f.check(value >= previous - 1e-9, || {
                format!("{name}: R fell at α = {alpha}: {previous:.9} -> {value:.9}")
            });
            previous = value;
        }
        // The KL limit, approached from both sides.
        let kl = a.kl_divergence(&b);
        for alpha in [1.0 - 1e-9, 1.0, 1.0 + 1e-9, 1.0 - 1e-6, 1.0 + 1e-6] {
            let value = a.renyi_divergence(&b, alpha);
            f.check(close(value, kl, 1e-5, 1e-9), || {
                format!("{name}: R({alpha}) = {value:.12} but KL = {kl:.12}")
            });
        }
    }
    f.finish("Rényi order behaviour");
}

/// Every divergence is zero exactly at a self-comparison, and Jeffreys is the
/// symmetric sum.
#[test]
fn a_self_comparison_is_zero_and_jeffreys_is_symmetric() {
    let mut f = Failures::default();
    for p in pairs() {
        if !p.name.contains(" vs ") {
            continue;
        }
        let (left, right) = p.name.split_once(" vs ").unwrap();
        if left != right {
            continue;
        }
        for (what, value) in [
            ("KL", p.kl),
            ("reverse KL", p.reverse),
            ("H²", p.hellinger_sq),
            ("χ²", p.chi_squared),
            ("R(1/2)", p.renyi_half),
            ("R(2)", p.renyi_two),
        ] {
            f.check(value.abs() < 1e-9, || {
                format!(
                    "{}: {what} against itself is {value:.6e}, expected 0",
                    p.name
                )
            });
        }
    }
    // Jeffreys is the sum of the two directions, exactly.
    let (a, b) = (Normal::new(0.0, 1.0), Normal::new(1.0, 2.0));
    f.check(
        close(
            a.jeffreys(&b),
            a.kl_divergence(&b) + b.kl_divergence(&a),
            0.0,
            1e-15,
        ),
        || "Jeffreys should be the sum of both KL directions".into(),
    );
    f.check(close(a.jeffreys(&b), b.jeffreys(&a), 0.0, 1e-15), || {
        "Jeffreys should be symmetric".into()
    });
    f.finish("self-comparison and symmetry");
}
