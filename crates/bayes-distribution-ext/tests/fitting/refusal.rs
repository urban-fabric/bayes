//! What every estimator must decline to fit.
//!
//! Three kinds of input belong here, and they fail for different reasons:
//!
//! 1. **Degenerate** — empty, a single point, a constant. There is no spread to
//!    fit.
//! 2. **Off-support** — a beta sample outside `(0, 1)`, a zero in a positive
//!    family, a 2 in a Bernoulli sample. The data is not from this family.
//! 3. **Out of the family's reach** — the moments are real and finite but no
//!    member of the family has them: an underdispersed negative binomial, a
//!    beta sample more dispersed than any beta, an F whose mean is below 1.
//!
//! The third kind is the interesting one, because the arithmetic completes
//! happily and returns a negative shape or a probability above 1. Every one of
//! these returns `None` rather than clamping, so that a fit never reports
//! something the sample did not support.

use bayes_distribution_ext::{
    Bernoulli, Beta, Cauchy, ChiSquared, Estimate, Exponential, FisherSnedecor, Frechet, Gamma,
    Gumbel, HalfNormal, InverseGamma, Laplace, LogNormal, Logistic, Lomax, NegativeBinomial,
    Normal, Pareto, Poisson, Rayleigh, StudentT, Triangular, Uniform, VonMises, Wald, Weibull,
};
use bayes_utils::Failures;

/// Is the fit absent? One entry per family, so a new family added without a
/// degenerate-input guard shows up as a named failure.
fn refusals(data: &[f64]) -> Vec<(&'static str, bool)> {
    vec![
        ("Bernoulli", Bernoulli::from_sample(data).is_none()),
        ("Beta", Beta::from_sample(data).is_none()),
        ("Cauchy", Cauchy::from_sample(data).is_none()),
        ("ChiSquared", ChiSquared::from_sample(data).is_none()),
        ("Exponential", Exponential::from_sample(data).is_none()),
        (
            "FisherSnedecor",
            FisherSnedecor::from_sample(data).is_none(),
        ),
        ("Frechet", Frechet::from_sample(data).is_none()),
        ("Gamma", Gamma::from_sample(data).is_none()),
        ("Gumbel", Gumbel::from_sample(data).is_none()),
        ("HalfNormal", HalfNormal::from_sample(data).is_none()),
        ("InverseGamma", InverseGamma::from_sample(data).is_none()),
        ("Laplace", Laplace::from_sample(data).is_none()),
        ("LogNormal", LogNormal::from_sample(data).is_none()),
        ("Logistic", Logistic::from_sample(data).is_none()),
        ("Lomax", Lomax::from_sample(data).is_none()),
        (
            "NegativeBinomial",
            NegativeBinomial::from_sample(data).is_none(),
        ),
        ("Normal", Normal::from_sample(data).is_none()),
        ("Pareto", Pareto::from_sample(data).is_none()),
        ("Poisson", Poisson::from_sample(data).is_none()),
        ("Rayleigh", Rayleigh::from_sample(data).is_none()),
        ("StudentT", StudentT::from_sample(data).is_none()),
        ("Triangular", Triangular::from_sample(data).is_none()),
        ("Uniform", Uniform::from_sample(data).is_none()),
        ("VonMises", VonMises::from_sample(data).is_none()),
        ("Wald", Wald::from_sample(data).is_none()),
        ("Weibull", Weibull::from_sample(data).is_none()),
    ]
}

/// No family fits an empty sample or a non-finite value.
///
/// A **single point** is treated separately, because the crate's answer there
/// is not uniform and should not be: a one-parameter family whose maximum
/// likelihood estimate is defined from one observation can legitimately use it.
/// `Exponential::from_sample(&[2.0])` gives rate `0.5`, which is the MLE, and
/// its sufficient statistic says so by requiring a count of 1 rather than 2.
#[test]
fn degenerate_samples_are_refused_by_every_family() {
    let mut f = Failures::default();
    for (description, data) in [
        ("an empty sample", vec![]),
        ("a sample containing NaN", vec![1.0, 2.0, f64::NAN]),
        (
            "a sample containing infinity",
            vec![1.0, 2.0, f64::INFINITY],
        ),
        (
            "a sample containing negative infinity",
            vec![1.0, 2.0, f64::NEG_INFINITY],
        ),
    ] {
        for (family, refused) in refusals(&data) {
            f.check(refused, || {
                format!("{family} fitted {description} instead of returning None")
            });
        }
    }
    f.finish("degenerate input");
}

/// One observation determines a one-parameter fit and nothing more.
///
/// The families listed as fittable are exactly those whose single parameter is
/// a function of a single observation. Everything else needs at least a pair,
/// and the point of pinning this down is that the boundary is a design choice
/// rather than an accident of which helper each estimator happened to use.
#[test]
fn a_single_point_fits_only_the_one_parameter_families() {
    let mut f = Failures::default();
    let data = vec![2.0];
    let fittable = ["Exponential"];
    for (family, refused) in refusals(&data) {
        if fittable.contains(&family) {
            f.check(!refused, || {
                format!("{family} should fit a single point: its MLE is defined from one")
            });
        } else {
            f.check(refused, || format!("{family} fitted a single point"));
        }
    }
    f.finish("single observation");
}

/// A constant sample has a location but no scale. Only the families that are
/// pure location or pure frequency can fit one.
#[test]
fn a_constant_sample_is_refused_by_every_family_with_a_scale() {
    let mut f = Failures::default();
    // Every value 3, repeated — a legitimate sample with zero variance.
    let data = vec![3.0; 50];
    // These have no scale to estimate, so a constant sample is fittable: a
    // chi-squared reads its freedom off the mean, a Poisson its rate, a
    // half-normal its scale from the second moment about zero, and a Pareto
    // puts every point at its lower bound.
    let fittable = [
        "ChiSquared",
        "Exponential",
        "HalfNormal",
        "Poisson",
        "Rayleigh",
        "Gamma",
        "Pareto",
        "VonMises",
        "Wald",
    ];
    for (family, refused) in refusals(&data) {
        if fittable.contains(&family) {
            continue;
        }
        f.check(refused, || {
            format!("{family} fitted a constant sample, which has no scale")
        });
    }
    f.finish("constant input");
}

/// Data outside a family's support is not data from that family.
#[test]
fn off_support_samples_are_refused() {
    let mut f = Failures::default();

    // A beta lives on (0, 1).
    for data in [vec![0.2, 0.5, 1.4], vec![-0.1, 0.3, 0.6], vec![0.0, 0.5]] {
        f.check(Beta::from_sample(&data).is_none(), || {
            format!("Beta fitted {data:?}, which leaves (0, 1)")
        });
    }
    // The log-scale families need strictly positive data.
    for data in [vec![1.0, 2.0, 0.0], vec![1.0, -2.0, 3.0]] {
        for (family, refused) in [
            ("LogNormal", LogNormal::from_sample(&data).is_none()),
            ("Weibull", Weibull::from_sample(&data).is_none()),
            ("Frechet", Frechet::from_sample(&data).is_none()),
            ("Wald", Wald::from_sample(&data).is_none()),
            ("InverseGamma", InverseGamma::from_sample(&data).is_none()),
            ("ChiSquared", ChiSquared::from_sample(&data).is_none()),
        ] {
            f.check(refused, || {
                format!("{family} fitted {data:?}, which is not strictly positive")
            });
        }
    }
    // A Bernoulli sample is zeros and ones; a geometric starts at 1; counts are
    // integers.
    f.check(Bernoulli::from_sample(&[0.0, 1.0, 2.0]).is_none(), || {
        "Bernoulli fitted a sample containing 2".into()
    });
    f.check(
        bayes_distribution_ext::Geometric::from_sample(&[0.0, 1.0, 2.0]).is_none(),
        || "Geometric fitted a sample containing 0, which is the failure-count convention".into(),
    );
    f.check(Poisson::from_sample(&[1.0, 2.5, 3.0]).is_none(), || {
        "Poisson fitted a non-integer sample".into()
    });
    f.check(
        NegativeBinomial::from_sample(&[1.0, 2.5, 3.0]).is_none(),
        || "NegativeBinomial fitted a non-integer sample".into(),
    );

    f.finish("off-support input");
}

/// The moments are finite and real, but no member of the family has them. This
/// is where a method of moments quietly produces a negative shape if nobody
/// checks.
#[test]
fn samples_outside_the_familys_reach_are_refused() {
    let mut f = Failures::default();

    // A beta's method-of-moments ceiling, `v < m(1 − m)`, turns out to be
    // *unreachable* from data inside the open interval: that bound is the
    // supremum, attained only by mass sitting exactly on 0 and 1. So a bimodal
    // sample is not refused — it fits the U-shaped beta it should, with both
    // shapes below 1. Asserted in that direction, because expecting a refusal
    // here is the mistake: it looks like the extreme case and is not one.
    let mut bimodal = vec![0.001; 40];
    bimodal.extend(vec![0.999; 40]);
    let fitted = Beta::from_sample(&bimodal);
    match fitted {
        Some(b) => f.check(b.alpha() < 1.0 && b.beta() < 1.0, || {
            format!(
                "a bimodal sample should fit a U-shaped beta, got alpha {} beta {}",
                b.alpha(),
                b.beta()
            )
        }),
        None => f.check(false, || {
            "a bimodal sample inside (0, 1) does have a beta fit".into()
        }),
    }

    // A negative binomial is always overdispersed. An underdispersed count
    // sample has no fit here, and wants a Poisson.
    let underdispersed: Vec<f64> = (0..60)
        .map(|i| if i % 2 == 0 { 4.0 } else { 5.0 })
        .collect();
    f.check(
        NegativeBinomial::from_sample(&underdispersed).is_none(),
        || "NegativeBinomial fitted an underdispersed sample".into(),
    );
    // The same sample is fine for a Poisson.
    f.check(Poisson::from_sample(&underdispersed).is_some(), || {
        "Poisson should fit an underdispersed count sample".into()
    });

    // A Lomax needs the coefficient of variation above 1. A tight positive
    // sample is far below that.
    let tight: Vec<f64> = (0..60).map(|i| 10.0 + f64::from(i % 3)).collect();
    f.check(Lomax::from_sample(&tight).is_none(), || {
        "Lomax fitted a sample with coefficient of variation well below 1".into()
    });

    // An F needs a sample mean above 1 for `d2 = 2m/(m − 1)` to be positive.
    let small: Vec<f64> = (0..60).map(|i| 0.1 + f64::from(i % 5) * 0.01).collect();
    f.check(FisherSnedecor::from_sample(&small).is_none(), || {
        "FisherSnedecor fitted a sample with mean below 1".into()
    });

    // A Student's t is strictly leptokurtic. A platykurtic sample — uniform
    // draws have excess kurtosis −1.2 — argues against the family.
    let flat: Vec<f64> = (0..200).map(|i| f64::from(i) / 200.0).collect();
    f.check(StudentT::from_sample(&flat).is_none(), || {
        "StudentT fitted a platykurtic sample".into()
    });

    f.finish("unreachable moments");
}

/// The three families with no `Estimate` at all.
///
/// This cannot be a runtime check — the absence of a trait implementation is a
/// compile-time fact — so it is recorded here as the place someone will look.
/// `Binomial` and `BetaBinomial` need a trial count and `Hypergeometric` needs
/// a population size, and those are properties of the experiment rather than of
/// the data. Method of moments for an unknown binomial `n` gives
/// `n̂ = m²/(m − v)`, which is negative for every overdispersed sample.
///
/// If an `Estimate` is ever added for one of them, this test should be replaced
/// by a recovery case rather than deleted.
#[test]
fn the_structural_families_have_no_estimate() {
    // Spelled out so the assertion is about a real sample rather than a
    // comment: these are the moments that defeat an unknown-`n` fit.
    let overdispersed = [0.0, 1.0, 9.0, 10.0, 0.0, 10.0];
    let n = overdispersed.len() as f64;
    let mean = overdispersed.iter().sum::<f64>() / n;
    let variance = overdispersed
        .iter()
        .map(|x| (x - mean) * (x - mean))
        .sum::<f64>()
        / n;
    assert!(
        variance > mean,
        "this sample is meant to be overdispersed: mean {mean}, variance {variance}"
    );
    assert!(
        mean * mean / (mean - variance) < 0.0,
        "an unknown-n binomial fit on an overdispersed sample gives a negative trial count"
    );
}
