//! Parameter recovery for the continuous families.

use super::harness::{Case, SAMPLE, case, recovers_its_parameters};
use bayes_distribution_ext::{
    Beta, Cauchy, ChiSquared, Estimate, Exponential, FisherSnedecor, Frechet, Gamma, Gumbel,
    HalfNormal, InverseGamma, Laplace, LogNormal, Logistic, Lomax, Moments, Normal, Pareto,
    Rayleigh, Triangular, Uniform, VonMises, Wald, Weibull,
};
use rand::distr::Distribution;

/// Each family, with parameters chosen to sit comfortably inside its
/// estimator's valid region — a Lomax shape above 4 so the sample variance has
/// a finite variance of its own, an F denominator above 8 for the same reason.
/// Putting them at the edge instead would test the sampling theory of the
/// moments rather than the formulas.
fn cases() -> Vec<Case> {
    let n = SAMPLE as f64;
    vec![
        case(
            "Beta(2.5, 4)",
            &["alpha", "beta"],
            &[2.5, 4.0],
            |rng| Beta::new(2.5, 4.0).unwrap().sample(rng),
            |d| Beta::from_sample(d).map(|x| vec![x.alpha(), x.beta()]),
        ),
        case(
            "Cauchy(1.5, 2)",
            &["location", "scale"],
            &[1.5, 2.0],
            |rng| Cauchy::new(1.5, 2.0).unwrap().sample(rng),
            |d| Cauchy::from_sample(d).map(|x| vec![x.location(), x.scale()]),
        ),
        case(
            "ChiSquared(7)",
            &["freedom"],
            &[7.0],
            |rng| ChiSquared::new(7.0).unwrap().sample(rng),
            |d| ChiSquared::from_sample(d).map(|x| vec![x.freedom()]),
        ),
        case(
            "Exponential(2)",
            &["rate"],
            &[2.0],
            |rng| Exponential::new(2.0).unwrap().sample(rng),
            |d| Exponential::from_sample(d).map(|x| vec![x.rate()]),
        ),
        case(
            "FisherSnedecor(6, 20)",
            &["d1", "d2"],
            &[6.0, 20.0],
            |rng| FisherSnedecor::new(6.0, 20.0).unwrap().sample(rng),
            |d| FisherSnedecor::from_sample(d).map(|x| vec![x.d1(), x.d2()]),
        ),
        case(
            "Frechet(3, 2)",
            &["shape", "scale"],
            &[3.0, 2.0],
            |rng| Frechet::new(3.0, 2.0).unwrap().sample(rng),
            |d| Frechet::from_sample(d).map(|x| vec![x.shape(), x.scale()]),
        ),
        case(
            "Gamma(3, 1.5)",
            &["shape", "rate"],
            &[3.0, 1.5],
            |rng| Gamma::new(3.0, 1.5).unwrap().sample(rng),
            |d| Gamma::from_sample(d).map(|x| vec![x.shape(), x.rate()]),
        ),
        case(
            "Gumbel(1, 2)",
            &["location", "scale"],
            &[1.0, 2.0],
            |rng| Gumbel::new(1.0, 2.0).unwrap().sample(rng),
            |d| Gumbel::from_sample(d).map(|x| vec![x.location(), x.scale()]),
        ),
        case(
            "HalfNormal(1.5)",
            &["scale"],
            &[1.5],
            |rng| HalfNormal::new(1.5).unwrap().sample(rng),
            |d| HalfNormal::from_sample(d).map(|x| vec![x.scale()]),
        ),
        case(
            "InverseGamma(6, 5)",
            &["shape", "scale"],
            &[6.0, 5.0],
            |rng| InverseGamma::new(6.0, 5.0).unwrap().sample(rng),
            |d| InverseGamma::from_sample(d).map(|x| vec![x.shape(), x.scale()]),
        ),
        case(
            "Laplace(1, 2)",
            &["location", "scale"],
            &[1.0, 2.0],
            |rng| Laplace::new(1.0, 2.0).unwrap().sample(rng),
            |d| Laplace::from_sample(d).map(|x| vec![x.location(), x.scale()]),
        ),
        case(
            "LogNormal(0.5, 0.8)",
            &["mu", "sigma"],
            &[0.5, 0.8],
            |rng| LogNormal::new(0.5, 0.8).unwrap().sample(rng),
            |d| LogNormal::from_sample(d).map(|x| vec![x.mu(), x.sigma()]),
        ),
        case(
            "Logistic(1, 0.7)",
            &["location", "scale"],
            &[1.0, 0.7],
            |rng| Logistic::new(1.0, 0.7).unwrap().sample(rng),
            |d| Logistic::from_sample(d).map(|x| vec![x.location(), x.scale()]),
        ),
        case(
            "Lomax(6, 5)",
            &["shape", "scale"],
            &[6.0, 5.0],
            |rng| Lomax::new(6.0, 5.0).unwrap().sample(rng),
            |d| Lomax::from_sample(d).map(|x| vec![x.shape(), x.scale()]),
        ),
        case(
            "Normal(2, 1.5)",
            &["mean", "std_dev"],
            &[2.0, 1.5],
            |rng| Normal::new(2.0, 1.5).unwrap().sample(rng),
            |d| Normal::from_sample(d).map(|x| vec![x.params().0, x.params().1]),
        ),
        // The Hill estimator's shape has a known `O(1/n)` bias: the sum of logs
        // is `Gamma(n − 1, α)`, so `n/S` has mean `nα/(n − 2)`, overshooting by
        // `2α/(n − 2)`. The scale is the sample minimum, biased up by about
        // `scale/(nα)`, which is far smaller.
        case(
            "Pareto(2, 3)",
            &["scale", "shape"],
            &[2.0, 3.0],
            |rng| Pareto::new(2.0, 3.0).unwrap().sample(rng),
            |d| Pareto::from_sample(d).map(|x| vec![x.scale(), x.shape()]),
        )
        .with_bias(2.0 * 3.0 / (n - 2.0)),
        case(
            "Rayleigh(2)",
            &["scale"],
            &[2.0],
            |rng| Rayleigh::new(2.0).unwrap().sample(rng),
            |d| Rayleigh::from_sample(d).map(|x| vec![x.scale()]),
        ),
        // A triangular density vanishes *linearly* at each endpoint, so its
        // extreme order statistics converge at `n^{-1/2}` rather than `n^{-1}`:
        // `F(x) ≈ (x − lo)²/30` near `lo` here, so the minimum sits about
        // `√(30/n) ≈ 0.12` inside it, and `√(70/n) ≈ 0.19` at the top. The mode
        // is `3x̄ − lo − hi`, which inherits both. Allowing their sum, 0.31,
        // rounded up — this is the loosest bound in the suite and the reason is
        // the family's shape, not the estimator.
        case(
            "Triangular(0, 10, 3)",
            &["lo", "hi", "mode"],
            &[0.0, 10.0, 3.0],
            |rng| Triangular::new(0.0, 10.0, 3.0).unwrap().sample(rng),
            |d| Triangular::from_sample(d).map(|x| vec![x.lo(), x.hi(), x.mode()]),
        )
        .with_bias((30.0 / n).sqrt() + (70.0 / n).sqrt()),
        // A uniform's density is positive at both endpoints, so each extreme is
        // biased inward by `range/(n + 1)`.
        case(
            "Uniform(-2, 5)",
            &["lo", "hi"],
            &[-2.0, 5.0],
            |rng| Uniform::new(-2.0, 5.0).unwrap().sample(rng),
            |d| Uniform::from_sample(d).map(|x| vec![x.lo(), x.hi()]),
        )
        .with_bias(7.0 / (n + 1.0)),
        case(
            "VonMises(1, 3)",
            &["mu", "kappa"],
            &[1.0, 3.0],
            |rng| VonMises::new(1.0, 3.0).unwrap().sample(rng),
            |d| VonMises::from_sample(d).map(|x| vec![x.mu(), x.kappa()]),
        )
        // Fisher's rational approximation to the inverse of `I₁/I₀` is accurate
        // to about 1% in this range, which is a property of the approximation
        // rather than of the sample and does not shrink with `n`.
        .with_bias(0.03),
        case(
            "Wald(2, 3)",
            &["mu", "shape"],
            &[2.0, 3.0],
            |rng| Wald::new(2.0, 3.0).unwrap().sample(rng),
            |d| Wald::from_sample(d).map(|x| vec![x.mu(), x.shape()]),
        ),
        case(
            "Weibull(3, 2)",
            &["scale", "shape"],
            &[3.0, 2.0],
            |rng| Weibull::new(3.0, 2.0).unwrap().sample(rng),
            |d| Weibull::from_sample(d).map(|x| vec![x.scale(), x.shape()]),
        ),
    ]
}

#[test]
fn the_continuous_families_recover_their_own_parameters() {
    recovers_its_parameters(cases(), 20_261_105, "continuous parameter recovery");
}

/// Student's t is held to a weaker claim than the rest, and deliberately.
///
/// Its degrees of freedom come from an excess kurtosis through
/// `ν̂ = 4 + 6/g₂`, which is strongly **convex** in `g₂`. So even with a
/// perfectly unbiased `g₂`, Jensen's inequality puts `E[ν̂]` above `ν` — the
/// estimator is consistent but biased at any finite `n`, and the bias is a
/// property of the reciprocal, not of the implementation. The replicate-mean
/// check the other families get would fail here for a correct formula, which
/// would make it a test of the arithmetic of `1/x` rather than of this crate.
///
/// What is checked instead is that the fit lands in the right *region*: within
/// 30% on the degrees of freedom and 5% on the scale, from a sample large
/// enough for a fourth moment to mean something. That still catches everything
/// a wrong formula does — a missing `+ 4`, the wrong kurtosis convention
/// (excess versus raw, a difference of exactly 3), or variance and scale
/// transposed all miss by far more than 30%.
#[test]
fn a_students_t_fit_lands_in_the_right_region() {
    use bayes_distribution_ext::StudentT;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(20_261_107);
    // Freedom 12 keeps the eighth moment finite, which is what the sampling
    // error of a fourth moment needs in order to exist at all.
    let truth = StudentT::new_location_scale(12.0, 1.0, 2.0).unwrap();
    let data: Vec<f64> = (0..200_000).map(|_| truth.sample(&mut rng)).collect();
    let fitted = StudentT::from_sample(&data).expect("a large t sample should fit");

    let freedom_error = (fitted.freedom() - 12.0).abs() / 12.0;
    assert!(
        freedom_error < 0.30,
        "freedom {} is {:.1}% from 12",
        fitted.freedom(),
        freedom_error * 100.0
    );
    assert!(
        (fitted.location() - 1.0).abs() < 0.05,
        "location {} is not near 1",
        fitted.location()
    );
    assert!(
        (fitted.scale() - 2.0).abs() / 2.0 < 0.05,
        "scale {} is not near 2",
        fitted.scale()
    );
    // And the direction of the bias is the predicted one.
    assert!(
        fitted.freedom() > 11.0,
        "the convexity of 6/g2 biases this fit upward, so a value far below 12 \
         means something other than Jensen is wrong: got {}",
        fitted.freedom()
    );
}
