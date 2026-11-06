//! **Discrete↔continuous bridges.** The strongest edges in the network.
//!
//! A `Poisson` CDF is a finite sum of exponentiated log-factorials. A
//! `ChiSquared` CDF is a regularized incomplete gamma evaluated by series or
//! continued fraction. They have no line of code in common — yet
//! `P(Poisson(λ) ≤ k)` and `P(χ²_{2(k+1)} > 2λ)` are *the same number*. An
//! identity across that much implementation distance is about as independent a
//! check as a single library can produce for itself.
//!
//! The three bridges here are the classical ones:
//!
//! ```text
//!   Poisson tail      P(N ≤ k)  = P(χ²_{2(k+1)} > 2λ)
//!   Binomial tail     P(N ≤ k)  = 1 − I_p(k+1, n−k)   = Beta(k+1, n−k).sf(p)
//!   NegBinomial tail  P(N ≤ k)  =     I_p(r, k+1)     = Beta(r, k+1).cdf(p)
//! ```
//!
//! Each is also checked against **direct summation of the PMF**, so a failure
//! localizes: if summation agrees with one side and not the other, the defect is
//! in the side that disagrees.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use crate::{DensityCase, Failures};
use bayes_distribution_ext::{Continuous, ContinuousCdf, Moments, *};

/// `P(Poisson(λ) ≤ k) = P(χ²_{2(k+1)} > 2λ)`.
#[test]
fn the_poisson_tail_is_a_chi_squared_tail() {
    let mut f = Failures::default();
    for lambda in [0.1, 1.0, 2.5, 10.0, 75.0, 500.0] {
        let pois = Poisson::new(lambda).unwrap();
        let mut summed = 0.0;
        for k in 0_u64..=40 {
            summed += Discrete::pmf(&pois, k);
            let chi = ChiSquared::new(2.0 * (k as f64 + 1.0)).unwrap();
            let via_chi = chi.sf(2.0 * lambda);
            let direct = DiscreteCdf::cdf(&pois, k);

            f.close(
                || {
                    format!(
                        "Poisson({lambda}).cdf({k}) vs ChiSquared({}).sf({})",
                        2 * (k + 1),
                        2.0 * lambda
                    )
                },
                direct,
                via_chi,
                1e-9,
            );
            // The same quantity by plain summation, so a disagreement above can
            // be attributed to one side rather than left ambiguous.
            f.close(
                || format!("Poisson({lambda}).cdf({k}) vs Σpmf"),
                direct,
                summed,
                1e-10,
            );
        }
    }
    f.finish("Poisson <-> ChiSquared");
}

/// `P(Binomial(n,p) ≤ k) = Beta(k+1, n−k).sf(p)`.
#[test]
fn the_binomial_tail_is_a_regularized_incomplete_beta() {
    let mut f = Failures::default();
    for (n, p) in [
        (10_u64, 0.5),
        (10, 0.05),
        (50, 0.3),
        (1000, 0.5),
        (1000, 0.001),
    ] {
        let bin = Binomial::new(n, p).unwrap();
        let mut summed = 0.0;
        for k in 0_u64..n.min(30) {
            summed += Discrete::pmf(&bin, k);
            let beta = Beta::new(k as f64 + 1.0, (n - k) as f64).unwrap();
            f.close(
                || {
                    format!(
                        "Binomial({n},{p}).cdf({k}) vs Beta({},{}).sf({p})",
                        k + 1,
                        n - k
                    )
                },
                DiscreteCdf::cdf(&bin, k),
                beta.sf(p),
                1e-9,
            );
            f.close(
                || format!("Binomial({n},{p}).cdf({k}) vs Σpmf"),
                DiscreteCdf::cdf(&bin, k),
                summed,
                1e-10,
            );
        }
    }
    f.finish("Binomial <-> Beta");
}

/// `P(NegativeBinomial(r,p) ≤ k) = Beta(r, k+1).cdf(p)`.
#[test]
fn the_negative_binomial_tail_is_a_regularized_incomplete_beta() {
    let mut f = Failures::default();
    for (r, p) in [(1.0, 0.5), (5.0, 0.5), (0.5, 0.3), (20.0, 0.8), (3.0, 0.05)] {
        let nb = NegativeBinomial::new(r, p).unwrap();
        let mut summed = 0.0;
        for k in 0_u64..30 {
            summed += Discrete::pmf(&nb, k);
            let beta = Beta::new(r, k as f64 + 1.0).unwrap();
            f.close(
                || {
                    format!(
                        "NegativeBinomial({r},{p}).cdf({k}) vs Beta({r},{}).cdf({p})",
                        k + 1
                    )
                },
                DiscreteCdf::cdf(&nb, k),
                beta.cdf(p),
                1e-9,
            );
            f.close(
                || format!("NegativeBinomial({r},{p}).cdf({k}) vs Σpmf"),
                DiscreteCdf::cdf(&nb, k),
                summed,
                1e-10,
            );
        }
    }
    f.finish("NegativeBinomial <-> Beta");
}

/// A continuous CDF against quadrature of its own density — the continuous
/// analogue of "CDF against PMF summation".
#[test]
fn continuous_cdf_increments_are_the_integral_of_the_density() {
    let mut f = Failures::default();
    let cases: Vec<DensityCase> = vec![
        {
            let d = Normal::new(0.0, 1.0).unwrap();
            (
                "Normal(0,1)".into(),
                Box::new(move |x| d.pdf(x)),
                Box::new(move |x| d.cdf(x)),
                -4.0,
                4.0,
            )
        },
        {
            let d = Gamma::new(3.0, 2.0).unwrap();
            (
                "Gamma(3,2)".into(),
                Box::new(move |x| d.pdf(x)),
                Box::new(move |x| d.cdf(x)),
                0.05,
                6.0,
            )
        },
        {
            let d = Beta::new(2.0, 5.0).unwrap();
            (
                "Beta(2,5)".into(),
                Box::new(move |x| d.pdf(x)),
                Box::new(move |x| d.cdf(x)),
                0.01,
                0.99,
            )
        },
        {
            let d = StudentT::new(4.0).unwrap();
            (
                "StudentT(4)".into(),
                Box::new(move |x| d.pdf(x)),
                Box::new(move |x| d.cdf(x)),
                -8.0,
                8.0,
            )
        },
        {
            let d = Wald::new(1.0, 3.0).unwrap();
            (
                "Wald(1,3)".into(),
                Box::new(move |x| d.pdf(x)),
                Box::new(move |x| d.cdf(x)),
                0.05,
                8.0,
            )
        },
        {
            let d = FisherSnedecor::new(6.0, 9.0).unwrap();
            (
                "FisherSnedecor(6,9)".into(),
                Box::new(move |x| d.pdf(x)),
                Box::new(move |x| d.cdf(x)),
                0.05,
                12.0,
            )
        },
        {
            let d = VonMises::new(0.0, 2.0).unwrap();
            // No CDF on VonMises, so integrate its density against itself: the
            // total over the circle must be 1. Handled in the density sweep;
            // here only the finite-interval form is exercised via Gumbel.
            let g = Gumbel::new(0.0, 1.0).unwrap();
            let _ = d;
            (
                "Gumbel(0,1)".into(),
                Box::new(move |x| g.pdf(x)),
                Box::new(move |x| g.cdf(x)),
                -4.0,
                10.0,
            )
        },
    ];
    for (name, pdf, cdf, lo, hi) in cases {
        let steps = 8;
        for i in 0..steps {
            let a = lo + (hi - lo) * i as f64 / steps as f64;
            let b = lo + (hi - lo) * (i + 1) as f64 / steps as f64;
            let integral = crate::simpson(&pdf, a, b, 4_000);
            f.close(
                || format!("{name}: ∫pdf over [{a:.4},{b:.4}] vs cdf difference"),
                integral,
                cdf(b) - cdf(a),
                1e-9,
            );
        }
    }
    f.finish("cdf increments vs density integral");
}

/// The hypergeometric is the sampling-without-replacement analogue of the
/// binomial, and its PMF has an exact symmetry under swapping the roles of
/// "successes" and "draws".
#[test]
fn the_hypergeometric_is_symmetric_in_successes_and_draws() {
    let mut f = Failures::default();
    for (population, successes, draws) in [(50_u64, 20_u64, 10_u64), (100, 7, 30), (12, 6, 6)] {
        let a = Hypergeometric::new(population, successes, draws).unwrap();
        // C(K,k)·C(N−K,n−k)/C(N,n) is unchanged when K and n are exchanged.
        let b = Hypergeometric::new(population, draws, successes).unwrap();
        let top = successes.min(draws);
        for k in 0..=top {
            f.close(
                || {
                    format!(
                        "Hypergeometric({population},{successes},{draws}) pmf({k}) \
                 vs swapped"
                    )
                },
                Discrete::pmf(&a, k),
                Discrete::pmf(&b, k),
                1e-11,
            );
        }
        f.close(
            || format!("Hypergeometric({population},{successes},{draws}) mean = n·K/N"),
            Moments::mean(&a),
            draws as f64 * successes as f64 / population as f64,
            1e-12,
        );
    }
    f.finish("Hypergeometric symmetry");
}

/// `BetaBinomial(n, a, b)` is a binomial whose success probability is itself a
/// `Beta(a, b)` draw, so its PMF must equal `∫₀¹ Binomial(n,p).pmf(k)·Beta(a,b).pdf(p) dp`.
#[test]
fn the_beta_binomial_is_a_beta_mixture_of_binomials() {
    let mut f = Failures::default();
    for (n, a, b) in [
        (10_u64, 2.0, 3.0),
        (5, 1.0, 1.0),
        (20, 0.5, 0.5),
        (8, 7.0, 2.0),
    ] {
        let bb = BetaBinomial::new(n, a, b).unwrap();
        let prior = Beta::new(a, b).unwrap();
        for k in 0..=n {
            // Integrate in the prior's **quantile** space: substituting p = Q(u)
            // turns ∫ g(p)·pdf(p) dp into ∫₀¹ g(Q(u)) du, which removes the
            // Beta density entirely. That matters here because `Beta(0.5, 0.5)`
            // has a p^(-1/2) singularity at each end that a uniform grid in p
            // resolves badly — it reported 0.2903 where the exact mass is
            // 0.12537, which is a defect in the probe, not in the library.
            let mixed = crate::simpson(
                |u| {
                    let p = prior.inverse_cdf(u);
                    if p <= 0.0 || p >= 1.0 {
                        return 0.0;
                    }
                    let bin = Binomial::new(n, p).expect("p in (0,1)");
                    Discrete::pmf(&bin, k)
                },
                1e-10,
                1.0 - 1e-10,
                20_000,
            );
            f.close(
                || format!("BetaBinomial({n},{a},{b}).pmf({k}) vs ∫Binomial(Q(u))du"),
                Discrete::pmf(&bb, k),
                mixed,
                1e-5,
            );
        }
        // Its mean is the binomial mean at the prior mean.
        f.close(
            || format!("BetaBinomial({n},{a},{b}) mean = n·a/(a+b)"),
            Moments::mean(&bb),
            n as f64 * a / (a + b),
            1e-11,
        );
    }
    f.finish("BetaBinomial = Beta-mixture of Binomial");
}
