//! Exact reparameterizations among the **discrete** families.
//!
//! The two that matter here are off-by-one conventions, which is exactly the
//! kind of thing that is invisible until something is compared against
//! something else: `Geometric` counts *trials until* the first success and so
//! starts at 1, while `NegativeBinomial` counts *failures before* the r-th and
//! starts at 0. The relation between them is the test that pins that down.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use crate::Failures;
use bayes_distribution_ext::{Moments, *};

const TOL: f64 = 1e-11;

#[test]
fn geometric_is_the_negative_binomial_of_one_success_shifted_by_one_trial() {
    let mut f = Failures::default();
    for p in [0.1, 0.5, 0.9] {
        let g = Geometric::new(p).unwrap();
        let nb = NegativeBinomial::new(1.0, p).unwrap();
        // `Geometric` counts *trials until* the first success (support from 1);
        // `NegativeBinomial` counts *failures before* the r-th success (from 0).
        // So they differ by exactly one trial.
        for k in 1_u64..12 {
            f.close(
                || {
                    format!(
                        "Geometric({p}) pmf({k}) = NegativeBinomial(1,{p}) pmf({})",
                        k - 1
                    )
                },
                Discrete::pmf(&g, k),
                Discrete::pmf(&nb, k - 1),
                TOL,
            );
            f.close(
                || {
                    format!(
                        "Geometric({p}) cdf({k}) = NegativeBinomial(1,{p}) cdf({})",
                        k - 1
                    )
                },
                DiscreteCdf::cdf(&g, k),
                DiscreteCdf::cdf(&nb, k - 1),
                TOL,
            );
        }
        f.close(
            || format!("Geometric({p}) mean = NegativeBinomial(1,{p}) mean + 1"),
            Moments::mean(&g),
            Moments::mean(&nb) + 1.0,
            TOL,
        );
        f.close(
            || format!("Geometric({p}) variance = NegativeBinomial(1,{p}) variance"),
            Moments::variance(&g),
            Moments::variance(&nb),
            TOL,
        );
    }
    f.finish("Geometric = shifted NegativeBinomial(1, p)");
}

#[test]
fn bernoulli_is_a_single_trial_binomial() {
    let mut f = Failures::default();
    for p in [0.0, 0.25, 0.5, 1.0] {
        let bern = Bernoulli::new(p).unwrap();
        let bin = Binomial::new(1, p).unwrap();
        for k in 0_u64..=1 {
            f.close(
                || format!("Bernoulli({p}) pmf({k}) = Binomial(1,{p}) pmf({k})"),
                Discrete::pmf(&bern, k),
                Discrete::pmf(&bin, k),
                TOL,
            );
            f.close(
                || format!("Bernoulli({p}) cdf({k}) = Binomial(1,{p}) cdf({k})"),
                DiscreteCdf::cdf(&bern, k),
                DiscreteCdf::cdf(&bin, k),
                TOL,
            );
        }
        f.close(
            || format!("Bernoulli({p}) variance = p(1−p)"),
            Moments::variance(&bern),
            p * (1.0 - p),
            TOL,
        );
    }
    f.finish("Bernoulli = Binomial(1, p)");
}
