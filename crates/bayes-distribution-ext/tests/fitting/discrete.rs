//! Parameter recovery for the discrete families.
//!
//! The samplers return integers of various widths — `bool` for a Bernoulli,
//! `i64` for a discrete uniform, `u64` for the rest — so each case converts to
//! `f64`, which is what [`Estimate`] consumes.

use super::harness::{Case, case, recovers_its_parameters};
use bayes_distribution_ext::{
    Bernoulli, Categorical, DiscreteUniform, Estimate, Geometric, NegativeBinomial, Poisson,
};
use rand::distr::Distribution;

fn cases() -> Vec<Case> {
    vec![
        case(
            "Bernoulli(0.35)",
            &["p"],
            &[0.35],
            |rng| {
                let draw: bool = Bernoulli::new(0.35).unwrap().sample(rng);
                f64::from(draw)
            },
            |d| Bernoulli::from_sample(d).map(|x| vec![x.p()]),
        ),
        case(
            "Categorical([0.2, 0.5, 0.3])",
            &["p0", "p1", "p2"],
            &[0.2, 0.5, 0.3],
            |rng| {
                let draw: u64 = Categorical::new(&[0.2, 0.5, 0.3]).unwrap().sample(rng);
                draw as f64
            },
            |d| Categorical::from_sample(d).map(|x| x.probabilities().to_vec()),
        ),
        // Support `{2, …, 8}`, since `new(lo, hi)` is half-open. The estimator
        // returns `max + 1` for `hi`, so recovering 9 here is the check that it
        // respects that convention rather than quietly dropping the top value.
        //
        // Both bounds are exact once every value has been seen, which at
        // `n = 2000` over seven values is overwhelming — hence no bias
        // allowance, unlike the continuous uniform.
        case(
            "DiscreteUniform(2, 9)",
            &["lo", "hi"],
            &[2.0, 9.0],
            |rng| {
                let draw: i64 = DiscreteUniform::new(2, 9).unwrap().sample(rng);
                draw as f64
            },
            |d| DiscreteUniform::from_sample(d).map(|x| vec![x.lo() as f64, x.hi() as f64]),
        ),
        // `1/x̄` is a nonlinear function of the sample mean, so Jensen gives it
        // an `O(1/n)` upward bias of about `p(1 − p)/n` — about `1e-4` here,
        // comfortably inside the four standard errors below.
        case(
            "Geometric(0.3)",
            &["p"],
            &[0.3],
            |rng| {
                let draw: u64 = Geometric::new(0.3).unwrap().sample(rng);
                draw as f64
            },
            |d| Geometric::from_sample(d).map(|x| vec![x.p()]),
        ),
        case(
            "NegativeBinomial(5, 0.4)",
            &["successes", "p"],
            &[5.0, 0.4],
            |rng| {
                let draw: u64 = NegativeBinomial::new(5.0, 0.4).unwrap().sample(rng);
                draw as f64
            },
            |d| NegativeBinomial::from_sample(d).map(|x| vec![x.successes(), x.p()]),
        ),
        case(
            "Poisson(4)",
            &["lambda"],
            &[4.0],
            |rng| {
                let draw: u64 = Poisson::new(4.0).unwrap().sample(rng);
                draw as f64
            },
            |d| Poisson::from_sample(d).map(|x| vec![x.lambda()]),
        ),
    ]
}

#[test]
fn the_discrete_families_recover_their_own_parameters() {
    recovers_its_parameters(cases(), 20_261_106, "discrete parameter recovery");
}
