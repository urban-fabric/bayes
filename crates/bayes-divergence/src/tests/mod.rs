#[allow(unused_imports)]
use super::*;
#[allow(unused_imports)]
use crate::{
    Bernoulli, Beta, Categorical, Cauchy, Continuous, Discrete, Exponential, Gamma, Geometric,
    Laplace, LogNormal, Moments, Normal, Pareto, Poisson, Rayleigh, Weibull,
};

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
}

/// Numerical Bhattacharyya coefficient `∫ √(f·g)` by the trapezoid rule, for
/// validating the closed-form Hellinger affinities.
fn bc_integral(f: impl Fn(f64) -> f64, g: impl Fn(f64) -> f64, lo: f64, hi: f64) -> f64 {
    let n = 100_000usize;
    let h = (hi - lo) / n as f64;
    let mut s = 0.0;
    for k in 0..=n {
        let x = lo + k as f64 * h;
        let w = if k == 0 || k == n { 0.5 } else { 1.0 };
        s += w * (f(x) * g(x)).max(0.0).sqrt();
    }
    s * h
}

mod part1;
mod part2;
