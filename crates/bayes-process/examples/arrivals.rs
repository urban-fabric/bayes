//! **How many support tickets will arrive before lunch?** — a Poisson process,
//! and the three facts about it that make it worth knowing.
//!
//! ```sh
//! cargo run -p bayes-process --example arrivals
//! ```
//!
//! A Poisson process is pinned down by one rate `λ`, and everything else follows:
//!
//! 1. the **count** in a window of width `t` is `Poisson(λt)`
//! 2. the **gaps** between arrivals are `Exponential(λ)`, independently
//! 3. given the count in a window, the arrival **times** are uniform in it
//!
//! Each of those is simulated and then checked against the closed form, which is
//! also a cross-check between `bayes-process` and `bayes-distribution-ext`.
//! Every printed number is asserted.

// An example's job is to be read, and threading `Result` through every line of a
// worked problem costs more clarity than the panic-freedom is worth here: every
// `expect` below is on a literal parameter that is valid by construction.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use bayes_distribution_ext::{ContinuousCdf, Discrete, DiscreteCdf, Exponential, Moments, Poisson};
use bayes_process::{
    GaussianRandomWalk, Kernel, Matern, MaternNu, PoissonProcess, Rbf, wiener_path,
};
use bayes_statistics::hypothesis::{chi_squared_goodness_of_fit, ks_one_sample};
use bayes_statistics::{mean, variance};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn main() {
    let rate = 4.5; // tickets per hour
    let horizon = 4.0; // a four-hour morning
    let process = PoissonProcess::new(rate).expect("positive rate");
    let mut rng = ChaCha8Rng::seed_from_u64(20_260_105);

    println!("Poisson arrivals at {rate}/hour over a {horizon}-hour window");

    // ---------------------------------------------------------------------
    // Fact 1: the count in the window is Poisson(rate * horizon).
    // ---------------------------------------------------------------------
    println!("\n-- the count --");
    let expected = Poisson::new(rate * horizon).expect("positive mean");
    println!("  closed form: Poisson({}) ", rate * horizon);
    println!(
        "    mean {:.3}, variance {:.3}",
        expected.mean(),
        expected.variance()
    );
    println!(
        "    P(exactly 18) = {:.5}",
        Discrete::pmf(&expected, 18_u64)
    );
    println!(
        "    P(at most 12) = {:.5}",
        DiscreteCdf::cdf(&expected, 12_u64)
    );
    println!(
        "    P(over 25)    = {:.5}",
        DiscreteCdf::sf(&expected, 25_u64)
    );

    // A Poisson has variance equal to its mean — the signature property, and the
    // reason overdispersed count data needs a negative binomial instead.
    assert!(
        (expected.mean() - expected.variance()).abs() < 1e-12,
        "a Poisson's variance equals its mean"
    );

    // Simulate many mornings and check the empirical count distribution.
    let mornings = 4_000;
    let counts: Vec<f64> = (0..mornings)
        .map(|_| process.arrivals(horizon, &mut rng).len() as f64)
        .collect();
    println!("  simulated {mornings} mornings:");
    println!(
        "    sample mean     {:.3} (closed form {:.3})",
        mean(&counts),
        expected.mean()
    );
    println!(
        "    sample variance {:.3} (closed form {:.3})",
        variance(&counts),
        expected.variance()
    );
    assert!(
        (mean(&counts) - expected.mean()).abs() < 0.2,
        "simulated mean matches"
    );
    assert!(
        (variance(&counts) - expected.variance()).abs() < 1.5,
        "simulated variance matches"
    );

    // A goodness-of-fit test against the closed-form masses. Buckets are the
    // counts 8..=28, with the tails pooled so no expected cell is tiny.
    let (lo, hi) = (8_u64, 28_u64);
    let mut observed = vec![0.0; (hi - lo + 3) as usize];
    for &c in &counts {
        let k = c as u64;
        let idx = if k < lo {
            0
        } else if k > hi {
            observed.len() - 1
        } else {
            (k - lo + 1) as usize
        };
        observed[idx] += 1.0;
    }
    let total = mornings as f64;
    let mut expected_cells = vec![0.0; observed.len()];
    expected_cells[0] = DiscreteCdf::cdf(&expected, lo - 1) * total;
    expected_cells[observed.len() - 1] = DiscreteCdf::sf(&expected, hi) * total;
    for k in lo..=hi {
        expected_cells[(k - lo + 1) as usize] = Discrete::pmf(&expected, k) * total;
    }
    let gof = chi_squared_goodness_of_fit(&observed, &expected_cells);
    println!("    chi-squared goodness of fit: p = {:.4}", gof.p_value);
    assert!(
        gof.p_value > 0.01,
        "the simulated counts match the closed form (p = {})",
        gof.p_value
    );

    // ---------------------------------------------------------------------
    // Fact 2: the gaps between arrivals are Exponential(rate).
    // ---------------------------------------------------------------------
    println!("\n-- the gaps --");
    let gap_law = Exponential::new(rate).expect("positive rate");
    println!("  closed form: Exponential({rate})");
    println!(
        "    mean gap {:.4} h = {:.1} minutes",
        gap_law.mean(),
        gap_law.mean() * 60.0
    );
    println!(
        "    P(next within 10 min) = {:.4}",
        gap_law.cdf(10.0 / 60.0)
    );

    // Collect gaps from a long run and test them against the exponential.
    let long = process.arrivals(2_000.0, &mut rng);
    let gaps: Vec<f64> = long.windows(2).map(|w| w[1] - w[0]).collect();
    println!("  {} gaps from a 2000-hour run:", gaps.len());
    println!(
        "    sample mean gap {:.4} (closed form {:.4})",
        mean(&gaps),
        gap_law.mean()
    );
    let ks = ks_one_sample(&gaps, |x| gap_law.cdf(x));
    println!(
        "    KS against Exponential({rate}): D = {:.4}, p = {:.4}",
        ks.statistic, ks.p_value
    );
    assert!(
        (mean(&gaps) - gap_law.mean()).abs() < 0.01,
        "mean gap matches 1/rate"
    );
    assert!(
        ks.p_value > 0.01,
        "gaps are exponential (KS p = {})",
        ks.p_value
    );

    // Memorylessness: having waited 20 minutes tells you nothing about the next
    // 10. This is the property that makes the process "no memory".
    let waited: f64 = gap_law.sf(30.0 / 60.0) / gap_law.sf(20.0 / 60.0);
    let fresh: f64 = gap_law.sf(10.0 / 60.0);
    println!("    P(wait > 30min | waited 20min) = {waited:.6}");
    println!("    P(wait > 10min | fresh)        = {fresh:.6}");
    assert!(
        (waited - fresh).abs() < 1e-12,
        "the exponential is memoryless"
    );

    // ---------------------------------------------------------------------
    // Fact 3: conditional on the count, arrival times are uniform.
    // ---------------------------------------------------------------------
    println!("\n-- the times --");
    let one_morning = process.arrivals(horizon, &mut rng);
    println!(
        "  one morning had {} arrivals at (hours):",
        one_morning.len()
    );
    print!("   ");
    for t in &one_morning {
        print!(" {t:.2}");
    }
    println!();
    assert!(
        one_morning.windows(2).all(|w| w[0] <= w[1]),
        "arrival times come back sorted"
    );
    assert!(
        one_morning.iter().all(|&t| (0.0..=horizon).contains(&t)),
        "arrivals lie inside the window"
    );

    // Pooling arrival times across many windows should look uniform on [0, horizon).
    let mut pooled = Vec::new();
    for _ in 0..1_500 {
        pooled.extend(process.arrivals(horizon, &mut rng));
    }
    let uniform_ks = ks_one_sample(&pooled, |x| (x / horizon).clamp(0.0, 1.0));
    println!(
        "  {} pooled times, KS against Uniform(0,{horizon}): p = {:.4}",
        pooled.len(),
        uniform_ks.p_value
    );
    assert!(
        uniform_ks.p_value > 0.01,
        "arrival times are uniform in the window"
    );

    // ---------------------------------------------------------------------
    // Brownian motion, for comparison: the other basic process.
    // ---------------------------------------------------------------------
    println!("\n-- Brownian motion --");
    let times: Vec<f64> = (0..=1_000).map(|i| i as f64 * 0.01).collect();
    let path = wiener_path(&times, 1.0, &mut rng);
    let increments: Vec<f64> = path.windows(2).map(|w| w[1] - w[0]).collect();
    println!("  1000 steps of dt = 0.01, diffusion 1.0");
    println!("    increment mean     {:.5} (expect 0)", mean(&increments));
    println!(
        "    increment variance {:.5} (expect dt = 0.01)",
        variance(&increments)
    );
    assert!(path[0] == 0.0, "a Wiener path starts at the origin");
    assert!(mean(&increments).abs() < 0.01, "increments are centred");
    assert!(
        (variance(&increments) - 0.01).abs() < 0.002,
        "increment variance is dt"
    );

    // A random walk with the same step size tracks the same variance growth.
    let walk = GaussianRandomWalk::new(0.0, 0.1);
    let trajectory = walk.trajectory(1_000, &mut rng);
    let steps: Vec<f64> = trajectory.windows(2).map(|w| w[1] - w[0]).collect();
    println!(
        "    random-walk step sd {:.5} (expect 0.1)",
        variance(&steps).sqrt()
    );
    assert!((variance(&steps).sqrt() - 0.1).abs() < 0.01);

    // ---------------------------------------------------------------------
    // Covariance kernels, if you are going on to a Gaussian process.
    // ---------------------------------------------------------------------
    println!("\n-- GP covariance kernels --");
    let rbf = Rbf::new(2.0, 1.5);
    let matern = Matern::new(2.0, 1.5, MaternNu::ThreeHalves);
    println!(
        "  Rbf(var 2, len 1.5):     k(x,x) = {:.4}",
        rbf.covariance(&[0.3], &[0.3])
    );
    println!(
        "  Matern(nu=3/2, same):    k(x,x) = {:.4}",
        matern.covariance(&[0.3], &[0.3])
    );
    // At zero separation every stationary kernel returns the variance.
    let rbf_at_zero: f64 = rbf.covariance(&[0.3], &[0.3]);
    let matern_at_zero: f64 = matern.covariance(&[0.3], &[0.3]);
    assert!((rbf_at_zero - 2.0).abs() < 1e-12);
    assert!((matern_at_zero - 2.0).abs() < 1e-12);
    // Covariance decays with distance and never exceeds the variance.
    let mut previous = f64::INFINITY;
    for d in [0.0, 0.5, 1.0, 2.0, 5.0] {
        let k: f64 = rbf.covariance(&[0.0], &[d]);
        assert!(
            k <= 2.0 + 1e-12 && k < previous,
            "Rbf must decay from the variance"
        );
        previous = k;
    }
    // Matern(1/2) is the exponential kernel, var * exp(-d/len) — a relation worth
    // checking because it ties the family to a closed form.
    let m_half = Matern::new(2.0, 1.5, MaternNu::Half);
    for d in [0.25, 1.0, 3.0] {
        let expected_k = 2.0 * (-d / 1.5_f64).exp();
        let got: f64 = m_half.covariance(&[0.0], &[d]);
        assert!(
            (got - expected_k).abs() < 1e-12,
            "Matern(nu=1/2) is the exponential kernel"
        );
    }
    println!("  (Matern(nu=1/2) == var*exp(-d/len) -- checked)");

    println!("\nall assertions held");
}
