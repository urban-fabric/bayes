//! **When will the pumps fail?** — fitting a lifetime distribution, reading
//! reliability off it, and checking the fit against the data.
//!
//! ```sh
//! cargo run -p bayes-statistics --example reliability
//! ```
//!
//! The Weibull is the workhorse here because its shape parameter *is* the failure
//! mode: `k < 1` is infant mortality (hazard falling), `k = 1` is memoryless
//! random failure (constant hazard — the exponential), `k > 1` is wear-out
//! (hazard rising). Reading `k` off a fit tells you which regime you are in, and
//! that changes what you should do about it.
//!
//! bayes ships no Weibull fitter, so this example also shows how to fit a family
//! yourself: linearize the CDF into a **Weibull plot** and regress. Every printed
//! number is asserted.

// An example's job is to be read, and threading `Result` through every line of a
// worked problem costs more clarity than the panic-freedom is worth here: every
// `expect` below is on a literal parameter that is valid by construction.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use bayes_distribution_ext::{Continuous, ContinuousCdf, Exponential, Moments, Weibull};
use bayes_statistics::{Ecdf, LinearRegression};
use rand::SeedableRng;
use rand::distr::Distribution;
use rand_chacha::ChaCha8Rng;

fn main() {
    // ---------------------------------------------------------------------
    // Observed lifetimes, in thousands of hours.
    // ---------------------------------------------------------------------
    // Generated from a known Weibull so the fit can be checked against truth —
    // in real use these would be field data.
    let truth = Weibull::new(12.0, 2.4).expect("valid Weibull");
    let mut rng = ChaCha8Rng::seed_from_u64(20_260_105);
    let lifetimes: Vec<f64> = (0..400).map(|_| truth.sample(&mut rng)).collect();

    println!(
        "Pump lifetimes: {} units, thousands of hours",
        lifetimes.len()
    );
    println!(
        "  generated from Weibull(scale = {}, shape = {})",
        truth.scale(),
        truth.shape()
    );

    // ---------------------------------------------------------------------
    // Fit by linearizing. Taking logs twice turns the Weibull CDF
    //     F(x) = 1 - exp(-(x/lambda)^k)
    // into a straight line in log-log coordinates:
    //     ln(-ln(1 - F)) = k*ln(x) - k*ln(lambda)
    // so an ordinary least-squares fit gives the shape as the slope and the
    // scale from the intercept. This is the classic "Weibull plot".
    // ---------------------------------------------------------------------
    let ecdf = Ecdf::new(&lifetimes);
    let sorted = ecdf.sorted();
    let n = sorted.len() as f64;
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (i, &x) in sorted.iter().enumerate() {
        // The median-rank plotting position, which is less biased at the ends
        // than i/n and avoids ln(0) at the last point.
        let fraction = (i as f64 + 1.0 - 0.3) / (n + 0.4);
        xs.push(x.ln());
        ys.push((-(1.0 - fraction).ln()).ln());
    }
    let line = LinearRegression::fit(&xs, &ys).expect("the plot supports a fit");
    let shape = line.slope;
    let scale = (-line.intercept / shape).exp();
    let fit = Weibull::new(scale, shape).expect("fitted parameters are valid");

    println!("\n-- fitted by Weibull plot --");
    println!(
        "  slope     = {:.4}  -> shape k = {:.4} (true {:.1})",
        line.slope,
        shape,
        truth.shape()
    );
    println!(
        "  intercept = {:.4}  -> scale L = {:.4} (true {:.1})",
        line.intercept,
        scale,
        truth.scale()
    );
    println!("  r^2 of the linearization = {:.5}", line.r_squared);
    println!("  k > 1, so this is wear-out: the hazard rises with age and");
    println!("  preventive replacement is worth doing.");

    assert!(
        line.r_squared > 0.99,
        "a Weibull plot of Weibull data should be near-linear"
    );
    assert!(
        (shape - truth.shape()).abs() < 0.2,
        "fitted shape {shape} should recover 2.4"
    );
    assert!(
        (scale - truth.scale()).abs() < 0.5,
        "fitted scale {scale} should recover 12.0"
    );
    assert!(shape > 1.0, "this data is a wear-out regime");

    // ---------------------------------------------------------------------
    // The numbers a reliability engineer actually asks for.
    // ---------------------------------------------------------------------
    println!("\n-- reliability --");
    println!("  MTTF (mean time to failure) = {:.3}k hours", fit.mean());
    println!("  median life (B50)           = {:.3}k hours", fit.median());

    // The "B10 life": the age by which 10% have failed — just a quantile.
    let b10: f64 = fit.inverse_cdf(0.10);
    let b01: f64 = fit.inverse_cdf(0.01);
    println!("  B10 life (10% failed)       = {b10:.3}k hours");
    println!("  B1 life  (1% failed)        = {b01:.3}k hours");
    assert!(b01 < b10 && b10 < fit.median(), "quantiles must be ordered");

    println!("\n  reliability R(t) = P(survive past t):");
    for t in [5.0, 10.0, 15.0, 20.0] {
        let r: f64 = fit.sf(t);
        println!("    R({t:>4.1}k) = {r:.4}");
    }
    let r0: f64 = fit.sf(0.0);
    assert!((r0 - 1.0).abs() < 1e-12, "everything survives age zero");
    let (r5, r10, r20): (f64, f64, f64) = (fit.sf(5.0), fit.sf(10.0), fit.sf(20.0));
    assert!(r5 > r10 && r10 > r20, "survival must fall");

    // The hazard rate h(t) = f(t)/R(t) is what distinguishes the regimes. For a
    // wear-out part it increases; a constant hazard would mean age tells you
    // nothing about imminent failure.
    println!("\n  hazard h(t) = f(t)/R(t), per thousand hours:");
    let mut previous = 0.0_f64;
    for t in [5.0, 10.0, 15.0, 20.0] {
        let density: f64 = fit.pdf(t);
        let survival: f64 = fit.sf(t);
        let h = density / survival;
        println!("    h({t:>4.1}k) = {h:.5}");
        assert!(h > previous, "hazard must rise for k > 1");
        previous = h;
    }

    // Conditional reliability: given it has already run 10k hours, what is the
    // chance of another 5k? For a Weibull this is *not* R(5) — that is the whole
    // point of a non-constant hazard.
    let r15: f64 = fit.sf(15.0);
    let conditional = r15 / r10;
    println!("\n  P(survive to 15k | survived 10k) = {conditional:.4}");
    println!("  P(survive to  5k | new)          = {r5:.4}");
    println!("  -- a used pump is *more* likely to fail soon; that is wear-out.");
    assert!(
        conditional < r5,
        "for k > 1 a used part has worse prospects than a new one"
    );

    // ---------------------------------------------------------------------
    // The memoryless contrast, which is the relation worth remembering.
    // ---------------------------------------------------------------------
    println!("\n-- contrast: the exponential is memoryless --");
    let exp = Exponential::new(1.0 / fit.mean()).expect("positive rate");
    let (e10, e15, e5): (f64, f64, f64) = (exp.sf(10.0), exp.sf(15.0), exp.sf(5.0));
    let exp_conditional = e15 / e10;
    println!("  Exponential with the same MTTF:");
    println!("    P(survive to 15k | survived 10k) = {exp_conditional:.4}");
    println!("    P(survive to  5k | new)          = {e5:.4}");
    println!("  -- identical, because an exponential forgets its age.");
    assert!(
        (exp_conditional - e5).abs() < 1e-12,
        "the exponential is memoryless: P(T>s+t | T>s) = P(T>t)"
    );

    // And a Weibull with shape exactly 1 *is* that exponential — the relation
    // that ties the two families together.
    let shape_one = Weibull::new(fit.mean(), 1.0).expect("valid Weibull");
    for t in [1.0, 5.0, 20.0] {
        let a: f64 = shape_one.sf(t);
        let b: f64 = exp.sf(t);
        assert!(
            (a - b).abs() < 1e-12,
            "Weibull(mean, 1) must equal Exponential(1/mean)"
        );
    }
    println!("  (and Weibull(L, k=1) is exactly that exponential -- checked)");

    // ---------------------------------------------------------------------
    // Does the fit actually describe the data? Compare empirical to fitted
    // quantiles at each decile.
    // ---------------------------------------------------------------------
    println!("\n-- goodness of fit --");
    let mut worst_gap = 0.0_f64;
    for decile in 1..10 {
        let p = decile as f64 / 10.0;
        let empirical = ecdf.quantile(p);
        let fitted: f64 = fit.inverse_cdf(p);
        worst_gap = worst_gap.max((empirical - fitted).abs());
        println!("    p={p:.1}: empirical {empirical:>6.3}k  fitted {fitted:>6.3}k");
    }
    println!("  worst quantile gap: {worst_gap:.4}k hours");
    assert!(worst_gap < 1.0, "the fit tracks the data to under 1k hours");

    println!("\nall assertions held");
}
