//! **Is the line still in control?** — statistical process control: limits,
//! capability, and detecting a shift.
//!
//! ```sh
//! cargo run -p bayes-statistics --example quality_control
//! ```
//!
//! Three questions a process engineer asks, each a different part of the library:
//!
//! 1. **Is it stable?** Control limits from the in-control distribution, and the
//!    false-alarm rate they imply.
//! 2. **Is it good enough?** Capability indices — the fraction out of spec, which
//!    is a CDF evaluation, not a guess.
//! 3. **Has it moved?** A two-sample test against the baseline, and the
//!    *run-length* of the control chart, which is where the Geometric shows up.
//!
//! The defect-count half uses a Poisson, where the control limits have to come
//! from the discrete quantile rather than from "mean ± 3 sigma" — the normal
//! approximation is poor at low counts and the example shows by how much.

// An example's job is to be read, and threading `Result` through every line of a
// worked problem costs more clarity than the panic-freedom is worth here: every
// `expect` below is on a literal parameter that is valid by construction.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use bayes_distribution_ext::{
    ContinuousCdf, Discrete, DiscreteCdf, Geometric, Moments, Normal, Poisson,
};
use bayes_statistics::hypothesis::{two_sample_t, variance_ratio_f};
use bayes_statistics::{RunningStats, mean, std_dev, variance};
use rand::SeedableRng;
use rand::distr::Distribution;
use rand_chacha::ChaCha8Rng;

fn main() {
    // Shaft diameter in millimetres. Spec is 20.00 +/- 0.15.
    let (target, spec_half_width) = (20.00_f64, 0.15_f64);
    let mut rng = ChaCha8Rng::seed_from_u64(20_260_105);

    // ---------------------------------------------------------------------
    // 1. Baseline: 200 measurements from a stable process.
    // ---------------------------------------------------------------------
    let in_control = Normal::new(20.002, 0.040).expect("positive sd");
    let baseline: Vec<f64> = (0..200).map(|_| in_control.sample(&mut rng)).collect();

    println!("Shaft diameter, spec {target} +/- {spec_half_width} mm");
    println!("\n-- 1. baseline (n = {}) --", baseline.len());
    let (mu, sd) = (mean(&baseline), std_dev(&baseline));
    println!("  mean {mu:.5} mm, sd {sd:.5} mm");

    // Shewhart limits: three standard deviations either side. The "3" is not
    // arbitrary — it fixes the false-alarm rate, which is worth stating.
    let (lcl, ucl) = (mu - 3.0 * sd, mu + 3.0 * sd);
    println!("  control limits: [{lcl:.5}, {ucl:.5}]");
    let z = Normal::new(0.0, 1.0).expect("standard normal");
    let false_alarm: f64 = 2.0 * z.sf(3.0);
    println!("  implied false-alarm rate per point: {false_alarm:.6}");
    println!("    = 1 in {:.0} points", 1.0 / false_alarm);
    assert!(
        (false_alarm - 0.002_699_796).abs() < 1e-8,
        "two-sided 3-sigma tail"
    );

    // The average run length to a false alarm is Geometric in that rate, so
    // its mean is 1/p — which the library can confirm rather than us asserting.
    let runs = Geometric::new(false_alarm).expect("probability in (0,1]");
    println!(
        "  average run length to a false alarm: {:.0} points",
        Moments::mean(&runs)
    );
    assert!(
        (Moments::mean(&runs) - 1.0 / false_alarm).abs() < 1e-6,
        "a Geometric's mean is 1/p"
    );

    // How many of the baseline points actually fall outside? Should be ~none.
    let outside = baseline.iter().filter(|&&x| x < lcl || x > ucl).count();
    println!(
        "  baseline points outside the limits: {outside} of {}",
        baseline.len()
    );
    assert!(
        outside <= 2,
        "a stable process should rarely trip its own limits"
    );

    // ---------------------------------------------------------------------
    // 2. Capability: what fraction is out of spec? A CDF question.
    // ---------------------------------------------------------------------
    println!("\n-- 2. capability --");
    let (lsl, usl) = (target - spec_half_width, target + spec_half_width);
    let fitted = Normal::new(mu, sd).expect("positive sd");

    // Cp ignores centring; Cpk penalizes it. The difference is the whole point.
    let cp = (usl - lsl) / (6.0 * sd);
    let cpk = ((usl - mu) / (3.0 * sd)).min((mu - lsl) / (3.0 * sd));
    println!("  Cp  = {cp:.4}   (spread only)");
    println!("  Cpk = {cpk:.4}   (spread and centring)");
    assert!(cpk <= cp + 1e-12, "Cpk can never exceed Cp");

    // The actual out-of-spec fraction, from the fitted distribution rather than
    // from a capability index. This is what Cpk is a proxy for.
    let below: f64 = fitted.cdf(lsl);
    let above: f64 = fitted.sf(usl);
    let defect_rate = below + above;
    println!("  P(below LSL) = {below:.3e}");
    println!("  P(above USL) = {above:.3e}");
    println!(
        "  total out of spec = {defect_rate:.3e} = {:.1} parts per million",
        defect_rate * 1e6
    );
    assert!(defect_rate > 0.0 && defect_rate < 1.0);

    // A Cpk of 1.0 corresponds to 3 sigma on the nearer side, i.e. a one-sided
    // tail of ~0.00135. Checking that correspondence ties the index to the CDF.
    let implied_tail: f64 = z.sf(3.0 * cpk);
    println!("  nearer-side tail implied by Cpk: {implied_tail:.3e}");
    assert!(
        (implied_tail - below.max(above)).abs() < 1e-9,
        "Cpk must correspond to the nearer-side tail probability"
    );

    // ---------------------------------------------------------------------
    // 3. A shift arrives. Can we see it?
    // ---------------------------------------------------------------------
    println!("\n-- 3. after a tool change --");
    // The mean moves by half a standard deviation and the spread grows.
    let shifted = Normal::new(20.022, 0.052).expect("positive sd");
    let after: Vec<f64> = (0..200).map(|_| shifted.sample(&mut rng)).collect();
    println!(
        "  mean {:.5} mm, sd {:.5} mm",
        mean(&after),
        std_dev(&after)
    );

    let shift_test = two_sample_t(&baseline, &after);
    println!(
        "  two-sample t: t = {:.4}, p = {:.3e}",
        shift_test.statistic, shift_test.p_value
    );
    assert!(
        shift_test.p_value < 0.01,
        "a half-sigma shift at n = 200 is detectable"
    );

    // The spread changed too, which a mean test cannot see. That needs an F test.
    let spread_test = variance_ratio_f(&baseline, &after);
    println!(
        "  variance ratio F: F = {:.4}, p = {:.4}",
        spread_test.statistic, spread_test.p_value
    );
    assert!(
        (spread_test.statistic - variance(&baseline) / variance(&after)).abs() < 1e-12,
        "the F statistic is the variance ratio"
    );

    // Streaming detection: how many points before the chart trips? A control
    // chart is run online, so this is the realistic form.
    let mut chart = RunningStats::new();
    let mut tripped_at = None;
    for (i, &x) in after.iter().enumerate() {
        if x < lcl || x > ucl {
            tripped_at = Some(i + 1);
            break;
        }
        chart.push(x);
    }
    match tripped_at {
        Some(i) => println!("  the chart tripped at point {i}"),
        None => println!("  the chart never tripped in 200 points"),
    }
    // The out-of-control rate per point, from the shifted distribution.
    let trip_rate: f64 = shifted.cdf(lcl) + shifted.sf(ucl);
    let detection = Geometric::new(trip_rate).expect("probability in (0,1]");
    println!("  P(trip | shifted) = {trip_rate:.5} per point");
    println!(
        "  expected points to detection: {:.1}",
        Moments::mean(&detection)
    );
    println!("  — a Shewhart chart is slow at small shifts; that is why CUSUM exists.");
    assert!(
        trip_rate > false_alarm,
        "a shifted process must trip more often"
    );

    // ---------------------------------------------------------------------
    // 4. Defect counts: the discrete half, where "mean +/- 3 sigma" misleads.
    // ---------------------------------------------------------------------
    println!("\n-- 4. defects per panel (count data) --");
    let rate = 2.5;
    let defects = Poisson::new(rate).expect("positive rate");
    println!(
        "  in control: Poisson({rate}), mean {:.3}, sd {:.4}",
        Moments::mean(&defects),
        Moments::std_dev(&defects)
    );

    println!("\n  exact tail probabilities:");
    let mut cumulative = 0.0;
    for k in 0_u64..=8 {
        cumulative += Discrete::pmf(&defects, k);
        println!(
            "    P(X = {k}) = {:.5}   P(X <= {k}) = {:.5}",
            Discrete::pmf(&defects, k),
            DiscreteCdf::cdf(&defects, k)
        );
        // The cdf is the running sum of the pmf — the same identity the test
        // suite checks across every discrete family.
        assert!(
            (DiscreteCdf::cdf(&defects, k) - cumulative).abs() < 1e-12,
            "the cdf is the running sum of the pmf"
        );
    }

    // The limit that matters: "mean + 3 sigma" is a normal-theory rule, and a
    // Poisson is not normal at these counts. Comparing the two across rates
    // shows the approximation lands **one count low** almost everywhere, so a
    // chart built on it raises materially more false alarms than it advertises.
    const NOMINAL: f64 = 0.00135; // the one-sided 3-sigma tail it is aiming for
    println!("\n  upper control limit, normal rule vs exact quantile:");
    println!("     rate   3-sigma rule   exact q   P(>exact)   P(>rule)");
    for lambda in [0.5_f64, 1.0, 2.5, 4.0, 9.0, 25.0, 100.0] {
        let d = Poisson::new(lambda).expect("positive rate");
        let rule = (lambda + 3.0 * lambda.sqrt()).ceil() as u64;
        let exact: u64 = DiscreteCdf::inverse_cdf(&d, 1.0 - NOMINAL);
        let tail_exact = DiscreteCdf::sf(&d, exact);
        let tail_rule = DiscreteCdf::sf(&d, rule);
        println!(
            "    {lambda:>6}   {rule:>12}   {exact:>7}   {tail_exact:>9.5}   {tail_rule:>8.5}"
        );

        // `inverse_cdf` is the smallest k reaching the nominal coverage, so the
        // tail beyond it cannot exceed the nominal rate. That is the guarantee
        // the normal rule does not give you.
        assert!(
            tail_exact <= NOMINAL + 1e-12,
            "the exact quantile must hold the nominal false-alarm rate at rate {lambda}"
        );
        // And the normal rule never sits *above* the exact quantile, so it can
        // only ever be anti-conservative, never conservative.
        assert!(
            rule <= exact,
            "the 3-sigma rule should not exceed the exact quantile at rate {lambda}"
        );
        assert!(
            tail_rule >= tail_exact - 1e-12,
            "a lower limit cannot have a smaller tail at rate {lambda}"
        );
    }
    println!("  -- the rule is one count low almost everywhere. At rate 1.0 its");
    println!("     true false-alarm rate is 0.0037, nearly three times the 0.00135");
    println!("     it claims, and it is still 1.3x off at rate 100.");

    // Spell that out for the worst case in the table.
    let low = Poisson::new(1.0).expect("positive rate");
    let rule_low = (1.0 + 3.0_f64.sqrt() * 1.0).ceil() as u64;
    let exact_low: u64 = DiscreteCdf::inverse_cdf(&low, 1.0 - NOMINAL);
    assert!(
        DiscreteCdf::sf(&low, rule_low) > 2.0 * NOMINAL,
        "at rate 1 the normal rule more than doubles the false-alarm rate"
    );
    assert!(
        exact_low > rule_low,
        "and the exact limit is strictly higher"
    );

    // A doubling of the defect rate, and the exact chance of catching it.
    let exact_ucl: u64 = DiscreteCdf::inverse_cdf(&defects, 1.0 - NOMINAL);
    let worse = Poisson::new(5.0).expect("positive rate");
    println!("\n  if the rate doubles from {rate} to 5.0:");
    println!(
        "    P(exceed the exact UCL of {exact_ucl}) rises from {:.5} to {:.4}",
        DiscreteCdf::sf(&defects, exact_ucl),
        DiscreteCdf::sf(&worse, exact_ucl)
    );
    let detect = Geometric::new(DiscreteCdf::sf(&worse, exact_ucl)).expect("probability in (0,1]");
    println!(
        "    expected panels to detection: {:.0}",
        Moments::mean(&detect)
    );
    assert!(
        DiscreteCdf::sf(&worse, exact_ucl) > DiscreteCdf::sf(&defects, exact_ucl),
        "a worse process must exceed the limit more often"
    );

    println!("\nall assertions held");
}
