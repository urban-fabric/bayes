//! Intervals, bounds, order statistics, and resampling.
//!
//! A confidence interval is a quantile of a reference distribution, so it is
//! checkable against that quantile directly. A concentration bound is only
//! useful if it actually *dominates* the true tail — a bound that sits below the
//! truth is worse than no bound — so each is checked against a distribution
//! whose tail we can compute exactly.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use bayes_distribution_ext::{ContinuousCdf, *};
use bayes_statistics::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use bayes_utils::Failures;

const SAMPLE: [f64; 12] = [4.1, 5.2, 3.8, 6.0, 4.7, 5.5, 4.2, 5.9, 3.3, 6.4, 4.9, 5.1];

/// Concentration bounds are only useful if they actually hold. Each is checked
/// against the true tail probability of a distribution we can compute exactly —
/// a bound that sits *below* the truth is worse than no bound at all.
#[test]
fn concentration_bounds_dominate_the_true_tail_probability() {
    let mut f = Failures::default();
    let n = Normal::new(0.0, 1.0).unwrap();

    // Chebyshev: P(|X − μ| ≥ t) ≤ min(σ²/t², 1). For a standard normal σ² = 1.
    for t in [1.5, 2.0, 3.0, 5.0] {
        let bound = chebyshev_bound(1.0, t);
        let truth = 2.0 * n.sf(t);
        f.check(bound >= truth, || {
            format!("chebyshev_bound(1,{t}) = {bound} is below the true normal tail {truth}")
        });
        f.close(
            || format!("chebyshev_bound(1,{t}) = min(1/t², 1)"),
            bound,
            (1.0 / (t * t)).min(1.0),
            1e-13,
        );
    }

    // Hoeffding: for X confined to [a, b], P(X − μ ≥ t) ≤ exp(−2t²/(b−a)²).
    // Checked against the exact tail of a uniform on the same interval.
    for (a, b) in [(0.0, 1.0), (-2.0, 3.0)] {
        let u = Uniform::new(a, b).unwrap();
        let mu = (a + b) / 2.0;
        for t in [0.1, 0.25, 0.4] {
            let scaled = t * (b - a);
            let bound = hoeffding_bound(a, b, scaled);
            let truth = u.sf(mu + scaled);
            f.check(bound >= truth, || {
                format!(
                    "hoeffding_bound({a},{b},{scaled}) = {bound} is below the \
                     true uniform tail {truth}"
                )
            });
            f.close(
                || format!("hoeffding_bound({a},{b},{scaled}) = exp(−2t²/(b−a)²)"),
                bound,
                (-2.0 * scaled * scaled / (b - a).powi(2)).exp(),
                1e-13,
            );
        }
    }

    // Markov on a nonnegative variable: P(X ≥ a) ≤ min(E[X]/a, 1). Checked
    // against the exact tail of an exponential with that mean.
    for (expectation, a) in [(1.0, 4.0), (2.5, 10.0), (3.0, 1.0)] {
        let bound = markov_bound(expectation, a);
        f.close(
            || format!("markov_bound({expectation},{a}) = min(E/a, 1)"),
            bound,
            (expectation / a).min(1.0),
            1e-13,
        );
        let e = Exponential::new(1.0 / expectation).unwrap();
        f.check(bound >= e.sf(a), || {
            format!(
                "markov_bound({expectation},{a}) = {bound} is below the \
                 exponential tail {}",
                e.sf(a)
            )
        });
    }

    // Popoviciu: a variable confined to [a, b] has variance at most (b−a)²/4,
    // and a two-point sample at the extremes attains it exactly.
    for (a, b) in [(0.0, 1.0), (-3.0, 7.0)] {
        let cap = max_variance_for_range(a, b);
        f.close(
            || format!("max_variance_for_range({a},{b}) = (b−a)²/4"),
            cap,
            (b - a).powi(2) / 4.0,
            1e-13,
        );
        f.close(
            || format!("a two-point sample at {a},{b} attains the cap"),
            population_variance(&[a, b]),
            cap,
            1e-12,
        );
        // And the uniform on the same interval, which has variance (b−a)²/12,
        // must sit under it.
        f.check(Uniform::new(a, b).unwrap().variance() <= cap, || {
            "the uniform variance exceeded the Popoviciu cap".into()
        });
    }

    f.finish("concentration bounds");
}

/// Order statistics, the ECDF, and the quantile function describe the same
/// sample from three directions.
#[test]
fn order_statistics_and_the_ecdf_agree() {
    let mut f = Failures::default();
    let e = Ecdf::new(&SAMPLE);

    f.close(
        || "median vs quantile(0.5)".into(),
        median(&SAMPLE),
        quantile(&SAMPLE, 0.5),
        1e-13,
    );
    f.close(
        || "iqr vs quantile(.75) − quantile(.25)".into(),
        iqr(&SAMPLE),
        quantile(&SAMPLE, 0.75) - quantile(&SAMPLE, 0.25),
        1e-13,
    );

    // The ECDF at each order statistic is the rank fraction.
    let sorted = e.sorted().to_vec();
    for (i, &x) in sorted.iter().enumerate() {
        f.close(
            || format!("Ecdf at order statistic {i}"),
            e.eval(x),
            (i + 1) as f64 / sorted.len() as f64,
            1e-13,
        );
    }
    // And below the minimum / above the maximum it saturates.
    f.close(
        || "Ecdf below the minimum".into(),
        e.eval(sorted[0] - 1.0),
        0.0,
        1e-15,
    );
    f.close(
        || "Ecdf above the maximum".into(),
        e.eval(sorted[sorted.len() - 1] + 1.0),
        1.0,
        1e-15,
    );

    // five_number_summary is the min, quartiles and max.
    let (fmin, q1, fmed, q3, fmax) = five_number_summary(&SAMPLE);
    f.close(
        || "five_number_summary minimum".into(),
        fmin,
        sorted[0],
        1e-15,
    );
    f.close(
        || "five_number_summary median".into(),
        fmed,
        median(&SAMPLE),
        1e-13,
    );
    f.close(
        || "five_number_summary maximum".into(),
        fmax,
        sorted[sorted.len() - 1],
        1e-15,
    );
    f.close(
        || "five_number_summary quartiles vs quantile".into(),
        q3 - q1,
        iqr(&SAMPLE),
        1e-13,
    );

    // Rank correlation of a strictly increasing map is exactly 1.
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let monotone = x.map(|v: f64| v.exp());
    f.close(
        || "spearman of a monotone map = 1".into(),
        spearman(&x, &monotone),
        1.0,
        1e-12,
    );
    f.close(
        || "kendall_tau of a monotone map = 1".into(),
        kendall_tau(&x, &monotone),
        1.0,
        1e-12,
    );
    let reversed = x.map(|v: f64| -v);
    f.close(
        || "spearman of a reversing map = −1".into(),
        spearman(&x, &reversed),
        -1.0,
        1e-12,
    );

    f.finish("order statistics / ECDF");
}

/// Resampling must land near the statistic it is resampling, and KDE must
/// integrate to one.
#[test]
fn resampling_and_kde_recover_what_they_estimate() {
    let mut f = Failures::default();

    // A bootstrap distribution of the mean is centred on the sample mean.
    let mut rng = ChaCha8Rng::seed_from_u64(20_260_105);
    let boots = bootstrap(&SAMPLE, 2_000, mean, &mut rng);
    f.close(
        || "bootstrap mean of means vs sample mean".into(),
        mean(&boots),
        mean(&SAMPLE),
        0.02,
    );
    let (lo_ci, hi_ci) = percentile_interval(&boots, 0.95);
    f.check(lo_ci < mean(&SAMPLE) && mean(&SAMPLE) < hi_ci, || {
        format!("bootstrap interval [{lo_ci}, {hi_ci}] misses the sample mean")
    });

    // Jackknife replicates of the mean are leave-one-out means.
    let jack = jackknife(&SAMPLE, mean);
    f.check(jack.len() == SAMPLE.len(), || {
        "jackknife replicate count".into()
    });
    let n = SAMPLE.len() as f64;
    for (i, &rep) in jack.iter().enumerate() {
        let expected = (mean(&SAMPLE) * n - SAMPLE[i]) / (n - 1.0);
        f.close(|| format!("jackknife replicate {i}"), rep, expected, 1e-12);
    }

    // A Gaussian KDE is a density, so it integrates to one.
    let kde = KernelDensity::new(&SAMPLE).unwrap();
    let (lo, hi) = (0.0, 10.0);
    let steps = 20_000;
    let h = (hi - lo) / steps as f64;
    let mass: f64 = (0..=steps)
        .map(|i| {
            let x = lo + h * i as f64;
            let w = if i == 0 || i == steps { 0.5 } else { 1.0 };
            w * kde.density(x)
        })
        .sum::<f64>()
        * h;
    f.close(|| "∫KDE over a wide interval = 1".into(), mass, 1.0, 1e-3);

    f.finish("resampling / KDE");
}
