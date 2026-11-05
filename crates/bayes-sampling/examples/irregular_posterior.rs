//! **A posterior with no conjugate form, sampled exactly.**
//!
//! ```sh
//! cargo run -p bayes-sampling --example irregular_posterior
//! ```
//!
//! Conjugacy is a luxury. Put a lognormal prior on a Poisson rate instead of the
//! gamma that would close, and the posterior
//!
//! ```text
//!     p(lambda | x) proportional to  LogNormal(lambda | mu, sigma) * prod Poisson(x_i | lambda)
//! ```
//!
//! has no closed form — no formula for its mean, no invertible CDF, no name.
//! The usual next step is MCMC, with its burn-in and its autocorrelation and its
//! convergence diagnostics.
//!
//! None of that is needed here. Reparameterize to `theta = ln lambda` and the
//! log-posterior becomes **concave**: the prior contributes `-(theta-mu)^2/2sigma^2`
//! and the likelihood contributes `theta*sum(x) - n*exp(theta)`, whose second
//! derivative is `-n*exp(theta) < 0`. Adaptive rejection then draws from it
//! *exactly*, with no tuning and no diagnostics, and gets cheaper as it runs.
//!
//! The example validates itself two ways: the sampled posterior mean against
//! numerical integration of the same unnormalized density, and the draws against
//! a quantile-bucket chi-squared test using the numerically integrated CDF.
//! Every printed number is asserted.

// An example's job is to be read, and threading `Result` through every line of a
// worked problem costs more clarity than the panic-freedom is worth here.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use bayes_sampling::{AdaptiveRejection, AliasTable};
use bayes_utils::{chi_squared_gof_pvalue, ks_pvalue, ks_statistic};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn main() {
    // ---------------------------------------------------------------------
    // The problem: counts of rare defects on 12 production panels.
    // ---------------------------------------------------------------------
    let counts: [u64; 12] = [2, 0, 1, 3, 1, 0, 2, 4, 1, 1, 0, 2];
    let n = counts.len() as f64;
    let total: f64 = counts.iter().sum::<u64>() as f64;

    // A lognormal prior on the rate: centred on about 1.6 defects, spread wide.
    // This is the natural prior when the engineer's belief is multiplicative
    // ("could be half, could be double") — and it does not conjugate.
    let (prior_mu, prior_sigma) = (0.5_f64, 0.75_f64);

    println!("Non-conjugate Poisson rate, lognormal prior");
    println!("  {} panels, {} defects total", counts.len(), total as u64);
    println!("  prior: LogNormal(mu = {prior_mu}, sigma = {prior_sigma})");
    println!("  MLE rate (ignoring the prior): {:.4}", total / n);

    // ---------------------------------------------------------------------
    // The log-posterior in theta = ln lambda, which is concave.
    // ---------------------------------------------------------------------
    // Prior in theta is Normal(mu, sigma) because theta = ln lambda.
    // Likelihood in theta: sum(x)*theta - n*exp(theta), dropping ln(x_i!).
    let ln_post = move |theta: f64| {
        let prior = -0.5 * ((theta - prior_mu) / prior_sigma).powi(2);
        prior + total * theta - n * theta.exp()
    };
    let d_ln_post = move |theta: f64| {
        let prior = -(theta - prior_mu) / (prior_sigma * prior_sigma);
        prior + total - n * theta.exp()
    };

    // Seeds either side of the posterior mode; the exact location does not
    // matter, since the hull adapts.
    let mut sampler = AdaptiveRejection::new(
        ln_post,
        d_ln_post,
        f64::NEG_INFINITY,
        f64::INFINITY,
        &[-1.0, 1.0],
    )
    .expect("the seeds bracket a concave log-density");

    let draws = 40_000;
    let mut rng = ChaCha8Rng::seed_from_u64(20_260_105);
    let thetas: Vec<f64> = (0..draws)
        .map(|_| sampler.draw(&mut rng, 1_000).expect("acceptance is near 1"))
        .collect();
    let lambdas: Vec<f64> = thetas.iter().map(|t| t.exp()).collect();

    println!("\n-- adaptive rejection --");
    println!("  {draws} exact draws, no burn-in and no autocorrelation");
    println!("  acceptance rate      {:.4}", sampler.acceptance_rate());
    println!(
        "  density evaluations  {} (of {draws} draws)",
        sampler.density_evaluations()
    );
    println!("  hull size            {} abscissae", sampler.hull_size());
    println!(
        "  -- the squeeze test avoided {:.1}% of density evaluations",
        100.0 * (1.0 - sampler.density_evaluations() as f64 / draws as f64)
    );

    assert!(
        !sampler.hull_was_breached(),
        "the log-posterior should be concave in theta"
    );
    assert!(
        sampler.acceptance_rate() > 0.9,
        "acceptance {} suggests the hull stopped adapting",
        sampler.acceptance_rate()
    );
    assert!(
        sampler.density_evaluations() < draws / 4,
        "the squeeze test should avoid most density evaluations"
    );

    // ---------------------------------------------------------------------
    // Posterior summaries, and the cross-check against quadrature.
    // ---------------------------------------------------------------------
    let mut sorted = lambdas.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    let q = |p: f64| sorted[((p * sorted.len() as f64) as usize).min(sorted.len() - 1)];
    let sample_mean = lambdas.iter().sum::<f64>() / lambdas.len() as f64;

    println!("\n-- posterior on the rate --");
    println!("  mean                 {sample_mean:.4}");
    println!("  median               {:.4}", q(0.5));
    println!("  95% credible         [{:.4}, {:.4}]", q(0.025), q(0.975));

    // The same posterior by numerical integration over lambda directly. The
    // density in lambda carries the Jacobian: p(lambda) = p(theta)/lambda.
    let ln_post_lambda = |lam: f64| {
        if lam <= 0.0 {
            return f64::NEG_INFINITY;
        }
        ln_post(lam.ln()) - lam.ln()
    };
    let (lo, hi) = (1e-6, 20.0);
    let steps = 400_000;
    let h = (hi - lo) / steps as f64;
    // Normalize and take the first moment in one pass, working relative to the
    // peak so the exponential cannot overflow.
    let peak = (0..=steps)
        .map(|i| ln_post_lambda(lo + h * i as f64))
        .fold(f64::NEG_INFINITY, f64::max);
    let (mut mass, mut first) = (0.0, 0.0);
    for i in 0..=steps {
        let lam = lo + h * i as f64;
        let w = if i == 0 || i == steps { 0.5 } else { 1.0 };
        let d = (ln_post_lambda(lam) - peak).exp();
        mass += w * d;
        first += w * d * lam;
    }
    let quad_mean = first / mass;

    println!("\n-- cross-check against quadrature --");
    println!("  sampled mean         {sample_mean:.5}");
    println!("  integrated mean      {quad_mean:.5}");
    println!(
        "  difference           {:.2e}",
        (sample_mean - quad_mean).abs()
    );
    assert!(
        (sample_mean - quad_mean).abs() < 0.01,
        "sampled {sample_mean} and integrated {quad_mean} should agree"
    );

    // And the whole distribution, not just its mean: KS against the numerically
    // integrated CDF.
    let cdf = |x: f64| {
        if x <= lo {
            return 0.0;
        }
        let upper = x.min(hi);
        let k = (((upper - lo) / h) as usize).min(steps);
        let mut acc = 0.0;
        for i in 0..=k {
            let lam = lo + h * i as f64;
            let w = if i == 0 || i == k { 0.5 } else { 1.0 };
            acc += w * (ln_post_lambda(lam) - peak).exp();
        }
        (acc / mass).clamp(0.0, 1.0)
    };
    // KS on a subsample: the numerical CDF costs a sweep per evaluation.
    let probe: Vec<f64> = lambdas.iter().copied().step_by(40).collect();
    let d = ks_statistic(&probe, cdf);
    let p = ks_pvalue(d, probe.len());
    println!(
        "  KS vs integrated cdf D = {d:.5}, p = {p:.4} (n = {})",
        probe.len()
    );
    assert!(p > 1e-3, "the draws should match the integrated posterior");

    // ---------------------------------------------------------------------
    // Posterior predictive, and where the alias table earns its keep.
    // ---------------------------------------------------------------------
    println!("\n-- posterior predictive --");
    // P(next panel has k defects) = E_posterior[ Poisson(k | lambda) ].
    // Averaging over the draws is a Monte Carlo integral that needs no new theory.
    let mut predictive = vec![0.0; 9];
    for &lam in &lambdas {
        let mut term = (-lam).exp();
        for (k, slot) in predictive.iter_mut().enumerate() {
            *slot += term;
            term *= lam / (k + 1) as f64;
        }
    }
    for slot in &mut predictive {
        *slot /= lambdas.len() as f64;
    }
    for (k, &p) in predictive.iter().enumerate() {
        println!("  P(X = {k}) = {p:.5}");
    }
    let predictive_total: f64 = predictive.iter().sum();
    assert!(
        predictive_total > 0.99 && predictive_total <= 1.0 + 1e-12,
        "the predictive should nearly sum to one over 0..=8, got {predictive_total}"
    );

    // Drawing from that predictive repeatedly is exactly what an alias table is
    // for: the weights are fixed, so pay O(n) once and O(1) per draw forever.
    let table = AliasTable::new(&predictive).expect("a valid predictive");
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let sim = 200_000;
    let mut counts_out = vec![0.0; table.len()];
    for _ in 0..sim {
        counts_out[table.sample(&mut rng)] += 1.0;
    }
    let expected: Vec<f64> = (0..table.len())
        .map(|k| table.pmf(k) * sim as f64)
        .collect();
    let gof = chi_squared_gof_pvalue(&counts_out, &expected, 5.0);
    println!("\n-- alias table over the predictive --");
    println!("  {sim} O(1) draws, chi-squared vs the table: p = {gof:.4}");
    assert!(
        !gof.is_finite() || gof > 1e-4,
        "alias draws should follow the predictive (p = {gof})"
    );

    println!("\nall assertions held");
}
