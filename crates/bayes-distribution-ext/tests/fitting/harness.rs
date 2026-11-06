//! The recovery harness: fit the same family many times and look at the bias.

use bayes_utils::Failures;
use rand_chacha::ChaCha8Rng;

/// Independent samples fitted per family.
pub(crate) const REPLICATES: usize = 32;
/// Observations per sample.
pub(crate) const SAMPLE: usize = 2_000;

/// Draw one observation, as an `f64` whatever the family's native sample type.
pub(crate) type Draw = Box<dyn Fn(&mut ChaCha8Rng) -> f64>;
/// Fit, and report the parameters in a fixed order, or decline.
pub(crate) type Fit = Box<dyn Fn(&[f64]) -> Option<Vec<f64>>>;

/// One family, with a known truth to recover.
pub(crate) struct Case {
    pub(crate) name: &'static str,
    pub(crate) labels: &'static [&'static str],
    pub(crate) truth: Vec<f64>,
    /// An allowance, in absolute units, for a **known finite-sample bias**.
    ///
    /// Zero for every moment-based fit here, and that is the point: a moment
    /// estimator's bias is `O(1/n)` while its standard error is `O(1/√n)`, so
    /// at `n = 2000` the bias is swamped by the bound below and needs no
    /// allowance at all.
    ///
    /// The exceptions are the fits built from **order statistics**. A uniform's
    /// sample minimum overestimates `lo`, and its error is `O(1/n)` — the same
    /// order as its own standard error, so it never becomes negligible relative
    /// to the bound however large the sample. Each case that sets this shows the
    /// arithmetic.
    pub(crate) bias: f64,
    pub(crate) draw: Draw,
    pub(crate) fit: Fit,
}

impl Case {
    pub(crate) fn with_bias(mut self, bias: f64) -> Self {
        self.bias = bias;
        self
    }
}

/// Build a case from anything that can be sampled into an `f64`.
pub(crate) fn case(
    name: &'static str,
    labels: &'static [&'static str],
    truth: &[f64],
    draw: impl Fn(&mut ChaCha8Rng) -> f64 + 'static,
    fit: impl Fn(&[f64]) -> Option<Vec<f64>> + 'static,
) -> Case {
    Case {
        name,
        labels,
        truth: truth.to_vec(),
        bias: 0.0,
        draw: Box::new(draw),
        fit: Box::new(fit),
    }
}

/// Fit each family [`REPLICATES`] times on independent samples and require the
/// **average** of the fits to sit within four standard errors of the truth.
///
/// # Why the bound is the fits' own scatter
///
/// The tolerance is computed from the spread of the replicate fits, not chosen.
/// That makes it a property of the estimator under test rather than a number
/// adjusted until the suite passed — the failure mode the repository's
/// convention about tolerances exists to prevent.
///
/// It also happens to be the right instrument for this particular job. A
/// formula with a swapped argument, a wrong constant or a missing square root
/// produces a *bias*: every replicate is wrong in the same direction, so the
/// average misses by many standard errors and the check fails however precise
/// or imprecise the estimator happens to be. Meanwhile an estimator that is
/// correct but noisy — a Frechet shape from 2000 points, say — widens its own
/// bound in proportion and passes. One threshold, `4σ`, therefore works across
/// estimators whose precisions differ by orders of magnitude.
///
/// Averaging over replicates is what buys the sensitivity: it shrinks the
/// standard error by `√REPLICATES` without shrinking a bias at all, so the test
/// is roughly `√32 ≈ 5.7` times more sensitive to a wrong formula than fitting
/// once would be.
pub(crate) fn recovers_its_parameters(cases: Vec<Case>, seed: u64, context: &str) {
    let mut f = Failures::default();

    for item in cases {
        let mut rng = <ChaCha8Rng as rand::SeedableRng>::seed_from_u64(seed);
        let mut fits: Vec<Vec<f64>> = Vec::with_capacity(REPLICATES);
        let mut refused = false;

        for replicate in 0..REPLICATES {
            let data: Vec<f64> = (0..SAMPLE).map(|_| (item.draw)(&mut rng)).collect();
            match (item.fit)(&data) {
                Some(parameters) => fits.push(parameters),
                None => {
                    // A clean sample from the family itself must always fit. A
                    // refusal here means a precondition is too strict, not that
                    // the data was bad.
                    f.check(false, || {
                        format!(
                            "{}: replicate {replicate} of {SAMPLE} draws from the family \
                             itself was refused by from_sample",
                            item.name
                        )
                    });
                    refused = true;
                    break;
                }
            }
        }
        if refused {
            continue;
        }

        for (index, label) in item.labels.iter().enumerate() {
            let values: Vec<f64> = fits.iter().map(|p| p[index]).collect();
            let mean = values.iter().sum::<f64>() / REPLICATES as f64;
            let spread = (values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>()
                / (REPLICATES - 1) as f64)
                .sqrt();
            let standard_error = spread / (REPLICATES as f64).sqrt();
            let allowance = 4.0 * standard_error + item.bias;
            f.close(
                || {
                    format!(
                        "{name} {label}: mean of {REPLICATES} fits (fit-to-fit spread \
                         {spread:.3e}, 4 se {four_se:.3e}, bias allowance {bias:.3e})",
                        name = item.name,
                        four_se = 4.0 * standard_error,
                        bias = item.bias
                    )
                },
                mean,
                item.truth[index],
                allowance,
            );
        }
    }

    f.finish(context);
}
