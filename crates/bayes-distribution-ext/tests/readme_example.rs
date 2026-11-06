//! The README's examples, compiled so they cannot go stale.
#![allow(clippy::unwrap_used)]

use bayes_distribution_ext::{BayesError, Continuous, ContinuousCdf, Moments, Normal};

#[test]
fn readme_opening_example() -> Result<(), BayesError> {
    let n = Normal::new(2.0, 3.0)?;

    n.pdf(2.0);
    n.inverse_cdf(0.975);
    n.entropy();

    // The tail claim the prose makes, at the standard normal where -40 really is
    // -40 sigma: the log-CDF stays finite where the naive `cdf(x).ln()` has
    // already underflowed to -inf.
    let z = Normal::new(0.0, 1.0)?;
    let ln_tail: f64 = z.ln_cdf(-40.0);
    let naive: f64 = z.cdf(-40.0);
    assert!(ln_tail.is_finite());
    assert!((ln_tail - -804.608_442).abs() < 1e-2);
    assert_eq!(naive.ln(), f64::NEG_INFINITY);
    Ok(())
}

#[test]
fn readme_lazy_vs_eager_constructors() {
    // Infallible on the carrier-generic side: on a deferred carrier there may be
    // nothing to validate yet.
    let _lazy = bayes_distribution::Normal::new(0.0, 1.0);
    // Validating on the eager side, where the parameters are host scalars.
    assert!(Normal::new(0.0, -1.0).is_err());
}
