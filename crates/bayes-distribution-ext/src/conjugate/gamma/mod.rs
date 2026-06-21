//! Gamma-prior conjugate models: **Gamma–Poisson** and **Gamma–Exponential**.

use super::{Conjugate, Pair};
use crate::{Gamma, Lomax};
use bayes_error::BayesError;
use euclid::{LazyNatural, LazyReal};

mod exponential;
mod poisson;

pub use exponential::GammaExponential;
pub use poisson::GammaPoisson;

fn validate_exposure(value: f64, name: &'static str) -> Result<(), BayesError> {
    if !value.is_finite() {
        return Err(BayesError::NonFinite { name, value });
    }
    if value < 0.0 {
        return Err(BayesError::Malformed {
            what: "exposures must be non-negative",
        });
    }
    Ok(())
}

fn exponential_data_are_valid(data: &[f64]) -> bool {
    data.iter().all(|&x| x.is_finite() && x >= 0.0)
}

fn ln_gamma_poisson_predictive(shape: f64, rate: f64, count: u64, exposure: f64) -> f64 {
    if validate_exposure(exposure, "exposure").is_err() {
        return f64::NAN;
    }
    if exposure == 0.0 {
        return if count == 0 { 0.0 } else { f64::NAN };
    }
    let count_f = count as f64;
    (shape + count_f).ln_gamma() - shape.ln_gamma() - count.ln_factorial()
        + shape * rate.ln()
        + count_f * exposure.ln()
        - (shape + count_f) * (rate + exposure).ln()
}

fn ln_gamma_exponential_predictive(shape: f64, rate: f64, x: f64) -> f64 {
    if !x.is_finite() || x < 0.0 {
        return f64::NAN;
    }
    shape.ln() + shape * rate.ln() - (shape + 1.0) * (rate + x).ln()
}

fn ln_gamma_exponential_exposure_evidence(
    shape: f64,
    rate: f64,
    failures: u64,
    total_exposure: f64,
) -> f64 {
    if validate_exposure(total_exposure, "total_exposure").is_err() {
        return f64::NAN;
    }
    let failures = failures as f64;
    shape * rate.ln() - shape.ln_gamma() + (shape + failures).ln_gamma()
        - (shape + failures) * (rate + total_exposure).ln()
}

fn poisson_exposure_stats(
    counts: &[u64],
    exposures: &[f64],
) -> Result<(f64, f64, f64), BayesError> {
    if counts.len() != exposures.len() {
        return Err(BayesError::Malformed {
            what: "counts and exposures must have the same length",
        });
    }
    let mut sum_counts = 0.0;
    let mut total_exposure = 0.0;
    let mut exposure_coeff = 0.0;
    for (&count, &exposure) in counts.iter().zip(exposures) {
        validate_exposure(exposure, "exposure")?;
        if count > 0 && exposure == 0.0 {
            return Err(BayesError::Malformed {
                what: "positive counts require positive exposure",
            });
        }
        sum_counts += count as f64;
        total_exposure += exposure;
        if count > 0 {
            exposure_coeff += count as f64 * exposure.ln();
        }
    }
    Ok((sum_counts, total_exposure, exposure_coeff))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::{Continuous, Discrete, Moments};

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn gamma_poisson_posterior_and_predictive() {
        let model = GammaPoisson::new(Gamma::new(2.0, 1.0).unwrap());
        let data = [3u64, 5, 2, 4];
        let post = model.posterior(&data);
        close(post.shape(), 16.0, 1e-12);
        close(post.rate(), 5.0, 1e-12);
        let pred = model.posterior_predictive(&data);
        close(pred.mean(), 16.0 / 5.0, 1e-12);
        assert!(pred.pmf(3_u64) > 0.0);
        close(
            model.ln_predictive(3).exp(),
            model.posterior_predictive(&[]).pmf(3_u64),
            1e-12,
        );
        close(
            model.ln_posterior_predictive(&data, 3).exp(),
            pred.pmf(3_u64),
            1e-12,
        );
        let evidence = 2.0 * 1.0_f64.ln()
            - 2.0_f64.ln_gamma()
            - data.iter().map(|&k| k.ln_factorial()).sum::<f64>()
            + 16.0_f64.ln_gamma()
            - 16.0 * 5.0_f64.ln();
        close(model.ln_marginal_likelihood(&data), evidence, 1e-12);
        close(
            model.ln_sequential_predictive(&data),
            model.ln_marginal_likelihood(&data),
            1e-12,
        );
        close(
            model.ln_posterior_predictive_likelihood(&data, &[3, 0]),
            GammaPoisson::new(post).ln_marginal_likelihood(&[3, 0]),
            1e-12,
        );
    }

    #[test]
    fn gamma_poisson_exposure_offsets_are_closed_form() {
        let model = GammaPoisson::new(Gamma::new(2.0, 1.0).unwrap());
        let counts = [3u64, 2];
        let exposures = [0.5, 1.5];
        let post = model.posterior_with_exposure(&counts, &exposures).unwrap();
        close(post.shape(), 7.0, 1e-12);
        close(post.rate(), 3.0, 1e-12);

        let pred = model
            .posterior_predictive_with_exposure(&counts, &exposures, 2.0)
            .unwrap();
        close(pred.mean(), 14.0 / 3.0, 1e-12);
        close(
            model.ln_predictive_with_exposure(3, 0.5).exp(),
            model
                .posterior_predictive_with_exposure(&[], &[], 0.5)
                .unwrap()
                .pmf(3_u64),
            1e-12,
        );
        close(
            model
                .ln_posterior_predictive_with_exposure(&counts, &exposures, 1, 2.0)
                .exp(),
            pred.pmf(1_u64),
            1e-12,
        );

        let evidence = 7.0_f64.ln_gamma() - 2.0_f64.ln_gamma() - 7.0 * 3.0_f64.ln()
            + 3.0 * 0.5_f64.ln()
            + 2.0 * 1.5_f64.ln()
            - 3_u64.ln_factorial()
            - 2_u64.ln_factorial();
        close(
            model.ln_marginal_likelihood_with_exposure(&counts, &exposures),
            evidence,
            1e-12,
        );
        close(
            model.ln_sequential_predictive_with_exposure(&counts, &exposures),
            model.ln_marginal_likelihood_with_exposure(&counts, &exposures),
            1e-12,
        );
        close(
            model.ln_posterior_predictive_likelihood_with_exposure(
                &counts,
                &exposures,
                &[1],
                &[2.0],
            ),
            GammaPoisson::new(post).ln_marginal_likelihood_with_exposure(&[1], &[2.0]),
            1e-12,
        );
        assert!(model.posterior_with_exposure(&[1], &[0.0]).is_err());
        assert!(model.posterior_with_exposure(&[0], &[-1.0]).is_err());
        assert!(
            model
                .posterior_predictive_with_exposure(&counts, &exposures, 0.0)
                .is_err()
        );
        assert!(model.ln_predictive_with_exposure(1, 0.0).is_nan());
        assert!(
            model
                .ln_sequential_predictive_with_exposure(&[1], &[0.0])
                .is_nan()
        );
    }

    #[test]
    fn gamma_exponential_posterior_and_lomax_predictive() {
        let model = GammaExponential::new(Gamma::new(1.0, 1.0).unwrap());
        let data = [0.5, 1.5, 2.0];
        let post = model.posterior(&data);
        close(post.shape(), 4.0, 1e-12);
        close(post.rate(), 5.0, 1e-12);
        // Predictive is Lomax(α′, β′) = Lomax(4, 5); mean = β′/(α′−1) = 5/3.
        let pred = model.posterior_predictive(&data);
        close(pred.shape(), 4.0, 1e-12);
        close(pred.scale(), 5.0, 1e-12);
        close(pred.mean(), 5.0 / 3.0, 1e-12);
        close(
            model.ln_predictive(0.5).exp(),
            model.posterior_predictive(&[]).pdf(0.5),
            1e-12,
        );
        close(
            model.ln_posterior_predictive(&data, 1.0).exp(),
            pred.pdf(1.0),
            1e-12,
        );
        close(
            model.ln_marginal_likelihood(&data),
            1.0 * 1.0_f64.ln() - 1.0_f64.ln_gamma() + 4.0_f64.ln_gamma() - 4.0 * 5.0_f64.ln(),
            1e-12,
        );
        close(
            model.ln_sequential_predictive(&data),
            model.ln_marginal_likelihood(&data),
            1e-12,
        );
        close(
            model.ln_posterior_predictive_likelihood(&data, &[0.25, 0.75]),
            GammaExponential::new(post).ln_marginal_likelihood(&[0.25, 0.75]),
            1e-12,
        );
        assert!(model.ln_marginal_likelihood(&[-1.0]).is_nan());
        assert!(model.ln_predictive(-1.0).is_nan());
        assert!(model.ln_posterior_predictive(&[-1.0], 1.0).is_nan());
    }

    #[test]
    fn gamma_exponential_censored_exposure_is_closed_form() {
        let model = GammaExponential::new(Gamma::new(1.0, 1.0).unwrap());
        let post = model.posterior_with_exposure(2, 4.0).unwrap();
        close(post.shape(), 3.0, 1e-12);
        close(post.rate(), 5.0, 1e-12);

        let pred = model.posterior_predictive_with_exposure(2, 4.0).unwrap();
        close(pred.shape(), 3.0, 1e-12);
        close(pred.scale(), 5.0, 1e-12);
        close(pred.mean(), 2.5, 1e-12);
        close(
            model.ln_posterior_predictive_with_exposure(2, 4.0, 1, 2.0),
            GammaExponential::new(post).ln_marginal_likelihood_with_exposure(1, 2.0),
            1e-12,
        );

        let evidence = 3.0_f64.ln_gamma() - 3.0 * 5.0_f64.ln();
        close(
            model.ln_marginal_likelihood_with_exposure(2, 4.0),
            evidence,
            1e-12,
        );
        close(model.ln_predictive_with_exposure(2, 4.0), evidence, 1e-12);
        close(
            model.ln_marginal_likelihood_with_exposure(0, 0.0),
            0.0,
            1e-12,
        );
        assert!(model.posterior_with_exposure(1, f64::NAN).is_err());
        assert!(model.ln_marginal_likelihood_with_exposure(1, -1.0).is_nan());
    }
}
