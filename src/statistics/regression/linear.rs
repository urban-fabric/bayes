//! Simple (one-predictor) ordinary least squares: fit `y ≈ slope·x + intercept`
//! with `R²`, standard errors, t-tests, information criteria, and residual
//! diagnostics (leverage, Cook's distance, confidence/prediction intervals).

use crate::distribution::{ContinuousCdf, StudentT};
use crate::statistics::{ConfidenceInterval, correlation, mean, variance};

/// A fitted simple (one-predictor) linear regression.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearRegression {
    pub slope: f64,
    pub intercept: f64,
    /// Coefficient of determination `R²` — the fraction of variance explained.
    pub r_squared: f64,
    /// Adjusted `R²`, penalized by the one-predictor model size.
    pub adjusted_r_squared: f64,
    /// Number of paired observations.
    pub n: usize,
    /// Mean of the predictor values.
    pub x_mean: f64,
    /// Sum of squared predictor deviations, `Σ(xᵢ - x̄)²`.
    pub sxx: f64,
    /// Residual sum of squares.
    pub rss: f64,
    /// Mean squared error, `RSS / (n - 2)`.
    pub mse: f64,
    /// Root mean squared residual, `sqrt(RSS / n)`.
    pub rmse: f64,
    /// Residual standard error, `sqrt(MSE)`.
    pub residual_std_error: f64,
    /// Standard error of the slope.
    pub slope_std_error: f64,
    /// Standard error of the intercept.
    pub intercept_std_error: f64,
    /// t-statistic for testing `slope = 0`.
    pub slope_t: f64,
    /// Two-sided p-value for testing `slope = 0`.
    pub slope_p_value: f64,
    /// t-statistic for testing `intercept = 0`.
    pub intercept_t: f64,
    /// Two-sided p-value for testing `intercept = 0`.
    pub intercept_p_value: f64,
    /// Akaike information criterion under Gaussian residuals, up to an additive constant.
    pub aic: f64,
    /// Bayesian information criterion under Gaussian residuals, up to an additive constant.
    pub bic: f64,
}

impl LinearRegression {
    /// Fit `y` against `x` by ordinary least squares. `None` if the lengths differ,
    /// there are fewer than 2 points, or `x` has zero variance (slope undefined).
    pub fn fit(x: &[f64], y: &[f64]) -> Option<Self> {
        if x.len() != y.len() || x.len() < 2 {
            return None;
        }
        if x.iter().chain(y).any(|v| !v.is_finite()) {
            return None;
        }
        let var_x = variance(x);
        if var_x <= 0.0 || var_x.is_nan() {
            return None;
        }
        let n = x.len();
        let x_mean = mean(x);
        let y_mean = mean(y);
        let sxx = x.iter().map(|&xi| (xi - x_mean).powi(2)).sum::<f64>();
        let sxy = x
            .iter()
            .zip(y)
            .map(|(&xi, &yi)| (xi - x_mean) * (yi - y_mean))
            .sum::<f64>();
        let slope = sxy / sxx;
        let intercept = y_mean - slope * x_mean;
        let r = correlation(x, y);
        let r_squared = r * r;
        let rss = x
            .iter()
            .zip(y)
            .map(|(&xi, &yi)| {
                let e = yi - (intercept + slope * xi);
                e * e
            })
            .sum::<f64>();
        let df = n as f64 - 2.0;
        let mse = if df > 0.0 { rss / df } else { f64::NAN };
        let rmse = (rss / n as f64).sqrt();
        let residual_std_error = mse.sqrt();
        let slope_std_error = (mse / sxx).sqrt();
        let intercept_std_error = (mse * (1.0 / n as f64 + x_mean * x_mean / sxx)).sqrt();
        let slope_t = ratio_or_infinite(slope, slope_std_error);
        let slope_p_value = t_p_value(slope_t, df);
        let intercept_t = ratio_or_infinite(intercept, intercept_std_error);
        let intercept_p_value = t_p_value(intercept_t, df);
        let adjusted_r_squared = if n > 2 {
            1.0 - (1.0 - r_squared) * (n as f64 - 1.0) / (n as f64 - 2.0)
        } else {
            f64::NAN
        };
        let gaussian_ic = if rss >= 0.0 {
            let log_mse = (rss / n as f64).ln();
            let k = 2.0;
            (
                n as f64 * log_mse + 2.0 * k,
                n as f64 * log_mse + k * (n as f64).ln(),
            )
        } else {
            (f64::NAN, f64::NAN)
        };
        Some(Self {
            slope,
            intercept,
            r_squared,
            adjusted_r_squared,
            n,
            x_mean,
            sxx,
            rss,
            mse,
            rmse,
            residual_std_error,
            slope_std_error,
            intercept_std_error,
            slope_t,
            slope_p_value,
            intercept_t,
            intercept_p_value,
            aic: gaussian_ic.0,
            bic: gaussian_ic.1,
        })
    }

    /// Predict `y` at a new `x`.
    pub fn predict(&self, x: f64) -> f64 {
        self.intercept + self.slope * x
    }

    /// Residuals `yᵢ - ŷᵢ`. Returns an empty vector for mismatched inputs.
    pub fn residuals(&self, x: &[f64], y: &[f64]) -> Vec<f64> {
        if x.len() != y.len() {
            return Vec::new();
        }
        x.iter()
            .zip(y)
            .map(|(&xi, &yi)| yi - self.predict(xi))
            .collect()
    }

    /// Leverage values for the fitted one-predictor design.
    pub fn leverage(&self, x: &[f64]) -> Vec<f64> {
        if self.n == 0 || self.sxx <= 0.0 {
            return Vec::new();
        }
        x.iter()
            .map(|&xi| 1.0 / self.n as f64 + (xi - self.x_mean).powi(2) / self.sxx)
            .collect()
    }

    /// Cook's distance for each observation. Returns an empty vector for malformed
    /// inputs or degenerate residual variance.
    pub fn cooks_distance(&self, x: &[f64], y: &[f64]) -> Vec<f64> {
        if x.len() != y.len() || self.mse <= 0.0 || self.mse.is_nan() {
            return Vec::new();
        }
        let residuals = self.residuals(x, y);
        let h = self.leverage(x);
        residuals
            .iter()
            .zip(h)
            .map(|(&e, hi)| e * e / (2.0 * self.mse) * hi / (1.0 - hi).powi(2))
            .collect()
    }

    /// Confidence interval for the fitted mean response at predictor value `x`.
    pub fn confidence_interval(&self, x: f64, level: f64) -> ConfidenceInterval {
        self.response_interval(x, level, false)
    }

    /// Prediction interval for a future observation at predictor value `x`.
    pub fn prediction_interval(&self, x: f64, level: f64) -> ConfidenceInterval {
        self.response_interval(x, level, true)
    }

    fn response_interval(&self, x: f64, level: f64, prediction: bool) -> ConfidenceInterval {
        if !(0.0..1.0).contains(&level)
            || self.n < 3
            || self.sxx <= 0.0
            || !x.is_finite()
            || self.residual_std_error.is_nan()
        {
            return invalid_interval(level);
        }
        let df = self.n as f64 - 2.0;
        let t = match StudentT::new(df) {
            Ok(d) => d.inverse_cdf(1.0 - (1.0 - level) / 2.0),
            Err(_) => return invalid_interval(level),
        };
        let prediction_term = if prediction { 1.0 } else { 0.0 };
        let se = self.residual_std_error
            * (prediction_term + 1.0 / self.n as f64 + (x - self.x_mean).powi(2) / self.sxx).sqrt();
        let y_hat = self.predict(x);
        ConfidenceInterval {
            lower: y_hat - t * se,
            upper: y_hat + t * se,
            level,
        }
    }
}

fn ratio_or_infinite(numerator: f64, denominator: f64) -> f64 {
    if denominator == 0.0 {
        if numerator == 0.0 {
            f64::NAN
        } else {
            numerator.signum() * f64::INFINITY
        }
    } else {
        numerator / denominator
    }
}

fn t_p_value(t: f64, df: f64) -> f64 {
    if df <= 0.0 || t.is_nan() {
        return f64::NAN;
    }
    if t.is_infinite() {
        return 0.0;
    }
    match StudentT::new(df) {
        Ok(d) => 2.0 * d.sf(t.abs()),
        Err(_) => f64::NAN,
    }
}

fn invalid_interval(level: f64) -> ConfidenceInterval {
    ConfidenceInterval {
        lower: f64::NAN,
        upper: f64::NAN,
        level,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn recovers_an_exact_line() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|v| 3.0 * v + 2.0).collect();
        let fit = LinearRegression::fit(&x, &y).unwrap();
        close(fit.slope, 3.0, 1e-12);
        close(fit.intercept, 2.0, 1e-12);
        close(fit.r_squared, 1.0, 1e-12);
        close(fit.rss, 0.0, 1e-12);
        close(fit.predict(10.0), 32.0, 1e-12);
    }

    #[test]
    fn noisy_fit_has_r_squared_below_one() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.1, 3.9, 6.2, 7.8, 10.1];
        let fit = LinearRegression::fit(&x, &y).unwrap();
        assert!(fit.slope > 1.8 && fit.slope < 2.2);
        assert!(fit.r_squared > 0.99 && fit.r_squared < 1.0);
        assert!(fit.rmse > 0.0);
        assert!(fit.slope_std_error > 0.0);
        assert!(fit.slope_p_value < 0.01);
        assert!(fit.intercept_std_error > 0.0);
    }

    #[test]
    fn regression_diagnostics_have_expected_shapes() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.1, 5.9, 8.2, 9.8];
        let fit = LinearRegression::fit(&x, &y).unwrap();
        assert_eq!(fit.n, 5);
        assert_eq!(fit.residuals(&x, &y).len(), 5);
        assert_eq!(fit.leverage(&x).len(), 5);
        assert_eq!(fit.cooks_distance(&x, &y).len(), 5);
        let ci = fit.confidence_interval(3.0, 0.95);
        let pi = fit.prediction_interval(3.0, 0.95);
        assert!(pi.width() > ci.width());
        assert!(ci.contains(fit.predict(3.0)));
    }

    #[test]
    fn degenerate_x_returns_none() {
        assert!(LinearRegression::fit(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]).is_none());
    }
}
