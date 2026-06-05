//! The error function family — `erf`, `erfc`, and the inverse `erf_inv` — the
//! workhorses of the Normal CDF (`Φ(x) = ½ erfc(-x/√2)`) and quantile.
//!
//! `erf`/`erfc` reduce to the regularized incomplete gamma `P(½, x²)`, which is
//! accurate across the whole range; `erf_inv` uses Giles's rational approximation
//! refined by one Newton step.

use super::gamma::gamma_lr;
use std::f64::consts::PI;

/// The error function `erf(x) = (2/√π) ∫₀ˣ e^{-t²} dt`.
pub fn erf(x: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else if x > 0.0 {
        gamma_lr(0.5, x * x)
    } else {
        -gamma_lr(0.5, x * x)
    }
}

/// The complementary error function `erfc(x) = 1 - erf(x)`.
pub fn erfc(x: f64) -> f64 {
    1.0 - erf(x)
}

/// The inverse error function: `erf_inv(erf(x)) = x` for `x ∈ (-1, 1)`.
pub fn erf_inv(x: f64) -> f64 {
    if x <= -1.0 {
        return f64::NEG_INFINITY;
    }
    if x >= 1.0 {
        return f64::INFINITY;
    }
    if x == 0.0 {
        return 0.0;
    }
    // Giles (2010) rational approximation as the initial guess.
    let w = -((1.0 - x) * (1.0 + x)).ln();
    let mut p;
    if w < 5.0 {
        let w = w - 2.5;
        p = 2.810_226_36e-08;
        p = 3.432_739_39e-07 + p * w;
        p = -3.523_387_7e-06 + p * w;
        p = -4.391_506_54e-06 + p * w;
        p = 0.000_218_580_87 + p * w;
        p = -0.001_253_725_03 + p * w;
        p = -0.004_177_681_64 + p * w;
        p = 0.246_640_727 + p * w;
        p = 1.501_409_41 + p * w;
    } else {
        let w = w.sqrt() - 3.0;
        p = -0.000_200_214_257;
        p = 0.000_100_950_24 + p * w;
        p = 0.001_349_343_22 + p * w;
        p = -0.003_673_428_4 + p * w;
        p = 0.005_739_507_73 + p * w;
        p = -0.007_622_461_3 + p * w;
        p = 0.009_438_870_47 + p * w;
        p = 1.001_674_06 + p * w;
        p = 2.832_976_82 + p * w;
    }
    let mut result = p * x;
    // One Newton step on erf to polish: f(y) = erf(y) - x, f'(y) = (2/√π)e^{-y²}.
    let err = erf(result) - x;
    result -= err / (2.0 / PI.sqrt() * (-result * result).exp());
    result
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "{a} vs {b}");
    }

    #[test]
    fn erf_known_values() {
        close(erf(0.0), 0.0, 1e-12);
        close(erf(1.0), 0.842_700_792_949_715, 1e-10);
        close(erf(-1.0), -0.842_700_792_949_715, 1e-10);
        close(erf(2.0), 0.995_322_265_018_953, 1e-10);
        close(erfc(0.0), 1.0, 1e-12);
    }

    #[test]
    fn erf_inv_round_trips() {
        for &x in &[-0.9, -0.5, -0.1, 0.1, 0.3, 0.5, 0.8, 0.95] {
            close(erf(erf_inv(x)), x, 1e-10);
        }
    }
}
