//! [`RunningCovariance`] — a single-pass, mergeable accumulator over **paired**
//! `(x, y)` observations: covariance, Pearson correlation, and simple OLS
//! slope/intercept (the bivariate analogue of [`RunningStats`](super::RunningStats),
//! via Welford's co-moment recurrence, mergeable by the Schubert/Pébay formula).

/// A running accumulator over paired `(x, y)` observations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunningCovariance {
    n: u64,
    mean_x: f64,
    mean_y: f64,
    c2: f64,   // Σ (xᵢ − x̄)(yᵢ − ȳ)
    m2_x: f64, // Σ (xᵢ − x̄)²
    m2_y: f64, // Σ (yᵢ − ȳ)²
}

impl Default for RunningCovariance {
    fn default() -> Self {
        Self::new()
    }
}

impl RunningCovariance {
    pub fn new() -> Self {
        Self {
            n: 0,
            mean_x: 0.0,
            mean_y: 0.0,
            c2: 0.0,
            m2_x: 0.0,
            m2_y: 0.0,
        }
    }

    /// Absorb one paired observation.
    pub fn push(&mut self, x: f64, y: f64) {
        self.n += 1;
        let n = self.n as f64;
        let dx = x - self.mean_x;
        self.mean_x += dx / n;
        let dy = y - self.mean_y;
        self.mean_y += dy / n;
        // Welford co-moment uses the *updated* y-mean and the *old* x-deviation.
        self.c2 += dx * (y - self.mean_y);
        self.m2_x += dx * (x - self.mean_x);
        self.m2_y += dy * (y - self.mean_y);
    }

    /// Combine another accumulator into this one (parallel/streamed reduction).
    pub fn merge(&mut self, other: &Self) {
        if other.n == 0 {
            return;
        }
        if self.n == 0 {
            *self = *other;
            return;
        }
        let (na, nb) = (self.n as f64, other.n as f64);
        let n = na + nb;
        let dx = other.mean_x - self.mean_x;
        let dy = other.mean_y - self.mean_y;
        *self = Self {
            n: self.n + other.n,
            mean_x: self.mean_x + dx * nb / n,
            mean_y: self.mean_y + dy * nb / n,
            c2: self.c2 + other.c2 + dx * dy * na * nb / n,
            m2_x: self.m2_x + other.m2_x + dx * dx * na * nb / n,
            m2_y: self.m2_y + other.m2_y + dy * dy * na * nb / n,
        };
    }

    pub fn count(&self) -> u64 {
        self.n
    }
    pub fn mean_x(&self) -> f64 {
        if self.n == 0 { f64::NAN } else { self.mean_x }
    }
    pub fn mean_y(&self) -> f64 {
        if self.n == 0 { f64::NAN } else { self.mean_y }
    }
    /// Unbiased sample covariance (`n−1`).
    pub fn covariance(&self) -> f64 {
        if self.n < 2 {
            f64::NAN
        } else {
            self.c2 / (self.n as f64 - 1.0)
        }
    }
    /// Population (biased) covariance (`n`).
    pub fn population_covariance(&self) -> f64 {
        if self.n == 0 {
            f64::NAN
        } else {
            self.c2 / self.n as f64
        }
    }
    /// Pearson correlation `c₂ / √(m2ₓ·m2ᵧ)`.
    pub fn correlation(&self) -> f64 {
        if self.n < 2 || self.m2_x <= 0.0 || self.m2_y <= 0.0 {
            return f64::NAN;
        }
        self.c2 / (self.m2_x * self.m2_y).sqrt()
    }
    /// OLS slope of `y` on `x` (`c₂ / m2ₓ`).
    pub fn slope(&self) -> f64 {
        if self.n < 2 || self.m2_x <= 0.0 {
            f64::NAN
        } else {
            self.c2 / self.m2_x
        }
    }
    /// OLS intercept (`ȳ − slope·x̄`).
    pub fn intercept(&self) -> f64 {
        if self.n < 2 {
            f64::NAN
        } else {
            self.mean_y - self.slope() * self.mean_x
        }
    }
}

impl Extend<(f64, f64)> for RunningCovariance {
    fn extend<I: IntoIterator<Item = (f64, f64)>>(&mut self, iter: I) {
        for (x, y) in iter {
            self.push(x, y);
        }
    }
}

impl FromIterator<(f64, f64)> for RunningCovariance {
    fn from_iter<I: IntoIterator<Item = (f64, f64)>>(iter: I) -> Self {
        let mut s = Self::new();
        s.extend(iter);
        s
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::statistics::summary::{correlation, covariance, variance};

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn covariance_accumulator_matches_batch_and_merges() {
        let xs: Vec<f64> = (0..200).map(|i| i as f64 * 0.1).collect();
        let ys: Vec<f64> = xs
            .iter()
            .map(|&x| 2.0 * x + 1.0 + (x * 0.7).sin())
            .collect();
        let rc: RunningCovariance = xs.iter().copied().zip(ys.iter().copied()).collect();
        close(rc.covariance(), covariance(&xs, &ys), 1e-7);
        close(rc.correlation(), correlation(&xs, &ys), 1e-9);
        // OLS slope = cov/var(x) (≈ 2 here, the linear part).
        close(rc.slope(), covariance(&xs, &ys) / variance(&xs), 1e-7);
        close(
            rc.intercept(),
            rc.mean_y() - rc.slope() * rc.mean_x(),
            1e-12,
        );

        // Split-and-merge equals the one-pass result.
        let mut a: RunningCovariance = xs[..70]
            .iter()
            .copied()
            .zip(ys[..70].iter().copied())
            .collect();
        let b: RunningCovariance = xs[70..]
            .iter()
            .copied()
            .zip(ys[70..].iter().copied())
            .collect();
        a.merge(&b);
        close(a.covariance(), rc.covariance(), 1e-7);
        close(a.correlation(), rc.correlation(), 1e-9);
        assert_eq!(a.count(), 200);
    }
}
