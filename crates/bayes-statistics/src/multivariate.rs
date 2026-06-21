//! Scalar/vector sample statistics that do not require a matrix backend.

/// Dot product of two equal-length real vectors.
pub fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}

/// Squared Euclidean norm.
pub fn norm2(x: &[f64]) -> f64 {
    dot(x, x)
}

/// Return `(rows, columns)` for finite rectangular observations with at least `min_rows`.
pub fn sample_shape(samples: &[Vec<f64>], min_rows: usize) -> Option<(usize, usize)> {
    let n = samples.len();
    let p = samples.first().map_or(0, Vec::len);
    if n < min_rows
        || p == 0
        || samples
            .iter()
            .any(|row| row.len() != p || row.iter().any(|x| !x.is_finite()))
    {
        return None;
    }
    Some((n, p))
}

/// Column means for row-major observations.
pub fn mean_vector(samples: &[Vec<f64>]) -> Vec<f64> {
    let Some((n, p)) = sample_shape(samples, 1) else {
        return Vec::new();
    };
    let mut means = vec![0.0; p];
    for row in samples {
        for j in 0..p {
            means[j] += row[j];
        }
    }
    for m in &mut means {
        *m /= n as f64;
    }
    means
}

/// A running accumulator over paired `(x, y)` observations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunningCovariance {
    n: u64,
    mean_x: f64,
    mean_y: f64,
    c2: f64,
    m2_x: f64,
    m2_y: f64,
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
        self.c2 += dx * (y - self.mean_y);
        self.m2_x += dx * (x - self.mean_x);
        self.m2_y += dy * (y - self.mean_y);
    }

    /// Combine another accumulator into this one.
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

    /// Unbiased sample covariance (`n - 1`).
    pub fn covariance(&self) -> f64 {
        if self.n < 2 {
            f64::NAN
        } else {
            self.c2 / (self.n as f64 - 1.0)
        }
    }

    /// Population covariance (`n`).
    pub fn population_covariance(&self) -> f64 {
        if self.n == 0 {
            f64::NAN
        } else {
            self.c2 / self.n as f64
        }
    }

    /// Pearson correlation `c2 / sqrt(m2_x * m2_y)`.
    pub fn correlation(&self) -> f64 {
        if self.n < 2 || self.m2_x <= 0.0 || self.m2_y <= 0.0 {
            return f64::NAN;
        }
        self.c2 / (self.m2_x * self.m2_y).sqrt()
    }

    /// OLS slope of `y` on `x` (`c2 / m2_x`).
    pub fn slope(&self) -> f64 {
        if self.n < 2 || self.m2_x <= 0.0 {
            f64::NAN
        } else {
            self.c2 / self.m2_x
        }
    }

    /// OLS intercept (`mean_y - slope * mean_x`).
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
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn dot_and_norm_match() {
        let x = [1.0, -2.0, 3.0];
        let y = [4.0, 5.0, -6.0];
        assert_eq!(dot(&x, &y), -24.0);
        assert_eq!(norm2(&x), 14.0);
    }

    #[test]
    fn means_validate_shape_and_finiteness() {
        let samples = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert_eq!(sample_shape(&samples, 1), Some((2, 2)));
        assert_eq!(mean_vector(&samples), vec![2.0, 3.0]);
        assert!(sample_shape(&[vec![1.0], vec![f64::NAN]], 1).is_none());
    }

    #[test]
    fn covariance_accumulator_matches_batch_and_merges() {
        let xs: Vec<f64> = (0..200).map(|i| i as f64 * 0.1).collect();
        let ys: Vec<f64> = xs
            .iter()
            .map(|&x| 2.0 * x + 1.0 + (x * 0.7).sin())
            .collect();
        let rc: RunningCovariance = xs.iter().copied().zip(ys.iter().copied()).collect();

        let mean_x = xs.iter().sum::<f64>() / xs.len() as f64;
        let mean_y = ys.iter().sum::<f64>() / ys.len() as f64;
        let cov = xs
            .iter()
            .zip(&ys)
            .map(|(&x, &y)| (x - mean_x) * (y - mean_y))
            .sum::<f64>()
            / (xs.len() as f64 - 1.0);
        let var_x = xs.iter().map(|&x| (x - mean_x).powi(2)).sum::<f64>() / (xs.len() as f64 - 1.0);
        let var_y = ys.iter().map(|&y| (y - mean_y).powi(2)).sum::<f64>() / (ys.len() as f64 - 1.0);

        close(rc.covariance(), cov, 1e-12);
        close(rc.correlation(), cov / (var_x * var_y).sqrt(), 1e-12);
        close(rc.slope(), cov / var_x, 1e-12);

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
        close(a.covariance(), rc.covariance(), 1e-12);
        close(a.correlation(), rc.correlation(), 1e-12);
        assert_eq!(a.count(), 200);
    }
}
