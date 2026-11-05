//! Scalar/vector sample statistics that do not require a matrix backend.

use euclid_numeric::{Boolean, Compare, Real};

/// Dot product of two equal-length real vectors.
pub fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}

/// Dot product over any [`Real`] carrier.
pub fn dot_real<R: Real>(x: &[R], y: &[R]) -> R {
    x.iter()
        .cloned()
        .zip(y.iter().cloned())
        .map(|(a, b)| a * b)
        .fold(R::zero(), |acc, x| acc + x)
}

/// Squared Euclidean norm.
pub fn norm2(x: &[f64]) -> f64 {
    dot(x, x)
}

/// Squared Euclidean norm over any [`Real`] carrier.
pub fn norm2_real<R: Real>(x: &[R]) -> R {
    dot_real(x, x)
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

/// Column means for row-major observations over any [`Real`] carrier.
pub fn mean_vector_real<R: Real>(samples: &[Vec<R>]) -> Vec<R> {
    let n = samples.len();
    let p = samples.first().map_or(0, Vec::len);
    if n == 0 || p == 0 || samples.iter().any(|row| row.len() != p) {
        return Vec::new();
    }
    let mut means = vec![R::zero(); p];
    for row in samples {
        for j in 0..p {
            means[j] = means[j].clone() + row[j].clone();
        }
    }
    let n_real = R::scalar(n as f64);
    for m in &mut means {
        *m = m.clone() / n_real.clone();
    }
    means
}

/// A running accumulator over paired `(x, y)` observations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunningCovariance<R: Real = f64> {
    n: u64,
    mean_x: R,
    mean_y: R,
    c2: R,
    m2_x: R,
    m2_y: R,
}

impl<R> Default for RunningCovariance<R>
where
    R: Real + Compare,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<R> RunningCovariance<R>
where
    R: Real + Compare,
{
    pub fn new() -> Self {
        Self {
            n: 0,
            mean_x: R::zero(),
            mean_y: R::zero(),
            c2: R::zero(),
            m2_x: R::zero(),
            m2_y: R::zero(),
        }
    }

    /// Absorb one paired observation.
    pub fn push(&mut self, x: R, y: R) {
        self.n += 1;
        let n = R::scalar(self.n as f64);
        let dx = x.clone() - self.mean_x.clone();
        self.mean_x = self.mean_x.clone() + dx.clone() / n.clone();
        let dy = y.clone() - self.mean_y.clone();
        self.mean_y = self.mean_y.clone() + dy.clone() / n;
        self.c2 = self.c2.clone() + dx.clone() * (y.clone() - self.mean_y.clone());
        self.m2_x = self.m2_x.clone() + dx * (x - self.mean_x.clone());
        self.m2_y = self.m2_y.clone() + dy * (y - self.mean_y.clone());
    }

    /// Combine another accumulator into this one.
    pub fn merge(&mut self, other: &Self) {
        if other.n == 0 {
            return;
        }
        if self.n == 0 {
            *self = other.clone();
            return;
        }
        let (na, nb) = (R::scalar(self.n as f64), R::scalar(other.n as f64));
        let n = na.clone() + nb.clone();
        let dx = other.mean_x.clone() - self.mean_x.clone();
        let dy = other.mean_y.clone() - self.mean_y.clone();
        *self = Self {
            n: self.n + other.n,
            mean_x: self.mean_x.clone() + dx.clone() * nb.clone() / n.clone(),
            mean_y: self.mean_y.clone() + dy.clone() * nb.clone() / n.clone(),
            c2: self.c2.clone()
                + other.c2.clone()
                + dx.clone() * dy.clone() * na.clone() * nb.clone() / n.clone(),
            m2_x: self.m2_x.clone()
                + other.m2_x.clone()
                + dx.clone() * dx * na.clone() * nb.clone() / n.clone(),
            m2_y: self.m2_y.clone() + other.m2_y.clone() + dy.clone() * dy * na * nb / n,
        };
    }

    pub fn count(&self) -> u64 {
        self.n
    }

    pub fn mean_x(&self) -> R {
        if self.n == 0 {
            R::nan()
        } else {
            self.mean_x.clone()
        }
    }

    pub fn mean_y(&self) -> R {
        if self.n == 0 {
            R::nan()
        } else {
            self.mean_y.clone()
        }
    }

    /// Unbiased sample covariance (`n - 1`).
    pub fn covariance(&self) -> R {
        if self.n < 2 {
            R::nan()
        } else {
            self.c2.clone() / R::scalar(self.n as f64 - 1.0)
        }
    }

    /// Population covariance (`n`).
    pub fn population_covariance(&self) -> R {
        if self.n == 0 {
            R::nan()
        } else {
            self.c2.clone() / R::scalar(self.n as f64)
        }
    }

    /// Pearson correlation `c2 / sqrt(m2_x * m2_y)`.
    pub fn correlation(&self) -> R {
        if self.n < 2 {
            return R::nan();
        }
        let positive = self.m2_x.gt(&R::zero()).and(&self.m2_y.gt(&R::zero()));
        let value = self.c2.clone() / (self.m2_x.clone() * self.m2_y.clone()).sqrt();
        R::select(&positive, value, R::nan())
    }

    /// OLS slope of `y` on `x` (`c2 / m2_x`).
    pub fn slope(&self) -> R {
        if self.n < 2 {
            return R::nan();
        }
        let positive = self.m2_x.gt(&R::zero());
        R::select(&positive, self.c2.clone() / self.m2_x.clone(), R::nan())
    }

    /// OLS intercept (`mean_y - slope * mean_x`).
    pub fn intercept(&self) -> R {
        if self.n < 2 {
            R::nan()
        } else {
            self.mean_y.clone() - self.slope() * self.mean_x.clone()
        }
    }
}

impl<R> Extend<(R, R)> for RunningCovariance<R>
where
    R: Real + Compare,
{
    fn extend<I: IntoIterator<Item = (R, R)>>(&mut self, iter: I) {
        for (x, y) in iter {
            self.push(x, y);
        }
    }
}

impl<R> FromIterator<(R, R)> for RunningCovariance<R>
where
    R: Real + Compare,
{
    fn from_iter<I: IntoIterator<Item = (R, R)>>(iter: I) -> Self {
        let mut s = Self::new();
        s.extend(iter);
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bayes_distribution_ext::assert_close;

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

        assert_close!(rc.covariance(), cov, 1e-12);
        assert_close!(rc.correlation(), cov / (var_x * var_y).sqrt(), 1e-12);
        assert_close!(rc.slope(), cov / var_x, 1e-12);

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
        assert_close!(a.covariance(), rc.covariance(), 1e-12);
        assert_close!(a.correlation(), rc.correlation(), 1e-12);
        assert_eq!(a.count(), 200);
    }
}
