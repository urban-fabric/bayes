//! **Probability generating functions** `G(z) = E[z^X]` for discrete families.
//!
//! The PGF is the discrete counterpart of the [`MomentGenerating`](crate::MomentGenerating)
//! surface (`G(z) = M(ln z)`): a power series in `z` whose coefficients are the
//! point masses `P(X = k)`. It is the natural tool for sums of independent counts
//! (PGFs multiply) and for branching/queueing recursions, and its derivatives at
//! `z = 1` give the factorial moments `E[X(X−1)…(X−k+1)] = G^{(k)}(1)`.
//!
//! `G(1) = 1` for every family. The series converges for `|z| ≤ 1` always; for the
//! geometric and negative-binomial it diverges past `z = 1/(1−p)`, where these
//! methods return `+∞`.

use crate::{Bernoulli, Binomial, DiscreteUniform, Geometric, NegativeBinomial, Poisson};

/// A discrete family with a closed-form probability generating function.
pub trait ProbabilityGenerating {
    /// `G(z) = E[z^X] = Σ_k P(X = k)·z^k`. Returns `+∞` where the series
    /// diverges.
    fn pgf(&self, z: f64) -> f64;
}

/// `G(z) = (1 − p) + p·z`.
impl ProbabilityGenerating for Bernoulli {
    fn pgf(&self, z: f64) -> f64 {
        let p = self.p();
        (1.0 - p) + p * z
    }
}

/// `G(z) = ((1 − p) + p·z)^n`.
impl ProbabilityGenerating for Binomial {
    fn pgf(&self, z: f64) -> f64 {
        let p = self.p();
        ((1.0 - p) + p * z).powi(self.trials() as i32)
    }
}

/// `G(z) = exp(λ(z − 1))`.
impl ProbabilityGenerating for Poisson {
    fn pgf(&self, z: f64) -> f64 {
        (self.lambda() * (z - 1.0)).exp()
    }
}

/// `G(z) = p·z / (1 − (1−p)·z)` for `(1−p)z < 1` (trial-count
/// parameterization), else `+∞`.
impl ProbabilityGenerating for Geometric {
    fn pgf(&self, z: f64) -> f64 {
        let p = self.p();
        let q_z = (1.0 - p) * z;
        if q_z < 1.0 {
            p * z / (1.0 - q_z)
        } else {
            f64::INFINITY
        }
    }
}

/// `G(z) = (p / (1 − (1−p)·z))^r` for `(1−p)z < 1`, else `+∞`.
impl ProbabilityGenerating for NegativeBinomial {
    fn pgf(&self, z: f64) -> f64 {
        let p = self.p();
        let q_z = (1.0 - p) * z;
        if q_z < 1.0 {
            (p / (1.0 - q_z)).powf(self.successes())
        } else {
            f64::INFINITY
        }
    }
}

/// `G(z) = z^lo·(z^n − 1) / (n(z − 1))` over the support `{lo, …, hi−1}`
/// (`hi` exclusive) with `n = hi−lo` (and `G(1) = 1`).
impl ProbabilityGenerating for DiscreteUniform {
    fn pgf(&self, z: f64) -> f64 {
        let (lo, hi) = (self.lo(), self.hi());
        let n = (hi - lo) as f64;
        if (z - 1.0).abs() < 1e-12 {
            return 1.0;
        }
        z.powi(lo as i32) * (z.powi((hi - lo) as i32) - 1.0) / (n * (z - 1.0))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::Moments;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn pgf_is_one_at_one() {
        let dists: [&dyn ProbabilityGenerating; 6] = [
            &Bernoulli::new(0.4).unwrap(),
            &Binomial::new(7, 0.3).unwrap(),
            &Poisson::new(3.0).unwrap(),
            &Geometric::new(0.5).unwrap(),
            &NegativeBinomial::new(3.0, 0.6).unwrap(),
            &DiscreteUniform::new(-2, 5).unwrap(),
        ];
        for d in dists {
            close(d.pgf(1.0), 1.0, 1e-12);
        }
    }

    #[test]
    fn pgf_closed_forms_and_divergence() {
        // Poisson(2): G(0) = e^{−2} = P(X = 0).
        close(Poisson::new(2.0).unwrap().pgf(0.0), (-2.0_f64).exp(), 1e-12);
        // Bernoulli(0.3): G(0) = 1 − p = P(X = 0).
        close(Bernoulli::new(0.3).unwrap().pgf(0.0), 0.7, 1e-12);
        // Geometric(0.5): G(z) finite below z = 1/(1−p) = 2; diverges at/after.
        let g = Geometric::new(0.5).unwrap();
        close(g.pgf(0.0), 0.0, 1e-12); // support starts at 1
        assert_eq!(g.pgf(2.0), f64::INFINITY);
        assert_eq!(NegativeBinomial::new(2.0, 0.5).unwrap().pgf(2.0), f64::INFINITY);
    }

    #[test]
    fn pgf_first_derivative_at_one_is_the_mean() {
        // G'(1) = E[X], checked by central difference.
        let check = |d: &dyn ProbabilityGenerating, mean: f64| {
            let h = 1e-5;
            close((d.pgf(1.0 + h) - d.pgf(1.0 - h)) / (2.0 * h), mean, 1e-5);
        };
        let po = Poisson::new(4.0).unwrap();
        check(&po, po.mean());
        let b = Binomial::new(10, 0.3).unwrap();
        check(&b, b.mean());
        let du = DiscreteUniform::new(1, 6).unwrap();
        check(&du, du.mean());
    }
}
