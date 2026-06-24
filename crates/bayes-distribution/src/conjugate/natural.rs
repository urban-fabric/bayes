//! Natural hyperparameters accumulated by conjugate-prior updates.
//!
//! Closed-form conjugate updates all share the same algebraic shape: start from
//! the prior's natural hyperparameters and add the sufficient statistic for each
//! datum. The types in this module are the small additive carriers used by
//! higher-level conjugate models.

/// A commutative additive monoid of natural hyperparameters.
///
/// [`combine`](Self::combine) is the monoid addition. Implementations should be
/// associative and commutative so that streaming updates agree with batch
/// updates and do not depend on data order.
pub trait NaturalParams: Clone {
    /// Pool two natural-hyperparameter increments.
    #[must_use]
    fn combine(self, other: &Self) -> Self;
}

/// Two real natural hyperparameters `(a, b)` under componentwise addition.
///
/// This carrier is shared by scalar conjugate families such as Beta and Gamma
/// updates.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pair {
    /// First coordinate.
    pub a: f64,
    /// Second coordinate.
    pub b: f64,
}

impl Pair {
    /// Construct a `(a, b)` increment.
    #[must_use]
    pub fn new(a: f64, b: f64) -> Self {
        Self { a, b }
    }
}

impl NaturalParams for Pair {
    fn combine(self, other: &Self) -> Self {
        Self {
            a: self.a + other.a,
            b: self.b + other.b,
        }
    }
}

/// Four natural hyperparameters for Normal-Inverse-Gamma style families.
///
/// The coordinates are `(lambda, lambda * mu, 2 * alpha, 2 * beta + lambda *
/// mu^2)`, represented in the additive coordinate system used by the update.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NigNatural {
    /// Mean pseudo-count `lambda`.
    pub lambda: f64,
    /// Precision-weighted mean sum `lambda * mu`.
    pub lambda_mu: f64,
    /// Twice the inverse-gamma shape, `2 * alpha`.
    pub two_alpha: f64,
    /// Natural sum-of-squares coordinate.
    pub sum_sq: f64,
}

impl NaturalParams for NigNatural {
    fn combine(self, other: &Self) -> Self {
        Self {
            lambda: self.lambda + other.lambda,
            lambda_mu: self.lambda_mu + other.lambda_mu,
            two_alpha: self.two_alpha + other.two_alpha,
            sum_sq: self.sum_sq + other.sum_sq,
        }
    }
}

/// Per-category counts under componentwise addition.
///
/// Different-length operands are zero-extended to the longer length.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Counts(pub Vec<f64>);

impl NaturalParams for Counts {
    fn combine(self, other: &Self) -> Self {
        let (mut acc, addend) = if self.0.len() >= other.0.len() {
            (self.0, &other.0)
        } else {
            (other.0.clone(), &self.0)
        };
        for (slot, &v) in acc.iter_mut().zip(addend.iter()) {
            *slot += v;
        }
        Counts(acc)
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn pair_combine_is_commutative_and_associative() {
        let (x, y, z) = (
            Pair::new(1.0, 2.0),
            Pair::new(3.0, -1.0),
            Pair::new(0.5, 4.0),
        );
        assert_eq!(x.combine(&y), y.combine(&x));
        assert_eq!(x.combine(&y).combine(&z), x.combine(&y.combine(&z)));
    }

    #[test]
    fn counts_combine_zero_extends_to_the_longer_operand() {
        let short = Counts(vec![1.0, 2.0]);
        let long = Counts(vec![10.0, 20.0, 30.0]);

        assert_eq!(short.clone().combine(&long), long.clone().combine(&short));
        assert_eq!(short.combine(&long).0, vec![11.0, 22.0, 30.0]);
    }
}
