//! Natural hyperparameters — the additive monoid a conjugate posterior
//! accumulates in.
//!
//! Every conjugate update in this module is the *same* algebraic move, proved
//! once (generically over the monoid) in the `riemann` Lean development
//! (`RiemannLean/Bayes/ExponentialFamily.lean`): the posterior hyperparameter is
//! the prior plus the data's sufficient statistic,
//!
//! ```text
//! posterior η T data = η + Σ T(xᵢ)
//! ```
//!
//! taken in a commutative additive monoid `H` of hyperparameters. **Associativity**
//! of that monoid is why online updating equals batch (`posterior_online_eq_batch`);
//! **commutativity** is why the posterior is order-independent (`posterior_perm`).
//!
//! [`NaturalParams`] is that monoid `H`; the carriers below are the `H` for the
//! concrete families: [`Pair`] (`ℝ²`, the Beta/Gamma/Gaussian pairs),
//! [`NigNatural`] (`ℝ⁴`, Normal–Inverse-Gamma / Normal-Gamma), and [`Counts`]
//! (`ℝᴷ`, Dirichlet).

/// A commutative additive monoid of natural hyperparameters — the `H` of the
/// riemann umbrella law `Bayes/ExponentialFamily` (`posterior η T data = η + Σ
/// T(xᵢ)`).
///
/// [`combine`](Self::combine) is the monoid `+`: it must be **associative** and
/// **commutative**. Those two laws — machine-checked in riemann — are exactly
/// what make the generic conjugate update
/// ([`Conjugate::posterior`](super::Conjugate::posterior)) agree across any
/// batch split (online = batch) and ignore data order.
///
/// This is the same commutative-monoid shape as
/// [`SufficientStatistic::merge`](crate::SufficientStatistic::merge),
/// the exponential-family MLE accumulator: there the monoid pools sufficient
/// statistics toward a point estimate, here it pools natural hyperparameters
/// toward a posterior. Both get streaming-equals-batch and order-independence
/// from the same associativity/commutativity; they stay distinct types because
/// the carriers and recovered objects differ (a fitted distribution vs. a
/// posterior law).
pub trait NaturalParams: Clone {
    /// Pool two natural-hyperparameter increments (the monoid `+`).
    #[must_use]
    fn combine(self, other: &Self) -> Self;
}

/// Two real natural hyperparameters `(a, b)` under componentwise addition — the
/// `H = ℝ²` shared by most conjugate families: the Beta pseudo-counts `(α, β)`,
/// the Gamma `(shape, rate)`, and the known-variance Gaussian
/// `(precision, precision·mean)`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pair {
    /// First coordinate (`α`, `shape`, or `precision`).
    pub a: f64,
    /// Second coordinate (`β`, `rate`, or `precision·mean`).
    pub b: f64,
}

impl Pair {
    /// A `(a, b)` increment.
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

/// The four natural hyperparameters of the Normal–Inverse-Gamma / Normal-Gamma family,
/// `(λ, λμ, 2α, 2β + λμ²)`, under componentwise addition — the coordinate system
/// in which the NIG update is the additive `η + Σ (1, x, 1, x²)` of riemann's
/// `normalInverseGamma_online_eq_batch` (`nigStat x = (1, x, x²)`, the two count
/// slots `λ` and `2α` sharing each datum's `+1`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NigNatural {
    /// Mean pseudo-count `λ`.
    pub lambda: f64,
    /// Precision-weighted mean sum `λμ`.
    pub lambda_mu: f64,
    /// Twice the Inverse-Gamma shape, `2α`.
    pub two_alpha: f64,
    /// The natural sum-of-squares coordinate `2β + λμ²`.
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

/// Per-category counts `H = ℝᴷ` for the Dirichlet families, under componentwise
/// addition (the finitely-supported sequences, zero-extended to the longer
/// operand). The `Fin K → ℝ` hyperparameter of riemann's
/// `dirichletCat_online_eq_batch`; the family's `statistic` confines each
/// increment to the prior's `K` categories so the accumulator stays length `K`.
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
        // commutative even at differing lengths
        assert_eq!(short.clone().combine(&long), long.clone().combine(&short));
        assert_eq!(short.combine(&long).0, vec![11.0, 22.0, 30.0]);
    }
}
