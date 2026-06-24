//! Support constraints for the lazy distribution families.

use bayes_core::{Boundary, Constrained, Constraint, IntegerConstraint, NaturalConstraint};
use euclid::{Compare, Integer, Natural, Real};

use crate::continuous::{
    Beta, Cauchy, ChiSquared, Exponential, FisherSnedecor, Frechet, Gamma, Gumbel, HalfNormal,
    InverseGamma, Laplace, LogNormal, Logistic, Lomax, Normal, Pareto, Rayleigh, StudentT, Uniform,
    VonMises, Wald, Weibull,
};
use crate::discrete::{
    Bernoulli, BetaBinomial, Binomial, DiscreteUniform, Geometric, Hypergeometric,
    NegativeBinomial, Poisson,
};

impl<R: Real> Constrained for Cauchy<R> {
    type Value = R;
    type Constraint = Constraint<R>;

    fn support_constraint(&self) -> Self::Constraint {
        Constraint::Real
    }
}

impl<R: Real> Constrained for Gumbel<R> {
    type Value = R;
    type Constraint = Constraint<R>;

    fn support_constraint(&self) -> Self::Constraint {
        Constraint::Real
    }
}

impl<R: Real> Constrained for Laplace<R> {
    type Value = R;
    type Constraint = Constraint<R>;

    fn support_constraint(&self) -> Self::Constraint {
        Constraint::Real
    }
}

impl<R: Real> Constrained for Logistic<R> {
    type Value = R;
    type Constraint = Constraint<R>;

    fn support_constraint(&self) -> Self::Constraint {
        Constraint::Real
    }
}

impl<R: Real> Constrained for Normal<R> {
    type Value = R;
    type Constraint = Constraint<R>;

    fn support_constraint(&self) -> Self::Constraint {
        Constraint::Real
    }
}

impl<R: Real> Constrained for StudentT<R> {
    type Value = R;
    type Constraint = Constraint<R>;

    fn support_constraint(&self) -> Self::Constraint {
        Constraint::Real
    }
}

impl<R: Real> Constrained for ChiSquared<R> {
    type Value = R;
    type Constraint = Constraint<R>;

    fn support_constraint(&self) -> Self::Constraint {
        Constraint::Positive
    }
}

impl<R: Real> Constrained for FisherSnedecor<R> {
    type Value = R;
    type Constraint = Constraint<R>;

    fn support_constraint(&self) -> Self::Constraint {
        Constraint::Positive
    }
}

impl<R: Real> Constrained for Frechet<R> {
    type Value = R;
    type Constraint = Constraint<R>;

    fn support_constraint(&self) -> Self::Constraint {
        Constraint::Positive
    }
}

impl<R: Real> Constrained for Gamma<R> {
    type Value = R;
    type Constraint = Constraint<R>;

    fn support_constraint(&self) -> Self::Constraint {
        Constraint::Positive
    }
}

impl<R: Real> Constrained for InverseGamma<R> {
    type Value = R;
    type Constraint = Constraint<R>;

    fn support_constraint(&self) -> Self::Constraint {
        Constraint::Positive
    }
}

impl<R: Real> Constrained for LogNormal<R> {
    type Value = R;
    type Constraint = Constraint<R>;

    fn support_constraint(&self) -> Self::Constraint {
        Constraint::Positive
    }
}

impl<R: Real> Constrained for Wald<R> {
    type Value = R;
    type Constraint = Constraint<R>;

    fn support_constraint(&self) -> Self::Constraint {
        Constraint::Positive
    }
}

impl<R: Real> Constrained for Exponential<R> {
    type Value = R;
    type Constraint = Constraint<R>;

    fn support_constraint(&self) -> Self::Constraint {
        Constraint::NonNegative
    }
}

impl<R: Real> Constrained for HalfNormal<R> {
    type Value = R;
    type Constraint = Constraint<R>;

    fn support_constraint(&self) -> Self::Constraint {
        Constraint::NonNegative
    }
}

impl<R: Real> Constrained for Lomax<R> {
    type Value = R;
    type Constraint = Constraint<R>;

    fn support_constraint(&self) -> Self::Constraint {
        Constraint::NonNegative
    }
}

impl<R: Real> Constrained for Rayleigh<R> {
    type Value = R;
    type Constraint = Constraint<R>;

    fn support_constraint(&self) -> Self::Constraint {
        Constraint::NonNegative
    }
}

impl<R: Real> Constrained for Weibull<R> {
    type Value = R;
    type Constraint = Constraint<R>;

    fn support_constraint(&self) -> Self::Constraint {
        Constraint::NonNegative
    }
}

impl<R: Real> Constrained for Beta<R> {
    type Value = R;
    type Constraint = Constraint<R>;

    fn support_constraint(&self) -> Self::Constraint {
        Constraint::UnitInterval
    }
}

impl<R: Real> Constrained for Pareto<R> {
    type Value = R;
    type Constraint = Constraint<R>;

    fn support_constraint(&self) -> Self::Constraint {
        Constraint::Lower {
            lo: self.scale(),
            boundary: Boundary::Closed,
        }
    }
}

impl<R: Real> Constrained for Uniform<R> {
    type Value = R;
    type Constraint = Constraint<R>;

    fn support_constraint(&self) -> Self::Constraint {
        Constraint::ClosedInterval {
            lo: self.lo(),
            hi: self.hi(),
        }
    }
}

impl<R: Real> Constrained for VonMises<R> {
    type Value = R;
    type Constraint = Constraint<R>;

    fn support_constraint(&self) -> Self::Constraint {
        Constraint::Bounded {
            lo: -R::pi(),
            hi: R::pi(),
            lower: Boundary::Closed,
            upper: Boundary::Open,
        }
    }
}

impl<R: Real> Constrained for Bernoulli<R> {
    type Value = u64;
    type Constraint = NaturalConstraint<u64>;

    fn support_constraint(&self) -> Self::Constraint {
        NaturalConstraint::Interval { lo: 0, hi: 1 }
    }
}

impl<N: Natural> Constrained for BetaBinomial<N> {
    type Value = N;
    type Constraint = NaturalConstraint<N>;

    fn support_constraint(&self) -> Self::Constraint {
        NaturalConstraint::Interval {
            lo: N::from_u64(0),
            hi: self.trials(),
        }
    }
}

impl<N: Natural> Constrained for Binomial<N> {
    type Value = N;
    type Constraint = NaturalConstraint<N>;

    fn support_constraint(&self) -> Self::Constraint {
        NaturalConstraint::Interval {
            lo: N::from_u64(0),
            hi: self.trials(),
        }
    }
}

impl<I: Integer> Constrained for DiscreteUniform<I> {
    type Value = I;
    type Constraint = IntegerConstraint<I>;

    fn support_constraint(&self) -> Self::Constraint {
        IntegerConstraint::Interval {
            lo: self.lo(),
            hi: self.hi() - I::from_i64(1),
        }
    }
}

impl<R: Real> Constrained for Geometric<R> {
    type Value = u64;
    type Constraint = NaturalConstraint<u64>;

    fn support_constraint(&self) -> Self::Constraint {
        NaturalConstraint::AtLeast { lo: 1 }
    }
}

impl<N> Constrained for Hypergeometric<N>
where
    N: Natural + Compare,
{
    type Value = N;
    type Constraint = NaturalConstraint<N>;

    fn support_constraint(&self) -> Self::Constraint {
        let successes = self.successes();
        let draws = self.draws();
        let successes_le_draws = successes.le(&draws);
        let hi = N::select(&successes_le_draws, successes, draws);
        NaturalConstraint::Interval {
            lo: N::from_u64(0),
            hi,
        }
    }
}

impl<R: Real> Constrained for NegativeBinomial<R> {
    type Value = u64;
    type Constraint = NaturalConstraint<u64>;

    fn support_constraint(&self) -> Self::Constraint {
        NaturalConstraint::AtLeast { lo: 0 }
    }
}

impl<R: Real> Constrained for Poisson<R> {
    type Value = u64;
    type Constraint = NaturalConstraint<u64>;

    fn support_constraint(&self) -> Self::Constraint {
        NaturalConstraint::AtLeast { lo: 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuous_distribution_constraints_match_support() {
        let gamma = Gamma::<f64>::new(2.0, 3.0);
        assert!(!gamma.support_constraint().contains(0.0));
        assert!(gamma.support_constraint().contains(1.0));

        let uniform = Uniform::<f64>::new(-2.0, 5.0);
        assert!(uniform.support_constraint().contains(-2.0));
        assert!(uniform.support_constraint().contains(5.0));
        assert!(!uniform.support_constraint().contains(6.0));

        let beta = Beta::<f64>::new(2.0, 5.0);
        assert!(beta.support_constraint().contains(0.5));
        assert!(!beta.support_constraint().contains(1.0));
    }

    #[test]
    fn discrete_distribution_constraints_match_support() {
        let binomial = Binomial::new(10_u64, 0.4);
        assert!(binomial.support_constraint().contains(0));
        assert!(binomial.support_constraint().contains(10));
        assert!(!binomial.support_constraint().contains(11));

        let uniform = DiscreteUniform::new(-3_i64, 4_i64);
        assert!(uniform.support_constraint().contains(-3));
        assert!(uniform.support_constraint().contains(3));
        assert!(!uniform.support_constraint().contains(4));
    }
}
