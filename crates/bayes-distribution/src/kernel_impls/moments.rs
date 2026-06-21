use crate::Moments;
use crate::continuous::{
    Beta, Cauchy, ChiSquared, Exponential, FisherSnedecor, Frechet, Gamma, Gumbel, HalfNormal,
    InverseGamma, Laplace, LogNormal, Logistic, Lomax, Normal, Pareto, Rayleigh, StudentT, Uniform,
    VonMises, Wald, Weibull,
};
use crate::discrete::{
    Bernoulli, BetaBinomial, Binomial, DiscreteUniform, Geometric, Hypergeometric,
    NegativeBinomial, Poisson,
};
use bayes_kernels as kernels;
use euclid::{LazyBool, LazyCompare, LazyInteger, LazyNatural, LazyReal};

fn moment2<T: LazyReal>(mean: T, variance: T) -> T {
    variance + mean.clone() * mean
}

fn select<T: LazyReal>(mask: &T::Mask, on_true: T, on_false: T) -> T {
    T::select(mask, on_true, on_false)
}

impl<T: LazyReal> Moments<T> for Beta<T> {
    fn mean(&self) -> T {
        let alpha = self.alpha();
        alpha.clone() / (alpha + self.beta())
    }

    fn variance(&self) -> T {
        kernels::beta_variance(self.alpha(), self.beta())
    }

    fn moment2(&self) -> T {
        moment2(self.mean(), self.variance())
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        kernels::beta_skewness(self.alpha(), self.beta())
    }

    fn kurtosis(&self) -> T {
        T::nan()
    }

    fn entropy(&self) -> T {
        kernels::beta_entropy(self.alpha(), self.beta())
    }

    fn median(&self) -> T {
        T::nan()
    }

    fn mode(&self) -> T {
        let alpha = self.alpha();
        let beta = self.beta();
        let valid = alpha.gt(&T::one()).and(&beta.gt(&T::one()));
        let body = (alpha.clone() - T::one()) / (alpha + beta - T::two());
        select(&valid, body, T::nan())
    }
}

impl<T: LazyReal> Moments<T> for Cauchy<T> {
    fn mean(&self) -> T {
        T::nan()
    }

    fn variance(&self) -> T {
        T::nan()
    }

    fn moment2(&self) -> T {
        T::nan()
    }

    fn std_dev(&self) -> T {
        T::nan()
    }

    fn skewness(&self) -> T {
        T::nan()
    }

    fn kurtosis(&self) -> T {
        T::nan()
    }

    fn entropy(&self) -> T {
        kernels::cauchy_entropy(self.scale())
    }

    fn median(&self) -> T {
        self.location()
    }

    fn mode(&self) -> T {
        self.location()
    }
}

impl<T: LazyReal> Moments<T> for ChiSquared<T> {
    fn mean(&self) -> T {
        self.freedom()
    }

    fn variance(&self) -> T {
        T::two() * self.freedom()
    }

    fn moment2(&self) -> T {
        moment2(self.mean(), self.variance())
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        kernels::chi_squared_skewness(self.freedom())
    }

    fn kurtosis(&self) -> T {
        kernels::chi_squared_excess_kurtosis(self.freedom())
    }

    fn entropy(&self) -> T {
        kernels::chi_squared_entropy(self.freedom() * T::half())
    }

    fn median(&self) -> T {
        T::nan()
    }

    fn mode(&self) -> T {
        (self.freedom() - T::two()).max(T::zero())
    }
}

impl<T: LazyReal> Moments<T> for Exponential<T> {
    fn mean(&self) -> T {
        self.rate().recip()
    }

    fn variance(&self) -> T {
        let rate = self.rate();
        T::one() / (rate.clone() * rate)
    }

    fn moment2(&self) -> T {
        moment2(self.mean(), self.variance())
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        T::two()
    }

    fn kurtosis(&self) -> T {
        T::scalar(6.0)
    }

    fn entropy(&self) -> T {
        kernels::exponential_entropy(self.rate())
    }

    fn median(&self) -> T {
        T::ln_2() / self.rate()
    }

    fn mode(&self) -> T {
        T::zero()
    }
}

impl<T: LazyReal> Moments<T> for FisherSnedecor<T> {
    fn mean(&self) -> T {
        let d2 = self.d2();
        let valid = d2.gt(&T::two());
        let body = d2.clone() / (d2 - T::two());
        select(&valid, body, T::nan())
    }

    fn variance(&self) -> T {
        let d1 = self.d1();
        let d2 = self.d2();
        let valid = d2.gt(&T::scalar(4.0));
        let body = T::two() * d2.clone() * d2.clone() * (d1.clone() + d2.clone() - T::two())
            / (d1 * (d2.clone() - T::two()) * (d2.clone() - T::two()) * (d2 - T::scalar(4.0)));
        select(&valid, body, T::nan())
    }

    fn moment2(&self) -> T {
        moment2(self.mean(), self.variance())
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        T::nan()
    }

    fn kurtosis(&self) -> T {
        T::nan()
    }

    fn entropy(&self) -> T {
        T::nan()
    }

    fn median(&self) -> T {
        T::nan()
    }

    fn mode(&self) -> T {
        let d1 = self.d1();
        let d2 = self.d2();
        let valid = d1.gt(&T::two());
        let body = (d1.clone() - T::two()) / d1 * d2.clone() / (d2 + T::two());
        select(&valid, body, T::zero())
    }
}

impl<T: LazyReal> Moments<T> for Frechet<T> {
    fn mean(&self) -> T {
        let shape = self.shape();
        let valid = shape.gt(&T::one());
        let body = self.scale() * (T::one() - T::one() / shape).gamma();
        select(&valid, body, T::nan())
    }

    fn variance(&self) -> T {
        let shape = self.shape();
        let valid = shape.gt(&T::two());
        let g1 = (T::one() - T::one() / shape.clone()).gamma();
        let g2 = (T::one() - T::two() / shape).gamma();
        let body = self.scale() * self.scale() * (g2 - g1.clone() * g1);
        select(&valid, body, T::nan())
    }

    fn moment2(&self) -> T {
        moment2(self.mean(), self.variance())
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        T::nan()
    }

    fn kurtosis(&self) -> T {
        T::nan()
    }

    fn entropy(&self) -> T {
        T::nan()
    }

    fn median(&self) -> T {
        self.scale() * T::ln_2().powf(-T::one() / self.shape())
    }

    fn mode(&self) -> T {
        let shape = self.shape();
        self.scale() * (shape.clone() / (shape.clone() + T::one())).powf(T::one() / shape)
    }
}

impl<T: LazyReal> Moments<T> for Gamma<T> {
    fn mean(&self) -> T {
        self.shape() / self.rate()
    }

    fn variance(&self) -> T {
        kernels::gamma_variance(self.shape(), self.rate())
    }

    fn moment2(&self) -> T {
        moment2(self.mean(), self.variance())
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        kernels::gamma_skewness(self.shape())
    }

    fn kurtosis(&self) -> T {
        kernels::gamma_excess_kurtosis(self.shape())
    }

    fn entropy(&self) -> T {
        kernels::gamma_entropy(self.shape(), self.rate())
    }

    fn median(&self) -> T {
        T::nan()
    }

    fn mode(&self) -> T {
        let shape = self.shape();
        let valid = shape.gt(&T::one());
        let body = (shape - T::one()) / self.rate();
        select(&valid, body, T::zero())
    }
}

impl<T: LazyReal> Moments<T> for Gumbel<T> {
    fn mean(&self) -> T {
        self.location() + self.scale() * T::euler_gamma()
    }

    fn variance(&self) -> T {
        T::pi() * T::pi() * self.scale() * self.scale() / T::scalar(6.0)
    }

    fn moment2(&self) -> T {
        moment2(self.mean(), self.variance())
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        T::scalar(1.139_547_099_404_649)
    }

    fn kurtosis(&self) -> T {
        T::nan()
    }

    fn entropy(&self) -> T {
        kernels::gumbel_entropy(self.scale())
    }

    fn median(&self) -> T {
        self.location() - self.scale() * T::ln_2().ln()
    }

    fn mode(&self) -> T {
        self.location()
    }
}

impl<T: LazyReal> Moments<T> for HalfNormal<T> {
    fn mean(&self) -> T {
        self.scale() * (T::two() / T::pi()).sqrt()
    }

    fn variance(&self) -> T {
        self.scale() * self.scale() * (T::one() - T::two() / T::pi())
    }

    fn moment2(&self) -> T {
        moment2(self.mean(), self.variance())
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        T::nan()
    }

    fn kurtosis(&self) -> T {
        T::nan()
    }

    fn entropy(&self) -> T {
        kernels::half_normal_entropy(self.scale())
    }

    fn median(&self) -> T {
        self.scale() * T::sqrt_2() * T::half().erf_inv()
    }

    fn mode(&self) -> T {
        T::zero()
    }
}

impl<T: LazyReal> Moments<T> for InverseGamma<T> {
    fn mean(&self) -> T {
        let shape = self.shape();
        let valid = shape.gt(&T::one());
        let body = self.scale() / (shape - T::one());
        select(&valid, body, T::nan())
    }

    fn variance(&self) -> T {
        let shape = self.shape();
        let valid = shape.gt(&T::two());
        let body = self.scale() * self.scale()
            / ((shape.clone() - T::one()) * (shape.clone() - T::one()) * (shape - T::two()));
        select(&valid, body, T::nan())
    }

    fn moment2(&self) -> T {
        moment2(self.mean(), self.variance())
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        T::nan()
    }

    fn kurtosis(&self) -> T {
        T::nan()
    }

    fn entropy(&self) -> T {
        kernels::inverse_gamma_entropy(self.shape(), self.scale())
    }

    fn median(&self) -> T {
        T::nan()
    }

    fn mode(&self) -> T {
        self.scale() / (self.shape() + T::one())
    }
}

impl<T: LazyReal> Moments<T> for Laplace<T> {
    fn mean(&self) -> T {
        self.location()
    }

    fn variance(&self) -> T {
        T::two() * self.scale() * self.scale()
    }

    fn moment2(&self) -> T {
        moment2(self.mean(), self.variance())
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        T::zero()
    }

    fn kurtosis(&self) -> T {
        T::scalar(3.0)
    }

    fn entropy(&self) -> T {
        kernels::laplace_entropy(self.scale())
    }

    fn median(&self) -> T {
        self.location()
    }

    fn mode(&self) -> T {
        self.location()
    }
}

impl<T: LazyReal> Moments<T> for Logistic<T> {
    fn mean(&self) -> T {
        self.location()
    }

    fn variance(&self) -> T {
        self.scale() * self.scale() * T::pi() * T::pi() / T::scalar(3.0)
    }

    fn moment2(&self) -> T {
        moment2(self.mean(), self.variance())
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        T::zero()
    }

    fn kurtosis(&self) -> T {
        T::scalar(1.2)
    }

    fn entropy(&self) -> T {
        kernels::logistic_entropy(self.scale())
    }

    fn median(&self) -> T {
        self.location()
    }

    fn mode(&self) -> T {
        self.location()
    }
}

impl<T: LazyReal> Moments<T> for LogNormal<T> {
    fn mean(&self) -> T {
        kernels::lognormal_mean(self.mu(), self.sigma())
    }

    fn variance(&self) -> T {
        kernels::lognormal_variance(self.mu(), self.sigma())
    }

    fn moment2(&self) -> T {
        moment2(self.mean(), self.variance())
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        T::nan()
    }

    fn kurtosis(&self) -> T {
        T::nan()
    }

    fn entropy(&self) -> T {
        kernels::lognormal_entropy(self.mu(), self.sigma())
    }

    fn median(&self) -> T {
        self.mu().exp()
    }

    fn mode(&self) -> T {
        let sigma = self.sigma();
        (self.mu() - sigma.clone() * sigma).exp()
    }
}

impl<T: LazyReal> Moments<T> for Lomax<T> {
    fn mean(&self) -> T {
        let shape = self.shape();
        let valid = shape.gt(&T::one());
        let body = self.scale() / (shape - T::one());
        select(&valid, body, T::nan())
    }

    fn variance(&self) -> T {
        let shape = self.shape();
        let valid = shape.gt(&T::two());
        let body = self.scale() * self.scale() * shape.clone()
            / ((shape.clone() - T::one()) * (shape.clone() - T::one()) * (shape - T::two()));
        select(&valid, body, T::nan())
    }

    fn moment2(&self) -> T {
        moment2(self.mean(), self.variance())
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        T::nan()
    }

    fn kurtosis(&self) -> T {
        T::nan()
    }

    fn entropy(&self) -> T {
        T::nan()
    }

    fn median(&self) -> T {
        self.scale() * (T::two().powf(T::one() / self.shape()) - T::one())
    }

    fn mode(&self) -> T {
        T::zero()
    }
}

impl<T: LazyReal> Moments<T> for Normal<T> {
    fn mean(&self) -> T {
        self.params().0
    }

    fn variance(&self) -> T {
        let std_dev = self.params().1;
        std_dev.clone() * std_dev
    }

    fn moment2(&self) -> T {
        moment2(self.mean(), self.variance())
    }

    fn std_dev(&self) -> T {
        self.params().1
    }

    fn skewness(&self) -> T {
        T::zero()
    }

    fn kurtosis(&self) -> T {
        T::zero()
    }

    fn entropy(&self) -> T {
        kernels::normal_entropy(self.params().1)
    }

    fn median(&self) -> T {
        self.params().0
    }

    fn mode(&self) -> T {
        self.params().0
    }
}

impl<T: LazyReal> Moments<T> for Pareto<T> {
    fn mean(&self) -> T {
        let shape = self.shape();
        let valid = shape.gt(&T::one());
        let body = shape.clone() * self.scale() / (shape - T::one());
        select(&valid, body, T::nan())
    }

    fn variance(&self) -> T {
        let shape = self.shape();
        let valid = shape.gt(&T::two());
        let body = self.scale() * self.scale() * shape.clone()
            / ((shape.clone() - T::one()) * (shape.clone() - T::one()) * (shape - T::two()));
        select(&valid, body, T::nan())
    }

    fn moment2(&self) -> T {
        moment2(self.mean(), self.variance())
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        T::nan()
    }

    fn kurtosis(&self) -> T {
        T::nan()
    }

    fn entropy(&self) -> T {
        kernels::pareto_entropy(self.scale(), self.shape())
    }

    fn median(&self) -> T {
        self.scale() * T::two().powf(T::one() / self.shape())
    }

    fn mode(&self) -> T {
        self.scale()
    }
}

impl<T: LazyReal> Moments<T> for Rayleigh<T> {
    fn mean(&self) -> T {
        self.scale() * (T::pi() / T::two()).sqrt()
    }

    fn variance(&self) -> T {
        (T::scalar(4.0) - T::pi()) * self.scale() * self.scale() / T::two()
    }

    fn moment2(&self) -> T {
        moment2(self.mean(), self.variance())
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        T::nan()
    }

    fn kurtosis(&self) -> T {
        T::nan()
    }

    fn entropy(&self) -> T {
        kernels::rayleigh_entropy(self.scale())
    }

    fn median(&self) -> T {
        self.scale() * (T::two() * T::ln_2()).sqrt()
    }

    fn mode(&self) -> T {
        self.scale()
    }
}

impl<T: LazyReal> Moments<T> for StudentT<T> {
    fn mean(&self) -> T {
        let freedom = self.freedom();
        let valid = freedom.gt(&T::one());
        select(&valid, self.location(), T::nan())
    }

    fn variance(&self) -> T {
        let freedom = self.freedom();
        let valid = freedom.gt(&T::two());
        let body = self.scale() * self.scale() * freedom.clone() / (freedom - T::two());
        select(&valid, body, T::nan())
    }

    fn moment2(&self) -> T {
        moment2(self.mean(), self.variance())
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        let freedom = self.freedom();
        let valid = freedom.gt(&T::scalar(3.0));
        select(&valid, T::zero(), T::nan())
    }

    fn kurtosis(&self) -> T {
        T::nan()
    }

    fn entropy(&self) -> T {
        kernels::students_t_entropy(self.scale(), self.freedom())
    }

    fn median(&self) -> T {
        self.location()
    }

    fn mode(&self) -> T {
        self.location()
    }
}

impl<T: LazyReal> Moments<T> for Uniform<T> {
    fn mean(&self) -> T {
        (self.lo() + self.hi()) * T::half()
    }

    fn variance(&self) -> T {
        let width = self.hi() - self.lo();
        width.clone() * width / T::scalar(12.0)
    }

    fn moment2(&self) -> T {
        moment2(self.mean(), self.variance())
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        T::zero()
    }

    fn kurtosis(&self) -> T {
        T::scalar(-1.2)
    }

    fn entropy(&self) -> T {
        kernels::uniform_entropy(self.hi() - self.lo())
    }

    fn median(&self) -> T {
        self.mean()
    }

    fn mode(&self) -> T {
        T::nan()
    }
}

impl<T: LazyReal> Moments<T> for VonMises<T> {
    fn mean(&self) -> T {
        self.mu()
    }

    fn variance(&self) -> T {
        T::nan()
    }

    fn moment2(&self) -> T {
        T::nan()
    }

    fn std_dev(&self) -> T {
        T::nan()
    }

    fn skewness(&self) -> T {
        T::nan()
    }

    fn kurtosis(&self) -> T {
        T::nan()
    }

    fn entropy(&self) -> T {
        kernels::von_mises_entropy(self.kappa())
    }

    fn median(&self) -> T {
        T::nan()
    }

    fn mode(&self) -> T {
        self.mu()
    }
}

impl<T: LazyReal> Moments<T> for Wald<T> {
    fn mean(&self) -> T {
        self.mu()
    }

    fn variance(&self) -> T {
        self.mu() * self.mu() * self.mu() / self.shape()
    }

    fn moment2(&self) -> T {
        moment2(self.mean(), self.variance())
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        kernels::wald_skewness(self.mu(), self.shape())
    }

    fn kurtosis(&self) -> T {
        kernels::wald_excess_kurtosis(self.mu(), self.shape())
    }

    fn entropy(&self) -> T {
        T::nan()
    }

    fn median(&self) -> T {
        T::nan()
    }

    fn mode(&self) -> T {
        let ratio = self.mu() / self.shape();
        self.mu()
            * ((T::one() + T::scalar(9.0) * ratio.clone() * ratio / T::scalar(4.0)).sqrt()
                - T::scalar(1.5) * self.mu() / self.shape())
    }
}

impl<T: LazyReal> Moments<T> for Weibull<T> {
    fn mean(&self) -> T {
        self.scale() * (T::one() + T::one() / self.shape()).gamma()
    }

    fn variance(&self) -> T {
        let g1 = (T::one() + T::one() / self.shape()).gamma();
        let g2 = (T::one() + T::two() / self.shape()).gamma();
        self.scale() * self.scale() * (g2 - g1.clone() * g1)
    }

    fn moment2(&self) -> T {
        moment2(self.mean(), self.variance())
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        T::nan()
    }

    fn kurtosis(&self) -> T {
        T::nan()
    }

    fn entropy(&self) -> T {
        kernels::weibull_entropy(self.scale(), self.shape())
    }

    fn median(&self) -> T {
        self.scale() * T::ln_2().powf(T::one() / self.shape())
    }

    fn mode(&self) -> T {
        let shape = self.shape();
        let valid = shape.gt(&T::one());
        let body =
            self.scale() * ((shape.clone() - T::one()) / shape.clone()).powf(T::one() / shape);
        select(&valid, body, T::zero())
    }
}

impl<T: LazyReal> Moments<T> for Bernoulli<T> {
    fn mean(&self) -> T {
        self.p()
    }

    fn variance(&self) -> T {
        self.p() * (T::one() - self.p())
    }

    fn moment2(&self) -> T {
        moment2(self.mean(), self.variance())
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        kernels::bernoulli_skewness(self.p())
    }

    fn kurtosis(&self) -> T {
        T::nan()
    }

    fn entropy(&self) -> T {
        let p = self.p();
        let q = T::one() - p.clone();
        -p.clone().xlogy(p) - q.clone().xlogy(q)
    }

    fn median(&self) -> T {
        T::nan()
    }

    fn mode(&self) -> T {
        let p = self.p();
        select(&p.gt(&T::half()), T::one(), T::zero())
    }
}

impl<N: LazyNatural> Moments<N::Real> for BetaBinomial<N> {
    fn mean(&self) -> N::Real {
        let sum = self.alpha() + self.beta();
        self.trials().to_real() * self.alpha() / sum
    }

    fn variance(&self) -> N::Real {
        let n = self.trials().to_real();
        let sum = self.alpha() + self.beta();
        n.clone() * self.alpha() * self.beta() * (sum.clone() + n)
            / (sum.clone() * sum.clone() * (sum + N::Real::one()))
    }

    fn moment2(&self) -> N::Real {
        moment2(self.mean(), self.variance())
    }

    fn std_dev(&self) -> N::Real {
        self.variance().sqrt()
    }

    fn skewness(&self) -> N::Real {
        N::Real::nan()
    }

    fn kurtosis(&self) -> N::Real {
        N::Real::nan()
    }

    fn entropy(&self) -> N::Real {
        N::Real::nan()
    }

    fn median(&self) -> N::Real {
        N::Real::nan()
    }

    fn mode(&self) -> N::Real {
        let alpha = self.alpha();
        let beta = self.beta();
        let valid = alpha.gt(&N::Real::one()).and(&beta.gt(&N::Real::one()));
        let body = self.trials().to_real() * (alpha.clone() - N::Real::one())
            / (alpha + beta - N::Real::two());
        select(&valid, body, N::Real::nan())
    }
}

impl<N: LazyNatural> Moments<N::Real> for Binomial<N> {
    fn mean(&self) -> N::Real {
        self.trials().to_real() * self.p()
    }

    fn variance(&self) -> N::Real {
        self.trials().to_real() * self.p() * (N::Real::one() - self.p())
    }

    fn moment2(&self) -> N::Real {
        moment2(self.mean(), self.variance())
    }

    fn std_dev(&self) -> N::Real {
        self.variance().sqrt()
    }

    fn skewness(&self) -> N::Real {
        kernels::binomial_skewness(self.trials().to_real(), self.p())
    }

    fn kurtosis(&self) -> N::Real {
        N::Real::nan()
    }

    fn entropy(&self) -> N::Real {
        N::Real::nan()
    }

    fn median(&self) -> N::Real {
        N::Real::nan()
    }

    fn mode(&self) -> N::Real {
        ((self.trials().to_real() + N::Real::one()) * self.p()).floor()
    }
}

impl<I: LazyInteger> Moments<I::Real> for DiscreteUniform<I> {
    fn mean(&self) -> I::Real {
        (self.lo().to_real() + self.hi().to_real() - I::Real::one()) * I::Real::half()
    }

    fn variance(&self) -> I::Real {
        let count = (self.hi() - self.lo()).to_real();
        (count.clone() * count - I::Real::one()) / I::Real::scalar(12.0)
    }

    fn moment2(&self) -> I::Real {
        moment2(self.mean(), self.variance())
    }

    fn std_dev(&self) -> I::Real {
        self.variance().sqrt()
    }

    fn skewness(&self) -> I::Real {
        I::Real::zero()
    }

    fn kurtosis(&self) -> I::Real {
        I::Real::nan()
    }

    fn entropy(&self) -> I::Real {
        kernels::discrete_uniform_entropy((self.hi() - self.lo()).to_real())
    }

    fn median(&self) -> I::Real {
        self.mean()
    }

    fn mode(&self) -> I::Real {
        I::Real::nan()
    }
}

impl<T: LazyReal> Moments<T> for Geometric<T> {
    fn mean(&self) -> T {
        self.p().recip()
    }

    fn variance(&self) -> T {
        (T::one() - self.p()) / (self.p() * self.p())
    }

    fn moment2(&self) -> T {
        moment2(self.mean(), self.variance())
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        kernels::geometric_skewness(self.p())
    }

    fn kurtosis(&self) -> T {
        T::nan()
    }

    fn entropy(&self) -> T {
        let p = self.p();
        let q = T::one() - p.clone();
        let body = (-q.clone() * q.ln() - p.clone() * p.clone().ln()) / p;
        select(&self.p().ge(&T::one()), T::zero(), body)
    }

    fn median(&self) -> T {
        T::nan()
    }

    fn mode(&self) -> T {
        T::one()
    }
}

impl<N: LazyNatural> Moments<N::Real> for Hypergeometric<N> {
    fn mean(&self) -> N::Real {
        self.draws().to_real() * self.successes().to_real() / self.population().to_real()
    }

    fn variance(&self) -> N::Real {
        let population = self.population().to_real();
        let successes = self.successes().to_real();
        let draws = self.draws().to_real();
        let body = draws.clone()
            * (successes.clone() / population.clone())
            * ((population.clone() - successes) / population.clone())
            * ((population.clone() - draws) / (population.clone() - N::Real::one()));
        select(&population.le(&N::Real::one()), N::Real::zero(), body)
    }

    fn moment2(&self) -> N::Real {
        moment2(self.mean(), self.variance())
    }

    fn std_dev(&self) -> N::Real {
        self.variance().sqrt()
    }

    fn skewness(&self) -> N::Real {
        N::Real::nan()
    }

    fn kurtosis(&self) -> N::Real {
        N::Real::nan()
    }

    fn entropy(&self) -> N::Real {
        N::Real::nan()
    }

    fn median(&self) -> N::Real {
        N::Real::nan()
    }

    fn mode(&self) -> N::Real {
        N::Real::nan()
    }
}

impl<T: LazyReal> Moments<T> for NegativeBinomial<T> {
    fn mean(&self) -> T {
        self.successes() * (T::one() - self.p()) / self.p()
    }

    fn variance(&self) -> T {
        self.successes() * (T::one() - self.p()) / (self.p() * self.p())
    }

    fn moment2(&self) -> T {
        moment2(self.mean(), self.variance())
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        kernels::negative_binomial_skewness(self.successes(), self.p())
    }

    fn kurtosis(&self) -> T {
        T::nan()
    }

    fn entropy(&self) -> T {
        T::nan()
    }

    fn median(&self) -> T {
        T::nan()
    }

    fn mode(&self) -> T {
        let successes = self.successes();
        let body = ((successes.clone() - T::one()) * (T::one() - self.p()) / self.p()).floor();
        select(&successes.gt(&T::one()), body, T::zero())
    }
}

impl<T: LazyReal> Moments<T> for Poisson<T> {
    fn mean(&self) -> T {
        self.lambda()
    }

    fn variance(&self) -> T {
        self.lambda()
    }

    fn moment2(&self) -> T {
        moment2(self.mean(), self.variance())
    }

    fn std_dev(&self) -> T {
        self.variance().sqrt()
    }

    fn skewness(&self) -> T {
        kernels::poisson_skewness(self.lambda())
    }

    fn kurtosis(&self) -> T {
        kernels::poisson_excess_kurtosis(self.lambda())
    }

    fn entropy(&self) -> T {
        T::nan()
    }

    fn median(&self) -> T {
        T::nan()
    }

    fn mode(&self) -> T {
        self.lambda().floor()
    }
}
