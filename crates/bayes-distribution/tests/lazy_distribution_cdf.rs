#![allow(clippy::panic)]
use bayes_core::assert_close_scaled;

use bayes_distribution::{
    Bernoulli, Beta, Binomial, Cauchy, ChiSquared, ContinuousCdf, DiscreteCdf, DiscreteUniform,
    Exponential, FisherSnedecor, Frechet, Gamma, Geometric, Gumbel, HalfNormal, IntegerDiscreteCdf,
    InverseGamma, Laplace, LogNormal, Logistic, Lomax, NegativeBinomial, Normal, Pareto, Poisson,
    Rayleigh, StudentT, Support, Uniform, VonMises, Wald, Weibull,
};

#[test]
fn lazy_continuous_cdfs_and_quantiles_are_public() {
    let p = 0.37;

    let normal = Normal::new(1.0, 2.0);
    assert_close_scaled!(normal.cdf(normal.inverse_cdf(p)), p, 1e-10);
    assert_close_scaled!(normal.ln_cdf(-2.0).exp(), normal.cdf(-2.0), 1e-12);

    let exponential = Exponential::new(1.7);
    assert_close_scaled!(exponential.cdf(exponential.inverse_cdf(p)), p, 1e-12);
    assert_close_scaled!(exponential.ln_sf(2.0), -3.4, 1e-12);

    let logistic = Logistic::new(-0.5, 1.3);
    assert_close_scaled!(logistic.cdf(logistic.inverse_cdf(p)), p, 1e-12);
    assert_close_scaled!(logistic.ln_sf(0.2).exp(), logistic.sf(0.2), 1e-12);

    let direct = [
        Cauchy::new(0.5, 1.7).cdf(Cauchy::new(0.5, 1.7).inverse_cdf(p)),
        Laplace::new(0.5, 1.7).cdf(Laplace::new(0.5, 1.7).inverse_cdf(p)),
        Uniform::new(-2.0, 3.0).cdf(Uniform::new(-2.0, 3.0).inverse_cdf(p)),
        Gumbel::new(1.0, 2.0).cdf(Gumbel::new(1.0, 2.0).inverse_cdf(p)),
        Weibull::new(1.4, 2.2).cdf(Weibull::new(1.4, 2.2).inverse_cdf(p)),
        Frechet::new(3.0, 2.0).cdf(Frechet::new(3.0, 2.0).inverse_cdf(p)),
        Pareto::new(2.0, 3.0).cdf(Pareto::new(2.0, 3.0).inverse_cdf(p)),
        Lomax::new(3.0, 2.0).cdf(Lomax::new(3.0, 2.0).inverse_cdf(p)),
        Rayleigh::new(1.4).cdf(Rayleigh::new(1.4).inverse_cdf(p)),
        HalfNormal::new(1.4).cdf(HalfNormal::new(1.4).inverse_cdf(p)),
        LogNormal::new(0.2, 0.7).cdf(LogNormal::new(0.2, 0.7).inverse_cdf(p)),
    ];
    for observed in direct {
        assert_close_scaled!(observed, p, 1e-10);
    }

    let iterative = [
        Gamma::new(2.5, 1.2).cdf(Gamma::new(2.5, 1.2).inverse_cdf(p)),
        ChiSquared::new(5.0).cdf(ChiSquared::new(5.0).inverse_cdf(p)),
        Beta::new(2.0, 5.0).cdf(Beta::new(2.0, 5.0).inverse_cdf(p)),
        StudentT::new_location_scale(6.0, -0.5, 1.3)
            .cdf(StudentT::new_location_scale(6.0, -0.5, 1.3).inverse_cdf(p)),
        FisherSnedecor::new(5.0, 10.0).cdf(FisherSnedecor::new(5.0, 10.0).inverse_cdf(p)),
        InverseGamma::new(3.0, 2.0).cdf(InverseGamma::new(3.0, 2.0).inverse_cdf(p)),
        Wald::new(1.0, 3.0).cdf(Wald::new(1.0, 3.0).inverse_cdf(p)),
    ];
    for observed in iterative {
        assert_close_scaled!(observed, p, 1e-8);
    }

    assert_close_scaled!(VonMises::new(0.0, 2.0).min(), -std::f64::consts::PI, 1e-12);
    assert_close_scaled!(VonMises::new(0.0, 2.0).max(), std::f64::consts::PI, 1e-12);
}

#[test]
fn lazy_discrete_cdfs_are_public() {
    let bernoulli = Bernoulli::new(0.25);
    assert_close_scaled!(bernoulli.cdf(0_u64), 0.75, 1e-12);
    assert_eq!(
        <Bernoulli<f64> as DiscreteCdf<u64>>::inverse_cdf(&bernoulli, 0.8),
        1
    );

    let binomial = Binomial::new(12_u64, 0.3);
    assert_close_scaled!(
        binomial.cdf(4),
        bayes_kernels::binomial_cdf_on_support(4_i64, 12_i64, 0.3),
        1e-12,
    );
    assert!(binomial.inverse_cdf(0.5) <= binomial.max());

    let geometric = Geometric::new(0.25);
    assert_close_scaled!(geometric.cdf(3_u64), 1.0 - 0.75_f64.powi(3), 1e-12);

    let negative_binomial = NegativeBinomial::new(5.0, 0.4);
    assert_close_scaled!(
        negative_binomial.cdf(3_u64),
        bayes_kernels::negative_binomial_cdf(3_i64, 5.0, 0.4),
        1e-12,
    );

    let poisson = Poisson::new(3.5);
    assert_close_scaled!(
        poisson.cdf(6_u64),
        bayes_kernels::poisson_cdf(6_i64, 3.5),
        1e-12,
    );

    let discrete_uniform = DiscreteUniform::new(-2_i64, 5_i64);
    assert_close_scaled!(discrete_uniform.cdf(1), 4.0 / 7.0, 1e-12);
    assert_eq!(discrete_uniform.inverse_cdf(0.5), 1);
}
