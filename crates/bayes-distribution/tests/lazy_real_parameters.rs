use bayes_core::assert_close;
use bayes_distribution::{
    Beta, BetaBinomial, Binomial, Cauchy, ChiSquared, Continuous, ContinuousDensityKernel,
    Discrete, DiscreteMassKernel, DiscreteUniform, Exponential, FisherSnedecor, Frechet, Gamma,
    Gumbel, HalfNormal, Hypergeometric, IntegerDiscrete, IntegerMassKernel, InverseGamma, Laplace,
    LogNormal, Logistic, Lomax, Moments, Normal, Pareto, Poisson, Rayleigh, StudentT, Uniform,
    VonMises, Wald, Weibull,
};

#[test]
fn moments_accept_lazy_real_parameters_and_explicit_nan() {
    let normal = Normal::new(1.0_f32, 2.0_f32);
    assert_close!(normal.mean() as f64, 1.0, 1e-6);
    assert_close!(normal.variance() as f64, 4.0, 1e-6);
    assert_close!(normal.std_dev() as f64, 2.0, 1e-6);

    let gamma = Gamma::new(2.5_f32, 1.25_f32);
    assert_close!(gamma.mean() as f64, 2.0, 1e-6);
    assert_close!(gamma.variance() as f64, 1.6, 1e-5);

    let cauchy = Cauchy::new(0.0_f32, 1.0_f32);
    assert!(cauchy.mean().is_nan());
    assert!(cauchy.variance().is_nan());

    let student = StudentT::new(1.0_f32);
    assert!(student.mean().is_nan());

    let pareto = Pareto::new(1.0_f32, 1.5_f32);
    assert!(pareto.variance().is_nan());

    let binomial = Binomial::new(10_u32, 0.3_f32);
    assert_close!(binomial.mean() as f64, 3.0, 1e-6);
    assert_close!(binomial.variance() as f64, 2.1, 1e-6);
}

#[test]
fn bounded_distributions_accept_lazy_real_parameters() {
    let beta = Beta::new(2.0_f32, 5.0_f32);
    let beta_reference = Beta::new(2.0, 5.0);
    assert_close!(beta.pdf(0.4_f32) as f64, beta_reference.pdf(0.4), 1e-5);
    assert_close!(
        beta.ln_pdf_kernel(0.4_f32) as f64,
        beta_reference.ln_pdf_kernel(0.4),
        1e-5,
    );

    let uniform = Uniform::new(-2.0_f32, 3.0_f32);
    let uniform_reference = Uniform::new(-2.0, 3.0);
    assert_close!(
        uniform.ln_pdf(0.0_f32) as f64,
        uniform_reference.ln_pdf(0.0),
        1e-6,
    );
}

#[test]
fn circular_distributions_accept_lazy_real_parameters() {
    let von_mises = VonMises::new(0.3_f32, 2.0_f32);
    let reference = VonMises::new(0.3, 2.0);
    assert_close!(
        von_mises.ln_pdf(0.1_f32) as f64,
        reference.ln_pdf(0.1),
        1e-5,
    );
}

#[test]
fn continuous_distributions_accept_lazy_real_parameters() {
    let normal = Normal::new(1.0_f32, 2.0_f32);
    let normal_reference = Normal::new(1.0, 2.0);
    assert_close!(
        normal.ln_pdf(0.25_f32) as f64,
        normal_reference.ln_pdf(0.25),
        1e-5,
    );

    let exponential = Exponential::new(1.7_f32);
    let exponential_reference = Exponential::new(1.7);
    assert_close!(
        exponential.ln_pdf(0.4_f32) as f64,
        exponential_reference.ln_pdf(0.4),
        1e-5,
    );
    assert_eq!(exponential.ln_pdf(-0.4_f32), f32::NEG_INFINITY);

    let gamma = Gamma::new(2.5_f32, 1.2_f32);
    let gamma_reference = Gamma::new(2.5, 1.2);
    assert_close!(
        gamma.ln_pdf(0.8_f32) as f64,
        gamma_reference.ln_pdf(0.8),
        1e-4,
    );

    let lognormal = LogNormal::new(0.2_f32, 0.7_f32);
    let lognormal_reference = LogNormal::new(0.2, 0.7);
    assert_close!(
        lognormal.ln_pdf(1.4_f32) as f64,
        lognormal_reference.ln_pdf(1.4),
        1e-5,
    );

    let logistic = Logistic::new(-0.5_f32, 1.3_f32);
    let logistic_reference = Logistic::new(-0.5, 1.3);
    assert_close!(
        logistic.ln_pdf(0.2_f32) as f64,
        logistic_reference.ln_pdf(0.2),
        1e-5,
    );

    let laplace = Laplace::new(0.5_f32, 1.7_f32);
    let laplace_reference = Laplace::new(0.5, 1.7);
    assert_close!(
        laplace.ln_pdf(-0.2_f32) as f64,
        laplace_reference.ln_pdf(-0.2),
        1e-5,
    );

    let cauchy = Cauchy::new(0.5_f32, 1.7_f32);
    let cauchy_reference = Cauchy::new(0.5, 1.7);
    assert_close!(
        cauchy.ln_pdf(-0.2_f32) as f64,
        cauchy_reference.ln_pdf(-0.2),
        1e-5,
    );

    let student = StudentT::new_location_scale(6.0_f32, -0.5_f32, 1.3_f32);
    let student_reference = StudentT::new_location_scale(6.0, -0.5, 1.3);
    assert_close!(
        student.ln_pdf(0.4_f32) as f64,
        student_reference.ln_pdf(0.4),
        1e-4,
    );

    let pareto = Pareto::new(2.0_f32, 3.0_f32);
    let pareto_reference = Pareto::new(2.0, 3.0);
    assert_close!(
        pareto.ln_pdf(4.0_f32) as f64,
        pareto_reference.ln_pdf(4.0),
        1e-5,
    );
    assert_eq!(pareto.ln_pdf(1.0_f32), f32::NEG_INFINITY);

    let rayleigh = Rayleigh::new(1.4_f32);
    let rayleigh_reference = Rayleigh::new(1.4);
    assert_close!(
        rayleigh.ln_pdf(0.9_f32) as f64,
        rayleigh_reference.ln_pdf(0.9),
        1e-5,
    );

    let weibull = Weibull::new(1.4_f32, 2.2_f32);
    let weibull_reference = Weibull::new(1.4, 2.2);
    assert_close!(
        weibull.ln_pdf(0.9_f32) as f64,
        weibull_reference.ln_pdf(0.9),
        1e-5,
    );

    let frechet = Frechet::new(3.0_f32, 2.0_f32);
    let frechet_reference = Frechet::new(3.0, 2.0);
    assert_close!(
        frechet.ln_pdf(4.0_f32) as f64,
        frechet_reference.ln_pdf(4.0),
        1e-5,
    );

    let half_normal = HalfNormal::new(1.4_f32);
    let half_normal_reference = HalfNormal::new(1.4);
    assert_close!(
        half_normal.ln_pdf(0.9_f32) as f64,
        half_normal_reference.ln_pdf(0.9),
        1e-5,
    );

    let inverse_gamma = InverseGamma::new(3.0_f32, 2.0_f32);
    let inverse_gamma_reference = InverseGamma::new(3.0, 2.0);
    assert_close!(
        inverse_gamma.ln_pdf(1.1_f32) as f64,
        inverse_gamma_reference.ln_pdf(1.1),
        1e-4,
    );

    let lomax = Lomax::new(3.0_f32, 2.0_f32);
    let lomax_reference = Lomax::new(3.0, 2.0);
    assert_close!(
        lomax.ln_pdf(1.1_f32) as f64,
        lomax_reference.ln_pdf(1.1),
        1e-5,
    );

    let chi_squared = ChiSquared::new(5.0_f32);
    let chi_squared_reference = ChiSquared::new(5.0);
    assert_close!(
        chi_squared.ln_pdf(1.2_f32) as f64,
        chi_squared_reference.ln_pdf(1.2),
        1e-4,
    );

    let fisher = FisherSnedecor::new(5.0_f32, 10.0_f32);
    let fisher_reference = FisherSnedecor::new(5.0, 10.0);
    assert_close!(
        fisher.ln_pdf(1.2_f32) as f64,
        fisher_reference.ln_pdf(1.2),
        1e-4,
    );

    let gumbel = Gumbel::new(1.0_f32, 2.0_f32);
    let gumbel_reference = Gumbel::new(1.0, 2.0);
    assert_close!(
        gumbel.ln_pdf(0.4_f32) as f64,
        gumbel_reference.ln_pdf(0.4),
        1e-5,
    );

    let wald = Wald::new(1.0_f32, 2.0_f32);
    let wald_reference = Wald::new(1.0, 2.0);
    assert_close!(
        wald.ln_pdf(0.9_f32) as f64,
        wald_reference.ln_pdf(0.9),
        1e-4,
    );
}

#[test]
fn natural_count_distributions_accept_lazy_natural_parameters() {
    let binomial = Binomial::new(10_u32, 0.3_f32);
    let binomial_reference = Binomial::new(10_u64, 0.3);
    assert_close!(
        binomial.pmf(3_u32) as f64,
        binomial_reference.pmf(3_u64),
        1e-5,
    );
    assert_close!(
        binomial.ln_pmf_kernel(3_u32) as f64,
        binomial_reference.ln_pmf_kernel(3_u64),
        1e-5,
    );

    let beta_binomial = BetaBinomial::from_beta(10_u32, Beta::new(2.0_f32, 5.0_f32));
    let beta_binomial_reference = BetaBinomial::new(10_u64, 2.0, 5.0);
    assert_close!(
        beta_binomial.pmf(3_u32) as f64,
        beta_binomial_reference.pmf(3_u64),
        1e-5,
    );

    let hypergeometric = Hypergeometric::new(30_u32, 12_u32, 5_u32);
    let hypergeometric_reference = Hypergeometric::new(30_u64, 12_u64, 5_u64);
    assert_close!(
        hypergeometric.pmf(2_u32) as f64,
        hypergeometric_reference.pmf(2_u64),
        1e-5,
    );

    let poisson = Poisson::new(3.5_f32);
    let poisson_reference = Poisson::new(3.5);
    assert_close!(
        poisson.pmf(4_u32) as f64,
        poisson_reference.pmf(4_u64),
        1e-5,
    );
}

#[test]
fn integer_distributions_accept_lazy_integer_parameters() {
    let discrete_uniform = DiscreteUniform::new(-2_i128, 5_i128);
    let reference = DiscreteUniform::new(-2_i64, 5_i64);
    assert_close!(discrete_uniform.pmf(3_i128), reference.pmf(3_i64), 1e-6);
    assert_close!(
        discrete_uniform.ln_pmf_kernel(3_i128),
        reference.ln_pmf_kernel(3_i64),
        1e-6,
    );
}
