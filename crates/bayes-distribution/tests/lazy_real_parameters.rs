use bayes_distribution::{
    Beta, BetaBinomial, Binomial, Continuous, ContinuousDensityKernel, Discrete,
    DiscreteMassKernel, DiscreteUniform, Hypergeometric, IntegerDiscrete, IntegerMassKernel,
    Poisson, Uniform, VonMises,
};

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
}

#[test]
fn bounded_distributions_accept_lazy_real_parameters() {
    let beta = Beta::from_lazy(2.0_f32, 5.0_f32);
    let beta_reference = Beta::new(2.0, 5.0).unwrap();
    close(beta.pdf(0.4_f32) as f64, beta_reference.pdf(0.4), 1e-5);
    close(
        beta.ln_pdf_kernel(0.4_f32) as f64,
        beta_reference.ln_pdf_kernel(0.4),
        1e-5,
    );

    let uniform = Uniform::from_lazy(-2.0_f32, 3.0_f32);
    let uniform_reference = Uniform::new(-2.0, 3.0).unwrap();
    close(
        uniform.ln_pdf(0.0_f32) as f64,
        uniform_reference.ln_pdf(0.0),
        1e-6,
    );
}

#[test]
fn circular_distributions_accept_lazy_real_parameters() {
    let von_mises = VonMises::from_lazy(0.3_f32, 2.0_f32);
    let reference = VonMises::new(0.3, 2.0).unwrap();
    close(
        von_mises.ln_pdf(0.1_f32) as f64,
        reference.ln_pdf(0.1),
        1e-5,
    );
}

#[test]
fn natural_count_distributions_accept_lazy_natural_parameters() {
    let binomial = Binomial::from_lazy(10_u32, 0.3_f32);
    let binomial_reference = Binomial::new(10, 0.3).unwrap();
    close(
        binomial.pmf(3_u32) as f64,
        binomial_reference.pmf(3_u64),
        1e-5,
    );
    close(
        binomial.ln_pmf_kernel(3_u32) as f64,
        binomial_reference.ln_pmf_kernel(3_u64),
        1e-5,
    );

    let beta_binomial = BetaBinomial::from_beta(10_u32, Beta::from_lazy(2.0_f32, 5.0_f32));
    let beta_binomial_reference = BetaBinomial::new(10, 2.0, 5.0).unwrap();
    close(
        beta_binomial.pmf(3_u32) as f64,
        beta_binomial_reference.pmf(3_u64),
        1e-5,
    );

    let hypergeometric = Hypergeometric::from_lazy(30_u32, 12_u32, 5_u32);
    let hypergeometric_reference = Hypergeometric::new(30, 12, 5).unwrap();
    close(
        hypergeometric.pmf(2_u32) as f64,
        hypergeometric_reference.pmf(2_u64),
        1e-5,
    );

    let poisson = Poisson::from_lazy(3.5_f32);
    let poisson_reference = Poisson::new(3.5).unwrap();
    close(
        poisson.pmf(4_u32) as f64,
        poisson_reference.pmf(4_u64),
        1e-5,
    );
}

#[test]
fn integer_distributions_accept_lazy_integer_parameters() {
    let discrete_uniform = DiscreteUniform::from_lazy(-2_i128, 5_i128);
    let reference = DiscreteUniform::new(-2, 5).unwrap();
    close(discrete_uniform.pmf(3_i128), reference.pmf(3_i64), 1e-6);
    close(
        discrete_uniform.ln_pmf_kernel(3_i128),
        reference.ln_pmf_kernel(3_i64),
        1e-6,
    );
}
