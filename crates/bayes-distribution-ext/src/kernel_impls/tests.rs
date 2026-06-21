use crate::{
    Beta, Binomial, Continuous, ContinuousDensityKernel, Discrete, DiscreteMassKernel, Gamma,
    Hypergeometric, NegativeBinomial, Normal, Poisson, StudentT,
};

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
}

#[test]
fn continuous_density_kernels_match_eager_on_support() {
    let cases: &[(Box<dyn Fn(f64) -> f64>, f64, f64)] = &[
        (
            Box::new(|x| Normal::new(1.0, 2.0).unwrap().ln_pdf_kernel(x)),
            0.25,
            Normal::new(1.0, 2.0).unwrap().ln_pdf(0.25),
        ),
        (
            Box::new(|x| Gamma::new(2.5, 1.2).unwrap().ln_pdf_kernel(x)),
            1.5,
            Gamma::new(2.5, 1.2).unwrap().ln_pdf(1.5),
        ),
        (
            Box::new(|x| Beta::new(2.0, 5.0).unwrap().ln_pdf_kernel(x)),
            0.4,
            Beta::new(2.0, 5.0).unwrap().ln_pdf(0.4),
        ),
        (
            Box::new(|x| {
                StudentT::new_location_scale(7.0, 0.5, 1.3)
                    .unwrap()
                    .ln_pdf_kernel(x)
            }),
            0.9,
            StudentT::new_location_scale(7.0, 0.5, 1.3)
                .unwrap()
                .ln_pdf(0.9),
        ),
    ];
    for &(ref kernel, x, eager) in cases {
        close(kernel(x), eager, 1e-12);
    }
}

#[test]
fn discrete_mass_kernels_match_eager_on_support() {
    let b = Binomial::new(10, 0.3).unwrap();
    close(b.ln_pmf_kernel(3_u64), b.ln_pmf(3_u64), 1e-12);

    let p = Poisson::new(4.5).unwrap();
    close(p.ln_pmf_kernel(6_u64), p.ln_pmf(6_u64), 1e-12);

    let nb = NegativeBinomial::new(2.5, 0.4).unwrap();
    close(nb.ln_pmf_kernel(4_u64), nb.ln_pmf(4_u64), 1e-12);

    let h = Hypergeometric::new(30, 12, 7).unwrap();
    close(h.ln_pmf_kernel(3_u64), h.pmf(3_u64).ln(), 1e-12);
}
