use crate::{
    Beta, Binomial, Continuous, ContinuousDensityKernel, Discrete, DiscreteMassKernel, Gamma,
    Hypergeometric, NegativeBinomial, Normal, Poisson, StudentT,
};
use bayes_core_ext::assert_close;

/// A lazy density kernel, the point to evaluate it at, and the eager value it
/// must match.
type DensityCase = (Box<dyn Fn(f64) -> f64>, f64, f64);

#[test]
fn continuous_density_kernels_match_eager_on_support() {
    let cases: &[DensityCase] = &[
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
        assert_close!(kernel(x), eager, 1e-12);
    }
}

#[test]
fn discrete_mass_kernels_match_eager_on_support() {
    let b = Binomial::new(10, 0.3).unwrap();
    assert_close!(b.ln_pmf_kernel(3_u64), b.ln_pmf(3_u64), 1e-12);

    let p = Poisson::new(4.5).unwrap();
    assert_close!(p.ln_pmf_kernel(6_u64), p.ln_pmf(6_u64), 1e-12);

    let nb = NegativeBinomial::new(2.5, 0.4).unwrap();
    assert_close!(nb.ln_pmf_kernel(4_u64), nb.ln_pmf(4_u64), 1e-12);

    let h = Hypergeometric::new(30, 12, 7).unwrap();
    assert_close!(h.ln_pmf_kernel(3_u64), h.pmf(3_u64).ln(), 1e-12);
}
