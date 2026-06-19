//! bayes densities, differentiated and optimized on XLA via maxwell's autograd.
//!
//! `bayes::lazy::normal_log_pdf` is generic over euclid's `LazyReal`. Because
//! `maxwell::Var` now implements that contract, the *same* function is called with
//! `T = Var` and **differentiated** by maxwell — here, maximum-likelihood fit of a
//! Gaussian. This is the bayes↔maxwell↔euclid unification: write-once densities,
//! run forward on `Tensor` (see `xla.rs`) *and* optimized on `Var`. (Dev-only;
//! bayes still depends only on euclid.)

use bayes::lazy::normal_log_pdf;
use maxwell::{Param, Session, Sgd, Tape, Tensor, Var};

const DATA: [f64; 8] = [1.0, 2.0, 3.0, 2.5, 1.5, 2.2, 1.8, 2.7];

#[test]
fn gaussian_mle_through_bayes_density() {
    let s = Session::cpu().unwrap();
    // Parameters: µ and log σ (so σ stays positive).
    let mut params = vec![Param::new(vec![0.0], &[]), Param::new(vec![0.0], &[])];
    let sgd = Sgd::new(0.05);

    // Negative log-likelihood = −Σ_i bayes::normal_log_pdf(x_i, µ, σ), built over
    // maxwell::Var so the whole thing is differentiated by maxwell on XLA.
    let build = |tape: &Tape, p: &[Var]| -> Var {
        let mu = &p[0];
        let sigma = p[1].exp();
        let mut nll = tape.constant(0.0);
        for &xi in DATA.iter() {
            let x = tape.var(Tensor::from_host(&[xi], &[]));
            let log_p = normal_log_pdf(x, mu.clone(), sigma.clone());
            nll = nll.sub(&log_p);
        }
        nll
    };

    let compiled = sgd.compile(&s, &params, build);
    for _ in 0..600 {
        compiled.step(&mut params);
    }

    let mu = params[0].value[0];
    let sigma = params[1].value[0].exp();
    // The Gaussian MLE is exactly the sample mean and (population) std.
    let n = DATA.len() as f64;
    let mean = DATA.iter().sum::<f64>() / n;
    let var = DATA.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    assert!((mu - mean).abs() < 0.02, "µ̂ = {mu} vs sample mean {mean}");
    assert!((sigma - var.sqrt()).abs() < 0.02, "σ̂ = {sigma} vs sample std {}", var.sqrt());
}
