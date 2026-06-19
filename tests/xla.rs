//! bayes on XLA.
//!
//! bayes' lazy closed forms are written once over euclid's contracts. This test
//! runs them on the compiled XLA backend (`maxwell`) and checks they match the
//! eager `f64` / `ramanujan` reference — the "write-once-math, many-backends"
//! payoff: the *same* density / CDF / multivariate-Gaussian source on both
//! backends. (Dev-only; bayes itself depends only on euclid.)

use bayes::lazy::{gamma_cdf_on_support, normal_cdf, normal_log_pdf};
use bayes::lazy_matrix::multivariate_normal_log_pdf;
use maxwell::{Session, Tensor};
use ramanujan::{DenseMatrix, DenseVector};

fn close_tol(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() < tol, "mismatch: {a} vs {b} (tol {tol})");
}
fn close(a: f64, b: f64) {
    close_tol(a, b, 1e-6);
}

/// Evaluate a tensor expression built inside a session scope.
fn xla(s: &Session, build: impl FnOnce() -> Tensor) -> f64 {
    let t = s.scope("xla", build);
    s.run(&t).unwrap()[0]
}

#[test]
fn scalar_closed_forms_match_f64() {
    let s = Session::cpu().unwrap();

    // Normal log-pdf — pure arithmetic + ln.
    close(
        normal_log_pdf(1.2_f64, 0.3, 0.7),
        xla(&s, || normal_log_pdf(Tensor::scalar(1.2), Tensor::scalar(0.3), Tensor::scalar(0.7))),
    );

    // Normal CDF — exercises erfc on the tensor backend.
    let phi = normal_cdf(1.0_f64, 0.0, 1.0);
    close(phi, xla(&s, || normal_cdf(Tensor::scalar(1.0), Tensor::scalar(0.0), Tensor::scalar(1.0))));
    close(phi, 0.841_344_746);

    // Gamma CDF — exercises the regularized lower incomplete gamma (Igamma).
    close_tol(
        gamma_cdf_on_support(2.0_f64, 3.0, 1.0),
        xla(&s, || gamma_cdf_on_support(Tensor::scalar(2.0), Tensor::scalar(3.0), Tensor::scalar(1.0))),
        1e-5,
    );
}

#[test]
fn multivariate_normal_matches_eager() {
    let s = Session::cpu().unwrap();

    // Σ = [[4,2],[2,3]], x = [1,2], µ = [0,0].
    // Eager reference: the same source on ramanujan's f64 DenseMatrix.
    let eager = multivariate_normal_log_pdf::<DenseMatrix>(
        DenseVector::new(vec![1.0, 2.0]),
        DenseVector::new(vec![0.0, 0.0]),
        DenseMatrix::from_rows(&[vec![4.0, 2.0], vec![2.0, 3.0]]),
    );

    // Compiled on XLA: the identical source on maxwell::Tensor.
    let on_xla = xla(&s, || {
        multivariate_normal_log_pdf::<Tensor>(
            Tensor::from_host(&[1.0, 2.0], &[2]),
            Tensor::from_host(&[0.0, 0.0], &[2]),
            Tensor::from_host(&[4.0, 2.0, 2.0, 3.0], &[2, 2]),
        )
    });

    close(eager, on_xla);
    close(eager, -3.565_097_837); // closed-form ln N(x | µ, Σ)
}
