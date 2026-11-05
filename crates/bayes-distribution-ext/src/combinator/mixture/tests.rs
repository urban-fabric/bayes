use super::*;
use crate::{Normal, Poisson};
use bayes_core_ext::assert_close;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn density_is_the_weighted_sum() {
    let m = Mixture::new(
        vec![0.3, 0.7],
        vec![
            Normal::new(-2.0, 1.0).unwrap(),
            Normal::new(3.0, 2.0).unwrap(),
        ],
    )
    .unwrap();
    let x = 0.5;
    let expect =
        0.3 * Normal::new(-2.0, 1.0).unwrap().pdf(x) + 0.7 * Normal::new(3.0, 2.0).unwrap().pdf(x);
    assert_close!(m.pdf(x), expect, 1e-12);
}

#[test]
fn continuous_responsibilities_are_stable_posteriors_over_components() {
    let m = Mixture::new(
        vec![0.25, 0.75],
        vec![
            Normal::new(0.0, 1.0).unwrap(),
            Normal::new(10.0, 1.0).unwrap(),
        ],
    )
    .unwrap();

    let x = 0.25;
    let resp = m.pdf_responsibilities(x);
    assert_eq!(resp.len(), 2);
    assert_close!(resp.iter().sum(), 1.0, 1e-12);
    assert_close!(resp[0], 0.25 * m.components()[0].pdf(x) / m.pdf(x), 1e-12);
    assert_close!(resp[1], 0.75 * m.components()[1].pdf(x) / m.pdf(x), 1e-12);
    assert!(resp[0] > 0.999);

    let log_resp = m.ln_pdf_responsibilities(-80.0);
    assert_eq!(log_resp.len(), 2);
    assert!(log_resp[0].is_finite());
    assert!(log_resp[1] < -700.0);
    assert_close!(log_resp[0].exp(), 1.0, 1e-12);
}

#[test]
fn responsibilities_handle_zero_weights_and_discrete_components() {
    let m = Mixture::new(
        vec![0.0, 1.0],
        vec![
            Normal::new(0.0, 1.0).unwrap(),
            Normal::new(5.0, 1.0).unwrap(),
        ],
    )
    .unwrap();
    assert_eq!(m.pdf_responsibilities(0.0), vec![0.0, 1.0]);
    assert_eq!(m.ln_pdf_responsibilities(0.0), vec![f64::NEG_INFINITY, 0.0]);

    let counts = Mixture::new(
        vec![0.4, 0.6],
        vec![Poisson::new(2.0).unwrap(), Poisson::new(8.0).unwrap()],
    )
    .unwrap();
    let k = 7;
    let resp = counts.pmf_responsibilities(k);
    assert_eq!(resp.len(), 2);
    assert_close!(resp.iter().sum(), 1.0, 1e-12);
    assert_close!(
        resp[0],
        0.4 * counts.components()[0].pmf(k) / counts.pmf(k),
        1e-12,
    );
    assert_close!(
        resp[1],
        0.6 * counts.components()[1].pmf(k) / counts.pmf(k),
        1e-12,
    );
    assert!(resp[1] > resp[0]);
}

#[test]
fn moments_follow_the_mixture_laws() {
    let m = Mixture::new(
        vec![0.5, 0.5],
        vec![
            Normal::new(0.0, 1.0).unwrap(),
            Normal::new(4.0, 1.0).unwrap(),
        ],
    )
    .unwrap();
    assert_close!(m.mean(), 2.0, 1e-12);
    assert_close!(m.variance(), 5.0, 1e-12);
    // Law of total variance: within = ½·1 + ½·1 = 1, between = ½·2² + ½·2² = 4.
    let (within, between) = m.variance_decomposition();
    assert_close!(within, 1.0, 1e-12);
    assert_close!(between, 4.0, 1e-12);
    // within + between == variance (law_of_total_variance).
    assert_close!(within + between, m.variance(), 1e-12);
}

#[test]
fn sampled_mean_matches() {
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    let m = Mixture::new(
        vec![0.5, 0.5],
        vec![
            Normal::new(0.0, 1.0).unwrap(),
            Normal::new(4.0, 1.0).unwrap(),
        ],
    )
    .unwrap();
    let n = 60_000;
    let s: f64 = (0..n).map(|_| m.sample(&mut rng)).sum();
    assert_close!(s / n as f64, 2.0, 0.05);
}
