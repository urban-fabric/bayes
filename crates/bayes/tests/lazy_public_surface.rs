use bayes::lazy::{
    Bernoulli, Continuous, HellingerDistance, Kl, KlDivergence, Moments, Normal, SampleWithKey,
    Uniform, brier_score_real, cross_entropy_real, f_divergence_real, jensen_shannon_real,
    shannon_entropy_real,
};
use euclid::{RandomKey, RandomSampler};

#[derive(Clone, Copy, Debug)]
struct FixedRandom;

impl RandomKey for FixedRandom {
    type Key = u64;

    fn key(seed: u64) -> Self::Key {
        seed
    }

    fn split(key: &Self::Key, count: usize) -> Vec<Self::Key> {
        (0..count).map(|i| *key + i as u64 + 1).collect()
    }

    fn fold_in(key: &Self::Key, data: u64) -> Self::Key {
        *key ^ data
    }
}

impl RandomSampler for FixedRandom {
    type Real = f64;
    type Bool = bool;
    type Int = i128;

    fn uniform(
        _key: &Self::Key,
        _shape: &[usize],
        low: Self::Real,
        high: Self::Real,
    ) -> Self::Real {
        low + 0.25 * (high - low)
    }

    fn normal(
        _key: &Self::Key,
        _shape: &[usize],
        mean: Self::Real,
        stddev: Self::Real,
    ) -> Self::Real {
        mean - 0.5 * stddev
    }

    fn bernoulli(_key: &Self::Key, _shape: &[usize], p: Self::Real) -> Self::Bool {
        p > 0.5
    }

    fn randint(_key: &Self::Key, _shape: &[usize], low: Self::Int, _high: Self::Int) -> Self::Int {
        low
    }
}

fn sample<D>(
    distribution: &D,
    key: &<FixedRandom as RandomKey>::Key,
) -> <D as SampleWithKey<FixedRandom>>::Output
where
    D: SampleWithKey<FixedRandom>,
{
    <D as SampleWithKey<FixedRandom>>::sample_with_key(distribution, key)
}

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-12, "{a} != {b}");
}

#[test]
fn lazy_namespace_exposes_generic_distribution_surface() {
    let p = Normal::new(0.0, 1.0);
    let q = Normal::new(1.0, 2.0);

    close(
        p.ln_pdf(0.25),
        bayes::kernels::normal_log_pdf(0.25, 0.0, 1.0),
    );
    close(
        p.entropy(),
        0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E).ln(),
    );
    assert!(p.kl_divergence(&q) > 0.0);
    assert!(p.hellinger_sq(&q) > 0.0);

    let key = FixedRandom::key(1);
    close(sample(&Uniform::new(2.0, 6.0), &key), 3.0);
    close(sample(&Normal::new(1.0, 2.0), &key), 0.0);
    assert!(sample(&Bernoulli::new(0.75), &key));

    let p = [0.25, 0.75];
    let q = [0.5, 0.5];
    close(shannon_entropy_real(&p), bayes::shannon_entropy(&p));
    close(cross_entropy_real(&p, &q), bayes::cross_entropy(&p, &q));
    close(jensen_shannon_real(&p, &q), bayes::jensen_shannon(&p, &q));
    close(
        f_divergence_real(&p, &q, &Kl),
        bayes::f_divergence(&p, &q, &Kl),
    );
    close(brier_score_real(&p, 1), bayes::brier_score(&p, 1));
}
