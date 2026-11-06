//! The count families, which need euclid's `CountSampler`.
//!
//! Same reasoning as the gamma family: a Poisson or binomial quantile has no
//! closed form, so the backend supplies the draw and these compose on it. The
//! beta-binomial is the compound case — a binomial whose success probability is
//! itself a beta draw — which is why it needs a split key rather than one.
use crate::{BetaBinomial, Binomial, Poisson};
use euclid_numeric::Natural;
use euclid_random::CountSampler;

use super::contract::SampleTensorWithKey;

// --- count family ----------------------------------------------------------

impl<B> SampleTensorWithKey<B> for Poisson<B::Real>
where
    B: CountSampler,
{
    type Output = B::Int;

    fn sample_tensor_with_key(&self, key: &B::Key, shape: &[usize]) -> B::Int {
        B::poisson(key, shape, self.lambda())
    }
}

impl<B, N> SampleTensorWithKey<B> for Binomial<N>
where
    B: CountSampler,
    N: Natural<Real = B::Real>,
{
    type Output = B::Int;

    fn sample_tensor_with_key(&self, key: &B::Key, shape: &[usize]) -> B::Int {
        B::binomial(key, shape, self.trials().to_real(), self.p())
    }
}

impl<B, N> SampleTensorWithKey<B> for BetaBinomial<N>
where
    B: CountSampler,
    N: Natural<Real = B::Real>,
{
    type Output = B::Int;

    fn sample_tensor_with_key(&self, key: &B::Key, shape: &[usize]) -> B::Int {
        B::beta_binomial(
            key,
            shape,
            self.trials().to_real(),
            self.alpha(),
            self.beta(),
        )
    }
}
