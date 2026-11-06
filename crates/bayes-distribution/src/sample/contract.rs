//! The explicit-key sampling contract.
//!
//! A convenience layer over euclid's `RandomSampler`: the backend still chooses
//! its own output carrier through `B::Real`, `B::Bool` and `B::Int`, so these
//! traits only supply the shape. Keys are explicit and immutable rather than a
//! mutable RNG handle, which is what lets the same draw be reproduced on a
//! tensor or graph carrier where there is no sequential call order to rely on.
use euclid_random::RandomSampler;

/// Sample a distribution from an explicit-key backend with scalar shape.
///
/// This is a bayes convenience trait over [`RandomSampler`]. Backends that represent tensor or
/// graph values still choose their own output carrier through `B::Real`, `B::Bool`, and `B::Int`;
/// this method supplies the empty host shape.
pub trait SampleWithKey<B: RandomSampler> {
    type Output;

    fn sample_with_key(&self, key: &B::Key) -> Self::Output;
}

/// Sample a distribution from an explicit-key backend with a host-known leading shape.
///
/// Euclid's current random contract uses one sampler trait for scalar and shaped draws. This
/// companion preserves bayes' shaped sampling surface while delegating to [`RandomSampler`].
pub trait SampleTensorWithKey<B: RandomSampler> {
    type Output;

    fn sample_tensor_with_key(&self, key: &B::Key, shape: &[usize]) -> Self::Output;
}

impl<D, B> SampleWithKey<B> for D
where
    B: RandomSampler,
    D: SampleTensorWithKey<B>,
{
    type Output = <D as SampleTensorWithKey<B>>::Output;

    fn sample_with_key(&self, key: &B::Key) -> Self::Output {
        self.sample_tensor_with_key(key, &[])
    }
}
