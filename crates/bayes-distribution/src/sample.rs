//! Explicit-key generic samplers for lazy distribution structs.
//!
//! The existing `rand::Distribution` impls in `bayes-distribution-ext` remain the eager host
//! sampling surface. This module is the backend-generic companion: immutable keys, pure draws,
//! and optional shaped samples through Euclid's random sampler contract.

use crate::{
    Bernoulli, Cauchy, DiscreteUniform, Exponential, Frechet, Gumbel, HalfNormal, Laplace,
    LogNormal, Logistic, Lomax, Normal, Pareto, Rayleigh, Uniform, Wald, Weibull,
};
use euclid::{Element, RandomKey, RandomSampler, Real};

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

fn uniform01<B: RandomSampler>(key: &B::Key, shape: &[usize]) -> B::Real {
    B::uniform(key, shape, B::Real::zero(), B::Real::one())
}

fn standard_normal<B: RandomSampler>(key: &B::Key, shape: &[usize]) -> B::Real {
    B::normal(key, shape, B::Real::zero(), B::Real::one())
}

fn lognormal_sample<R: Real>(z: R, mu: R, sigma: R) -> R {
    (mu + sigma * z).exp()
}

fn half_normal_sample<R: Real>(z: R, scale: R) -> R {
    (scale * z).abs()
}

fn exponential_sample<R: Real>(u: R, rate: R) -> R {
    -((R::one() - u).ln()) / rate
}

fn cauchy_sample<R: Real>(u: R, location: R, scale: R) -> R {
    location + scale * ((u - R::half()) * R::pi()).tan()
}

fn logistic_sample<R: Real>(u: R, location: R, scale: R) -> R {
    let logit = u.clone().ln() - (R::one() - u).ln();
    location + scale * logit
}

fn gumbel_sample<R: Real>(u: R, location: R, scale: R) -> R {
    location - scale * (-(u.ln())).ln()
}

fn laplace_sample<R: Real>(u: R, location: R, scale: R) -> R {
    let centered = u - R::half();
    let sign = R::select(&centered.lt(&R::zero()), -R::one(), R::one());
    let magnitude = -((R::one() - R::two() * centered.abs()).ln());
    location + scale * sign * magnitude
}

fn pareto_sample<R: Real>(u: R, scale: R, shape: R) -> R {
    scale * (R::one() - u).powf(-(R::one() / shape))
}

fn frechet_sample<R: Real>(u: R, scale: R, shape: R) -> R {
    scale * (-(u.ln())).powf(-(R::one() / shape))
}

fn lomax_sample<R: Real>(u: R, shape: R, scale: R) -> R {
    scale * ((R::one() - u).powf(-(R::one() / shape)) - R::one())
}

fn weibull_sample<R: Real>(u: R, scale: R, shape: R) -> R {
    scale * (-((R::one() - u).ln())).powf(R::one() / shape)
}

fn rayleigh_sample<R: Real>(u: R, scale: R) -> R {
    scale * (-(R::two() * (R::one() - u).ln())).sqrt()
}

fn wald_sample<R: Real>(z: R, u: R, mu: R, shape: R) -> R {
    let y = z.square();
    let mu_sq = mu.clone() * mu.clone();
    let two_shape = R::two() * shape.clone();
    let root =
        (R::scalar(4.0) * mu.clone() * shape * y.clone() + mu_sq.clone() * y.clone() * y).sqrt();
    let candidate =
        mu.clone() + mu_sq.clone() * z.square() / two_shape.clone() - mu.clone() * root / two_shape;
    let threshold = mu.clone() / (mu + candidate.clone());
    R::select(&u.le(&threshold), candidate.clone(), mu_sq / candidate)
}

impl<B> SampleTensorWithKey<B> for Uniform<B::Real>
where
    B: RandomSampler,
{
    type Output = B::Real;

    fn sample_tensor_with_key(&self, key: &B::Key, shape: &[usize]) -> B::Real {
        B::uniform(key, shape, self.lo(), self.hi())
    }
}

impl<B> SampleTensorWithKey<B> for Normal<B::Real>
where
    B: RandomSampler,
{
    type Output = B::Real;

    fn sample_tensor_with_key(&self, key: &B::Key, shape: &[usize]) -> B::Real {
        let (mean, std_dev) = self.params();
        B::normal(key, shape, mean, std_dev)
    }
}

impl<B> SampleTensorWithKey<B> for LogNormal<B::Real>
where
    B: RandomSampler,
{
    type Output = B::Real;

    fn sample_tensor_with_key(&self, key: &B::Key, shape: &[usize]) -> B::Real {
        lognormal_sample(standard_normal::<B>(key, shape), self.mu(), self.sigma())
    }
}

impl<B> SampleTensorWithKey<B> for HalfNormal<B::Real>
where
    B: RandomSampler,
{
    type Output = B::Real;

    fn sample_tensor_with_key(&self, key: &B::Key, shape: &[usize]) -> B::Real {
        half_normal_sample(standard_normal::<B>(key, shape), self.scale())
    }
}

impl<B> SampleTensorWithKey<B> for Exponential<B::Real>
where
    B: RandomSampler,
{
    type Output = B::Real;

    fn sample_tensor_with_key(&self, key: &B::Key, shape: &[usize]) -> B::Real {
        exponential_sample(uniform01::<B>(key, shape), self.rate())
    }
}

impl<B> SampleTensorWithKey<B> for Cauchy<B::Real>
where
    B: RandomSampler,
{
    type Output = B::Real;

    fn sample_tensor_with_key(&self, key: &B::Key, shape: &[usize]) -> B::Real {
        cauchy_sample(uniform01::<B>(key, shape), self.location(), self.scale())
    }
}

impl<B> SampleTensorWithKey<B> for Logistic<B::Real>
where
    B: RandomSampler,
{
    type Output = B::Real;

    fn sample_tensor_with_key(&self, key: &B::Key, shape: &[usize]) -> B::Real {
        logistic_sample(uniform01::<B>(key, shape), self.location(), self.scale())
    }
}

impl<B> SampleTensorWithKey<B> for Gumbel<B::Real>
where
    B: RandomSampler,
{
    type Output = B::Real;

    fn sample_tensor_with_key(&self, key: &B::Key, shape: &[usize]) -> B::Real {
        gumbel_sample(uniform01::<B>(key, shape), self.location(), self.scale())
    }
}

impl<B> SampleTensorWithKey<B> for Laplace<B::Real>
where
    B: RandomSampler,
{
    type Output = B::Real;

    fn sample_tensor_with_key(&self, key: &B::Key, shape: &[usize]) -> B::Real {
        laplace_sample(uniform01::<B>(key, shape), self.location(), self.scale())
    }
}

impl<B> SampleTensorWithKey<B> for Pareto<B::Real>
where
    B: RandomSampler,
{
    type Output = B::Real;

    fn sample_tensor_with_key(&self, key: &B::Key, shape: &[usize]) -> B::Real {
        pareto_sample(uniform01::<B>(key, shape), self.scale(), self.shape())
    }
}

impl<B> SampleTensorWithKey<B> for Frechet<B::Real>
where
    B: RandomSampler,
{
    type Output = B::Real;

    fn sample_tensor_with_key(&self, key: &B::Key, shape: &[usize]) -> B::Real {
        frechet_sample(uniform01::<B>(key, shape), self.scale(), self.shape())
    }
}

impl<B> SampleTensorWithKey<B> for Lomax<B::Real>
where
    B: RandomSampler,
{
    type Output = B::Real;

    fn sample_tensor_with_key(&self, key: &B::Key, shape: &[usize]) -> B::Real {
        lomax_sample(uniform01::<B>(key, shape), self.shape(), self.scale())
    }
}

impl<B> SampleTensorWithKey<B> for Weibull<B::Real>
where
    B: RandomSampler,
{
    type Output = B::Real;

    fn sample_tensor_with_key(&self, key: &B::Key, shape: &[usize]) -> B::Real {
        weibull_sample(uniform01::<B>(key, shape), self.scale(), self.shape())
    }
}

impl<B> SampleTensorWithKey<B> for Rayleigh<B::Real>
where
    B: RandomSampler,
{
    type Output = B::Real;

    fn sample_tensor_with_key(&self, key: &B::Key, shape: &[usize]) -> B::Real {
        rayleigh_sample(uniform01::<B>(key, shape), self.scale())
    }
}

impl<B> SampleTensorWithKey<B> for Wald<B::Real>
where
    B: RandomSampler,
{
    type Output = B::Real;

    fn sample_tensor_with_key(&self, key: &B::Key, shape: &[usize]) -> B::Real {
        let keys = <B as RandomKey>::split(key, 2);
        wald_sample(
            standard_normal::<B>(&keys[0], shape),
            uniform01::<B>(&keys[1], shape),
            self.mu(),
            self.shape(),
        )
    }
}

impl<B> SampleTensorWithKey<B> for Bernoulli<B::Real>
where
    B: RandomSampler,
{
    type Output = B::Bool;

    fn sample_tensor_with_key(&self, key: &B::Key, shape: &[usize]) -> B::Bool {
        B::bernoulli(key, shape, self.p())
    }
}

impl<B> SampleTensorWithKey<B> for DiscreteUniform<B::Int>
where
    B: RandomSampler,
{
    type Output = B::Int;

    fn sample_tensor_with_key(&self, key: &B::Key, shape: &[usize]) -> B::Int {
        B::randint(key, shape, self.lo(), self.hi())
    }
}
