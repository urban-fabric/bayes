//! The gamma family: the draws that need a keystone primitive.
//!
//! A gamma quantile has no closed form, so unlike the location-scale families
//! these cannot be built from a uniform alone. euclid's `GammaSampler` supplies
//! `standard_gamma` as the one primitive a backend must implement, and
//! everything here composes on top of it: `Beta` from two gammas, `ChiSquared`
//! and `StudentT` and `FisherSnedecor` and `InverseGamma` from those.
use super::location_scale::standard_normal;
use crate::{Beta, ChiSquared, FisherSnedecor, Gamma, InverseGamma, StudentT};
use euclid_numeric::Real;
use euclid_random::{GammaSampler, RandomKey};

use super::contract::SampleTensorWithKey;

// --- gamma family ----------------------------------------------------------
//
// The single backend primitive is `GammaSampler::standard_gamma`; the rest of the family
// composes from it (plus `normal` and key splitting), so a backend that supplies one kernel
// gets all six distributions.

impl<B> SampleTensorWithKey<B> for Gamma<B::Real>
where
    B: GammaSampler,
{
    type Output = B::Real;

    fn sample_tensor_with_key(&self, key: &B::Key, shape: &[usize]) -> B::Real {
        // Gamma(shape, rate) = Gamma(shape, 1) / rate.
        B::standard_gamma(key, shape, self.shape()) / self.rate()
    }
}

impl<B> SampleTensorWithKey<B> for Beta<B::Real>
where
    B: GammaSampler,
{
    type Output = B::Real;

    fn sample_tensor_with_key(&self, key: &B::Key, shape: &[usize]) -> B::Real {
        // Beta(a, b) = Gₐ / (Gₐ + G_b) with independent unit-rate gammas.
        let keys = <B as RandomKey>::split(key, 2);
        let ga = B::standard_gamma(&keys[0], shape, self.alpha());
        let gb = B::standard_gamma(&keys[1], shape, self.beta());
        ga.clone() / (ga + gb)
    }
}

impl<B> SampleTensorWithKey<B> for ChiSquared<B::Real>
where
    B: GammaSampler,
{
    type Output = B::Real;

    fn sample_tensor_with_key(&self, key: &B::Key, shape: &[usize]) -> B::Real {
        // χ²(ν) = 2 · Gamma(ν/2, 1).
        let two = B::Real::two();
        let half = self.freedom() / two.clone();
        two * B::standard_gamma(key, shape, half)
    }
}

impl<B> SampleTensorWithKey<B> for StudentT<B::Real>
where
    B: GammaSampler,
{
    type Output = B::Real;

    fn sample_tensor_with_key(&self, key: &B::Key, shape: &[usize]) -> B::Real {
        // t(ν) = Z / √(χ²(ν) / ν), then shifted/scaled.
        let keys = <B as RandomKey>::split(key, 2);
        let z = standard_normal::<B>(&keys[0], shape);
        let nu = self.freedom();
        let chi2 = B::Real::two() * B::standard_gamma(&keys[1], shape, nu.clone() / B::Real::two());
        let t = z / (chi2 / nu).sqrt();
        self.location() + self.scale() * t
    }
}

impl<B> SampleTensorWithKey<B> for FisherSnedecor<B::Real>
where
    B: GammaSampler,
{
    type Output = B::Real;

    fn sample_tensor_with_key(&self, key: &B::Key, shape: &[usize]) -> B::Real {
        // F(d1, d2) = (χ²(d1)/d1) / (χ²(d2)/d2).
        let keys = <B as RandomKey>::split(key, 2);
        let (d1, d2) = (self.d1(), self.d2());
        let u = B::Real::two() * B::standard_gamma(&keys[0], shape, d1.clone() / B::Real::two());
        let v = B::Real::two() * B::standard_gamma(&keys[1], shape, d2.clone() / B::Real::two());
        (u / d1) / (v / d2)
    }
}

impl<B> SampleTensorWithKey<B> for InverseGamma<B::Real>
where
    B: GammaSampler,
{
    type Output = B::Real;

    fn sample_tensor_with_key(&self, key: &B::Key, shape: &[usize]) -> B::Real {
        // InverseGamma(shape, scale) = scale / Gamma(shape, 1).
        self.scale() / B::standard_gamma(key, shape, self.shape())
    }
}
