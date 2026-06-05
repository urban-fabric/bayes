use rand::Rng;
use rand::distr::Distribution;

use super::uniform::Uniform;
use crate::error::BayesError;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Uniform2D {
    x: Uniform,
    y: Uniform,
}

impl Uniform2D {
    pub fn new(lo_x: f64, hi_x: f64, lo_y: f64, hi_y: f64) -> Result<Self, BayesError> {
        Ok(Self {
            x: Uniform::new(lo_x, hi_x)?,
            y: Uniform::new(lo_y, hi_y)?,
        })
    }

    pub fn unit_square() -> Self {
        Self {
            x: Uniform::unit_interval(),
            y: Uniform::unit_interval(),
        }
    }
}

impl Distribution<(f64, f64)> for Uniform2D {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> (f64, f64) {
        let x = self.x.sample(rng);
        let y = self.y.sample(rng);
        (x, y)
    }
}
#[cfg(test)]
mod tests;
