use super::{finite, positive};
use bayes_error::BayesError;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frechet {
    shape: f64,
    scale: f64,
}

impl Frechet {
    pub fn new(shape: f64, scale: f64) -> Result<Self, BayesError> {
        Ok(Self {
            shape: positive("shape", shape)?,
            scale: positive("scale", scale)?,
        })
    }

    pub fn shape(&self) -> f64 {
        self.shape
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HalfNormal {
    scale: f64,
}

impl HalfNormal {
    pub fn new(scale: f64) -> Result<Self, BayesError> {
        Ok(Self {
            scale: positive("scale", scale)?,
        })
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogNormal {
    mu: f64,
    sigma: f64,
}

impl LogNormal {
    pub fn new(mu: f64, sigma: f64) -> Result<Self, BayesError> {
        Ok(Self {
            mu: finite("mu", mu)?,
            sigma: positive("sigma", sigma)?,
        })
    }

    pub fn mu(&self) -> f64 {
        self.mu
    }

    pub fn sigma(&self) -> f64 {
        self.sigma
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lomax {
    shape: f64,
    scale: f64,
}

impl Lomax {
    pub fn new(shape: f64, scale: f64) -> Result<Self, BayesError> {
        Ok(Self {
            shape: positive("shape", shape)?,
            scale: positive("scale", scale)?,
        })
    }

    pub fn shape(&self) -> f64 {
        self.shape
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pareto {
    scale: f64,
    shape: f64,
}

impl Pareto {
    pub fn new(scale: f64, shape: f64) -> Result<Self, BayesError> {
        Ok(Self {
            scale: positive("scale", scale)?,
            shape: positive("shape", shape)?,
        })
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }

    pub fn shape(&self) -> f64 {
        self.shape
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rayleigh {
    scale: f64,
}

impl Rayleigh {
    pub fn new(scale: f64) -> Result<Self, BayesError> {
        Ok(Self {
            scale: positive("scale", scale)?,
        })
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wald {
    mu: f64,
    shape: f64,
}

impl Wald {
    pub fn new(mu: f64, shape: f64) -> Result<Self, BayesError> {
        Ok(Self {
            mu: positive("mu", mu)?,
            shape: positive("shape", shape)?,
        })
    }

    pub fn mu(&self) -> f64 {
        self.mu
    }

    pub fn shape(&self) -> f64 {
        self.shape
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Weibull {
    scale: f64,
    shape: f64,
}

impl Weibull {
    pub fn new(scale: f64, shape: f64) -> Result<Self, BayesError> {
        Ok(Self {
            scale: positive("scale", scale)?,
            shape: positive("shape", shape)?,
        })
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }

    pub fn shape(&self) -> f64 {
        self.shape
    }
}
