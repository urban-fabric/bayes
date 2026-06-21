use super::positive;
use bayes_error::BayesError;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChiSquared {
    freedom: f64,
}

impl ChiSquared {
    pub fn new(freedom: f64) -> Result<Self, BayesError> {
        Ok(Self {
            freedom: positive("freedom", freedom)?,
        })
    }

    pub fn freedom(&self) -> f64 {
        self.freedom
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Exponential {
    rate: f64,
}

impl Exponential {
    pub fn new(rate: f64) -> Result<Self, BayesError> {
        Ok(Self {
            rate: positive("rate", rate)?,
        })
    }

    pub fn rate(&self) -> f64 {
        self.rate
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FisherSnedecor {
    d1: f64,
    d2: f64,
}

impl FisherSnedecor {
    pub fn new(d1: f64, d2: f64) -> Result<Self, BayesError> {
        Ok(Self {
            d1: positive("d1", d1)?,
            d2: positive("d2", d2)?,
        })
    }

    pub fn d1(&self) -> f64 {
        self.d1
    }

    pub fn d2(&self) -> f64 {
        self.d2
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gamma {
    shape: f64,
    rate: f64,
}

impl Gamma {
    pub fn new(shape: f64, rate: f64) -> Result<Self, BayesError> {
        Ok(Self {
            shape: positive("shape", shape)?,
            rate: positive("rate", rate)?,
        })
    }

    pub fn shape(&self) -> f64 {
        self.shape
    }

    pub fn rate(&self) -> f64 {
        self.rate
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InverseGamma {
    shape: f64,
    scale: f64,
}

impl InverseGamma {
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
