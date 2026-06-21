use super::{finite, positive};
use bayes_error::BayesError;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cauchy {
    location: f64,
    scale: f64,
}

impl Cauchy {
    pub fn new(location: f64, scale: f64) -> Result<Self, BayesError> {
        Ok(Self {
            location: finite("location", location)?,
            scale: positive("scale", scale)?,
        })
    }

    pub fn location(&self) -> f64 {
        self.location
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gumbel {
    location: f64,
    scale: f64,
}

impl Gumbel {
    pub fn new(location: f64, scale: f64) -> Result<Self, BayesError> {
        Ok(Self {
            location: finite("location", location)?,
            scale: positive("scale", scale)?,
        })
    }

    pub fn location(&self) -> f64 {
        self.location
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Laplace {
    location: f64,
    scale: f64,
}

impl Laplace {
    pub fn new(location: f64, scale: f64) -> Result<Self, BayesError> {
        Ok(Self {
            location: finite("location", location)?,
            scale: positive("scale", scale)?,
        })
    }

    pub fn location(&self) -> f64 {
        self.location
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Logistic {
    location: f64,
    scale: f64,
}

impl Logistic {
    pub fn new(location: f64, scale: f64) -> Result<Self, BayesError> {
        Ok(Self {
            location: finite("location", location)?,
            scale: positive("scale", scale)?,
        })
    }

    pub fn location(&self) -> f64 {
        self.location
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Normal {
    mean: f64,
    std_dev: f64,
}

impl Normal {
    pub fn new(mean: f64, std_dev: f64) -> Result<Self, BayesError> {
        Ok(Self {
            mean: finite("mean", mean)?,
            std_dev: positive("std_dev", std_dev)?,
        })
    }

    pub fn params(&self) -> (f64, f64) {
        (self.mean, self.std_dev)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StudentT {
    freedom: f64,
    location: f64,
    scale: f64,
}

impl StudentT {
    pub fn new(freedom: f64) -> Result<Self, BayesError> {
        Self::new_location_scale(freedom, 0.0, 1.0)
    }

    pub fn new_location_scale(freedom: f64, location: f64, scale: f64) -> Result<Self, BayesError> {
        Ok(Self {
            freedom: positive("freedom", freedom)?,
            location: finite("location", location)?,
            scale: positive("scale", scale)?,
        })
    }

    pub fn freedom(&self) -> f64 {
        self.freedom
    }

    pub fn location(&self) -> f64 {
        self.location
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }
}
