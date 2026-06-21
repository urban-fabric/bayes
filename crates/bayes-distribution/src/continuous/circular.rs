use super::finite;
use bayes_error::BayesError;
use euclid::LazyReal;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VonMises<R: LazyReal = f64> {
    mu: R,
    kappa: R,
}

impl VonMises<f64> {
    pub fn new(mu: f64, kappa: f64) -> Result<Self, BayesError> {
        finite("mu", mu)?;
        finite("kappa", kappa)?;
        if kappa < 0.0 {
            return Err(BayesError::Malformed {
                what: "kappa must be non-negative",
            });
        }
        Ok(Self { mu, kappa })
    }
}

impl<R: LazyReal> VonMises<R> {
    pub fn from_lazy(mu: R, kappa: R) -> Self {
        Self { mu, kappa }
    }

    pub fn mu(&self) -> R {
        self.mu.clone()
    }

    pub fn kappa(&self) -> R {
        self.kappa.clone()
    }
}
