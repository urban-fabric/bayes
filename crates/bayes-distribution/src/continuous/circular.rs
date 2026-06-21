use euclid::LazyReal;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VonMises<R: LazyReal = f64> {
    mu: R,
    kappa: R,
}

impl<R: LazyReal> VonMises<R> {
    pub fn new(mu: R, kappa: R) -> Self {
        Self { mu, kappa }
    }

    pub fn mu(&self) -> R {
        self.mu.clone()
    }

    pub fn kappa(&self) -> R {
        self.kappa.clone()
    }
}
