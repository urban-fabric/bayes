use euclid::LazyReal;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frechet<R: LazyReal = f64> {
    shape: R,
    scale: R,
}

impl<R: LazyReal> Frechet<R> {
    pub fn new(shape: R, scale: R) -> Self {
        Self { shape, scale }
    }

    pub fn shape(&self) -> R {
        self.shape.clone()
    }

    pub fn scale(&self) -> R {
        self.scale.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HalfNormal<R: LazyReal = f64> {
    scale: R,
}

impl<R: LazyReal> HalfNormal<R> {
    pub fn new(scale: R) -> Self {
        Self { scale }
    }

    pub fn scale(&self) -> R {
        self.scale.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogNormal<R: LazyReal = f64> {
    mu: R,
    sigma: R,
}

impl<R: LazyReal> LogNormal<R> {
    pub fn new(mu: R, sigma: R) -> Self {
        Self { mu, sigma }
    }

    pub fn mu(&self) -> R {
        self.mu.clone()
    }

    pub fn sigma(&self) -> R {
        self.sigma.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lomax<R: LazyReal = f64> {
    shape: R,
    scale: R,
}

impl<R: LazyReal> Lomax<R> {
    pub fn new(shape: R, scale: R) -> Self {
        Self { shape, scale }
    }

    pub fn shape(&self) -> R {
        self.shape.clone()
    }

    pub fn scale(&self) -> R {
        self.scale.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pareto<R: LazyReal = f64> {
    scale: R,
    shape: R,
}

impl<R: LazyReal> Pareto<R> {
    pub fn new(scale: R, shape: R) -> Self {
        Self { scale, shape }
    }

    pub fn scale(&self) -> R {
        self.scale.clone()
    }

    pub fn shape(&self) -> R {
        self.shape.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rayleigh<R: LazyReal = f64> {
    scale: R,
}

impl<R: LazyReal> Rayleigh<R> {
    pub fn new(scale: R) -> Self {
        Self { scale }
    }

    pub fn scale(&self) -> R {
        self.scale.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wald<R: LazyReal = f64> {
    mu: R,
    shape: R,
}

impl<R: LazyReal> Wald<R> {
    pub fn new(mu: R, shape: R) -> Self {
        Self { mu, shape }
    }

    pub fn mu(&self) -> R {
        self.mu.clone()
    }

    pub fn shape(&self) -> R {
        self.shape.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Weibull<R: LazyReal = f64> {
    scale: R,
    shape: R,
}

impl<R: LazyReal> Weibull<R> {
    pub fn new(scale: R, shape: R) -> Self {
        Self { scale, shape }
    }

    pub fn scale(&self) -> R {
        self.scale.clone()
    }

    pub fn shape(&self) -> R {
        self.shape.clone()
    }
}
