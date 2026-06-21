use euclid::LazyReal;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChiSquared<R: LazyReal = f64> {
    freedom: R,
}

impl<R: LazyReal> ChiSquared<R> {
    pub fn new(freedom: R) -> Self {
        Self { freedom }
    }

    pub fn freedom(&self) -> R {
        self.freedom.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Exponential<R: LazyReal = f64> {
    rate: R,
}

impl<R: LazyReal> Exponential<R> {
    pub fn new(rate: R) -> Self {
        Self { rate }
    }

    pub fn rate(&self) -> R {
        self.rate.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FisherSnedecor<R: LazyReal = f64> {
    d1: R,
    d2: R,
}

impl<R: LazyReal> FisherSnedecor<R> {
    pub fn new(d1: R, d2: R) -> Self {
        Self { d1, d2 }
    }

    pub fn d1(&self) -> R {
        self.d1.clone()
    }

    pub fn d2(&self) -> R {
        self.d2.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gamma<R: LazyReal = f64> {
    shape: R,
    rate: R,
}

impl<R: LazyReal> Gamma<R> {
    pub fn new(shape: R, rate: R) -> Self {
        Self { shape, rate }
    }

    pub fn shape(&self) -> R {
        self.shape.clone()
    }

    pub fn rate(&self) -> R {
        self.rate.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InverseGamma<R: LazyReal = f64> {
    shape: R,
    scale: R,
}

impl<R: LazyReal> InverseGamma<R> {
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
