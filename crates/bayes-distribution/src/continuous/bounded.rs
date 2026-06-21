use euclid::LazyReal;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Beta<R: LazyReal = f64> {
    alpha: R,
    beta: R,
}

impl<R: LazyReal> Beta<R> {
    pub fn new(alpha: R, beta: R) -> Self {
        Self { alpha, beta }
    }

    pub fn alpha(&self) -> R {
        self.alpha.clone()
    }

    pub fn beta(&self) -> R {
        self.beta.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Uniform<R: LazyReal = f64> {
    lo: R,
    hi: R,
}

impl<R: LazyReal> Uniform<R> {
    pub fn new(lo: R, hi: R) -> Self {
        Self { lo, hi }
    }

    pub fn lo(&self) -> R {
        self.lo.clone()
    }

    pub fn hi(&self) -> R {
        self.hi.clone()
    }
}
