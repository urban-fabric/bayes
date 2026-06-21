use euclid::LazyReal;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cauchy<R: LazyReal = f64> {
    location: R,
    scale: R,
}

impl<R: LazyReal> Cauchy<R> {
    pub fn new(location: R, scale: R) -> Self {
        Self { location, scale }
    }

    pub fn location(&self) -> R {
        self.location.clone()
    }

    pub fn scale(&self) -> R {
        self.scale.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gumbel<R: LazyReal = f64> {
    location: R,
    scale: R,
}

impl<R: LazyReal> Gumbel<R> {
    pub fn new(location: R, scale: R) -> Self {
        Self { location, scale }
    }

    pub fn location(&self) -> R {
        self.location.clone()
    }

    pub fn scale(&self) -> R {
        self.scale.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Laplace<R: LazyReal = f64> {
    location: R,
    scale: R,
}

impl<R: LazyReal> Laplace<R> {
    pub fn new(location: R, scale: R) -> Self {
        Self { location, scale }
    }

    pub fn location(&self) -> R {
        self.location.clone()
    }

    pub fn scale(&self) -> R {
        self.scale.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Logistic<R: LazyReal = f64> {
    location: R,
    scale: R,
}

impl<R: LazyReal> Logistic<R> {
    pub fn new(location: R, scale: R) -> Self {
        Self { location, scale }
    }

    pub fn location(&self) -> R {
        self.location.clone()
    }

    pub fn scale(&self) -> R {
        self.scale.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Normal<R: LazyReal = f64> {
    mean: R,
    std_dev: R,
}

impl<R: LazyReal> Normal<R> {
    pub fn new(mean: R, std_dev: R) -> Self {
        Self { mean, std_dev }
    }

    pub fn params(&self) -> (R, R) {
        (self.mean.clone(), self.std_dev.clone())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StudentT<R: LazyReal = f64> {
    freedom: R,
    location: R,
    scale: R,
}

impl<R: LazyReal> StudentT<R> {
    pub fn new(freedom: R) -> Self {
        Self {
            freedom,
            location: R::zero(),
            scale: R::one(),
        }
    }

    pub fn new_location_scale(freedom: R, location: R, scale: R) -> Self {
        Self {
            freedom,
            location,
            scale,
        }
    }

    pub fn freedom(&self) -> R {
        self.freedom.clone()
    }

    pub fn location(&self) -> R {
        self.location.clone()
    }

    pub fn scale(&self) -> R {
        self.scale.clone()
    }
}
