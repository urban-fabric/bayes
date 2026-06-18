//! The error-function family (`erf`/`erfc`/`erf_inv`) — now owned by
//! [`euclid::special`] and re-exported here. It is the workhorse of the Normal CDF
//! (`Φ(x) = ½ erfc(-x/√2)`) and quantile, and is the concrete reference behind the
//! [`euclid::numeric::LazyReal`] `erf`/`erfc`/`erf_inv` methods.

pub use euclid::special::erf::{erf, erf_inv, erfc};
