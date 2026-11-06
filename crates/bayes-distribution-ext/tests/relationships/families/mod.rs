//! Exact reparameterizations: algebraic identities between families, at a
//! tolerance near machine precision because none of them is asymptotic.
//!
//! | module | the relation |
//! |--------|--------------|
//! | [`reparameterized`] | one family *is* another at particular parameters |
//! | [`transformed`] | one family is a fold, log, reciprocal or power of another |
//! | [`derived`] | one family is a ratio of others; plus exact closed forms |
//! | [`discrete`] | the same, among the discrete families |

mod derived;
mod discrete;
mod reparameterized;
mod transformed;
