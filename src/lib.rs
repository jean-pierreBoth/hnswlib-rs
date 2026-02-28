#![cfg_attr(feature = "stdsimd", feature(portable_simd))]

pub mod api;
pub mod datamap;
pub mod filter;
pub mod flatten;
pub mod hnsw;
pub mod hnswio;
pub mod libext;
pub mod prelude;

// we impose our version of anndists
pub use anndists;
