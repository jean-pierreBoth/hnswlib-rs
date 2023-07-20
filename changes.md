1. Upgrade of many dependencies. Change from simple_logger to env_logger. The logger is initialized one for all in file src/lib.rs and cannot be intialized twice. The level of log can be modulated by the RUST_LOG env variable on a module basis or switched off. See the *env_logger* crate doc.
2. A rust crate *edlib_rs* provides an interface to the *excellent* edlib C++ library  [(Cf edlib)](https://github.com/Martinsos/edlib) can be found at [edlib_rs](https://github.com/jean-pierreBoth/edlib-rs) or on crate.io. It can be used to define a user adhoc distance on &[u8] with normal, prefix or infix mode (which is useful in genomics alignment).
3. The library do not depend anymore on hdf5 and ndarray. They are dev-dependancies needed for examples, this simplify compatibility issues.
4. Added insertion methods for slices for easier use with the ndarray crate.
5. simd/avx2 requires now the feature "simdeez_f". So by default the crate can compile on M1 chip and transitions to std::simd.
6. Added DistPtr and possiblity to dump/reload with this distance type. (See *load_hnsw_with_dist* function)
7. Implementation of Hamming for f64 exclusively in the context SuperMinHash in crate [probminhash](https://crates.io/crates/probminhash)
8. Implementation of filtering and simplification of the interface for reloading a previous dump.