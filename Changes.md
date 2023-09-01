- The **version 0.2** introduces 
    1. possibility to use mmap on the data file storing  the vectors represented in the hnsw structure. This is mostly usefule for
    large vectors, where data needs more space than the graph part.
    As a consequence the format of this file changed. Old format can be read but new dumps will be in the new format.  
    In case of mmap usage, a dump after inserting new elements must ensure that the old file is not overwritten, so a unique file name is
  generated if necessary. See documentation of module Hnswio

    2. the filtering trait
  
   
-  Upgrade of many dependencies. Change from simple_logger to env_logger. The logger is initialized one for all in file src/lib.rs and cannot be intialized twice. The level of log can be modulated by the RUST_LOG env variable on a module basis or switched off. See the *env_logger* crate doc.
  
- A rust crate *edlib_rs* provides an interface to the *excellent* edlib C++ library  [(Cf edlib)](https://github.com/Martinsos/edlib) can be found at [edlib_rs](https://github.com/jean-pierreBoth/edlib-rs) or on crate.io. It can be used to define a user adhoc distance on &[u8] with normal, prefix or infix mode (which is useful in genomics alignment).
  
- The library do not depend anymore on hdf5 and ndarray. They are dev-dependancies needed for examples, this simplify compatibility issues.
- Added insertion methods for slices for easier use with the ndarray crate.
  
- simd/avx2 requires now the feature "simdeez_f". So by default the crate can compile on M1 chip and transitions to std::simd.
  
- Added DistPtr and possiblity to dump/reload with this distance type. (See *load_hnsw_with_dist* function)
  
- Implementation of Hamming for f64 exclusively in the context SuperMinHash in crate [probminhash](https://crates.io/crates/probminhash)

