# hnsw-rs

This crate provides a Rust implementation of the paper by Yu.A. Malkov and D.A Yashunin:

"Efficient and Robust approximate nearest neighbours using Hierarchical Navigable Small World Graphs" (2016,2018)
[https://arxiv.org/abs/1603.09320]

## License

Licensed under either of

* Apache License, Version 2.0, [LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>
* MIT license [LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>

at your option.

This software was written on my own while working at [CEA](http://www.cea.fr/).

## Functionalities

The crate provides:

* usual distances as L1, L2, Cosine, Jaccard, Hamming for vectors of standard numeric types.

* Hellinger distance and Jeffreys divergence between probability distributions (f32 and f64). It must be noted that the Jeffreys divergence
(a symetrized Kullback-Leibler divergence) do not satisfy the triangle inequality. (Neither Cosine distance !).

* Jensen-Shannon distance between probability distributions (f32 and f64). It is defined as the **square root** of the Jensen-Shannon divergence and is a bounded metric. See [Nielsen F. in Entropy 2019, 21(5), 485](https://doi.org/10.3390/e21050485)

* Levenshtein distance on u16.

* A Trait to enable the user to implement its own distances.
  It takes as data slices of types T satisfying T:Serialize+Clone+Send+Sync. It is also possible to use C extern functions or closures.

* An interface towards C and more specifically to the [Julia](https://julialang.org/) language.
See the companion Julia package [HnswAnn.jl](https://gitlab.com/jpboth/HnswAnn.jl) and the building paragraph for some help for Julia users.

* Multithreaded insertion and search requests.
  
* Dump and reload functions (*See module hnswio*) to store the data and the graph once it is built. These facilities rely partly on Serde so T needs to implement Serialize and Deserialized as derived by Serde.
  It is also possible to reload only the graph and not the data themselves. A specific type (struct NoData, associated to the NoDist distance is dedicated to this functionality.

* A flattening conversion of the Hnsw structure to keep only neighborhood relationships between points (without their internal data) internal to the Hnsw structure (*see module flatten.rs, FlatPoint and FlatNeighborhood*). It is thus possible to keep some topology information with low memory usage.

## Implementation

The graph construction and searches are multithreaded with the **parking_lot** crate (See **parallel_insert_data** and **parallel_search_neighbours** functions and also examples files).  
Simd Avx2 implementation, currently based on the **simdeez** crate, is provided for most distances in the **f32** heavily used case and for the Hamming distance for **i32**. See *Building*.

## Building

### Simd

- The simd provided by the simdeez crate is accessible with the feature "simdeez_f" for x86_64 processors.
Compile with **cargo build --release --features "simdeez_f"** .... 
To compile this crate on a M1 chip just do not activate this feature.
As soon as std::simd lands in rust stable, it will be the default.

- It is nevertheless possible to experiment with std::simd. Compiling with the feature stdsimd
  (**cargo build --release --features "stdsimd"**), activates the transitory crate packed_simd_2. This requires rust nightly and also to uncomment the first line of file *lib.rs*.
  **This is discouraged** as (now) only the Hamming distance with the u32x16 and u64x8 types is provided.


### Julia interface

By default the crate is a standalone project and builds a static libray and executable.
To be used with the companion Julia package it is necessary to build a dynamic library.
This can be done by just uncommenting (i.e get rid of the #) in file Cargo.toml the line:

*#crate-type = ["cdylib"]*

and rerun the command: cargo build --release.

This will generate a .so file in the target/release directory.

## Algorithm and Input Parameters

The algorithm stores points in layers (at most 16), and a graph is constructed to enable a search from less densely populated levels to most densely populated levels by constructing links from less dense layers to the most dense layer (level 0).

Roughly the algorithm goes along runs as follows:

Upon insertion, the level ***l*** of a new point is sampled with an exponential law, limiting the number of levels to 16,
so that level 0 is the most densely populated layer, upper layers being exponentially less populated as level increases.  
The nearest neighbour of the point is searched in lookup tables from the upper level to the level just above its layer (***l***), so we should arrive near the new point at its level at a relatively low cost. Then the ***max_nb_connection*** nearest neighbours are searched in neighbours of neighbours table (with a reverse updating of tables) recursively from its layer ***l*** down to the most populated level 0.  

The scale parameter of the exponential law depends on the maximum number of connection possible for a point (parameter ***max_nb_connection***) to others.  
Explicitly the scale parameter is chosen as : `scale=1/ln(max_nb_connection)`.

The main parameters occuring in constructing the graph or in searching are:

* max_nb_connection (in hnsw initialization)
    The maximum number of links from one point to others. Values ranging from 16 to 64 are standard initialising values, the higher the more time consuming.

* ef_construction (in hnsw initialization)  
  This parameter controls the width of the search for neighbours during insertion. Values from 200 to 800 are standard initialising values, the higher the more time consuming.

* max_layer (in hnsw initialization)  
    The maximum number of layers in graph. Must be less or equal than 16.

* ef_arg (in search methods)  
    This parameter controls the width of the search in the lowest level, it must be greater than number of neighbours asked but can be less than ***ef_construction***.
    As a rule of thumb could be between the number of neighbours we will ask for (knbn arg in search method) and max_nb_connection.

* keep_pruned and extend_candidates.  
    These parameters are described in the paper by Malkov and Yashunin can be used to
    modify the search strategy. The interested user should check the paper to see the impact. By default
    the values are as recommended in the paper.

## Examples and Benchmarks

Some examples are taken from the [ann-benchmarks site](https://github.com/erikbern/ann-benchmarks)
and recall rates and request/s are given in comments in the examples files for some input parameters.
The annhdf5 module implements reading the standardized data files
of the [ann-benchmarks site](https://github.com/erikbern/ann-benchmarks),
just download the necessary benchmark data files and modify path in sources accordingly.
Then run: cargo build --examples --release.  
It is possible in these examples to change from parallel searches to serial searches to check for speeds
or modify parameters to see the impact on performance.

For example:
1. On the fashion-mnist-784-euclidean benchmark search requests run at 22197 req/s with a recall rate of 0.9775 on a laptop with 8 i7-cores at 2.3Ghz
2. On the sift1m benchmark (1 million points in 128 dimension) search requests for the 10 first neighbours runs at 6100 req/s with a recall rate of 0.9907


Some lines extracted from this benchmark show how it works for f32 and L2 norm

```rust
    //  reading data
    let anndata = AnnBenchmarkData::new(fname).unwrap();
    let nb_elem = anndata.train_data.len();
    let max_nb_connection = 24;
    let nb_layer = 16.min((nb_elem as f32).ln().trunc() as usize);
    let ef_c = 400;
    // allocating network
    let mut hnsw =  Hnsw::<f32, DistL2>::new(max_nb_connection, nb_elem, nb_layer, ef_c, DistL2{});
    hnsw.set_extend_candidates(false);
    // parallel insertion of train data
    let data_for_par_insertion = anndata.train_data.iter().map( |x| (&x.0, x.1)).collect();
    hnsw.parallel_insert(&data_for_par_insertion);
    //
    hnsw.dump_layer_info();
    //  Now the bench with 10 neighbours
    let mut knn_neighbours_for_tests = Vec::<Vec<Neighbour>>::with_capacity(nb_elem);
    hnsw.set_searching_mode(true);
    let knbn = 10;
    let ef_c = max_nb_connection;
    // search 10 nearest neighbours for test data
    knn_neighbours_for_tests = hnsw.parallel_search(&anndata.test_data, knbn, ef_c);
    ....
```



## Contributions

Petter Egesund added the DistLevenshtein distance.

## Notes

1. Upgrade of many dependencies. Change from simple_logger to env_logger. The logger is initialized one for all in file src/lib.rs and cannot be intialized twice. The level of log can be modulated by the RUST_LOG env variable on a module basis or switched off. See the *env_logger* crate doc.
2. A rust crate *edlib_rs* provides an interface to the *excellent* edlib C++ library  [(Cf edlib)](https://github.com/Martinsos/edlib) can be found at [edlib_rs](https://github.com/jean-pierreBoth/edlib-rs) or on crate.io. It can be used to define a user adhoc distance on &[u8] with normal, prefix or infix mode (which is useful in genomics alignment).
3. The library do not depend anymore on hdf5 and ndarray. They are dev-dependancies needed for examples, this simplify compatibility issues.
4. Added insertion methods for slices for easier use with the ndarray crate.
5. simd/avx2 requires now the feature "simdeez_f". So by default the crate can compile on M1 chip and transitions to std::simd.
