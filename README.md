# hnsw-rs

This crate provides a Rust implementation of the paper by Yu.A. Malkov and D.A Yashunin:

"Efficient and Robust approximate nearest neighbours using Hierarchical Navigable Small World Graphs" (2016,2018)
[https://arxiv.org/abs/1603.09320]

## License

Licensed under either of

* Apache License, Version 2.0, [LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>
* MIT license [LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>

at your option.

This software was written on my own while working at [CEA](http://www.cea.fr/), [CEA-LIST](http://www-list.cea.fr/en/)

## Functionalities

The crate provides:

* usual distances as L1, L2, Cosine, Jaccard, Hamming for vectors of standard numeric types.

* Hellinger and Jeffreys distances between probability distributions (f32 and f64). It must be noted that the Jeffreys distance
(a symetrized Kullback-Leibler divergence) do not satisfy the triangle inequality. (Neither Cosine distance !).

* Levenshtein distance on u16.

* A structure to enable the user to implement its own distances. It takes as data, vectors of types T:Copy+Clone+Send+Sync.

* An interface towards C and more specifically to the [Julia](https://julialang.org/) language.
See the companion Julia package [HnswAnn.jl](https://gitlab.com/jpboth/HnswAnn.jl) and the building paragraph for some help for Julia users.

* Dump and reload functions (Cf module hnswio) to store the graph once it is built. As the time necessary to compute the graph can be important it can be useful to store it for future use.

## Implementation

The graph construction and searches are multithreaded with the **parking_lot** crate (See **parallel_insert_data** and **parallel_search_neighbours** functions and also examples files).
For the heavily used case (f32) we provide simd avx2 implementation in distance computations
currently based on the **simdeez** crate.

## Building

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

For example on the fashion-mnist-784-euclidean benchmark search requests run at 12791 req/s with a recall rate of 0.9765 on a laptop with 4 i7 cores at 2.7Ghz

## Contributions

Petter Egesund added Levenshtein distance.
