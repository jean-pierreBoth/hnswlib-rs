//! Reachability probe for jean-pierreBoth/hnswlib-rs#38.
//!
//! Answers the maintainer's three questions with one harness:
//!   * a level-scale sweep (incl. 0.2, where the graph is flat and the layer
//!     bug cannot exist by construction),
//!   * clustered data, including many-clusters/few-points-each,
//!   * and it REPORTS THE GRAPH SHAPE so "flat" is measured, not assumed.
//!
//! usage: orphan_probe <n> <M> <dim> <scale> <mode> <nclusters> <sigma> <seed>
//!        mode = uniform | clustered

use anndists::dist::*;
use hnsw_rs::prelude::*;

fn rand_unit(seed: u64, dim: usize) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2654435761) % (1u64 << 31);
    let mut v = vec![0.0f32; dim];
    let mut norm = 0.0f32;
    for slot in v.iter_mut() {
        x = x.wrapping_mul(1103515245).wrapping_add(12345) % (1u64 << 31);
        *slot = (x as f32) / (1u32 << 30) as f32 - 1.0;
        norm += *slot * *slot;
    }
    let norm = norm.sqrt();
    for slot in v.iter_mut() {
        *slot /= norm;
    }
    v
}

/// centre + sigma * noise, renormalised. sigma -> 0 gives near-duplicate points.
fn clustered(i: usize, dim: usize, nclusters: usize, sigma: f32, seed: u64) -> Vec<f32> {
    let c = rand_unit(seed.wrapping_add(1_000_000 + (i % nclusters) as u64), dim);
    let n = rand_unit(seed.wrapping_add(7_000_000 + i as u64), dim);
    let mut v: Vec<f32> = c.iter().zip(n.iter()).map(|(a, b)| a + sigma * b).collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    for s in v.iter_mut() {
        *s /= norm;
    }
    v
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n: usize = a[1].parse().unwrap();
    let m: usize = a[2].parse().unwrap();
    let dim: usize = a[3].parse().unwrap();
    let scale: f64 = a[4].parse().unwrap();
    let mode = a[5].clone();
    let nclusters: usize = a[6].parse().unwrap();
    let sigma: f32 = a[7].parse().unwrap();
    let seed: u64 = a[8].parse().unwrap();
    let k: usize = a.get(9).map(|v| v.parse().unwrap()).unwrap_or(200);
    let ef: usize = a.get(10).map(|v| v.parse().unwrap()).unwrap_or(1024);
    let efc: usize = a.get(11).map(|v| v.parse().unwrap()).unwrap_or(100);

    let pts: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            if mode == "clustered" {
                clustered(i, dim, nclusters, sigma, seed)
            } else {
                rand_unit(seed.wrapping_add(i as u64), dim)
            }
        })
        .collect();

    let mut hnsw = Hnsw::<f32, DistCosine>::new(m, n, 16, efc, DistCosine {});
    if (scale - 1.0).abs() > f64::EPSILON {
        hnsw.modify_level_scale(scale);
    }
    for (i, p) in pts.iter().enumerate() {
        hnsw.insert_data(p, i);
    }

    // Graph shape: how many points were drawn ABOVE layer 0. If this is 0 the
    // graph is flat and the layer bug cannot manifest -- that is the control.
    let maxlev = hnsw.get_max_level_observed();
    let at0 = hnsw.get_point_indexation().get_layer_nb_point(0);
    let above0 = n - at0;

    // Deliberately generous: at k and ef this far above what a correct graph
    // needs, a point missing from its OWN result is unreachable, not crowded out.
    let orphans = pts
        .iter()
        .enumerate()
        .filter(|(i, p)| !hnsw.search(p, k, ef).iter().any(|nb| nb.d_id == *i))
        .count();

    println!(
        "n={n} M={m} dim={dim} scale={scale} mode={mode} clusters={nclusters} sigma={sigma} seed={seed} maxlevel={maxlev} above0={above0} k={k} ef={ef} efc={efc} orphans={orphans}"
    );
}
