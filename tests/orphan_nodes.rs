//! Every inserted point must remain reachable by a search for its own vector.
//!
//! Regression test for symmetric edges being stored on the wrong layer.
//!
//! `reverse_update_neighborhood_simple` wrote every backlink into the
//! neighbour's list at index `new_point.p_id.0` — the new point's *top* level —
//! rather than at `l`, the layer the forward edge was built on.  A point drawn
//! at level >= 1 therefore received no layer-0 in-edge at all and could only be
//! found when it happened to be the entry point.  Distance to a point's own
//! vector is zero, so failing to return it means the graph has no path to it;
//! raising `ef` does not compensate, because there is no edge to traverse.
//!
//! These are construction bugs: they reproduce with sequential `insert_data` on
//! a single thread and need no `parallel_insert`.

use anndists::dist::*;
use hnsw_rs::prelude::*;

#[allow(unused)]
fn log_init_test() {
    let _ = env_logger::builder().is_test(true).try_init();
}

/// Deterministic unit vector, so a failure can be replayed from its seed.
fn unit_vector(seed: u64, dim: usize) -> Vec<f32> {
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

/// Builds an index by sequential insertion and returns the origin ids that a
/// search for their own vector fails to bring back.
fn unreachable_points(nb_point: usize, max_nb_connection: usize, dim: usize) -> Vec<usize> {
    let mut hnsw =
        Hnsw::<f32, DistCosine>::new(max_nb_connection, nb_point, 16, 100, DistCosine {});
    let points: Vec<Vec<f32>> = (0..nb_point)
        .map(|i| unit_vector(i as u64 * 7919, dim))
        .collect();
    for (i, point) in points.iter().enumerate() {
        hnsw.insert_data(point, i);
    }
    assert_eq!(hnsw.get_nb_point(), nb_point);

    // The probe is deliberately generous: with k and ef this far above what a
    // correct graph needs, a point missing from the result is unreachable, not
    // merely crowded out by near neighbours.  A tight probe (k=10, ef=64)
    // conflates the two on dense data.
    points
        .iter()
        .enumerate()
        .filter(|(i, point)| !hnsw.search(point, 200, 1024).iter().any(|n| n.d_id == *i))
        .map(|(i, _)| i)
        .collect()
}

#[test]
fn every_point_is_reachable_from_its_own_vector() {
    log_init_test();
    let orphans = unreachable_points(3_000, 16, 128);
    assert!(
        orphans.is_empty(),
        "{} of 3000 points are not returned by a search for their own vector \
         (the graph holds no in-edge to them): {:?}",
        orphans.len(),
        &orphans[..orphans.len().min(10)]
    );
}

/// The number of layers drives how many points are drawn at level >= 1, and so
/// how many were affected: smaller `max_nb_connection` means a larger level
/// scale (`1/ln(max_nb_connection)`) and a taller graph.  Sweeping it exercises
/// several genuinely different layer distributions rather than one.
#[test]
fn reachability_holds_across_connectivity_settings() {
    log_init_test();
    let mut report = Vec::new();
    for max_nb_connection in [6usize, 8, 16, 32] {
        let orphans = unreachable_points(3_000, max_nb_connection, 64);
        if !orphans.is_empty() {
            report.push(format!(
                "max_nb_connection = {max_nb_connection}: {} unreachable",
                orphans.len()
            ));
        }
    }
    assert!(
        report.is_empty(),
        "points unreachable from their own vector:\n  {}",
        report.join("\n  ")
    );
}
