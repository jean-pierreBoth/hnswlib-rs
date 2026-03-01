//! Tests for soft-deletion (mark_deleted) and neighbor repair in HNSW.
//!
//! Run with: cargo test --test deletion_test -- --nocapture

use anndists::dist::*;
use hnsw_rs::prelude::*;
use rand::distr::Uniform;
use rand::prelude::*;

fn gen_random_vectors(dim: usize, count: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::rng();
    let unif = Uniform::<f32>::new(0., 1.).unwrap();
    (0..count)
        .map(|_| (0..dim).map(|_| rng.sample(unif)).collect())
        .collect()
}

#[test]
fn test_basic_deletion() {
    // Insert 100 points, delete one, verify it's excluded from search results
    let dim = 8;
    let max_nb_connection = 16;
    let ef_construction = 200;
    let nb_elem = 100;

    let hnsw = Hnsw::<f32, DistL2>::new(max_nb_connection, nb_elem, 16, ef_construction, DistL2);
    let data = gen_random_vectors(dim, nb_elem);

    for (i, v) in data.iter().enumerate() {
        hnsw.insert((v, i));
    }

    // Search before deletion — origin_id 0 should be findable
    let results_before = hnsw.search(&data[0], 10, 30);
    let found_before = results_before.iter().any(|r| r.d_id == 0);
    assert!(found_before, "Point 0 should be in search results before deletion");

    // Delete point with origin_id 0
    let deleted = hnsw.mark_deleted(0);
    assert!(deleted, "mark_deleted should return true for existing point");
    assert_eq!(hnsw.get_deleted_count(), 1, "Deleted count should be 1");

    // Search after deletion — origin_id 0 should NOT be in results
    let results_after = hnsw.search(&data[0], 10, 30);
    let found_after = results_after.iter().any(|r| r.d_id == 0);
    assert!(!found_after, "Deleted point should NOT appear in search results");
}

#[test]
fn test_delete_nonexistent() {
    let hnsw = Hnsw::<f32, DistL2>::new(16, 100, 16, 200, DistL2);
    let data = gen_random_vectors(8, 10);
    for (i, v) in data.iter().enumerate() {
        hnsw.insert((v, i));
    }

    // Try to delete a point that doesn't exist
    let deleted = hnsw.mark_deleted(999);
    assert!(!deleted, "Deleting nonexistent point should return false");
    assert_eq!(hnsw.get_deleted_count(), 0);
}

#[test]
fn test_double_deletion() {
    let hnsw = Hnsw::<f32, DistL2>::new(16, 100, 16, 200, DistL2);
    let data = gen_random_vectors(8, 10);
    for (i, v) in data.iter().enumerate() {
        hnsw.insert((v, i));
    }

    // Delete same point twice
    assert!(hnsw.mark_deleted(0));
    assert!(hnsw.mark_deleted(0), "Double delete should return true (already deleted)");
    assert_eq!(hnsw.get_deleted_count(), 1, "Count should still be 1 after double delete");
}

#[test]
fn test_delete_multiple_points() {
    let dim = 8;
    let nb_elem = 50;
    let hnsw = Hnsw::<f32, DistL2>::new(16, nb_elem, 16, 200, DistL2);
    let data = gen_random_vectors(dim, nb_elem);

    for (i, v) in data.iter().enumerate() {
        hnsw.insert((v, i));
    }

    // Delete points 0, 10, 20, 30, 40
    for id in [0, 10, 20, 30, 40] {
        assert!(hnsw.mark_deleted(id));
    }
    assert_eq!(hnsw.get_deleted_count(), 5);

    // Search should never return any deleted point
    for query_idx in [0, 10, 20, 30, 40] {
        let results = hnsw.search(&data[query_idx], 10, 30);
        for r in &results {
            assert!(
                ![0, 10, 20, 30, 40].contains(&r.d_id),
                "Deleted point {} found in search results when querying for point {}",
                r.d_id,
                query_idx
            );
        }
    }
}

#[test]
fn test_delete_all_points() {
    let dim = 4;
    let nb_elem = 20;
    let hnsw = Hnsw::<f32, DistL2>::new(8, nb_elem, 16, 50, DistL2);
    let data = gen_random_vectors(dim, nb_elem);

    for (i, v) in data.iter().enumerate() {
        hnsw.insert((v, i));
    }

    // Delete every point
    for i in 0..nb_elem {
        assert!(hnsw.mark_deleted(i));
    }
    assert_eq!(hnsw.get_deleted_count(), nb_elem);

    // Search should return empty
    let results = hnsw.search(&data[0], 10, 30);
    assert!(results.is_empty(), "Search should return empty when all points deleted");
}

#[test]
fn test_higher_layer_deletion() {
    // With enough points, some will be assigned to layers > 0.
    // Use a large dataset to ensure multi-layer structure, then verify
    // we can delete points from any layer.
    let dim = 8;
    let nb_elem = 1000;
    let max_nb_connection = 16;

    let hnsw = Hnsw::<f32, DistL2>::new(max_nb_connection, nb_elem, 16, 200, DistL2);
    let data = gen_random_vectors(dim, nb_elem);

    let data_refs: Vec<(&Vec<f32>, usize)> = data.iter().enumerate().map(|(i, v)| (v, i)).collect();
    hnsw.parallel_insert(&data_refs);

    // Check that we have multiple layers
    let max_layer = hnsw.get_max_level_observed();
    println!("Max layer observed: {}", max_layer);
    assert!(max_layer > 0, "With 1000 points, we should have multiple layers");

    // Delete all points and verify all are found
    let mut deleted_count = 0;
    for i in 0..nb_elem {
        if hnsw.mark_deleted(i) {
            deleted_count += 1;
        }
    }
    assert_eq!(
        deleted_count, nb_elem,
        "Should be able to delete ALL {} points including those in higher layers",
        nb_elem
    );
    assert_eq!(hnsw.get_deleted_count(), nb_elem);
}

#[test]
fn test_search_quality_after_deletion() {
    // Insert clustered data, delete points from one cluster,
    // verify remaining cluster is still searchable with good quality
    let dim = 4;
    let nb_elem = 200;

    let hnsw = Hnsw::<f32, DistL2>::new(16, nb_elem, 16, 200, DistL2);

    // Cluster A: centered around [1,1,1,1], ids 0-99
    // Cluster B: centered around [10,10,10,10], ids 100-199
    let mut rng = rand::rng();
    let noise = Uniform::<f32>::new(-0.1, 0.1).unwrap();
    let mut data = Vec::new();

    for _ in 0..100 {
        let v: Vec<f32> = (0..dim).map(|_| 1.0 + rng.sample(noise)).collect();
        data.push(v);
    }
    for _ in 0..100 {
        let v: Vec<f32> = (0..dim).map(|_| 10.0 + rng.sample(noise)).collect();
        data.push(v);
    }

    for (i, v) in data.iter().enumerate() {
        hnsw.insert((v, i));
    }

    // Delete all of cluster A (ids 0-99)
    for i in 0..100 {
        hnsw.mark_deleted(i);
    }

    // Search for a cluster A point — should only get cluster B results
    let query = vec![1.0, 1.0, 1.0, 1.0];
    let results = hnsw.search(&query, 10, 30);

    for r in &results {
        assert!(
            r.d_id >= 100,
            "After deleting cluster A, result {} should be from cluster B (id >= 100)",
            r.d_id
        );
    }

    // Search for a cluster B point — should still work well
    let query_b = vec![10.0, 10.0, 10.0, 10.0];
    let results_b = hnsw.search(&query_b, 10, 30);
    assert!(!results_b.is_empty(), "Cluster B search should return results");
    for r in &results_b {
        assert!(r.d_id >= 100, "Cluster B results should be from cluster B");
    }
}

#[test]
fn test_deletion_with_cosine_distance() {
    // Verify deletion works with different distance metrics
    let dim = 8;
    let nb_elem = 50;
    let hnsw =
        Hnsw::<f32, DistCosine>::new(16, nb_elem, 16, 200, DistCosine);
    let data = gen_random_vectors(dim, nb_elem);

    for (i, v) in data.iter().enumerate() {
        hnsw.insert((v, i));
    }

    // Delete point 5
    assert!(hnsw.mark_deleted(5));

    // Verify it's not in results
    let results = hnsw.search(&data[5], 10, 30);
    assert!(
        !results.iter().any(|r| r.d_id == 5),
        "Deleted point should not appear in cosine search results"
    );
}

#[test]
fn test_deletion_with_parallel_insert() {
    // Verify deletion works after parallel_insert
    let dim = 16;
    let nb_elem = 500;
    let hnsw = Hnsw::<f32, DistL2>::new(16, nb_elem, 16, 200, DistL2);
    let data = gen_random_vectors(dim, nb_elem);

    let data_refs: Vec<(&Vec<f32>, usize)> = data.iter().enumerate().map(|(i, v)| (v, i)).collect();
    hnsw.parallel_insert(&data_refs);

    // Delete every 5th point
    let deleted_ids: Vec<usize> = (0..nb_elem).step_by(5).collect();
    for &id in &deleted_ids {
        assert!(hnsw.mark_deleted(id), "Failed to delete point {}", id);
    }

    assert_eq!(hnsw.get_deleted_count(), deleted_ids.len());

    // Verify none of the deleted points appear in search results
    for &id in &deleted_ids {
        let results = hnsw.search(&data[id], 20, 50);
        assert!(
            !results.iter().any(|r| deleted_ids.contains(&r.d_id)),
            "Deleted point found in search results when querying point {}",
            id
        );
    }
}

#[test]
fn test_empty_graph_deletion() {
    let hnsw = Hnsw::<f32, DistL2>::new(16, 100, 16, 200, DistL2);
    assert!(!hnsw.mark_deleted(0), "Deleting from empty graph should return false");
    assert_eq!(hnsw.get_deleted_count(), 0);
}

#[test]
fn test_single_point_deletion() {
    let hnsw = Hnsw::<f32, DistL2>::new(16, 10, 16, 200, DistL2);
    let v = vec![1.0, 2.0, 3.0, 4.0];
    hnsw.insert((&v, 0));

    // Search finds it
    let results = hnsw.search(&v, 1, 10);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].d_id, 0);

    // Delete it
    assert!(hnsw.mark_deleted(0));

    // Search returns empty
    let results = hnsw.search(&v, 1, 10);
    assert!(results.is_empty(), "Search should return empty after deleting the only point");
}
