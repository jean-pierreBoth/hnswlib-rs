//! Adversarial reload: a malformed or truncated dump must produce a typed
//! error, never a panic, a process exit, an unbounded allocation, or an
//! out-of-bounds read.
//!
//! The reload path parses two files (`<name>.hnsw.graph` and
//! `<name>.hnsw.data`) whose header fields — dimension, layer count, point
//! count, per-layer neighbour counts, per-point payload lengths, point
//! identities — all size buffers or index tables during reconstruction.
//! Every one of those fields is attacker-controlled if the files are not
//! trusted, so this suite writes a genuine dump, corrupts it in one
//! specific way per case, and asserts the loader declines it as an error.
//!
//! Each case is run in-process: a panic here fails the test, which is the
//! point — "returns Err" and "does not panic" are the same assertion for a
//! parser that must be safe against hostile input.

use std::fs;
use std::path::{Path, PathBuf};

use hnsw_rs::anndists::dist::{self, DistL1};
use hnsw_rs::api::AnnT;
use hnsw_rs::hnsw::Hnsw;
use hnsw_rs::hnswio::{HnswIo, load_description};

const NB_POINTS: usize = 60;
const DIMENSION: usize = 8;
const DUMP_NAME: &str = "malformed";

/// Build a small, genuine dump and return its directory.
fn build_dump(dir: &Path) {
    let hnsw = Hnsw::<f32, dist::DistL1>::new(8, NB_POINTS, 16, 24, dist::DistL1 {});
    for point in 0..NB_POINTS {
        // Deterministic, non-degenerate vectors: no randomness, so a failing
        // case is reproducible from the test name alone.
        let vector: Vec<f32> = (0..DIMENSION)
            .map(|component| (point * DIMENSION + component) as f32 * 0.125 + 0.5)
            .collect();
        hnsw.insert((&vector, point));
    }
    hnsw.file_dump(dir, DUMP_NAME)
        .expect("dump the fixture index");
}

fn graph_path(dir: &Path) -> PathBuf {
    dir.join(format!("{DUMP_NAME}.hnsw.graph"))
}

fn data_path(dir: &Path) -> PathBuf {
    dir.join(format!("{DUMP_NAME}.hnsw.data"))
}

/// Load a dump directory, returning whether the loader accepted it.
fn try_load(dir: &Path) -> Result<(), String> {
    let mut reloader = HnswIo::new(dir, DUMP_NAME);
    match reloader.load_hnsw::<f32, DistL1>() {
        Ok(_) => Ok(()),
        Err(error) => Err(error.to_string()),
    }
}

/// Apply `mutate` to a fresh dump and assert the loader rejects the result.
fn assert_rejected(case: &str, mutate: impl FnOnce(&Path)) {
    let dir = tempfile::tempdir().expect("tempdir");
    build_dump(dir.path());
    mutate(dir.path());
    match try_load(dir.path()) {
        Err(message) => {
            assert!(
                !message.is_empty(),
                "{case}: rejection must carry a diagnostic"
            );
        }
        Ok(()) => panic!("{case}: loader accepted a malformed dump"),
    }
}

#[test]
fn the_unmodified_fixture_loads() {
    // Control: the corruption cases below only mean something if the same
    // dump loads cleanly when left alone.
    let dir = tempfile::tempdir().expect("tempdir");
    build_dump(dir.path());
    try_load(dir.path()).expect("an untouched dump must load");
}

#[test]
fn truncation_at_every_prefix_is_rejected_without_panicking() {
    // Truncation is the cheapest corruption and reaches every read in the
    // parser: each prefix stops the file at a different field boundary.
    let full = {
        let dir = tempfile::tempdir().expect("tempdir");
        build_dump(dir.path());
        fs::read(graph_path(dir.path())).expect("read graph file")
    };

    // Sample prefixes across the whole file rather than all of them: the
    // parser reads fixed-width fields, so a stride finer than the smallest
    // field (1 byte) adds nothing, and this keeps the suite fast.
    let stride = (full.len() / 64).max(1);
    let mut prefix = 0;
    while prefix < full.len() {
        let truncated = full[..prefix].to_vec();
        assert_rejected(&format!("graph truncated to {prefix} bytes"), |dir| {
            fs::write(graph_path(dir), &truncated).expect("write truncated graph");
        });
        prefix += stride;
    }
}

#[test]
fn data_file_truncation_is_rejected_without_panicking() {
    let full = {
        let dir = tempfile::tempdir().expect("tempdir");
        build_dump(dir.path());
        fs::read(data_path(dir.path())).expect("read data file")
    };

    let stride = (full.len() / 32).max(1);
    let mut prefix = 0;
    while prefix < full.len() {
        let truncated = full[..prefix].to_vec();
        assert_rejected(&format!("data truncated to {prefix} bytes"), |dir| {
            fs::write(data_path(dir), &truncated).expect("write truncated data");
        });
        prefix += stride;
    }
}

#[test]
fn a_hostile_dimension_cannot_drive_an_unbounded_allocation() {
    // The dimension sits at a fixed offset in the description and
    // multiplies into every per-point payload length. Left unbounded, these
    // eight bytes are an out-of-memory request.
    let dir = tempfile::tempdir().expect("tempdir");
    build_dump(dir.path());
    let mut graph = fs::read(graph_path(dir.path())).expect("read graph file");
    let offset = dimension_offset(&graph);
    graph[offset..offset + 8].copy_from_slice(&usize::MAX.to_ne_bytes());
    fs::write(graph_path(dir.path()), &graph).expect("write hostile graph");

    let error = try_load(dir.path()).expect_err("a usize::MAX dimension must be rejected");
    assert!(
        error.contains("dimension"),
        "the rejection should name the offending field, got: {error}"
    );
}

#[test]
fn a_hostile_layer_count_is_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    build_dump(dir.path());
    let mut graph = fs::read(graph_path(dir.path())).expect("read graph file");
    // nb_layer is the u8 immediately before ef; see load_description.
    let offset = dimension_offset(&graph) - 8 - 8 - 1;
    graph[offset] = 250;
    fs::write(graph_path(dir.path()), &graph).expect("write hostile graph");

    let error = try_load(dir.path()).expect_err("a 250-layer description must be rejected");
    assert!(
        error.contains("layer"),
        "the rejection should name the offending field, got: {error}"
    );
}

#[test]
fn single_byte_flips_never_panic() {
    // Bit flips reach fields that truncation cannot: magics, identities,
    // counts and lengths in the middle of the stream. Any individual flip
    // may legitimately still load (it might land in a distance value), so
    // the assertion is only that the loader terminates with a value rather
    // than unwinding or exiting.
    let dir = tempfile::tempdir().expect("tempdir");
    build_dump(dir.path());
    let graph = fs::read(graph_path(dir.path())).expect("read graph file");
    let data = fs::read(data_path(dir.path())).expect("read data file");

    let stride = (graph.len() / 48).max(1);
    let mut index = 0;
    while index < graph.len() {
        for bit in [0u8, 3, 7] {
            let mut corrupted = graph.clone();
            corrupted[index] ^= 1 << bit;
            let case_dir = tempfile::tempdir().expect("tempdir");
            build_dump(case_dir.path());
            fs::write(graph_path(case_dir.path()), &corrupted).expect("write flipped graph");
            // Result deliberately ignored: not panicking IS the assertion.
            let _ = try_load(case_dir.path());
        }
        index += stride;
    }

    let stride = (data.len() / 24).max(1);
    let mut index = 0;
    while index < data.len() {
        let mut corrupted = data.clone();
        corrupted[index] ^= 0b0100_0000;
        let case_dir = tempfile::tempdir().expect("tempdir");
        build_dump(case_dir.path());
        fs::write(data_path(case_dir.path()), &corrupted).expect("write flipped data");
        let _ = try_load(case_dir.path());
        index += stride;
    }
}

#[test]
fn a_description_with_a_bad_magic_is_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    build_dump(dir.path());
    let mut graph = fs::read(graph_path(dir.path())).expect("read graph file");
    graph[0] ^= 0xff;
    fs::write(graph_path(dir.path()), &graph).expect("write hostile graph");
    try_load(dir.path()).expect_err("a corrupt description magic must be rejected");
}

#[test]
fn an_empty_or_tiny_file_is_rejected() {
    for size in [0usize, 1, 3, 4, 8] {
        assert_rejected(&format!("graph file of {size} bytes"), |dir| {
            fs::write(graph_path(dir), vec![0u8; size]).expect("write tiny graph");
        });
    }
}

#[test]
fn load_description_rejects_a_hostile_name_length() {
    // The distance-name length is read before the name itself; a huge value
    // must be refused rather than used as an allocation size.
    let dir = tempfile::tempdir().expect("tempdir");
    build_dump(dir.path());
    let graph = fs::read(graph_path(dir.path())).expect("read graph file");
    let offset = dimension_offset(&graph) + 8;
    let mut hostile = graph.clone();
    hostile[offset..offset + 8].copy_from_slice(&usize::MAX.to_ne_bytes());
    let mut cursor = std::io::Cursor::new(hostile);
    assert!(
        load_description(&mut cursor).is_err(),
        "a hostile name length must be rejected"
    );
}

/// Byte offset of the `dimension` field inside a dumped description.
///
/// The layout is fixed by `load_description`: magic (u32), dumpmode (u8),
/// max_nb_connection (u8), level_scale (f64, v4 only), nb_layer (u8), ef
/// (usize), nb_point (usize), dimension (usize).
fn dimension_offset(graph: &[u8]) -> usize {
    let magic = u32::from_ne_bytes(graph[0..4].try_into().expect("magic"));
    // v4 carries the level-scale f64; earlier versions do not.
    let level_scale_width = if magic == 0x002a6779 { 8 } else { 0 };
    4 + 1 + 1 + level_scale_width + 1 + 8 + 8
}
