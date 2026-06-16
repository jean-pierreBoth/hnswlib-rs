/*
 check retrieve of data with query being the inserted data

 Notes:
  - For a fair parallel/serial comparison, both indexes must use the same parameters,
    including modify_level_scale when it is used.
  - After building the serial index, search must be recomputed on the serial index.
  - Long neighbour dumps for missing self-retrieval are logged at debug level.
  - Info level only prints compact summaries.
*/

use anndists::dist::*;
use cpu_time::ProcessTime;

use hnsw_rs::prelude::*;
use rand::distr::{Distribution, Uniform};

#[derive(Debug, Clone)]
struct SearchStats {
    nb_found: usize,
    #[allow(unused)]
    nb_dist_equal: usize,
    missing_ids: Vec<usize>,
}

fn report_search_results<T: std::fmt::Debug>(
    label: &str,
    data_neighbours: &[Vec<Neighbour>],
    epsil: f32,
) -> SearchStats {
    let mut nb_found = 0usize;
    let mut nb_dist_equal = 0usize;
    let mut missing_ids = Vec::new();

    for (id, neighbours) in data_neighbours.iter().enumerate() {
        let mut found_id = false;

        for neighbour in neighbours {
            if neighbour.get_distance() <= epsil {
                nb_dist_equal += 1;
            }

            if neighbour.get_origin_id() == id {
                found_id = true;
                nb_found += 1;
            }
        }

        if !found_id {
            missing_ids.push(id);
        }
    }

    log::info!(
        "{} : nb_found = {}, nb_dist_equal = {:.5e}, missing_count = {}",
        label,
        nb_found,
        nb_dist_equal,
        missing_ids.len()
    );

    if !missing_ids.is_empty() {
        log::info!("{} : missing_ids = {:?}", label, missing_ids);
    }

    for id in &missing_ids {
        log::debug!(
            "{} missing id = {}, neighbours = {:?}",
            label,
            id,
            data_neighbours[*id]
        );
    }

    SearchStats {
        nb_found,
        nb_dist_equal,
        missing_ids,
    }
}

#[test]
fn test_equality_float() {
    let _ = env_logger::builder().is_test(true).try_init();

    println!("\n\n test_equality_float");

    let mut rng = rand::rng();
    let unif = Uniform::<f32>::new(0., 1.).unwrap();

    let nbdata = 10000;
    let dim = 10;

    let mut datas: Vec<Vec<f32>> = Vec::with_capacity(nbdata);

    for _ in 0..nbdata {
        datas.push(
            (0..dim)
                .map(|_| unif.sample(&mut rng))
                .collect::<Vec<f32>>(),
        );
    }

    let data_refs: Vec<(&Vec<f32>, usize)> = (0..nbdata).map(|j| (&datas[j], j)).collect();

    let ef_construct = 128;
    let nb_connection = 32;
    let max_layer = 16;
    let search_ef = 1024;
    let search_k = 16;
    let epsil = 1.0E-5;

    //
    // parallel insertion
    //
    log::info!("running DistL1 parallel insertion");

    let _start = ProcessTime::now();

    let mut hns =
        Hnsw::<f32, DistL1>::new(nb_connection, nbdata, max_layer, ef_construct, DistL1 {});

    hns.modify_level_scale(0.5);
    hns.parallel_insert(&data_refs);
    hns.dump_layer_info();

    let data_neighbours = hns.parallel_search(&datas, search_k, search_ef);
    assert_eq!(data_neighbours.len(), nbdata);

    let parallel_stats =
        report_search_results::<f32>("DistL1 parallel insertion", &data_neighbours, epsil);

    //
    // serial insertion
    //
    log::info!("running DistL1 serial insertion");

    let mut hns =
        Hnsw::<f32, DistL1>::new(nb_connection, nbdata, max_layer, ef_construct, DistL1 {});

    hns.modify_level_scale(0.5);

    for data in data_refs {
        hns.insert((data.0, data.1));
    }

    let data_neighbours = hns.parallel_search(&datas, search_k, search_ef);
    assert_eq!(data_neighbours.len(), nbdata);

    let serial_stats =
        report_search_results::<f32>("DistL1 serial insertion", &data_neighbours, epsil);

    log::info!(
        "DistL1 comparison : parallel_found = {}, serial_found = {}, parallel_missing = {}, serial_missing = {}",
        parallel_stats.nb_found,
        serial_stats.nb_found,
        parallel_stats.missing_ids.len(),
        serial_stats.missing_ids.len()
    );
}

#[test]
fn test_equality_cosine() {
    let _ = env_logger::builder().is_test(true).try_init();

    println!("\n\n test_equality_cosine");

    let mut rng = rand::rng();
    let unif = Uniform::<f32>::new(0., 1.).unwrap();

    let nbdata = 10000;
    let dim = 10;

    let mut datas: Vec<Vec<f32>> = Vec::with_capacity(nbdata);

    for _ in 0..nbdata {
        datas.push(
            (0..dim)
                .map(|_| unif.sample(&mut rng))
                .collect::<Vec<f32>>(),
        );
    }

    let data_refs: Vec<(&Vec<f32>, usize)> = (0..nbdata).map(|j| (&datas[j], j)).collect();

    let ef_construct = 128;
    let nb_connection = 32;
    let max_layer = 16;
    let search_ef = 1024;
    let search_k = 16;
    let epsil = 1.0E-5;

    //
    // parallel insertion
    //
    log::info!("running DistCosine parallel insertion");

    let _start = ProcessTime::now();

    let mut hns = Hnsw::<f32, DistCosine>::new(
        nb_connection,
        nbdata,
        max_layer,
        ef_construct,
        DistCosine {},
    );

    hns.modify_level_scale(0.5);
    hns.parallel_insert(&data_refs);
    hns.dump_layer_info();

    let data_neighbours = hns.parallel_search(&datas, search_k, search_ef);
    assert_eq!(data_neighbours.len(), nbdata);

    let parallel_stats =
        report_search_results::<f32>("DistCosine parallel insertion", &data_neighbours, epsil);

    //
    // serial insertion
    //
    log::info!("running DistCosine serial insertion");

    let mut hns = Hnsw::<f32, DistCosine>::new(
        nb_connection,
        nbdata,
        max_layer,
        ef_construct,
        DistCosine {},
    );

    hns.modify_level_scale(0.5);

    for data in data_refs {
        hns.insert((data.0, data.1));
    }

    let data_neighbours = hns.parallel_search(&datas, search_k, search_ef);
    assert_eq!(data_neighbours.len(), nbdata);

    let serial_stats =
        report_search_results::<f32>("DistCosine serial insertion", &data_neighbours, epsil);

    log::info!(
        "DistCosine comparison : parallel_found = {}, serial_found = {}, parallel_missing = {}, serial_missing = {}",
        parallel_stats.nb_found,
        serial_stats.nb_found,
        parallel_stats.missing_ids.len(),
        serial_stats.missing_ids.len()
    );
}

#[test]
fn test_equality_int() {
    let _ = env_logger::builder().is_test(true).try_init();

    println!("\n\n test_equality_int");

    let mut rng = rand::rng();
    let unif = Uniform::<u32>::new(0, 1000).unwrap();

    let nbdata = 10000;
    let dim = 10;

    let mut datas: Vec<Vec<u32>> = Vec::with_capacity(nbdata);

    for _ in 0..nbdata {
        datas.push(
            (0..dim)
                .map(|_| unif.sample(&mut rng))
                .collect::<Vec<u32>>(),
        );
    }

    let data_refs: Vec<(&Vec<u32>, usize)> = (0..nbdata).map(|j| (&datas[j], j)).collect();

    let ef_construct = 128;
    let nb_connection = 32;
    let max_layer = 16;
    let search_ef = 1024;
    let search_k = 16;
    let epsil = 1.0E-5;

    //
    // parallel insertion
    //
    log::info!("running DistJaccard parallel insertion");

    let _start = ProcessTime::now();

    let hns = Hnsw::<u32, DistJaccard>::new(
        nb_connection,
        nbdata,
        max_layer,
        ef_construct,
        DistJaccard {},
    );

    hns.parallel_insert(&data_refs);
    hns.dump_layer_info();

    let data_neighbours = hns.parallel_search(&datas, search_k, search_ef);
    assert_eq!(data_neighbours.len(), nbdata);

    let parallel_stats =
        report_search_results::<u32>("DistJaccard parallel insertion", &data_neighbours, epsil);

    //
    // serial insertion
    //
    log::info!("running DistJaccard serial insertion");

    let hns = Hnsw::<u32, DistJaccard>::new(
        nb_connection,
        nbdata,
        max_layer,
        ef_construct,
        DistJaccard {},
    );

    for data in data_refs {
        hns.insert((data.0, data.1));
    }

    let data_neighbours = hns.parallel_search(&datas, search_k, search_ef);
    assert_eq!(data_neighbours.len(), nbdata);

    let serial_stats =
        report_search_results::<u32>("DistJaccard serial insertion", &data_neighbours, epsil);

    log::info!(
        "DistJaccard comparison : parallel_found = {}, serial_found = {}, parallel_missing = {}, serial_missing = {}",
        parallel_stats.nb_found,
        serial_stats.nb_found,
        parallel_stats.missing_ids.len(),
        serial_stats.missing_ids.len()
    );
}
