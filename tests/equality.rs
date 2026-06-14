/*
 check retrieve of data with query being the inserted data

 playing with parameters we can see that:
  - parallel (versus serial) insertion can degrade performance. Pb is more acute with DistL1 than DistJaccard
  - degradation reduces when using modify_level_scale
  - BUT it can also occur that parallel insertion has better recall than serial
*/

use anndists::dist::*;
use cpu_time::ProcessTime;

use hnsw_rs::prelude::*;
use rand::{Rng, distr::Uniform};

#[test]
fn test_equality_float() {
    //
    let _ = env_logger::builder().is_test(true).try_init();
    //
    println!("\n\n test_equality_retrieve");
    //
    let mut rng = rand::rng();
    let unif = Uniform::<f32>::new(0., 1.).unwrap();
    let nbdata = 10000;
    let dim = 10;
    let mut datas: Vec<Vec<f32>> = Vec::with_capacity(nbdata);
    //
    for _ in 0..nbdata {
        datas.push((0..dim).map(|_| rng.sample(unif)).collect::<Vec<f32>>());
        // data_refs.push((&datas[j], j));
    }
    let data_refs: Vec<(&Vec<f32>, usize)> = (0..nbdata).map(|j| (&datas[j], j)).collect();
    // insertion
    let ef_construct = 128;
    let nb_connection = 32;
    let _start = ProcessTime::now();
    let mut hns = Hnsw::<f32, DistL1>::new(nb_connection, nbdata, 16, ef_construct, DistL1 {});
    hns.modify_level_scale(0.5);

    // parallel insertion
    hns.parallel_insert(&data_refs);
    hns.dump_layer_info();
    //
    // check how many we retrieve each data in a neighborhood
    //
    let data_neighbours = hns.parallel_search(&datas, 16, 16);
    assert_eq!(data_neighbours.len(), nbdata);

    let epsil = 1.0E-5;
    let mut nb_found = 0usize;
    let mut nb_dist_equal = 0;
    for (id, neighbours) in data_neighbours.iter().enumerate() {
        for neighbour in neighbours {
            if neighbour.get_distance() <= epsil {
                nb_dist_equal += 1;
            }
            if neighbour.get_origin_id() == id {
                nb_found += 1;
            }
        }
    }
    //
    log::info!(
        "DistL1 parallel insertion : nb_found = {}, nb_dist_equal = {:.4e}",
        nb_found,
        nb_dist_equal
    );
    //
    // serial insertion
    //
    log::info!("running serial insertion");
    let hns = Hnsw::<f32, DistL1>::new(nb_connection, nbdata, 16, ef_construct, DistL1 {});
    for data in data_refs {
        hns.insert((data.0, data.1));
    }
    let data_neighbours = hns.parallel_search(&datas, 16, 16);
    assert_eq!(data_neighbours.len(), nbdata);
    let mut nb_found = 0usize;
    let mut nb_dist_equal = 0;
    for (id, neighbours) in data_neighbours.iter().enumerate() {
        for neighbour in neighbours {
            if neighbour.get_distance() <= epsil {
                nb_dist_equal += 1;
                if neighbour.get_origin_id() == id {
                    nb_found += 1;
                }
            }
        }
    }
    //
    log::info!(
        "\n\n DistL1 serial insertion : nb_found = {}, nb_dist_equal = {:.4e}",
        nb_found,
        nb_dist_equal
    );
}

#[test]
fn test_equality_int() {
    //
    let _ = env_logger::builder().is_test(true).try_init();
    //
    println!("\n\n test_equality_retrieve");
    //
    let mut rng = rand::rng();
    let unif = Uniform::<u32>::new(0, 1000).unwrap();
    let nbdata = 10000;
    let dim = 10;
    let mut datas: Vec<Vec<u32>> = Vec::with_capacity(nbdata);
    //
    for _ in 0..nbdata {
        datas.push((0..dim).map(|_| rng.sample(unif)).collect::<Vec<u32>>());
        // data_refs.push((&datas[j], j));
    }
    let data_refs: Vec<(&Vec<u32>, usize)> = (0..nbdata).map(|j| (&datas[j], j)).collect();
    // insertion
    let ef_construct = 32;
    let nb_connection = 16;
    let _start = ProcessTime::now();
    #[allow(unused_mut)]
    let mut hns =
        Hnsw::<u32, DistJaccard>::new(nb_connection, nbdata, 16, ef_construct, DistJaccard {});
    //    hns.modify_level_scale(0.5);
    // parallel insertion
    hns.parallel_insert(&data_refs);
    hns.dump_layer_info();

    //
    // check how many we retrieve each data in a neighborhood
    //
    let data_neighbours = hns.parallel_search(&datas, 16, 16);
    assert_eq!(data_neighbours.len(), nbdata);

    let epsil = 1.0E-5;
    let mut nb_found = 0usize;
    let mut nb_dist_equal = 0;
    for (id, neighbours) in data_neighbours.iter().enumerate() {
        for neighbour in neighbours {
            if neighbour.get_distance() <= epsil {
                nb_dist_equal += 1;
            }
            if neighbour.get_origin_id() == id {
                nb_found += 1;
            }
        }
    }
    //
    log::info!(
        "DistJaccard parallel insertion : nb_found = {}, nb_dist_equal = {:.5e}",
        nb_found,
        nb_dist_equal
    );
    //
    // serial insertion
    //
    log::info!("running serial insertion");
    let hns =
        Hnsw::<u32, DistJaccard>::new(nb_connection, nbdata, 16, ef_construct, DistJaccard {});
    for data in data_refs {
        hns.insert((data.0, data.1));
    }
    let mut nb_found = 0usize;
    let mut nb_dist_equal = 0;
    for (id, neighbours) in data_neighbours.iter().enumerate() {
        for neighbour in neighbours {
            if neighbour.get_distance() <= epsil {
                nb_dist_equal += 1;
                if neighbour.get_origin_id() == id {
                    nb_found += 1;
                }
            }
        }
    }
    //
    log::info!(
        "\n\n DistJaccard serial insertion : nb_found = {}, nb_dist_equal = {:.5e}",
        nb_found,
        nb_dist_equal
    );
}
