// This is a retrieval test: estimate how many data we will retrieve (at null distance in a search)

#![allow(unused)]

use rand::{SeedableRng, distr::Distribution, distr::Uniform, rngs::Xoshiro256PlusPlus};

use crate::hnsw::*;
use anndists::dist::distances::Distance;

// TODO: parametrize ef_search

/// computes fraction retrived, based on sampling_fraction of data
pub fn retrieval_fraction<'b, T: Sized + Clone + Send + Sync + 'b, D: Distance<T> + Send + Sync>(
    hnsw: &Hnsw<T, D>,
    s_fraction: f32,
    block_size: usize,
) -> f32 {
    //
    let block_size = 5000;
    let knbn = 10;
    let ef_search = 32;
    let mut nb_retrieved = 0;
    let nb_to_sample = (hnsw.get_nb_point() as f32 * s_fraction).floor() as usize;
    let mut nb_sampled = 0;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(773);
    // iterates through data, selecting point by block
    let point_indexation = hnsw.get_point_indexation();
    let mut d_iter = IterPoint::new(point_indexation);
    let nbrow = 10;
    let unif = Uniform::<f32>::new(0., 1.).unwrap();
    // search
    let mut data_block = Vec::<Vec<T>>::with_capacity(block_size);
    for point in d_iter {
        if unif.sample(&mut rng) <= s_fraction {
            nb_sampled += 1;
            let p = point.get_v().to_vec();
            if data_block.len() < block_size {
                data_block.push(p);
            } else {
                // block full we search
                log::debug!("searching block");
                let neighbours = hnsw.parallel_search(&data_block, knbn, ef_search);
                //
                nb_retrieved += neighbours.iter().filter(|v| (*v)[0].distance <= 0.).count();
                data_block.clear();
                data_block.push(p);
            }
        }
    }
    // check possible unfilled block
    if !data_block.is_empty() {
        let neighbours = hnsw.parallel_search(&data_block, knbn, ef_search);
        //
        nb_retrieved += neighbours.iter().filter(|v| (*v)[0].distance <= 0.).count();
        data_block.clear();
    }
    //
    let retrieved_fraction = nb_retrieved as f32 / nb_sampled as f32;
    log::info!("fraction retrived : {:.2e}", retrieved_fraction);
    //
    retrieved_fraction
}
