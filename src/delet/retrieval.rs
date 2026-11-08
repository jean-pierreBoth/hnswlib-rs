// This is a retrieval test: estimate how many data we will retrieve (at null distance in a search)

#![allow(unused)]

use rand::{SeedableRng, distr::Distribution, distr::Uniform, rngs::Xoshiro256PlusPlus};

use crate::hnsw::*;
use anndists::dist::distances::Distance;

// TODO: returns DataId (as DistDot we get often 1.e8 as first distance)

/// computes fraction retrived, based on sampling_fraction of data
pub fn retrieval_fraction<'b, T: Sized + Clone + Send + Sync + 'b, D: Distance<T> + Send + Sync>(
    hnsw: &Hnsw<T, D>,
    s_fraction: f32,
    block_size: usize,
    ef_search: usize,
) -> f32 {
    //
    let block_size = 5000;
    let knbn = 40;
    //
    let mut nb_retrieved = 0;
    let mut mean_dist: f32 = 0.;
    let mut max_dist: f32 = 0.;
    //
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
    let mut dataids = Vec::<DataId>::with_capacity(block_size);
    //
    for point in d_iter {
        if unif.sample(&mut rng) <= s_fraction {
            nb_sampled += 1;
            let p = point.get_v().to_vec();
            if data_block.len() < block_size {
                data_block.push(p);
                dataids.push(point.get_origin_id());
            } else {
                // block full we search
                log::debug!("searching block");
                let neighbours = hnsw.parallel_search(&data_block, knbn, ef_search);
                //
                for (rank, n) in neighbours.iter().enumerate() {
                    if dataids[rank] == n[0].get_origin_id() {
                        nb_retrieved += 1;
                    }
                    let dist = n[0].get_distance();
                    mean_dist += dist;
                    max_dist = max_dist.max(dist);
                }
                data_block.clear();
                dataids.clear();
                // prepare new block iteration
                dataids.push(point.get_origin_id());
                data_block.push(p);
            }
        }
    }
    // check possible unfilled block
    if !data_block.is_empty() {
        log::info!("residual block size : {}", data_block.len());
        let neighbours = hnsw.parallel_search(&data_block, knbn, ef_search);
        //
        for (rank, n) in neighbours.iter().enumerate() {
            if dataids[rank] == n[0].get_origin_id() {
                nb_retrieved += 1;
            }
            let dist = n[0].get_distance();
            mean_dist += dist;
            max_dist = max_dist.max(dist);
        }
        data_block.clear();
    }
    //
    let retrieved_fraction = nb_retrieved as f32 / nb_sampled as f32;
    log::info!(
        "fraction retrived : {:.2e}, nb_sampled : {}, mean_dist : {:.3e}, max_dist : {:.3e}",
        retrieved_fraction,
        nb_sampled,
        mean_dist / nb_sampled as f32,
        max_dist
    );
    //
    retrieved_fraction
}

#[cfg(test)]
mod tests {
    use super::*;
    use anndists::dist;
    use cpu_time::ProcessTime;
    use dist::l2_normalize;
    use log::error;
    use rand::prelude::*;
    use std::time::Duration;

    use rand::distr::{Distribution, Uniform};

    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    /// return nbcolumn vectors of dimension nbrow
    fn gen_random_matrix_f32(nbrow: usize, nbcolumn: usize) -> Vec<Vec<f32>> {
        let mut rng = rand::rng();
        let unif = Uniform::<f32>::new(0., 1.).unwrap();
        let mut data = Vec::with_capacity(nbcolumn);
        for _ in 0..nbcolumn {
            let column = (0..nbrow).map(|_| rng.sample(unif)).collect::<Vec<f32>>();
            data.push(column);
        }
        data
    }

    #[test]
    fn retrieval_random() {
        //
        log_init_test();
        //
        let nb_elem = 21000;
        let dim = 25;
        let knbn = 10;
        let ef_c = 800;
        let max_nb_connection = 48;
        let ef = 128;
        //
        //
        let mut data = gen_random_matrix_f32(dim, nb_elem);
        for v in &mut data {
            l2_normalize(v);
        }
        let data_with_id = data.iter().zip(0..data.len()).collect::<Vec<_>>();
        let nb_layer = 16.min((nb_elem as f32).ln().trunc() as usize);
        let mut hnsw = Hnsw::<f32, dist::DistDot>::new(
            max_nb_connection,
            nb_elem,
            nb_layer,
            ef_c,
            dist::DistDot {},
        );
        // !
        // hns.
        (true);
        let mut start = ProcessTime::now();
        let now = std::time::SystemTime::now();
        // parallel insertion
        hnsw.parallel_insert(&data_with_id);
        let mut cpu_time: Duration = start.elapsed();
        println!(
            "\n hnsw data parallel insertion cpu time {:?} , system time {:?}",
            cpu_time,
            now.elapsed()
        );
        // now we retrieve
        let retrieval_rate = retrieval_fraction(&hnsw, 0.9, 1000, 128);
        log::info!("retrieval_random, retrieval rate : {:.3e}", retrieval_rate);
    }
} // end test module
