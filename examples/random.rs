extern crate hnsw_rs;


use std::time::{Duration, SystemTime};
use cpu_time::ProcessTime;
use rand::distributions::{Uniform};
use rand::prelude::*;

use hnsw_rs::prelude::*;

//use crate::point::*;

use hnsw_rs::test::*;



fn main() {
    env_logger::Builder::from_default_env().init();
    //
    let nb_elem = 500000;
    let dim = 25;
    let data = gen_random_matrix_f32(dim, nb_elem);
    let data_with_id = data.iter().zip(0..data.len()).collect();

    let ef_c = 200;
    let max_nb_connection = 15;
    let nb_layer = 16.min((nb_elem as f32).ln().trunc() as usize);
    let hns = Hnsw::<f32, DistL2>::new(max_nb_connection, nb_elem, nb_layer, ef_c, DistL2{});
    let mut start = ProcessTime::now();
    let mut begin_t = SystemTime::now();
    hns.parallel_insert(&data_with_id);
    let mut cpu_time: Duration = start.elapsed();
    println!(" hnsw data insertion  cpu time {:?}", cpu_time); 
    println!(" hnsw data insertion parallel,   system time {:?} \n", begin_t.elapsed().unwrap()); 
    hns.dump_layer_info();
    println!(" parallel hnsw data nb point inserted {:?}", hns.get_nb_point());
    //
    // serial insertion
    //
    let hns = Hnsw::<f32, DistL2>::new(max_nb_connection, nb_elem, nb_layer, ef_c, DistL2{});
    start = ProcessTime::now();
    begin_t = SystemTime::now();
    for _i in 0..data_with_id.len() {
        hns.insert(data_with_id[_i]);
    }
    cpu_time = start.elapsed();
    println!("\n\n serial hnsw data insertion {:?}", cpu_time); 
    println!(" hnsw data insertion serial,  system time {:?}", begin_t.elapsed().unwrap()); 
    hns.dump_layer_info();
    println!(" serial hnsw data nb point inserted {:?}", hns.get_nb_point());

    let ef_search = max_nb_connection * 2;
    let knbn = 10;
    //
    for _iter in 0..100 {
        let mut r_vec = Vec::<f32>::with_capacity(dim);
        let mut rng = thread_rng();
        let unif = Uniform::<f32>::new(0.,1.);
        for _ in 0..dim {
            r_vec.push(rng.sample(unif));
        }
        //
        let _neighbours = hns.search(&r_vec, knbn, ef_search);
    }

}