use std::time::{Duration, SystemTime};
use cpu_time::ProcessTime;
// search in serial mode
//  max_nb_conn   ef_cons    ef_search   scale_factor    extend  keep pruned  recall        req/s      last ratio
//
//     12           400         12           1              0          0        0.917        6486       1.005
//     24           400         24           1              1          0        0.9779       3456       1.001

// parallel mode
//  max_nb_conn   ef_cons    ef_search   scale_factor    extend  keep pruned  recall        req/s      last ratio
//     24           400         24           1              0          0        0.977        12566       1.001
//     24           400         12           1              0          0        0.947        18425       1.003


use hnsw_rs::prelude::*;

mod annhdf5;
use annhdf5::*;


pub fn main() {
    let mut parallel = true;
    //
    let fname = String::from("/home.2/Data/ANN/fashion-mnist-784-euclidean.hdf5");
    println!("\n\n test_load_hdf5 {:?}", fname);
    // now recall that data are stored in row order.
    let anndata = AnnBenchmarkData::new(fname).unwrap();
    // run bench
    let nb_elem = anndata.train_data.len();
    let max_nb_connection = 24;
    let nb_layer = 16.min((nb_elem as f32).ln().trunc() as usize);
    let ef_c = 400;
    println!(" number of elements to insert {:?} , setting max nb layer to {:?} ef_construction {:?}", nb_elem, nb_layer, ef_c);
    println!(" =====================================================================================");
    let nb_search = anndata.test_data.len();
    println!(" number of search {:?}", nb_search);

    let mut hnsw =  Hnsw::<f32, DistL2>::new(max_nb_connection, nb_elem, nb_layer, ef_c, DistL2{});
    hnsw.set_extend_candidates(false);
    // parallel insertion
    let mut start = ProcessTime::now();
    let mut now = SystemTime::now();
    let data_for_par_insertion = anndata.train_data.iter().map( |x| (&x.0, x.1)).collect();
    if parallel {
        println!(" \n parallel insertion");
        hnsw.parallel_insert(&data_for_par_insertion);
    }
    else {
        println!(" \n serial insertion");
        for d in data_for_par_insertion {
            hnsw.insert(d);
        }
    }
    let mut cpu_time: Duration = start.elapsed();
    //
    println!("\n hnsw data insertion cpu time  {:?}  system time {:?} ", cpu_time, now.elapsed()); 
    hnsw.dump_layer_info();
    println!(" hnsw data nb point inserted {:?}", hnsw.get_nb_point());
    //
    //  Now the bench with 10 neighbours
    //
    let mut recalls = Vec::<usize>::with_capacity(nb_elem);
    let mut nb_returned = Vec::<usize>::with_capacity(nb_elem);
    let mut last_distances_ratio = Vec::<f32>::with_capacity(nb_elem);
    let mut knn_neighbours_for_tests = Vec::<Vec<Neighbour>>::with_capacity(nb_elem);
    hnsw.set_searching_mode(true);
    let knbn = 10;
    let ef_c = max_nb_connection;
    println!("\n searching with ef : {:?}", ef_c);
    start = ProcessTime::now();
    now = SystemTime::now();
    // search
    parallel = true;
    if parallel {
        println!(" \n parallel search");
        knn_neighbours_for_tests = hnsw.parallel_search(&anndata.test_data, knbn, ef_c);
    } else {
        println!(" \n serial search");
        for i in 0..anndata.test_data.len() {
          let knn_neighbours : Vec<Neighbour> = hnsw.search(&anndata.test_data[i], knbn, ef_c);
          knn_neighbours_for_tests.push(knn_neighbours);
        }
    }
    cpu_time = start.elapsed();
    let search_sys_time = now.elapsed().unwrap().as_micros() as f32;
    let search_cpu_time = cpu_time.as_micros() as f32;
    println!("total cpu time for search requests {:?} , system time {:?} ", search_cpu_time, search_sys_time);
    // now compute recall rate
    for i in 0..anndata.test_data.len() {
        let true_distances = anndata.test_distances.row(i);
        let max_dist = true_distances[knbn-1];
        let mut _knn_neighbours_id : Vec<usize> = knn_neighbours_for_tests[i].iter().map(|p| p.d_id).collect();
        let knn_neighbours_dist : Vec<f32> = knn_neighbours_for_tests[i].iter().map(|p| p.distance).collect();
        nb_returned.push(knn_neighbours_dist.len());
        // count how many distances of knn_neighbours_dist are less than
        let recall = knn_neighbours_dist.iter().filter(|x| *x <= &max_dist).count();
        recalls.push(recall);
        let mut ratio = 0.;
        if knn_neighbours_dist.len() >= 1 {
            ratio = knn_neighbours_dist[knn_neighbours_dist.len()-1]/max_dist;
        }
        last_distances_ratio.push(ratio);
    }
    let mean_recall = (recalls.iter().sum::<usize>() as f32)/((knbn * recalls.len()) as f32);
    println!("\n mean fraction nb returned by search {:?} ", (nb_returned.iter().sum::<usize>() as f32)/ ((nb_returned.len() * knbn) as f32));
    println!("\n last distances ratio {:?} ", last_distances_ratio.iter().sum::<f32>() / last_distances_ratio.len() as f32);
    println!("\n recall rate for {:?} is {:?} , nb req /s {:?}", anndata.fname, mean_recall, (nb_search as f32)*1.0e+6_f32/search_sys_time);
}
