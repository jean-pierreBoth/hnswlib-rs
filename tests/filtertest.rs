
use hnsw_rs::prelude::*;
use hnsw_rs::dist::{DistLevenshtein};
use rand::Rng;
use std::iter;

// Shows two ways to do filtering, by a sorted vector or with a closure
// We define a hnsw-index with 500 entries
// Only ids within 300-400 should be in the result-set 

// Used to create a random string
fn generate_random_string(len: usize) -> String {
    const CHARSET: &[u8] = b"abcdefghij";
    let mut rng = rand::thread_rng();
    let one_char = || CHARSET[rng.gen_range(0..CHARSET.len())] as char;
    iter::repeat_with(one_char).take(len).collect()
}

// this function uses a sorted vector as a filter
fn search_closure_filter(word: &str, hns: &Hnsw<u16, DistLevenshtein>, words: &Vec<String>, filter_vector : &Vec<usize>) {
    
    // transform string to u16 values
    let vec : Vec<u16> = word.chars().map(|c| c as u16).collect();
    // now create a closure using this filter_vector
    // here we can off course implement more advanced filter logic
    let filter = |id: &usize| -> bool {
        match filter_vector.binary_search(id) {
            Ok(_) => true,
            Err(_) => false,
        }
    };
    
    // Now let us do the search by using the defined clojure, which in turn uses our vector
    // ids not in the vector will not be indluced in the search results
    println!("========== Search with closure filter"); 
    let ef_search = 30;
    let res = hns.search_possible_filter(&vec, 10, ef_search, Some(&filter));
    for r in res {
        println!("Word: {:?} Id: {:?} Distance: {:?}", words[r.d_id], r.d_id, r.distance);
    }
}



// this function uses a sorted vector as a filter
fn search_vector_filter(word: &str, hns: &Hnsw<u16, DistLevenshtein>, words: &Vec<String>, filter : &Vec<usize>) -> Vec<Neighbour> {
    
    // transform string to u16 values
    let vec : Vec<u16> = word.chars().map(|c| c as u16).collect();

    let ef_search = 30;
    // Then do a "normal search without filter"
    println!("========== Search without filter");   
    let res3 = hns.search_possible_filter(&vec, 10, ef_search, None);
    for r in res3 {
        println!("Word: {:?} Id: {:?} Distance: {:?}", words[r.d_id], r.d_id, r.distance);
    }

    // Now let us do the search with the vector filter
    // ids not in the vector will not be indluced in the search results
    println!("========== Search with vector filter"); 
    let res = hns.search_possible_filter(&vec, 10, ef_search, Some(filter));

    for r in &res {
        println!("Word: {:?} Id: {:?} Distance: {:?}", words[r.d_id], r.d_id, r.distance);
    }
    //
    return res;
} // end of search_vector_filter


#[test]
fn levenstein() {

    let nb_elem = 500000; // number of possible words in the dictionary
    let max_nb_connection = 15;
    let nb_layer = 16.min((nb_elem as f32).ln().trunc() as usize);
    let ef_c = 200;
    let hns = Hnsw::<u16, DistLevenshtein>::new(max_nb_connection, nb_elem, nb_layer, ef_c, DistLevenshtein{});
    let mut words = vec![];
    for _n in 1..1000 {
        let tw = generate_random_string(8);
        words.push(tw);
    }

    let mut i = 0;
    for w in &words {
        let vec : Vec<u16> = w.chars().map(|c| c as u16).collect();
        hns.insert((&vec, i));
        i = i + 1;
        if i % 1000 == 0 {
            println!("Inserting: {:?}", i);
        }
    }
    // Create a sorted vector of ids
    // the ids in the vector will be used as a filter
    let filtered_hns = Hnsw::<u16, DistLevenshtein>::new(max_nb_connection, nb_elem, nb_layer, ef_c, DistLevenshtein{});
    let mut filter_vector: Vec<usize> = Vec::new();
    for i in 300..400 {
        filter_vector.push(i);
        let v : Vec<u16> = words[i].chars().map(|c| c as u16).collect();
        filtered_hns.insert((&v, i));
    }
    //
    println!("========== Search in filtered_hns" );
    let ef_search = 30;
    let tosearch = "abcdefg";
    let vec : Vec<u16> = tosearch.chars().map(|c| c as u16).collect();
    let res: Vec<Neighbour> = filtered_hns.search_possible_filter(&vec, 10, ef_search, None);
    for r in &res {
        println!("Word: {:?} Id: {:?} Distance: {:?}", words[r.d_id], r.d_id, r.distance);
    }
    //
    // search with filter
    let tosearch = "abcdefg";
    search_closure_filter(tosearch, &hns, &words, &filter_vector);
    let filter_vec_res = search_vector_filter(tosearch, &hns, &words, &filter_vector);
    // how many neighbours in res are in filter_vec_res
    let mut nb_found : usize = 0; 
    for n in &res {
        let found = filter_vec_res.iter().find(|&&m| m.d_id == n.d_id);
        if found.is_some() {
            nb_found += 1;
            assert_eq!(n.distance, found.unwrap().distance);
        }
    }    
    println!(" recall : {}", nb_found as f32/res.len() as f32);
    println!(" last distances ratio : {} ", res.last().unwrap().distance / filter_vec_res.last().unwrap().distance);

}