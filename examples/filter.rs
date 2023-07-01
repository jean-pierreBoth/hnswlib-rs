
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
fn search_closure_filter(word: &str, hns: &Hnsw<u16, DistLevenshtein>, words: &Vec<String>) {
    
    // transform string to u16 values
    let mut vec: Vec<u16> = Vec::new();
    for c in word.chars() {
        vec.push(c as u16);
    }

    // Create a sorted vector of ids
    // the ids in the vector will be used as a filter
    let mut filter_vector: Vec<usize> = Vec::new();
    for i in 300..400 {
        filter_vector.push(i);
    }

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
fn search_vector_filter(word: &str, hns: &Hnsw<u16, DistLevenshtein>, words: &Vec<String>) {
    
    // transform string to u16 values
    let mut vec: Vec<u16> = Vec::new();
    for c in word.chars() {
        vec.push(c as u16);
    }

    // Create a sorted vector of ids
    // the ids in the vector will be used as a filter
    let mut filter: Vec<usize> = Vec::new();
    for i in 300..400 {
        filter.push(i);
    }

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
    let res = hns.search_possible_filter(&vec, 10, ef_search, Some(&filter));

    for r in res {
        println!("Word: {:?} Id: {:?} Distance: {:?}", words[r.d_id], r.d_id, r.distance);
    }

}


fn main() {

    let nb_elem = 500000; // number of possible words in the dictionary
    let max_nb_connection = 15;
    let nb_layer = 16.min((nb_elem as f32).ln().trunc() as usize);
    let ef_c = 200;
    let hns = Hnsw::<u16, DistLevenshtein>::new(max_nb_connection, nb_elem, nb_layer, ef_c, DistLevenshtein{});
    let mut words = vec![];
    for _n in 1..500 {
        let tw = generate_random_string(5);
        words.push(tw);
    }

    let mut i = 0;
    for w in &words {
        let mut vec: Vec<u16> = Vec::new();
        for c in w.chars() {
            vec.push(c as u16);
        }
        hns.insert((&vec, i));
        i = i + 1;
        if i % 1000 == 0 {
            println!("Inserting: {:?}", i);
        }
    }
    search_vector_filter(&"abcde", &hns, &words);
    search_closure_filter(&"abcde", &hns, &words);
    println!("Words generated");

}