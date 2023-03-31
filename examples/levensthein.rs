
use hnsw_rs::prelude::*;
use hnsw_rs::dist::{DistLevenshtein};
use rand::Rng;
use std::iter;

// Used to create a random string
fn generate(len: usize) -> String {
    const CHARSET: &[u8] = b"abcdefghij";
    let mut rng = rand::thread_rng();
    let one_char = || CHARSET[rng.gen_range(0..CHARSET.len())] as char;
    iter::repeat_with(one_char).take(len).collect()
}

// this function uses a sorted vector as a filter
fn search_vector_filter(word: &str, hns: &Hnsw<u16, DistLevenshtein>, words: &Vec<String>) {
    // Create a sorted vector of ids
    // the ids in the vector will be used as a filter
    let mut vec: Vec<u16> = Vec::new();
    for c in word.chars() {
        vec.push(c as u16);
    }

    let mut filter: Vec<usize> = Vec::new();
    for i in 300..400 {
        filter.push(i);
    }

    // Now let us do the search
    // ids not in the vector will not be indluced in the search results
    let ef_search = 30;
    let res = hns.search_possible_filter(&vec, 10, ef_search, Some(&filter));

    let mut i = 0;
    for r in res {
        i = i + 1;
        println!("{:?}  Word: {:?} Id: {:?} Distance: {:?}", i, words[r.d_id], r.d_id, r.distance);
    }

    // Then do a "normal search without filter"
    println!("==========");   
    let res3 = hns.search_possible_filter(&vec, 10, ef_search, None);
    for r in res3 {
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
        let tw = generate(5);
        words.push(tw);
    }
    println!("Words generated");
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

}