extern crate hnsw_rs;

use hnsw_rs::test::*;
use hnsw_rs::dist::{DistLevenshtein};



fn search(word: &str, hns: &Hnsw<u16, DistLevenshtein>, words: &Vec<&str>) {
    let mut vec: Vec<u16> = Vec::new();
    for c in word.chars() {
        vec.push(c as u16);
    }
    let ef_search = 30;
    let knbn = words.len();
    let res = hns.search(&vec, knbn, ef_search);
    for r in res {
        println!("Word: {:?} distance: {:?}", words[r.d_id], r.distance);
    }
}

fn main() {

    let nb_elem = 500000; // number of possible words in the dictionary
    let max_nb_connection = 15;
    let nb_layer = 16.min((nb_elem as f32).ln().trunc() as usize);
    let ef_c = 200;
    let hns = Hnsw::<u16, DistLevenshtein>::new(max_nb_connection, nb_elem, nb_layer, ef_c, DistLevenshtein{});
    let words = vec!["abcd", "efgh", "ijkl", "abc", "test", "abbdc"];
    let mut i = 0;
    for w in &words {
        let mut vec: Vec<u16> = Vec::new();
        for c in w.chars() {
            vec.push(c as u16);
        }
        hns.insert((&vec, i));
        i = i + 1;
    }
    search(&"abcd", &hns, &words);

}