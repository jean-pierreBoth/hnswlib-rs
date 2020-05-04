//! Api for external language.
//! This file provides a trait to be used as an opaque pointer for C or Julia calls used in file libext.rs


use std::io::prelude::*;
use std::fs::OpenOptions;
use std::io::{BufWriter};
use std::path::{PathBuf};

use typename::TypeName;

use crate::hnsw::*;
use crate::hnswio::*;


pub trait AnnT {
    /// type of data vectors 
    type Val;
    ///
    fn insert_data(&mut self, data: &Vec<Self::Val>, id: usize);
    ///
    fn search_neighbours(&self,  data :&Vec<Self::Val> , knbn : usize, ef_s: usize) -> Vec<Neighbour>;
    ///
    fn parallel_insert_data(&mut self, data: &Vec<(&Vec<Self::Val>, usize)> );
    ///
    fn parallel_search_neighbours(&self, data: &Vec<Vec<Self::Val> >, knbn : usize, ef_s: usize) -> Vec<Vec<Neighbour>> ;
    ///
    /// dumps a data and graph in 2 files. 
    /// Datas are dumped in file filename.hnsw.data and graph in filename.hnsw.graph
    fn file_dump(&self, filename: &String) -> Result<i32, String>;
 }


impl <T,D> AnnT for Hnsw<T,D>  where T:Copy+Clone+Send+Sync+TypeName , D: Distance<T>+TypeName+Send+Sync {
    type Val= T;
    ///
    fn insert_data(&mut self, data: &Vec<Self::Val>, id: usize) {
       self.insert((data, id));
    }
    ///
    fn search_neighbours(&self, data : &Vec<T>, knbn : usize, ef_s: usize) -> Vec<Neighbour> {
        self.search(data, knbn, ef_s)
    }
    fn parallel_insert_data(&mut self, data: &Vec<(&Vec<Self::Val>, usize)> ) {
        self.parallel_insert(data);
    }

   fn parallel_search_neighbours(&self, data: &Vec<Vec<Self::Val>>, knbn : usize, ef_s: usize) -> Vec<Vec<Neighbour>> {
       self.parallel_search(data, knbn, ef_s)
   }
   /// 
   fn file_dump(&self, filename: &String) -> Result<i32, String> {
       let mut graphname = filename.clone();
        graphname.push_str(".hnsw.graph");
        let graphpath = PathBuf::from(graphname);
        let fileres = OpenOptions::new().write(true).create(true).truncate(true).open(&graphpath);
        if fileres.is_err() {
            println!("could not open file {:?}", graphpath.as_os_str());
            return Err("could not open file".to_string());            
        }
        let graphfile = fileres.unwrap();
        //
        let mut dataname = filename.clone();
        dataname.push_str(".hnsw.data");
        let datapath = PathBuf::from(dataname);
        let fileres = OpenOptions::new().write(true).create(true).truncate(true).open(&datapath);
        if fileres.is_err() {
            println!("could not open file {:?}", datapath.as_os_str());
            return Err("could not open file".to_string());            
        }
        let datafile = fileres.unwrap();
        let mut graphbufw = BufWriter::with_capacity(10000000 , graphfile);
        let mut databufw = BufWriter::with_capacity(10000000 , datafile);
        let res = self.dump(DumpMode::Full, &mut graphbufw, &mut databufw);
        graphbufw.flush().unwrap();
        databufw.flush().unwrap();
        log::debug!("\n end of dump");
        return res;
   }   // end of dump
} // end of impl block AnnT for Hnsw<T,D>


// macro export makes the macro export t the root of the crate
#[macro_export]
macro_rules! mapdist_t(
    ("DistL1")       => (crate::dist::DistL1);
    ("DistL2")       => (crate::dist::DistL2);
    ("DistL2")       => (crate::dist::DistL2);
    ("DistDot")      => (crate::dist::DistDot);
    ("DistHamming")  => (crate::dist::DistHamming);
    ("DistJaccard")  => (crate::dist::DistJaccard);
    ("DistPtr")      => (crate::dist::DistPtr);
    ("DistLevenshtein") => (crate::dist::DistLevenshtein);
);


#[macro_export]
macro_rules! mapdist_c(
    ("DistL1")  => (crate::dist::DistL1{});
);


#[macro_export]
macro_rules! genhnsw(
    ($val:ty, $dist:tt, $max_nb_conn:tt, $ef_const:tt) => (
        Hnsw::<$val, mapdist_t!($dist) >::new($max_nb_conn, 10000, 16, $ef_const, mapdist_c!($dist))
    )
);





//////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {

use super::*;


#[test]
    fn test_genhnsw() {
        let h = genhnsw!(f32, "DistL1", 24, 48);
        println!("test constructed Hnsw with distance : {:?} ", h.get_distance_name());
    }
} // end of module test
