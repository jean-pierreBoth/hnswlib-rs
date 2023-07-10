//! This module provides reload of the dump consisting in Data vectors.
//! We mmap the file and provide 
//!     - a Hashmap from DataId to address
//!     - an interface for retrieving just data vectors loaded in the hnsw structure.

#![allow(unused)]

use std::io::BufReader;

use std::path::{PathBuf};
use std::fs::{File,OpenOptions};

use mmap_rs::{MmapOptions,Mmap};
use hashbrown::HashMap;

use crate::prelude::DataId;
use crate::hnswio::load_description;


/// This structure uses the data part of the dump of a Hnsw structure to retrieve the data.
/// The data is access via a mmap of the data file, so memory is spared at the expense of page loading.
// possibly to be used in graph to spare memory?
pub struct DataMap {
    /// File containing Points data
    datapath : PathBuf,
    /// The mmap structure
    mmap : Mmap,
    /// map a dataId to an address
    hmap: HashMap<DataId, u64>,
    /// type name of Data
    t_name : String,
} // end of DataMap


impl DataMap {

    pub fn new(dir : &str, fname : &String) -> Self {
        // we know data filename is hnswdump.hnsw.data
        let mut datapath = PathBuf::new();
        datapath.push(dir);
        let mut filename = fname.clone();
        filename.push_str(".hnsw.data");
        datapath.push(filename);
        //
        let meta = std::fs::metadata(&datapath);
        if meta.is_err() {
            log::error!("could not open file : {:?}", &datapath);
            std::process::exit(1);            
        }
        let fsize = meta.unwrap().len().try_into().unwrap();
        //
        let file_res = File::open(&datapath);
        if file_res.is_err() {
            log::error!("could not open file : {:?}", &datapath);
            std::process::exit(1);
        }
        let file = file_res.unwrap();
        let offset  = 0;
        //
        let mmap_opt = MmapOptions::new(fsize).unwrap();
        let mmap_opt = unsafe { mmap_opt.with_file(&file, offset)};
        let mapping_res = mmap_opt.map();
        if mapping_res.is_err() {
            log::error!("could not memory map : {:?}", &datapath);
            std::process::exit(1);            
        }
        let mmap = mapping_res.unwrap();
        //
        let hmap = HashMap::<DataId, u64>::new();
        log::info!("mmap done on file : {:?}", &datapath);
        //
        // reload description to have data type
        let mut graphpath = PathBuf::new();
        graphpath.push(dir);
        let mut filename = fname.clone();
        filename.push_str(".hnsw.graph");
        graphpath.push(filename);
        let graphfileres = OpenOptions::new().read(true).open(&graphpath);
        if graphfileres.is_err() {
            println!("DataMap: could not open file {:?}", graphpath.as_os_str());
            std::process::exit(1);            
        }
        let graphfile = graphfileres.unwrap();
        let mut graph_in = BufReader::new(graphfile);
        // we need to call load_description first to get distance name
        let hnsw_description = crate::hnswio::load_description(&mut graph_in).unwrap();
        let t_name = hnsw_description.get_typename();
        // fill hmap to have address of each data point in file
        //
        return DataMap{datapath, mmap, hmap, t_name};
    } // end of new

    /// get adress of data related to dataid
    fn get_data_address(&self, dataid : DataId) -> u64 {
        panic!("not yet implemented");
    }

    /// return the data corresponding to dataid. Access is done via mmap
    pub fn get_data<T>(&self, dataid : DataId) -> &[T] {
        panic!("not yet implemented");
    }
} // end of impl DataMap


//=====================================================================================


#[cfg(test)]


mod tests {

use super::*;

use crate::dist;

use crate::prelude::*;
pub use crate::api::AnnT;

use rand::distributions::{Distribution, Uniform};


fn log_init_test() {
    let _ = env_logger::builder().is_test(true).try_init();
}


#[test]
fn test_file_mmap() {
    println!("\n\n test_file_mmap");
    log_init_test();
    // generate a random test
    let mut rng = rand::thread_rng();
    let unif =  Uniform::<f32>::new(0.,1.);
    // 1000 vectors of size 10 f32
    let nbcolumn = 1000;
    let nbrow = 10;
    let mut xsi;
    let mut data = Vec::with_capacity(nbcolumn);
    for j in 0..nbcolumn {
        data.push(Vec::with_capacity(nbrow));
        for _ in 0..nbrow {
            xsi = unif.sample(&mut rng);
            data[j].push(xsi);
        }
    } 
    // define hnsw
    let ef_construct= 25;
    let nb_connection = 10;
    let hnsw = Hnsw::<f32, dist::DistL1>::new(nb_connection, nbcolumn, 16, ef_construct, dist::DistL1{});
    for i in 0..data.len() {
        hnsw.insert((&data[i], i));
    }
    // some loggin info
    hnsw.dump_layer_info();
    // dump in a file.  Must take care of name as tests runs in // !!!
    let fname = String::from("mmap_test");
    let _res = hnsw.file_dump(&fname);
    //
    //
    //
    let datamap = DataMap::new(".", &fname);
} // end of test_file_mmap

} // end of mod tests 