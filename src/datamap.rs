//! This module provides a memory mapping of Data vectors filling the Hnsw structure.
//! It is used by the module [hnswio] and also gives access to an iterator over data without loading the graph.
//!
//! We mmap the file and provide   
//!   - a Hashmap from DataId to address  
//!   - an interface for retrieving just data vectors loaded in the hnsw structure.

use std::io::BufReader;

use std::fs::{File, OpenOptions};
use std::path::PathBuf;

use indexmap::map::IndexMap;
use log::log_enabled;
use mmap_rs::{Mmap, MmapOptions};

use crate::hnsw::DataId;
use crate::hnswio;

use crate::hnswio::MAGICDATAP;
/// This structure uses the data part of the dump of a Hnsw structure to retrieve the data.
/// The data is access via a mmap of the data file, so memory is spared at the expense of page loading.
// possibly to be used in graph to spare memory?
pub struct DataMap {
    /// File containing Points data
    _datapath: PathBuf,
    /// The mmap structure
    mmap: Mmap,
    /// map a dataId to an address where we get a bson encoded vector of type T
    hmap: IndexMap<DataId, usize>,
    /// type name of Data
    t_name: String,
    /// dimension of data vector
    dimension: usize,
    ///
    distname: String,
} // end of DataMap

impl DataMap {
    // TODO: specifiy mmap option
    /// The fname argument corresponds to the basename of the dump.  
    /// To reload from file fname.hnsw.data just pass fname as argument.
    /// The dir argument is the directory where the fname.hnsw.data and fname.hnsw.graph reside.
    pub fn from_hnswdump<T: std::fmt::Debug>(dir: &str, fname: &String) -> Result<DataMap, String> {
        // reload description to have data type, and check for dump version
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
        let hnsw_description = hnswio::load_description(&mut graph_in).unwrap();
        if hnsw_description.format_version <= 2 {
            let msg = String::from("from_hnsw::from_hnsw : data mapping is only possible for dumps with the version > 0.1.19 of this crate");
            log::error!("from_hnsw::from_hnsw : data mapping is only possible for dumps with the version > 0.1.19 of this crate");
            return Err(msg);
        }
        let distname = hnsw_description.distname.clone();
        let t_name = hnsw_description.get_typename();
        // check typename coherence
        log::info!("got typename from reload : {:?}", t_name);
        if std::any::type_name::<T>() != t_name {
            log::error!(
                "description has typename {:?}, function type argument is : {:?}",
                t_name,
                std::any::type_name::<T>()
            );
            return Err(String::from("type error"));
        }
        // get dimension as declared in description
        let descr_dimension = hnsw_description.get_dimension();
        drop(graph_in);
        //
        // we know data filename is hnswdump.hnsw.data
        //
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
        let offset = 0;
        //
        let mmap_opt = MmapOptions::new(fsize).unwrap();
        let mmap_opt = unsafe { mmap_opt.with_file(&file, offset) };
        let mapping_res = mmap_opt.map();
        if mapping_res.is_err() {
            log::error!("could not memory map : {:?}", &datapath);
            std::process::exit(1);
        }
        let mmap = mapping_res.unwrap();
        //
        log::info!("mmap done on file : {:?}", &datapath);
        //
        // where are we in decoding mmap slice? at beginning
        //
        let mapped_slice = mmap.as_slice();
        //
        // where are we in decoding mmap slice?
        let mut current_mmap_addr = 0usize;
        let mut usize_slice = [0u8; std::mem::size_of::<usize>()];
        // check magic
        let mut u32_slice = [0u8; std::mem::size_of::<u32>()];
        u32_slice.copy_from_slice(
            &mapped_slice[current_mmap_addr..current_mmap_addr + std::mem::size_of::<u32>()],
        );
        current_mmap_addr += std::mem::size_of::<u32>();
        let magic = u32::from_ne_bytes(u32_slice);
        assert_eq!(magic, MAGICDATAP, "magic not equal to MAGICDATAP in mmap");
        // get dimension
        usize_slice.copy_from_slice(
            &mapped_slice[current_mmap_addr..current_mmap_addr + std::mem::size_of::<usize>()],
        );
        current_mmap_addr += std::mem::size_of::<usize>();
        let dimension = usize::from_ne_bytes(usize_slice) as usize;
        if dimension as usize != descr_dimension {
            log::error!("description and data do not agree on dimension, data got : {:?}, description got : {:?}",dimension, descr_dimension);
            return Err(String::from(
                "description and data do not agree on dimension",
            ));
        } else {
            log::info!(" got dimension : {:?}", dimension);
        }
        //
        // now we know that each record consists in
        //   - MAGICDATAP (u32), DataId  (u64), dimension (u64) and then  (length of type in bytes * dimension)
        //
        let record_size = std::mem::size_of::<u32>()
            + 2 * std::mem::size_of::<u64>()
            + dimension * std::mem::size_of::<T>();
        let residual = mmap.size() - current_mmap_addr;
        log::info!(
            "mmap size {}, current_mmap_addr {}, residual : {}",
            mmap.size(),
            current_mmap_addr,
            residual
        );
        let nb_record = residual / record_size;
        log::debug!("record size : {}, nb_record : {}", record_size, nb_record);
        // allocate hmap with correct capacity
        let mut hmap = IndexMap::<DataId, usize>::with_capacity(nb_record);
        // fill hmap to have address of each data point in file
        let mut u64_slice = [0u8; std::mem::size_of::<u64>()];
        //
        // now we loop on records
        //
        for i in 0..nb_record {
            log::debug!("record i : {}, addr : {}", i, current_mmap_addr);
            // decode Magic
            u32_slice.copy_from_slice(
                &mapped_slice[current_mmap_addr..current_mmap_addr + std::mem::size_of::<u32>()],
            );
            current_mmap_addr += std::mem::size_of::<u32>();
            let magic = u32::from_ne_bytes(u32_slice);
            assert_eq!(magic, MAGICDATAP, "magic not equal to MAGICDATAP in mmap");
            // decode DataId
            u64_slice.copy_from_slice(
                &mapped_slice[current_mmap_addr..current_mmap_addr + std::mem::size_of::<u64>()],
            );
            current_mmap_addr += std::mem::size_of::<u64>();
            let data_id = u64::from_ne_bytes(u64_slice) as usize;
            log::debug!(
                "inserting in hmap : got dataid : {:?} current map address : {:?}",
                data_id,
                current_mmap_addr
            );
            // Note we store address where we have to decode dimension*size_of::<T>  and full bson encoded vector
            hmap.insert(data_id, current_mmap_addr);
            // now read serialized length
            u64_slice.copy_from_slice(
                &mapped_slice[current_mmap_addr..current_mmap_addr + std::mem::size_of::<u64>()],
            );
            current_mmap_addr += std::mem::size_of::<u64>();
            let serialized_len = u64::from_ne_bytes(u64_slice) as usize;
            if log_enabled!(log::Level::Debug) && i == 0 {
                log::debug!("serialized bytes len to reload {:?}", serialized_len);
            }
            let mut v_serialized = Vec::<u8>::with_capacity(serialized_len);
            // TODO avoid initialization
            v_serialized.resize(serialized_len as usize, 0);
            v_serialized.copy_from_slice(
                &mapped_slice[current_mmap_addr..current_mmap_addr + serialized_len],
            );
            current_mmap_addr += serialized_len;
            let slice_t = unsafe {
                std::slice::from_raw_parts(v_serialized.as_ptr() as *const T, dimension as usize)
            };
            log::trace!(
                "deserialized v : {:?} address : {:?} ",
                slice_t,
                v_serialized.as_ptr() as *const T
            );
        } // end of for on record
          //
        log::debug!("\n end of DataMap::from_hnsw \n");
        //
        let datamap = DataMap {
            _datapath: datapath,
            mmap,
            hmap,
            t_name,
            dimension: descr_dimension,
            distname,
        };
        //
        return Ok(datamap);
    } // end of from_datas

    //

    /// returns true if type T corresponds to type as retrieved in DataMap.
    /// This function can (should!) be used before calling [Self::get_data()]
    pub fn check_data_type<T>(&self) -> bool
    where
        T: 'static + Sized,
    {
        // we check last part of name of type
        let tname_vec = self.t_name.rsplit_terminator("::").collect::<Vec<&str>>();

        if tname_vec.last().is_none() {
            let errmsg = "DataMap::check_data_type() cannot determine data type name ";
            log::error!("DataMap::check_data_type() cannot determine data type name ");
            std::panic!("DataMap::check_data_type(), {}", errmsg);
        }
        let tname_last = tname_vec.last().unwrap();
        //
        let datat_name_arg = std::any::type_name::<T>().to_string();
        let datat_name_vec = datat_name_arg
            .rsplit_terminator("::")
            .collect::<Vec<&str>>();

        let datat_name_arg_last = datat_name_vec.last().unwrap();
        //
        if datat_name_arg_last == tname_last {
            return true;
        } else {
            log::info!(
                "data type in DataMap : {},  type arg = {}",
                tname_last,
                datat_name_arg_last
            );
            return false;
        }
    } // end of check_data_type

    //

    /// return the data corresponding to dataid. Access is done using mmap.  
    /// Function returns None if address is invalid
    /// This function requires you know the type T.  
    /// **As mmap loading calls an unsafe function it is recommended to check the type name with  [Self::check_data_type()]**
    pub fn get_data<'a, T: Clone + std::fmt::Debug>(&'a self, dataid: &DataId) -> Option<&'a [T]> {
        //
        log::trace!("in DataMap::get_data, dataid : {:?}", dataid);
        let address = self.hmap.get(dataid);
        if address.is_none() {
            return None;
        }
        log::debug!(" adress for id : {}, address : {:?}", dataid, address);
        let mut current_mmap_addr = *address.unwrap();
        let mapped_slice = self.mmap.as_slice();
        let mut u64_slice = [0u8; std::mem::size_of::<u64>()];
        u64_slice.copy_from_slice(
            &mapped_slice[current_mmap_addr..current_mmap_addr + std::mem::size_of::<u64>()],
        );
        let serialized_len = u64::from_ne_bytes(u64_slice) as usize;
        current_mmap_addr += std::mem::size_of::<u64>();
        log::trace!("serialized bytes len to reload {:?}", serialized_len);
        let slice_t = unsafe {
            std::slice::from_raw_parts(
                mapped_slice[current_mmap_addr..].as_ptr() as *const T,
                self.dimension as usize,
            )
        };
        Some(slice_t)
    }

    /// returns Keys in order they are in the file, thus optimizing file/memory access.  
    /// Note that in case of parallel insertion this can be different from insertion odrer.
    pub fn get_dataid_iter(&self) -> indexmap::map::Keys<DataId, usize> {
        return self.hmap.keys();
    }

    /// returns full data type name
    pub fn get_data_typename(&self) -> String {
        return self.t_name.clone();
    }

    /// returns full data type name
    pub fn get_distname(&self) -> String {
        return self.distname.clone();
    }

    /// return the number of data in mmap
    pub fn get_nb_data(&self) -> usize {
        self.hmap.len()
    }
} // end of impl DataMap

//=====================================================================================

#[cfg(test)]

mod tests {

    use super::*;

    use crate::hnswio::HnswIo;
    use anndists::dist::*;

    pub use crate::api::AnnT;
    use crate::prelude::*;

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
        let unif = Uniform::<f32>::new(0., 1.);
        // 1000 vectors of size 10 f32
        let nbcolumn = 50;
        let nbrow = 11;
        let mut xsi;
        let mut data = Vec::with_capacity(nbcolumn);
        for j in 0..nbcolumn {
            data.push(Vec::with_capacity(nbrow));
            for _ in 0..nbrow {
                xsi = unif.sample(&mut rng);
                data[j].push(xsi);
            }
            log::debug!("j : {:?}, data : {:?} ", j, &data[j]);
        }
        // define hnsw
        let ef_construct = 25;
        let nb_connection = 10;
        let hnsw = Hnsw::<f32, DistL1>::new(nb_connection, nbcolumn, 16, ef_construct, DistL1 {});
        for i in 0..data.len() {
            hnsw.insert((&data[i], i));
        }
        // some loggin info
        hnsw.dump_layer_info();
        // dump in a file.  Must take care of name as tests runs in // !!!
        let fname = String::from("mmap_test");
        let _res = hnsw.file_dump(&fname);

        let check_reload = false;
        if check_reload {
            // We check we can reload
            log::debug!("\n\n  hnsw reload");
            let directory = PathBuf::from(".");
            let mut reloader = HnswIo::new(directory, String::from("mmap_test"));
            let hnsw_loaded: Hnsw<f32, DistL1> = reloader.load_hnsw::<f32, DistL1>().unwrap();
            check_graph_equality(&hnsw_loaded, &hnsw);
            log::info!("\n ========= reload success, going to mmap reloading ========= \n");
        }
        //
        // now we have check that datamap seems  ok, test reload of hnsw with mmap
        let datamap: DataMap = DataMap::from_hnswdump::<f32>(".", &fname).unwrap();
        let nb_test = 30;
        log::info!("checking random access of id , nb test : {}", nb_test);
        for _ in 0..nb_test {
            // sample an id in 0..nb_data
            let unif = Uniform::<usize>::new(0, nbcolumn);
            let id = unif.sample(&mut rng);
            let d = datamap.get_data::<f32>(&id);
            assert!(d.is_some());
            if d.is_some() {
                log::debug!("id = {}, v = {:?}", id, d.as_ref().unwrap());
                assert_eq!(d.as_ref().unwrap(), &data[id]);
            }
        }
        // test iterator from datamap
        let keys = datamap.get_dataid_iter();
        for k in keys {
            let _data = datamap.get_data::<f32>(k);
        }
        // rm files generated!
        let _ = std::fs::remove_file("mmap_test.hnsw.data");
        let _ = std::fs::remove_file("mmap_test.hnsw.graph");
    } // end of test_file_mmap

    #[test]
    fn test_mmap_iter() {
        log_init_test();
        // generate a random test
        let mut rng = rand::thread_rng();
        let unif = Uniform::<u32>::new(0, 10000);
        // 1000 vectors of size 10 f32
        let nbcolumn = 50;
        let nbrow = 11;
        let mut xsi;
        let mut data = Vec::with_capacity(nbcolumn);
        for j in 0..nbcolumn {
            data.push(Vec::with_capacity(nbrow));
            for _ in 0..nbrow {
                xsi = unif.sample(&mut rng);
                data[j].push(xsi);
            }
            log::debug!("j : {:?}, data : {:?} ", j, &data[j]);
        }
        // define hnsw
        let ef_construct = 25;
        let nb_connection = 10;
        let hnsw = Hnsw::<u32, DistL1>::new(nb_connection, nbcolumn, 16, ef_construct, DistL1 {});
        for i in 0..data.len() {
            hnsw.insert((&data[i], i));
        }
        // some loggin info
        hnsw.dump_layer_info();
        // dump in a file.  Must take care of name as tests runs in // !!!
        let fname = String::from("mmap_order_test");
        let _res = hnsw.file_dump(&fname);
        // now we have check that datamap seems  ok, test reload of hnsw with mmap
        let datamap: DataMap = DataMap::from_hnswdump::<u32>(".", &fname).unwrap();
        // testing type check
        assert!(datamap.check_data_type::<u32>());
        assert!(!datamap.check_data_type::<f32>());
        log::info!("Datamap iteration order checking");
        let keys = datamap.get_dataid_iter();
        let mut ukey = 0usize;
        for dataid in keys {
            let v = datamap.get_data::<u32>(dataid).unwrap();
            assert_eq!(v, &data[*dataid], "dataid = {}, ukey = {}", dataid, ukey);
            ukey += 1;
        }
        // rm files generated!
        let _ = std::fs::remove_file("mmap_order_test.hnsw.data");
        let _ = std::fs::remove_file("mmap_order_test.hnsw.graph");
    }
    //
} // end of mod tests
