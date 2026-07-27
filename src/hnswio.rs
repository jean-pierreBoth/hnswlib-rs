//! This module provides io dump/ reload of computed graph via the structure Hnswio.  
//! This structure stores references to data points if memory map is used.  
//!
//! A dump is constituted of 2 files.
//! One file stores just the graph (or topology) with id of points.  
//! The other file stores the ids and vector in point and can be reloaded via a mmap scheme.
//! The graph file is suffixed by "hnsw.graph" the other is suffixed by "hnsw.data"
//!
//! Examples of dump and reload of structure Hnsw is given in the tests (see test_dump_reload, reload_with_mmap)
// datafile
// MAGICDATAP : u32
// dimension : usize!!
// The for each point the triplet: (MAGICDATAP, origin_id , dimension , array of values bson encoded) ( u32, u64, ....)
//
// A point is dumped in graph file as given by its external id (type DataId i.e : a usize, possibly a hash value)
// and layer (u8) and rank_in_layer:i32.
// In the data file the point dump consist in the triplet: (MAGICDATAP, origin_id , array of values.)
//
use serde::{Serialize, de::DeserializeOwned};
use std::sync::atomic::{AtomicUsize, Ordering};
//
use std::time::SystemTime;

// io
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

// synchro
use parking_lot::RwLock;
use std::sync::Arc;

use std::collections::HashMap;

use anyhow::*;
use std::any::type_name;

use anndists::dist::distances::*;

use self::hnsw::*;
use crate::datamap::*;
use crate::hnsw;
use log::{debug, error, info, trace};
use std::io::prelude::*;

// magic before each graph point data for each point
const MAGICPOINT: u32 = 0x000a678f;
// magic at beginning of description format v2 of dump
const MAGICDESCR_2: u32 = 0x002a677f;

// magic at beginning of description format v3 of dump
// format where we can use mmap to provide acces to data (not graph) via a memory mapping of file data ,
// useful when data vector are large and data uses more space than graph.
// differ from v2 as we do not use bincode encoding for point. We dump pure binary
// This help use mmap as we can return directly a slice.
const MAGICDESCR_3: u32 = 0x002a6771;

// magic for v4
// we dump level scale modififcation factor
const MAGICDESCR_4: u32 = 0x002a6779;

// magic at beginning of a layer dump
const MAGICLAYER: u32 = 0x000a676f;
// magic head of data file and before each data vector
pub(crate) const MAGICDATAP: u32 = 0xa67f0000;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DumpMode {
    Light,
    Full,
}

/// The main interface for dumping struct Hnsw.
pub(crate) trait HnswIoT {
    fn dump(&self, mode: DumpMode, dumpinit: &mut DumpInit) -> anyhow::Result<i32>;
}

/// Describe options accessible for reload
///
///  - datamap : a bool for mmap usage.  
///    The data point can be reloaded via mmap of data file dump.  
///    This can be useful when data points consist in large vectors (as in genomic sketching)
///    as in this case data needs more space than the graph.  
///
///  - mmap_threshold : the number of itmes above which we use mmap. Default is 0, meaning always use mmap data
///    Can be useful for search speed in hnsw if we have part of data resident in memory.
#[derive(Copy, Clone)]
pub struct ReloadOptions {
    datamap: bool,
    /// number of data items above which we use mmap.
    mmap_threshold: usize,
}

impl Default for ReloadOptions {
    /// default is no mmap
    fn default() -> Self {
        ReloadOptions {
            datamap: false,
            mmap_threshold: 0,
        }
    }
}

impl ReloadOptions {
    pub fn new(datamap: bool) -> Self {
        ReloadOptions {
            datamap,
            mmap_threshold: 0,
        }
    }

    /// set mmap uasge to true
    pub fn set_mmap(&mut self, val: bool) -> Self {
        self.datamap = val;
        *self
    }

    /// set mmap threshold i.e : The maximum number of data that will be reloaded in memory by reading file dump, the other points will be mmapped.  
    /// As the upper layers are the most frequently used, these points will be loaded in memory during reading, the others will be mmaped.
    /// See test *reload_with_mmap()*
    pub fn set_mmap_threshold(&mut self, threshold: usize) -> Self {
        if threshold > 0 {
            self.datamap = true;
            self.mmap_threshold = threshold;
        }
        *self
    }

    /// return a 2-uple, (datamap, threshold)
    pub fn use_mmap(&self) -> (bool, usize) {
        (self.datamap, self.mmap_threshold)
    }
} // end of ReloadOptions

//===============================================================================================

// initialize datafile and graphfile for io ops
// This structure will check existence of dumps of same name and generate a unique filename if necessary according to overwrite flag
pub struct DumpInit {
    // basename dump
    basename: String,
    // to dump data
    pub(crate) data_out: BufWriter<File>,
    // to dump graph
    pub(crate) graph_out: BufWriter<File>,
} // end of

impl DumpInit {
    // This structure will check existence of dumps of same name and generate a unique filename if necessary according to overwrite flag
    pub fn new(dir: &Path, basename_default: &str, overwrite: bool) -> Self {
        // if we cannot overwrite data files (in case of mmap in particular)
        // we will ensure we have a unique basename
        let basename = match overwrite {
            true => basename_default.to_string(),
            false => {
                // we check
                let mut dataname = basename_default.to_string();
                dataname.push_str(".hnsw.data");
                let mut datapath = PathBuf::from(dir);
                datapath.push(dataname);
                let exist_res = std::fs::metadata(datapath.as_os_str());
                if exist_res.is_ok() {
                    loop {
                        let mut unique_basename;
                        let mut dataname: String;
                        let id: usize = rand::random_range(0..10000);
                        let strid: String = id.to_string();
                        unique_basename = basename_default.to_string();
                        unique_basename.push('-');
                        unique_basename.push_str(&strid);
                        dataname = unique_basename.clone();
                        dataname.push_str(".hnsw.data");
                        let mut datapath = PathBuf::from(dir);
                        datapath.push(dataname);
                        let exist_res = std::fs::metadata(datapath.as_os_str());
                        if exist_res.is_err() {
                            break unique_basename;
                        }
                    }
                } else {
                    basename_default.to_string()
                }
            }
        };
        //
        debug!(
            "DumpInit : dumping in dir {:#?} with (unique) basename : {}",
            dir, basename
        );
        //
        let mut graphname = basename.clone();
        graphname.push_str(".hnsw.graph");
        let mut graphpath = PathBuf::from(dir);
        graphpath.push(graphname);
        let graphfileres = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&graphpath);
        if graphfileres.is_err() {
            println!(
                "DumpInit::new : could not open file {:?}",
                graphpath.as_os_str()
            );
            std::panic::panic_any("HnswIo::init : could not open file".to_string());
        }
        let graphfile = graphfileres.unwrap();
        //  same thing for data file
        let mut dataname = basename.clone();
        dataname.push_str(".hnsw.data");
        let mut datapath = PathBuf::from(dir);
        datapath.push(dataname);
        let datafileres = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&datapath);
        if datafileres.is_err() {
            println!(
                "DumpInit::init : could not open file {:?}",
                datapath.as_os_str()
            );
            std::panic::panic_any("HnswIo::init : could not open file".to_string());
        }
        let datafile = datafileres.unwrap();
        //
        let graph_out = BufWriter::new(graphfile);
        let data_out = BufWriter::new(datafile);
        //
        DumpInit {
            basename,
            data_out,
            graph_out,
        }
    }

    /// returns the basename used for the dump. May be it has been made unique to void overwriting a previous or mmapped dump
    pub fn get_basename(&self) -> &String {
        &self.basename
    }

    pub fn flush(&mut self) -> Result<()> {
        self.data_out.flush()?;
        self.graph_out.flush()?;
        Ok(())
    }
} // end impl for DumpInit

//====================================================
// basic block used to provide arguments to load_hnsw and load_hnsw_with_dist
struct LoadInit {
    descr: Description,
    //
    graphfile: BufReader<File>,
    //
    datafile: BufReader<File>,
} // end of LoadInit

/// a structure to provide simplified methods for reloading a previous dump.  
///  
/// The data point can be reloaded via mmap of data file dump.  
/// This can be useful when data points consist in large vectors (as in genomic sketching)
/// as in this case data needs more space than the graph.  
/// Note : **As this structure potentially contains the mmap data used in hnsw after reload it must not be dropped
/// before the reloaded hnsw.**
/// Example:
///
/// See example in  tests::reload_with_mmap
/// ```text
///     let directory = Path::new(".");
///     let mut reloader = HnswIo::new(directory, "mmapreloadtest");
///     let options = ReloadOptions::default().set_mmap(true);
///     reloader.set_options(options);
///     let hnsw_loaded : Hnsw<f32,DistL1>= reloader.load_hnsw::<f32, DistL1>().unwrap();
/// ```  
///   
/// In some cases we need a hnsw variable that can come from a reload **OR** a direct initialization.  
///   
/// Hnswio must be defined before Hnsw as drop is done in reverse order of definition, and the function [load_hnsw](Self::load_hnsw())
/// borrows Hnswio. (Hnswio stores the mmap address Hnsw can refer to if mmap is used)
/// It is also possible to preinitialize a Hnswio with the default() function which leaves all the fields with blank values and use
/// the function [set_values](Self::set_values()) after.  
/// We get something like:
///
/// ```text
///     let need_reload : bool;
///     ....................
///     let mut hnswio : Hnswio::default();
///     let hnsw : Hnsw<>;
///     if need_reload {
///         hnswio.set_values(...);
///         hnsw = hnswio.reload_hnsw(...)
///     }
///     else {
///         hnsw = Hnsw::new(...)
///     }
/// ````
#[derive(Default)]
pub struct HnswIo {
    dir: PathBuf,
    /// basename is used to build $basename.hnsw.data and $basename.hnsw.graph
    basename: String,
    /// options
    options: ReloadOptions,
    datamap: Option<DataMap>,
    /// for Hnswio to be async
    nb_point_loaded: Arc<AtomicUsize>,
    initialized: bool,
} // end of struct ReloadOptions

impl HnswIo {
    /// - directory is directory containing the dumped files,
    /// - basename is used to build $basename.hnsw.data and $basename.hnsw.graph
    ///
    ///  default is to use default ReloadOptions.
    pub fn new(directory: &Path, basename: &str) -> Self {
        HnswIo {
            dir: directory.to_path_buf(),
            basename: basename.to_string(),
            options: ReloadOptions::default(),
            datamap: None,
            nb_point_loaded: Arc::new(AtomicUsize::new(0)),
            initialized: true,
        }
    }

    /// same as preceding, avoids the call to [set_options](Self::set_options())
    pub fn new_with_options(directory: &Path, basename: &str, options: ReloadOptions) -> Self {
        HnswIo {
            dir: directory.to_path_buf(),
            basename: basename.to_string(),
            options,
            datamap: None,
            nb_point_loaded: Arc::new(AtomicUsize::new(0)),
            initialized: true,
        }
    }

    /// return basename of dump
    pub fn get_basename(&self) -> &str {
        &self.basename
    }
    /// this method enables effective initialization after default allocation.
    /// It is an error to call set_values on an already defined Hswnio by any function other than [default](Self::default())
    pub fn set_values(
        &mut self,
        directory: &Path,
        basename: String,
        options: ReloadOptions,
    ) -> Result<()> {
        if self.initialized {
            return Err(anyhow!("Hnswio already initialized"));
        };
        //
        self.dir = directory.to_path_buf();
        self.basename = basename;
        self.options = options;
        self.datamap = None;
        //
        self.initialized = true;
        //
        Ok(())
    } // end of set_values

    //
    fn init(&self) -> Result<LoadInit> {
        //
        info!("reloading from basename : {}", &self.basename);
        //
        let mut graphname = self.basename.clone();
        graphname.push_str(".hnsw.graph");
        let mut graphpath = self.dir.clone();
        graphpath.push(graphname);
        let graphfileres = OpenOptions::new().read(true).open(&graphpath);
        if graphfileres.is_err() {
            println!(
                "HnswIo::init : could not open file {:?}",
                graphpath.as_os_str()
            );
            error!(
                "HnswIo::init : could not open file {:?}",
                graphpath.as_os_str()
            );
            return Err(anyhow!(
                "HnswIo::init : could not open file {:?}",
                graphpath.as_os_str()
            ));
        }
        let graphfile = graphfileres.unwrap();
        //  same thing for data file
        let mut dataname = self.basename.clone();
        dataname.push_str(".hnsw.data");
        let mut datapath = self.dir.clone();
        datapath.push(dataname);
        let datafileres = OpenOptions::new().read(true).open(&datapath);
        if datafileres.is_err() {
            println!(
                "HnswIo::init : could not open file {:?}",
                datapath.as_os_str()
            );
            error!(
                "HnswIo::init : could not open file {:?}",
                datapath.as_os_str()
            );
            return Err(anyhow!(
                "HnswIo::init : could not open file {:?}",
                datapath.as_os_str()
            ));
        }
        let datafile = datafileres.unwrap();
        //
        let mut graph_in = BufReader::new(graphfile);
        let data_in = BufReader::new(datafile);
        // we need to call load_description first to get distance name.
        // A rejected description is a normal outcome for an untrusted or
        // damaged dump, so it must stay an error here: unwrapping would turn
        // every header validation below into a panic at this line.
        let hnsw_description = load_description(&mut graph_in)
            .map_err(|e| anyhow!("HnswIo::init : could not load description: {}", e))?;
        //
        Ok(LoadInit {
            descr: hnsw_description,
            graphfile: graph_in,
            datafile: data_in,
        })
    }

    /// to set non default options, in particular to ask for mmap of data file
    pub fn set_options(&mut self, options: ReloadOptions) {
        self.options = options;
    }

    /// reload a previously dumped hnsw structure
    pub fn load_hnsw<'b, 'a, T, D>(&'a mut self) -> Result<Hnsw<'b, T, D>>
    where
        T: 'static + Serialize + DeserializeOwned + Clone + Sized + Send + Sync + std::fmt::Debug,
        D: Distance<T> + Default + Send + Sync,
        'a: 'b,
    {
        //
        debug!("HnswIo::load_hnsw ");
        let start_t = SystemTime::now();
        //
        let mut init = self
            .init()
            .map_err(|e| anyhow!("could not reload HNSW structure: {}", e))?;
        let data_in = &mut init.datafile;
        let graph_in = &mut init.graphfile;
        let description = init.descr;
        info!("format version : {}", description.format_version);
        //  In datafile , we must read MAGICDATAP and dimension and check
        let mut it_slice = [0u8; std::mem::size_of::<u32>()];
        data_in.read_exact(&mut it_slice)?;
        let magic = u32::from_ne_bytes(it_slice);
        check_data_header(magic, data_in, &description)?;
        //
        let _mode = description.dumpmode;
        let distname = description.distname.clone();
        // We must ensure that the distance stored matches the one asked for in loading hnsw
        // for that we check for short names equality stripping
        debug!("distance in description = {:?}", distname);
        let d_type_name = type_name::<D>().to_string();
        let d_type_name_split: Vec<&str> = d_type_name.rsplit_terminator("::").collect();
        for s in &d_type_name_split {
            info!(" distname in generic type argument {:?}", s);
        }
        let distname_split: Vec<&str> = distname.rsplit_terminator("::").collect();
        if (std::any::TypeId::of::<T>() != std::any::TypeId::of::<NoData>())
            && (d_type_name_split[0] != distname_split[0])
        {
            // for all types except NoData , distance asked in reload declaration and distance in dump must be equal!
            let mut errmsg = String::from("error in distances : dumped distance is : ");
            errmsg.push_str(&distname);
            errmsg.push_str(" asked distance in loading is : ");
            errmsg.push_str(&d_type_name);
            error!(" distance in type argument : {:?}", d_type_name);
            error!("error , dump is for distance = {:?}", distname);
            return Err(anyhow!(errmsg));
        }
        let t_type = description.t_name.clone();
        debug!("T type name in dump = {:?}", t_type);
        // Do we use mmap at reload
        if self.options.use_mmap().0 {
            let datamap_res = DataMap::from_hnswdump::<T>(self.dir.as_path(), &self.basename);
            if let Result::Ok(datamap) = datamap_res {
                info!("reload using mmap");
                self.datamap = Some(datamap);
            } else {
                error!("load_hnsw could not initialize mmap")
            }
        }
        // reloader can use datamap
        let layer_point_indexation = self.load_point_indexation(graph_in, &description, data_in)?;
        let data_dim = layer_point_indexation.get_data_dimension();
        //
        let hnsw: Hnsw<T, D> = Hnsw {
            max_nb_connection: description.max_nb_connection as usize,
            ef_construction: description.ef,
            extend_candidates: true,
            keep_pruned: false,
            max_layer: description.nb_layer as usize,
            layer_indexed_points: layer_point_indexation,
            data_dimension: data_dim,
            dist_f: D::default(),
            searching: false,
            datamap_opt: true, // set datamap_opt to true
        };
        //
        debug!("load_hnsw completed");
        let elapsed_t = start_t.elapsed().unwrap().as_secs() as f32;
        info!("reload_hnsw : elapsed system time(s) {}", elapsed_t);
        Ok(hnsw)
    } // end of load_hnsw

    /// reload a previously dumped hnsw structure
    /// This function makes reload of a Hnsw dump with a given Dist.  
    /// It is dedicated to distance of type DistPtr (see crate [anndist](https://crates.io/crates/anndists)) that cannot implement Default.  
    /// **It is the user responsability to reload with the same function as used in the dump**
    ///
    pub fn load_hnsw_with_dist<'b, 'a, T, D>(&'a self, f: D) -> anyhow::Result<Hnsw<'b, T, D>>
    where
        T: 'static + Serialize + DeserializeOwned + Clone + Sized + Send + Sync + std::fmt::Debug,
        D: Distance<T> + Send + Sync,
        'a: 'b,
    {
        //
        debug!("HnswIo::load_hnsw_with_dist");
        //
        let mut init = self
            .init()
            .map_err(|e| anyhow!("Could not reload hnsw structure: {}", e))?;
        //
        let data_in = &mut init.datafile;
        let graph_in = &mut init.graphfile;
        let description = init.descr;
        //  In datafile , we must read MAGICDATAP and dimension and check
        let mut it_slice = [0u8; std::mem::size_of::<u32>()];
        data_in.read_exact(&mut it_slice)?;
        let magic = u32::from_ne_bytes(it_slice);
        check_data_header(magic, data_in, &description)?;
        //
        let _mode = description.dumpmode;
        let distname = description.distname.clone();
        // We must ensure that the distance stored matches the one asked for in loading hnsw
        // for that we check for short names equality stripping
        info!("distance in description = {:?}", distname);
        let d_type_name = type_name::<D>().to_string();
        let v: Vec<&str> = d_type_name.rsplit_terminator("::").collect();
        for s in v {
            info!(" distname in generic type argument {:?}", s);
        }
        if (std::any::TypeId::of::<T>() != std::any::TypeId::of::<NoData>())
            && (d_type_name != distname)
        {
            // for all types except NoData , distance asked in reload declaration and distance in dump must be equal!
            let mut errmsg = String::from("error in distances : dumped distance is : ");
            errmsg.push_str(&distname);
            errmsg.push_str(" asked distance in loading is : ");
            errmsg.push_str(&d_type_name);
            error!(" distance in type argument : {:?}", d_type_name);
            error!("error , dump is for distance = {:?}", distname);
            return Err(anyhow!(errmsg));
        }
        let t_type = description.t_name.clone();
        info!("T type name in dump = {:?}", t_type);
        //
        //
        let layer_point_indexation = self.load_point_indexation(graph_in, &description, data_in)?;
        let data_dim = layer_point_indexation.get_data_dimension();
        //
        let hnsw: Hnsw<T, D> = Hnsw {
            max_nb_connection: description.max_nb_connection as usize,
            ef_construction: description.ef,
            extend_candidates: true,
            keep_pruned: false,
            max_layer: description.nb_layer as usize,
            layer_indexed_points: layer_point_indexation,
            data_dimension: data_dim,
            dist_f: f,
            searching: false,
            datamap_opt: false,
        };
        //
        debug!("load_hnsw_with_dist completed");
        // We cannot check that the pointer function was the same as the dump
        //
        Ok(hnsw)
    } // end of load_hnsw_with_dist

    fn load_point_indexation<'b, 'a, T>(
        &'a self,
        graph_in: &mut dyn Read,
        descr: &Description,
        data_in: &mut dyn Read,
    ) -> anyhow::Result<PointIndexation<'b, T>>
    where
        T: 'static + Serialize + DeserializeOwned + Clone + Sized + Send + Sync + std::fmt::Debug,
        'a: 'b,
    {
        //
        debug!(" in load_point_indexation");
        //
        // now we check that except for the case NoData, the typename are the sames.
        if std::any::TypeId::of::<T>() != std::any::TypeId::of::<NoData>()
            && std::any::type_name::<T>() != descr.t_name
        {
            error!(
                "typename loaded in  description {:?} do not correspond to instanciation type {:?}",
                descr.t_name,
                std::any::type_name::<T>()
            );
            return Err(anyhow!(
                "dump was written for element type {:?} but is being reloaded as {:?}",
                descr.t_name,
                std::any::type_name::<T>()
            ));
        }
        //
        let mut points_by_layer: Vec<Vec<Arc<Point<T>>>> =
            Vec::with_capacity(NB_LAYER_MAX as usize);
        let mut neighbourhood_map: HashMap<PointId, Vec<Vec<Neighbour>>> = HashMap::new();
        // load max layer
        let mut it_slice = [0u8; ::std::mem::size_of::<u8>()];
        graph_in.read_exact(&mut it_slice)?;
        let nb_layer = u8::from_ne_bytes(it_slice);
        debug!("nb layer {:?}", nb_layer);
        if nb_layer > NB_LAYER_MAX {
            return Err(anyhow!("inconsistent number of layErrers"));
        }
        //
        let mut nb_points_loaded: usize = 0;
        let mut nb_still_to_load = descr.nb_point as i64;
        let (use_mmap, max_nbpoint_in_memory) = self.options.use_mmap();
        //
        for l in 0..nb_layer as usize {
            // read and check magic
            debug!("loading layer {:?}", l);
            let mut it_slice = [0u8; ::std::mem::size_of::<u32>()];
            graph_in.read_exact(&mut it_slice)?;
            let magic = u32::from_ne_bytes(it_slice);
            if magic != MAGICLAYER {
                return Err(anyhow!("bad magic at layer beginning"));
            }
            let mut it_slice = [0u8; ::std::mem::size_of::<usize>()];
            graph_in.read_exact(&mut it_slice)?;
            let nbpoints = usize::from_ne_bytes(it_slice);
            debug!(" layer {:?} , nb points {:?}", l, nbpoints);
            // No layer can hold more points than the whole index declares;
            // rejecting here keeps a hostile count from driving allocation
            // and from silently disagreeing with the description.
            if nbpoints > descr.nb_point {
                return Err(anyhow!(
                    "layer {} declares {} points, above the {} the description declares",
                    l,
                    nbpoints,
                    descr.nb_point
                ));
            }
            let mut vlayer: Vec<Arc<Point<T>>> =
                Vec::with_capacity(nbpoints.min(RESERVE_POINT_HINT));
            // load graph and data part of point. Points are dumped in the same order.
            for r in 0..nbpoints {
                // do we use mmap? for this point. We must load into memory up to threshold points, and we also want the  most
                // frequently accessed points, i.e those in upper layers! to be physically loaded.
                // So we do use mmap from the moment the number of points yet to be loaded is less than threshold.
                let point_use_mmap = match use_mmap {
                    false => false,
                    true => {
                        if nb_still_to_load <= max_nbpoint_in_memory as i64 {
                            if log::log_enabled!(log::Level::Info)
                                && nb_still_to_load == max_nbpoint_in_memory as i64
                            {
                                info!(
                                    "Switching to points in memory. nb points stiil to load {:?}",
                                    nb_still_to_load
                                );
                            }
                            false
                        } else {
                            true
                        }
                    }
                };
                let load_point_res = self
                    .load_point(graph_in, descr, data_in, point_use_mmap)
                    .map_err(|other| {
                        error!("in load_point_indexation, loading of point {} failed", r);
                        anyhow!(other)
                    })?;
                let point = load_point_res.0;
                let p_id = point.get_point_id();
                // Points are dumped layer by layer in rank order, so a point
                // whose own identity contradicts its position means the graph
                // file is inconsistent with itself.
                if l != p_id.0 as usize || r != p_id.1 as usize {
                    debug!("Origin= {:?},  p_id = {:?}", point.get_origin_id(), p_id);
                    debug!("Storing at l {:?}, r {:?}", l, r);
                    return Err(anyhow!(
                        "point stored at layer {} rank {} carries identity {:?}",
                        l,
                        r,
                        p_id
                    ));
                }
                // store neoghbour info of this point
                neighbourhood_map.insert(p_id, load_point_res.1);
                vlayer.push(point);
                nb_points_loaded += 1;
                nb_still_to_load -= 1;
                if nb_still_to_load < 0 {
                    return Err(anyhow!(
                        "graph file holds more points than the {} the description declares",
                        descr.nb_point
                    ));
                }
            }
            points_by_layer.push(vlayer);
        }
        // at this step all points are loaded , but without their neighbours fileds are not yet initialized
        let mut nbp: usize = 0;
        for (p_id, neighbours) in &neighbourhood_map {
            let point = resolve_in_layers(&points_by_layer, *p_id)?;
            for (l, neighbours) in neighbours.iter().enumerate() {
                for n in neighbours {
                    // A neighbour identity is file-controlled: resolve it
                    // through a checked lookup so a dangling reference is a
                    // typed error instead of an indexing panic.
                    let n_point = resolve_in_layers(&points_by_layer, n.p_id)?;
                    // now n_point is the Arc<Point> corresponding to neighbour n of point,
                    // construct a corresponding PointWithOrder
                    let n_pwo = PointWithOrder::<T>::new(n_point, n.distance);
                    point.neighbours.write()[l].push(Arc::new(n_pwo));
                } // end of for n
                //  must sort
                point.neighbours.write()[l].sort_unstable();
            } // end of for l
            nbp += 1;
            if nbp.is_multiple_of(500_000) {
                debug!("reloading nb_points neighbourhood completed : {}", nbp);
            }
        } // end loop in neighbourhood_map
        //
        // get id of entry_point
        // load entry point
        info!(
            "end of layer loading, allocating PointIndexation, nb points loaded {:?}",
            nb_points_loaded
        );
        //
        let mut it_slice = [0u8; std::mem::size_of::<DataId>()];
        graph_in.read_exact(&mut it_slice)?;
        let origin_id = DataId::from_ne_bytes(it_slice);
        //
        let mut it_slice = [0u8; ::std::mem::size_of::<u8>()];
        graph_in.read_exact(&mut it_slice)?;
        let layer = u8::from_ne_bytes(it_slice);
        //
        let mut it_slice = [0u8; std::mem::size_of::<i32>()];
        graph_in.read_exact(&mut it_slice)?;
        let rank_in_l = i32::from_ne_bytes(it_slice);
        //
        info!(
            "found entry point, origin_id {:?} , layer {:?}, rank in layer {:?} ",
            origin_id, layer, rank_in_l
        );
        let entry_point = Arc::clone(&points_by_layer[layer as usize][rank_in_l as usize]);
        info!(
            " loaded entry point, origin_id {:} p_id {:?}",
            entry_point.get_origin_id(),
            entry_point.get_point_id()
        );
        //
        let point_indexation = PointIndexation {
            max_nb_connection: descr.max_nb_connection as usize,
            max_layer: NB_LAYER_MAX as usize,
            points_by_layer: Arc::new(RwLock::new(points_by_layer)),
            layer_g: LayerGenerator::new_with_scale(
                descr.max_nb_connection as usize,
                descr.level_scale,
                NB_LAYER_MAX as usize,
            ),
            nb_point: Arc::new(RwLock::new(nb_points_loaded)), // CAVEAT , we should increase , the whole thing is to be able to increment graph ?
            entry_point: Arc::new(RwLock::new(Some(entry_point))),
        };
        //
        debug!("Exiting load_pointIndexation");
        Ok(point_indexation)
    } // end of load_pointIndexation

    //
    //  Reload a point from a dump.
    //
    //  The graph part is loaded from graph_in file
    // the data vector itself is loaded from data_in
    //
    #[allow(clippy::type_complexity)]
    fn load_point<'b, 'a, T>(
        &'a self,
        graph_in: &mut dyn Read,
        descr: &Description,
        data_in: &mut dyn Read,
        point_use_mmap: bool,
    ) -> Result<(Arc<Point<'b, T>>, Vec<Vec<Neighbour>>)>
    where
        T: 'static + DeserializeOwned + Clone + Sized + Send + Sync + std::fmt::Debug,
        'a: 'b,
    {
        //
        //    debug!(" point load {:?} {:?}  ", p_id, origin_id);
        // Now  for each layer , read neighbours
        let (origin_id, p_id, neighborhood) = load_point_graph(graph_in, descr).map_err(|e| {
            error!("load_point error reading graph data for point p_id");
            anyhow!("error reading graph data for point: {}", e)
        })?;
        //
        let point = match point_use_mmap {
            false => {
                let v = load_point_data::<T>(origin_id, data_in, descr)
                    .map_err(|e| anyhow!("loading data for point {}: {}", origin_id, e))?;
                Point::<T>::new(v, origin_id, p_id)
            }
            true => {
                skip_point_data(origin_id, data_in, descr)?; // keep cohrence between data file and graph file!
                debug!("constructing point from datamap, dataid : {:?}", origin_id);
                let datamap = self.datamap.as_ref().ok_or_else(|| {
                    anyhow!(
                        "mmap reload requested for point {} without a datamap",
                        origin_id
                    )
                })?;
                let s: &'b [T] = datamap.get_data::<T>(&origin_id).ok_or_else(|| {
                    anyhow!("data file has no mmap entry for point {}", origin_id)
                })?;
                Point::<T>::new_from_mmap(s, origin_id, p_id)
            }
        };
        self.nb_point_loaded.fetch_add(1, Ordering::Relaxed);
        trace!(
            "load_point  origin {:?} allocated size {:?}, dim {:?}",
            origin_id,
            point.get_v().len(),
            descr.dimension
        );
        //
        Ok((Arc::new(point), neighborhood))
    } // end of load_point
} // end of Hnswio

/// structure describing main parameters for hnsnw data and written at the beginning of a dump file.
///
/// Name of distance and type of data must be encoded in the dump file for a coherent reload.
#[repr(C)]
pub struct Description {
    /// to keep track of format version
    pub format_version: usize,
    ///  value is 1 for Full 0 for Light
    pub dumpmode: u8,
    /// max number of connections in layers != 0
    pub max_nb_connection: u8,
    /// scale used in level sampling
    pub level_scale: f64,
    /// number of observed layers
    pub nb_layer: u8,
    /// search parameter
    pub ef: usize,
    /// total number of points
    pub nb_point: usize,
    /// data dimension
    pub dimension: usize,
    /// name of distance
    pub distname: String,
    /// T typename
    pub t_name: String,
}

impl Description {
    /// The dump of Description consists in :
    /// . The value MAGICDESCR_* as a u32 (4 u8)
    /// . The type of dump as u8
    /// . max_nb_connection as u8
    /// . ef (search parameter used in construction) as usize
    /// . nb_point (the number points dumped) as a usize
    /// . the name of distance used. (nb byes as a usize then list of bytes)
    ///
    fn dump<W: Write>(&self, argmode: DumpMode, out: &mut BufWriter<W>) -> Result<i32> {
        info!("in dump of description");
        out.write_all(&MAGICDESCR_4.to_ne_bytes())?;
        let mode: u8 = match argmode {
            DumpMode::Full => 1,
            _ => 0,
        };
        // CAVEAT should check mode == self.mode
        out.write_all(&mode.to_ne_bytes())?;
        // dump of max_nb_connection as u8!!
        out.write_all(&self.max_nb_connection.to_ne_bytes())?;
        // with MAGICDESCR_4 we must dump self.level_scale
        out.write_all(&self.level_scale.to_ne_bytes())?;
        //
        out.write_all(&self.nb_layer.to_ne_bytes())?;
        if self.nb_layer != NB_LAYER_MAX {
            println!("dump of Description, nb_layer != NB_MAX_LAYER");
            return Err(anyhow!("dump of Description, nb_layer != NB_MAX_LAYER"));
        }
        //
        info!("dumping ef {:?}", self.ef);
        out.write_all(&self.ef.to_ne_bytes())?;
        //
        info!("dumping nb point {:?}", self.nb_point);
        out.write_all(&self.nb_point.to_ne_bytes())?;
        //
        info!("dumping dimension of data {:?}", self.dimension);
        out.write_all(&self.dimension.to_ne_bytes())?;

        // dump of distance name
        let namelen: usize = self.distname.len();
        info!("distance name {:?} ", self.distname);
        out.write_all(&namelen.to_ne_bytes())?;
        out.write_all(self.distname.as_bytes())?;
        // dump of T value typename
        let namelen: usize = self.t_name.len();
        info!("T name {:?} ", self.t_name);
        out.write_all(&namelen.to_ne_bytes())?;
        out.write_all(self.t_name.as_bytes())?;
        //
        Ok(1)
    } // end fo dump

    /// return data typename
    pub fn get_typename(&self) -> String {
        self.t_name.clone()
    }

    /// returns dimension of data
    pub fn get_dimension(&self) -> usize {
        self.dimension
    }
} // end of HnswIO impl for Descr

//

/// This method is internally used by Hnswio.  
/// It is make *pub* as it can be used to retrieve the description of a dump.
/// It takes as input the graph part of the dump.
/// Validate the data file's own header against the description.
///
/// The data file repeats the magic and the dimension; both must agree with
/// the graph file's description, because every later payload length is
/// computed from the description while the bytes come from here. A
/// disagreement means the two files are not a matched pair and the load
/// must stop before any point is reconstructed.
fn check_data_header(magic: u32, data_in: &mut dyn Read, descr: &Description) -> Result<()> {
    if magic != MAGICDATAP {
        return Err(anyhow!(
            "bad magic at data file beginning: expected {:x}, got {:x}",
            MAGICDATAP,
            magic
        ));
    }
    let mut it_slice = [0u8; std::mem::size_of::<usize>()];
    data_in.read_exact(&mut it_slice)?;
    let dimension = usize::from_ne_bytes(it_slice);
    if dimension != descr.dimension {
        return Err(anyhow!(
            "data file declares dimension {} but the description declares {}",
            dimension,
            descr.dimension
        ));
    }
    Ok(())
}

/// Largest vector dimension a dump may declare.
///
/// The dimension multiplies into every per-point buffer length, so an
/// unbounded value turns a few header bytes into an arbitrary allocation
/// demand. The ceiling is far above any real embedding width.
const MAX_DIMENSION: usize = 1 << 24;

pub fn load_description(io_in: &mut dyn Read) -> Result<Description> {
    //
    let mut descr = Description {
        format_version: 0,
        dumpmode: 0,
        max_nb_connection: 0,
        level_scale: 1.0f64,
        nb_layer: 0,
        ef: 0,
        nb_point: 0,
        dimension: 0,
        distname: String::from(""),
        t_name: String::from(""),
    };
    //
    let mut it_slice = [0u8; std::mem::size_of::<u32>()];
    io_in.read_exact(&mut it_slice)?;
    let magic = u32::from_ne_bytes(it_slice);
    debug!(" magic {:X} ", magic);
    match magic {
        MAGICDESCR_2 => {
            descr.format_version = 2;
        }
        MAGICDESCR_3 => {
            descr.format_version = 3;
        }
        MAGICDESCR_4 => {
            descr.format_version = 4;
        }
        _ => {
            error!("bad magic");
            return Err(anyhow!("bad magic at descr beginning"));
        }
    }
    let mut it_slice = [0u8; std::mem::size_of::<u8>()];
    io_in.read_exact(&mut it_slice)?;
    descr.dumpmode = u8::from_ne_bytes(it_slice);
    info!(" dumpmode {:?} ", descr.dumpmode);
    //
    let mut it_slice = [0u8; std::mem::size_of::<u8>()];
    io_in.read_exact(&mut it_slice)?;
    descr.max_nb_connection = u8::from_ne_bytes(it_slice);
    info!(" max_nb_connection {:?} ", descr.max_nb_connection);
    //
    if descr.format_version == 4 {
        // we read modification for level sampling
        let mut it_slice = [0u8; std::mem::size_of::<f64>()];
        io_in.read_exact(&mut it_slice)?;
        descr.level_scale = f64::from_ne_bytes(it_slice);
        info!(" level scale : {:.2e}", descr.level_scale);
    }
    //
    let mut it_slice = [0u8; std::mem::size_of::<u8>()];
    io_in.read_exact(&mut it_slice)?;
    descr.nb_layer = u8::from_ne_bytes(it_slice);
    info!("nb_layer  {:?} ", descr.nb_layer);
    // ef
    let mut it_slice = [0u8; std::mem::size_of::<usize>()];
    io_in.read_exact(&mut it_slice)?;
    descr.ef = usize::from_ne_bytes(it_slice);
    info!("ef  {:?} ", descr.ef);
    // nb_point
    let mut it_slice = [0u8; std::mem::size_of::<usize>()];
    io_in.read_exact(&mut it_slice)?;
    descr.nb_point = usize::from_ne_bytes(it_slice);
    // read dimension
    let mut it_slice = [0u8; std::mem::size_of::<usize>()];
    io_in.read_exact(&mut it_slice)?;
    descr.dimension = usize::from_ne_bytes(it_slice);
    info!(
        "nb_point {:?} dimension {:?} ",
        descr.nb_point, descr.dimension
    );
    // distance name
    let mut it_slice = [0u8; std::mem::size_of::<usize>()];
    io_in.read_exact(&mut it_slice)?;
    let len: usize = usize::from_ne_bytes(it_slice);
    debug!("length of distance name {:?} ", len);
    if len > 256 {
        info!(" length of distance name > 256");
        println!(" length of distance name should not exceed 256");
        return Err(anyhow!("bad length for distance name"));
    }
    let mut distv = vec![0; len];
    io_in.read_exact(distv.as_mut_slice())?;
    let distname =
        String::from_utf8(distv).map_err(|e| anyhow!("distance name is not valid UTF-8: {}", e))?;
    debug!("distance name {:?} ", distname);
    descr.distname = distname;
    // reload of type name
    let mut it_slice = [0u8; std::mem::size_of::<usize>()];
    io_in.read_exact(&mut it_slice)?;
    let len: usize = usize::from_ne_bytes(it_slice);
    debug!("length of T  name {:?} ", len);
    if len > 256 {
        println!(" length of T name should not exceed 256");
        return Err(anyhow!("bad lenght for T name"));
    }
    let mut tnamev = vec![0; len];
    io_in.read_exact(tnamev.as_mut_slice())?;
    let t_name =
        String::from_utf8(tnamev).map_err(|e| anyhow!("T type name is not valid UTF-8: {}", e))?;
    debug!("T type name {:?} ", t_name);
    descr.t_name = t_name;
    debug!(" end of description load \n");
    //
    // Every later stage sizes buffers and indexes tables from these fields,
    // so validate them once here rather than at each use.
    if descr.nb_layer > NB_LAYER_MAX {
        return Err(anyhow!(
            "description declares {} layers, above the {} maximum",
            descr.nb_layer,
            NB_LAYER_MAX
        ));
    }
    if descr.dimension > MAX_DIMENSION {
        return Err(anyhow!(
            "description declares dimension {}, above the {} maximum",
            descr.dimension,
            MAX_DIMENSION
        ));
    }
    if !descr.level_scale.is_finite() || descr.level_scale <= 0.0 {
        return Err(anyhow!(
            "description declares a non-positive or non-finite level scale {}",
            descr.level_scale
        ));
    }
    //
    Ok(descr)
}

//
// dump and load of Point<T>
// ==========================
//

///  Graph part of point dump
/// dump of a point consist in  
///  1. The value MAGICPOINT
///  2. its identity ( a usize  rank in original data , hash value or else , and PointId)
///  3. for each layer dump of the number of neighbours followed by :
///     for each neighbour dump of its identity (: usize) and then distance (): u32) to point dumped.
///
/// identity of a point is in full mode the triplet origin_id (: usize), layer (: u8) rank_in_layer (: u32)
///                           light mode only origin_id (: usize)
///  For data dump
///  1. The value MAGICDATAP (u32)
///  2. origin_id as a u64
///  3. The vector of data (the length is known from Description)
///
fn dump_point<T: Serialize + Clone + Sized + Send + Sync, W: Write>(
    point: &Point<T>,
    mode: DumpMode,
    graphout: &mut BufWriter<W>,
    dataout: &mut BufWriter<W>,
) -> Result<i32> {
    //
    graphout.write_all(&MAGICPOINT.to_ne_bytes())?;
    // dump ext_id: usize , layer : u8 , rank in layer : i32
    graphout.write_all(&point.get_origin_id().to_ne_bytes())?;
    let p_id = point.get_point_id();
    if mode == DumpMode::Full {
        graphout.write_all(&p_id.0.to_ne_bytes())?;
        graphout.write_all(&p_id.1.to_ne_bytes())?;
    }
    trace!(" point dump {:?} {:?}  ", p_id, point.get_origin_id());
    // then dump neighborhood info : nb neighbours : u32 , then list of origin_id, layer, rank_in_layer
    let neighborhood = point.get_neighborhood_id();
    // in any case nb_layers are dumped with possibly 0 neighbours at a layer, but this does not occur by construction
    for (l, neighbours_at_l) in neighborhood.iter().enumerate() {
        // Caution : we dump number of neighbours as a usize, even if it cannot be so large!
        let nbg_l: usize = neighbours_at_l.len();
        trace!("\t dumping nbng : {} at l {}", nbg_l, l);
        graphout.write_all(&nbg_l.to_ne_bytes())?;
        for n in neighbours_at_l {
            // dump d_id : uszie , distance : f32, layer : u8, rank in layer : i32
            graphout.write_all(&n.d_id.to_ne_bytes())?;
            if mode == DumpMode::Full {
                graphout.write_all(&n.p_id.0.to_ne_bytes())?;
                graphout.write_all(&n.p_id.1.to_ne_bytes())?;
            }
            graphout.write_all(&n.distance.to_ne_bytes())?;
            //                debug!("        voisins  {:?}  {:?}  {:?}", n.p_id,  n.d_id , n.distance);
        }
    }
    // now we dump data vector!
    dataout.write_all(&MAGICDATAP.to_ne_bytes())?;
    let origin_u64 = point.get_origin_id() as u64;
    dataout.write_all(&origin_u64.to_ne_bytes())?;
    //
    let serialized = unsafe {
        std::slice::from_raw_parts(
            point.get_v().as_ptr() as *const u8,
            std::mem::size_of_val(point.get_v()),
        )
    };
    trace!("serializing len {:?}", serialized.len());
    let len_64 = serialized.len() as u64;
    dataout.write_all(&len_64.to_ne_bytes())?;
    dataout.write_all(serialized)?;
    //
    Ok(1)
} // end of dump for Point<T>

/// Upper bound on a single serialized point payload, in bytes.
///
/// The `NoData` case and any future format carry no description-derived
/// length, so they need a standalone ceiling; v3/v4 payload lengths are
/// additionally pinned to exactly `dimension * size_of::<T>()`. Without a
/// ceiling, an 8-byte hostile length field would drive an unbounded
/// allocation before a single data byte is read.
const MAX_SERIALIZED_POINT_BYTES: u64 = 1 << 30;

/// Validate a declared payload length against the description, returning it
/// as a `usize` that is safe to allocate.
///
/// For the raw-binary formats (v3/v4) the length is fully determined by the
/// description, so anything else means the pair of files disagree and the
/// point must be rejected rather than reconstructed from a mismatched
/// buffer.
fn check_serialized_len<T: 'static>(
    serialized_len: u64,
    descr: &Description,
    origin_id: usize,
) -> Result<usize> {
    if serialized_len > MAX_SERIALIZED_POINT_BYTES {
        return Err(anyhow!(
            "point {} declares {} serialized bytes, above the {} byte ceiling",
            origin_id,
            serialized_len,
            MAX_SERIALIZED_POINT_BYTES
        ));
    }
    let serialized_len = usize::try_from(serialized_len)
        .map_err(|_| anyhow!("point {} declares a length exceeding usize", origin_id))?;
    let is_no_data = std::any::TypeId::of::<T>() == std::any::TypeId::of::<NoData>();
    if !is_no_data && matches!(descr.format_version, 3 | 4) {
        let expected = descr
            .dimension
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| anyhow!("dimension {} overflows a byte length", descr.dimension))?;
        if serialized_len != expected {
            return Err(anyhow!(
                "point {} declares {} serialized bytes but the description implies {} \
                 ({} elements of {} bytes)",
                origin_id,
                serialized_len,
                expected,
                descr.dimension,
                std::mem::size_of::<T>()
            ));
        }
    }
    Ok(serialized_len)
}

/// Decode `dimension` values of `T` from a raw little-endian-by-construction
/// binary payload.
///
/// The v3/v4 dump writes the in-memory bytes of `T` directly, so reload has
/// to reinterpret them. Two properties matter and are why this is not a
/// `slice::from_raw_parts` cast:
///
/// - **Length.** The caller must have validated `bytes.len()` against
///   `dimension * size_of::<T>()` (see [`check_serialized_len`]); this
///   function additionally refuses to read past the buffer, so a mismatch
///   can never become an out-of-bounds read.
/// - **Alignment.** `bytes` comes from a `Vec<u8>` (alignment 1) while `T`
///   may require greater alignment, so each element is read with
///   `read_unaligned` instead of through a misaligned `*const T`.
///
/// The formats remain plain-data-only by construction: the dump side writes
/// `T`'s raw bytes, so a `T` owning heap memory was never representable in
/// these versions.
fn decode_raw_binary_vector<T>(bytes: &[u8], dimension: usize) -> Vec<T> {
    let width = std::mem::size_of::<T>();
    let mut values = Vec::with_capacity(dimension);
    if width == 0 {
        return values;
    }
    for index in 0..dimension {
        let Some(offset) = index.checked_mul(width) else {
            break;
        };
        let Some(end) = offset.checked_add(width) else {
            break;
        };
        if end > bytes.len() {
            break;
        }
        // SAFETY: `offset + width <= bytes.len()` was just checked, so the
        // read stays inside the buffer, and `read_unaligned` imposes no
        // alignment requirement on the source pointer.
        let value = unsafe { std::ptr::read_unaligned(bytes.as_ptr().add(offset) as *const T) };
        values.push(value);
    }
    values
}

// just reload data vector for point from file where data were dumped
// used when we do not used memory map in reload
fn load_point_data<T>(
    origin_id: usize,
    data_in: &mut dyn Read,
    descr: &Description,
) -> Result<Vec<T>>
where
    T: 'static + DeserializeOwned + Clone + Sized + Send + Sync,
{
    //
    trace!("load_point_data , origin id : {}", origin_id);
    //
    // construct a point from data_in
    //
    let mut it_slice = [0u8; std::mem::size_of::<u32>()];
    data_in.read_exact(&mut it_slice)?;
    let magic = u32::from_ne_bytes(it_slice);
    if magic != MAGICDATAP {
        return Err(anyhow!(
            "bad magic in data file for point {}: expected {:x}, got {:x}",
            origin_id,
            MAGICDATAP,
            magic
        ));
    }
    // read origin id
    let mut it_slice = [0u8; std::mem::size_of::<u64>()];
    data_in.read_exact(&mut it_slice)?;
    let origin_id_data = u64::from_ne_bytes(it_slice) as usize;
    if origin_id != origin_id_data {
        return Err(anyhow!(
            "origin_id incoherent between graph and data: graph {}, data {}",
            origin_id,
            origin_id_data
        ));
    }
    // now read data. we use size_t that is in description, to take care of the casewhere we reload
    let mut it_slice = [0u8; std::mem::size_of::<u64>()];
    data_in.read_exact(&mut it_slice)?;
    let serialized_len = u64::from_ne_bytes(it_slice);
    trace!("serialized len to reload {:?}", serialized_len);
    let serialized_len = check_serialized_len::<T>(serialized_len, descr, origin_id)?;
    // Read into a capacity-bounded buffer rather than pre-allocating the
    // declared length: a truncated file then fails in read_exact instead of
    // letting a hostile length reserve gigabytes before any I/O happens.
    let mut v_serialized = vec![0u8; serialized_len];
    data_in.read_exact(&mut v_serialized)?;

    let v: Vec<T> = if std::any::TypeId::of::<T>() != std::any::TypeId::of::<NoData>() {
        match descr.format_version {
            2 => {
                error!("format bincode of dump no more used");
                return Err(anyhow!(
                    "dump for point {} uses the retired bincode format (version 2)",
                    origin_id
                ));
            }
            3 | 4 => decode_raw_binary_vector::<T>(&v_serialized, descr.dimension),
            other => {
                return Err(anyhow!(
                    "unknown format_version {} while loading point {}",
                    other,
                    origin_id
                ));
            }
        }
    } else {
        Vec::new()
    };
    //
    Ok(v)
} // end of load_point_data

// We need to maintain coherence in data and graph stream, so we read to keep in phase
fn skip_point_data(origin_id: usize, data_in: &mut dyn Read, descr: &Description) -> Result<()> {
    //
    let mut it_slice = [0u8; std::mem::size_of::<u32>()];
    data_in.read_exact(&mut it_slice)?;
    let magic = u32::from_ne_bytes(it_slice);
    if magic != MAGICDATAP {
        return Err(anyhow!(
            "bad magic in data file while skipping point {}: expected {:x}, got {:x}",
            origin_id,
            MAGICDATAP,
            magic
        ));
    }
    // read origin id
    let mut it_slice = [0u8; std::mem::size_of::<u64>()];
    data_in.read_exact(&mut it_slice)?;
    let origin_id_data = u64::from_ne_bytes(it_slice) as usize;
    if origin_id != origin_id_data {
        return Err(anyhow!(
            "origin_id incoherent between graph and data while skipping: graph {}, data {}",
            origin_id,
            origin_id_data
        ));
    }
    //
    // now read data. we use size_t that is in description, to take care of the casewhere we reload
    let mut it_slice = [0u8; std::mem::size_of::<u64>()];
    data_in.read_exact(&mut it_slice)?;
    let serialized_len = u64::from_ne_bytes(it_slice);
    trace!(
        "skip_point_data : serialized len to reload {:?}",
        serialized_len
    );
    // The skipped payload is never interpreted, so bound it by the ceiling
    // alone; consuming it through a fixed-size scratch buffer keeps a hostile
    // length from reserving memory it never fills.
    if serialized_len > MAX_SERIALIZED_POINT_BYTES {
        return Err(anyhow!(
            "point {} declares {} serialized bytes, above the {} byte ceiling",
            origin_id,
            serialized_len,
            MAX_SERIALIZED_POINT_BYTES
        ));
    }
    let _ = descr;
    let mut remaining = serialized_len;
    let mut scratch = [0u8; 8192];
    while remaining > 0 {
        let take = remaining.min(scratch.len() as u64) as usize;
        data_in.read_exact(&mut scratch[..take])?;
        remaining -= take as u64;
    }
    //
    Ok(())
} // end of skip_point_data

//==================================================================================

/// This structure gathers info loaded in dumped graph file for a point.
type PointGraphInfo = (usize, PointId, Vec<Vec<Neighbour>>);

/// Capacity hint used when reading a neighbour list, independent of the
/// count the file declares (see the call site for why).
const RESERVE_NEIGHBOUR_HINT: usize = 64;

/// Capacity hint used when reading a layer's point list, bounded for the
/// same reason as [`RESERVE_NEIGHBOUR_HINT`].
const RESERVE_POINT_HINT: usize = 4096;

/// Resolve a point identity against the reloaded layer tables.
///
/// Both the identity of a point and the identities inside its neighbour
/// lists come from the file, so every lookup is checked: a layer or rank
/// that does not name a loaded point yields a typed error rather than an
/// out-of-bounds panic.
fn resolve_in_layers<'layers, 'point, T: Clone + Send + Sync>(
    points_by_layer: &'layers [Vec<Arc<Point<'point, T>>>],
    p_id: PointId,
) -> Result<&'layers Arc<Point<'point, T>>> {
    let layer = points_by_layer.get(p_id.0 as usize).ok_or_else(|| {
        anyhow!(
            "point identity names layer {} but only {} layers were loaded",
            p_id.0,
            points_by_layer.len()
        )
    })?;
    if p_id.1 < 0 {
        return Err(anyhow!(
            "point identity names a negative rank {} in layer {}",
            p_id.1,
            p_id.0
        ));
    }
    layer.get(p_id.1 as usize).ok_or_else(|| {
        anyhow!(
            "point identity names rank {} in layer {} but that layer loaded {} points",
            p_id.1,
            p_id.0,
            layer.len()
        )
    })
}

/// Largest neighbour count a single layer of a well-formed dump can hold.
///
/// Construction caps a neighbourhood at `max_nb_connection` (doubled at
/// layer 0 by the usual HNSW rule), and no layer can hold more neighbours
/// than the index holds points. Taking the larger of the two keeps honest
/// dumps loadable — including ones written by a build with a different
/// connection policy — while still bounding a hostile count.
fn max_neighbours_per_layer(descr: &Description) -> usize {
    let by_connection = (descr.max_nb_connection as usize).saturating_mul(2);
    by_connection.max(descr.nb_point)
}

/// Reject a point identity that cannot address a layer table.
///
/// Reconstruction indexes `points_by_layer[layer][rank]`, so a layer at or
/// above `NB_LAYER_MAX`, a negative rank, or a rank beyond the point count
/// must fail here as a typed error. Left to the indexing site they become a
/// panic (or, for a negative rank widened to `usize`, an absurd index) on
/// purely file-controlled values.
fn check_point_id(layer: u8, rank_in_layer: i32, descr: &Description) -> Result<()> {
    if layer >= NB_LAYER_MAX {
        return Err(anyhow!(
            "point identity names layer {}, at or above the {} layer maximum",
            layer,
            NB_LAYER_MAX
        ));
    }
    if rank_in_layer < 0 {
        return Err(anyhow!(
            "point identity names a negative rank {} in layer {}",
            rank_in_layer,
            layer
        ));
    }
    if descr.nb_point > 0 && rank_in_layer as usize >= descr.nb_point {
        return Err(anyhow!(
            "point identity names rank {} in layer {}, beyond the {} points the description \
             declares",
            rank_in_layer,
            layer,
            descr.nb_point
        ));
    }
    Ok(())
}

// This function reads neighbourhood info and returns neighbourhood info.
// It suppose and requires that the file graph_in is just at beginning of info related to origin_id
fn load_point_graph(graph_in: &mut dyn Read, descr: &Description) -> Result<PointGraphInfo> {
    //
    trace!("in load_point_graph");
    // read and check magic
    let mut it_slice = [0u8; std::mem::size_of::<u32>()];
    graph_in.read_exact(&mut it_slice)?;
    let magic = u32::from_ne_bytes(it_slice);
    if magic != MAGICPOINT {
        error!("got instead of MAGICPOINT {:x}", magic);
        return Err(anyhow!("bad magic at point beginning"));
    }
    let mut it_slice = [0u8; std::mem::size_of::<DataId>()];
    graph_in.read_exact(&mut it_slice)?;
    let origin_id = DataId::from_ne_bytes(it_slice);
    //
    // read point_id
    let mut it_slice = [0u8; std::mem::size_of::<u8>()];
    graph_in.read_exact(&mut it_slice)?;
    let layer = u8::from_ne_bytes(it_slice);
    //
    let mut it_slice = [0u8; std::mem::size_of::<i32>()];
    graph_in.read_exact(&mut it_slice)?;
    let rank_in_l = i32::from_ne_bytes(it_slice);
    // A point identity is used to index the layer tables during
    // reconstruction, so reject out-of-range identities at the parse
    // boundary rather than letting them become an indexing panic later.
    check_point_id(layer, rank_in_l, descr)?;
    let p_id = PointId(layer, rank_in_l);
    debug!(
        "in load_point_graph, got origin_id : {}, p_id : {:?}",
        origin_id, p_id
    );
    //
    // Now  for each layer , read neighbours
    let nb_layer = descr.nb_layer;
    // The tail loop below pads to exactly NB_LAYER_MAX entries, which only
    // holds if the description's layer count fits. load_description rejects
    // an over-large count, so this is a cheap restatement of that invariant
    // at the point that depends on it.
    if nb_layer > NB_LAYER_MAX {
        return Err(anyhow!(
            "description declares {} layers, above the {} maximum",
            nb_layer,
            NB_LAYER_MAX
        ));
    }
    let mut neighborhood = Vec::<Vec<Neighbour>>::with_capacity(NB_LAYER_MAX as usize);
    for _l in 0..nb_layer {
        let mut neighbour: Neighbour = Default::default();
        // read nb_neighbour as usize!!! CAUTION, then nb_neighbours times identity(depends on Full or Light) distance : f32
        let mut it_slice = [0u8; std::mem::size_of::<usize>()];
        graph_in.read_exact(&mut it_slice)?;
        let nb_neighbours = usize::from_ne_bytes(it_slice);
        if nb_neighbours > max_neighbours_per_layer(descr) {
            return Err(anyhow!(
                "point {} declares {} neighbours at one layer, above the {} bound implied by \
                 max_nb_connection {} and nb_point {}",
                origin_id,
                nb_neighbours,
                max_neighbours_per_layer(descr),
                descr.max_nb_connection,
                descr.nb_point
            ));
        }
        // Capacity is a hint bounded independently of the declared count: a
        // truncated file must fail in read_exact, not after reserving for
        // neighbours that are not in the file.
        let mut neighborhood_l: Vec<Neighbour> =
            Vec::with_capacity(nb_neighbours.min(RESERVE_NEIGHBOUR_HINT));
        for _j in 0..nb_neighbours {
            let mut it_slice = [0u8; std::mem::size_of::<DataId>()];
            graph_in.read_exact(&mut it_slice)?;
            neighbour.d_id = DataId::from_ne_bytes(it_slice);
            if descr.dumpmode == 1 {
                let mut it_slice = [0u8; std::mem::size_of::<u8>()];
                graph_in.read_exact(&mut it_slice)?;
                neighbour.p_id.0 = u8::from_ne_bytes(it_slice);
                //
                let mut it_slice = [0u8; std::mem::size_of::<i32>()];
                graph_in.read_exact(&mut it_slice)?;
                neighbour.p_id.1 = i32::from_ne_bytes(it_slice);
                check_point_id(neighbour.p_id.0, neighbour.p_id.1, descr)?;
            }
            let mut it_slice = [0u8; std::mem::size_of::<f32>()];
            graph_in.read_exact(&mut it_slice)?;
            neighbour.distance = f32::from_ne_bytes(it_slice);
            //  debug!("        voisins  load {:?} {:?} {:?} ", neighbour.p_id, neighbour.d_id , neighbour.distance);
            // now we have a new neighbour, we must really fill neighbourhood info, so it means going from Neighbour to PointWithOrder
            neighborhood_l.push(neighbour);
        }
        neighborhood.push(neighborhood_l);
    }
    for _l in nb_layer..NB_LAYER_MAX {
        neighborhood.push(Vec::<Neighbour>::new());
    }
    //
    let point_grap_info = (origin_id, p_id, neighborhood);
    //
    Ok(point_grap_info)
} // end of load_point_graph

//
// dump and load of PointIndexation<T>
// ===================================
//
//
// nb_layer : 8
// a magick at each Layer : u32
// . number of points in layer (usize),
// . list of point of layer
// dump entry point
//
impl<T: Serialize + DeserializeOwned + Clone + Send + Sync> HnswIoT for PointIndexation<'_, T> {
    fn dump(&self, mode: DumpMode, dumpinit: &mut DumpInit) -> Result<i32> {
        let graphout = &mut dumpinit.graph_out;
        let dataout = &mut dumpinit.data_out;
        // dump max_layer
        let layers = self.points_by_layer.read();
        let nb_layer = layers.len() as u8;
        graphout.write_all(&nb_layer.to_ne_bytes())?;
        // dump layers from lower (most populatated to higher level)
        for i in 0..layers.len() {
            let nb_point = layers[i].len();
            debug!("dumping layer {:?}, nb_point {:?}", i, nb_point);
            graphout.write_all(&MAGICLAYER.to_ne_bytes())?;
            graphout.write_all(&nb_point.to_ne_bytes())?;
            for j in 0..layers[i].len() {
                assert_eq!(layers[i][j].get_point_id(), PointId(i as u8, j as i32));
                dump_point(&layers[i][j], mode, graphout, dataout)?;
            }
        }
        // dump id of entry point
        let ep_read = self.entry_point.read();
        let ep = ep_read
            .as_ref()
            .ok_or(anyhow!("entry point not initialized"))?;
        //let ep = ep_read.as_ref().unwrap();
        graphout.write_all(&ep.get_origin_id().to_ne_bytes())?;
        let p_id = ep.get_point_id();
        if mode == DumpMode::Full {
            graphout.write_all(&p_id.0.to_ne_bytes())?;
            graphout.write_all(&p_id.1.to_ne_bytes())?;
        }
        info!(
            "dumped entry_point origin_d {:?}, p_id {:?} ",
            ep.get_origin_id(),
            p_id
        );
        //
        Ok(1)
    } // end of dump for PointIndexation<T>
} // end of impl HnswIO

//
// dump and load of Hnsw<T>
// =========================
//
//

impl<T: Serialize + DeserializeOwned + Clone + Sized + Send + Sync, D: Distance<T> + Send + Sync>
    HnswIoT for Hnsw<'_, T, D>
{
    /// The dump method for hnsw.  
    /// - graphout is a BufWriter dedicated to the dump of the graph part of Hnsw
    /// - dataout is a bufWriter dedicated to the dump of the data stored in the Hnsw structure.
    fn dump(&self, mode: DumpMode, dumpinit: &mut DumpInit) -> anyhow::Result<i32> {
        //
        let graphout = &mut dumpinit.graph_out;
        let dataout = &mut dumpinit.data_out;
        // dump description , then PointIndexation
        let dumpmode: u8 = match mode {
            DumpMode::Full => 1,
            _ => 0,
        };
        let datadim: usize = self.layer_indexed_points.get_data_dimension();
        let level_scale = self.layer_indexed_points.get_level_scale();
        let description = Description {
            format_version: 3,
            //  value is 1 for Full 0 for Light
            dumpmode,
            max_nb_connection: self.get_max_nb_connection(),
            level_scale,
            nb_layer: self.get_max_level() as u8,
            ef: self.get_ef_construction(),
            nb_point: self.get_nb_point(),
            dimension: datadim,
            distname: self.get_distance_name(),
            t_name: type_name::<T>().to_string(),
        };
        debug!("dump  obtained typename {:?}", type_name::<T>());
        description.dump(mode, graphout)?;
        // We must dump a header for dataout.
        dataout.write_all(&MAGICDATAP.to_ne_bytes())?;
        dataout.write_all(&datadim.to_ne_bytes())?;
        //
        self.layer_indexed_points.dump(mode, dumpinit)?;
        Ok(1)
    }
} // end impl block for Hnsw

//===============================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    pub use crate::api::AnnT;
    use anndists::dist;
    use log::error;

    use rand::distr::{Distribution, Uniform};

    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    fn description_for(format_version: usize, dimension: usize, nb_point: usize) -> Description {
        Description {
            format_version,
            dumpmode: 1,
            max_nb_connection: 8,
            level_scale: 1.0,
            nb_layer: 4,
            ef: 24,
            nb_point,
            dimension,
            distname: String::from("DistL1"),
            t_name: String::from("f32"),
        }
    }

    /// The raw-binary decoder must never read past the buffer, whatever the
    /// declared dimension says, and must tolerate a source buffer with no
    /// alignment guarantee (a `Vec<u8>` reinterpreted as `f32`).
    #[test]
    fn raw_binary_decode_is_bounded_and_alignment_free() {
        let values: [f32; 4] = [1.5, -2.25, 0.0, 1e30];
        let mut bytes = Vec::new();
        for value in &values {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }

        let decoded = decode_raw_binary_vector::<f32>(&bytes, 4);
        assert_eq!(decoded, values.to_vec(), "exact-length decode round-trips");

        // A dimension larger than the buffer must stop at the buffer end
        // rather than reading out of bounds.
        let truncated = decode_raw_binary_vector::<f32>(&bytes, 64);
        assert_eq!(truncated.len(), 4);

        // A partial trailing element is not a value: it must be dropped, not
        // completed with whatever follows in memory.
        let ragged = decode_raw_binary_vector::<f32>(&bytes[..bytes.len() - 1], 4);
        assert_eq!(ragged.len(), 3);

        // Decode from a deliberately misaligned offset: `f32` wants 4-byte
        // alignment and this slice starts one byte in.
        let mut offset_bytes = vec![0u8];
        offset_bytes.extend_from_slice(&bytes);
        let unaligned = decode_raw_binary_vector::<f32>(&offset_bytes[1..], 4);
        assert_eq!(unaligned, values.to_vec());
    }

    /// A payload length that disagrees with the description must be rejected
    /// before it can be used to reconstruct a point.
    #[test]
    fn serialized_length_must_match_the_description() {
        let descr = description_for(3, 8, 100);
        let exact = (8 * std::mem::size_of::<f32>()) as u64;
        assert_eq!(
            check_serialized_len::<f32>(exact, &descr, 0).expect("exact length is accepted"),
            exact as usize
        );
        assert!(
            check_serialized_len::<f32>(exact - 4, &descr, 0).is_err(),
            "a short payload must be rejected, not silently under-read"
        );
        assert!(
            check_serialized_len::<f32>(exact + 4, &descr, 0).is_err(),
            "an over-long payload means the files disagree"
        );
        assert!(
            check_serialized_len::<f32>(u64::MAX, &descr, 0).is_err(),
            "a hostile length must never reach an allocation"
        );
    }

    /// Point identities index the layer tables, so out-of-range values must
    /// be refused at the parse boundary.
    #[test]
    fn point_identities_are_range_checked() {
        let descr = description_for(4, 8, 10);
        check_point_id(0, 0, &descr).expect("a valid identity is accepted");
        check_point_id(3, 9, &descr).expect("the last valid rank is accepted");
        assert!(
            check_point_id(NB_LAYER_MAX, 0, &descr).is_err(),
            "a layer at the maximum is out of range"
        );
        assert!(
            check_point_id(0, -1, &descr).is_err(),
            "a negative rank must not widen into a huge index"
        );
        assert!(
            check_point_id(0, 10, &descr).is_err(),
            "a rank beyond the declared point count is out of range"
        );
    }

    fn my_fn(v1: &[f32], v2: &[f32]) -> f32 {
        let norm_l1: f32 = v1.iter().zip(v2.iter()).map(|t| (*t.0 - *t.1).abs()).sum();
        norm_l1
    }

    #[test]
    fn test_dump_reload_1() {
        println!("\n\n test_dump_reload_1");
        log_init_test();
        // generate a random test
        let mut rng = rand::rng();
        let unif = Uniform::<f32>::new(0., 1.).unwrap();
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
        let ef_construct = 25;
        let nb_connection = 10;
        let hnsw = Hnsw::<f32, dist::DistL1>::new(
            nb_connection,
            nbcolumn,
            16,
            ef_construct,
            dist::DistL1 {},
        );
        for (i, d) in data.iter().enumerate() {
            hnsw.insert((d, i));
        }
        // some loggin info
        hnsw.dump_layer_info();
        // dump in a file.  Must take care of name as tests runs in // !!!
        let fname = "dumpreloadtest1";
        let directory = tempfile::tempdir().unwrap();
        let _res = hnsw.file_dump(directory.path(), fname);
        //
        // reload
        debug!("\n\n test_dump_reload_1 hnsw reload");
        // we will need a procedural macro to get from distance name to its instanciation.
        // from now on we test with DistL1
        let mut reloader = HnswIo::new(directory.path(), fname);
        let hnsw_loaded: Hnsw<f32, DistL1> = reloader.load_hnsw::<f32, DistL1>().unwrap();
        // test equality
        check_graph_equality(&hnsw_loaded, &hnsw);
    } // end of test_dump_reload

    #[test]
    fn test_dump_reload_myfn() {
        println!("\n\n test_dump_reload_myfn");
        log_init_test();
        // generate a random test
        let mut rng = rand::rng();
        let unif = Uniform::<f32>::new(0., 1.).unwrap();
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
        let ef_construct = 25;
        let nb_connection = 10;
        let mydist = dist::DistPtr::<f32, f32>::new(my_fn);
        let hnsw = Hnsw::<f32, dist::DistPtr<f32, f32>>::new(
            nb_connection,
            nbcolumn,
            16,
            ef_construct,
            mydist,
        );
        for (i, d) in data.iter().enumerate() {
            hnsw.insert((d, i));
        }
        // some loggin info
        hnsw.dump_layer_info();
        let fname = "dumpreloadtest_myfn";
        let directory = tempfile::tempdir().unwrap();

        let _res = hnsw.file_dump(directory.path(), fname);
        // This will dump in 2 files named dumpreloadtest.hnsw.graph and dumpreloadtest.hnsw.data
        //
        // reload
        debug!("HNSW reload");
        let reloader = HnswIo::new(directory.path(), fname);
        let mydist = dist::DistPtr::<f32, f32>::new(my_fn);
        let _hnsw_loaded: Hnsw<f32, DistPtr<f32, f32>> =
            reloader.load_hnsw_with_dist(mydist).unwrap();
    } // end of test_dump_reload_myfn

    #[test]
    fn test_dump_reload_graph_only() {
        println!("\n\n test_dump_reload_graph_only");
        log_init_test();
        // generate a random test
        let mut rng = rand::rng();
        let unif = Uniform::<f32>::new(0., 1.).unwrap();
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
        let ef_construct = 25;
        let nb_connection = 10;
        let hnsw = Hnsw::<f32, dist::DistL1>::new(
            nb_connection,
            nbcolumn,
            16,
            ef_construct,
            dist::DistL1 {},
        );
        for (i, d) in data.iter().enumerate() {
            hnsw.insert((d, i));
        }
        // some loggin info
        hnsw.dump_layer_info();
        // dump in a file. Must take care of name as tests runs in // !!!
        let fname = "dumpreloadtestgraph";
        let directory = tempfile::tempdir().unwrap();
        let _res = hnsw.file_dump(directory.path(), fname);
        // This will dump in 2 files named dumpreloadtest.hnsw.graph and dumpreloadtest.hnsw.data
        //
        // reload
        debug!("\n\n  hnsw reload");
        let mut reloader = HnswIo::new(directory.path(), fname);
        let hnsw_loaded: Hnsw<NoData, NoDist> = reloader.load_hnsw().unwrap();
        // test equality
        check_graph_equality(&hnsw_loaded, &hnsw);
    } // end of test_dump_reload

    // this tests reloads a dump with memory mapping of data, inserts new data and redump
    #[test]
    fn reload_with_mmap() {
        println!("\n\n hnswio tests : reload_with_mmap");
        log_init_test();
        // generate a random test
        let mut rng = rand::rng();
        let unif = Uniform::<f32>::new(0., 1.).unwrap();
        // 100 vectors of size 10 f32
        let nbcolumn = 100;
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
        //
        let first: Vec<f32> = data[0].clone();
        info!("data[0] = {:?}", first);
        // define hnsw
        let ef_construct = 25;
        let nb_connection = 10;
        let hnsw = Hnsw::<f32, dist::DistL1>::new(
            nb_connection,
            nbcolumn,
            16,
            ef_construct,
            dist::DistL1 {},
        );
        for (i, d) in data.iter().enumerate() {
            hnsw.insert((d, i));
        }
        // some loggin info
        hnsw.dump_layer_info();
        // dump in a file.  Must take care of name as tests runs in // !!!
        let fname = "mmapreloadtest";
        let directory = tempfile::tempdir().unwrap();
        let dumpname = hnsw.file_dump(directory.path(), fname).unwrap();
        debug!("dump succeeded in file basename : {}", dumpname);
        //
        // reload reload_with_mmap
        debug!("HNSW reload");
        let mut reloader = HnswIo::new(directory.path(), &dumpname);
        // use mmap for points after half number of points
        let options = ReloadOptions::default().set_mmap_threshold(nbcolumn / 2);
        reloader.set_options(options);
        let hnsw_loaded: Hnsw<f32, DistL1> = reloader.load_hnsw::<f32, DistL1>().unwrap();
        // test equality
        check_graph_equality(&hnsw_loaded, &hnsw);
        // We add nbcolumn new vectors
        info!("adding points in hnsw reloaded");
        let nbcolumn = 5;
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
        let first_with_mmap: Vec<f32> = data[0].clone();
        info!(
            "first added after reloading with mmap : data[0] = {:?}",
            first_with_mmap
        );
        let nb_in = hnsw.get_nb_point();
        for (i, d) in data.iter().enumerate() {
            hnsw.insert((d, i + nb_in));
        }
        //
        let search_res = hnsw.search(&first, 5, ef_construct);
        info!("neighbours od first point inserted");
        for n in &search_res {
            info!("neighbour: {:?}", n);
        }
        assert_eq!(search_res[0].d_id, 0);
        assert_eq!(search_res[0].distance, 0.);
        let search_res = hnsw.search(&first_with_mmap, 5, ef_construct);
        info!("neighbours of first point inserted after reload with mmap");
        for n in &search_res {
            info!("neighbour {:?}", n);
        }
        if search_res[0].d_id != nb_in {
            // with very low probability it could happen that we find a very near point!
            // then distance should very small
            info!(
                "neighbour found for point id : {}, distance : {:.2e}, should have been id : {}, dist : {:.2e}",
                search_res[0].d_id, search_res[0].distance, nb_in, 0.
            );
        }
        assert_eq!(search_res[0].d_id, nb_in);
        assert_eq!(search_res[0].distance, 0.);
        //
        // TODO: redump  and care about mmapped file, so we do not overwrite
        //
        let dump_init = DumpInit::new(directory.path(), fname, false);
        info!("will use basename : {}", dump_init.get_basename());
        let res = hnsw.file_dump(directory.path(), dump_init.get_basename());
        if res.is_err() {
            error!("hnsw.file_dump failed");
            std::panic!("hnsw.file_dump failed");
        }
    } // end of reload_with_mmap

    #[test]
    fn read_write_empty_db() -> Result<()> {
        log_init_test();
        let ef_construct = 25;
        let nb_connection = 10;
        let hnsw =
            Hnsw::<f32, dist::DistL1>::new(nb_connection, 0, 16, ef_construct, dist::DistL1 {});
        let fname = "empty_db";
        let directory = tempfile::tempdir()?;
        let _res = hnsw.file_dump(directory.path(), fname);
        let mut reloader = HnswIo::new(directory.path(), fname);
        let hnsw_loaded_res = reloader.load_hnsw::<f32, DistL1>();
        assert!(hnsw_loaded_res.is_err());
        Ok(())
    }
} // end module tests
