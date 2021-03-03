//! This file provides io dump/ reload of computed graph.
//!
//! A dump is constituted of 2 files. 
//! One file stores just the graph (or topology) with id of points.  
//! The other file stores the ids and vector in point.
//! The graph file is suffixed by "hnsw.graph" the other is suffixed by "hnsw.data"
//! 
//! An example of dump and reload of structure Hnsw is given in the tests (see test_dump_reload)
/// 
/// 
// datafile
// MAGICDATAP : u32
// dimension : u32
// The for each point the triplet: (MAGICDATAP, origin_id , array of values.) ( u32, u64, ....)
//
// A point is dumped in graph file as given by its external id (type DataId i.e : a usize, possibly a hash value) 
// and layer (u8) and rank_in_layer:i32.
// In the data file the point dump consist in the triplet: (MAGICDATAP, origin_id , array of values.)
//

use serde::{Serialize, de::DeserializeOwned};


use std::io;
use std::mem;

use parking_lot::{RwLock};
use std::sync::Arc;
use std::collections::HashMap;
#[allow(unused_imports)]
use std::path::PathBuf;

use std::any::{type_name};

use std::io::prelude::*;
use crate::hnsw;
use self::hnsw::*;
use crate::dist::Distance;

// magic before each graph point data for each point
const MAGICPOINT : u32 = 0x000a678f;
// magic at beginning of description format v& of dump
const MAGICDESCR_1 : u32 = 0x001a677f;
// magic at beginning of description format v& of dump
const MAGICDESCR_2 : u32 = 0x002a677f;
// magic at beginning of a layer dump
const MAGICLAYER : u32 = 0x000a676f;
// magic head of data file and before each data vector
const MAGICDATAP : u32 = 0xa67f0000;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DumpMode {
    Light,
    Full,
}


/// The main interface for dumping struct Hnsw.
pub trait HnswIO {
    fn dump<W:Write>(&self, mode : DumpMode, outgraph : &mut io::BufWriter<W>, outdata: &mut io::BufWriter<W>) -> Result<i32, String>;
}




/// structure describing main parameters for hnsnw data and written at the beginning of a dump file.
/// 
/// Name of distance and type of data must be encoded in the dump file for a coherent reload.
#[repr(C)]
pub struct Description {
    ///  value is 1 for Full 0 for Light
    pub dumpmode : u8,
    /// max number of connections in layers != 0
    pub max_nb_connection : u8,
    /// number of observed layers
    pub nb_layer : u8,
    /// search parameter
    pub ef: usize,
    /// total number of points
    pub nb_point: usize,
    /// data dimension
    pub dimension : usize,
    /// name of distance
    pub distname : String,
    /// T typename
    pub t_name : String,
}

impl Description {
    /// The dump of Description consists in :
    /// . The value MAGICDESCR_1 as a u32 (4 u8)
    /// . The type of dump as u8
    /// . max_nb_connection as u8
    /// . ef (search parameter used in construction) as usize
    /// . nb_point (the number points dumped) as a usize
    /// . the name of distance used. (nb byes as a usize then list of bytes)
    /// 
    fn dump<W:Write>(&self, argmode : DumpMode, out : &mut io::BufWriter<W>) -> Result<i32, String> {
        log::info!("in dump of description");
        out.write(unsafe { &mem::transmute::<u32, [u8; std::mem::size_of::<u32>()]>(MAGICDESCR_2) } ).unwrap();
        let mode : u8 = match argmode {
            DumpMode::Full => 1,
            _              => 0,
        };
        // CAVEAT should check mode == self.mode
        out.write(unsafe { &mem::transmute::<u8, [u8;1]>(mode) } ).unwrap();
        out.write(unsafe { &mem::transmute::<u8, [u8;1]>(self.max_nb_connection) } ).unwrap();
        out.write(unsafe { &mem::transmute::<u8, [u8;1]>(self.nb_layer) } ).unwrap();
        if self.nb_layer != NB_LAYER_MAX {
            println!("dump of Description, nb_layer != NB_MAX_LAYER");
            return Err(String::from("dump of Description, nb_layer != NB_MAX_LAYER"));
        }
        out.write(unsafe { &mem::transmute::<usize, [u8;std::mem::size_of::<usize>()]>(self.ef) } ).unwrap();
        log::info!("dumping nb point {:?}", self.nb_point);
        // 
        out.write(unsafe { &mem::transmute::<usize, [u8;std::mem::size_of::<usize>()]>(self.nb_point) } ).unwrap();
        //
        log::info!("dumping dimension of data {:?}", self.dimension);
        out.write(unsafe { &mem::transmute::<usize, [u8;std::mem::size_of::<usize>()]>(self.dimension) } ).unwrap();
        // dump of distance name
        let namelen : usize = self.distname.len();
        log::info!("distance name {:?} ", self.distname);
        out.write(unsafe { &mem::transmute::<usize, [u8;std::mem::size_of::<usize>()]>(namelen) } ).unwrap();
        out.write(self.distname.as_bytes()).unwrap();
        // dump of T value typename
        let namelen : usize = self.t_name.len();
        log::info!("T name {:?} ", self.t_name);
        out.write(unsafe { &mem::transmute::<usize, [u8;std::mem::size_of::<usize>()]>(namelen) } ).unwrap();
        out.write(self.t_name.as_bytes()).unwrap();
        //
        return Ok(1);
    } // end fo dump

} // end of HnswIO impl for Descr


/// This method is a preliminary to do a full reload from a dump.
/// The method load_hnsw needs to know the typename , distance used, and construction parameters.
/// So the reload is made in two steps.
pub fn load_description(io_in: &mut dyn Read)  -> io::Result<Description> {
    //
    let mut descr = Description{ dumpmode: 0, max_nb_connection: 0, nb_layer: 0, 
                                ef: 0, nb_point: 0, dimension : 0, 
                                distname: String::from(""), t_name : String::from("")};
    let magic : u32 = 0;
    let it_slice = unsafe {::std::slice::from_raw_parts_mut((&magic as *const u32) as *mut u8, ::std::mem::size_of::<u32>() )};
    io_in.read_exact(it_slice)?;
    log::debug!(" magic {:X} ", magic);
    if magic !=  MAGICDESCR_1 && magic !=  MAGICDESCR_2 {
        log::info!("bad magic");
        return Err(io::Error::new(io::ErrorKind::Other, "bad magic at descr beginning"));
    }
    else if magic ==  MAGICDESCR_1 {
        log::info!("old version of dump..., exiting");
        println!("old version of dump");
        return Err(io::Error::new(io::ErrorKind::Other, "old format of dump"));
    }
    let it_slice = unsafe {::std::slice::from_raw_parts_mut((&descr.dumpmode as *const u8) as *mut u8, ::std::mem::size_of::<u8>() )};
    io_in.read_exact(it_slice)?;
    log::info!(" dumpmode {:?} ", descr.dumpmode);
    let it_slice = unsafe {::std::slice::from_raw_parts_mut((&descr.max_nb_connection as *const u8) as *mut u8, ::std::mem::size_of::<u8>() )};
    io_in.read_exact(it_slice)?;
    log::info!(" max_nb_connection {:?} ", descr.max_nb_connection);
    let it_slice = unsafe {::std::slice::from_raw_parts_mut((&descr.nb_layer as *const u8) as *mut u8, ::std::mem::size_of::<u8>() )};
    io_in.read_exact(it_slice)?;
    log::info!("nb_layer  {:?} ", descr.nb_layer);
    // ef 
    let it_slice = unsafe {::std::slice::from_raw_parts_mut((&descr.ef as *const usize) as *mut u8, ::std::mem::size_of::<usize>() )};
    io_in.read_exact(it_slice)?;
    log::info!("ef  {:?} ", descr.ef);
    // nb_point
    let it_slice = unsafe {::std::slice::from_raw_parts_mut((&descr.nb_point as *const usize) as *mut u8, ::std::mem::size_of::<usize>() )};
    io_in.read_exact(it_slice)?;
    // read dimension
    let it_slice = unsafe {::std::slice::from_raw_parts_mut((&descr.dimension as *const usize) as *mut u8, ::std::mem::size_of::<usize>() )};
    io_in.read_exact(it_slice)?;    
    log::info!("nb_point {:?} dimension {:?} ", descr.nb_point, descr.dimension);    
    // distance name
    let len : usize = 0;
    let it_slice = unsafe {::std::slice::from_raw_parts_mut((&len as *const usize) as *mut u8, ::std::mem::size_of::<usize>() )};
    io_in.read_exact(it_slice)?;
    log::debug!("length of distance name {:?} ", len);
    if len > 256 {
        log::info!(" length of distance name > 256");
        println!(" length of distance name should not exceed 256");
        return Err(io::Error::new(io::ErrorKind::Other, "bad lenght for distance name"));
    }
    let mut distv = Vec::<u8>::new();
    distv.resize(len , 0);
    io_in.read_exact(distv.as_mut_slice())?;
    let distname = String::from_utf8(distv).unwrap();
    log::debug!("distance name {:?} ", distname);
    descr.distname = distname;
    // reload of type name
    let len : usize = 0;
    let it_slice = unsafe {::std::slice::from_raw_parts_mut((&len as *const usize) as *mut u8, ::std::mem::size_of::<usize>() )};
    io_in.read_exact(it_slice)?;
    log::debug!("length of T  name {:?} ", len);
    if len > 256 {
        println!(" length of T name should not exceed 256");
        return Err(io::Error::new(io::ErrorKind::Other, "bad lenght for T name"));
    }
    let mut tnamev = Vec::<u8>::new();
    tnamev.resize(len , 0);
    io_in.read_exact(tnamev.as_mut_slice())?;
    let t_name = String::from_utf8(tnamev).unwrap();
    log::debug!("T type name {:?} ", t_name);
    descr.t_name = t_name;   
    log::debug!(" end of description load \n");
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
    ///      for each neighbour dump of its identity (: usize) and then distance (): u32) to point dumped.
    ///
    /// identity of a point is in full mode the triplet origin_id (: usize), layer (: u8) rank_in_layer (: u32)
    ///                           light mode only origin_id (: usize)
    ///  For data dump
    ///  1. The value MAGICDATAP (u32)
    ///  2. origin_id as a u64
    ///  3. The vector of data (the length is known from Description)
    
fn dump_point<'a, T:Serialize+Clone+Sized+Send+Sync, W:Write>(point : &Point<T> , mode : DumpMode, 
                    graphout : &mut io::BufWriter<W>, dataout : &mut io::BufWriter<W>) -> Result<i32, String> {
    //
    graphout.write(unsafe { &mem::transmute::<u32, [u8;4]>(MAGICPOINT) } ).unwrap();
    // dump ext_id: usize , layer : u8 , rank in layer : i32
    graphout.write(unsafe { &mem::transmute::<usize, [u8;8]>(point.get_origin_id()) } ).unwrap();
    let p_id = point.get_point_id();
    if mode == DumpMode::Full {
        graphout.write(unsafe { &mem::transmute::<u8, [u8;1]>(p_id.0) } ).unwrap();
        graphout.write(unsafe { &mem::transmute::<i32, [u8;4]>(p_id.1) } ).unwrap();
    }
//        log::debug!(" point dump {:?} {:?}  ", p_id, self.get_origin_id());
    // then dump neighborhood info : nb neighbours : u32 , then list of origin_id, layer, rank_in_layer
    let neighborhood = point.get_neighborhood_id();
    // in any case nb_layers are dumped with possibly 0 neighbours at a layer, but this does not occur by construction
    for l in 0..neighborhood.len() {
        let neighbours_at_l = &neighborhood[l];
        graphout.write(unsafe { &mem::transmute::<u8, [u8;1]>(neighbours_at_l.len() as u8) } ).unwrap();
        for n in neighbours_at_l { // dump d_id : uszie , distance : f32, layer : u8, rank in layer : i32
            graphout.write(unsafe { &mem::transmute::<usize, [u8;8]>(n.d_id) } ).unwrap();
            if mode == DumpMode::Full {
                graphout.write(unsafe { &mem::transmute::<u8, [u8;1]>(n.p_id.0) } ).unwrap();
                graphout.write(unsafe { &mem::transmute::<i32, [u8;std::mem::size_of::<i32>()]>(n.p_id.1) } ).unwrap();
            }
            graphout.write(unsafe { &mem::transmute::<f32, [u8;std::mem::size_of::<f32>()]>(n.distance) } ).unwrap();
//                log::debug!("        voisins  {:?}  {:?}  {:?}", n.p_id,  n.d_id , n.distance);
        }
    }
    // now we dump data vector!
    dataout.write(unsafe { &mem::transmute::<u32, [u8;4]>(MAGICDATAP) } ).unwrap();
    dataout.write(unsafe { &mem::transmute::<u64, [u8;8]>(point.get_origin_id() as u64) } ).unwrap();
    //
    let serialized : Vec<u8> = bincode::serialize(point.get_v()).unwrap();
//    log::debug!("serializing len {:?}", serialized.len());
    dataout.write(unsafe { &mem::transmute::<u64, [u8;8]>(serialized.len() as u64) } ).unwrap();
    dataout.write_all(&serialized).unwrap();
    //
    return Ok(1);
} // end of dump for Point<T>



//
//  Reload a point from a dump.
// 
//  The graph part is loaded from graph_in file
// the data vector itself is loaded from data_in
// 
fn load_point<T:'static+DeserializeOwned+Clone+Sized+Send+Sync>(graph_in: &mut dyn Read, descr: &Description, 
                                                data_in: &mut dyn Read) -> io::Result<(Arc<Point<T>>, Vec<Vec<Neighbour> >) > {
    //
    // read and check magic
    let magic : u32 = 0;
    let it_slice = unsafe {::std::slice::from_raw_parts_mut((&magic as *const u32) as *mut u8, ::std::mem::size_of::<u32>() )};
    graph_in.read_exact(it_slice)?;
    if magic != MAGICPOINT {
        log::debug!("got instead of MAGICPOINT {:x}", magic);
        return Err(io::Error::new(io::ErrorKind::Other, "bad magic at point beginning"));
    }
    let origin_id : DataId = 0;
    let it_slice = unsafe {::std::slice::from_raw_parts_mut((&origin_id as *const DataId) as *mut u8, 
                                ::std::mem::size_of::<DataId>())};
    graph_in.read_exact(it_slice)?;
    //
    let layer : u8 = 0;
    let it_slice = unsafe {::std::slice::from_raw_parts_mut((&layer as *const u8) as *mut u8, 
                                ::std::mem::size_of::<u8>() )};
    graph_in.read_exact(it_slice)?;
    //
    let rank_in_l : i32 = 0;
    let it_slice = unsafe {::std::slice::from_raw_parts_mut((&rank_in_l as *const i32) as *mut u8, 
                                ::std::mem::size_of::<i32>() )};
    graph_in.read_exact(it_slice)?;
    let p_id = PointId{0: layer, 1:rank_in_l};
//    log::debug!(" point load {:?} {:?}  ", p_id, origin_id);
    // Now  for each layer , read neighbours
    let nb_layer = descr.nb_layer;
    let nb_neighbours : u8 = 0;
    let mut neighborhood = Vec::<Vec<Neighbour> >::with_capacity(NB_LAYER_MAX as usize);
    for _l in 0..nb_layer {
        let neighbour : Neighbour = Default::default();
        // read nb_neighbour : u8, then nb_neighbours times identity(depends on Full or Light) distance : f32 
        let it_slice = unsafe {::std::slice::from_raw_parts_mut((&nb_neighbours as *const u8) as *mut u8, 
                                        ::std::mem::size_of::<u8>() )};
        graph_in.read_exact(it_slice)?;
        let mut neighborhood_l : Vec<Neighbour> = Vec::with_capacity(nb_neighbours as usize);
        for _j in 0..nb_neighbours {
            let it_slice = unsafe {::std::slice::from_raw_parts_mut((&neighbour.d_id as *const DataId) as *mut u8, 
                                        ::std::mem::size_of::<DataId>() )};
            graph_in.read_exact(it_slice)?;
            if descr.dumpmode == 1 {
                let it_slice = unsafe {::std::slice::from_raw_parts_mut((&neighbour.p_id.0 as *const u8) as *mut u8, 
                                        ::std::mem::size_of::<u8>() )};
                graph_in.read_exact(it_slice)?;
                let it_slice = unsafe {::std::slice::from_raw_parts_mut((&neighbour.p_id.1 as *const i32) as *mut u8, 
                                        ::std::mem::size_of::<i32>() )};
                graph_in.read_exact(it_slice)?;
            }
            let it_slice = unsafe {::std::slice::from_raw_parts_mut((&neighbour.distance as *const f32) as *mut u8, 
                                        ::std::mem::size_of::<f32>() )};
            graph_in.read_exact(it_slice)?;
        //  log::debug!("        voisins  load {:?} {:?} {:?} ", neighbour.p_id, neighbour.d_id , neighbour.distance);
            // now we have a new neighbour, we must really fill neighbourhood info, so it means going from Neighbour to PointWithOrder
            neighborhood_l.push(neighbour);
        }
        neighborhood.push(neighborhood_l);
    }
    for _l in nb_layer..NB_LAYER_MAX {
        neighborhood.push(Vec::<Neighbour>::new());
    }
    //
    // construct a point from data_in
    //
    let magic : u32 = 0;
    let it_slice = unsafe {::std::slice::from_raw_parts_mut( (&magic as *const u32) as *mut u8, 
                                        ::std::mem::size_of::<u32>() )}; 
    data_in.read_exact(it_slice)?;
    assert_eq!(magic, MAGICDATAP, "magic not equal to MAGICDATAP in load_point, point_id : {:?} ", origin_id);
    // read origin id
    let origin_id_data : usize = 0;
    let it_slice = unsafe {::std::slice::from_raw_parts_mut( (&origin_id_data as *const usize) as *mut u8, ::std::mem::size_of::<usize>() )};
    data_in.read_exact(it_slice)?;
    assert_eq!(origin_id, origin_id_data, "origin_id incoherent between graph and data");
    // now read data. we use size_t that is in description, to take care of the casewhere we reload
    let serialized_len : u64 = 0;
    let it_slice = unsafe {::std::slice::from_raw_parts_mut( (&serialized_len as *const u64) as *mut u8, ::std::mem::size_of::<u64>() )};
    data_in.read_exact(it_slice)?; 
//    log::debug!("serialized len to reload {:?}", serialized_len);
    // could do allocation with_capacity() and then using unsafe set_len() which avoid explicit initialization
    let mut v_serialized = Vec::<u8>::new();
    v_serialized.resize(serialized_len as usize, 0);
    data_in.read_exact(&mut v_serialized)?;
    let v : Vec<T>;
    if std::any::TypeId::of::<T>() != std::any::TypeId::of::<NoData>() {
        v = bincode::deserialize(&v_serialized).unwrap();
    }
    else {
        v = Vec::<T>::new();
    }
    let point = Point::<T>::new(&v, origin_id as usize, p_id);
    log::trace!("load_point  origin {:?} allocated size {:?}, dim {:?}", origin_id, point.get_v().len(), descr.dimension);
    //
    return Ok((Arc::new(point), neighborhood));
}  // end of load_point



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
impl <T:Serialize+DeserializeOwned+Clone+Send+Sync> HnswIO for PointIndexation<T> {
    fn dump<W:Write>(&self, mode : DumpMode, graphout : &mut io::BufWriter<W>, dataout : &mut io::BufWriter<W>) -> Result<i32, String> {
        // dump max_layer
        let layers = self.points_by_layer.read();
        let nb_layer = layers.len() as u8;
        graphout.write(unsafe { &mem::transmute::<u8, [u8;1]>(nb_layer) } ).unwrap();
        // dump layers from lower (most populatated to higher level)
        for i in 0..layers.len() {
            let nb_point = layers[i].len();
            log::debug!("dumping layer {:?}, nb_point {:?}", i, nb_point);
            graphout.write(unsafe { &mem::transmute::<u32, [u8;4]>(MAGICLAYER) } ).unwrap();
            graphout.write(unsafe { &mem::transmute::<usize, [u8;8]>(nb_point) } ).unwrap();
            for j in 0..layers[i].len() {
                assert_eq!(layers[i][j].get_point_id() , PointId{0: i as u8,1:j as i32 });
                dump_point(&layers[i][j], mode, graphout, dataout)?;
            }
        }
        // dump id of entry point
        let ep_read = self.entry_point.read();
        assert!(ep_read.is_some());
        let ep = ep_read.as_ref().unwrap();
        graphout.write(unsafe { &mem::transmute::<DataId,[u8; ::std::mem::size_of::<DataId>()] >(ep.get_origin_id()) } ).unwrap();
        let p_id = ep.get_point_id();
        if mode == DumpMode::Full {
            graphout.write(unsafe { &mem::transmute::<u8, [u8;1]>(p_id.0) } ).unwrap();
            graphout.write(unsafe { &mem::transmute::<i32, [u8;4]>(p_id.1) } ).unwrap();
        }
        log::info!("dumped entry_point origin_d {:?}, p_id {:?} ", ep.get_origin_id(), p_id);
        //
        Ok(1)
    } // end of dump for PointIndexation<T>
} // end of impl HnswIO


fn load_point_indexation<T:'static+Serialize+DeserializeOwned+Clone+Sized+Send+Sync>(graph_in: &mut dyn Read, 
                descr : &Description, 
                data_in:  &mut dyn Read) -> io::Result<PointIndexation<T> > {
    //
    log::debug!(" in load_point_indexation");
    //
    // now we check that except for the case NoData, the typename are the sames.
    if std::any::TypeId::of::<T>() != std::any::TypeId::of::<NoData>() && std::any::type_name::<T>() != descr.t_name {
        log::error!("typename loaded in  description {:?} do not correspond to instanciation type {:?}", 
                            descr.t_name,std::any::type_name::<T>() );
        panic!("incohrent size of T in description");
    }
    //
    let mut points_by_layer : Vec<Vec<Arc<Point<T>> > >= Vec::with_capacity(NB_LAYER_MAX as usize);
    let mut neighbourhood_map : HashMap<PointId, Vec<Vec<Neighbour>> > =  HashMap::new();
    // load max layer
    let nb_layer : u8 = 0;
    let it_slice = unsafe {::std::slice::from_raw_parts_mut((&nb_layer as *const u8) as *mut u8, ::std::mem::size_of::<u8>() )};
    graph_in.read_exact(it_slice)?;
    log::debug!("nb layer {:?}", nb_layer);
    if nb_layer > NB_LAYER_MAX {
        return Err(io::Error::new(io::ErrorKind::Other, "inconsistent number of layErrers"));
    }
    //
    let mut nb_points_loaded : usize = 0;
    //
    for l in 0..nb_layer as usize {
        // read and check magic
        log::debug!("loading layer {:?}", l);
        let magic : u32 = 0;
        let it_slice = unsafe {::std::slice::from_raw_parts_mut((&magic as *const u32) as *mut u8, ::std::mem::size_of::<u32>() )};
        graph_in.read_exact(it_slice)?;
        if magic != MAGICLAYER {
            return Err(io::Error::new(io::ErrorKind::Other, "bad magic at layer beginning"));
        }
        let nbpoints : usize = 0;
        let it_slice = unsafe {::std::slice::from_raw_parts_mut((&nbpoints as *const usize) as *mut u8, ::std::mem::size_of::<usize>() )};
        graph_in.read_exact(it_slice)?;
        log::debug!(" layer {:?} , nb points {:?}", l ,  nbpoints);
        let mut vlayer : Vec<Arc<Point<T>>> = Vec::with_capacity(nbpoints);
        for r in 0..nbpoints {
            // load graph and data part of point. Points are dumped in the same order.
            let load_point_res = load_point(graph_in, descr, data_in)?;
            let point = load_point_res.0;
            let p_id = point.get_point_id();
            // some checks
            assert_eq!(l, p_id.0 as usize);
            if r != p_id.1 as usize {
                log::debug!("\n\n origin= {:?},  p_id = {:?}", point.get_origin_id(), p_id);
                log::debug!("storing at l {:?}, r {:?}",  l, r);
            }
            assert_eq!(r , p_id.1 as usize);
            // store neoghbour info of this point
            neighbourhood_map.insert(p_id, load_point_res.1);
            vlayer.push(Arc::clone(&point));
        }
        points_by_layer.push(vlayer);
        nb_points_loaded += nbpoints;
    }
    // at this step all points are loaded , but without their neighbours fileds are not yet initialized
    for (p_id , neighbours) in &neighbourhood_map {
        let point = &points_by_layer[p_id.0 as usize][p_id.1 as usize];
        for l in 0..neighbours.len() {
            for n in &neighbours[l] {
                let n_point = &points_by_layer[n.p_id.0 as usize][n.p_id.1 as usize];
                // now n_point is the Arc<Point> corresponding to neighbour n of point, 
                // construct a corresponding PointWithOrder
                let n_pwo = PointWithOrder::<T>::new(&Arc::clone(&n_point), n.distance);
                point.neighbours.write()[l].push(Arc::new(n_pwo));
            } // end of for n
            //  must sort
            point.neighbours.write()[l].sort_unstable();
        } // end of for l
    } // end loop in neighbourhood_map
    // 
    // get id of entry_point
    // load entry point
    log::info!("\n end of layer loading, allocating PointIndexation, nb points loaded {:?}", nb_points_loaded);
    //
    let origin_id : usize = 0;
    let it_slice = unsafe {::std::slice::from_raw_parts_mut((&origin_id as *const DataId) as *mut u8, ::std::mem::size_of::<DataId>() )};
    graph_in.read_exact(it_slice)?;
    //
    let layer : u8 = 0;
    let it_slice = unsafe {::std::slice::from_raw_parts_mut((&layer as *const u8) as *mut u8, ::std::mem::size_of::<u8>() )};
    graph_in.read_exact(it_slice)?;
    //
    let rank_in_l : i32 = 0;
    let it_slice = unsafe {::std::slice::from_raw_parts_mut((&rank_in_l as *const i32) as *mut u8, ::std::mem::size_of::<i32>() )};
    graph_in.read_exact(it_slice)?;
    //
    log::info!("found entry point, origin_id {:?} , layer {:?}, rank in layer {:?} ", origin_id, layer, rank_in_l);
    let entry_point = Arc::clone(&points_by_layer[layer as usize][rank_in_l as usize]);
    log::info!(" loaded entry point, origin_id {:} p_id {:?}", entry_point.get_origin_id(),entry_point.get_point_id());

    //
    let point_indexation = PointIndexation {
        max_nb_connection : descr.max_nb_connection as usize,
        max_layer : NB_LAYER_MAX as usize,
        points_by_layer : Arc::new(RwLock::new(points_by_layer)),
        layer_g : LayerGenerator::new(descr.max_nb_connection as usize , NB_LAYER_MAX as usize),
        nb_point : Arc::new(RwLock::new(nb_points_loaded)),   // CAVEAT , we should increase , the whole thing is to be able to increment graph ?
        entry_point : Arc::new(RwLock::new(Some(entry_point))),
    };
    //  
    log::debug!("\n exiting load_pointIndexation");
    Ok(point_indexation)
} // end of load_pointIndexation



//
// dump and load of Hnsw<T>
// =========================
//
//

impl <T:Serialize+DeserializeOwned+Clone+Sized+Send+Sync, D: Distance<T>+Send+Sync> HnswIO for Hnsw<T, D> {
    /// The dump method for hnsw.  
    /// - graphout is a BufWriter dedicated to the dump of the graph part of Hnsw
    /// - dataout is a bufWriter dedicated to the dump of the data stored in the Hnsw structure.
    fn dump<W:Write>(&self, mode : DumpMode, graphout : &mut io::BufWriter<W>, dataout : &mut io::BufWriter<W>) -> Result<i32, String> {
        // dump description , then PointIndexation
        let dumpmode : u8 = match mode {
                DumpMode::Full => 1,
                            _  => 0,
        };  
        let datadim : usize = self.layer_indexed_points.get_data_dimension();

        let description = Description {
               ///  value is 1 for Full 0 for Light
            dumpmode : dumpmode,
            max_nb_connection : self.get_max_nb_connection(),
            nb_layer : self.get_max_level() as u8,
            ef: self.get_ef_construction(),
            nb_point: self.get_nb_point(),
            dimension : datadim,
            distname : self.get_distance_name(),
            t_name: type_name::<T>().to_string(),
        };
        log::debug!("dump  obtained typename {:?}", type_name::<T>());
        description.dump(mode, graphout)?;
        // We must dump a header for dataout.
        dataout.write(unsafe { &mem::transmute::<u32, [u8;4]>(MAGICDATAP) } ).unwrap();
        dataout.write(unsafe { &mem::transmute::<usize, [u8;::std::mem::size_of::<usize>()]>(datadim) } ).unwrap();
        //
        self.layer_indexed_points.dump(mode, graphout, dataout)?;
        Ok(1)
    }
}   // end impl block for Hnsw



/// The reload is made in two steps.
/// First a call to load_description must be used to get basic information
/// about structure to reload (Typename, distance type, construction parameters).  
/// Cf fn load_description(io_in: &mut dyn Read) -> io::Result<Description>
///
pub fn load_hnsw<T:'static+Serialize+DeserializeOwned+Clone+Sized+Send+Sync, D:Distance<T>+Default+Send+Sync>(graph_in: &mut dyn Read, 
                                            description: &Description, 
                                            data_in : &mut dyn Read) -> io::Result<Hnsw<T,D> > {
    //  In datafile , we must read MAGICDATAP and dimension and check
    let magic : u32 = 0;
    let it_slice = unsafe {::std::slice::from_raw_parts_mut( (&magic as *const u32) as *mut u8, 
                                        ::std::mem::size_of::<u32>() )}; 
    data_in.read_exact(it_slice)?;
    assert_eq!(magic, MAGICDATAP, "magic not equal to MAGICDATAP in load_point");
    let dimension : usize = 0;
    let it_slice = unsafe {::std::slice::from_raw_parts_mut( (&dimension as *const usize) as *mut u8, 
        ::std::mem::size_of::<usize>() )};
    data_in.read_exact(it_slice)?;
    assert_eq!(dimension, description.dimension, "data dimension incoherent {:?} {:?} ", 
            dimension, description.dimension);
    //
    let _mode = description.dumpmode;
    let distname = description.distname.clone();
    // We must ensure that the distance stored matches the one asked for in loading hnsw
    // for that we check for short names equality stripping 
    log::debug!("distance in description = {:?}", distname);
    let d_type_name = type_name::<D>().to_string();
    let v: Vec<&str> = d_type_name.rsplit_terminator("::").collect();
    for s in v {
        log::info!(" distname in generic type argument {:?}", s);
    }
    if (std::any::TypeId::of::<T>() != std::any::TypeId::of::<NoData>())  &&  (d_type_name != distname) {
        // for all types except NoData , distance asked in reload declaration and distance in dump must be equal!
        let mut errmsg = String::from("error in distances : dumped distance is : ");
        errmsg.push_str(&distname);
        errmsg.push_str(" asked distance in loading is : ");
        errmsg.push_str(&d_type_name);
        log::error!(" distance in type argument : {:?}", d_type_name);
        log::error!("error , dump is for distance = {:?}", distname);
        return Err(io::Error::new(io::ErrorKind::Other, errmsg));
    }
    let t_type = description.t_name.clone();
    log::debug!("T type name in dump = {:?}", t_type);
    let layer_point_indexation = load_point_indexation(graph_in, &description, data_in)?;
    let data_dim = layer_point_indexation.get_data_dimension();
    //
    let hnsw : Hnsw::<T,D> =  Hnsw{  max_nb_connection : description.max_nb_connection as usize,
                        ef_construction : description.ef, 
                        extend_candidates : true, 
                        keep_pruned: false,
                        max_layer: description.nb_layer as usize, 
                        layer_indexed_points: layer_point_indexation,
                        data_dimension : data_dim,
                        dist_f: D::default(),
                        searching : false,
                    } ;
    //
    Ok(hnsw)
}  // end of load_hnsw




//===============================================================================================================

#[cfg(test)]


mod tests {

use super::*;
use crate::dist;


use std::fs::OpenOptions;
use std::io::{BufReader};
use std::path::PathBuf;

pub use crate::dist::*;
pub use crate::api::AnnT;

use rand::distributions::{Distribution, Uniform};


fn log_init_test() {
    let _ = env_logger::builder().is_test(true).try_init();
}



#[test]
fn test_dump_reload_1() {
    println!("\n\n test_dump_reload_1");
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
    let fname = String::from("dumpreloadtest1");
    let _res = hnsw.file_dump(&fname);
    // This will dump in 2 files named dumpreloadtest.hnsw.graph and dumpreloadtest.hnsw.data
    //
    // reload
    log::debug!("\n\n  hnsw reload");
    // we will need a procedural macro to get from distance name to its instanciation. 
    // from now on we test with DistL1
    let graphfname = String::from("dumpreloadtest1.hnsw.graph");
    let graphpath = PathBuf::from(graphfname);
    let graphfileres = OpenOptions::new().read(true).open(&graphpath);
    if graphfileres.is_err() {
        println!("test_dump_reload: could not open file {:?}", graphpath.as_os_str());
        panic!("test_dump_reload: could not open file".to_string());            
    }
    let graphfile = graphfileres.unwrap();
    //  
    let datafname = String::from("dumpreloadtest1.hnsw.data");
    let datapath = PathBuf::from(datafname);
    let datafileres = OpenOptions::new().read(true).open(&datapath);
    if datafileres.is_err() {
        println!("test_dump_reload : could not open file {:?}", datapath.as_os_str());
        panic!("test_dump_reload : could not open file".to_string());            
    }
    let datafile = datafileres.unwrap();
    //
    let mut graph_in = BufReader::new(graphfile);
    let mut data_in = BufReader::new(datafile);
    // we need to call load_description first to get distance name
    let hnsw_description = load_description(&mut graph_in).unwrap();
    let hnsw_loaded : Hnsw<f32,DistL1>= load_hnsw(&mut graph_in, &hnsw_description, &mut data_in).unwrap();
    // test equality
    check_graph_equality(&hnsw_loaded, &hnsw);
}  // end of test_dump_reload




#[test]
fn test_dump_reload_graph_only() {
    println!("\n\n test_dump_reload_graph_only");
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
    // dump in a file. Must take care of name as tests runs in // !!!
    let fname = String::from("dumpreloadtestgraph");
    let _res = hnsw.file_dump(&fname);
    // This will dump in 2 files named dumpreloadtest.hnsw.graph and dumpreloadtest.hnsw.data
    //
    // reload
    log::debug!("\n\n  hnsw reload");
    // we will need a procedural macro to get from distance name to its instanciation. 
    // from now on we test with DistL1
    let graphfname = String::from("dumpreloadtestgraph.hnsw.graph");
    let graphpath = PathBuf::from(graphfname);
    let graphfileres = OpenOptions::new().read(true).open(&graphpath);
    if graphfileres.is_err() {
        println!("test_dump_reload: could not open file {:?}", graphpath.as_os_str());
        panic!("test_dump_reload: could not open file".to_string());            
    }
    let graphfile = graphfileres.unwrap();
    //  
    let datafname = String::from("dumpreloadtestgraph.hnsw.data");
    let datapath = PathBuf::from(datafname);
    let datafileres = OpenOptions::new().read(true).open(&datapath);
    if datafileres.is_err() {
        println!("test_dump_reload : could not open file {:?}", datapath.as_os_str());
        panic!("test_dump_reload : could not open file".to_string());            
    }
    let datafile = datafileres.unwrap();
    //
    let mut graph_in = BufReader::new(graphfile);
    let mut data_in = BufReader::new(datafile);
    // we need to call load_description first to get distance name
    let hnsw_description = load_description(&mut graph_in).unwrap();
    let hnsw_loaded : Hnsw<NoData,NoDist>= load_hnsw(&mut graph_in, &hnsw_description, &mut data_in).unwrap();
    // test equality
    check_graph_equality(&hnsw_loaded, &hnsw);
}  // end of test_dump_reload



#[test]
fn test_bincode() {
    let mut rng = rand::thread_rng();
    let unif =  Uniform::<f32>::new(0.,1.);
    let size = 10;
    let mut xsi;
    let mut data = Vec::with_capacity(size);
    for _ in 0..size {
         xsi = unif.sample(&mut rng);
         println!("xsi = {:?}", xsi);
        data.push(xsi);
    } 
    println!("to serialized {:?}", data);

    let v_serialized : Vec<u8> = bincode::serialize(&data).unwrap();
    log::debug!("serializing len {:?}", v_serialized.len());
    let v_deserialized : Vec<f32> = bincode::deserialize(&v_serialized).unwrap();
    println!("deserialized {:?}", v_deserialized);

}



}  // end module tests