//! This module provides conversion of Hnsw into a HashTable of point and neighbourhood
//! It can be used when reloading only the graph part of the data to have knowledge
//! of relaitve proximity of points as described just by their DataId
//! 

use hashbrown::HashMap;


use crate::hnsw;
use hnsw::*;

use crate::dist::NoDist;

pub struct FlatPoint {
    /// an id coming from client using hnsw, should identify point uniquely
    origin_id : DataId,
    /// a point id identifying point as stored in our structure
    p_id: PointId, 
    /// neighbours info
    neighbours:Vec<Vec<Neighbour>>,
}


pub fn flatten_point<T:Clone+Send+Sync>(point :&Point<T>) -> FlatPoint {
    let fpoint = FlatPoint { origin_id : point.get_origin_id(), p_id : point.get_point_id() ,neighbours: point.get_neighborhood_id() };
    fpoint
}

/// extract from the Hnsw strucure a hashtable mapping original DataId into a  Point<T> srtructure with its neighbours information
/// Useful after reloading from a dump with T=NoData and D = NoDist as points are then reloaded with neighbourhood information only.
pub fn intohash_table<T,D>(hnsw: &Hnsw<NoData,NoDist>) -> HashMap<DataId, FlatPoint> {
    let mut hash_t = HashMap::new();
    let mut ptiter = hnsw.get_point_indexation().into_iter();
    //
    loop {
        if let Some(point) = ptiter.next() {
        //    println!("point : {:?}", _point.p_id);
            let res_insert = hash_t.insert(point.get_origin_id(), flatten_point(&point));
            match res_insert {
                Some(old_point) => {
                    println!("2 points with same origin id {:?}", old_point.origin_id);
                    log::error!("2 points with same origin id {:?}", old_point.origin_id);
                }
                _ => ()
            } // end match
        }
        else {
            break;
        }
    } // end while
    return hash_t;
} // end of intohash_table
