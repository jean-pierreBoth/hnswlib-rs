extern crate rand;

// for logging (debug mostly, switched at compile time in cargo.toml)
extern crate log;
extern crate simple_logger;

#[macro_use]
extern crate lazy_static;


pub mod hnsw;
pub mod dist;
pub mod annhdf5;
pub mod hnswio;
pub mod test;
pub mod prelude;
pub mod api;
pub mod libext;

lazy_static! {
    #[allow(dead_code)]
    static ref LOG: u64 = {
        let res = init_log();
        res
    };
}

// install a logger facility
fn init_log() -> u64 {
    simple_logger::init().unwrap();
    println!("\n ************** initializing logger *****************\n");    
    return 1;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    // initialize once log system for tests.
    fn init_log() {
        let _res = simple_logger::init();
    }
}  // end of tests
