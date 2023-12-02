// for logging (debug mostly, switched at compile time in cargo.toml)
use env_logger::Builder;

use lazy_static::lazy_static;

pub mod api;
pub mod datamap;
pub mod dist;
pub mod filter;
pub mod flatten;
pub mod hnsw;
pub mod hnswio;
pub mod libext;
pub mod prelude;

lazy_static! {
    static ref LOG: u64 = {
        let res = init_log();
        res
    };
}

// install a logger facility
fn init_log() -> u64 {
    Builder::from_default_env().init();
    println!("\n ************** initializing logger *****************\n");
    return 1;
}
