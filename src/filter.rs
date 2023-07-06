//! defines a trait for filtering requests


pub trait FilterT {
    fn hnsw_filter(&self, id:&usize) -> bool;
}

impl FilterT for Vec<usize> {
    fn hnsw_filter(&self, id:&usize) -> bool {
        return match &self.binary_search(id) {
            Ok(_) => true,
            _ => false }
    }
}

impl<F> FilterT for F
where F: Fn(&usize) -> bool,
{
    fn hnsw_filter(&self, id:&usize) -> bool {
        return self(id)
    }
}
