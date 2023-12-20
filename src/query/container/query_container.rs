use std::fmt::Debug;

use super::query::Query;

trait_set::trait_set! {
    pub trait QueryKey= Send + Sync + Ord + 'static + Debug;
    pub trait QueryValue= Send + Sync + Ord + 'static + Debug;
}

pub trait QueryContainer<K, V>: Sync
where
    K: QueryKey,
    V: QueryValue,
{
    fn query(&self, query: &Query<V>) -> Vec<K>;
}

pub trait DataContainer<K, V>
where
    K: QueryKey,
    V: QueryValue,
{
    fn insert(&mut self, key: K, value: V);
    fn get_queryable(&mut self) -> Box<dyn QueryContainer<K, V>>;
    //    fn delete(&mut self, key: K);
}
