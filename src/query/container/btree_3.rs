#[cfg(test)]
mod tests {
    use core::panic;
    use std::{
        collections::BTreeMap,
        ops::{Range, RangeBounds, RangeFrom, RangeTo},
        time::Instant,
        usize,
    };

    use rand::{
        distributions::{Distribution, Standard},
        Rng,
    };

    use crate::query::container::{Query, QueryKey, QueryValue};

    fn get_random_query<V>() -> Query<V>
    where
        V: QueryValue + Clone + Copy,
        Standard: Distribution<V>,
    {
        let mut rng = rand::thread_rng();
        let query_type: u32 = rng.gen_range(0..3);
        let query: Query<V> = match query_type {
            0 => Query::Gt { value: rng.gen() },
            1 => Query::Lt { value: rng.gen() },
            2 => {
                let (v1, v2): (V, V) = (rng.gen(), rng.gen());
                Query::Bt {
                    value_low: v1.min(v2),
                    value_high: v1.max(v2),
                }
            }
            _ => panic!("Invalid RNG generated for query type"),
        };
        query
    }

    fn query_benchmark<K, V>()
    where
        K: QueryKey + Clone + Copy,
        V: QueryValue + Clone + Copy,
        Standard: Distribution<K>,
        Standard: Distribution<V>,
        RangeFrom<V>: RangeBounds<K>,
        RangeTo<V>: RangeBounds<K>,
        Range<V>: RangeBounds<K>,
    {
        let time = Instant::now();
        let mut rng = rand::thread_rng();
        let mut container = BTreeMap::new();

        let pair_count: usize = 100_000;

        let mut kv_pair: Vec<(K, V)> = vec![];
        for it in 0..pair_count {
            if it % 10000 == 0 {
                println!("starting {it} at {:?}", time.elapsed());
            };
            let (k, v) = loop {
                let (k, v): (K, V) = rng.gen();
                if !kv_pair.iter().any(|(k1, _)| *k1 == k) {
                    break (k, v);
                };
            };
            kv_pair.push((k, v));
            container.insert(k, v);
        }

        let query_count = 10_000;
        let time = Instant::now();
        for i in 0..query_count {
            if i % 1000 == 0 {
                println!("Query #{i} at {:?}", time.elapsed());
            };
            let query: Query<V> = get_random_query();
            let _: Vec<_> = match query {
                Query::Gt { value } => container.range(value..).collect(),
                Query::Lt { value } => container.range(..value).collect(),
                Query::Bt {
                    value_low,
                    value_high,
                } => container.range(value_low..value_high).collect(),
            };
        }
    }

    #[test]
    fn queries_test() {
        query_benchmark::<u32, u32>();
    }
}
