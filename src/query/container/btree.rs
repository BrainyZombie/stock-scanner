use std::{collections::HashMap, hash::Hash, sync::Arc};

use super::{DataContainer, Query, QueryContainer, QueryKey, QueryValue};

#[derive(PartialEq, Eq, Clone)]
pub struct InnerNodeData<K, V, const C: usize>
where
    K: QueryKey + Clone + Copy + Hash,
    V: QueryValue + Clone + Copy,
{
    value_range: [V; 2],
    keys_in_range: Vec<K>,
    first_child: BTreeNode<K, V, C>,
    children: [Option<(V, BTreeNode<K, V, C>)>; C],
}

#[derive(PartialEq, Eq, Clone)]
pub struct InnerNode<K, V, const C: usize>
where
    K: QueryKey + Clone + Copy + Hash,
    V: QueryValue + Clone + Copy,
{
    node: Arc<InnerNodeData<K, V, C>>,
}

#[derive(PartialEq, Eq, Clone)]
pub struct LeafNodeData<K, V, const C: usize> {
    value: V,
    keys: Vec<K>,
}

#[derive(PartialEq, Eq, Clone)]
pub struct LeafNode<K, V, const C: usize>
where
    K: QueryKey + Clone + Copy + Hash,
    V: QueryValue + Clone + Copy,
{
    node: Arc<LeafNodeData<K, V, C>>,
}

#[derive(PartialEq, Eq, Clone)]
pub enum BTreeNode<K, V, const C: usize>
where
    K: QueryKey + Clone + Copy + Hash,
    V: QueryValue + Clone + Copy,
{
    InnerNode(InnerNode<K, V, C>),
    LeafNode(LeafNode<K, V, C>),
}

pub struct BTree<K, V, const C: usize>
where
    K: QueryKey + Clone + Copy + Hash,
    V: QueryValue + Clone + Copy,
{
    root_node: BTreeNode<K, V, C>,
    node_lookup: HashMap<K, V>,
}

impl<K, V, const C: usize> InnerNode<K, V, C>
where
    K: QueryKey + Clone + Copy + Hash,
    V: QueryValue + Clone + Copy,
{
    fn query(&self, query: &Query<V>) -> Vec<K> {
        let node = &self.node;
        let [range_min, range_max] = node.value_range;
        match query {
            Query::Gt { value } => {
                if *value <= range_min {
                    return node.keys_in_range.clone();
                };
                if *value > range_max {
                    return vec![];
                };
            }
            Query::Lt { value } => {
                if *value < range_min {
                    return vec![];
                };
                if *value >= range_max {
                    return node.keys_in_range.clone();
                };
            }
            Query::Bt {
                value_low,
                value_high,
            } => {
                if *value_low <= range_min && *value_high >= range_max {
                    return node.keys_in_range.clone();
                }
                if *value_high < range_min || *value_low > range_max {
                    return vec![];
                }
            }
        }

        return node
            .children
            .iter()
            .fold(node.first_child.query(query), |mut acc, child_info| {
                if let Some((_, child)) = child_info {
                    let mut result = child.query(query);
                    acc.append(&mut result);
                    acc
                } else {
                    acc
                }
            });
    }
    fn insert(&self, key: K, value: V) -> (BTreeNode<K, V, C>, Option<BTreeNode<K, V, C>>) {
        let mut child_nodes: Vec<_> = std::iter::once(self.node.first_child.clone())
            .chain(self.node.children.iter().cloned().map_while(|child| {
                if let Some((_, child)) = child {
                    Some(child)
                } else {
                    None
                }
            }))
            .collect();

        // will be forwarded to the child in the prev index as the one we calculate here
        // since we index into the child_nodes array later which has first_child prepended
        // Hence getting idx 2 here means we actually forward to idx 1
        // getting idx 0 means we forward to first_child
        let idx_to_forward = self
            .node
            .children
            .iter()
            .enumerate()
            .find(|(_, child)| {
                if let Some((child_value, _)) = child {
                    if value < *child_value {
                        return true;
                    };
                };
                return false;
            })
            .map_or(child_nodes.len() - 1, |(idx, _)| idx);

        let result = child_nodes[idx_to_forward].insert(key, value);

        match result {
            (new_child, None) => {
                child_nodes[idx_to_forward] = new_child;
                let first_child = child_nodes[0].clone();
                let children: Vec<_> = child_nodes.into_iter().skip(1).collect();
                (BTreeNode::new_inner_node(first_child, children), None)
            }
            (new_child1, Some(new_child2)) => {
                child_nodes.splice(
                    idx_to_forward..(idx_to_forward + 1),
                    [new_child1, new_child2].into_iter(),
                );
                if child_nodes.len() <= C + 1 {
                    let mut iter = child_nodes.into_iter();
                    let first_child = iter.next().unwrap();
                    let children: Vec<_> = iter.collect();
                    (BTreeNode::new_inner_node(first_child, children), None)
                } else {
                    let child_nodes2 = child_nodes.split_off(child_nodes.len() / 2);

                    let mut iter = child_nodes.into_iter();
                    let first_child = iter.next().unwrap();
                    let children: Vec<_> = iter.collect();

                    let mut iter2 = child_nodes2.into_iter();
                    let first_child2 = iter2.next().unwrap();
                    let children2: Vec<_> = iter2.collect();
                    (
                        BTreeNode::new_inner_node(first_child, children),
                        Some(BTreeNode::new_inner_node(first_child2, children2)),
                    )
                }
            }
        }
    }
}

impl<K, V, const C: usize> LeafNode<K, V, C>
where
    K: QueryKey + Clone + Copy + Hash,
    V: QueryValue + Clone + Copy,
{
    fn query(&self, query: &Query<V>) -> Vec<K> {
        let node = &self.node;
        match query {
            Query::Gt { value } => {
                if node.value >= *value {
                    node.keys.clone()
                } else {
                    vec![]
                }
            }
            Query::Lt { value } => {
                if node.value <= *value {
                    node.keys.clone()
                } else {
                    vec![]
                }
            }
            Query::Bt {
                value_low,
                value_high,
            } => {
                if node.value <= *value_high && node.value >= *value_low {
                    node.keys.clone()
                } else {
                    vec![]
                }
            }
        }
    }
    fn insert(&self, key: K, value: V) -> (BTreeNode<K, V, C>, Option<BTreeNode<K, V, C>>) {
        if self.node.value == value {
            (
                BTreeNode::new_leaf_node(
                    self.node
                        .keys
                        .clone()
                        .into_iter()
                        .chain(std::iter::once(key))
                        .collect(),
                    value,
                ),
                None,
            )
        } else if self.node.value < value {
            (
                BTreeNode::LeafNode(LeafNode {
                    node: self.node.clone(),
                }),
                Some(BTreeNode::new_leaf_node(vec![key], value)),
            )
        } else {
            (
                BTreeNode::new_leaf_node(vec![key], value),
                Some(BTreeNode::LeafNode(LeafNode {
                    node: self.node.clone(),
                })),
            )
        }
    }
}

impl<K, V, const C: usize> BTreeNode<K, V, C>
where
    K: QueryKey + Clone + Copy + Hash,
    V: QueryValue + Clone + Copy,
{
    fn new_inner_node(
        first_child: BTreeNode<K, V, C>,
        children: Vec<BTreeNode<K, V, C>>,
    ) -> BTreeNode<K, V, C> {
        if children.len() > C {
            panic!("Children len should never be greater than C");
        }
        let start_value = {
            match first_child {
                BTreeNode::LeafNode(ref leaf) => leaf.node.value,
                BTreeNode::InnerNode(ref inner) => inner.node.value_range[0],
            }
        };
        let end_value = children.last().map_or(start_value, |node| match node {
            BTreeNode::InnerNode(ref inner) => inner.node.value_range[1],
            BTreeNode::LeafNode(ref leaf) => leaf.node.value,
        });
        let keys_in_range: Vec<K> = children.iter().fold(
            match first_child {
                BTreeNode::LeafNode(ref leaf) => leaf.node.keys.clone(),
                BTreeNode::InnerNode(ref inner) => inner.node.keys_in_range.clone(),
            },
            |mut acc, child| {
                match child {
                    BTreeNode::LeafNode(ref leaf) => acc.append(&mut leaf.node.keys.clone()),
                    BTreeNode::InnerNode(ref inner) => {
                        acc.append(&mut inner.node.keys_in_range.clone())
                    }
                };
                acc
            },
        );

        BTreeNode::InnerNode(InnerNode {
            node: Arc::new(InnerNodeData {
                keys_in_range,
                children: std::array::from_fn(|i| {
                    children
                        .get(i)
                        .map(|child| match child {
                            BTreeNode::LeafNode(leaf) => Some((leaf.node.value, child.clone())),
                            BTreeNode::InnerNode(inner) => {
                                Some((inner.node.value_range[0], child.clone()))
                            }
                        })
                        .unwrap_or(None)
                }),
                first_child,
                value_range: [start_value, end_value],
            }),
        })
    }
    fn new_leaf_node(keys: Vec<K>, value: V) -> BTreeNode<K, V, C> {
        BTreeNode::LeafNode(LeafNode {
            node: Arc::new(LeafNodeData { keys, value }),
        })
    }
    fn insert(&self, key: K, value: V) -> (BTreeNode<K, V, C>, Option<BTreeNode<K, V, C>>) {
        match self {
            BTreeNode::InnerNode(node) => node.insert(key, value),
            BTreeNode::LeafNode(node) => node.insert(key, value),
        }
    }
}

impl<K, V, const C: usize> BTree<K, V, C>
where
    K: QueryKey + Clone + Copy + Hash,
    V: QueryValue + Clone + Copy,
{
    pub fn new(init: (K, V)) -> BTree<K, V, C> {
        BTree {
            root_node: BTreeNode::new_leaf_node(vec![init.0], init.1),
            node_lookup: HashMap::new(),
        }
    }
}

impl<K, V, const C: usize> DataContainer<K, V, BTreeNode<K, V, C>> for BTree<K, V, C>
where
    K: QueryKey + Clone + Copy + Hash,
    V: QueryValue + Clone + Copy,
{
    fn insert(&mut self, key: K, value: V) {
        let root = self.root_node.clone();

        let new_nodes = {
            if let Some(_) = self.node_lookup.get(&key) {
                panic!("test");
                root.insert(key, value)
            } else {
                self.node_lookup.insert(key, value);
                root.insert(key, value)
            }
        };

        if let (node1, Some(node2)) = new_nodes {
            let new_root = BTreeNode::new_inner_node(node1, vec![node2]);
            self.root_node = new_root;
        } else if let (node1, None) = new_nodes {
            self.root_node = node1;
        }
    }
    fn get_queryable(&mut self) -> BTreeNode<K, V, C> {
        self.root_node.clone()
    }
}

impl<K, V, const C: usize> QueryContainer<K, V> for BTreeNode<K, V, C>
where
    K: QueryKey + Clone + Copy + Hash,
    V: QueryValue + Clone + Copy,
{
    fn query(&self, query: &Query<V>) -> Vec<K> {
        match self {
            BTreeNode::InnerNode(node) => node.query(query),
            BTreeNode::LeafNode(node) => node.query(query),
        }
    }
}

#[cfg(test)]
mod tests {
    use core::panic;
    use std::{time::Instant, usize};

    use rand::{
        distributions::{Distribution, Standard},
        Rng,
    };

    use super::*;
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
            2 => Query::Bt {
                value_low: rng.gen(),
                value_high: rng.gen(),
            },
            _ => panic!("Invalid RNG generated for query type"),
        };
        query
    }
    fn verify_queryable<K, V>(query_container: &Box<dyn QueryContainer<K, V>>, items: &Vec<(K, V)>)
    where
        K: QueryKey + Clone + Copy + Hash,
        V: QueryValue + Clone + Copy,
        Standard: Distribution<K>,
        Standard: Distribution<V>,
    {
        let query_count = 1000;
        for i in 0..query_count {
            println!("Query #{i}");
            let query = get_random_query();
            let mut expected_results: Vec<_> = match query {
                Query::Gt { value } => items.iter().filter(|(_, v)| *v >= value).collect(),
                Query::Lt { value } => items.iter().filter(|(_, v)| *v <= value).collect(),
                Query::Bt {
                    value_low,
                    value_high,
                } => items
                    .iter()
                    .filter(|(_, v)| *v >= value_low && *v <= value_high)
                    .collect(),
            };

            let mut actual_results: Vec<_> = query_container
                .query(&query)
                .iter()
                .map(|k| items.iter().find(|(k1, _)| k1 == k).unwrap())
                .collect();
            expected_results.sort();
            actual_results.sort();
            assert_eq!(
                expected_results, actual_results,
                "testing query {:?}",
                query
            );
        }
    }
    fn verify_tree_leaf_depth<K, V, const C: usize>(root: &BTreeNode<K, V, C>) -> u64
    where
        K: QueryKey + Clone + Copy + Hash,
        V: QueryValue + Clone + Copy,
    {
        match root {
            BTreeNode::LeafNode(_) => 0,
            BTreeNode::InnerNode(inner) => {
                let first_depth = verify_tree_leaf_depth(&inner.node.first_child);
                inner.node.children.iter().for_each(|child| {
                    if let Some((_, child)) = child {
                        assert_eq!(first_depth, verify_tree_leaf_depth(child));
                    }
                });

                first_depth + 1
            }
        }
    }
    fn verify_tree_contents<K, V, const C: usize>(root: &BTreeNode<K, V, C>, items: &Vec<(K, V)>)
    where
        K: QueryKey + Clone + Copy + Hash,
        V: QueryValue + Clone + Copy,
    {
        match root {
            BTreeNode::LeafNode(leaf) => {
                leaf.node.keys.iter().for_each(|k| {
                    let item = items.iter().find(|(k1, _)| k1 == k);
                    if let Some((_, v1)) = item {
                        assert!(*v1 == leaf.node.value);
                    } else {
                        assert!(false);
                    }
                });

                let expected_keys: Vec<_> = items
                    .iter()
                    .filter_map(|(k, v)| {
                        if *v == leaf.node.value {
                            Some(*k)
                        } else {
                            None
                        }
                    })
                    .collect();
                assert_eq!(expected_keys, leaf.node.keys);
            }
            BTreeNode::InnerNode(inner) => {
                assert!(inner.node.value_range[0] <= inner.node.value_range[1]);

                inner.node.keys_in_range.iter().for_each(|k| {
                    let item = items.iter().find(|(k1, _)| k1 == k);
                    if let Some((_, v1)) = item {
                        assert!(
                            inner.node.value_range[0] <= *v1 && inner.node.value_range[1] >= *v1
                        );
                    } else {
                        assert!(false);
                    }
                });

                let mut expected_keys: Vec<_> = items
                    .iter()
                    .filter_map(|(k, v)| {
                        if *v >= inner.node.value_range[0] && *v <= inner.node.value_range[1] {
                            Some(*k)
                        } else {
                            None
                        }
                    })
                    .collect();
                let mut actual_keys = inner.node.keys_in_range.clone();

                expected_keys.sort();
                actual_keys.sort();
                assert_eq!(expected_keys, actual_keys);

                match &inner.node.first_child {
                    BTreeNode::LeafNode(leaf_child) => {
                        assert!(
                            leaf_child.node.value <= inner.node.value_range[1]
                                && leaf_child.node.value >= inner.node.value_range[0]
                        );
                    }
                    BTreeNode::InnerNode(inner_child) => {
                        assert!(
                            inner_child.node.value_range[0] >= inner.node.value_range[0]
                                && inner_child.node.value_range[1] <= inner.node.value_range[1]
                        );
                    }
                }
                verify_tree_contents(&inner.node.first_child, items);

                inner.node.children.iter().for_each(|child| {
                    if let Some((v, child)) = child {
                        assert!(*v <= inner.node.value_range[1]);
                        match child {
                            BTreeNode::LeafNode(leaf_child) => {
                                assert!(
                                    leaf_child.node.value <= inner.node.value_range[1]
                                        && leaf_child.node.value >= inner.node.value_range[0]
                                );
                            }
                            BTreeNode::InnerNode(inner_child) => {
                                assert!(
                                    inner_child.node.value_range[0] >= inner.node.value_range[0]
                                        && inner_child.node.value_range[1]
                                            <= inner.node.value_range[1]
                                );
                            }
                        }
                        verify_tree_contents(child, items);
                    }
                });
                inner
                    .node
                    .children
                    .iter()
                    .skip(1)
                    .zip(inner.node.children.iter())
                    .for_each(|children| {
                        if let (Some((v1, _)), Some((v2, _))) = children {
                            assert!(v2 < v1, "testing {:?} {:?}", v1, v2);
                        }
                    });
            }
        }
    }
    fn verify_tree<K, V, const C: usize>(root: &BTreeNode<K, V, C>, items: &Vec<(K, V)>)
    where
        K: QueryKey + Clone + Copy + Hash,
        V: QueryValue + Clone + Copy,
    {
        verify_tree_leaf_depth(root);
        verify_tree_contents(root, items);
    }

    fn random_inserts<K, V, const C: usize>()
    where
        K: QueryKey + Clone + Copy + Hash,
        V: QueryValue + Clone + Copy,
        Standard: Distribution<K>,
        Standard: Distribution<V>,
    {
        let time = Instant::now();
        let mut rng = rand::thread_rng();
        let init: (K, V) = (rng.gen(), rng.gen());
        let mut container: BTree<K, V, C> = BTree::new(init);

        let iterations: usize = 10;
        let pairs_per_iteration: usize = 100;

        let mut kv_pairs: Vec<Vec<(K, V)>> = vec![vec![init]];
        let mut queryables: Vec<Box<dyn QueryContainer<K, V>>> =
            vec![Box::new(container.get_queryable())];
        let mut tree_roots: Vec<BTreeNode<K, V, C>> = vec![container.get_queryable()];
        for it in 0..iterations {
            println!("starting {it} at {:?}", time.elapsed());
            if let Some(last_pairs) = kv_pairs.last() {
                kv_pairs.push(last_pairs.clone());
            } else {
                kv_pairs.push(vec![]);
            }
            let kv_pair = kv_pairs.last_mut().unwrap();
            for _ in 0..pairs_per_iteration {
                let (k, v) = loop {
                    let (k, v): (K, V) = rng.gen();
                    if !kv_pair.iter().any(|(k1, _)| *k1 == k) {
                        break (k, v);
                    };
                };
                kv_pair.push((k, v));
                container.insert(k, v);
            }
            queryables.push(Box::new(container.get_queryable()));
            tree_roots.push(container.get_queryable());

            verify_queryable(queryables.last().unwrap(), kv_pairs.last().unwrap());
            verify_tree(tree_roots.last().unwrap(), kv_pairs.last().unwrap());
        }

        kv_pairs
            .iter()
            .zip(tree_roots.iter().zip(queryables.iter()))
            .for_each(|(kv_pairs, (root, queryable))| {
                verify_tree(root, kv_pairs);
                verify_queryable(queryable, kv_pairs);
            });
    }

    fn query_benchmark<K, V, const C: usize>()
    where
        K: QueryKey + Clone + Copy + Hash,
        V: QueryValue + Clone + Copy,
        Standard: Distribution<K>,
        Standard: Distribution<V>,
    {
        let time = Instant::now();
        let mut rng = rand::thread_rng();
        let mut container = BTree::<K, V, C>::new((rng.gen(), rng.gen()));

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
        let queriable = container.get_queryable();
        let time = Instant::now();
        for i in 0..query_count {
            if i % 1000 == 0 {
                println!("Query #{i} at {:?}", time.elapsed());
            };
            let query = get_random_query();
            queriable.query(&query);
        }

        println!(
            "Tree depth {}",
            verify_tree_leaf_depth(&container.root_node)
        );
    }

    #[test]
    fn random_inserts_test() {
        random_inserts::<u32, u32, 9>();
    }

    #[test]
    fn queries_test() {
        query_benchmark::<u32, u32, 9>();
    }
}
