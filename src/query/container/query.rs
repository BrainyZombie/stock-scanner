#[derive(Debug)]
pub enum Query<V>
where
    V: Ord + Send,
{
    Gt { value: V },
    Lt { value: V },
    Bt { value_low: V, value_high: V },
}
