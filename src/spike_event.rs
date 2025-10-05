use std::cmp::Ordering;

#[derive(Debug, Clone, Copy)]
pub struct SpikeEvent {
    pub synapse_index: usize,
    pub delivery_time: usize,
}

impl PartialEq for SpikeEvent {
    fn eq(&self, other: &Self) -> bool {
        self.delivery_time == other.delivery_time
    }
}

impl Eq for SpikeEvent {}

impl PartialOrd for SpikeEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.delivery_time.partial_cmp(&other.delivery_time)
    }
}

impl Ord for SpikeEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse to make BinaryHeap a min-heap for delivery_time
        self.delivery_time
            .partial_cmp(&other.delivery_time)
            .unwrap_or(Ordering::Equal)
    }
}
