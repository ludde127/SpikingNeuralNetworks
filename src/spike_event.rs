use std::cmp::Ordering;

#[derive(Debug, Clone)]
pub struct SpikeEvent {
    pub source_neuron: usize,
    pub target_neuron: usize,
    pub synapse_index: usize,
    pub spike_time: f64,
    pub arrival_time: f64,
    pub weight: f64,
}

impl PartialEq for SpikeEvent {
    fn eq(&self, other: &Self) -> bool {
        self.arrival_time == other.arrival_time
    }
}

impl Eq for SpikeEvent {}

impl PartialOrd for SpikeEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.arrival_time.partial_cmp(&other.arrival_time)
    }
}

impl Ord for SpikeEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse to make BinaryHeap a min-heap for arrival_time
        self.arrival_time
            .partial_cmp(&other.arrival_time)
            .unwrap_or(Ordering::Equal)
    }
}

