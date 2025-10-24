use crate::synapse::ChemicalSynapse;
use std::cmp::Ordering;
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone)]
pub struct SpikeEvent {
    pub synapse: Arc<RwLock<ChemicalSynapse>>,
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
