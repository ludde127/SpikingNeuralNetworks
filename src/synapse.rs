use rand::Rng;
use crate::constants::{MAX_SYNAPSE_DELAY_MS, MIN_SYNAPSE_DELAY_MS};

pub trait Synapse: Send + Sync {
    fn get_presynaptic_neuron(&self) -> usize;
    fn get_postsynaptic_neuron(&self) -> usize;
    fn get_weight(&self) -> f32;
    fn get_delay(&self) -> f32;
    fn update_weight(&mut self, delta_w: f32);
    fn clone_box(&self) -> Box<dyn Synapse>;
}

impl Clone for Box<dyn Synapse> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

#[derive(Clone, Debug)]
pub struct ChemicalSynapse {
    pub presynaptic_neuron: usize,
    pub postsynaptic_neuron: usize,
    pub weight: f32,
    pub delay: f32, // in milliseconds
}

impl ChemicalSynapse {
    pub fn new(presynaptic_neuron: usize, postsynaptic_neuron: usize) -> Self {
        let mut rng = rand::thread_rng();
        ChemicalSynapse {
            presynaptic_neuron,
            postsynaptic_neuron,
            weight: rng.gen_range(0.0..1.0),
            delay: rng.gen_range(MIN_SYNAPSE_DELAY_MS..MAX_SYNAPSE_DELAY_MS),
        }
    }
}

impl Synapse for ChemicalSynapse {
    fn get_presynaptic_neuron(&self) -> usize {
        self.presynaptic_neuron
    }

    fn get_postsynaptic_neuron(&self) -> usize {
        self.postsynaptic_neuron
    }

    fn get_weight(&self) -> f32 {
        self.weight
    }

    fn get_delay(&self) -> f32 {
        self.delay
    }

    fn update_weight(&mut self, delta_w: f32) {
        self.weight += delta_w;
    }

    fn clone_box(&self) -> Box<dyn Synapse> {
        Box::new(self.clone())
    }
}
