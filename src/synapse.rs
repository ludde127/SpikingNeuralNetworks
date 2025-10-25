use std::sync::{Arc, RwLock};
use rand::Rng;
use crate::constants::{MAX_SYNAPSE_DELAY_MS, MAX_SYNAPSE_WEIGHT, MIN_SYNAPSE_DELAY_MS, MIN_SYNAPSE_WEIGHT};
use crate::neuron::Neuron;
use crate::utils::get_clamped_normal;

pub trait Synapse: Send + Sync {
    fn get_presynaptic_neuron(&self) -> Arc<RwLock<Neuron>>;
    fn get_postsynaptic_neuron(&self) -> Arc<RwLock<Neuron>>;
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
    pub presynaptic_neuron: Arc<RwLock<Neuron>>,
    pub postsynaptic_neuron: Arc<RwLock<Neuron>>,
    pub weight: f32,
    pub delay: f32, // in milliseconds
}

impl ChemicalSynapse {
    pub fn new(presynaptic_neuron: Arc<RwLock<Neuron>>, postsynaptic_neuron: Arc<RwLock<Neuron>>, rng: &mut impl Rng) -> Self {
        ChemicalSynapse {
            presynaptic_neuron,
            postsynaptic_neuron,
            weight: get_clamped_normal(MIN_SYNAPSE_WEIGHT, MAX_SYNAPSE_WEIGHT, rng),
            delay: get_clamped_normal(MIN_SYNAPSE_DELAY_MS, MAX_SYNAPSE_DELAY_MS, rng),
        }
    }
}

impl Synapse for ChemicalSynapse {
    fn get_presynaptic_neuron(&self) -> Arc<RwLock<Neuron>> {
        self.presynaptic_neuron.clone()
    }

    fn get_postsynaptic_neuron(&self) -> Arc<RwLock<Neuron>> {
        self.postsynaptic_neuron.clone()
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
