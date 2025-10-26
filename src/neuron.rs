use std::sync::{Arc, RwLock};
use crate::constants::{MAX_NEURON_DECAY_RATE, MAX_REFRACTORY_PERIOD_MS, MIN_NEURON_DECAY_RATE, MIN_REFRACTORY_PERIOD_MS, REWARD_AVERAGE_DURATION};
use rand::Rng;
use crate::datastructures::exponential_rolling_mean::EmaMeanF32;
use crate::synapse::ChemicalSynapse;

#[derive(Clone, Debug)]
pub struct Neuron {
    pub id: usize,
    membrane_potential: f32,
    threshold: f32,
    refractory_period: f32,
    decay_rate: f32,
    pub exiting_synapses: Vec<Arc<RwLock<ChemicalSynapse>>>,
    pub entering_synapses: Vec<Arc<RwLock<ChemicalSynapse>>>,
    time_of_last_fire: f32,
    last_update_time: f32,

    // Reward properties
    last_spike_magnitude: f32,
    ema_activation: EmaMeanF32,
    ema_firing_rate: EmaMeanF32
}

pub trait NeuronBehavior {
    fn new(threshold: f32, id: usize) -> Self;
    fn receive(&mut self, input_current: f32, time: f32) -> f32;
    fn full_reset(&mut self);
    fn get_potential(&mut self, time: f32) -> f32;
    fn step(&mut self, time: f32) -> bool;  // Simply return whether the neuron fired, handle the logic for propagation elsewhere
    fn will_fire(&mut self, time: f32) -> bool;
    fn time_of_last_fire(&self) -> f32;
    fn ema_activation_average(&self, time: f32) -> f32;
    fn ema_firing_rate_average(&self, time: f32) -> f32;
    fn last_spike_magnitude(&self) -> f32;
}

impl NeuronBehavior for Neuron {
    /// Constructor to create a new neuron.
    fn new(threshold: f32, id: usize) -> Self {
        // Create random generator
        let mut rng = rand::thread_rng();
        Neuron {
            id,
            membrane_potential: 0.0,
            threshold,
            refractory_period: rng.gen_range(MIN_REFRACTORY_PERIOD_MS..MAX_REFRACTORY_PERIOD_MS),
            time_of_last_fire: -f32::INFINITY,
            exiting_synapses: Vec::new(),
            entering_synapses: Vec::new(),
            last_update_time: 0.0,
            decay_rate: rng.gen_range(MIN_NEURON_DECAY_RATE..MAX_NEURON_DECAY_RATE),
            ema_activation: EmaMeanF32::new(REWARD_AVERAGE_DURATION),
            ema_firing_rate: EmaMeanF32::new(REWARD_AVERAGE_DURATION),
            last_spike_magnitude: 0.0,
        }
    }

    fn receive(&mut self, input_current: f32, time: f32) -> f32 {
        // If not in refractory period, update membrane potential
        if (time - self.time_of_last_fire) >= self.refractory_period {
            self.membrane_potential = self.get_potential(time) + input_current;
            self.last_update_time = time;
        }
        self.membrane_potential
    }
    fn full_reset(&mut self) {
        self.membrane_potential = 0.0;
        self.time_of_last_fire = 0.0;
    }

    fn get_potential(&mut self, time: f32) -> f32 {
        // Get potential of the neuron
        // account for decay since this method was last called
        let dt = time - self.last_update_time;
        self.membrane_potential = self.membrane_potential * (-self.decay_rate * dt).exp();
        self.membrane_potential
    }

    fn step(&mut self, time: f32) -> bool {
        if self.will_fire(time) {
            self.ema_activation.add(time, self.membrane_potential);

            // 2. Add to firing rate (spike) EMA. We add 1.0 to represent one spike.
            self.ema_firing_rate.add(time, 1.0);

            self.last_spike_magnitude = self.membrane_potential;
            self.time_of_last_fire = time;
            self.membrane_potential = 0.0; // Reset potential after firing
            true
        } else {
            // If it didn't fire, its rate is 0 for this time.
            self.ema_firing_rate.add(time, 0.0);
            false
        }
    }

    fn will_fire(&mut self, time: f32) -> bool {
        let potential = self.get_potential(time);
        potential >= self.threshold
    }

    fn time_of_last_fire(&self) -> f32 {
        self.time_of_last_fire
    }
    fn last_spike_magnitude(&self) -> f32 {
        self.last_spike_magnitude
    }
    fn ema_activation_average(&self, time: f32) -> f32 {
        self.ema_activation.get_mean(time).unwrap_or(0.0)
    }

    fn ema_firing_rate_average(&self, time: f32) -> f32 {
        self.ema_firing_rate.get_mean(time).unwrap_or(0.0)
    }
}
