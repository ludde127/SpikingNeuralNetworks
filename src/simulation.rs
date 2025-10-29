use crate::neuron::{Neuron, NeuronBehavior};
use crate::reward_system::RewardSystem;
use crate::spike_event::SpikeEvent;
use crate::synapse::{ChemicalSynapse, Synapse};
use graphviz_rust::print;
use rand::Rng;
use std::collections::VecDeque;
use std::sync::{Arc, RwLock};

pub struct Simulation {
    spike_queue: VecDeque<Arc<RwLock<SpikeEvent>>>,
    dt: f32,
    pub time: f32,
    neurons: Vec<Arc<RwLock<Neuron>>>,
    reward_system: RewardSystem,
    current_trial_spike_events: Vec<Arc<RwLock<SpikeEvent>>>,
}

impl Simulation {
    pub fn new(dt: f32, input_neurons: Vec<Arc<RwLock<Neuron>>>) -> Self {
        Simulation {
            spike_queue: VecDeque::new(),
            dt,
            time: 0.0,
            neurons: input_neurons,
            reward_system: RewardSystem::new(),
            current_trial_spike_events: Vec::new(),
        }
    }

    pub fn reward(&mut self, reward: f32) {
        self.reward_system.add_reward(self.time, reward);
        if self.current_trial_spike_events.is_empty() {
            return;
        }
        self.reward_system
            .update_synapses(self.time, &self.current_trial_spike_events);
        self.current_trial_spike_events = Vec::new();
    }

    fn send_action_potential(&mut self, neuron: Arc<RwLock<Neuron>>) {
        let n = neuron.read().unwrap();
        for syn in &n.exiting_synapses {
            let wsyn = syn.write().unwrap();
            let spike_event = Arc::new(RwLock::new(SpikeEvent {
                synapse: syn.clone(),
                delivery_time: self.time + wsyn.delay,
                presynaptic_ema_firing_rate_before_spike: neuron.read().unwrap().ema_firing_rate_before_last_spike
            }));
            self.spike_queue.push_back(spike_event);
        }
    }

    fn step_process_nodes(&mut self, neurons: Vec<Arc<RwLock<Neuron>>>) {
        let mut firing_neurons = Vec::with_capacity(neurons.len() * 5);
        for neuron in &neurons {
            let fired = {
                let mut n = neuron.write().unwrap();
                n.step(self.time)
            };
            if fired {
                // Neuron fired, create spike events
                firing_neurons.push(neuron.clone());
            }
        }
        for neuron in firing_neurons {
            self.send_action_potential(neuron.clone());
        }
    }

    pub fn input_external_stimuli(&mut self, node: Arc<RwLock<Neuron>>, magnitude: f32) {
        node.write().unwrap().receive(magnitude, self.time);
        self.step_process_nodes(vec![node]);
    }

    pub fn random_noise(&mut self, min: f32, max: f32, percent: f32, rng: &mut impl Rng) {
        // Adds random noise to a percentage of neurons
        let num_neurons = (self.neurons.len() as f32 * percent).ceil() as usize;
        let mut selected_indices = Vec::with_capacity(num_neurons);
        let mut modified_neurons = Vec::with_capacity(num_neurons);
        while selected_indices.len() < num_neurons {
            let idx = rng.gen_range(0..self.neurons.len());
            if !selected_indices.contains(&idx) {
                selected_indices.push(idx);
                self.neurons[idx]
                    .write()
                    .unwrap()
                    .receive(rng.gen_range(min..max), self.time);
                modified_neurons.push(self.neurons[idx].clone());
            }
        }
        self.step_process_nodes(modified_neurons);
    }

    pub(crate) fn step(&mut self) {
        // Process external stimuli
        self.time += self.dt;
        self.process_events();
    }

    fn process_events(&mut self) {
        // May be roughly correct size
        let mut new_firing_neurons = Vec::with_capacity(self.spike_queue.len() * 5);
        while let Some(event) = self.spike_queue.front() {
            let delivery_time = event.read().unwrap().delivery_time as f32;
            if delivery_time <= self.time {
                let event = self.spike_queue.pop_front().unwrap();
                let synapse = event.read().unwrap().synapse.clone();
                let wsyn = synapse.write().unwrap();
                let post_neuron = wsyn.get_postsynaptic_neuron();
                let mut n = post_neuron.write().unwrap();
                n.receive(wsyn.get_weight(), self.time);

                if n.will_fire(delivery_time) {
                    // Neuron fired, create spike events
                    new_firing_neurons.push(post_neuron.clone());
                    self.current_trial_spike_events.push(event)
                }
            } else {
                break;
            }
        }

        // Could be more optimized if we check for duplicated nodes in the new_firing_neurons
        self.step_process_nodes(new_firing_neurons);
    }
}
