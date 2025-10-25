use std::collections::VecDeque;
use std::sync::{Arc, RwLock};
use crate::neuron::{Neuron, NeuronBehavior};
use crate::reward_system::RewardSystem;
use crate::spike_event::SpikeEvent;
use crate::synapse::{ChemicalSynapse, Synapse};

pub struct Simulation {
    spike_queue: VecDeque<Arc<RwLock<SpikeEvent>>>,
    dt: f32,
    pub time: f32,
    neurons_with_external_stimuli: Vec<Arc<RwLock<Neuron>>>,
    reward_system: RewardSystem,
    last_iteration_processes_spike_events : Vec<Arc<RwLock<SpikeEvent>>>
}

impl Simulation {
    pub fn new(dt: f32, input_neurons: Vec<Arc<RwLock<Neuron>>>) -> Self {
        Simulation {
            spike_queue: VecDeque::new(),
            dt,
            time: 0.0,
            neurons_with_external_stimuli: input_neurons,
            reward_system: RewardSystem::new(),
            last_iteration_processes_spike_events: Vec::new(),
        }
    }
    
    pub fn reward(&mut self, reward: f32) {
        self.reward_system.update_synapses(self.time, &self.last_iteration_processes_spike_events);
        self.reward_system.add_reward(self.time, reward);
        self.last_iteration_processes_spike_events = Vec::new();
    }

    fn send_action_potential(&mut self, neuron: Arc<RwLock<Neuron>>) {
        let n = neuron.read().unwrap();
        for syn in &n.exiting_synapses {
            let wsyn = syn.write().unwrap();
            let delivery_time = self.time + wsyn.delay;
            let spike_event = Arc::new(RwLock::new(SpikeEvent {
                synapse: syn.clone(),
                delivery_time: delivery_time as usize,
            }));
            self.spike_queue.push_back(spike_event);
        }
    }

    fn step_process_nodes(&mut self, neurons: Vec<Arc<RwLock<Neuron>>>) {
        let mut firing_neurons = Vec::with_capacity(neurons.len() * 5);
        for neuron in &neurons{
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

    pub(crate) fn step(&mut self) {
        // Process external stimuli
        self.time += self.dt;
        self.step_process_nodes(self.neurons_with_external_stimuli.clone());
        self.process_events();
    }

    fn process_events(&mut self) {
        // May be roughly correct size
        self.last_iteration_processes_spike_events = vec![];
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
                    self.last_iteration_processes_spike_events.push(event)
                }
            } else {
                break;
            }
        }

        // Could be more optimized if we check for duplicated nodes in the new_firing_neurons
        self.step_process_nodes(new_firing_neurons);
    }
}

