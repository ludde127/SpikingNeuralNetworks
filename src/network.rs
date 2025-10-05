use crate::constants::*;
use crate::neuron::Neuron;
use crate::spike_event::SpikeEvent;
use crate::synapse::{ChemicalSynapse, Synapse};
use indicatif::ProgressBar;
use std::collections::VecDeque;

/// A simple struct to hold the network components.
pub struct Network {
    pub neurons: Vec<Neuron>,
    pub synapses: Vec<ChemicalSynapse>,
    pub event_queue: VecDeque<SpikeEvent>,
    pub current_time: f64,
    pub input_neurons: Vec<usize>,
    pub output_neurons: Vec<usize>,
}

impl Network {
    pub fn new(
        neurons: Vec<Neuron>,
        synapses: Vec<ChemicalSynapse>,
        input_neurons: Vec<usize>,
        output_neurons: Vec<usize>,
    ) -> Self {
        Network {
            neurons,
            synapses,
            event_queue: VecDeque::new(),
            current_time: 0.0,
            input_neurons,
            output_neurons,
        }
    }

    pub fn describe(&self) {
        println!(
            "Network created with {} input neurons, {} output neurons, {} hidden neurons, and {} synapses.",
            self.input_neurons.len(),
            self.output_neurons.len(),
            self.neurons.len() - self.input_neurons.len() - self.output_neurons.len(),
            self.synapses.len()
        );

        let total_weight: f64 = self.synapses.iter().map(|s| s.weight).sum();
        let avg_weight = total_weight / self.synapses.len() as f64;
        let max_weight = self
            .synapses
            .iter()
            .map(|s| s.weight)
            .fold(MINIMUM_CHEMICAL_SYNAPSE_WEIGHT, |a, b| a.max(b));
        let min_weight = self
            .synapses
            .iter()
            .map(|s| s.weight)
            .fold(MAXIMUM_CHEMICAL_SYNAPSE_WEIGHT, |a, b| a.min(b));
        println!(
            "Synapse weights - Avg: {:.4}, Min: {:.4}, Max: {:.4}",
            avg_weight, min_weight, max_weight
        );
        let total_plasticity: f64 = self.synapses.iter().map(|s| s.plasticity).sum();
        let avg_plasticity = total_plasticity / self.synapses.len() as f64;
        let max_plasticity = self
            .synapses
            .iter()
            .map(|s| s.plasticity)
            .fold(0.0f64, |a, b| a.max(b));
        let min_plasticity = self
            .synapses
            .iter()
            .map(|s| s.plasticity)
            .fold(f64::INFINITY, |a, b| a.min(b));
        println!(
            "Synapse plasticity - Avg: {:.4}, Min: {:.4}, Max: {:.4}",
            avg_plasticity, min_plasticity, max_plasticity
        );

        // Num synapses with weight equal to minimum
        let num_min_weight = self
            .synapses
            .iter()
            .filter(|s| (s.weight - MINIMUM_CHEMICAL_SYNAPSE_WEIGHT).abs() < f64::EPSILON)
            .count();
        let num_max_weight = self
            .synapses
            .iter()
            .filter(|s| (s.weight - MAXIMUM_CHEMICAL_SYNAPSE_WEIGHT).abs() < f64::EPSILON)
            .count();
        println!(
            "Synapses at weight bounds - Min weight: {}, Max weight: {}",
            num_min_weight, num_max_weight
        );
        println!("---");
    }

    pub fn print_synapse_weight(&self) {
        for synapse in &self.synapses {
            println!(
                "Synapse ({:_>2} -> {:_>2}): {:.4}",
                synapse.source_neuron, synapse.target_neuron, synapse.weight
            );
        }
    }

    pub fn simulate(
        &mut self,
        steps_to_simulate: usize,
        step_size_ms: f64,
        inputs: &mut Vec<Vec<f64>>,
    ) -> Vec<Vec<f64>> {
        let mut potentials: Vec<Vec<f64>> = Vec::new();

        let mut spike_event_counter: usize = 0;
        assert_eq!(
            steps_to_simulate,
            inputs.len(),
            "Steps to simulate and inputs length should be equal"
        );
        let simulation_end = self.current_time + (steps_to_simulate as f64) * step_size_ms;
        let pbar = ProgressBar::new(steps_to_simulate as u64);
        while self.current_time + step_size_ms < simulation_end {
            self.current_time += step_size_ms;

            // 1. Present the input pattern

            let input = inputs.pop().unwrap();
            assert_eq!(
                input.len(),
                self.input_neurons.len(),
                "Inputs at each timestep must have values for all neurons"
            );

            for (input_neuron_idx, value) in input.iter().enumerate() {
                let spike =
                    self.neurons[input_neuron_idx].receive(value.clone(), self.current_time);
                if spike > 0.0 {
                    //println!("Input Neuron {} spiked", input_neuron_idx);
                    // The neuron spiked so we must propagate it
                    let exiting_synapses = &self.neurons[input_neuron_idx].exiting_synapses;
                    for &synapse_idx in exiting_synapses {
                        let synapse = &self.synapses[synapse_idx];
                        if synapse.weight <= MINIMUM_CHEMICAL_SYNAPSE_WEIGHT {
                            continue;
                        }
                        self.event_queue.push_back(SpikeEvent {
                            source_neuron: input_neuron_idx,
                            target_neuron: synapse.target_neuron,
                            spike_time: self.current_time,
                            arrival_time: self.current_time + SYNAPSE_SPIKE_TIME,
                            weight: synapse.weight,
                            synapse_index: synapse_idx,
                        });
                    }
                    self.neurons[input_neuron_idx].last_spike_time = self.current_time;
                }
            }

            // Deliver all spike events scheduled for this timestep
            self.process_events(&mut spike_event_counter);
            let snapshot: Vec<f64> = self.neurons.iter().map(|n| n.membrane_potential).collect();
            potentials.push(snapshot);
            pbar.inc(1);
        }
        println!(
            "Finished simulation, handled {} spike events",
            spike_event_counter
        );
        potentials
    }

    fn process_events(&mut self, spike_event_counter: &mut usize) {
        while let Some(event) = self.event_queue.front() {
            if event.arrival_time > self.current_time {
                break;
            }
            let event = self.event_queue.pop_front().unwrap();
            *spike_event_counter += 1;
            let target_idx = event.target_neuron;
            let target = &mut self.neurons[target_idx];
            let mut target_last_spike_time = target.last_spike_time;
            let potential = POSTSYNAPTIC_POTENTIAL_AMPLITUDE * event.weight;
            let action_potential = target.receive(potential, self.current_time);
            let exiting = &target.exiting_synapses;
            if action_potential > 0.0 {
                target_last_spike_time = self.current_time;
                for out_syn_idx in exiting {
                    let out_syn = &self.synapses[*out_syn_idx];
                    if out_syn.weight <= MINIMUM_CHEMICAL_SYNAPSE_WEIGHT {
                        continue;
                    }
                    self.event_queue.push_back(SpikeEvent {
                        source_neuron: target_idx,
                        target_neuron: out_syn.target_neuron,
                        spike_time: self.current_time,
                        arrival_time: self.current_time + SYNAPSE_SPIKE_TIME,
                        weight: out_syn.weight,
                        synapse_index: *out_syn_idx,
                    });
                }
            }
            for syn_idx in target.entering_synapses.clone() {
                let synapse = &mut self.synapses[syn_idx];
                if synapse.weight <= MINIMUM_CHEMICAL_SYNAPSE_WEIGHT {
                    continue;
                }
                let source_idx = synapse.source_neuron;
                let neuron = &self.neurons[source_idx];
                let pre_time = if neuron.last_spike_time == self.current_time
                    && source_idx == event.source_neuron
                {
                    event.spike_time
                } else {
                    neuron.last_spike_time
                };
                if pre_time.is_finite() {
                    synapse.update_weight(pre_time, target_last_spike_time);
                }
            }
        }
    }
}
