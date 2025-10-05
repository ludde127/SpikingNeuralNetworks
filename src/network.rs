use crate::constants::*;
use crate::neuron::Neuron;
use crate::spike_event::SpikeEvent;
use crate::synapse::{ChemicalSynapse, Synapse};
use indicatif::ProgressBar;
use rayon::prelude::*;
use std::collections::VecDeque;

/// A simple struct to hold the network components.
#[derive(Clone)]
pub struct Network {
    pub neurons: Vec<Neuron>,
    pub synapses: Vec<ChemicalSynapse>,
    pub event_queue: VecDeque<SpikeEvent>,
    pub current_time: f64,
    pub input_neurons: Vec<usize>,
    pub output_neurons: Vec<usize>,
    pub reward_neurons: Vec<usize>,  // Reward neurons for correct classifications
    pub pain_neurons: Vec<usize>,    // Pain neurons for incorrect classifications
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
            reward_neurons: Vec::new(),
            pain_neurons: Vec::new(),
        }
    }

    pub fn new_supervised(
        neurons: Vec<Neuron>,
        synapses: Vec<ChemicalSynapse>,
        input_neurons: Vec<usize>,
        output_neurons: Vec<usize>,
        reward_neurons: Vec<usize>,
        pain_neurons: Vec<usize>,
    ) -> Self {
        Network {
            neurons,
            synapses,
            event_queue: VecDeque::new(),
            current_time: 0.0,
            input_neurons,
            output_neurons,
            reward_neurons,
            pain_neurons,
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
            let source_idx = event.source_neuron;

            let (source_neuron, target_neuron) = if source_idx < target_idx {
                let (left, right) = self.neurons.split_at_mut(target_idx);
                (&left[source_idx], &mut right[0])
            } else if source_idx > target_idx {
                let (left, right) = self.neurons.split_at_mut(source_idx);
                (&right[0], &mut left[target_idx])
            } else {
                // This case should ideally not happen in a well-formed network graph
                // where a neuron does not synapse itself.
                continue;
            };

            let mut target_last_spike_time = target_neuron.last_spike_time;
            let potential = POSTSYNAPTIC_POTENTIAL_AMPLITUDE * event.weight;
            let action_potential = target_neuron.receive(potential, self.current_time);
            let exiting = target_neuron.exiting_synapses.clone();

            if action_potential > 0.0 {
                target_last_spike_time = self.current_time;
                for out_syn_idx in &exiting {
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

            for syn_idx in target_neuron.entering_synapses.clone() {
                let synapse = &mut self.synapses[syn_idx];
                if synapse.weight <= MINIMUM_CHEMICAL_SYNAPSE_WEIGHT {
                    continue;
                }
                let pre_time = if source_neuron.last_spike_time == self.current_time
                    && synapse.source_neuron == event.source_neuron
                {
                    event.spike_time
                } else {
                    source_neuron.last_spike_time
                };
                if pre_time.is_finite() {
                    synapse.update_weight(pre_time, target_last_spike_time);
                }
            }
        }
    }

    /// Simulate with supervised learning using reward and pain signals
    pub fn simulate_supervised(
        &mut self,
        steps_to_simulate: usize,
        step_size_ms: f64,
        inputs: &Vec<Vec<f64>>,
        target_label: Option<usize>,
        show_progress: bool,
        track_potentials: bool,
    ) -> Vec<Vec<f64>> {
        let mut potentials: Vec<Vec<f64>> = if track_potentials {
            Vec::with_capacity(steps_to_simulate)
        } else {
            Vec::new()
        };
        let mut spike_event_counter: usize = 0;

        assert_eq!(
            steps_to_simulate,
            inputs.len(),
            "Steps to simulate and inputs length should be equal"
        );

        let simulation_end = self.current_time + (steps_to_simulate as f64) * step_size_ms;
        let pbar = if show_progress {
            Some(ProgressBar::new(steps_to_simulate as u64))
        } else {
            None
        };

        // Track output neuron spikes for reward/punishment
        let mut output_spike_counts = vec![0; self.output_neurons.len()];

        let mut step_idx = 0;
        while self.current_time + step_size_ms < simulation_end {
            self.current_time += step_size_ms;

            // 1. Present the input pattern - use indexing instead of popping
            let input = &inputs[step_idx];
            step_idx += 1;

            assert_eq!(
                input.len(),
                self.input_neurons.len(),
                "Inputs at each timestep must have values for all neurons"
            );

            for (input_neuron_idx, value) in input.iter().enumerate() {
                let spike =
                    self.neurons[input_neuron_idx].receive(*value, self.current_time);
                if spike > 0.0 {
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
            self.process_events_supervised(&mut spike_event_counter, &mut output_spike_counts);

            if track_potentials {
                let snapshot: Vec<f64> = self.neurons.iter().map(|n| n.membrane_potential).collect();
                potentials.push(snapshot);
            }

            if let Some(ref pb) = pbar {
                pb.inc(1);
            }
        }

        // Apply reward or punishment at the end of the simulation
        if let Some(correct_label) = target_label {
            self.apply_supervision(correct_label, &output_spike_counts);
        }

        if show_progress {
            println!(
                "Finished simulation, handled {} spike events",
                spike_event_counter
            );
        }
        potentials
    }

    fn process_events_supervised(
        &mut self,
        spike_event_counter: &mut usize,
        output_spike_counts: &mut Vec<usize>,
    ) {
        // Pre-create a lookup map for faster output neuron checking
        let output_neuron_indices: Vec<bool> = {
            let mut indices = vec![false; self.neurons.len()];
            for &neuron_id in self.output_neurons.iter() {
                indices[neuron_id] = true;
            }
            indices
        };

        while let Some(event) = self.event_queue.front() {
            if event.arrival_time > self.current_time {
                break;
            }
            let event = self.event_queue.pop_front().unwrap();
            *spike_event_counter += 1;
            let target_idx = event.target_neuron;
            let source_idx = event.source_neuron;

            // Extract entering_synapses before borrowing target mutably
            let entering_synapses = self.neurons[target_idx].entering_synapses.clone();

            let (source_neuron, target_neuron) = if source_idx < target_idx {
                let (left, right) = self.neurons.split_at_mut(target_idx);
                (&left[source_idx], &mut right[0])
            } else if source_idx > target_idx {
                let (left, right) = self.neurons.split_at_mut(source_idx);
                (&right[0], &mut left[target_idx])
            } else {
                // This case should ideally not happen in a well-formed network graph
                // where a neuron does not synapse itself.
                continue;
            };

            let mut target_last_spike_time = target_neuron.last_spike_time;
            let potential = POSTSYNAPTIC_POTENTIAL_AMPLITUDE * event.weight;
            let action_potential = target_neuron.receive(potential, self.current_time);
            let exiting = target_neuron.exiting_synapses.clone();

            if action_potential > 0.0 {
                target_last_spike_time = self.current_time;

                // Track output neuron spikes - optimized with lookup
                if output_neuron_indices[target_idx] {
                    for (idx, &output_neuron_id) in self.output_neurons.iter().enumerate() {
                        if target_idx == output_neuron_id {
                            output_spike_counts[idx] += 1;
                            break;
                        }
                    }
                }

                for out_syn_idx in &exiting {
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

            // Standard STDP update
            for &syn_idx in &entering_synapses {
                let synapse = &mut self.synapses[syn_idx];
                if synapse.weight <= MINIMUM_CHEMICAL_SYNAPSE_WEIGHT {
                    continue;
                }

                let pre_time = if source_neuron.last_spike_time == self.current_time
                    && synapse.source_neuron == event.source_neuron
                {
                    event.spike_time
                } else {
                    source_neuron.last_spike_time
                };
                if pre_time.is_finite() {
                    synapse.update_weight(pre_time, target_last_spike_time);
                }
            }
        }
    }

    /// Apply reward or punishment after observing network output
    fn apply_supervision(&mut self, correct_label: usize, output_spike_counts: &[usize]) {
        // Determine which output neuron spiked the most
        let predicted_label = output_spike_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let correct_output_neuron = self.output_neurons[correct_label];

        // === SOFT SUPERVISION: Scale rewards/punishments by spike proportions ===

        // Calculate total spikes across all output neurons
        let total_spikes: usize = output_spike_counts.iter().sum();

        if total_spikes == 0 {
            // No output activity - strongly reward the correct output to encourage activity
            if !self.reward_neurons.is_empty() {
                let reward_neuron_idx = self.reward_neurons[correct_label];
                self.neurons[reward_neuron_idx].last_spike_time = self.current_time;
                self.modulate_synapses_to_neuron(correct_output_neuron, 3.0);
            }
            return;
        }

        // For each output neuron, apply modulation based on its performance
        for (output_idx, &spike_count) in output_spike_counts.iter().enumerate() {
            let output_neuron = self.output_neurons[output_idx];
            let spike_fraction = spike_count as f64 / total_spikes as f64;

            if output_idx == correct_label {
                // This is the CORRECT output neuron
                // Reward proportional to how much it spiked
                // If it spiked 100% of the time: full reward (1.0)
                // If it spiked 50% of the time: partial reward (0.5)
                // If it didn't spike: no reward, but also punish to encourage it

                let modulation = if spike_fraction > 0.5 {
                    // It's already dominant - reinforce it moderately
                    1.0 + spike_fraction
                } else if spike_fraction > 0.0 {
                    // It's spiking but not dominant - strongly encourage it
                    2.0 + (1.0 - spike_fraction) * 2.0
                } else {
                    // It's not spiking at all - very strong encouragement
                    3.0
                };

                if !self.reward_neurons.is_empty() {
                    let reward_neuron_idx = self.reward_neurons[output_idx];
                    self.neurons[reward_neuron_idx].last_spike_time = self.current_time;
                    self.modulate_synapses_to_neuron(output_neuron, modulation);
                }

            } else {
                // This is an INCORRECT output neuron
                // Punish proportional to how much it spiked
                // If it spiked 100% of the time: strong punishment (-2.0)
                // If it spiked 50% of the time: moderate punishment (-1.0)
                // If it didn't spike: no punishment needed (already doing the right thing)

                if spike_fraction > 0.0 {
                    let punishment = -(1.0 + spike_fraction * 2.0);

                    if !self.pain_neurons.is_empty() && output_idx < self.pain_neurons.len() {
                        let pain_neuron_idx = self.pain_neurons[output_idx];
                        self.neurons[pain_neuron_idx].last_spike_time = self.current_time;
                        self.modulate_synapses_to_neuron(output_neuron, punishment);
                    }
                }
            }
        }
    }

    /// Apply modulation to all synapses targeting a specific neuron
    fn modulate_synapses_to_neuron(&mut self, target_neuron_idx: usize, modulation_factor: f64) {
        let entering_synapses = self.neurons[target_neuron_idx].entering_synapses.clone();
        let target_spike_time = self.neurons[target_neuron_idx].last_spike_time;

        for syn_idx in entering_synapses {
            let synapse = &mut self.synapses[syn_idx];
            let source_idx = synapse.source_neuron;
            let source_spike_time = self.neurons[source_idx].last_spike_time;

            if source_spike_time.is_finite() && target_spike_time.is_finite() {
                synapse.update_weight_modulated(source_spike_time, target_spike_time, modulation_factor);
            }
        }
    }

    /// Average synapse weights from multiple network instances (for parallel batch training)
    pub fn average_weights(&mut self, other_networks: &[Network]) {
        let num_networks = other_networks.len() + 1; // +1 for self

        for syn_idx in 0..self.synapses.len() {
            let mut total_weight = self.synapses[syn_idx].weight;
            let mut total_plasticity = self.synapses[syn_idx].plasticity;

            for other in other_networks {
                total_weight += other.synapses[syn_idx].weight;
                total_plasticity += other.synapses[syn_idx].plasticity;
            }

            self.synapses[syn_idx].weight = total_weight / num_networks as f64;
            self.synapses[syn_idx].plasticity = total_plasticity / num_networks as f64;
        }
    }

    /// Reset the network state for a new simulation (clear event queue, reset time)
    pub fn reset_state(&mut self) {
        self.event_queue.clear();
        self.current_time = 0.0;
        // Reset neuron membrane potentials and spike history to resting state
        for neuron in &mut self.neurons {
            neuron.membrane_potential = neuron.resting_potential;
            neuron.last_accessed_time = 0.0;
            neuron.last_spike_time = f64::NEG_INFINITY; // Reset spike history so neurons don't carry over action potentials
        }
    }
}
