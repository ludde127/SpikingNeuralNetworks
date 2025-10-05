use crate::constants::{
    GAMMA_HID, GAMMA_OUT, HASI_K, HASI_L1_HID, HASI_L1_OUT, HASI_L2_HID, HASI_L2_OUT, HASI_V_TH,
    LEARNING_RATE,
};
use crate::neuron::Neuron;
use crate::spike_event::SpikeEvent;
use crate::synapse::Synapse;
use plotters::prelude::*;
use std::collections::VecDeque;

#[derive(Clone)]
pub struct Network {
    pub neurons: Vec<Neuron>,
    pub synapses: Vec<Box<dyn Synapse>>,
    pub input_neuron_indices: Vec<usize>,
    pub output_neuron_indices: Vec<usize>,
}

impl Network {
    pub fn new(
        neurons: Vec<Neuron>,
        synapses: Vec<Box<dyn Synapse>>,
        input_neuron_indices: Vec<usize>,
        output_neuron_indices: Vec<usize>,
    ) -> Self {
        Network {
            neurons,
            synapses,
            input_neuron_indices,
            output_neuron_indices,
        }
    }

    pub fn reset_state(&mut self) {
        for neuron in &mut self.neurons {
            neuron.reset();
        }
    }

    // --- HaSiST Functions ---

    /// Applies the Hard Sigmoid (HaSi) activation function.
    fn hasi_activation(&self, potential: f64, layer: u8) -> f64 {
        let (l1, l2) = if layer == 1 {
            (HASI_L1_HID, HASI_L2_HID)
        } else {
            (HASI_L1_OUT, HASI_L2_OUT)
        };

        if potential < l1 {
            0.0
        } else if potential > l2 {
            1.0
        } else {
            (HASI_K * (potential - HASI_V_TH) + 1.0) / 2.0
        }
    }

    /// Calculates the derivative of the HaSi function.
    fn hasi_derivative(&self, potential: f64, layer: u8) -> f64 {
        let (l1, l2, gamma) = if layer == 1 {
            (HASI_L1_HID, HASI_L2_HID, GAMMA_HID)
        } else {
            (HASI_L1_OUT, HASI_L2_OUT, GAMMA_OUT)
        };

        if potential >= l1 && potential <= l2 {
            (gamma * HASI_K) / 2.0
        } else {
            0.0
        }
    }

    /// Performs one forward pass for the surrogate network (ANN mode).
    fn forward_pass_surrogate(
        &self,
        inputs: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut hidden_potentials = vec![0.0; self.neurons.len()];
        let mut hidden_activations = vec![0.0; self.neurons.len()];
        let mut output_potentials = vec![0.0; self.neurons.len()];
        let mut output_activations = vec![0.0; self.neurons.len()];

        // Hidden Layer
        for neuron in &self.neurons {
            if neuron.layer == 1 {
                // Hidden
                let mut potential = 0.0;
                for &syn_idx in &neuron.entering_synapses {
                    let synapse = &self.synapses[syn_idx];
                    let presynaptic_neuron_id = synapse.get_presynaptic_neuron();
                    // Check if the presynaptic neuron is in the input layer
                    if self.neurons[presynaptic_neuron_id].layer == 0 {
                        potential += synapse.get_weight() * inputs[presynaptic_neuron_id];
                    }
                }
                hidden_potentials[neuron.id] = potential;
                hidden_activations[neuron.id] = self.hasi_activation(potential, 1);
            }
        }

        // Output Layer
        for neuron in &self.neurons {
            if neuron.layer == 2 {
                // Output
                let mut potential = 0.0;
                for &syn_idx in &neuron.entering_synapses {
                    let synapse = &self.synapses[syn_idx];
                    let presynaptic_neuron_id = synapse.get_presynaptic_neuron();
                    // Check if the presynaptic neuron is in the hidden layer
                    if self.neurons[presynaptic_neuron_id].layer == 1 {
                        potential +=
                            synapse.get_weight() * hidden_activations[presynaptic_neuron_id];
                    }
                }
                output_potentials[neuron.id] = potential;
                output_activations[neuron.id] = self.hasi_activation(potential, 2);
            }
        }

        (
            hidden_potentials,
            hidden_activations,
            output_potentials,
            output_activations,
        )
    }

    /// Performs backpropagation and updates weights.
    fn backward_pass(
        &mut self,
        inputs: &[f64],
        label: usize,
        hidden_potentials: &[f64],
        hidden_activations: &[f64],
        output_potentials: &[f64],
        output_activations: &[f64],
    ) -> f64 {
        let mut targets = vec![0.0; self.output_neuron_indices.len()];
        targets[label] = 1.0;

        let mut output_errors = vec![0.0; self.neurons.len()];
        let mut hidden_errors = vec![0.0; self.neurons.len()];
        let mut total_loss = 0.0;

        // 1. Calculate output layer errors and loss
        for (i, &neuron_idx) in self.output_neuron_indices.iter().enumerate() {
            let error = output_activations[neuron_idx] - targets[i];
            total_loss += error * error; // MSE
            let derivative = self.hasi_derivative(output_potentials[neuron_idx], 2);
            output_errors[neuron_idx] = error * derivative;
        }

        // 2. Calculate hidden layer errors
        for neuron in &self.neurons {
            if neuron.layer == 1 {
                // Hidden
                let mut error = 0.0;
                for &syn_idx in &neuron.exiting_synapses {
                    let synapse = &self.synapses[syn_idx];
                    let postsynaptic_neuron_id = synapse.get_postsynaptic_neuron();
                    if self.neurons[postsynaptic_neuron_id].layer == 2 {
                        error += output_errors[postsynaptic_neuron_id] * synapse.get_weight();
                    }
                }
                let derivative = self.hasi_derivative(hidden_potentials[neuron.id], 1);
                hidden_errors[neuron.id] = error * derivative;
            }
        }

        // 3. Update output layer weights (Hidden -> Output)
        for neuron in &self.neurons {
            if neuron.layer == 2 {
                // Output
                for &syn_idx in &neuron.entering_synapses {
                    let synapse = &self.synapses[syn_idx];
                    let presynaptic_id = synapse.get_presynaptic_neuron();
                    if self.neurons[presynaptic_id].layer == 1 {
                        let delta_w = -LEARNING_RATE
                            * output_errors[neuron.id]
                            * hidden_activations[presynaptic_id];
                        self.synapses[syn_idx].update_weight(delta_w);
                    }
                }
            }
        }

        // 4. Update hidden layer weights (Input -> Hidden)
        for neuron in &self.neurons {
            if neuron.layer == 1 {
                // Hidden
                for &syn_idx in &neuron.entering_synapses {
                    let synapse = &self.synapses[syn_idx];
                    let presynaptic_id = synapse.get_presynaptic_neuron();
                    if self.neurons[presynaptic_id].layer == 0 {
                        let delta_w =
                            -LEARNING_RATE * hidden_errors[neuron.id] * inputs[presynaptic_id];
                        self.synapses[syn_idx].update_weight(delta_w);
                    }
                }
            }
        }

        total_loss / 2.0 // Return average loss for the sample
    }

    /// Train on a single sample using the HaSiST algorithm.
    pub fn train_on_sample(
        &mut self,
        input_vector: &[Vec<f64>],
        label: usize,
        _steps: usize,
        _step_size_ms: f64,
    ) -> (f64, usize) {
        // For HaSiST, we ignore temporal dynamics during training as per the paper.
        // We use the input from the first timestep.
        let inputs = &input_vector[0];

        // Forward pass
        let (hidden_potentials, hidden_activations, output_potentials, output_activations) =
            self.forward_pass_surrogate(inputs);

        // Backward pass and weight update
        let loss = self.backward_pass(
            inputs,
            label,
            &hidden_potentials,
            &hidden_activations,
            &output_potentials,
            &output_activations,
        );

        // Get prediction for this sample
        let predicted_label = output_activations
            .iter()
            .enumerate()
            .filter(|(i, _)| self.output_neuron_indices.contains(i))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| self.output_neuron_indices.iter().position(|&n| n == i).unwrap())
            .unwrap_or(0);

        (loss, predicted_label)
    }

    /// Predict the class for a given input vector using the SNN.
    pub fn predict(
        &self,
        input_vector: &[Vec<f64>],
        steps: usize,
        step_size_ms: f64,
        output_neuron_indices: &[usize],
    ) -> usize {
        let mut temp_network = self.clone();
        temp_network.reset_state();

        let mut spike_events = VecDeque::new();
        let mut output_spike_counts = vec![0; self.output_neuron_indices.len()];

        for t in 0..steps {
            // Apply input currents
            for (neuron_idx, &current) in self.input_neuron_indices.iter().zip(input_vector[t].iter()) {
                if temp_network.neurons[*neuron_idx].refractory_period == 0 {
                    temp_network.neurons[*neuron_idx].membrane_potential += current;
                }
            }

            // Process neurons and generate spikes
            for i in 0..temp_network.neurons.len() {
                let neuron = &mut temp_network.neurons[i];
                if neuron.refractory_period > 0 {
                    neuron.refractory_period -= 1;
                    continue;
                }

                // Leaky integrate
                neuron.membrane_potential *= 1.0 - (step_size_ms / 100.0); // Leak

                if neuron.membrane_potential > neuron.threshold {
                    neuron.membrane_potential = 0.0; // Reset potential
                    neuron.refractory_period = 5; // 5 ms refractory period

                    // Generate spike events for exiting synapses
                    for &syn_idx in &neuron.exiting_synapses {
                        let synapse = &temp_network.synapses[syn_idx];
                        spike_events.push_back(SpikeEvent {
                            synapse_index: syn_idx,
                            delivery_time: t + synapse.get_delay() as usize,
                        });
                    }

                    // Count output spikes
                    if let Some(output_idx) = output_neuron_indices.iter().position(|&n| n == i) {
                        output_spike_counts[output_idx] += 1;
                    }
                }
            }

            // Process spike events
            let current_time = t;
            let mut processed_spikes = 0;
            for event in &spike_events {
                if event.delivery_time == current_time {
                    let synapse = &temp_network.synapses[event.synapse_index];
                    let postsynaptic_neuron = &mut temp_network.neurons[synapse.get_postsynaptic_neuron()];
                    if postsynaptic_neuron.refractory_period == 0 {
                        postsynaptic_neuron.membrane_potential += synapse.get_weight();
                    }
                    processed_spikes += 1;
                } else if event.delivery_time < current_time {
                    processed_spikes += 1;
                }
            }
            for _ in 0..processed_spikes {
                spike_events.pop_front();
            }
        }

        // Determine prediction based on spike counts
        output_spike_counts
            .iter()
            .enumerate()
            .max_by_key(|&(_, &count)| count)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    // --- Utility and Visualization ---

    pub fn describe(&self) {
        println!("--- Network State ---");
        println!("Total neurons: {}", self.neurons.len());
        println!("Total synapses: {}", self.synapses.len());

        let mut min_w = f64::MAX;
        let mut max_w = f64::MIN;
        let mut avg_w = 0.0;
        for s in &self.synapses {
            let w = s.get_weight();
            if w < min_w {
                min_w = w;
            }
            if w > max_w {
                max_w = w;
            }
            avg_w += w;
        }
        avg_w /= self.synapses.len() as f64;

        println!(
            "Synapse weights: min={:.4}, max={:.4}, avg={:.4}",
            min_w, max_w, avg_w
        );
    }

    pub fn plot_synapse_weights(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        let weights: Vec<f64> = self.synapses.iter().map(|s| s.get_weight()).collect();
        if weights.is_empty() {
            return Ok(());
        }
        let max_weight = weights.iter().cloned().fold(f64::MIN, f64::max);
        let min_weight = weights.iter().cloned().fold(f64::MAX, f64::min);

        let n_bins = 20;
        let mut bins = vec![0; n_bins];
        let bucket_size = (max_weight - min_weight) / n_bins as f64;

        if bucket_size <= 0.0 {
            // Handle case where all weights are the same
            let mut chart = ChartBuilder::on(&root)
                .caption(
                    "Synapse Weight Distribution (All weights are equal)",
                    ("sans-serif", 50).into_font(),
                )
                .build_cartesian_2d(min_weight..max_weight, 0..weights.len() as u32)?;
            chart.configure_mesh().draw()?;
            return Ok(());
        }

        for &weight in &weights {
            let mut bin_index = ((weight - min_weight) / bucket_size).floor() as usize;
            if bin_index >= n_bins {
                bin_index = n_bins - 1;
            }
            bins[bin_index] += 1;
        }

        let max_count = *bins.iter().max().unwrap_or(&0) as u32;

        let mut chart = ChartBuilder::on(&root)
            .caption("Synapse Weight Distribution", ("sans-serif", 50).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(min_weight..max_weight, 0..max_count)?;

        chart.configure_mesh().draw()?;

        chart.draw_series((0..n_bins).map(|i| {
            let x0 = min_weight + i as f64 * bucket_size;
            let x1 = min_weight + (i + 1) as f64 * bucket_size;
            let y = bins[i] as u32;
            let mut bar = Rectangle::new([(x0, 0), (x1, y)], RED.mix(0.5).filled());
            bar.set_margin(0, 0, 5, 5);
            bar
        }))?;

        root.present()?;
        Ok(())
    }
}
