mod constants;
mod neuron;
mod spike_event;
mod synapse;
mod network;
mod visualization;

use image::GrayImage;
use rand::Rng;
use std::time::Instant;

use crate::neuron::Neuron;
use crate::network::Network;
use crate::synapse::{ChemicalSynapse, Synapse};

fn simple_test() {
    println!("--- Starting Neuromorphic Network Test ---");

    // --- 1. Network Setup ---
    const NUM_INPUT_NEURONS: usize = 2;
    const NUM_OUTPUT_NEURONS: usize = 1;
    const NUM_HIDDEN_NEURONS: usize = 500;
    const TOTAL_NEURONS: usize = NUM_INPUT_NEURONS + NUM_OUTPUT_NEURONS + NUM_HIDDEN_NEURONS;

    let mut neurons = Vec::with_capacity(TOTAL_NEURONS);
    let mut synapses = Vec::with_capacity(TOTAL_NEURONS);
    let mut input_neurons = Vec::with_capacity(NUM_INPUT_NEURONS);
    let mut output_neurons = Vec::with_capacity(NUM_OUTPUT_NEURONS);

    // Create all neurons
    for _ in 0..TOTAL_NEURONS {
        neurons.push(Neuron::new(0.0));
    }

    // Connect every input neuron to every output neuron
    let mut synapse_index = 0;
    for i in 0..TOTAL_NEURONS {
        for j in 0..TOTAL_NEURONS {
            if i == j {
                continue;
            }
            synapses.push(ChemicalSynapse::new(i, j));
            neurons[i].exiting_synapses.push(synapse_index);
            neurons[j].entering_synapses.push(synapse_index);
            synapse_index += 1;
        }
    }

    for i in 0..NUM_INPUT_NEURONS {
        input_neurons.push(i);
    }

    for i in TOTAL_NEURONS - NUM_OUTPUT_NEURONS..TOTAL_NEURONS {
        output_neurons.push(i);
    }

    let mut network = Network::new(neurons, synapses, input_neurons, output_neurons);
    network.describe();

    let steps_to_simulate = 1000;

    let mut input_vector = Vec::with_capacity(steps_to_simulate);

    let mut rng = rand::rng();
    let amplitude = 0.05;
    for i in 0..steps_to_simulate {
        // Every 20 steps, present either pattern A or B
        if i % 50 == 0 {
            if rng.random_bool(0.5) {
                // Pattern A: neurons 0 and 2 spike
                input_vector.push(vec![amplitude, 0.0]);
            } else {
                // Pattern B: neurons 1 and 3 spike
                input_vector.push(vec![0.0, amplitude]);
            }
        } else {
            // Silence otherwise
            input_vector.push(vec![0.0, 0.0]);
        }
    }
    input_vector.reverse();

    let start = Instant::now();
    let potentials = network.simulate(steps_to_simulate, 0.3, &mut input_vector);
    // --- 3. Results ---
    //println!("\n--- Final Synapse Weights after simulation ---",);
    //network.print_synapse_weight();
    println!("--- Simulation completed in {:.2?} ---", start.elapsed());

    // Export network graph
    //network.visualize_graph("network.dot");

    // Plot membrane potentials
    network
        .plot_membrane_potentials(&potentials, "membrane.png")
        .unwrap();
    network.describe();
}

fn main() {
    println!("--- Starting Neuromorphic Network Test ---");

    // --- 1. Image Loading and Preprocessing ---
    let image_path = "test.jpg";

    let img = image::open(image_path).expect("Failed to open image file");

    let resized_img: GrayImage = img
        .resize_exact(28, 28, image::imageops::Lanczos3)
        .into_luma8();
    let (width, height) = resized_img.dimensions();

    // Convert pixel data to a normalized vector of firing rates (0.0 to 1.0)
    let firing_rates: Vec<f64> = resized_img.pixels().map(|p| p[0] as f64 / 255.0).collect();

    // --- 2. Network Setup ---
    const NUM_OUTPUT_NEURONS: usize = 1;
    const NUM_HIDDEN_NEURONS: usize = 50;

    let num_input_neurons: usize = (width * height) as usize;
    let total_neurons: usize = num_input_neurons + NUM_OUTPUT_NEURONS + NUM_HIDDEN_NEURONS;

    let mut neurons = Vec::with_capacity(total_neurons);
    let mut synapses = Vec::new();
    let mut input_neurons = Vec::with_capacity(num_input_neurons);
    let mut output_neurons = Vec::with_capacity(NUM_OUTPUT_NEURONS);

    for _ in 0..total_neurons {
        neurons.push(Neuron::new(0.0));
    }

    let mut synapse_index = 0;
    for i in 0..total_neurons {
        for j in 0..total_neurons {
            if i == j {
                continue;
            }
            synapses.push(ChemicalSynapse::new(i, j));
            neurons[i].exiting_synapses.push(synapse_index);
            neurons[j].entering_synapses.push(synapse_index);
            synapse_index += 1;
        }
    }

    for i in 0..num_input_neurons {
        input_neurons.push(i);
    }

    for i in total_neurons - NUM_OUTPUT_NEURONS..total_neurons {
        output_neurons.push(i);
    }

    let mut network = Network::new(neurons, synapses, input_neurons, output_neurons);
    network.describe();

    // --- 3. Poisson Spike Encoding and Simulation ---
    let steps_to_simulate = 100;
    let mut input_vector = Vec::with_capacity(steps_to_simulate);

    let mut rng = rand::rng();

    let max_rate = 0.5; // Maximum firing rate (e.g., 500 Hz if dt=1ms)

    // Generate a spike train for each input neuron over time
    for _ in 0..steps_to_simulate {
        let mut step_input = Vec::with_capacity(num_input_neurons);
        for &rate in &firing_rates {
            // The probability of a spike is proportional to the firing rate
            if rng.random::<f64>() < rate * max_rate {
                step_input.push(1.0); // A spike event is represented by a value of 1.0
            } else {
                step_input.push(0.0); // No spike
            }
        }
        input_vector.push(step_input);
    }

    let start = Instant::now();
    let potentials = network.simulate(steps_to_simulate, 0.3, &mut input_vector);

    println!("--- Simulation completed in {:.2?} ---", start.elapsed());

    network
        .plot_membrane_potentials(&potentials, "membrane.png")
        .unwrap();
    network.plot_synapse_weights("synapse_weights.png").unwrap();
    network.describe();
}
