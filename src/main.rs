mod constants;
mod neuron;
mod spike_event;
mod synapse;
mod network;
mod visualization;

use image::GrayImage;
use rand::Rng;
use std::time::Instant;
use std::fs;
use std::path::Path;

use crate::neuron::Neuron;
use crate::network::Network;
use crate::synapse::{ChemicalSynapse, Synapse};

fn load_iris_images(class_path: &str) -> Vec<(GrayImage, usize)> {
    let mut images = Vec::new();
    let path = Path::new(class_path);

    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries {
            if let Ok(entry) = entry {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("jpg") {
                    if let Ok(img) = image::open(&path) {
                        let resized_img: GrayImage = img
                            .resize_exact(28, 28, image::imageops::Lanczos3)
                            .into_luma8();
                        // Store image along with a dummy label (will assign proper labels later)
                        images.push((resized_img, 0));
                    }
                }
            }
        }
    }
    images
}

fn iris_classification_test() {
    println!("--- Starting Iris Classification Training with Supervised Learning ---");

    // --- 1. Load two iris classes ---
    let class1_path = "iris_dataset/iris-setosa";
    let class2_path = "iris_dataset/iris-versicolour";

    println!("Loading images from {} and {}", class1_path, class2_path);

    let mut class1_images = load_iris_images(class1_path);
    let mut class2_images = load_iris_images(class2_path);

    // Assign labels: 0 for setosa, 1 for versicolour
    for (_, label) in &mut class1_images {
        *label = 0;
    }
    for (_, label) in &mut class2_images {
        *label = 1;
    }

    println!("Loaded {} setosa images and {} versicolour images",
             class1_images.len(), class2_images.len());

    // Combine and shuffle the dataset
    let mut all_images = Vec::new();
    all_images.extend(class1_images);
    all_images.extend(class2_images);

    // Shuffle the dataset
    let mut rng = rand::rng();
    for i in (1..all_images.len()).rev() {
        let j = rng.random_range(0..=i);
        all_images.swap(i, j);
    }

    // Split into training (80%) and testing (20%)
    let split_idx = (all_images.len() as f64 * 0.8) as usize;
    let training_set = &all_images[..split_idx];
    let test_set = &all_images[split_idx..];

    println!("Training set: {} images, Test set: {} images",
             training_set.len(), test_set.len());

    // --- 2. Network Setup with Reward and Pain Neurons ---
    const NUM_OUTPUT_NEURONS: usize = 2; // Two output neurons for binary classification
    const NUM_HIDDEN_NEURONS: usize = 100;
    const NUM_REWARD_NEURONS: usize = 2; // One reward neuron per class
    const NUM_PAIN_NEURONS: usize = 2;   // One pain neuron per class

    let num_input_neurons: usize = 28 * 28; // 784 pixels
    let total_neurons: usize = num_input_neurons + NUM_OUTPUT_NEURONS + NUM_HIDDEN_NEURONS
                               + NUM_REWARD_NEURONS + NUM_PAIN_NEURONS;

    let mut neurons = Vec::with_capacity(total_neurons);
    let mut synapses = Vec::new();
    let mut input_neurons = Vec::with_capacity(num_input_neurons);
    let mut output_neurons = Vec::with_capacity(NUM_OUTPUT_NEURONS);
    let mut reward_neurons = Vec::with_capacity(NUM_REWARD_NEURONS);
    let mut pain_neurons = Vec::with_capacity(NUM_PAIN_NEURONS);

    for _ in 0..total_neurons {
        neurons.push(Neuron::new(0.0));
    }

    // Define neuron ranges
    let input_start = 0;
    let input_end = num_input_neurons;
    let hidden_start = input_end;
    let hidden_end = hidden_start + NUM_HIDDEN_NEURONS;
    let output_start = hidden_end;
    let output_end = output_start + NUM_OUTPUT_NEURONS;
    let reward_start = output_end;
    let reward_end = reward_start + NUM_REWARD_NEURONS;
    let pain_start = reward_end;
    let pain_end = pain_start + NUM_PAIN_NEURONS;

    // Fully connected network (except reward/pain neurons)
    let mut synapse_index = 0;
    for i in 0..total_neurons {
        for j in 0..total_neurons {
            if i == j {
                continue;
            }

            // Skip connections from/to reward and pain neurons (they are special)
            if (i >= reward_start && i < pain_end) || (j >= reward_start && j < pain_end) {
                continue;
            }

            synapses.push(ChemicalSynapse::new(i, j));
            neurons[i].exiting_synapses.push(synapse_index);
            neurons[j].entering_synapses.push(synapse_index);
            synapse_index += 1;
        }
    }

    // Set up neuron indices
    for i in input_start..input_end {
        input_neurons.push(i);
    }
    for i in output_start..output_end {
        output_neurons.push(i);
    }
    for i in reward_start..reward_end {
        reward_neurons.push(i);
    }
    for i in pain_start..pain_end {
        pain_neurons.push(i);
    }

    let mut network = Network::new_supervised(
        neurons,
        synapses,
        input_neurons,
        output_neurons,
        reward_neurons,
        pain_neurons,
    );

    println!("\n--- Network Architecture ---");
    println!("Input neurons: {} (indices {}-{})", num_input_neurons, input_start, input_end-1);
    println!("Hidden neurons: {} (indices {}-{})", NUM_HIDDEN_NEURONS, hidden_start, hidden_end-1);
    println!("Output neurons: {} (indices {}-{})", NUM_OUTPUT_NEURONS, output_start, output_end-1);
    println!("Reward neurons: {} (indices {}-{})", NUM_REWARD_NEURONS, reward_start, reward_end-1);
    println!("Pain neurons: {} (indices {}-{})", NUM_PAIN_NEURONS, pain_start, pain_end-1);
    network.describe();

    // --- 3. Training Loop with Supervised Learning ---
    let steps_per_image = 50; // Simulation steps per image

    println!("\n--- Training Phase (Supervised with Reward/Pain) ---");
    let training_start = Instant::now();

    for (epoch, (img, label)) in training_set.iter().enumerate() {
        if epoch % 10 == 0 {
            println!("Processing training image {}/{} (label: {})", epoch + 1, training_set.len(), label);
        }

        // Convert image to rate coding: pixel intensity = firing rate
        let mut input_vector = Vec::with_capacity(steps_per_image);
        for _ in 0..steps_per_image {
            let mut step_input = Vec::with_capacity(num_input_neurons);
            for pixel in img.pixels() {
                let intensity = pixel[0] as f64 / 255.0;
                // Rate coding: higher intensity = stronger input current
                step_input.push(intensity * 0.1); // Scale factor for input current
            }
            input_vector.push(step_input);
        }

        // Simulate with supervised learning signal
        let _potentials = network.simulate_supervised(
            steps_per_image,
            0.3,
            &mut input_vector,
            Some(*label), // Provide the correct label
        );
    }

    println!("Training completed in {:.2?}", training_start.elapsed());
    network.describe();

    // --- 4. Testing Phase ---
    println!("\n--- Testing Phase ---");
    let mut correct_predictions = 0;

    // Confusion matrix: [true_label][predicted_label]
    // Index 0 = setosa, Index 1 = versicolour
    let mut confusion_matrix = vec![vec![0; 2]; 2];

    for (test_idx, (img, actual_label)) in test_set.iter().enumerate() {
        if test_idx % 10 == 0 {
            println!("Processing test image {}/{}", test_idx + 1, test_set.len());
        }

        // Convert image to rate coding
        let mut input_vector = Vec::with_capacity(steps_per_image);
        for _ in 0..steps_per_image {
            let mut step_input = Vec::with_capacity(num_input_neurons);
            for pixel in img.pixels() {
                let intensity = pixel[0] as f64 / 255.0;
                step_input.push(intensity * 0.1);
            }
            input_vector.push(step_input);
        }

        // Simulate without supervision (testing mode)
        let potentials = network.simulate_supervised(
            steps_per_image,
            0.3,
            &mut input_vector,
            None, // No label during testing
        );

        // Determine prediction based on output neuron activity
        let output_neuron_1_idx = output_start;
        let output_neuron_2_idx = output_start + 1;

        let mut output1_spike_count = 0;
        let mut output2_spike_count = 0;

        for potential_snapshot in &potentials {
            if potential_snapshot[output_neuron_1_idx] > 30.0 {
                output1_spike_count += 1;
            }
            if potential_snapshot[output_neuron_2_idx] > 30.0 {
                output2_spike_count += 1;
            }
        }

        let predicted_label = if output1_spike_count > output2_spike_count { 0 } else { 1 };

        // Update confusion matrix
        confusion_matrix[*actual_label][predicted_label] += 1;

        if predicted_label == *actual_label {
            correct_predictions += 1;
        }
    }

    let accuracy = (correct_predictions as f64 / test_set.len() as f64) * 100.0;

    println!("\n--- Results ---");
    println!("Test Accuracy: {:.2}% ({}/{})",
             accuracy, correct_predictions, test_set.len());

    // Display confusion matrix
    println!("\n--- Confusion Matrix ---");
    println!("                  Predicted");
    println!("                  Setosa  Versicolour");
    println!("Actual Setosa       {:>3}      {:>3}",
             confusion_matrix[0][0], confusion_matrix[0][1]);
    println!("       Versicolour  {:>3}      {:>3}",
             confusion_matrix[1][0], confusion_matrix[1][1]);

    // Calculate per-class metrics
    let setosa_total = confusion_matrix[0][0] + confusion_matrix[0][1];
    let versicolour_total = confusion_matrix[1][0] + confusion_matrix[1][1];

    if setosa_total > 0 {
        let setosa_accuracy = (confusion_matrix[0][0] as f64 / setosa_total as f64) * 100.0;
        println!("\nSetosa accuracy: {:.2}% ({}/{})",
                 setosa_accuracy, confusion_matrix[0][0], setosa_total);
    }

    if versicolour_total > 0 {
        let versicolour_accuracy = (confusion_matrix[1][1] as f64 / versicolour_total as f64) * 100.0;
        println!("Versicolour accuracy: {:.2}% ({}/{})",
                 versicolour_accuracy, confusion_matrix[1][1], versicolour_total);
    }

    // Calculate precision and recall
    let predicted_setosa_total = confusion_matrix[0][0] + confusion_matrix[1][0];
    let predicted_versicolour_total = confusion_matrix[0][1] + confusion_matrix[1][1];

    if predicted_setosa_total > 0 {
        let setosa_precision = (confusion_matrix[0][0] as f64 / predicted_setosa_total as f64) * 100.0;
        println!("\nSetosa precision: {:.2}% (when predicted setosa, correct {}/{} times)",
                 setosa_precision, confusion_matrix[0][0], predicted_setosa_total);
    }

    if predicted_versicolour_total > 0 {
        let versicolour_precision = (confusion_matrix[1][1] as f64 / predicted_versicolour_total as f64) * 100.0;
        println!("Versicolour precision: {:.2}% (when predicted versicolour, correct {}/{} times)",
                 versicolour_precision, confusion_matrix[1][1], predicted_versicolour_total);
    }

    network.describe();
}

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
    iris_classification_test();
    //simple_test();
}
