mod constants;
mod data;
mod network;
mod neuron;
mod setup;
mod spike_event;
mod synapse;
mod testing;
mod training;
mod visualization;

use crate::data::load_and_prep_iris_data;
use crate::setup::setup_network;
use crate::testing::test_network;
use crate::training::train_network;

fn iris_classification_test() {
    println!("--- Starting Iris Classification Training with Supervised Learning ---");

    // --- 1. Load and Prepare Data ---
    let (training_set, test_set) = load_and_prep_iris_data();

    // --- 2. Network Setup ---
    let (
        mut network,
        _input_neuron_indices,
        output_neuron_indices,
        _hidden_neuron_indices,
        _reward_neuron_indices,
        _pain_neuron_indices,
    ) = setup_network();

    network.describe();

    // --- 3. Training Phase ---
    train_network(&mut network, &training_set);
    network.describe();

    // --- 4. Testing Phase ---
    test_network(&network, &test_set, &output_neuron_indices);

    network.plot_synapse_weights("synapse_weights.png").unwrap();
    network.describe();
}

fn main() {
    iris_classification_test();
}
