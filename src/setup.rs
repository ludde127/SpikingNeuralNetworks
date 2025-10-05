use crate::constants::{
    NUM_HIDDEN_NEURONS, NUM_INPUT_NEURONS, NUM_OUTPUT_NEURONS,
};
use crate::network::Network;
use crate::neuron::Neuron;
use crate::synapse::{ChemicalSynapse, Synapse};

pub fn setup_network() -> (
    Network,
    Vec<usize>, // input
    Vec<usize>, // output
    Vec<usize>, // hidden
    Vec<usize>, // reward (empty)
    Vec<usize>, // pain (empty)
) {
    let total_neurons: usize = NUM_INPUT_NEURONS + NUM_HIDDEN_NEURONS + NUM_OUTPUT_NEURONS;

    let mut neurons = Vec::with_capacity(total_neurons);
    for i in 0..total_neurons {
        neurons.push(Neuron::new(0.0, i));
    }

    let mut synapses: Vec<Box<dyn Synapse>> = Vec::new();
    let mut input_neurons = Vec::with_capacity(NUM_INPUT_NEURONS);
    let mut output_neurons = Vec::with_capacity(NUM_OUTPUT_NEURONS);
    let mut hidden_neurons = Vec::with_capacity(NUM_HIDDEN_NEURONS);

    // Define neuron ranges
    let input_start = 0;
    let input_end = NUM_INPUT_NEURONS;
    let hidden_start = input_end;
    let hidden_end = hidden_start + NUM_HIDDEN_NEURONS;
    let output_start = hidden_end;
    let output_end = output_start + NUM_OUTPUT_NEURONS;

    // --- Connect Neurons ---
    let mut synapse_index = 0;

    // 1. Input -> Hidden
    for i in input_start..input_end {
        for j in hidden_start..hidden_end {
            synapses.push(Box::new(ChemicalSynapse::new(i, j, 0.0))); // Use 0.0 for random init
            neurons[i].exiting_synapses.push(synapse_index);
            neurons[j].entering_synapses.push(synapse_index);
            synapse_index += 1;
        }
    }

    // 2. Hidden -> Output
    for i in hidden_start..hidden_end {
        for j in output_start..output_end {
            synapses.push(Box::new(ChemicalSynapse::new(i, j, 0.0))); // Use 0.0 for random init
            neurons[i].exiting_synapses.push(synapse_index);
            neurons[j].entering_synapses.push(synapse_index);
            synapse_index += 1;
        }
    }
    
    // 3. Inhibitory connections between output neurons
    for i in output_start..output_end {
        for j in output_start..output_end {
            if i == j { continue; }
            // Negative weight for inhibition
            synapses.push(Box::new(ChemicalSynapse::new(i, j, -1.0)));
            neurons[i].exiting_synapses.push(synapse_index);
            neurons[j].entering_synapses.push(synapse_index);
            synapse_index += 1;
        }
    }


    // Set up neuron indices
    for i in input_start..input_end {
        input_neurons.push(i);
        neurons[i].layer = 0; // Input layer
    }
    for i in hidden_start..hidden_end {
        hidden_neurons.push(i);
        neurons[i].layer = 1; // Hidden layer
    }
    for i in output_start..output_end {
        output_neurons.push(i);
        neurons[i].layer = 2; // Output layer
    }

    let network = Network::new(
        neurons,
        synapses,
        input_neurons.clone(),
        output_neurons.clone(),
    );

    println!("\n--- Network Architecture (HaSiST) ---");
    println!(
        "Input neurons: {} (indices {}-{})",
        NUM_INPUT_NEURONS,
        input_start,
        input_end - 1
    );
    println!(
        "Hidden neurons: {} (indices {}-{})",
        NUM_HIDDEN_NEURONS,
        hidden_start,
        hidden_end - 1
    );
    println!(
        "Output neurons: {} (indices {}-{})",
        NUM_OUTPUT_NEURONS,
        output_start,
        output_end - 1
    );

    (
        network,
        input_neurons,
        output_neurons,
        hidden_neurons,
        Vec::new(), // No reward neurons in HaSiST
        Vec::new(), // No pain neurons in HaSiST
    )
}
