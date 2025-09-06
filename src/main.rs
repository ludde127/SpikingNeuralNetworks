use rand::Rng;
use rand_distr::{Normal, Distribution};
/*
## Synapse
A synapse is a structure that allows a neuron to signal another neuron, these can be
either electrical or chemical.

A synapse most often connects many-to-one neurons or one-to-many, but it can also be one to one.

Plasticity in synapses is only possible for chemical synapses. Where it works in two ways:
Long-term potentiation (LTP) and Long-term depression (LTD).

LTP: The connection is strengthened (the sensitivity increased) when a presynaptic neuron commonly stimulates a postsynaptic neuron.

LTD: This is the opposite of LTP and the connection is weakened, this happens when a synapse is repeatedly activated at a low frequency.

### Electrical  (trough gap junctions).

The communication is almost instant.

Most often bidirectional, sometimes rectified, primarily transmitting in one direction.

Low capacity for signal modulation, cannot be modified or amplified.

Used for synchronizing the firing of groups of neurons.


### Chemical synapses

Chemical synapses are much slower but more flexible.

Speed: 1-100 ms

Is unidirectional

Chemical synapses has good ways of modulating the signal, to amplify signals whose sensitivity can
be altered. This makes plasticity possible.

Chemical synapses allow summing up (integrating) all the inputs from other neurons for the postsynapic
neuron. Chemical synapses can both send excitatory and inhibitory signals (both negative and positive towards the sum).


### Neuron

Neurons sends an all-or-nothing signal called action potential if they receive a large enough voltage
change over a small timeframe.  They have a voltage gradient across their membranes

A neuron integrates all incoming signals.
At rest, they have a negative charge over its membrane of around -70 millivolt

If the neuron receives signals which makes the voltage change to around -55 millivolt it sends
its action potential down the axon.

If a neuron gets a sum of 20 as input and has threshold as 3 it would send a signal removing
threshold from the sum, and at next refractory time send again removing 3 and on and on.

*/
use std::sync::Arc;

const MINIMUM_CHEMICAL_SYNAPSE_WEIGHT: f64 = 0.001;
const MAXIMUM_CHEMICAL_SYNAPSE_WEIGHT: f64 = 0.999;
const ELECTRICAL_SYNAPSE_WEIGHT: f64 = (MAXIMUM_CHEMICAL_SYNAPSE_WEIGHT - MINIMUM_CHEMICAL_SYNAPSE_WEIGHT) / 2.0;
const LONG_TERM_POTENTIATION_TIME_WINDOW: f64 = 20.0;
const LONG_TERM_DEPRESSION_TIME_WINDOW: f64 = 20.0;
const SYNAPSE_LTP_DECAY: f64 = 10.0;
const SYNAPSE_LTD_DECAY: f64 = 10.0;

const ADAPTIVE_LEARNING_RATE_SCALING_FACTOR: f64 = 0.5;
const WEIGHT_NORMALIZATION_FACTOR: f64 = 2.0;
const WEIGHT_RANGE_END_VALUE: f64 = 1.0;

const MEAN_NEURON_RESTING_POTENTIAL: f64 = -70e-3; // -70 millivolt
const MEAN_NEURON_THRESHOLD: f64 = -55e-3; // -55 millivolt
const MEAN_NEURON_ABSOLUTE_REFRACTORY_TIME: f64 = 1.5; // ms

const MEAN_NEURON_MEMBRANE_TIME_CONSTANT: f64 = 15.0; // ms
const MEAN_HYPERPOLARIZATION_DEPTH: f64 = -77.5e-3; // V
const MEAN_HYPERPOLARIZATION_TIME_CONSTANT: f64 = 3.5; // ms

const SYNAPSE_SPIKE_TIME: f64 = 2.0;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Clone, Debug)]
struct Neuron {
    // Synapses are stored elsewhere
    resting_potential: f64,
    membrane_potential: f64,
    threshold: f64,
    membrane_time_constant: f64,
    last_spike_time: f64, // in ms
    absolute_refractory_time: f64,
    exiting_synapses: Vec<usize>, // Indexes of the synapses in the main synapse array (these are the outgoing, may however be bidirectional)

    relative_refractory_duration: f64, // Time it is hyperpolarized after spiking but not fully blocked
    hyperpolarization_depth: f64, // The most negative voltage reached over the membrane after an action potential, around -75 to -80 mV
    hyperpolarization_time_constant: f64, // How quickly the neurons membrane returns to resting state

    last_accessed_time: f64, // The time these values where valid, used to correctly apply decay
}

impl Neuron {
    /// Constructor to create a new neuron with properties randomized around the mean.
    fn new(current_time: f64) -> Self {
        let mut rng = rand::rng();

        // Use a normal distribution for slight variations. Standard deviation is 5% of the mean.
        let potential_dist = Normal::new(0.0, 0.05).unwrap();
        let time_dist = Normal::new(0.0, 0.05).unwrap();

        Neuron {
            resting_potential: MEAN_NEURON_RESTING_POTENTIAL * (1.0 + potential_dist.sample(&mut rng)),
            membrane_potential: MEAN_NEURON_RESTING_POTENTIAL, // Start at rest
            threshold: MEAN_NEURON_THRESHOLD * (1.0 + potential_dist.sample(&mut rng)),
            membrane_time_constant: MEAN_NEURON_MEMBRANE_TIME_CONSTANT * (1.0 + time_dist.sample(&mut rng)),
            last_spike_time: -f64::INFINITY, // Initialize to never have spiked
            absolute_refractory_time: MEAN_NEURON_ABSOLUTE_REFRACTORY_TIME * (1.0 + time_dist.sample(&mut rng)),
            exiting_synapses: Vec::new(),
            relative_refractory_duration: 5.0, // Example value
            hyperpolarization_depth: MEAN_HYPERPOLARIZATION_DEPTH * (1.0 + potential_dist.sample(&mut rng)),
            hyperpolarization_time_constant: MEAN_HYPERPOLARIZATION_TIME_CONSTANT * (1.0 + time_dist.sample(&mut rng)),
            last_accessed_time: current_time,
        }
    }

    fn current_threshold(&self, time: f64) -> f64 {
        // If the neuron has had time to refractor but not enough to stop being hyperpolarized
        let time_since_spike = time - self.last_spike_time;

        // Absolute refractory period: cannot fire.
        if time_since_spike < self.absolute_refractory_time {
            return f64::INFINITY;
        }

        // Relative refractory period: threshold is elevated and decays back to normal.
        let time_in_relative_period = time_since_spike - self.absolute_refractory_time;
        if time_in_relative_period < self.relative_refractory_duration {
            let recovery_factor = (-time_in_relative_period / self.hyperpolarization_time_constant).exp();
            let elevated_threshold = self.threshold + self.hyperpolarization_depth * recovery_factor;
            return elevated_threshold;
        }

        // Normal state: threshold is at its base value.
        self.threshold
    }

    /// Simulates the neuron receiving an input and potentially firing an action potential.
    ///
    /// The function updates the neuron's membrane potential based on the incoming signal,
    /// checks if it can fire an action potential (based on threshold and refractory period),
    /// and returns the action potential's voltage if it fires, otherwise returns 0.0.
    ///
    /// # Arguments
    /// * `potential` - The change in membrane potential received from a presynaptic neuron.
    /// * `current_time` - The current time in the simulation.
    ///
    /// # Returns
    /// * `f64` - The action potential's voltage if the neuron fires; otherwise, 0.0.
    pub fn receive(&mut self, potential: f64, current_time: f64) -> f64 {
        // Calculate the time elapsed since the last decay calculation.
        let dt = current_time - self.last_accessed_time;
        // Apply exponential decay to the membrane potential.
        let decay_factor = (-dt / self.membrane_time_constant).exp();
        self.membrane_potential = self.resting_potential + (self.membrane_potential - self.resting_potential) * decay_factor;
        self.last_accessed_time = current_time;

        // Integrate the incoming potential into the membrane potential regardless of the refractory state.
        self.membrane_potential += potential;

        // Check if the neuron is ready to fire. It must be outside the refractory period and
        // its membrane potential must have reached the threshold.
        if current_time - self.last_spike_time >= self.absolute_refractory_time && self.membrane_potential >= self.current_threshold(current_time) {
            // The neuron fires an action potential.
            self.last_spike_time = current_time;

            // Reset the membrane potential to its resting state after firing.
            self.membrane_potential = self.resting_potential;

            // Return a standard action potential value.
            return 1.0;
        }

        // If the neuron did not fire, return 0.0.
        0.0
    }
}

#[derive(Clone, Debug)]
struct ElectricalSynapse {
    // This synapse is bidirectional from source_neuron to target_neuron
    source_neuron: usize,
    target_neuron: usize,

    weight: f64, // This weight is constant for the synapse.
}

trait Synapse {
    /// Applies the STDP learning rule to update the synapse weight.
    /// `pre_spike_time` is the time the source neuron fired.
    /// `post_spike_time` is the time the target neuron fired.
    /// `learning_rate` determines the magnitude of the weight change.
    fn update_weight(&mut self, pre_spike_time: f64, post_spike_time: f64);

    fn new(source_neuron: usize, target_neuron: usize) -> Self;

    fn get_source(&self) -> usize;
    fn get_target(&self) -> usize;
}

#[derive(Clone, Debug)]
struct ChemicalSynapse {
    // This synapse is unidirectional and plastic. And learns its weight using Spike-Timing-Dependent Plasticity (STDP)
    source_neuron: usize,
    target_neuron: usize,

    weight: f64,     // This weight is learned
    plasticity: f64, // This is a factor which is similar to learning rate. It is updated based on how far the weight is from the max (or min)
}

impl Synapse for ChemicalSynapse {
    /// Applies the STDP learning rule to update the synapse weight.
    /// `pre_spike_time` is the time the source neuron fired.
    /// `post_spike_time` is the time the target neuron fired.
    /// `learning_rate` determines the magnitude of the weight change.
    fn update_weight(&mut self, pre_spike_time: f64, post_spike_time: f64) {
        let delta_t = post_spike_time - pre_spike_time;

        // Long-Term Potentiation (LTP): Pre-synaptic spike before post-synaptic spike
        if delta_t > 0.0 && delta_t <= LONG_TERM_POTENTIATION_TIME_WINDOW {
            let delta_w = self.plasticity * (-delta_t / SYNAPSE_LTP_DECAY).exp(); // Exponential decay
            self.weight += delta_w;
        }
        // Long-Term Depression (LTD): Post-synaptic spike before pre-synaptic spike
        else if delta_t < 0.0 && delta_t >= -LONG_TERM_DEPRESSION_TIME_WINDOW {
            // 20ms window for LTD
            let delta_w = self.plasticity * (-(-delta_t) / SYNAPSE_LTD_DECAY).exp(); // Exponential decay
            self.weight -= delta_w;
        }
        self.plasticity = ADAPTIVE_LEARNING_RATE_SCALING_FACTOR
            * (WEIGHT_RANGE_END_VALUE
                - (WEIGHT_NORMALIZATION_FACTOR * self.weight - WEIGHT_RANGE_END_VALUE).abs());
        // Clamp the weight to a valid range to prevent it from growing indefinitely
        self.weight = self.weight.clamp(MINIMUM_CHEMICAL_SYNAPSE_WEIGHT, MAXIMUM_CHEMICAL_SYNAPSE_WEIGHT);
    }

    /// Constructor for a new chemical synapse with a random initial weight.
    fn new(source_neuron: usize, target_neuron: usize) -> Self {
        let initial_weight = rand::rng().random_range(0.4..=0.6);
        let plasticity = ADAPTIVE_LEARNING_RATE_SCALING_FACTOR * (WEIGHT_RANGE_END_VALUE - (WEIGHT_NORMALIZATION_FACTOR * initial_weight - WEIGHT_RANGE_END_VALUE).abs());

        ChemicalSynapse {
            source_neuron,
            target_neuron,
            weight: initial_weight,
            plasticity,
        }
    }
    fn get_source(&self) -> usize { self.source_neuron }
    fn get_target(&self) -> usize { self.target_neuron }
}

impl Synapse for ElectricalSynapse {
    fn update_weight(&mut self, pre_spike_time: f64, post_spike_time: f64) {
        // Do nothing, electrical synapses are not plastic
    }
    fn new(source_neuron: usize, target_neuron: usize) -> Self {
        ElectricalSynapse {
            source_neuron,
            target_neuron,
            weight: ELECTRICAL_SYNAPSE_WEIGHT
        }
    }
    fn get_source(&self) -> usize { self.source_neuron }
    fn get_target(&self) -> usize { self.target_neuron }
}

/// A simple struct to hold the network components.
struct Network {
    neurons: Vec<Neuron>,
    synapses: Vec<ChemicalSynapse>,
}

fn main() {
    println!("--- Starting Neuromorphic Network Test ---");

    // --- 1. Network Setup ---
    const NUM_INPUT_NEURONS: usize = 4;
    const NUM_OUTPUT_NEURONS: usize = 2;
    const TOTAL_NEURONS: usize = NUM_INPUT_NEURONS + NUM_OUTPUT_NEURONS;

    let mut network = Network {
        neurons: Vec::with_capacity(TOTAL_NEURONS),
        synapses: Vec::new(),
    };

    // Create all neurons
    for _ in 0..TOTAL_NEURONS {
        network.neurons.push(Neuron::new(0.0));
    }

    // Connect every input neuron to every output neuron
    let mut synapse_index = 0;
    for i in 0..NUM_INPUT_NEURONS {
        for j in NUM_INPUT_NEURONS..TOTAL_NEURONS {
            network.synapses.push(ChemicalSynapse::new(i, j));
            network.neurons[i].exiting_synapses.push(synapse_index);
            synapse_index += 1;
        }
    }

    println!("Network created with {} input neurons, {} output neurons, and {} synapses.", NUM_INPUT_NEURONS, NUM_OUTPUT_NEURONS, network.synapses.len());
    println!("\n--- Initial Synapse Weights ---");
    for synapse in &network.synapses {
        println!("Synapse ({:_>2} -> {:_>2}): {:.4}", synapse.source_neuron, synapse.target_neuron, synapse.weight);
    }

    // --- 2. Simulation ---
    let time_step = 0.1; // ms
    let simulation_duration = 2000.0; // ms
    let input_spike_potential = 18e-3; // 18mV potential change
    let pattern_presentation_interval = 10.0; // ms

    // Define the pattern to learn: fire input neurons 0 and 2 simultaneously.
    let target_pattern = vec![0, 2];
    println!("\nTraining network to recognize pattern: Input spikes on neurons {:?}", target_pattern);


    let mut current_time = 0.0;
    let mut last_pattern_time = -f64::INFINITY;

    while current_time < simulation_duration {
        // Present the input pattern at regular intervals
        if current_time - last_pattern_time >= pattern_presentation_interval {
            last_pattern_time = current_time;

            // Fire the neurons in the target pattern
            for &input_neuron_idx in &target_pattern {

                // Clone the list of synapse indices to avoid borrowing `network.neurons` in the loop.
                let exiting_synapses = network.neurons[input_neuron_idx].exiting_synapses.clone();
                // Propagate the spike through all outgoing synapses of this input neuron
                for &synapse_idx in &exiting_synapses {
                    let synapse = &network.synapses[synapse_idx];
                    let target_neuron_idx = synapse.target_neuron;

                    let potential = input_spike_potential * synapse.weight;
                    let output_neuron = &mut network.neurons[target_neuron_idx];

                    // Check if the output neuron fires in response
                    if output_neuron.receive(potential, current_time) > 0.0 {
                        let pre_spike_time = network.neurons[synapse.get_source()].last_spike_time;
                        let post_spike_time = network.neurons[synapse.get_target()].last_spike_time;

                        // Update weights for all synapses connected to the neuron that just fired
                        for s in network.synapses.iter_mut().filter(|s| s.target_neuron == target_neuron_idx) {
                            s.update_weight(pre_spike_time, post_spike_time);
                        }
                    }
                }
            }
        }
        current_time += time_step;
    }

    // --- 3. Results ---
    println!("\n--- Final Synapse Weights after {}ms simulation ---", simulation_duration);
    for synapse in &network.synapses {
        println!("Synapse ({:_>2} -> {:_>2}): {:.4}", synapse.source_neuron, synapse.target_neuron, synapse.weight);
    }

    println!("\n--- Analysis ---");
    println!("Observe how weights for synapses from neurons 0 and 2 have increased,");
    println!("while weights from neurons 1 and 3 have likely decreased (due to LTD not being strongly triggered).");
    println!("One of the output neurons should now be specialized to fire in response to the pattern {:?}.", target_pattern);
}
