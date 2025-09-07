use rand::Rng;
use rand_distr::{Distribution, Normal};
use plotters::prelude::*;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::dot::{Dot, Config};
use std::fs::File;
use std::io::Write;
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
use rand_distr::num_traits::float::FloatCore;

const MINIMUM_CHEMICAL_SYNAPSE_WEIGHT: f64 = 0.0;
const MAXIMUM_CHEMICAL_SYNAPSE_WEIGHT: f64 = 1.0;
const ELECTRICAL_SYNAPSE_WEIGHT: f64 =
    (MAXIMUM_CHEMICAL_SYNAPSE_WEIGHT - MINIMUM_CHEMICAL_SYNAPSE_WEIGHT) / 2.0;
const LONG_TERM_POTENTIATION_TIME_WINDOW: f64 = 20.0;
const LONG_TERM_DEPRESSION_TIME_WINDOW: f64 = 20.0;
const SYNAPSE_LTP_DECAY: f64 = 10.0;
const SYNAPSE_LTD_DECAY: f64 = 10.0;

const ADAPTIVE_LEARNING_RATE_SCALING_FACTOR: f64 = 0.05;
const WEIGHT_NORMALIZATION_FACTOR: f64 = 2.0;
const WEIGHT_RANGE_END_VALUE: f64 = 1.0;

const MEAN_NEURON_RESTING_POTENTIAL: f64 = 0.0; // -70 millivolt
const MEAN_NEURON_THRESHOLD: f64 = 43e-3; // -55 millivolt
const MEAN_NEURON_ABSOLUTE_REFRACTORY_TIME: f64 = 1.5; // ms

const MEAN_NEURON_MEMBRANE_TIME_CONSTANT: f64 = 15.0; // ms
const MEAN_HYPERPOLARIZATION_DEPTH: f64 = 25e-3; // V
const MEAN_HYPERPOLARIZATION_TIME_CONSTANT: f64 = 3.5; // ms

const SYNAPSE_SPIKE_TIME: f64 = 2.0;
const POSTSYNAPTIC_POTENTIAL_AMPLITUDE: f64 = 2e-3; // 20 millivolt change

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
            resting_potential: MEAN_NEURON_RESTING_POTENTIAL
                * (1.0 + potential_dist.sample(&mut rng)),
            membrane_potential: MEAN_NEURON_RESTING_POTENTIAL, // Start at rest
            threshold: MEAN_NEURON_THRESHOLD * (1.0 + potential_dist.sample(&mut rng)),
            membrane_time_constant: MEAN_NEURON_MEMBRANE_TIME_CONSTANT
                * (1.0 + time_dist.sample(&mut rng)),
            last_spike_time: -1.0, // Initialize to never have spiked
            absolute_refractory_time: MEAN_NEURON_ABSOLUTE_REFRACTORY_TIME
                * (1.0 + time_dist.sample(&mut rng)),
            exiting_synapses: Vec::new(),
            relative_refractory_duration: 5.0, // Example value
            hyperpolarization_depth: MEAN_HYPERPOLARIZATION_DEPTH
                * (1.0 + potential_dist.sample(&mut rng)),
            hyperpolarization_time_constant: MEAN_HYPERPOLARIZATION_TIME_CONSTANT
                * (1.0 + time_dist.sample(&mut rng)),
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
            let recovery_factor =
                (-time_in_relative_period / self.hyperpolarization_time_constant).exp();
            let elevated_threshold =
                self.threshold + self.hyperpolarization_depth * recovery_factor;
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
        self.membrane_potential = self.resting_potential
            + (self.membrane_potential - self.resting_potential) * decay_factor;
        self.last_accessed_time = current_time;

        // Integrate the incoming potential into the membrane potential regardless of the refractory state.
        self.membrane_potential += potential;

        // Check if the neuron is ready to fire. It must be outside the refractory period and
        // its membrane potential must have reached the threshold.
        //println!("mem: {}, threshold: {}", self.membrane_potential, self.current_threshold(current_time));
        if current_time - self.last_spike_time >= self.absolute_refractory_time
            && self.membrane_potential >= self.current_threshold(current_time)
        {
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
        let mut delta_t = post_spike_time - pre_spike_time;
        let mut delta_w = 0.0;
        // Long-Term Potentiation (LTP): Pre-synaptic spike before post-synaptic spike
        if delta_t > 0.0 {
            delta_w = self.plasticity * (-delta_t / SYNAPSE_LTP_DECAY).exp(); // Exponential decay
        }
        // Long-Term Depression (LTD): Post-synaptic spike before pre-synaptic spike
        else if delta_t < 0.0 {
            // 20ms window for LTD
            delta_t = delta_t.clamp(-SYNAPSE_LTP_DECAY, 0.0);
            delta_w = -self.plasticity * (-(-delta_t) / SYNAPSE_LTD_DECAY).exp(); // Exponential decay
        } else {
            // This happens if simultaneous firing or if the prespike neuron have never fired
            return;
        }
        self.weight += delta_w;

        self.plasticity = ADAPTIVE_LEARNING_RATE_SCALING_FACTOR
            * (WEIGHT_RANGE_END_VALUE
                - (WEIGHT_NORMALIZATION_FACTOR * self.weight - WEIGHT_RANGE_END_VALUE).abs());
        // Clamp the weight to a valid range to prevent it from growing indefinitely

        //println!("{}->{}, delta weight {}, delta_t {}", self.source_neuron, self.target_neuron, delta_w, delta_t);
        self.weight = self.weight.clamp(
            MINIMUM_CHEMICAL_SYNAPSE_WEIGHT,
            MAXIMUM_CHEMICAL_SYNAPSE_WEIGHT,
        );

        /*println!(
            "STDP: pre={} post={} Δt={:.2} Δw={:.4} new_w={:.4}, pre_s={}, post_s={}",
            self.source_neuron, self.target_neuron, delta_t, delta_w, self.weight, pre_spike_time, post_spike_time
        );*/
    }

    /// Constructor for a new chemical synapse with a random initial weight.
    fn new(source_neuron: usize, target_neuron: usize) -> Self {
        let initial_weight = rand::rng().random_range(0.4..=0.6);
        let plasticity = ADAPTIVE_LEARNING_RATE_SCALING_FACTOR
            * (WEIGHT_RANGE_END_VALUE
                - (WEIGHT_NORMALIZATION_FACTOR * initial_weight - WEIGHT_RANGE_END_VALUE).abs());

        ChemicalSynapse {
            source_neuron,
            target_neuron,
            weight: initial_weight,
            plasticity,
        }
    }
    fn get_source(&self) -> usize {
        self.source_neuron
    }
    fn get_target(&self) -> usize {
        self.target_neuron
    }
}

impl Synapse for ElectricalSynapse {
    fn update_weight(&mut self, pre_spike_time: f64, post_spike_time: f64) {
        // Do nothing, electrical synapses are not plastic
    }
    fn new(source_neuron: usize, target_neuron: usize) -> Self {
        ElectricalSynapse {
            source_neuron,
            target_neuron,
            weight: ELECTRICAL_SYNAPSE_WEIGHT,
        }
    }
    fn get_source(&self) -> usize {
        self.source_neuron
    }
    fn get_target(&self) -> usize {
        self.target_neuron
    }
}

#[derive(Clone, Debug)]
struct SpikeEvent {
    source_neuron: usize,
    target_neuron: usize,
    synapse_index: usize,
    spike_time: f64,
    arrival_time: f64,
    weight: f64,
}

/// A simple struct to hold the network components.
struct Network {
    neurons: Vec<Neuron>,
    synapses: Vec<ChemicalSynapse>,
    event_queue: Vec<SpikeEvent>,
    current_time: f64,
    input_neurons: Vec<usize>,
    output_neurons: Vec<usize>,
}

impl Network {
    fn new(
        neurons: Vec<Neuron>,
        synapses: Vec<ChemicalSynapse>,
        input_neurons: Vec<usize>,
        output_neurons: Vec<usize>,
    ) -> Self {
        Network {
            neurons,
            synapses,
            event_queue: Vec::new(),
            current_time: 0.0,
            input_neurons,
            output_neurons,
        }
    }

    fn print_synapse_weight(&self) {
        for synapse in &self.synapses {
            println!(
                "Synapse ({:_>2} -> {:_>2}): {:.4}",
                synapse.source_neuron, synapse.target_neuron, synapse.weight
            );
        }
    }

    fn simulate(
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
                if (spike > 0.0) {
                    //println!("Input Neuron {} spiked", input_neuron_idx);
                    // The neuron spiked so we must propagate it
                    let exiting_synapses = self.neurons[input_neuron_idx].exiting_synapses.clone();
                    for &synapse_idx in &exiting_synapses {
                        let synapse = &self.synapses[synapse_idx];
                        if synapse.weight <= MINIMUM_CHEMICAL_SYNAPSE_WEIGHT {
                            continue;
                        }
                        self.event_queue.push(SpikeEvent {
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
        }
        println!(
            "Finished simulation, handled {} spike events",
            spike_event_counter
        );
        potentials
    }

    fn process_events(&mut self, spike_event_counter: &mut usize) {
        let mut i = 0;
        while i < self.event_queue.len() {
            if self.event_queue[i].arrival_time <= self.current_time {
                *spike_event_counter += 1;
                let event = self.event_queue.remove(i);
                let target_idx = event.target_neuron;

                let incoming_syn_indices: Vec<usize> = self
                    .synapses
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, s)| {
                        if s.target_neuron == event.target_neuron {
                            Some(idx)
                        } else {
                            None
                        }
                    })
                    .collect();

                let target = &mut self.neurons[target_idx];
                let mut target_last_spike_time = target.last_spike_time;
                let potential = POSTSYNAPTIC_POTENTIAL_AMPLITUDE * event.weight; // All have the same potential in this simplification
                let action_potential = target.receive(potential, self.current_time);
                let exiting = target.exiting_synapses.clone();

                if action_potential > 0.0 {
                    // --- POST neuron spiked now ---
                    target_last_spike_time = self.current_time;

                    // propagate spikes
                    for out_syn_idx in exiting {
                        let out_syn = &self.synapses[out_syn_idx];
                        if out_syn.weight <= MINIMUM_CHEMICAL_SYNAPSE_WEIGHT {
                            continue;
                        }
                        self.event_queue.push(SpikeEvent {
                            source_neuron: target_idx,
                            target_neuron: out_syn.target_neuron,
                            spike_time: self.current_time,
                            arrival_time: self.current_time + SYNAPSE_SPIKE_TIME,
                            weight: out_syn.weight,
                            synapse_index: out_syn_idx,
                        });
                    }
                }
                for &syn_idx in &incoming_syn_indices {
                    let source_idx = self.synapses[syn_idx].source_neuron;
                    let neuron = &self.neurons[source_idx];
                    let pre_time = if neuron.last_spike_time == self.current_time
                        && source_idx == event.source_neuron
                    {
                        event.spike_time
                    } else {
                        neuron.last_spike_time
                    };

                    if pre_time.is_finite() {
                        self.synapses[syn_idx]
                            .update_weight(pre_time, target_last_spike_time);
                    }
                }
            } else {
                i += 1;
            }
        }
    }

    /// Visualize network as a directed graph with weights on edges.
    fn visualize_graph(&self, filename: &str) {
        let mut g = DiGraph::<usize, (f64, String)>::new();
        let mut node_indices = Vec::with_capacity(self.neurons.len());

        // Add neurons
        for i in 0..self.neurons.len() {
            node_indices.push(g.add_node(i));
        }

        // Add synapses with weights > 0
        for s in &self.synapses {
            if s.weight > 0.0 {
                let color = if s.weight < 0.33 {
                    "blue"
                } else if s.weight < 0.66 {
                    "green"
                } else {
                    "red"
                };
                g.add_edge(
                    node_indices[s.source_neuron],
                    node_indices[s.target_neuron],
                    (s.weight, color.to_string()),
                );
            }
        }

        // Export DOT with labels + colors
        let dot = Dot::with_attr_getters(
            &g,
            &[Config::EdgeNoLabel],
            &|_, e| {
                let (w, color) = e.weight();
                format!(
                    "label=\"{:.2}\" color={} fontcolor={} penwidth={}",
                    w,
                    color,
                    color,
                    1.0 + 4.0 * w, // thickness based on weight
                )
            },
            &|_, n| {
                format!("label=\"N{}\"", n.0.index())
            },
        );

        let mut f = File::create(filename).unwrap();
        writeln!(f, "{:?}", dot).unwrap();
        println!("Graph exported to {}", filename);
        println!("Run: dot -Tpng {} -o network.png", filename);
    }

    /// Plot membrane potentials of input and output neurons over time.
    fn plot_membrane_potentials(
        &self,
        potentials: &Vec<Vec<f64>>,
        filename: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(filename, (1280, 720)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Membrane Potentials of Input & Output Neurons", ("sans-serif", 30))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(0..potentials.len(), -0.1f64..0.1f64)?;

        chart.configure_mesh().draw()?;

        for (i, idx) in self.input_neurons.iter().chain(self.output_neurons.iter()).enumerate() {
            let series: Vec<(usize, f64)> =
                potentials.iter().enumerate().map(|(t, v)| (t, v[*idx])).collect();
            chart
                .draw_series(LineSeries::new(series, &Palette99::pick(i)))?
                .label(format!("Neuron {}", idx))
                .legend(move |(x, y)| {
                    PathElement::new(vec![(x, y), (x + 20, y)], Palette99::pick(i))
                });
        }

        chart.configure_series_labels().border_style(&BLACK).draw()?;

        Ok(())
    }
}

fn main() {
    println!("--- Starting Neuromorphic Network Test ---");

    // --- 1. Network Setup ---
    const NUM_INPUT_NEURONS: usize = 4;
    const NUM_OUTPUT_NEURONS: usize = 2;
    const NUM_HIDDEN_NEURONS: usize = 3;
    const TOTAL_NEURONS: usize = NUM_INPUT_NEURONS + NUM_OUTPUT_NEURONS + NUM_HIDDEN_NEURONS;
    const SYNAPSE_DELAY: f64 = 1.5; // ms delay for chemical synapse

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
            if (i == j) {continue};
            synapses.push(ChemicalSynapse::new(i, j));
            neurons[i].exiting_synapses.push(synapse_index);
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

    println!(
        "Network created with {} input neurons, {} output neurons, and {} synapses.",
        NUM_INPUT_NEURONS,
        NUM_OUTPUT_NEURONS,
        network.synapses.len()
    );

    // --- 2. Simulation ---
    let target_pattern = vec![0, 2]; // neurons that fire together
    println!(
        "\nTraining network to recognize pattern: Input spikes on neurons {:?}",
        target_pattern
    );

    let steps_to_simulate = 1000;

    let mut input_vector = Vec::with_capacity(steps_to_simulate);

    let mut rng = rand::rng();
    let amplitude = 90e-4;
    for i in 0..steps_to_simulate {
        // Every 20 steps, present either pattern A or B
        if i % 5 == 0 {
            if rng.gen_bool(0.5) {
                // Pattern A: neurons 0 and 2 spike
                input_vector.push(vec![amplitude, 0.0, amplitude, 0.0]);
            } else {
                // Pattern B: neurons 1 and 3 spike
                input_vector.push(vec![0.0, amplitude, 0.0, amplitude]);
            }
        } else {
            // Silence otherwise
            input_vector.push(vec![0.0, 0.0, 0.0, 0.0]);
        }
    }
    input_vector.reverse();
    let potentials = network.simulate(steps_to_simulate, 0.01, &mut input_vector);

    // Export network graph
    network.visualize_graph("network.dot");

    // Plot membrane potentials
    network.plot_membrane_potentials(&potentials, "membrane.png").unwrap();

    // --- 3. Results ---
    println!("\n--- Final Synapse Weights after simulation ---",);
    network.print_synapse_weight();
}
