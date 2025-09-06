use rand::prelude::*;
use std::collections::VecDeque;

// Axons connect neurons
//
//

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
*/

const MINIMUM_CHEMICAL_SYNAPSE_WEIGHT: f64 = 0.001;
const MAXIMUM_CHEMICAL_SYNAPSE_WEIGHT: f64 = 0.999;
const LONG_TERM_POTENTIATION_TIME_WINDOW: f64 = 20.0;
const LONG_TERM_DEPRESSION_TIME_WINDOW: f64 = 20.0;
const SYNAPSE_LTP_DECAY: f64 = 10.0;
const SYNAPSE_LTD_DECAY: f64 = 10.0;

fn modified_sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-(x-0.5).exp()))
}

#[derive(Clone, Debug)]
struct Neuron {

}

impl Neuron {}

#[derive(Clone, Debug)]
struct ElectricalSynapse {
    // This synapse is unidirectional from source_neuron to target_neuron

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
}

#[derive(Clone, Debug)]
struct ChemicalSynapse {
    // This synapse is bidirectional and plastic. And learns its weight using Spike-Timing-Dependent Plasticity (STDP)
    source_neuron: usize,
    target_neuron: usize,

    weight: f64,  // This weight is learned
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
        else if delta_t < 0.0 && delta_t >= -LONG_TERM_DEPRESSION_TIME_WINDOW { // 20ms window for LTD
            let delta_w = self.plasticity * (-(-delta_t) / SYNAPSE_LTD_DECAY).exp(); // Exponential decay
            self.weight -= delta_w;
        }

        // Clamp the weight to a valid range to prevent it from growing indefinitely
        self.weight = modified_sigmoid(self.weight);
    }
}

impl Synapse for ElectricalSynapse {
    fn update_weight(&mut self, pre_spike_time: f64, post_spike_time: f64) {
        // Do nothing, electrical synapses are not plastic
    }
}

fn main() {
    println!("Basic Neuromorphic network.");
}