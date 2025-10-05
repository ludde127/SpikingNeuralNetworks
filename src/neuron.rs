use crate::constants::*;
use rand_distr::{Distribution, Normal};

#[derive(Clone, Debug)]
pub struct Neuron {
    // Synapses are stored elsewhere
    pub resting_potential: f64,
    pub membrane_potential: f64,
    pub threshold: f64,
    pub membrane_time_constant: f64,
    pub last_spike_time: f64, // in ms
    pub absolute_refractory_time: f64,
    pub exiting_synapses: Vec<usize>, // Indexes of the synapses in the main synapse array (these are the outgoing, may however be bidirectional)
    pub entering_synapses: Vec<usize>, // Indexes of the synapses in the main synapse array (these are the incoming, may however be bidirectional)

    pub relative_refractory_duration: f64, // Time it is hyperpolarized after spiking but not fully blocked
    pub hyperpolarization_depth: f64, // The most negative voltage reached over the membrane after an action potential, around -75 to -80 mV
    pub hyperpolarization_time_constant: f64, // How quickly the neurons membrane returns to resting state

    pub last_accessed_time: f64, // The time these values where valid, used to correctly apply decay
}

impl Neuron {
    /// Constructor to create a new neuron with properties randomized around the mean.
    pub fn new(current_time: f64) -> Self {
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
            entering_synapses: Vec::new(),
            exiting_synapses: Vec::new(),
            relative_refractory_duration: 5.0, // Example value
            hyperpolarization_depth: MEAN_HYPERPOLARIZATION_DEPTH
                * (1.0 + potential_dist.sample(&mut rng)),
            hyperpolarization_time_constant: MEAN_HYPERPOLARIZATION_TIME_CONSTANT
                * (1.0 + time_dist.sample(&mut rng)),
            last_accessed_time: current_time,
        }
    }

    pub fn current_threshold(&self, time: f64) -> f64 {
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

