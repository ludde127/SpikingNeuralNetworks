use crate::constants::*;
use rand::Rng;

pub trait Synapse {
    /// Applies the STDP learning rule to update the synapse weight.
    /// `pre_spike_time` is the time the source neuron fired.
    /// `post_spike_time` is the time the target neuron fired.
    fn update_weight(&mut self, pre_spike_time: f64, post_spike_time: f64);

    /// Applies reward-modulated STDP learning rule
    /// `modulation_factor` is the reward (+) or punishment (-) signal
    fn update_weight_modulated(&mut self, pre_spike_time: f64, post_spike_time: f64, modulation_factor: f64);

    /// Dopamine-modulated STDP
    fn update_weight_dopamine(&mut self, pre_spike_time: f64, post_spike_time: f64);

    fn new(source_neuron: usize, target_neuron: usize) -> Self;

    fn get_source(&self) -> usize;
    fn get_target(&self) -> usize;
}

#[derive(Clone, Debug)]
pub struct ChemicalSynapse {
    // This synapse is unidirectional and plastic. And learns its weight using Spike-Timing-Dependent Plasticity (STDP)
    pub source_neuron: usize,
    pub target_neuron: usize,

    pub weight: f64,     // This weight is learned
    pub plasticity: f64, // This is a factor which is similar to learning rate. It is updated based on how far the weight is from the max (or min)
    pub base_plasticity: f64, // Base learning rate
    pub dopamine: f64, // Dopamine level for reward-based learning
}

impl Synapse for ChemicalSynapse {
    /// Applies the STDP learning rule to update the synapse weight.
    /// `pre_spike_time` is the time the source neuron fired.
    /// `post_spike_time` is the time the target neuron fired.
    fn update_weight(&mut self, pre_spike_time: f64, post_spike_time: f64) {
        let mut delta_t = post_spike_time - pre_spike_time;
        let delta_w;
        // Long-Term Potentiation (LTP): Pre-synaptic spike before post-synaptic spike
        if delta_t > 0.0 {
            delta_w = self.plasticity * (-delta_t / SYNAPSE_LTP_DECAY).exp(); // Exponential decay
        }
        // Long-Term Depression (LTD): Post-synaptic spike before pre-synaptic spike
        else if delta_t < 0.0 {
            // 20ms window for LTD
            delta_t = delta_t.clamp(-SYNAPSE_LTP_DECAY, 0.0);
            delta_w = -self.plasticity * (-(-delta_t) / SYNAPSE_LTD_DECAY).exp();
        // Exponential decay
        } else {
            // This happens if simultaneous firing or if the prespike neuron have never fired
            return;
        }
        self.weight += delta_w;
        self.weight = self.weight.clamp(
            MINIMUM_CHEMICAL_SYNAPSE_WEIGHT,
            MAXIMUM_CHEMICAL_SYNAPSE_WEIGHT,
        );

        /*println!(
            "STDP: pre={} post={} Δt={:.2} Δw={:.4} new_w={:.4}, pre_s={}, post_s={}",
            self.source_neuron, self.target_neuron, delta_t, delta_w, self.weight, pre_spike_time, post_spike_time
        );*/
    }

    /// Reward-modulated STDP: now sets dopamine level
    fn update_weight_modulated(&mut self, _pre_spike_time: f64, _post_spike_time: f64, modulation_factor: f64) {
        self.dopamine = modulation_factor;
    }

    /// Dopamine-modulated STDP
    fn update_weight_dopamine(&mut self, pre_spike_time: f64, post_spike_time: f64) {
        if self.dopamine == 0.0 {
            return; // No learning if no dopamine signal
        }

        let delta_t = post_spike_time - pre_spike_time;

        // Only apply learning for causal events (pre-before-post)
        if delta_t > 0.0 && delta_t < 20.0 { // 20ms window
            let eligibility = (-delta_t / 10.0).exp(); // Eligibility trace
            let delta_w = self.base_plasticity * self.dopamine * eligibility;

            self.weight += delta_w;
            self.weight = self.weight.clamp(
                MINIMUM_CHEMICAL_SYNAPSE_WEIGHT,
                MAXIMUM_CHEMICAL_SYNAPSE_WEIGHT,
            );
        }
        self.dopamine = 0.0; // Reset dopamine after use
    }

    /// Constructor for a new chemical synapse with a random initial weight.
    fn new(source_neuron: usize, target_neuron: usize) -> Self {
        let initial_weight = rand::rng().random_range(0.4..=0.6);
        let base_plasticity = 0.01; // Default base plasticity
        let plasticity = base_plasticity
            * (WEIGHT_RANGE_END_VALUE
                - (WEIGHT_NORMALIZATION_FACTOR * initial_weight - WEIGHT_RANGE_END_VALUE).abs());

        ChemicalSynapse {
            source_neuron,
            target_neuron,
            weight: initial_weight,
            plasticity,
            base_plasticity,
            dopamine: 0.0,
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
pub struct ElectricalSynapse {
    // This synapse is bidirectional from source_neuron to target_neuron
    pub source_neuron: usize,
    pub target_neuron: usize,

    pub weight: f64, // This weight is constant for the synapse.
}

impl Synapse for ElectricalSynapse {
    fn update_weight(&mut self, _pre_spike_time: f64, _post_spike_time: f64) {
        // Do nothing, electrical synapses are not plastic
    }

    fn update_weight_modulated(&mut self, _pre_spike_time: f64, _post_spike_time: f64, _modulation_factor: f64) {
        // Do nothing, electrical synapses are not plastic
    }

    fn update_weight_dopamine(&mut self, _pre_spike_time: f64, _post_spike_time: f64) {
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
