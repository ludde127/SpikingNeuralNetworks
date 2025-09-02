use rand::prelude::*;
use std::collections::VecDeque;

#[derive(Clone, Debug)]
struct Neuron {
    membrane_potential: f32,
    resting_potential: f32,
    threshold_voltage: f32,
    reset_voltage: f32,
    membrane_time_constant: f32, // in milliseconds
    refractory_period_steps: u32,
    refractory_countdown: u32,
    synaptic_current: f32,
}

impl Neuron {
    fn new_leaky_integrate_and_fire(
        resting_potential: f32,
        threshold_voltage: f32,
        reset_voltage: f32,
        membrane_time_constant: f32,
        refractory_period_steps: u32,
    ) -> Self {
        Self {
            membrane_potential: resting_potential,
            resting_potential,
            threshold_voltage,
            reset_voltage,
            membrane_time_constant,
            refractory_period_steps,
            refractory_countdown: 0,
            synaptic_current: 0.0,
        }
    }

    /// Advance neuron by one time step of size `dt` (in ms) with an external current `i_ext`.
    /// Returns true if the neuron spikes in this step.
    fn step(&mut self, dt: f32, external_current: f32) -> bool {
        if self.refractory_countdown > 0 {
            self.refractory_countdown -= 1;
            self.membrane_potential = self.reset_voltage;
            return false;
        }

        // Simple LIF Euler integration: dv/dt = (-(v - v_rest) + I) / tau_m
        let total_current = self.synaptic_current + external_current;
        let dv = (-(self.membrane_potential - self.resting_potential) + total_current)
            * (dt / self.membrane_time_constant);
        self.membrane_potential += dv;

        let has_spiked = self.membrane_potential >= self.threshold_voltage;
        if has_spiked {
            self.membrane_potential = self.reset_voltage;
            self.refractory_countdown = self.refractory_period_steps;
        }

        // Clear synaptic current accumulator after integration
        self.synaptic_current = 0.0;
        has_spiked
    }
}

#[derive(Clone, Debug)]
struct Synapse {
    pre_neuron_index: usize,
    post_neuron_index: usize,
    weight: f32,                 // synaptic efficacy (current units)
    delay_steps: usize,          // integer delay in time steps
    spike_buffer: VecDeque<f32>, // circular buffer for axonal delay
}

impl Synapse {
    fn new(
        pre_neuron_index: usize,
        post_neuron_index: usize,
        weight: f32,
        delay_steps: usize,
    ) -> Self {
        let mut spike_buffer = VecDeque::with_capacity(delay_steps.max(1));
        for _ in 0..delay_steps {
            spike_buffer.push_back(0.0);
        }
        Self {
            pre_neuron_index,
            post_neuron_index,
            weight,
            delay_steps,
            spike_buffer,
        }
    }

    /// Propagate one tick: push spike effect if `pre` spiked, pop front to deliver to `post`.
    fn tick(&mut self, pre_spiked: bool) -> f32 {
        let current_to_propagate = if pre_spiked { self.weight } else { 0.0 };
        self.spike_buffer.push_back(current_to_propagate);
        self.spike_buffer.pop_front().unwrap_or(0.0)
    }
}

struct Network {
    neurons: Vec<Neuron>,
    synapses: Vec<Synapse>,
}

impl Network {
    fn new(neurons: Vec<Neuron>, synapses: Vec<Synapse>) -> Self {
        Self { neurons, synapses }
    }

    fn number_of_neurons(&self) -> usize {
        self.neurons.len()
    }

    fn number_of_synapses(&self) -> usize {
        self.synapses.len()
    }

    fn new_evolving_network(num_input_neurons: usize, num_output_neurons: usize) -> Self {
        // Build neurons: inputs + outputs (all LIF for simplicity)
        let mut neurons = Vec::new();
        for _ in 0..(num_input_neurons + num_output_neurons) {
            neurons.push(
                Neuron::new_leaky_integrate_and_fire(
                    0.0, 1.0, 0.0, 20.0, 3
                )
            );
        }


        // Random feedforward synapses from inputs -> outputs
        let mut rng = StdRng::seed_from_u64(42);
        let mut synapses = Vec::new();
        for input_neuron_index in 0..num_input_neurons {
            for output_neuron_index in 0..num_output_neurons {
                let weight = rng.gen_range(0.05..0.25);
                let delay_steps = rng.gen_range(1..5); // 1..4 ms delay
                let post_neuron_index = num_input_neurons + output_neuron_index;
                synapses.push(Synapse::new(
                    input_neuron_index,
                    post_neuron_index,
                    weight,
                    delay_steps,
                ));
            }
        }

        Self { neurons, synapses }

    }

    /// Step the entire network once. `external_currents` is per-neuron external current.
    /// Returns a Vec<bool> marking spikes per neuron for this step.
    fn step(&mut self, dt: f32, external_currents: &[f32]) -> Vec<bool> {
        assert_eq!(self.neurons.len(), external_currents.len());

        // 1) Integrate neurons with their current accumulators
        let mut spiked = self
            .neurons
            .iter_mut()
            .zip(external_currents.iter().copied())
            .map(|(neuron, current)| neuron.step(dt, current))
            .collect::<Vec<bool>>();

        // 2) Propagate synapses based on spikes that occurred *this* step
        for synapse in &mut self.synapses {
            let delivered_current = synapse.tick(spiked[synapse.pre_neuron_index]);
            self.neurons[synapse.post_neuron_index].synaptic_current += delivered_current;
        }

        spiked
    }
}

/// Simple Poisson spike generator driving a set of input channels as external current pulses.
struct PoissonInputs {
    firing_rates_hz: Vec<f32>,
    amplitude: f32,
    dt_ms: f32,
    rng: StdRng,
}

impl PoissonInputs {
    fn new(firing_rates_hz: Vec<f32>, amplitude: f32, dt_ms: f32, seed: u64) -> Self {
        Self {
            firing_rates_hz,
            amplitude,
            dt_ms,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    fn sample_currents(&mut self) -> Vec<f32> {
        let dt_s = self.dt_ms / 1000.0;
        self.firing_rates_hz
            .iter()
            .map(|&rate_hz| {
                let probability = rate_hz * dt_s; // Bernoulli approx to Poisson per bin
                if self.rng.gen::<f32>() < probability {
                    self.amplitude
                } else {
                    0.0
                }
            })
            .collect()
    }
}

fn main() {
    // --- Config ---
    let dt_ms = 1.0; // 1 ms timestep
    let num_timesteps = 1000; // simulate 1 second

    let n_in = 3;
    let n_out = 3;

    let mut network = Network::new_evolving_network(n_in, n_out);

    // Poisson drive for inputs only; outputs get 0 external current.
    let mut input_drive = PoissonInputs::new(
        vec![15.0; n_in],
        /*amplitude*/ 20.0,
        dt_ms,
        123,
    );

    // Buffers for logging spikes
    let total_neurons = network.number_of_neurons();
    let mut spike_history: Vec<Vec<bool>> = Vec::with_capacity(num_timesteps);

    for _t in 0..num_timesteps {
        let mut external_currents = vec![0.0; total_neurons];
        let input_pulses = input_drive.sample_currents();
        for (k, &value) in input_pulses.iter().enumerate() {
            external_currents[k] = value;
        }

        let spiked_this_step = network.step(dt_ms, &external_currents);
        spike_history.push(spiked_this_step);
    }

    // Print a simple ASCII raster plot: rows=neurons, cols=time (downsampled)
    let downsample_factor = 2usize;
    println!(
        "\nASCII Raster ('.'=no spike, '*'=spike). Inputs: 0..{} Outputs: {}..{}\n",
        n_in - 1,
        n_in,
        total_neurons - 1
    );
    for neuron_index in 0..total_neurons {
        print!("n{:02} ", neuron_index);
        for t in (0..num_timesteps).step_by(downsample_factor) {
            let character = if spike_history[t][neuron_index] {
                '*'
            } else {
                '.'
            };
            print!("{}", character);
        }
        println!();
    }

    // Also print total spikes per neuron
    println!("\nSpike counts per neuron:");
    for neuron_index in 0..total_neurons {
        let count: usize = spike_history.iter().map(|s| s[neuron_index] as usize).sum();
        println!("n{:02}: {}", neuron_index, count);
    }
}
