// main.rs — Minimal Leaky-Integrate-and-Fire (LIF) SNN in pure Rust
// Run as a single-binary Cargo project. See chat for Cargo.toml.

use rand::prelude::*;
use std::collections::VecDeque;

#[derive(Clone, Debug)]
struct Neuron {
    v: f32,          // membrane potential
    v_rest: f32,     // resting potential
    v_th: f32,       // spiking threshold
    v_reset: f32,    // reset potential
    tau_m: f32,      // membrane time constant (ms)
    refrac_ms: u32,  // absolute refractory (in timesteps)
    refrac_left: u32,
    i_syn: f32,      // accumulated synaptic current this step
}

impl Neuron {
    fn lif(v_rest: f32, v_th: f32, v_reset: f32, tau_m: f32, refrac_ms: u32) -> Self {
        Self {
            v: v_rest,
            v_rest,
            v_th,
            v_reset,
            tau_m,
            refrac_ms,
            refrac_left: 0,
            i_syn: 0.0,
        }
    }

    /// Advance neuron by one time step of size dt (ms) with an external current i_ext.
    /// Returns true if the neuron spikes in this step.
    fn step(&mut self, dt: f32, i_ext: f32) -> bool {
        let spiked: bool;
        if self.refrac_left > 0 {
            self.refrac_left -= 1;
            self.v = self.v_reset;
            spiked = false;
        } else {
            // Simple LIF Euler integration: dv/dt = (-(v - v_rest) + I) / tau_m
            let total_i = self.i_syn + i_ext;
            let dv = (-(self.v - self.v_rest) + total_i) * (dt / self.tau_m);
            self.v += dv;
            if self.v >= self.v_th {
                spiked = true;
                self.v = self.v_reset;
                self.refrac_left = self.refrac_ms;
            } else {
                spiked = false;
            }
        }
        // Clear synaptic current accumulator after integration
        self.i_syn = 0.0;
        spiked
    }
}

#[derive(Clone, Debug)]
struct Synapse {
    pre: usize,
    post: usize,
    weight: f32,        // synaptic efficacy (current units)
    delay_steps: usize, // integer delay in time steps
    buffer: VecDeque<f32>, // circular buffer implementing axonal delay
}

impl Synapse {
    fn new(pre: usize, post: usize, weight: f32, delay_steps: usize) -> Self {
        let mut buffer = VecDeque::with_capacity(delay_steps.max(1));
        for _ in 0..delay_steps { buffer.push_back(0.0); }
        Self { pre, post, weight, delay_steps, buffer }
    }

    /// Propagate one tick: push spike effect if pre spiked, pop front to deliver to post.
    fn tick(&mut self, pre_spiked: bool) -> f32 {
        if pre_spiked {
            // push at the back to arrive after delay_steps pops
            self.buffer.push_back(self.weight);
        } else {
            self.buffer.push_back(0.0);
        }
        // deliver what's due this step
        self.buffer.pop_front().unwrap_or(0.0)
    }
}

struct Network {
    neurons: Vec<Neuron>,
    synapses: Vec<Synapse>,
}

impl Network {
    fn new(neurons: Vec<Neuron>, synapses: Vec<Synapse>) -> Self { Self { neurons, synapses } }

    /// Step the entire network once. `i_ext` is per-neuron external current.
    /// Returns a Vec<bool> marking spikes per neuron for this step.
    fn step(&mut self, dt: f32, i_ext: &[f32]) -> Vec<bool> {
        assert_eq!(self.neurons.len(), i_ext.len());

        // 1) Integrate neurons with their current accumulators (syn inputs added later this step)
        let mut spiked = vec![false; self.neurons.len()];
        for (n, i) in self.neurons.iter_mut().zip(i_ext.iter().copied()) {
            let s = n.step(dt, i);
            spiked.push(s);
        }
        // The above pushed extra values; fix length properly.
        spiked.drain(0..self.neurons.len());

        // 2) Propagate synapses based on spikes that occurred *this* step
        //    (discrete-time; you can choose pre- or post-integration ordering).
        // Here we use: spike this step -> queued -> delivered after delay.
        for syn in &mut self.synapses {
            let deliver = syn.tick(spiked[syn.pre]);
            self.neurons[syn.post].i_syn += deliver;
        }

        spiked
    }
}

/// Simple Poisson spike generator driving a set of input channels as external current pulses.
struct PoissonInputs {
    rates_hz: Vec<f32>, // firing rates per input (Hz)
    amp: f32,           // current amplitude per spike
    dt_ms: f32,
    rng: StdRng,
}

impl PoissonInputs {
    fn new(rates_hz: Vec<f32>, amp: f32, dt_ms: f32, seed: u64) -> Self {
        Self { rates_hz, amp, dt_ms, rng: StdRng::seed_from_u64(seed) }
    }
    fn sample_currents(&mut self) -> Vec<f32> {
        let dt_s = self.dt_ms / 1000.0;
        self.rates_hz
            .iter()
            .map(|&r| {
                let p = r * dt_s; // Bernoulli approx to Poisson per bin
                if self.rng.gen::<f32>() < p { self.amp } else { 0.0 }
            })
            .collect()
    }
}

fn main() {
    // --- Config ---
    let dt_ms = 1.0; // 1 ms timestep
    let t_steps = 1000; // simulate 1 second

    let n_in = 10;
    let n_out = 5;

    // Build neurons: inputs + outputs (all LIF for simplicity)
    let mut neurons = Vec::new();
    for _ in 0..(n_in + n_out) {
        neurons.push(Neuron::lif(
            /*v_rest*/ 0.0,
            /*v_th*/ 1.0,
            /*v_reset*/ 0.0,
            /*tau_m*/ 20.0,
            /*refrac_ms*/ 3,
        ));
    }

    // Random feedforward synapses from inputs -> outputs
    let mut rng = StdRng::seed_from_u64(42);
    let mut synapses = Vec::new();
    for i in 0..n_in {
        for o in 0..n_out {
            let w = rng.gen_range(0.05..0.25);
            let d = rng.gen_range(1..5); // 1..4 ms delay
            synapses.push(Synapse::new(i, n_in + o, w, d));
        }
    }

    let mut net = Network::new(neurons, synapses);

    // Poisson drive for inputs only; outputs get 0 external current.
    let mut drive = PoissonInputs::new(vec![15.0; n_in], /*amp*/ 20.0, dt_ms, 123);

    // Buffers for logging spikes
    let total_neurons = n_in + n_out;
    let mut spikes_over_time: Vec<Vec<bool>> = Vec::with_capacity(t_steps);

    for _t in 0..t_steps {
        let mut i_ext = vec![0.0; total_neurons];
        let input_pulses = drive.sample_currents();
        for (k, val) in input_pulses.iter().enumerate() { i_ext[k] = *val; }

        let spiked = net.step(dt_ms, &i_ext);
        spikes_over_time.push(spiked);
    }

    // Print a simple ASCII raster plot: rows=neurons, cols=time (downsampled)
    let ds = 2usize; // downsample columns for compact print
    println!("\nASCII Raster ('.'=no spike, '*'=spike). Inputs: 0..{} Outputs: {}..{}\n",
             n_in-1, n_in, total_neurons-1);
    for n in 0..total_neurons {
        print!("n{:02} ", n);
        for t in (0..t_steps).step_by(ds) {
            let c = if spikes_over_time[t][n] { '*' } else { '.' };
            print!("{}", c);
        }
        println!();
    }

    // Also print total spikes per neuron
    println!("\nSpike counts per neuron:");
    for n in 0..total_neurons {
        let count: usize = spikes_over_time.iter().map(|s| s[n] as usize).sum();
        println!("n{:02}: {}", n, count);
    }
}