use crate::network::{Network, VisualizeNetwork};

mod constants;
mod data;
mod network;
mod neuron;
mod spike_event;
mod synapse;

fn main() {
    println!("Spiking Neural Network Simulation");
    let network = Network::create_dense(10);
    network.describe();
    network.plot_synapse_weights("synapse_weights.png").unwrap();
}
