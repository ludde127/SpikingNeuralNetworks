use crate::network::{Network, VisualizeNetwork};
use crate::neuron::NeuronBehavior;
use crate::simulation::Simulation;

mod constants;
mod data;
mod network;
mod neuron;
mod spike_event;
mod synapse;
mod simulation;

fn main() {
    println!("Spiking Neural Network Simulation");
    let network = Network::create_dense(10);

    let mut simulation = Simulation::new(1.0, vec![network.neurons[0].clone(), network.neurons[1].clone()]);

    // Create random poisson spike trains for input neurons
    for _ in 0..100 {
        {
            network.neurons[0].write().unwrap().receive(
                1.0,
                simulation.time,
            );
        }

        simulation.step();
    }
}
