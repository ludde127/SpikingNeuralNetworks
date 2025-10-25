use rand::{thread_rng, Rng};
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
mod utils;
mod reward_system;
mod datastructures;

fn main() {
    println!("Spiking Neural Network Simulation");
    let network = Network::create_dense(100);

    let mut simulation = Simulation::new(1.0, vec![network.neurons[0].clone(), network.neurons[1].clone()]);
    let mut rng = thread_rng();

    network.plot_synapse_weights("synapse_weights_start.png").unwrap();
    // Create random poisson spike trains for input neurons
    for _ in 0..100 {
        {
            network.neurons[0].write().unwrap().receive(
                1.0,
                simulation.time,
            );
        }

        simulation.step();
        simulation.reward(rng.gen_range(-1.0..1.0));
        
    }
    network.plot_synapse_weights("synapse_weights_end.png").unwrap();
}
