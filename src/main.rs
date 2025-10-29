use rand::{thread_rng, Rng, SeedableRng};
use plotters::prelude::*;
use rand::rngs::StdRng;
// Added for reward plotting
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

/// Plots a 1D vector of f32 data as a line chart.
fn plot_reward_over_time(
    data: &[f32],
    path: &str,
    title: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if data.is_empty() {
        println!("No reward data to plot.");
        return Ok(());
    }

    let root = BitMapBackend::new(path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    // Find min and max reward for y-axis
    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);

    // Add some padding, but cap at reward limits
    let y_max = (max_val + 0.2).min(1.0);
    let y_min = (min_val - 0.2).max(-1.0);

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 40).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0..data.len(), y_min..y_max)?;

    chart.configure_mesh()
        .x_desc("Trial Batch")
        .y_desc("Average Reward")
        .draw()?;

    chart.draw_series(LineSeries::new(
        data.iter().enumerate().map(|(x, y)| (x, *y)),
        &BLUE
    ))?;

    root.present()?;
    println!("Reward plot saved to {}", path);
    Ok(())
}


fn main() {
    println!("Spiking Neural Network Simulation");
    //let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducibility
    let mut rng = thread_rng(); // Non-deterministic seed for variability
    let network = Network::create_dense(100, &mut rng);
    let mut simulation = Simulation::new(1.0, network.neurons.clone());

    network.plot_synapse_weights("synapse_weights_start.png").unwrap();

    const INPUT_NEURON_A_IDX: usize = 0; // "Go" signal
    const INPUT_NEURON_B_IDX: usize = 1; // "No-Go" signal
    const OUTPUT_NEURON_IDX: usize = 2; // Target output
    const REWARD_MAGNITUDE: f32 = 1.0; // Changed to f32

    const NUM_TRIALS: u32 = 20_000;
    const TRIAL_WINDOW_STEPS: u32 = 25;

    println!("Starting simulation... Target: Neuron {} spikes for Input {}, not for Input {}.",
             OUTPUT_NEURON_IDX, INPUT_NEURON_A_IDX, INPUT_NEURON_B_IDX);
    println!("Running {} trials, each with a {}-step response window.", NUM_TRIALS, TRIAL_WINDOW_STEPS);


    let mut total_reward_batch = 0.0f32;
    let mut all_trial_rewards: Vec<f32> = Vec::new();
    let mut batch_avg_rewards: Vec<f32> = Vec::new();
    const BATCH_SIZE: u32 = 50;

    for trial in 0..NUM_TRIALS {
        let (input_idx, is_go_signal) = if rng.gen_bool(0.5) {
            (INPUT_NEURON_A_IDX, true)
        } else {
            (INPUT_NEURON_B_IDX, false)
        };

        // Get spike time *before* the trial starts
        let last_spike_time_before_trial = {
            network.neurons[OUTPUT_NEURON_IDX].read().unwrap().time_of_last_fire()
        };

        // pct_rand should decrease exponentially over trials
        let pct_rand = (-0.3 * (trial as f32)).exp();

        // --- This is the inner trial loop ---
        for _ in 0..TRIAL_WINDOW_STEPS {
            simulation.input_external_stimuli(network.neurons[input_idx].clone(), 1.0);
            if pct_rand > 0.0001 {
                simulation.random_noise(-pct_rand, pct_rand, pct_rand, &mut rng);
            }
            simulation.step();
            // DO NOT apply reward here
        }
        // --- End of inner trial loop ---

        // Now, check the *overall* outcome of the trial
        let output_spiked_during_trial = {
            network.neurons[OUTPUT_NEURON_IDX].read().unwrap().time_of_last_fire() > last_spike_time_before_trial
        };
        //println!("Output spiked during trial: {}", output_spiked_during_trial);

        let reward: f32;
        if is_go_signal {
            if output_spiked_during_trial {
                reward = REWARD_MAGNITUDE * 5.0; // Correct: "Go" and it spiked
            } else {
                reward = -REWARD_MAGNITUDE; // Correct: "Go" and it didn't spike
            }
        } else { // is_no_go_signal
            if output_spiked_during_trial {
                reward = -REWARD_MAGNITUDE; // Correct: "No-Go" and it spiked
            } else {
                reward = REWARD_MAGNITUDE; // Correct: "No-Go" and it didn't spike
            }
        }

        // Apply the *single* reward for the *entire* trial
        simulation.reward(reward);

        // This intra-trial result tracking is fine, but it was just for logging.
        // We'll use the final reward for logging.
        let intra_trial_result = reward;
        all_trial_rewards.push(intra_trial_result);
        total_reward_batch += intra_trial_result;

        if (trial + 1) % BATCH_SIZE == 0 {
            let avg_reward = total_reward_batch / (BATCH_SIZE as f32);
            println!("Trials {}-{}: Average Reward = {:.2}", trial + 1 - BATCH_SIZE, trial + 1, avg_reward);
            batch_avg_rewards.push(avg_reward);
            total_reward_batch = 0.0f32;
        }
    }

    println!("Simulation finished.");

    if let Err(e) = plot_reward_over_time(
        &batch_avg_rewards,
        "reward_over_time.png",
        "Average Reward per Batch",
    ) {
        eprintln!("Error plotting reward: {}", e);
    }
    // ---

    // Calculate and print moving average of rewards
    let moving_avg_window = 100; // This is 100 *trials*, not batches
    if all_trial_rewards.len() > moving_avg_window {
        println!("Moving average of reward (window={}):", moving_avg_window);
        let moving_averages: Vec<f32> = all_trial_rewards // Changed to f32
            .windows(moving_avg_window)
            .map(|w| w.iter().sum::<f32>() / (moving_avg_window as f32)) // Changed to f32
            .collect();

        // Print a subset of moving averages to avoid flooding console
        for (i, avg) in moving_averages.iter().enumerate().step_by(moving_avg_window) {
            println!("  Trials {}-{}: {:.2}", i, i + moving_avg_window, avg);
        }
    }

    network.plot_synapse_weights("synapse_weights_end.png").unwrap();
    network.describe();

    println!("\nCheck synapse_weights_start.png, synapse_weights_end.png, and reward_over_time.png.");
    println!("If learning occurred, you should see a rising trend in the average rewards.");
    println!("The weights plot might show stronger connections from {} -> {}", INPUT_NEURON_A_IDX, OUTPUT_NEURON_IDX);
    println!("and weaker (or inhibitory) connections from {} -> {}", INPUT_NEURON_B_IDX, OUTPUT_NEURON_IDX);
}


