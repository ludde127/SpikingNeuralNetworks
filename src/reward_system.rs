use crate::constants::{
    LEARNING_RATE, MAX_SYNAPSE_WEIGHT, MIN_SYNAPSE_WEIGHT, REWARD_AVERAGE_DURATION,
};
use crate::datastructures::exponential_rolling_mean::EmaMeanF32;
use crate::neuron::NeuronBehavior;
use crate::spike_event::SpikeEvent;
use std::sync::{Arc, RwLock};

pub struct RewardSystem {
    learning_rate: f32,
    average_reward: EmaMeanF32,
    last_reward: f32,
    // Track the delta error (reward prediction error) per learning step
    delta_error_history: Vec<f32>,
}

impl RewardSystem {
    pub(crate) fn new() -> Self {
        RewardSystem {
            learning_rate: LEARNING_RATE,
            average_reward: EmaMeanF32::new(REWARD_AVERAGE_DURATION), // This is global so we use a slower mean function than in synapses.
            last_reward: 0.0,
            delta_error_history: Vec::new(),
        }
    }

    pub(crate) fn add_reward(&mut self, time: f32, reward: f32) {
        self.average_reward.add(time, reward);
        self.last_reward = reward;
    }

    pub(crate) fn update_synapses(
        &mut self,
        time: f32,
        spike_events: &Vec<Arc<RwLock<SpikeEvent>>>,
    ) {
        // We will iterate all the synapses so only send synapses which have been active lately.
        let average_reward = self.average_reward.get_mean(time).unwrap_or(0.0);
        let weighted_average_reward = average_reward * (time / REWARD_AVERAGE_DURATION).min(1.0);

        // If the average is close to 0 we scale up the learning rate to introduce randomness
        // This will be done using the formula: scaled_learning_rate = learning_rate / (|average_reward| + epsilon)
        // where epsilon is a small constant to prevent division by zero.
        let scaled_learning_rate =  {
            if average_reward.abs() < 0.15 && time > REWARD_AVERAGE_DURATION * 2.0 {
                self.learning_rate / (average_reward.abs() + f32::EPSILON).powf(2.0)
            } else if average_reward.abs() > 0.6 {
                // Scale down learning rate exponentially when average reward is high to stabilize learning
                self.learning_rate * (1.0 - average_reward.abs())
            }
            else {
                self.learning_rate
            }
        };


        let delta_reward = self.last_reward - weighted_average_reward;

        // Record delta error for plotting regardless of whether we update.
        self.delta_error_history.push(delta_reward);

        if delta_reward.abs() < 0.01 {
            return;
        }

        let mut delta_sum = 0.0;
        for spike_event in spike_events {
            // a_i(t) - a_i_avg(t)
            let synapse = spike_event.read().unwrap().clone().synapse;
            let event = spike_event.read().unwrap();

            // --- This is the missing part ---
            // 1. Get Presynaptic Activity: x_j(t)
            let presynaptic_activity = event.presynaptic_ema_firing_rate_before_spike;
            let post_synaptic_neuron_deviation = {
                let read = synapse.read().unwrap();
                let post_synaptic_neuron = read.postsynaptic_neuron.read().unwrap();
                let first_spike_after =
                    post_synaptic_neuron.first_spike_after(event.delivery_time - f32::EPSILON);
                if first_spike_after.is_none() {
                    //println!("No spike after time {} for neuron {}", event.delivery_time, post_synaptic_neuron.id);
                    None
                } else {
                    Some(
                        first_spike_after.unwrap().membrane_potential_at_spike
                            - post_synaptic_neuron
                                .ema_activation_average(event.delivery_time - f32::EPSILON),
                    )
                }
            };
            let delta = {
                if post_synaptic_neuron_deviation.is_none() {
                    -scaled_learning_rate * delta_reward // Punish for silent synapse
                } else {
                    scaled_learning_rate
                        * presynaptic_activity
                        * post_synaptic_neuron_deviation.unwrap()
                        * delta_reward
                }
            };
            delta_sum += delta;
            //println!("Delta Reward: {}, Presynaptic Activity: {}, Post Synaptic Deviation: {}, Weight Change: {}", delta_reward, presynaptic_activity, post_synaptic_neuron_deviation, delta);

            let current_synapse_weight = synapse.read().unwrap().weight;
            synapse.write().unwrap().weight =
                (current_synapse_weight + delta).clamp(MIN_SYNAPSE_WEIGHT, MAX_SYNAPSE_WEIGHT);
        }
        println!(
            "Total weight change this reward: {}, And average: {}, num spikes: {}, scaled learning rate: {}",
            delta_sum,
            delta_sum / (spike_events.len() as f32),
            spike_events.len(),
            scaled_learning_rate
        );
        self.last_reward = 0.0;
    }

    // New: expose the current EMA average reward (decayed to the provided time)
    pub fn get_average_reward(&self, current_time: f32) -> Option<f32> {
        self.average_reward.get_mean(current_time)
    }

    // New: expose the delta error history for plotting
    pub fn delta_error_history(&self) -> &[f32] {
        &self.delta_error_history
    }
}
