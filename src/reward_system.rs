use crate::constants::{LEARNING_RATE, MAX_SYNAPSE_WEIGHT, MIN_SYNAPSE_WEIGHT, REWARD_AVERAGE_DURATION};
use crate::neuron::NeuronBehavior;
use crate::synapse::{Synapse};
use std::sync::{Arc, RwLock};
use crate::datastructures::exponential_rolling_mean::EmaMeanF32;
use crate::spike_event::SpikeEvent;

pub struct RewardSystem {
    learning_rate: f32,
    average_reward: EmaMeanF32,
    last_reward: f32,
}

impl RewardSystem {
    pub(crate) fn new() -> Self {
        RewardSystem {
            learning_rate: LEARNING_RATE,
            average_reward: EmaMeanF32::new(REWARD_AVERAGE_DURATION), // This is global so we use a slower mean function than in synapses.
            last_reward: 0.0,
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
        let delta_reward = self.last_reward - average_reward;
        if delta_reward.abs() < 0.01 {return}

        let mut delta_sum = 0.0;
        for spike_event in spike_events {
            // a_i(t) - a_i_avg(t)
            let synapse = spike_event.read().unwrap().clone().synapse;
            let event = spike_event.read().unwrap();

            // --- This is the missing part ---
            // 1. Get Presynaptic Activity: x_j(t)
            let presynaptic_activity = event.presynaptic_ema_firing_rate_before_spike;
            let post_synaptic_neuron_deviation =
                {
                    let read = synapse.read().unwrap();
                    let post_synaptic_neuron = read.postsynaptic_neuron.read().unwrap();
                    let first_spike_after = post_synaptic_neuron.first_spike_after(event.delivery_time - f32::EPSILON);
                    if first_spike_after.is_none() {
                        continue;
                    }
                    first_spike_after.unwrap().membrane_potential_at_spike - post_synaptic_neuron.ema_activation_average(time)
                };
            let delta = self.learning_rate * presynaptic_activity * post_synaptic_neuron_deviation * delta_reward;
            delta_sum += delta;
            //println!("Delta Reward: {}, Presynaptic Activity: {}, Post Synaptic Deviation: {}, Weight Change: {}", delta_reward, presynaptic_activity, post_synaptic_neuron_deviation, delta);

            let current_synapse_weight = synapse.read().unwrap().weight;
            synapse.write().unwrap().weight = (current_synapse_weight + delta).clamp(MIN_SYNAPSE_WEIGHT, MAX_SYNAPSE_WEIGHT);
        }
        println!("Total weight change this reward: {}, And average: {}, num spikes: {}", delta_sum, delta_sum/(spike_events.len() as f32), spike_events.len());
        self.last_reward = 0.0;
    }
}
