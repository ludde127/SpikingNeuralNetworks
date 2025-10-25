use crate::constants::{LEARNING_RATE, REWARD_AVERAGE_DURATION};
use crate::datastructures::rolling_mean::RollingMeanF32;
use crate::neuron::NeuronBehavior;
use crate::synapse::{ChemicalSynapse, Synapse};
use std::sync::{Arc, RwLock};
use crate::spike_event::SpikeEvent;

pub struct RewardSystem {
    learning_rate: f32,
    average_reward: RollingMeanF32,
    last_reward: f32,
}

impl RewardSystem {
    pub(crate) fn new() -> Self {
        RewardSystem {
            learning_rate: LEARNING_RATE,
            average_reward: RollingMeanF32::new(REWARD_AVERAGE_DURATION), // This is global so we use a slower mean function than in synapses.
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

        for spike_event in spike_events {
            // a_i(t) - a_i_avg(t)
            let synapse = spike_event.read().unwrap().clone().synapse;
            let post_synaptic_neuron_deviation =
                {
                    let read = synapse.read().unwrap();
                    let post_synaptic_neuron = read.postsynaptic_neuron.read().unwrap();
                    post_synaptic_neuron.last_spike_magnitude()
                        - post_synaptic_neuron.ema_spike_average(time)
                };
            let delta = self.learning_rate * post_synaptic_neuron_deviation * delta_reward;
            synapse.write().unwrap().weight += delta;

            // println debugging info
            println!("delta {}, deviation {}, delta reward {}, last reward {}, average reward {}", delta, post_synaptic_neuron_deviation, delta_reward, self.last_reward, average_reward);
        }
        self.last_reward = 0.0;
    }
}
