use crate::constants::{MAX_NEURON_THRESHOLD, MIN_NEURON_THRESHOLD};
use crate::neuron::{Neuron, NeuronBehavior};
use crate::synapse::{ChemicalSynapse, Synapse};
use crate::utils::get_clamped_normal;
use plotters::prelude::*;
use rand::Rng;
use std::sync::{Arc, RwLock};

#[derive(Clone)]
pub struct Network {
    pub neurons: Vec<Arc<RwLock<Neuron>>>,
    pub synapses: Vec<Arc<RwLock<ChemicalSynapse>>>,
}

impl Network {
    pub fn new(
        neurons: Vec<Arc<RwLock<Neuron>>>,
        synapses: Vec<Arc<RwLock<ChemicalSynapse>>>,
    ) -> Self {
        Network { neurons, synapses }
    }

    pub fn create_dense(num_neurons: usize, rng: &mut impl Rng) -> Self {
        let mut neurons = Vec::with_capacity(num_neurons);
        for i in 0..num_neurons {
            neurons.push(Arc::new(RwLock::new(Neuron::new(
                get_clamped_normal(MIN_NEURON_THRESHOLD, MAX_NEURON_THRESHOLD, rng),
                i,
            ))));
        }

        let mut synapses = Vec::new();
        for pre in 0..neurons.len() {
            for post in 0..neurons.len() {
                if pre != post {
                    let synapse = Arc::new(RwLock::new(ChemicalSynapse::new(
                        neurons.get(pre).unwrap().clone(),
                        neurons.get(post).unwrap().clone(),
                        rng,
                    )));
                    neurons
                        .get(pre)
                        .unwrap()
                        .write()
                        .unwrap()
                        .exiting_synapses
                        .push(synapse.clone());
                    neurons
                        .get(post)
                        .unwrap()
                        .write()
                        .unwrap()
                        .entering_synapses
                        .push(synapse.clone());
                    synapses.push(synapse);
                }
            }
        }

        Network { neurons, synapses }
    }

    pub fn reset_state(&mut self) {
        for neuron in &self.neurons {
            let mut neuron = neuron.write().unwrap();
            neuron.full_reset();
        }
    }
}

pub trait VisualizeNetwork {
    fn describe(&self);
    fn plot_synapse_weights(&self, path: &str) -> Result<(), Box<dyn std::error::Error>>;
}

impl VisualizeNetwork for Network {
    fn describe(&self) {
        println!("--- Network State ---");
        println!("Total neurons: {}", self.neurons.len());
        println!("Total synapses: {}", self.synapses.len());

        let mut min_w = f32::MAX;
        let mut max_w = f32::MIN;
        let mut avg_w = 0.0;
        for _s in &self.synapses {
            let s = _s.read().unwrap();
            let w = s.get_weight();
            if w < min_w {
                min_w = w;
            }
            if w > max_w {
                max_w = w;
            }
            avg_w += w;
        }
        avg_w /= self.synapses.len() as f32;

        println!(
            "Synapse weights: min={:.4}, max={:.4}, avg={:.4}",
            min_w, max_w, avg_w
        );

        // If it has less than 10 neurons create a connectivity matrix
        if self.neurons.len() <= 10 {
            println!("Connectivity Matrix:");
            let n = self.neurons.len();
            let mut matrix = vec![vec![0.0; n]; n];
            for synapse in &self.synapses {
                let s = synapse.read().unwrap();
                let pre_id = s.get_presynaptic_neuron().read().unwrap().id;
                let post_id = s.get_postsynaptic_neuron().read().unwrap().id;
                matrix[pre_id][post_id] = s.weight;
            }
            // Add headers
            println!("Columns are Post-synaptic Neurons, Rows are Pre-synaptic Neurons");
            for i in 0..n {
                print!("{:>5} ", i);
            }
            println!();
            for row in matrix {
                for val in row {
                    print!("{:>5.2} ", val);
                }
                println!();
            }
        }
    }

    fn plot_synapse_weights(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        let weights: Vec<f32> = self
            .synapses
            .iter()
            .map(|s| s.read().unwrap().get_weight())
            .collect();
        if weights.is_empty() {
            return Ok(());
        }
        let max_weight = weights.iter().cloned().fold(f32::MIN, f32::max);
        let min_weight = weights.iter().cloned().fold(f32::MAX, f32::min);

        let n_bins = 20;
        let mut bins = vec![0; n_bins];
        let bucket_size = (max_weight - min_weight) / n_bins as f32;

        if bucket_size <= 0.0 {
            // Handle case where all weights are the same
            let mut chart = ChartBuilder::on(&root)
                .caption(
                    "Synapse Weight Distribution (All weights are equal)",
                    ("sans-serif", 50).into_font(),
                )
                .build_cartesian_2d(min_weight..max_weight, 0..weights.len() as u32)?;
            chart.configure_mesh().draw()?;
            return Ok(());
        }

        for &weight in &weights {
            let mut bin_index = ((weight - min_weight) / bucket_size).floor() as usize;
            if bin_index >= n_bins {
                bin_index = n_bins - 1;
            }
            bins[bin_index] += 1;
        }

        let max_count = *bins.iter().max().unwrap_or(&0) as u32;

        let mut chart = ChartBuilder::on(&root)
            .caption(
                "Synapse Weight Distribution",
                ("sans-serif", 50).into_font(),
            )
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(min_weight..max_weight, 0..max_count)?;

        chart.configure_mesh().draw()?;

        chart.draw_series((0..n_bins).map(|i| {
            let x0 = min_weight + i as f32 * bucket_size;
            let x1 = min_weight + (i + 1) as f32 * bucket_size;
            let y = bins[i] as u32;
            let mut bar = Rectangle::new([(x0, 0), (x1, y)], RED.mix(0.5).filled());
            bar.set_margin(0, 0, 5, 5);
            bar
        }))?;

        root.present()?;
        Ok(())
    }
}
