use crate::neuron::{Neuron, NeuronBehavior};
use crate::synapse::Synapse;
use plotters::prelude::*;

#[derive(Clone)]
pub struct Network {
    pub neurons: Vec<Neuron>,
    pub synapses: Vec<Box<dyn Synapse>>,
}

impl Network {
    pub fn new(
        neurons: Vec<Neuron>,
        synapses: Vec<Box<dyn Synapse>>,
    ) -> Self {
        Network {
            neurons,
            synapses,
        }
    }

    pub fn reset_state(&mut self) {
        for neuron in &mut self.neurons {
            neuron.reset();
        }
    }

    // --- Utility and Visualization ---

    pub fn describe(&self) {
        println!("--- Network State ---");
        println!("Total neurons: {}", self.neurons.len());
        println!("Total synapses: {}", self.synapses.len());

        let mut min_w = f32::MAX;
        let mut max_w = f32::MIN;
        let mut avg_w = 0.0;
        for s in &self.synapses {
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
    }

    pub fn plot_synapse_weights(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        let weights: Vec<f32> = self.synapses.iter().map(|s| s.get_weight()).collect();
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
            .caption("Synapse Weight Distribution", ("sans-serif", 50).into_font())
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
