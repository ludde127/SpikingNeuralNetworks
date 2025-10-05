use crate::constants::*;
use crate::network::Network;
use petgraph::dot::{Config, Dot};
use petgraph::graph::DiGraph;
use plotters::prelude::*;
use std::fs::File;
use std::io::Write;

impl Network {
    /// Visualize network as a directed graph with weights on edges.
    pub fn visualize_graph(&self, filename: &str) {
        let mut g = DiGraph::<usize, (f64, String)>::new();
        let mut node_indices = Vec::with_capacity(self.neurons.len());

        // Add neurons
        for i in 0..self.neurons.len() {
            node_indices.push(g.add_node(i));
        }

        // Add synapses with weights > 0
        for s in &self.synapses {
            if s.weight > 0.0 {
                let color = if s.weight < 0.33 {
                    "blue"
                } else if s.weight < 0.66 {
                    "green"
                } else {
                    "red"
                };
                g.add_edge(
                    node_indices[s.source_neuron],
                    node_indices[s.target_neuron],
                    (s.weight, color.to_string()),
                );
            }
        }

        // Export DOT with labels + colors
        let dot = Dot::with_attr_getters(
            &g,
            &[Config::EdgeNoLabel],
            &|_, e| {
                let (w, color) = e.weight();
                format!(
                    "label=\"{:.2}\" color={} fontcolor={} penwidth={}",
                    w,
                    color,
                    color,
                    1.0 + 4.0 * w, // thickness based on weight
                )
            },
            &|_, n| format!("label=\"N{}\"", n.0.index()),
        );

        let mut f = File::create(filename).unwrap();
        writeln!(f, "{:?}", dot).unwrap();
        println!("Graph exported to {}", filename);
        println!("Run: dot -Tpng {} -o network.png", filename);
    }

    /// Plot membrane potentials of input and output neurons over time.
    pub fn plot_membrane_potentials(
        &self,
        potentials: &Vec<Vec<f64>>,
        filename: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(filename, (1280, 720)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption(
                "Membrane Potentials of Input & Output Neurons",
                ("sans-serif", 30),
            )
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(0..potentials.len(), -0.1f64..0.1f64)?;

        chart.configure_mesh().draw()?;

        for (i, idx) in self
            .input_neurons
            .iter()
            .chain(self.output_neurons.iter())
            .enumerate()
        {
            let series: Vec<(usize, f64)> = potentials
                .iter()
                .enumerate()
                .map(|(t, v)| (t, v[*idx]))
                .collect();
            chart
                .draw_series(LineSeries::new(series, &Palette99::pick(i)))?
                .label(format!("Neuron {}", idx))
                .legend(move |(x, y)| {
                    PathElement::new(vec![(x, y), (x + 20, y)], Palette99::pick(i))
                });
        }

        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .draw()?;

        Ok(())
    }

    /// Plot a histogram of synapse weights (bucketed barplot).
    pub fn plot_synapse_weights(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        const NUM_BUCKETS: usize = 20;
        let min_weight = MINIMUM_CHEMICAL_SYNAPSE_WEIGHT;
        let max_weight = MAXIMUM_CHEMICAL_SYNAPSE_WEIGHT;
        let bucket_width = (max_weight - min_weight) / NUM_BUCKETS as f64;
        let mut buckets = vec![0usize; NUM_BUCKETS];
        for s in &self.synapses {
            let mut idx = ((s.weight - min_weight) / bucket_width).floor() as usize;
            if idx >= NUM_BUCKETS {
                idx = NUM_BUCKETS - 1;
            }
            buckets[idx] += 1;
        }
        let root = BitMapBackend::new(filename, (1280, 720)).into_drawing_area();
        root.fill(&WHITE)?;
        let max_count = *buckets.iter().max().unwrap_or(&1) as u32;
        let mut chart = ChartBuilder::on(&root)
            .caption("Synapse Weight Distribution", ("sans-serif", 30))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(0..NUM_BUCKETS, 0u32..max_count)?;
        chart
            .configure_mesh()
            .x_desc("Weight Bucket")
            .y_desc("Count")
            .x_labels(NUM_BUCKETS)
            .x_label_formatter(&|idx| {
                let left = min_weight + *idx as f64 * bucket_width;
                let right = left + bucket_width;
                format!("{:.2}-{:.2}", left, right)
            })
            .draw()?;
        chart.draw_series(buckets.iter().enumerate().map(|(i, &count)| {
            let left = i;
            let right = i + 1;
            let color = if (min_weight + i as f64 * bucket_width) < 0.33 {
                BLUE.filled()
            } else if (min_weight + i as f64 * bucket_width) < 0.66 {
                GREEN.filled()
            } else {
                RED.filled()
            };
            Rectangle::new([(left, 0), (right, count as u32)], color)
        }))?;
        Ok(())
    }
}

