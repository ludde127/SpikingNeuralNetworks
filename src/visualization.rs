use crate::network::Network;
use graphviz_rust::{exec_dot, parse};
use std::fs::File;
use std::io::Write;

impl Network {
    /// Visualize network using Graphviz.
    pub fn visualize(&self, path: &str) -> Result<(), std::io::Error> {
        let mut dot_string = "digraph network {\n  rankdir=LR;\n  splines=line;\n".to_string();

        // Add nodes
        for neuron in &self.neurons {
            let color = match neuron.layer {
                0 => "blue",
                1 => "orange",
                2 => "green",
                _ => "black",
            };
            dot_string.push_str(&format!(
                "  N{} [color={}, style=filled];\n",
                neuron.id, color
            ));
        }

        // Add edges
        for synapse in &self.synapses {
            let weight = synapse.get_weight();
            if weight > 0.0 {
                let color = if weight < 0.33 {
                    "gray"
                } else if weight < 0.66 {
                    "black"
                } else {
                    "red"
                };
                dot_string.push_str(&format!(
                    "  N{} -> N{} [label=\"{:.2}\", color={}];\n",
                    synapse.get_presynaptic_neuron(),
                    synapse.get_postsynaptic_neuron(),
                    weight,
                    color
                ));
            }
        }

        dot_string.push_str("}");

        exec_dot(
            dot_string,
            vec![
                graphviz_rust::cmd::Format::Png.into(),
                graphviz_rust::cmd::CommandArg::Output(path.to_string()),
            ],
        )?;
        Ok(())
    }
}
