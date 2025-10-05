#[derive(Clone, Debug)]
pub struct Neuron {
    pub id: usize,
    pub membrane_potential: f64,
    pub threshold: f64,
    pub refractory_period: u32,
    pub exiting_synapses: Vec<usize>,
    pub entering_synapses: Vec<usize>,
    pub layer: u8, // 0: input, 1: hidden, 2: output
}

impl Neuron {
    /// Constructor to create a new neuron.
    pub fn new(threshold: f64, id: usize) -> Self {
        Neuron {
            id,
            membrane_potential: 0.0,
            threshold,
            refractory_period: 0,
            exiting_synapses: Vec::new(),
            entering_synapses: Vec::new(),
            layer: 0, // Default to input layer
        }
    }

    pub fn reset(&mut self) {
        self.membrane_potential = 0.0;
        self.refractory_period = 0;
    }
}
