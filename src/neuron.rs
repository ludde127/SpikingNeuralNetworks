
#[derive(Clone, Debug)]
pub struct Neuron {
    pub id: usize,
    pub membrane_potential: f64,
    pub threshold: f64,
    pub refractory_period: f32,
    pub exiting_synapses: Vec<usize>,
    pub entering_synapses: Vec<usize>,
    pub time_since_fire: f32
}

pub trait NeuronBehavior {
    fn new(threshold: f64, id: usize) -> Self;
    fn integrate(&mut self, input_current: f64);
    fn is_firing(&self) -> bool;
    fn reset(&mut self);
    
}

impl NeuronBehavior for Neuron {
    /// Constructor to create a new neuron.
    fn new(threshold: f64, id: usize) -> Self {
        Neuron {
            id,
            membrane_potential: 0.0,
            threshold,
            refractory_period: 0.0,
            time_since_fire: f32::INFINITY,
            exiting_synapses: Vec::new(),
            entering_synapses: Vec::new(),
        }
    }

    fn integrate(&mut self, input_current: f64) {
        if (self.refractory_period <= 0.0) {
            self.membrane_potential += input_current;
        } else {
            self.refractory_period -= 1.0;
        }
    }

    fn is_firing(&self) -> bool {
        self.membrane_potential >= self.threshold
    }

    fn reset(&mut self) {
        self.membrane_potential = 0.0;
        self.refractory_period = 0.0;
    }
}
