use rand::Rng;

pub trait Synapse: Send + Sync {
    fn get_presynaptic_neuron(&self) -> usize;
    fn get_postsynaptic_neuron(&self) -> usize;
    fn get_weight(&self) -> f64;
    fn get_delay(&self) -> u32;
    fn update_weight(&mut self, delta_w: f64);
    fn clone_box(&self) -> Box<dyn Synapse>;
}

impl Clone for Box<dyn Synapse> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

#[derive(Clone, Debug)]
pub struct ChemicalSynapse {
    pub presynaptic_neuron: usize,
    pub postsynaptic_neuron: usize,
    pub weight: f64,
    pub delay: u32,
}

impl ChemicalSynapse {
    pub fn new(presynaptic_neuron: usize, postsynaptic_neuron: usize, weight: f64) -> Self {
        let mut rng = rand::thread_rng();
        ChemicalSynapse {
            presynaptic_neuron,
            postsynaptic_neuron,
            // If weight is 1.0 or -1.0, it's likely an initial setup call, not a random one.
            weight: if weight == 1.0 || weight == -1.0 {
                weight
            } else {
                rng.gen_range(-0.5..0.5)
            },
            delay: rng.gen_range(1..=5),
        }
    }
}

impl Synapse for ChemicalSynapse {
    fn get_presynaptic_neuron(&self) -> usize {
        self.presynaptic_neuron
    }

    fn get_postsynaptic_neuron(&self) -> usize {
        self.postsynaptic_neuron
    }

    fn get_weight(&self) -> f64 {
        self.weight
    }

    fn get_delay(&self) -> u32 {
        self.delay
    }

    fn update_weight(&mut self, delta_w: f64) {
        self.weight += delta_w;
    }

    fn clone_box(&self) -> Box<dyn Synapse> {
        Box::new(self.clone())
    }
}
