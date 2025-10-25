/// Calculates an Exponential Moving Average using f32 timestamps.
///
/// This structure approximates a rolling mean using O(1) memory.
/// It does not store a vector of data, only the current mean
/// and the timestamp of the last update.
#[derive(Clone, Debug)]
pub struct EmaMeanF32 {
    // The time window (delta_t) to use as the decay constant
    decay_window: f32,
    current_mean: f32,
    last_timestamp: f32,
    is_initialized: bool,
}

impl EmaMeanF32 {
    /// Creates a new EMA calculator with a `decay_window` (delta_t).
    pub fn new(decay_window: f32) -> Self {
        if decay_window <= 0.0 {
            panic!("Decay window must be a positive value");
        }
        EmaMeanF32 {
            decay_window,
            current_mean: 0.0,
            last_timestamp: 0.0,
            is_initialized: false,
        }
    }

    /// Adds a new value at a specific `timestamp` (t).
    pub fn add(&mut self, timestamp: f32, value: f32) {
        if !self.is_initialized {
            // This is the first value.
            self.current_mean = value;
            self.last_timestamp = timestamp;
            self.is_initialized = true;
            return;
        }

        // Ensure time moves forward
        let delta_t = (timestamp - self.last_timestamp).max(0.0);

        // 1. Calculate the decay factor.
        // This is how much weight the *previous* mean gets.
        // `exp(-delta_t / window)`
        let decay_factor = (-delta_t / self.decay_window).exp();

        // 2. The new value's weight is the complement
        let new_val_weight = 1.0 - decay_factor;

        // 3. Calculate the new mean
        self.current_mean = (value * new_val_weight) + (self.current_mean * decay_factor);
        self.last_timestamp = timestamp;
    }

    /// Gets the current mean, optionally decaying it to the `current_time`.
    ///
    /// If no new data has arrived, the mean's influence "decays" over time.
    pub fn get_mean(&self, current_time: f32) -> Option<f32> {
        if !self.is_initialized {
            return None;
        }

        // How long has it been since the last *add*?
        let delta_t = (current_time - self.last_timestamp).max(0.0);

        // If we are asking for a time *before* the last data, just return
        // the last calculated mean.
        if delta_t == 0.0 {
            return Some(self.current_mean);
        }

        // Decay the stored mean to the present time.
        // This represents the "activity level" fading over time.
        let decay_factor = (-delta_t / self.decay_window).exp();

        Some(self.current_mean * decay_factor)
    }
}