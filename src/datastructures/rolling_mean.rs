use std::collections::VecDeque;

/// Calculates a rolling mean over a time window using f32 timestamps.
///
/// This structure efficiently stores data points and calculates the mean
/// of values that fall within a given `window` (delta_t) relative
/// to a query time.
///
/// It assumes that timestamps provided to `add` are monotonically
/// increasing (i.e., added in chronological order).
pub struct RollingMeanF32 {
    // Stores (timestamp, value)
    data: VecDeque<(f32, f32)>,
    // The time window (delta_t)
    window: f32,
    // The running sum of values *within* the window
    sum: f32,
}

impl RollingMeanF32 {
    /// Creates a new rolling mean calculator with a fixed `window` duration (delta_t).
    pub fn new(window: f32) -> Self {
        if window <= 0.0 {
            panic!("Window must be a positive value");
        }
        RollingMeanF32 {
            data: VecDeque::new(),
            window,
            sum: 0.0,
        }
    }

    /// Removes data points older than the window relative to `current_time`.
    fn prune(&mut self, current_time: f32) {
        let cutoff = current_time - self.window;

        // Loop while the front item is older than the cutoff time
        while let Some((ts, val)) = self.data.front() {
            if *ts < cutoff {
                // Remove the old item and subtract it from the sum
                self.sum -= val;
                self.data.pop_front();
            } else {
                // The rest of the items are in the window, so we stop
                break;
            }
        }
    }

    /// Adds a new value at a specific `timestamp` (t).
    ///
    /// This method also prunes any data that falls out of the
    /// window relative to this new `timestamp`.
    pub fn add(&mut self, timestamp: f32, value: f32) {
        // Enforce the assumption that time only moves forward
        if let Some((last_ts, _)) = self.data.back() {
            if timestamp < *last_ts {
                panic!("Timestamps must be monotonically increasing.");
            }
        }

        // Add the new item
        self.data.push_back((timestamp, value));
        self.sum += value;

        // Prune old items based on this new timestamp
        self.prune(timestamp);
    }

    /// Calculates the mean of all values since `current_time - window`.
    ///
    /// This call will first prune any data that has become stale since
    /// the last `add` operation.
    pub fn get_mean(&mut self, current_time: f32) -> Option<f32> {
        // Prune any remaining old data relative to the query time
        self.prune(current_time);

        if self.data.is_empty() {
            None
        } else {
            Some(self.sum / self.data.len() as f32)
        }
    }

    /// Gets the current count of items in the window.
    pub fn count(&self) -> usize {
        self.data.len()
    }
}
