use rand::prelude::*;
use rand_distr::{Normal, Distribution};

/// Generates a random value from a normal distribution defined by a min/max
/// range corresponding to +/- 5 standard deviations, and clamps the result.
///
/// - `min`: The value at -5 std devs.
/// - `max`: The value at +5 std devs.
/// - `rng`: A mutable reference to a random number generator.
pub fn get_clamped_normal(min: f32, max: f32, rng: &mut impl Rng) -> f32 {
    // 1. Calculate the distribution's parameters (mean and std_dev)
    // mean = (min + max) / 2
    let mean = (min + max) / 2.0;

    // std_dev = (max - min) / 10
    let std_dev = (max - min) / 10.0;

    // 2. Create the normal distribution
    // We unwrap(), assuming min < max, which makes std_dev positive.
    let dist = Normal::new(mean, std_dev).unwrap();

    // 3. Sample a value from the distribution
    let val = dist.sample(rng);

    // 4. Clamp the value to the [min, max] range
    // This ensures that values beyond +/- 5 sigma are capped.
    val.clamp(min, max)
}