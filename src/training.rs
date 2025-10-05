use crate::constants::{NUM_EPOCHS, STEPS_PER_IMAGE, STEP_SIZE_MS};
use crate::network::Network;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::time::Instant;

pub fn train_network(network: &mut Network, training_set: &[(Vec<f64>, usize)]) {
    println!("\n--- Training Phase (HaSiST) ---");
    let training_start = Instant::now();

    // Pre-convert all training images to input vectors to avoid repeated computation
    println!("Pre-processing training images...");
    let training_inputs: Vec<(Vec<Vec<f64>>, usize)> = training_set
        .par_iter()
        .map(|(img_data, label)| {
            let input_vector: Vec<Vec<f64>> = (0..STEPS_PER_IMAGE)
                .map(|_| {
                    img_data
                        .iter()
                        .map(|&pixel_val| pixel_val * 0.20) // Scale factor for input current
                        .collect()
                })
                .collect();
            (input_vector, *label)
        })
        .collect();
    println!("Training on {} images...", training_inputs.len());

    let num_samples = training_inputs.len();
    let pb = ProgressBar::new((NUM_EPOCHS * num_samples) as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} samples - {msg}")
            .unwrap()
            .progress_chars("##-"),
    );

    for epoch in 0..NUM_EPOCHS {
        let mut total_loss = 0.0;
        let mut correct_predictions = 0;

        for (input_vector, label) in &training_inputs {
            let (loss, predicted_label) =
                network.train_on_sample(input_vector, *label, STEPS_PER_IMAGE, STEP_SIZE_MS);
            total_loss += loss;
            if predicted_label == *label {
                correct_predictions += 1;
            }
            pb.inc(1);
        }

        let avg_loss = total_loss / num_samples as f64;
        let accuracy = (correct_predictions as f64 / num_samples as f64) * 100.0;

        // Update the progress bar's message
        pb.set_message(format!(
            "Epoch {}/{} - Avg Loss: {:.4}, Acc: {:.2}%",
            epoch + 1,
            NUM_EPOCHS,
            avg_loss,
            accuracy
        ));
    }

    pb.finish_with_message("Training complete");
    println!("Training completed in {:.2?}", training_start.elapsed());
}
