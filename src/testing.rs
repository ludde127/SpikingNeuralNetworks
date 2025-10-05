use crate::constants::{STEPS_PER_IMAGE, STEP_SIZE_MS};
use crate::network::Network;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::ParallelIterator;
use rayon::prelude::*;

pub fn test_network(
    network: &Network,
    test_set: &[(Vec<f64>, usize)],
    output_neuron_indices: &[usize],
) {
    println!("\n--- Testing Phase ---");

    // Pre-convert all test images to input vectors
    println!("Pre-processing test images...");
    let test_inputs: Vec<(Vec<Vec<f64>>, usize)> = test_set
        .par_iter()
        .map(|(img_data, label)| {
            let input_vector: Vec<Vec<f64>> = (0..STEPS_PER_IMAGE)
                .map(|_| {
                    img_data
                        .iter()
                        .map(|&pixel_val| pixel_val * 0.20)
                        .collect()
                })
                .collect();
            (input_vector, *label)
        })
        .collect();

    let mut correct_predictions = 0;
    let mut confusion_matrix = vec![vec![0; 2]; 2];

    let test_pb = ProgressBar::new(test_inputs.len() as u64);
    test_pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.green/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("##-"),
    );

    for (input_vector, actual_label) in test_inputs.iter() {
        let predicted_label =
            network.predict(input_vector, STEPS_PER_IMAGE, STEP_SIZE_MS, output_neuron_indices);

        confusion_matrix[*actual_label][predicted_label] += 1;

        if predicted_label == *actual_label {
            correct_predictions += 1;
        }

        test_pb.inc(1);
    }
    test_pb.finish_with_message("Testing complete");

    let accuracy = (correct_predictions as f64 / test_inputs.len() as f64) * 100.0;

    println!("\n--- Results ---");
    println!(
        "Test Accuracy: {:.2}% ({}/{})",
        accuracy,
        correct_predictions,
        test_inputs.len()
    );

    // Display confusion matrix
    println!("\n--- Confusion Matrix ---");
    println!("                  Predicted");
    println!("                  Setosa  Versicolour");
    println!(
        "Actual Setosa       {:>3}      {:>3}",
        confusion_matrix[0][0], confusion_matrix[0][1]
    );
    println!(
        "       Versicolour  {:>3}      {:>3}",
        confusion_matrix[1][0], confusion_matrix[1][1]
    );
}
