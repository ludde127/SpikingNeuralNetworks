use image::{GrayImage, Luma};
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::fs;
use std::path::Path;

fn load_iris_images(class_path: &str) -> Vec<(Vec<f64>, usize)> {
    let path = Path::new(class_path);
    if let Ok(entries) = fs::read_dir(path) {
        entries
            .par_bridge()
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("jpg") {
                    if let Ok(img) = image::open(&path) {
                        let resized_img: GrayImage =
                            img.resize_exact(28, 28, image::imageops::Lanczos3).into_luma8();
                        let pixel_data: Vec<f64> = resized_img
                            .pixels()
                            .map(|&Luma([p])| p as f64 / 255.0)
                            .collect();
                        // Dummy label, will be assigned later
                        return Some((pixel_data, 0));
                    }
                }
                None
            })
            .collect()
    } else {
        Vec::new()
    }
}

pub fn load_and_prep_iris_data() -> (Vec<(Vec<f64>, usize)>, Vec<(Vec<f64>, usize)>) {
    println!("--- Loading and Preparing Iris Data ---");

    let class1_path = "iris_dataset/iris-setosa";
    let class2_path = "iris_dataset/iris-versicolour";

    println!("Loading images from {} and {}", class1_path, class2_path);

    let mut class1_images = load_iris_images(class1_path);
    let mut class2_images = load_iris_images(class2_path);

    // Assign labels: 0 for setosa, 1 for versicolour
    for (_, label) in &mut class1_images {
        *label = 0;
    }
    for (_, label) in &mut class2_images {
        *label = 1;
    }

    println!(
        "Loaded {} setosa images and {} versicolour images",
        class1_images.len(),
        class2_images.len()
    );

    // Balance the dataset
    let min_samples = class1_images.len().min(class2_images.len());
    class1_images.truncate(min_samples);
    class2_images.truncate(min_samples);
    println!("Balanced dataset to {} samples per class.", min_samples);

    // Combine and shuffle
    let mut all_images = Vec::new();
    all_images.extend(class1_images);
    all_images.extend(class2_images);
    all_images.shuffle(&mut rand::thread_rng());

    // Split into training (80%) and testing (20%)
    let split_idx = (all_images.len() as f64 * 0.8) as usize;
    let (training_set, test_set) = all_images.split_at(split_idx);

    println!(
        "Training set: {} images, Test set: {} images",
        training_set.len(),
        test_set.len()
    );

    (training_set.to_vec(), test_set.to_vec())
}
