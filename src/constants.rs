// --- Network Dimensions ---
pub const NUM_INPUT_NEURONS: usize = 28 * 28;
pub const NUM_OUTPUT_NEURONS: usize = 2;
pub const NUM_HIDDEN_NEURONS: usize = 20; // As per paper's hardware implementation for 8x8 digits

// --- Simulation Parameters ---
pub const STEPS_PER_IMAGE: usize = 15; // As per paper Table 1 for 8x8 Digits
pub const STEP_SIZE_MS: f64 = 1.0;

// --- Training Parameters ---
pub const NUM_EPOCHS: usize = 64; // As per paper Table 1
pub const LEARNING_RATE: f64 = 1.0 / 8.0; // As per paper Table 1

// --- HaSiST Parameters ---
pub const HASI_K: f64 = 0.5; // Steepness, as per paper Table 1
pub const HASI_V_TH: f64 = 0.0; // Threshold, as per paper Table 1

// Hidden Layer l1, l2
pub const HASI_L1_HID: f64 = -2.0; // As per paper Table 1
pub const HASI_L2_HID: f64 = 2.0;  // As per paper Table 1

// Output Layer l1, l2
pub const HASI_L1_OUT: f64 = -20.0; // As per paper Table 1
pub const HASI_L2_OUT: f64 = 20.0;  // As per paper Table 1

// Gradient controllers (gamma)
pub const GAMMA_HID: f64 = 1.0; // As per paper Table 1
pub const GAMMA_OUT: f64 = 1.0; // As per paper Table 1
