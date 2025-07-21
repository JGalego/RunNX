//! Command-line ONNX model runner
//!
//! This binary provides a command-line interface for running ONNX models
//! with the RunNX runtime.

use runnx::{Model, Tensor};
use std::collections::HashMap;
use std::env;
use std::fs;

/// Command-line arguments
struct Args {
    model_path: String,
    input_path: Option<String>,
    output_path: Option<String>,
    verbose: bool,
    show_summary: bool,
}

impl Args {
    /// Parse command-line arguments
    fn parse() -> Result<Self, String> {
        let args: Vec<String> = env::args().collect();

        if args.len() < 2 {
            return Err("Usage: runnx-runner --model <model.json> [options]".to_string());
        }

        let mut model_path = None;
        let mut input_path = None;
        let mut output_path = None;
        let mut verbose = false;
        let mut show_summary = false;

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--model" | "-m" => {
                    if i + 1 >= args.len() {
                        return Err("--model requires a value".to_string());
                    }
                    model_path = Some(args[i + 1].clone());
                    i += 2;
                }
                "--input" | "-i" => {
                    if i + 1 >= args.len() {
                        return Err("--input requires a value".to_string());
                    }
                    input_path = Some(args[i + 1].clone());
                    i += 2;
                }
                "--output" | "-o" => {
                    if i + 1 >= args.len() {
                        return Err("--output requires a value".to_string());
                    }
                    output_path = Some(args[i + 1].clone());
                    i += 2;
                }
                "--verbose" | "-v" => {
                    verbose = true;
                    i += 1;
                }
                "--summary" | "-s" => {
                    show_summary = true;
                    i += 1;
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                _ => {
                    return Err(format!("Unknown argument: {}", args[i]));
                }
            }
        }

        let model_path = model_path.ok_or("--model is required")?;

        Ok(Args {
            model_path,
            input_path,
            output_path,
            verbose,
            show_summary,
        })
    }
}

/// Input data format for JSON files
#[derive(serde::Deserialize)]
struct InputData {
    #[serde(flatten)]
    inputs: HashMap<String, InputTensor>,
}

#[derive(serde::Deserialize)]
struct InputTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

/// Output data format for JSON files
#[derive(serde::Serialize)]
struct OutputData {
    #[serde(flatten)]
    outputs: HashMap<String, OutputTensor>,
}

#[derive(serde::Serialize)]
struct OutputTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

fn main() {
    // Initialize logging
    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    // Parse arguments
    let args = match Args::parse() {
        Ok(args) => args,
        Err(e) => {
            eprintln!("Error: {e}");
            print_help();
            std::process::exit(1);
        }
    };

    // Set logging level based on verbosity
    if args.verbose {
        env::set_var("RUST_LOG", "debug");
    }

    // Load model
    println!("Loading model from: {}", args.model_path);
    let model = match Model::from_file(&args.model_path) {
        Ok(model) => model,
        Err(e) => {
            eprintln!("Error loading model: {e}");
            std::process::exit(1);
        }
    };

    if args.verbose {
        println!("Model loaded successfully");
    }

    // Show model summary if requested
    if args.show_summary {
        println!("\nModel Summary:");
        println!("{}", model.summary());
        if args.input_path.is_none() {
            return;
        }
    }

    // Load input data
    let inputs = match load_inputs(&model, args.input_path.as_deref()) {
        Ok(inputs) => inputs,
        Err(e) => {
            eprintln!("Error loading inputs: {e}");
            std::process::exit(1);
        }
    };

    if args.verbose {
        println!("Input tensors:");
        for (name, tensor) in &inputs {
            println!("  {}: shape {:?}", name, tensor.shape());
        }
    }

    // Run inference
    println!("Running inference...");
    let start_time = std::time::Instant::now();

    let outputs = match model.run(&inputs) {
        Ok(outputs) => outputs,
        Err(e) => {
            eprintln!("Error during inference: {e}");
            std::process::exit(1);
        }
    };

    let inference_time = start_time.elapsed();
    println!("Inference completed in {:.2}ms", inference_time.as_millis());

    if args.verbose {
        println!("Output tensors:");
        for (name, tensor) in &outputs {
            println!("  {}: shape {:?}", name, tensor.shape());
        }
    }

    // Save or print outputs
    match save_outputs(&outputs, args.output_path.as_deref()) {
        Ok(()) => {
            if args.output_path.is_some() {
                println!("Outputs saved to: {}", args.output_path.unwrap());
            }
        }
        Err(e) => {
            eprintln!("Error saving outputs: {e}");
            std::process::exit(1);
        }
    }
}

/// Load input tensors from file or create default inputs
fn load_inputs(
    model: &Model,
    input_path: Option<&str>,
) -> Result<HashMap<String, Tensor>, Box<dyn std::error::Error>> {
    match input_path {
        Some(path) => {
            // Load from JSON file
            let content = fs::read_to_string(path)?;
            let input_data: InputData = serde_json::from_str(&content)?;

            let mut inputs = HashMap::new();
            for (name, input_tensor) in input_data.inputs {
                let tensor = Tensor::from_shape_vec(&input_tensor.shape, input_tensor.data)?;
                inputs.insert(name, tensor);
            }

            Ok(inputs)
        }
        None => {
            // Create default inputs (zeros) based on model input specs
            let mut inputs = HashMap::new();
            for input_spec in model.input_specs() {
                // Convert Option<usize> to usize, using 1 for dynamic dimensions
                let shape: Vec<usize> = input_spec
                    .shape
                    .iter()
                    .map(|&dim| dim.unwrap_or(1))
                    .collect();

                let tensor = Tensor::zeros(&shape);
                inputs.insert(input_spec.name.clone(), tensor);

                println!(
                    "Created default input '{}' with shape {:?}",
                    input_spec.name, shape
                );
            }

            Ok(inputs)
        }
    }
}

/// Save outputs to file or print to console
fn save_outputs(
    outputs: &HashMap<String, Tensor>,
    output_path: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    match output_path {
        Some(path) => {
            // Save to JSON file
            let mut output_data = HashMap::new();
            for (name, tensor) in outputs {
                let output_tensor = OutputTensor {
                    data: tensor.data().iter().cloned().collect(),
                    shape: tensor.shape().to_vec(),
                };
                output_data.insert(name.clone(), output_tensor);
            }

            let json_data = OutputData {
                outputs: output_data,
            };
            let content = serde_json::to_string_pretty(&json_data)?;
            fs::write(path, content)?;
        }
        None => {
            // Print to console
            println!("\nOutputs:");
            for (name, tensor) in outputs {
                println!("{name}:");
                println!("  Shape: {:?}", tensor.shape());

                // Get the raw data as a 1D slice for display
                if let Some(data_slice) = tensor.data().as_slice() {
                    let display_len = 10.min(data_slice.len());
                    println!("  Data: {:?}", &data_slice[..display_len]);
                    if data_slice.len() > 10 {
                        println!("  ... ({} more elements)", data_slice.len() - 10);
                    }
                } else {
                    // Fallback for non-contiguous arrays
                    println!(
                        "  Data: (non-contiguous layout, shape: {:?})",
                        tensor.shape()
                    );
                }
                println!();
            }
        }
    }

    Ok(())
}

fn print_help() {
    println!("RunNX Runner");
    println!();
    println!("USAGE:");
    println!("    runnx-runner --model <MODEL> [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("    -m, --model <MODEL>      Path to the ONNX model file (.json format)");
    println!("    -i, --input <INPUT>      Path to input data file (.json format)");
    println!("    -o, --output <OUTPUT>    Path to save output data (.json format)");
    println!("    -v, --verbose            Enable verbose logging");
    println!("    -s, --summary            Show model summary");
    println!("    -h, --help               Print this help message");
    println!();
    println!("EXAMPLES:");
    println!("    # Run model with default inputs and print outputs");
    println!("    runnx-runner --model model.json");
    println!();
    println!("    # Run with custom inputs from file");
    println!("    runnx-runner --model model.json --input inputs.json");
    println!();
    println!("    # Save outputs to file");
    println!("    runnx-runner --model model.json --output outputs.json");
    println!();
    println!("    # Show model summary only");
    println!("    runnx-runner --model model.json --summary");
    println!();
    println!("INPUT FORMAT (JSON):");
    println!("    {{");
    println!("      \"input_name\": {{");
    println!("        \"data\": [1.0, 2.0, 3.0, 4.0],");
    println!("        \"shape\": [2, 2]");
    println!("      }}");
    println!("    }}");
    println!();
    println!("OUTPUT FORMAT (JSON):");
    println!("    {{");
    println!("      \"output_name\": {{");
    println!("        \"data\": [5.0, 7.0, 9.0, 11.0],");
    println!("        \"shape\": [2, 2]");
    println!("      }}");
    println!("    }}");
}
