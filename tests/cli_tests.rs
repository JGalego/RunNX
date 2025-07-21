//! CLI runner tests for the ONNX runtime
//!
//! These tests verify the complete functionality of the CLI runner
//! by testing command-line argument parsing, file I/O, and integration workflows.

use runnx::{Model, Tensor};
use std::collections::HashMap;
use std::fs;
use std::process::Command;
use tempfile::{NamedTempFile, TempDir};

/// Test CLI argument parsing and help functionality
#[test]
fn test_cli_help() {
    let output = Command::new("cargo")
        .args(["run", "--bin", "runnx-runner", "--", "--help"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("RunNX Runner"));
    assert!(stdout.contains("USAGE:"));
    assert!(stdout.contains("--model"));
    assert!(stdout.contains("INPUT FORMAT"));
}

/// Test CLI with missing required arguments
#[test]
fn test_cli_missing_model() {
    let output = Command::new("cargo")
        .args(["run", "--bin", "runnx-runner"])
        .output()
        .expect("Failed to execute command");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Usage:") || stderr.contains("--model"));
}

/// Test CLI with invalid arguments
#[test]
fn test_cli_invalid_argument() {
    let temp_model = create_temp_model_file();
    
    let output = Command::new("cargo")
        .args(["run", "--bin", "runnx-runner", "--", "--model", temp_model.path().to_str().unwrap(), "--invalid-arg"])
        .output()
        .expect("Failed to execute command");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Unknown argument") || stderr.contains("invalid-arg"));
}

/// Test CLI model summary functionality
#[test]
fn test_cli_model_summary() {
    let temp_model = create_temp_model_file();
    
    let output = Command::new("cargo")
        .args(["run", "--bin", "runnx-runner", "--", "--model", temp_model.path().to_str().unwrap(), "--summary"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Model Summary"));
}

/// Test CLI with verbose output
#[test]
fn test_cli_verbose() {
    let temp_model = create_temp_model_file();
    
    let output = Command::new("cargo")
        .args(["run", "--bin", "runnx-runner", "--", "--model", temp_model.path().to_str().unwrap(), "--verbose", "--summary"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Model loaded successfully") || stdout.contains("Loading model"));
}

/// Test CLI with default inputs (no input file)
#[test]
fn test_cli_default_inputs() {
    let temp_model = create_temp_model_file();
    
    let output = Command::new("cargo")
        .args(["run", "--bin", "runnx-runner", "--", "--model", temp_model.path().to_str().unwrap()])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Running inference") || stdout.contains("Outputs:"));
}

/// Test CLI with custom input file
#[test]
fn test_cli_custom_inputs() {
    let temp_model = create_temp_model_file();
    let temp_input = create_temp_input_file();
    
    let output = Command::new("cargo")
        .args([
            "run", "--bin", "runnx-runner", "--", 
            "--model", temp_model.path().to_str().unwrap(),
            "--input", temp_input.path().to_str().unwrap()
        ])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Running inference") || stdout.contains("Outputs:"));
}

/// Test CLI with output file
#[test]
fn test_cli_output_file() {
    let temp_model = create_temp_model_file();
    let temp_output = NamedTempFile::new().unwrap();
    let output_path = temp_output.path().to_str().unwrap();
    
    let output = Command::new("cargo")
        .args([
            "run", "--bin", "runnx-runner", "--", 
            "--model", temp_model.path().to_str().unwrap(),
            "--output", output_path
        ])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Outputs saved to"));
    
    // Verify output file was created and contains valid JSON
    let output_content = fs::read_to_string(output_path).unwrap();
    assert!(serde_json::from_str::<serde_json::Value>(&output_content).is_ok());
}

/// Test CLI with invalid model file
#[test]
fn test_cli_invalid_model() {
    let temp_file = NamedTempFile::new().unwrap();
    fs::write(temp_file.path(), "invalid json").unwrap();
    
    let output = Command::new("cargo")
        .args(["run", "--bin", "runnx-runner", "--", "--model", temp_file.path().to_str().unwrap()])
        .output()
        .expect("Failed to execute command");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Error loading model") || stderr.contains("Error"));
}

/// Test CLI with invalid input file
#[test]
fn test_cli_invalid_input() {
    let temp_model = create_temp_model_file();
    let temp_input = NamedTempFile::new().unwrap();
    fs::write(temp_input.path(), "invalid json").unwrap();
    
    let output = Command::new("cargo")
        .args([
            "run", "--bin", "runnx-runner", "--", 
            "--model", temp_model.path().to_str().unwrap(),
            "--input", temp_input.path().to_str().unwrap()
        ])
        .output()
        .expect("Failed to execute command");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Error loading inputs") || stderr.contains("Error"));
}

/// Test CLI with nonexistent model file
#[test]
fn test_cli_nonexistent_model() {
    let output = Command::new("cargo")
        .args(["run", "--bin", "runnx-runner", "--", "--model", "/nonexistent/path.json"])
        .output()
        .expect("Failed to execute command");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Error loading model") || stderr.contains("No such file"));
}

/// Test CLI with nonexistent input file
#[test]
fn test_cli_nonexistent_input() {
    let temp_model = create_temp_model_file();
    
    let output = Command::new("cargo")
        .args([
            "run", "--bin", "runnx-runner", "--", 
            "--model", temp_model.path().to_str().unwrap(),
            "--input", "/nonexistent/input.json"
        ])
        .output()
        .expect("Failed to execute command");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Error loading inputs") || stderr.contains("No such file"));
}

/// Test CLI argument parsing edge cases
#[test]
fn test_cli_argument_parsing_edge_cases() {
    let temp_model = create_temp_model_file();
    
    // Test missing argument value
    let output = Command::new("cargo")
        .args(["run", "--bin", "runnx-runner", "--", "--model"])
        .output()
        .expect("Failed to execute command");
    
    assert!(!output.status.success());
    
    // Test model required error
    let output = Command::new("cargo")
        .args(["run", "--bin", "runnx-runner", "--", "--verbose"])
        .output()
        .expect("Failed to execute command");
    
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--model is required") || stderr.contains("model"));
}

/// Test CLI with all flags combined
#[test]
fn test_cli_all_flags() {
    let temp_model = create_temp_model_file();
    let temp_input = create_temp_input_file();
    let temp_output = NamedTempFile::new().unwrap();
    
    let output = Command::new("cargo")
        .args([
            "run", "--bin", "runnx-runner", "--", 
            "--model", temp_model.path().to_str().unwrap(),
            "--input", temp_input.path().to_str().unwrap(),
            "--output", temp_output.path().to_str().unwrap(),
            "--verbose",
            "--summary"
        ])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Model Summary"));
    assert!(stdout.contains("Outputs saved to"));
}

/// Helper function to create a temporary model file
fn create_temp_model_file() -> NamedTempFile {
    let model = Model::create_simple_linear();
    let temp_file = NamedTempFile::new().unwrap();
    model.to_file(temp_file.path()).unwrap();
    temp_file
}

/// Helper function to create a temporary input file
fn create_temp_input_file() -> NamedTempFile {
    let input_data = serde_json::json!({
        "input": {
            "data": [1.0, 2.0, 3.0],
            "shape": [1, 3]
        }
    });
    
    let temp_file = NamedTempFile::new().unwrap();
    fs::write(temp_file.path(), serde_json::to_string_pretty(&input_data).unwrap()).unwrap();
    temp_file
}
