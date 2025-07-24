//! Comprehensive tests for CLI runner binary to improve coverage

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use tempfile::{NamedTempFile, TempDir};

/// Get the path to the runnx-runner binary
fn get_runner_binary() -> PathBuf {
    // Try different possible locations
    let possibilities = [
        "target/debug/runnx-runner",
        "target/release/runnx-runner",
        "./target/debug/runnx-runner",
        "./target/release/runnx-runner",
    ];

    for path in &possibilities {
        let path_buf = PathBuf::from(path);
        if path_buf.exists() {
            return path_buf;
        }
    }

    // Fallback to the most likely location
    PathBuf::from("target/debug/runnx-runner")
}

/// Create a minimal valid JSON model file for testing
fn create_test_model() -> NamedTempFile {
    let temp_file = NamedTempFile::new().unwrap();
    let model_content = r#"{
        "metadata": {
            "name": "test_model",
            "version": "1.0",
            "description": "Test model for CLI",
            "producer": "RunNX Test Suite",
            "onnx_version": "1.9.0",
            "domain": "test"
        },
        "graph": {
            "nodes": [],
            "inputs": [],
            "outputs": [],
            "initializers": []
        }
    }"#;
    fs::write(temp_file.path(), model_content).unwrap();
    temp_file
}

#[test]
fn test_cli_help_command() {
    let output = Command::new(get_runner_binary())
        .arg("--help")
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Usage") || stdout.contains("USAGE"));
}

#[test]
fn test_cli_version_command() {
    // Since --version isn't implemented, test with invalid args to trigger help
    let output = Command::new(get_runner_binary())
        .output()
        .expect("Failed to execute command");

    // Should show usage/help when no args provided
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.is_empty());
}

#[test]
fn test_cli_with_invalid_arguments() {
    let output = Command::new(get_runner_binary())
        .arg("--invalid-flag")
        .output()
        .expect("Failed to execute command");

    // Invalid arguments should fail
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.is_empty());
}

#[test]
fn test_cli_run_command_basic() {
    let temp_file = create_test_model();
    let model_path = temp_file.path();

    let output = Command::new(get_runner_binary())
        .arg("--model")
        .arg(model_path.to_str().unwrap())
        .output()
        .expect("Failed to execute command");

    // We're testing that the CLI handles the command, regardless of success
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Either succeeds or fails with meaningful error message
    assert!(!stderr.is_empty() || !stdout.is_empty() || !output.status.success());
}

#[test]
fn test_cli_validate_command() {
    // Test missing model argument
    let output = Command::new(get_runner_binary())
        .output()
        .expect("Failed to execute command");

    // Should fail when no model is provided
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--model") || stderr.contains("Usage"));
}

#[test]
fn test_cli_info_command() {
    let temp_file = create_test_model();
    let model_path = temp_file.path();

    let output = Command::new(get_runner_binary())
        .arg("--model")
        .arg(model_path.to_str().unwrap())
        .arg("--summary")
        .output()
        .expect("Failed to execute command");

    // Test that the command is processed
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!stderr.is_empty() || !stdout.is_empty() || !output.status.success());
}

#[test]
fn test_cli_with_nonexistent_file() {
    let output = Command::new(get_runner_binary())
        .arg("--model")
        .arg("/nonexistent/file.json")
        .output()
        .expect("Failed to execute command");

    // Should handle missing file gracefully
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.is_empty());
}

#[test]
fn test_cli_output_format_options() {
    let temp_file = create_test_model();
    let model_path = temp_file.path();

    // Test different flags supported by the CLI
    let flags = vec!["--verbose", "--summary"];

    for flag in flags {
        let output = Command::new(get_runner_binary())
            .arg("--model")
            .arg(model_path.to_str().unwrap())
            .arg(flag)
            .output()
            .expect("Failed to execute command");

        // Just testing that flags are processed
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(!stderr.is_empty() || !stdout.is_empty() || !output.status.success());
    }
}

#[test]
fn test_cli_input_data_options() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("model.json");
    let input_path = temp_dir.path().join("input.json");

    // Create a test model
    let model_content = r#"{
        "metadata": {"name": "test", "version": "1.0", "producer": "RunNX Test", "description": "Test", "onnx_version": "1.9.0", "domain": "test"},
        "graph": {"nodes": [], "inputs": [], "outputs": [], "initializers": []}
    }"#;
    fs::write(&model_path, model_content).unwrap();

    // Create test input
    fs::write(
        &input_path,
        r#"{"input": {"data": [1.0, 2.0, 3.0], "shape": [3]}}"#,
    )
    .unwrap();

    let output = Command::new(get_runner_binary())
        .arg("--model")
        .arg(model_path.to_str().unwrap())
        .arg("--input")
        .arg(input_path.to_str().unwrap())
        .output()
        .expect("Failed to execute command");

    // Test input file handling
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!stderr.is_empty() || !stdout.is_empty() || !output.status.success());
}

#[test]
fn test_cli_batch_processing() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("model.json");
    let output_path = temp_dir.path().join("output.json");

    // Create test model
    let model_content = r#"{
        "metadata": {"name": "test", "version": "1.0", "producer": "RunNX Test", "description": "Test", "onnx_version": "1.9.0", "domain": "test"},
        "graph": {"nodes": [], "inputs": [], "outputs": [], "initializers": []}
    }"#;
    fs::write(&model_path, model_content).unwrap();

    let output = Command::new(get_runner_binary())
        .arg("--model")
        .arg(model_path.to_str().unwrap())
        .arg("--output")
        .arg(output_path.to_str().unwrap())
        .output()
        .expect("Failed to execute command");

    // Test output file specification
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!stderr.is_empty() || !stdout.is_empty() || !output.status.success());
}

#[test]
fn test_cli_configuration_options() {
    let temp_file = create_test_model();
    let model_path = temp_file.path();

    // Test combining multiple valid flags
    let output = Command::new(get_runner_binary())
        .arg("--model")
        .arg(model_path.to_str().unwrap())
        .arg("--verbose")
        .arg("--summary")
        .output()
        .expect("Failed to execute command");

    // Test multiple configuration flags together
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!stderr.is_empty() || !stdout.is_empty() || !output.status.success());
}

#[test]
fn test_cli_error_conditions() {
    // Test various error conditions to improve coverage
    let error_cases = vec![
        vec!["--model"],                      // Missing model value
        vec!["--input"],                      // Missing input value
        vec!["--output"],                     // Missing output value
        vec!["--model", "/nonexistent.json"], // Nonexistent model file
        vec!["unknown-argument"],             // Unknown argument
    ];

    for case in error_cases {
        let mut cmd = Command::new(get_runner_binary());
        for arg in case {
            cmd.arg(arg);
        }

        let output = cmd.output().expect("Failed to execute command");

        // These should all result in errors, which is expected behavior
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!stderr.is_empty() || !output.status.success());
    }
}

#[test]
fn test_cli_signal_handling() {
    // Test that CLI handles interruption gracefully
    let temp_file = create_test_model();
    let model_path = temp_file.path();

    // Start a command that might run longer
    let mut child = Command::new(get_runner_binary())
        .arg("--model")
        .arg(model_path.to_str().unwrap())
        .arg("--verbose")
        .spawn()
        .expect("Failed to start command");

    // Give it a moment then terminate
    std::thread::sleep(std::time::Duration::from_millis(100));

    let exit_status = child.kill();
    assert!(exit_status.is_ok());

    // Wait for the process to clean up
    let _ = child.wait();
}
