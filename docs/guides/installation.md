# Installation Guide

This guide provides detailed installation instructions for RunNX across different platforms and use cases, including YOLOv8 object detection capabilities.

## System Requirements

### Minimum Requirements
- **Rust**: 1.85.0 or later (latest stable recommended)
- **Protocol Buffers**: `protoc` compiler version 3.12 or later
- **Memory**: 4GB RAM minimum (8GB+ recommended for YOLOv8 models)
- **Storage**: 200MB for basic installation, additional space for YOLO models (~6MB for YOLOv8n)

### Supported Platforms
- **Linux**: Ubuntu 18.04+, Debian 10+, CentOS 7+, Arch Linux
- **macOS**: 10.15 (Catalina) or later (including Apple Silicon M1/M2)
- **Windows**: Windows 10 or later (Windows 11 recommended)

### Recommended for Object Detection
- **Memory**: 8GB+ RAM for optimal YOLOv8 performance
- **Storage**: 1GB+ for model storage and test images
- **Graphics**: No GPU required, but improves performance for large models

## Installing Dependencies

### Protocol Buffers Compiler

RunNX requires `protoc` to compile protocol buffer definitions.

#### Ubuntu/Debian
```bash
# Update package list
sudo apt-get update

# Install protobuf compiler
sudo apt-get install protobuf-compiler

# Verify installation
protoc --version
```

#### CentOS/RHEL/Fedora
```bash
# CentOS/RHEL 7
sudo yum install protobuf-compiler

# CentOS/RHEL 8+ / Fedora
sudo dnf install protobuf-compiler

# Verify installation
protoc --version
```

#### macOS
```bash
# Using Homebrew (recommended)
brew install protobuf

# Using MacPorts
sudo port install protobuf3-cpp

# Verify installation
protoc --version
```

#### Windows

**Option 1: Using Chocolatey (Recommended)**
```powershell
# Install Chocolatey if not already installed
# See: https://chocolatey.org/install

# Install protoc
choco install protoc

# Verify installation
protoc --version
```

**Option 2: Manual Installation**
1. Download pre-compiled binaries from [Protocol Buffers releases](https://github.com/protocolbuffers/protobuf/releases)
2. Extract to a directory (e.g., `C:\protobuf`)
3. Add `C:\protobuf\bin` to your PATH environment variable
4. Verify with `protoc --version`

**Option 3: Using vcpkg**
```powershell
# Install vcpkg if not already installed
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# Install protobuf
.\vcpkg install protobuf
```

### Rust Installation

If you don't have Rust installed:

```bash
# Install Rust using rustup (recommended)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Follow the on-screen instructions, then:
source $HOME/.cargo/env

# Verify installation
rustc --version
cargo --version
```

For Windows, download and run the installer from [rustup.rs](https://rustup.rs/).

## Installing RunNX

### Option 1: As a Library Dependency

Add to your `Cargo.toml`:

```toml
[dependencies]
runnx = "0.2.0"

# Optional features
runnx = { version = "0.2.0", features = ["async"] }
```

Then in your Rust code:
```rust
use runnx::{Model, Tensor};
```

### Option 2: From crates.io

```bash
# Install the CLI tool globally
cargo install runnx

# Run the CLI
runnx-runner --help
```

### Option 3: From Source

```bash
# Clone the repository
git clone https://github.com/jgalego/runnx.git
cd runnx

# Build in release mode
cargo build --release

# The binary will be available at
./target/release/runnx-runner

# Optionally install globally
cargo install --path .
```

### Option 4: Development Installation

For contributing or development:

```bash
# Clone with all submodules
git clone --recursive https://github.com/jgalego/runnx.git
cd runnx

# Install development dependencies
cargo install just  # Task runner (optional but recommended)

# Run tests to verify installation
cargo test

# Build documentation
cargo doc --open
```

## Feature Flags

RunNX supports several optional features:

```toml
[dependencies]
runnx = { version = "0.2.0", features = ["async", "formal-verification"] }
```

### Available Features

- **`async`**: Enables async/await support for model operations
- **`formal-verification`**: Enables formal verification contracts and runtime checks
- **`serde`**: Enables serialization support (enabled by default)

## Verification

### Test Your Installation

Create a simple test file (`test_runnx.rs`):

```rust
use runnx::{Model, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("RunNX installation test");
    
    // Test basic tensor creation
    let tensor = Tensor::from_array(ndarray::array![[1.0, 2.0, 3.0]]);
    println!("Created tensor with shape: {:?}", tensor.shape());
    
    println!("✅ RunNX is working correctly!");
    Ok(())
}
```

Run it:
```bash
rustc test_runnx.rs --extern runnx
./test_runnx
```

### Test CLI Installation

```bash
# Check CLI is available
runnx-runner --version

# Test with a simple model (if you have one)
runnx-runner --model model.onnx --summary
```

## Troubleshooting

### Common Issues

#### `protoc` not found
```
error: failed to run custom build command for `prost-build`
```
**Solution**: Install Protocol Buffers compiler as described above.

#### Rust version too old
```
error: package `runnx` requires Rust 1.70.0 or newer
```
**Solution**: Update Rust:
```bash
rustup update stable
```

#### Permission denied on Unix systems
```bash
# If you get permission errors, try:
sudo chown -R $USER:$USER ~/.cargo
```

#### Windows MSVC build tools missing
**Solution**: Install Visual Studio Build Tools or Visual Studio Community with C++ development tools.

#### Out of memory during compilation
**Solution**: 
- Close other applications
- Increase virtual memory
- Use `cargo build` instead of `cargo build --release` initially

### Getting Help

If you encounter issues not covered here:

1. **Check existing issues**: [GitHub Issues](https://github.com/jgalego/runnx/issues)
2. **Create a new issue**: Include your OS, Rust version, and error messages
3. **Join discussions**: [GitHub Discussions](https://github.com/jgalego/runnx/discussions)

### Debugging Installation

Enable verbose output to diagnose issues:

```bash
# Verbose cargo output
RUST_LOG=debug cargo build --verbose

# Check system information
rustc --version --verbose
cargo --version --verbose
protoc --version
```

## Next Steps

Once RunNX is installed:

1. **[Quick Start Guide](quick-start.md)** - Basic usage examples
2. **[Usage Examples](examples.md)** - Comprehensive tutorials
3. **[Development Setup](../development/setup.md)** - If you plan to contribute
4. **[API Documentation](../api/)** - Detailed API reference

## Updating RunNX

### Library Updates
```bash
# Update to latest version
cargo update runnx
```

### CLI Updates
```bash
# Update globally installed CLI
cargo install runnx --force
```

### Development Updates
```bash
# Pull latest changes
git pull origin main

# Update dependencies
cargo update

# Rebuild
cargo build --release
```
