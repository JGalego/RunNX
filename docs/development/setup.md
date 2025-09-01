# Development Setup Guide

This guide covers setting up a development environment for contributing to RunNX, including all necessary tools, dependencies, and workflow configurations.

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows 10+
- **Memory**: 4GB RAM minimum (8GB+ recommended for YOLOv8 development)
- **Storage**: 3GB free space for development environment and models
- **Network**: Internet connection for downloading dependencies and models

### Required Tools

#### 1. Rust Toolchain

Install the latest stable Rust (1.81+):

```bash
# Install Rust using rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Source the environment
source $HOME/.cargo/env

# Verify installation
rustc --version
cargo --version
```

#### 2. Protocol Buffers Compiler

RunNX requires `protoc` for building ONNX protobuf support:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install protobuf-compiler
```

**macOS:**
```bash
brew install protobuf
```

**Windows:**
```bash
choco install protoc
# or download from https://github.com/protocolbuffers/protobuf/releases
```

#### 3. Additional Development Tools

Install helpful development tools:

```bash
# Code formatting and linting
rustup component add rustfmt clippy

# Task runner (highly recommended for RunNX development)
cargo install just

# Coverage tools for testing
cargo install cargo-tarpaulin

# Documentation generation
cargo install cargo-doc

# Benchmarking tools (for performance work)
cargo install cargo-criterion

# Image processing tools (for YOLOv8 development)
# These are handled by Cargo dependencies, but you may want Graphviz for visualization
sudo apt-get install graphviz  # Ubuntu/Debian
brew install graphviz          # macOS
choco install graphviz         # Windows
```

#### 4. Optional Tools for YOLOv8 Development

```bash
# Python for model conversion and comparison scripts
python3 -m pip install torch torchvision ultralytics onnx

# Image processing tools
sudo apt-get install imagemagick  # Ubuntu/Debian  
brew install imagemagick          # macOS
```

## Repository Setup

### Clone the Repository

```bash
# Clone with all submodules
git clone --recursive https://github.com/jgalego/runnx.git
cd runnx

# If you forgot --recursive, initialize submodules
git submodule update --init --recursive
```

### Verify Installation

```bash
# Test basic build
cargo build

# Run test suite
cargo test

# Run clippy linting
cargo clippy

# Format code
cargo fmt

# Build documentation
cargo doc --open
```

## Development Environment

### IDE Setup

#### Visual Studio Code

Recommended extensions:
- **rust-analyzer**: Rust language support
- **CodeLLDB**: Debugging support
- **Better TOML**: Better Cargo.toml editing
- **Error Lens**: Inline error display

VS Code settings (`.vscode/settings.json`):
```json
{
    "rust-analyzer.cargo.features": "all",
    "rust-analyzer.checkOnSave.command": "clippy",
    "rust-analyzer.rustfmt.rangeFormatting.enable": true,
    "[rust]": {
        "editor.formatOnSave": true,
        "editor.defaultFormatter": "rust-lang.rust-analyzer"
    }
}
```

#### Other IDEs

- **IntelliJ IDEA/CLion**: Install Rust plugin
- **Vim/Neovim**: Use rust.vim + ALE or coc.nvim
- **Emacs**: Use rust-mode + flycheck

### Git Configuration

#### Pre-commit Hooks

RunNX includes automated pre-commit hooks. Install them:

```bash
# The hooks are automatically installed when you first commit
# Or manually copy:
cp scripts/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

The pre-commit hook runs:
1. Code formatting (`cargo fmt --check`)
2. Linting (`cargo clippy`)
3. Tests (`cargo test`)
4. Build verification

#### Git Configuration

Set up your Git identity:
```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Enable GPG signing (recommended)
git config commit.gpgsign true
git config user.signingkey YOUR_GPG_KEY
```

## Development Workflow

### Branch Strategy

RunNX uses Git Flow:
- **`main`**: Stable releases only
- **`develop`**: Main development branch (target for PRs)
- **`feature/*`**: Feature development branches
- **`hotfix/*`**: Critical bug fixes

### Creating a Feature

```bash
# Start from develop
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes, commit frequently
git add .
git commit -m "Add initial implementation"

# Push and create PR targeting develop
git push origin feature/your-feature-name
```

### Quality Assurance

#### Manual Quality Check

Run the complete quality check:
```bash
# Use the quality check script
./scripts/quality-check.sh

# Or run individual steps
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo build --release
```

#### Automated Checks

The CI system runs:
- **Formatting**: Ensures consistent code style
- **Linting**: Catches common issues with Clippy
- **Testing**: Full test suite across platforms
- **Coverage**: Code coverage reporting
- **Formal Verification**: Mathematical correctness checks

### Testing Strategy

#### Unit Tests

```bash
# Run all tests
cargo test

# Run specific test module
cargo test operators

# Run with output
cargo test -- --nocapture

# Run with threads (for debugging)
cargo test -- --test-threads=1
```

#### Integration Tests

```bash
# Run integration tests specifically
cargo test --test integration_tests

# Run CLI tests
cargo test --test cli_tests
```

#### Property-Based Tests

```bash
# Run formal verification tests
cargo test formal --features formal-verification

# Run property-based tests
cargo test property
```

#### Coverage Analysis

```bash
# Generate coverage report
cargo tarpaulin --out html

# View coverage
open tarpaulin-report.html
```

### Benchmarking

#### Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench inference

# Save baseline for comparison
cargo bench -- --save-baseline main
```

#### Adding New Benchmarks

Create benchmarks in `benches/` directory:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use runnx::{Model, Tensor};

fn benchmark_inference(c: &mut Criterion) {
    let model = Model::from_file("test_model.onnx").unwrap();
    let input = Tensor::from_array(ndarray::Array2::zeros((1, 784)));
    
    c.bench_function("model inference", |b| {
        b.iter(|| {
            model.run(&[("input", black_box(input.clone()))]).unwrap()
        })
    });
}

criterion_group!(benches, benchmark_inference);
criterion_main!(benches);
```

## Development Tools

### Just Commands

RunNX uses Just for task automation. Common commands:

```bash
# Show all available commands
just --list

# Run tests
just test

# Run formal verification
just formal-test

# Run quality checks
just check

# Build documentation
just docs

# Run benchmarks
just bench

# Clean build artifacts
just clean
```

### Custom Just Recipes

Add to `justfile` for project-specific tasks:

```justfile
# Run development server
dev:
    cargo watch -x "run --bin runnx-runner -- --model examples/simple.onnx --summary"

# Update dependencies
update:
    cargo update
    cargo audit

# Profile performance
profile:
    cargo build --release --features profiling
    perf record ./target/release/runnx-runner --model large_model.onnx
    perf report
```

### Debugging

#### Using rust-lldb/rust-gdb

```bash
# Debug mode build
cargo build

# Debug with lldb
rust-lldb ./target/debug/runnx-runner -- --model model.onnx --input input.json

# Debug with gdb
rust-gdb ./target/debug/runnx-runner
```

#### Debug Configuration

Enable debug logging:
```bash
# Set log level
RUST_LOG=debug cargo run --bin runnx-runner -- --model model.onnx --verbose

# Full trace logging
RUST_LOG=trace cargo test test_name -- --nocapture
```

#### VS Code Debugging

`.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "type": "lldb",
            "request": "launch",
            "name": "Debug runnx-runner",
            "cargo": {
                "args": ["build", "--bin=runnx-runner"],
                "filter": {
                    "name": "runnx-runner",
                    "kind": "bin"
                }
            },
            "args": ["--model", "test.onnx", "--summary"],
            "cwd": "${workspaceFolder}"
        }
    ]
}
```

## Formal Verification Development

### Why3 Setup

For formal verification development:

```bash
# Install Why3 (Ubuntu/Debian)
sudo apt-get install why3

# Install theorem provers
sudo apt-get install alt-ergo z3

# Verify installation
why3 --list-provers
```

### Verification Workflow

```bash
# Run formal verification
just formal-verify

# Test specific operator
./formal/verify_operators.py --operator Add

# Generate property tests
./formal/verify_operators.py --generate-tests
```

### Adding New Formal Specifications

1. Add specifications to `formal/simple_specs.mlw`
2. Implement runtime contracts in `src/operators.rs`
3. Add property-based tests
4. Run verification pipeline

## Performance Optimization

### Profiling Setup

#### CPU Profiling

```bash
# Install profiling tools
cargo install flamegraph

# Generate flame graph
cargo flamegraph --bin runnx-runner -- --model model.onnx --input input.json
```

#### Memory Profiling

```bash
# Use valgrind (Linux)
valgrind --tool=massif ./target/release/runnx-runner --model model.onnx

# Use heaptrack (Linux)
heaptrack ./target/release/runnx-runner --model model.onnx
```

#### Optimization Guidelines

1. **Profile first**: Always measure before optimizing
2. **Focus on hot paths**: Optimize inference loops
3. **Memory efficiency**: Minimize allocations in critical paths
4. **SIMD utilization**: Use ndarray's optimized operations
5. **Async opportunities**: Identify I/O bound operations

## Contribution Guidelines

### Code Style

Follow Rust conventions:
- Use `cargo fmt` for formatting
- Follow naming conventions (snake_case for functions/variables)
- Add documentation for public APIs
- Include examples in documentation

### Documentation Requirements

```rust
/// Brief description of the function
/// 
/// # Arguments
/// 
/// * `param1` - Description of parameter
/// * `param2` - Description of parameter
/// 
/// # Returns
/// 
/// Description of return value
/// 
/// # Examples
/// 
/// ```
/// use runnx::Tensor;
/// let tensor = Tensor::from_array(ndarray::array![1.0, 2.0]);
/// assert_eq!(tensor.shape(), &[2]);
/// ```
/// 
/// # Errors
/// 
/// This function will return an error if...
pub fn example_function(param1: f32, param2: &str) -> Result<Tensor, Error> {
    // Implementation
}
```

### Testing Requirements

All new code must include:
1. **Unit tests**: Test individual functions
2. **Integration tests**: Test end-to-end workflows
3. **Property tests**: For mathematical operations
4. **Documentation tests**: Ensure examples work

### Pull Request Process

1. **Create feature branch** from `develop`
2. **Implement changes** with tests
3. **Run quality checks**: `./scripts/quality-check.sh`
4. **Update documentation** if needed
5. **Submit PR** targeting `develop`
6. **Address review feedback**
7. **Merge after approval**

## Troubleshooting

### Common Issues

**Build failures with protoc:**
```
error: failed to run custom build command for `prost-build`
```
*Solution: Install Protocol Buffers compiler*

**Test failures in CI but not locally:**
- Check for platform-specific code
- Verify test determinism
- Check for missing test files

**Clippy warnings:**
- Fix all warnings before submitting
- Use `#[allow(clippy::lint_name)]` sparingly
- Prefer code fixes over suppressions

**Coverage drops:**
- Add tests for new code paths
- Check for unreachable code
- Verify test completeness

### Getting Help

1. **Read documentation**: Check docs/ directory
2. **Search issues**: Look for similar problems
3. **Ask questions**: Create GitHub discussion
4. **Join community**: Follow project updates

## Advanced Development

### Adding New Operators

1. **Define operator** in `src/operators.rs`
2. **Add formal specification** in `formal/simple_specs.mlw`
3. **Implement tests** including property-based tests
4. **Add benchmarks** if performance-critical
5. **Update documentation** with examples

### Extending Format Support

1. **Add parser** in `src/converter.rs`
2. **Update auto-detection** logic
3. **Add comprehensive tests** for format conversion
4. **Document new format** in guides

### Performance Improvements

1. **Profile current performance**
2. **Identify bottlenecks**
3. **Implement optimizations**
4. **Verify correctness** with tests
5. **Measure improvements** with benchmarks

## Next Steps

Once your development environment is set up:

1. **Read the codebase**: Start with `src/lib.rs`
2. **Run examples**: Try examples in `examples/`
3. **Pick an issue**: Look for "good first issue" labels
4. **Join discussions**: Follow GitHub discussions
5. **Contribute documentation**: Help improve guides

For specific contribution areas:
- **[Testing Guide](testing.md)** - Comprehensive testing strategies
- **[API Documentation](../api/)** - Understanding the codebase
- **[Contributing Guidelines](../../CONTRIBUTING.md)** - Detailed contribution process
