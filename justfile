# RunNX Development Justfile
# Run `just --list` to see all available commands

# Default recipe - runs basic development checks
default: check test

# Show all available commands
help:
    @just --list

# === Build Commands ===

# Build the project
build:
    cargo build

# Build the project in release mode
build-release:
    cargo build --release

# Build with all features enabled
build-all:
    cargo build --all-features

# Clean build artifacts
clean:
    cargo clean

# === Testing Commands ===

# Run all tests
test:
    cargo test --all-features

# Run only library tests
test-lib:
    cargo test --lib

# Run integration tests
test-integration:
    cargo test --test integration_tests

# Run CLI tests
test-cli:
    cargo test --test cli_tests

# Run property-based tests (formal verification)
test-formal:
    cargo test formal --lib

# Run tests with coverage report
test-coverage:
    cargo llvm-cov --all-features --ignore-filename-regex "bin/" --workspace --lcov --output-path lcov.info --open

# === Code Quality Commands ===

# Run quality checks (format, lint, test)
check: format-check lint test

# Format code
format:
    cargo fmt

# Check if code is formatted
format-check:
    cargo fmt --all -- --check

# Run clippy linter
lint:
    cargo clippy --all-targets --all-features -- -D warnings

# Run clippy with auto-fixes
lint-fix:
    cargo clippy --all-targets --all-features --fix --allow-dirty

# Run the quality check script
quality:
    ./scripts/quality-check.sh

# === Documentation Commands ===

# Build documentation
docs:
    cargo doc --all-features --no-deps --document-private-items

# Build and open documentation
docs-open:
    cargo doc --all-features --no-deps --document-private-items --open

# === Example Commands ===

# Run the simple model example
example-simple:
    cargo run --example simple_model

# Run the tensor operations example
example-tensor:
    cargo run --example tensor_ops

# Run the ONNX demo (format compatibility)
example-onnx:
    cargo run --example onnx_demo

# Run the format conversion example
example-convert:
    cargo run --example format_conversion

# Run the formal verification example
example-formal:
    cargo run --example formal_verification

# Run the ONNX support test
example-test-onnx:
    cargo run --example test_onnx_support

# Run all examples
examples: example-simple example-tensor example-onnx example-convert example-formal example-test-onnx

# === CLI Commands ===

# Run the CLI runner with help
cli-help:
    cargo run --bin runnx-runner -- --help

# Run CLI with sample model (requires model file)
cli-run model input:
    cargo run --bin runnx-runner -- --model {{model}} --input {{input}}

# Run CLI with async features
cli-async model input:
    cargo run --features async --bin runnx-runner -- --model {{model}} --input {{input}}

# === Benchmarking Commands ===

# Run benchmarks
bench:
    cargo bench

# Run benchmarks with HTML report
bench-report:
    cargo bench -- --output-format html

# Run specific benchmark
bench-inference:
    cargo bench --bench inference_benchmark

# === Formal Verification Commands ===

# Test formal verification setup
formal-test:
    cd formal && ./test-verification.sh

# Run formal verification (requires Why3)
formal-verify:
    cd formal && make verify

# Setup formal verification environment
formal-setup:
    cd formal && make setup-provers

# Install Why3 for formal verification  
formal-install:
    cd formal && make install-why3

# Run all formal verification tasks
formal: formal-test formal-verify

# === Development Commands ===

# Watch for changes and run tests
watch:
    cargo watch -x test

# Watch for changes and run checks
watch-check:
    cargo watch -x check -x test

# Install development dependencies
dev-setup:
    rustup component add rustfmt clippy llvm-tools-preview
    cargo install cargo-llvm-cov cargo-watch

# === Release Commands ===

# Check if ready for release
release-check: clean build-release test docs lint

# Dry-run of cargo publish
release-dry:
    cargo publish --dry-run

# Publish to crates.io (requires authentication)
release-publish:
    cargo publish

# === Platform-specific Commands ===

# Install protobuf compiler (Ubuntu/Debian)
install-protoc-ubuntu:
    sudo apt-get update && sudo apt-get install -y protobuf-compiler

# Install protobuf compiler (macOS)
install-protoc-macos:
    brew install protobuf

# === File Generation Commands ===

# Generate a new example template
new-example name:
    @echo "Creating new example: {{name}}"
    @echo 'use runnx::*;\n\nfn main() -> runnx::Result<()> {\n    println!("{{name}} example");\n    Ok(())\n}' > examples/{{name}}.rs
    @echo "Created examples/{{name}}.rs"

# Create a sample ONNX model for testing
create-sample-model:
    cargo run --example onnx_demo

# === Utility Commands ===

# Check for security vulnerabilities
audit:
    cargo audit

# Update dependencies
update:
    cargo update

# Show project info
info:
    @echo "RunNX - A minimal, verifiable ONNX runtime in Rust"
    @echo "Version: $(cargo read-manifest | jq -r .version)"
    @echo "Repository: $(cargo read-manifest | jq -r .repository)"
    @echo ""
    @echo "Available examples:"
    @ls examples/*.rs | sed 's/examples\//  - /' | sed 's/\.rs//'
    @echo ""
    @echo "Build targets:"
    @echo "  - Library: runnx"
    @echo "  - Binary: runnx-runner"

# Show dependencies
deps:
    cargo tree

# Show outdated dependencies
deps-outdated:
    cargo outdated

# === CI Simulation Commands ===

# Simulate CI checks locally
ci: clean format-check lint test docs

# Simulate the full CI pipeline
ci-full: ci bench audit

# === Container Commands ===

# Build Docker image (if Dockerfile exists)
docker-build:
    docker build -t runnx .

# Run in Docker container
docker-run:
    docker run --rm -it runnx

# === Shortcuts ===

# Quick development cycle
dev: format lint test

# Full quality check
qa: quality

# Quick test
t: test

# Quick build
b: build

# Quick format
f: format

# Quick lint
l: lint
