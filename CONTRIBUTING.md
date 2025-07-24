# Contributing

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

## Pull Requests

Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Development Setup

1. Install Rust: https://rustup.rs/
2. Install Protocol Buffers compiler:
   - Ubuntu/Debian: `sudo apt-get install protobuf-compiler`
   - macOS: `brew install protobuf`  
   - Windows: `choco install protoc`
3. Clone the repository
4. Run `cargo test` to run the test suite
4. Run `cargo bench` to run benchmarks
5. Run `cargo doc --open` to build and view documentation

## Code Style

We use standard Rust formatting. Please run `cargo fmt` before submitting.

## Testing

- Write unit tests for new functions
- Write integration tests for new features  
- Add benchmarks for performance-critical code
- Test both JSON and ONNX binary format compatibility for model-related changes
- Ensure all tests pass with `cargo test`
- Run format compatibility tests: `cargo run --example onnx_demo`

## Documentation

- Document all public APIs with rustdoc comments
- Include examples in documentation
- Update README.md if needed
- Write clear commit messages

## Any contributions you make will be under the MIT/Apache-2.0 Software License

In short, when you submit code changes, your submissions are understood to be under the same [MIT/Apache-2.0 License](LICENSE) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using GitHub's [issues](https://github.com/JGalego/runnx/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/JGalego/runnx/issues/new); it's that easy!

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Use a Consistent Coding Style

* Use `cargo fmt` for formatting
* Use `cargo clippy` for linting
* Follow Rust naming conventions
* Add comprehensive documentation for public APIs

## Development Tools

RunNX provides convenient development tools to streamline your workflow:

### Using Justfile (Recommended)
```bash
# Install just (one time)
cargo install just

# Quick development cycle
just dev           # Format, lint, and test
just quality       # Run comprehensive quality checks  
just examples      # Run all examples
just ci            # Simulate CI locally
```

### Using dev.sh Script (Alternative)
```bash
# No installation required
./dev.sh dev       # Quick development cycle
./dev.sh quality   # Run comprehensive quality checks
./dev.sh examples  # Run all examples  
./dev.sh ci        # Simulate CI locally
```

### Manual Commands
```bash
# Individual quality checks
cargo fmt           # Format code
cargo clippy        # Run linting
cargo test          # Run all tests
./scripts/quality-check.sh  # Full quality check script
```

## License

By contributing, you agree that your contributions will be licensed under its MIT/Apache-2.0 License.

## References

This document was adapted from the open-source contribution guidelines for [Facebook's Draft](https://github.com/facebook/draft-js/blob/a9316a723f9e918afde44dea68b5f9f39b7d9b00/CONTRIBUTING.md)
