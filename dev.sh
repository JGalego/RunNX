#!/bin/bash
# RunNX Development Scripts
# Alternative to Justfile for users without 'just' installed

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_usage() {
    echo -e "${BLUE}RunNX Development Scripts${NC}"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Build Commands:"
    echo "  build         Build the project"
    echo "  build-release Build in release mode"
    echo "  clean         Clean build artifacts"
    echo ""
    echo "Testing Commands:"
    echo "  test          Run all tests"
    echo "  test-lib      Run library tests only"
    echo "  test-formal   Run formal verification tests"
    echo ""
    echo "Code Quality:"
    echo "  check         Run format check, lint, and test"
    echo "  format        Format code"
    echo "  lint          Run clippy linter"
    echo "  quality       Run quality check script"
    echo ""
    echo "Examples:"
    echo "  example-onnx      Run ONNX demo"
    echo "  example-convert   Run format conversion example"
    echo "  example-simple    Run simple model example"
    echo "  examples          Run all examples"
    echo ""
    echo "Documentation:"
    echo "  docs          Build documentation"
    echo "  docs-open     Build and open documentation"
    echo ""
    echo "Other:"
    echo "  bench         Run benchmarks"
    echo "  formal-test   Test formal verification setup"
    echo "  ci            Simulate CI checks"
    echo "  info          Show project information"
    echo "  dev           Quick development cycle (format + lint + test)"
}

run_command() {
    local cmd="$1"
    echo -e "${YELLOW}Running: ${cmd}${NC}"
    eval "$cmd"
}

case "$1" in
    # Build commands
    "build")
        run_command "cargo build"
        ;;
    "build-release")
        run_command "cargo build --release"
        ;;
    "clean")
        run_command "cargo clean"
        ;;
    
    # Testing commands
    "test")
        run_command "cargo test --all-features"
        ;;
    "test-lib")
        run_command "cargo test --lib"
        ;;
    "test-formal")
        run_command "cargo test formal --lib"
        ;;
    
    # Code quality
    "check")
        run_command "cargo fmt --all -- --check"
        run_command "cargo clippy --all-targets --all-features -- -D warnings"
        run_command "cargo test --all-features"
        ;;
    "format")
        run_command "cargo fmt"
        ;;
    "lint")
        run_command "cargo clippy --all-targets --all-features -- -D warnings"
        ;;
    "quality")
        run_command "./scripts/quality-check.sh"
        ;;
    
    # Examples
    "example-onnx")
        run_command "cargo run --example onnx_demo"
        ;;
    "example-convert")
        run_command "cargo run --example format_conversion"
        ;;
    "example-simple")
        run_command "cargo run --example simple_model"
        ;;
    "example-tensor")
        run_command "cargo run --example tensor_ops"
        ;;
    "example-formal")
        run_command "cargo run --example formal_verification"
        ;;
    "examples")
        echo -e "${BLUE}Running all examples...${NC}"
        run_command "cargo run --example simple_model"
        run_command "cargo run --example tensor_ops"
        run_command "cargo run --example onnx_demo"
        run_command "cargo run --example format_conversion"
        run_command "cargo run --example formal_verification"
        ;;
    
    # Documentation
    "docs")
        run_command "cargo doc --all-features --no-deps --document-private-items"
        ;;
    "docs-open")
        run_command "cargo doc --all-features --no-deps --document-private-items --open"
        ;;
    
    # Other commands
    "bench")
        run_command "cargo bench"
        ;;
    "formal-test")
        run_command "cd formal && ./test-verification.sh"
        ;;
    "ci")
        echo -e "${BLUE}Simulating CI checks...${NC}"
        run_command "cargo fmt --all -- --check"
        run_command "cargo clippy --all-targets --all-features -- -D warnings"
        run_command "cargo test --all-features"
        run_command "cargo doc --all-features --no-deps --document-private-items"
        ;;
    "info")
        echo -e "${BLUE}RunNX - A minimal, verifiable ONNX runtime in Rust${NC}"
        echo "Version: $(cargo read-manifest 2>/dev/null | grep version | head -1 | cut -d'"' -f4 || echo 'unknown')"
        echo ""
        echo "Available examples:"
        ls examples/*.rs 2>/dev/null | sed 's/examples\//  - /' | sed 's/\.rs//' || echo "  No examples found"
        echo ""
        echo "Build targets:"
        echo "  - Library: runnx"
        echo "  - Binary: runnx-runner"
        ;;
    "dev")
        echo -e "${BLUE}Quick development cycle...${NC}"
        run_command "cargo fmt"
        run_command "cargo clippy --all-targets --all-features -- -D warnings"
        run_command "cargo test --all-features"
        echo -e "${GREEN}✅ Development cycle completed!${NC}"
        ;;
    
    # Help and default
    "help" | "--help" | "-h" | "")
        print_usage
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo ""
        print_usage
        exit 1
        ;;
esac
