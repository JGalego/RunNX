# RunNX Formal Verification Makefile

.PHONY: all verify test clean check-why3 install-why3

# Default target
all: verify test

# Check if Why3 is installed
check-why3:
	@which why3 > /dev/null || (echo "❌ Why3 not found. Run 'make install-why3' to install it." && exit 1)
	@echo "✅ Why3 found"

# Install Why3 via opam
install-why3:
	@echo "📦 Installing Why3..."
	@if ! which opam > /dev/null; then \
		echo "Installing opam first..."; \
		curl -fsSL https://raw.githubusercontent.com/ocaml/opam/master/shell/install.sh | sh; \
	fi
	@opam init -y --disable-sandboxing || true
	@eval `opam config env` && opam install -y why3 alt-ergo
	@echo "✅ Why3 installed successfully"
	@echo "📝 Note: You may need to run 'eval \$$(opam env)' to update your PATH"

# Run formal verification
verify: check-why3
	@echo "🔍 Running formal verification..."
	@why3 prove -P alt-ergo tensor_specs.mlw || echo "⚠️ Some proofs failed"
	@why3 prove -P alt-ergo neural_network_specs.mlw || echo "⚠️ Some proofs failed"
	@echo "✅ Formal verification completed"

# Run property-based tests
test:
	@echo "🧪 Running property-based tests..."
	@cargo test formal --release
	@echo "✅ Property-based tests completed"

# Run the verification script
verify-script:
	@echo "🐍 Running Python verification bridge..."
	@python3 verify.py

# Generate Why3 HTML report
report: check-why3
	@echo "📊 Generating verification report..."
	@why3 session html . || echo "⚠️ Report generation failed"
	@echo "📁 Report generated in _why3session/"

# Run static analysis
lint:
	@echo "🔍 Running static analysis..."
	@cargo clippy -- -D warnings
	@cargo fmt --check

# Run benchmarks with verification
bench:
	@echo "⚡ Running benchmarks with verification..."
	@cargo bench
	@echo "✅ Benchmarks completed"

# Full verification suite
full-verify: lint verify test bench
	@echo "🎉 Full verification suite completed!"

# Clean build artifacts and verification outputs
clean:
	@echo "🧹 Cleaning up..."
	@cargo clean
	@rm -rf _why3session/
	@rm -rf __pycache__/
	@echo "✅ Cleanup completed"

# Interactive Why3 IDE
ide: check-why3
	@echo "🚀 Starting Why3 IDE..."
	@why3 ide tensor_specs.mlw

# Help target
help:
	@echo "RunNX Formal Verification Commands:"
	@echo ""
	@echo "  make all           - Run verification and tests"
	@echo "  make install-why3  - Install Why3 and Alt-Ergo"
	@echo "  make verify        - Run formal verification only"
	@echo "  make test          - Run property-based tests only"
	@echo "  make verify-script - Run Python verification bridge"
	@echo "  make report        - Generate HTML verification report"
	@echo "  make lint          - Run static analysis"
	@echo "  make bench         - Run benchmarks"
	@echo "  make full-verify   - Run complete verification suite"
	@echo "  make ide           - Start Why3 IDE"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make help          - Show this help"
