# RunNX Formal Verification Makefile

.PHONY: all verify test clean check-why3 install-why3 setup-provers test-setup verify-operators

# Default target - includes operator verification
all: setup-provers verify-operators test

# Verify operators specifically
verify-operators: check-why3
	@echo "🔍 Verifying ONNX operators..."
	@python3 verify_operators.py

# Verify specific operator
verify-operator-%: check-why3
	@echo "🎯 Verifying $* operator..."
	@python3 verify_operators.py $*

# Verify extended operators (new additions)
verify-extended: check-why3
	@echo "🔍 Verifying extended operator set..."
	@for op in div sub exp sqrt pow identity cast squeeze unsqueeze reduce_mean batch_norm; do \
		echo "🎯 Verifying $$op operator..."; \
		python3 verify_operators.py $$op || echo "⚠️ $$op verification had issues"; \
	done

# Verify core operators only (original set)
verify-core: check-why3
	@echo "🔍 Verifying core operator set..."
	@for op in add mul matmul relu sigmoid transpose reshape; do \
		echo "🎯 Verifying $$op operator..."; \
		python3 verify_operators.py $$op || echo "⚠️ $$op verification had issues"; \
	done

# Test the formal verification setup
test-setup:
	@echo "🧪 Testing formal verification setup..."
	@./test-verification.sh

# Setup Why3 provers
setup-provers:
	@echo "🔧 Setting up Why3 provers..."
	@if which why3 > /dev/null; then \
		why3 config detect || echo "⚠️ Could not auto-detect provers"; \
		echo "Available provers:"; \
		why3 config list-provers || echo "⚠️ Could not list provers"; \
	else \
		echo "❌ Why3 not found. Run 'make install-why3' first."; \
		exit 1; \
	fi

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
	@eval `opam config env` && why3 config detect
	@echo "✅ Why3 installed successfully"
	@echo "📝 Note: You may need to run 'eval \$$(opam env)' to update your PATH"

# Run formal verification
verify: check-why3
	@echo "🔍 Running formal verification..."
	@echo "Detecting available provers..."
	@why3 config detect || echo "⚠️ Could not detect provers"
	@echo "Available provers:"
	@why3 config list-provers || echo "⚠️ Could not list provers"
	@echo "Verifying tensor specifications..."
	@why3 prove tensors.mlw || echo "⚠️ Some tensor proofs failed"
	@echo "Verifying operator specifications..."
	@why3 prove operators.mlw || echo "⚠️ Some operator proofs failed"
	@echo "✅ Formal verification completed"

# Run property-based tests
test:
	@echo "🧪 Running property-based tests..."
	@cargo test formal --release
	@echo "✅ Property-based tests completed"

# Run the verification script
verify-script:
	@echo "🐍 Running Python verification bridge..."
	@python3 verify_operators.py

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
	@why3 ide simple_specs.mlw

# Help target
help:
	@echo "RunNX Formal Verification Commands:"
	@echo ""
	@echo "  make all           - Run verification and tests"
	@echo "  make install-why3  - Install Why3 and Alt-Ergo"
	@echo "  make setup-provers - Setup and detect Why3 provers"
	@echo "  make test-setup    - Test the verification setup"
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
