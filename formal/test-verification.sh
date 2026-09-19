#!/bin/bash

# Test script for formal verification of operators

set -e

echo "🧪 Testing Formal Verification for ONNX Operators"
echo "================================================="

echo ""
echo "📋 Step 1: Check formal verification feature compilation"
echo "-------------------------------------------------------"
cargo check --features formal-verification
echo "✅ Formal verification compilation check passed"

echo ""
echo "📋 Step 2: Run formal operator tests"
echo "------------------------------------"
cargo test formal --release
echo "✅ Formal operator tests passed"

echo ""
echo "📋 Step 3: Run property-based formal tests"
echo "------------------------------------------"
cargo test --lib formal_verification_tests --release
echo "✅ Property-based formal tests passed"

echo ""
echo "📋 Step 4: Check Why3 availability"
echo "-----------------------------------"
if command -v why3 &> /dev/null; then
    echo "✅ Why3 found: $(why3 --version)"
    
    echo ""
    echo "📋 Step 5: Detect available provers"
    echo "------------------------------------"
    why3 config detect || echo "⚠️ Could not auto-detect provers"
    echo "Available provers:"
    why3 config list-provers || echo "⚠️ Could not list provers"
    
    echo ""
    echo "📋 Step 6: Verify operator specifications"
    echo "------------------------------------------"
    python3 verify_operators.py
    echo "✅ Formal verification completed successfully"
else
    echo "⚠️ Why3 not found - skipping formal proofs"
    echo "   To install Why3: cd formal && make install-why3"
fi

echo ""
echo "📋 Step 7: Build with formal verification enabled"
echo "--------------------------------------------------"
cargo build --features formal-verification
echo "✅ Build with formal verification succeeded"

echo ""
echo "📋 Step 8: Test example with formal verification"
echo "-------------------------------------------------"
echo "Testing addition operator with formal contracts..."
RUST_LOG=debug cargo run --features formal-verification --example tensor_ops
echo "✅ Example execution completed"

echo ""
echo "🎉 Formal verification pipeline completed successfully!"
echo ""
echo "Summary:"
echo "  ✅ Formal verification compilation check"
echo "  ✅ Formal operator tests"
echo "  ✅ Property-based formal tests"
if command -v why3 &> /dev/null; then
    echo "  ✅ Why3 availability and prover detection"
    echo "  ⚖️ Formal specification verification"
else
    echo "  ⚠️ Why3 not available (install with: make install-why3)"
fi
echo "  ✅ Formal verification enabled build"
echo "  ✅ Formal verification example execution"
echo ""
echo "🎯 All formal verification checks passed!"
