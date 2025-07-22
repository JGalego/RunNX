#!/bin/bash
# Test script for formal verification setup

echo "🧪 Testing formal verification setup..."

# Test 1: Check if Why3 is available
echo "Test 1: Checking Why3 availability..."
if command -v why3 >/dev/null 2>&1; then
    echo "✅ Why3 found"
    
    # Test 2: Detect provers
    echo "Test 2: Detecting provers..."
    why3 config detect && echo "✅ Prover detection successful" || echo "⚠️ Prover detection failed"
    
    # Test 3: List available provers
    echo "Test 3: Listing available provers..."
    why3 config list-provers && echo "✅ Prover listing successful" || echo "⚠️ Prover listing failed"
    
    # Test 4: Try to verify without specifying prover
    echo "Test 4: Testing verification without specifying prover..."
    why3 prove tensor_specs.mlw && echo "✅ Verification without prover spec successful" || echo "⚠️ Verification failed"
    
else
    echo "❌ Why3 not found. This is expected in environments without Why3."
fi

# Test 5: Run Python verification script
echo "Test 5: Running Python verification script..."
python3 verify.py && echo "✅ Python verification successful" || echo "⚠️ Python verification had issues"

echo "🏁 Test completed!"
