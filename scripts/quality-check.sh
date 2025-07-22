#!/bin/bash
#
# RunNX Development Quality Checks
# Run this script to perform the same checks as the pre-commit hook
#

set -e

echo "🔍 Running RunNX quality checks..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# 1. Format code
source $HOME/.cargo/env  # Ensure Cargo is in PATH
print_status $YELLOW "📝 Formatting code..."
cargo fmt
print_status $GREEN "✅ Code formatted"

# 2. Check formatting
print_status $YELLOW "🔍 Checking formatting..."
if ! cargo fmt --check; then
    print_status $RED "❌ Code formatting check failed after fmt!"
    exit 1
fi
print_status $GREEN "✅ Formatting verified"

# 3. Run clippy with fixes
print_status $YELLOW "🔧 Running clippy with auto-fixes..."
cargo clippy --fix --allow-dirty --allow-staged
print_status $GREEN "✅ Clippy fixes applied"

# 4. Check clippy without warnings
print_status $YELLOW "🔍 Running clippy checks..."
if ! cargo clippy --all-targets --all-features -- -D warnings; then
    print_status $RED "❌ Clippy found remaining issues!"
    exit 1
fi
print_status $GREEN "✅ No clippy issues"

# 5. Run tests
print_status $YELLOW "🧪 Running tests..."
if ! cargo test; then
    print_status $RED "❌ Tests failed!"
    exit 1
fi
print_status $GREEN "✅ All tests passed"

# 6. Build project
print_status $YELLOW "🔨 Building project..."
if ! cargo build --all-targets; then
    print_status $RED "❌ Build failed!"
    exit 1
fi
print_status $GREEN "✅ Build successful"

print_status $GREEN "🎉 All quality checks passed! Your code is ready for commit."
