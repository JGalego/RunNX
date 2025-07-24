#!/bin/bash
# Helper script to run Why3 commands with proper environment setup

set -e

# Function to check if Why3 is available
check_why3() {
    if command -v why3 >/dev/null 2>&1; then
        return 0
    elif command -v opam >/dev/null 2>&1; then
        eval "$(opam env)"
        if command -v why3 >/dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# Function to run Why3 command with proper environment
run_why3() {
    if command -v why3 >/dev/null 2>&1; then
        "$@"
    elif command -v opam >/dev/null 2>&1; then
        eval "$(opam env)"
        "$@"
    else
        echo "❌ Why3 not found. Please install it first."
        exit 1
    fi
}

# Function to setup Why3 provers
setup_provers() {
    echo "🔍 Detecting Why3 provers..."
    if run_why3 why3 config detect; then
        echo "✅ Provers detected successfully"
        echo "📋 Available provers:"
        run_why3 why3 config list-provers
    else
        echo "⚠️ Warning: Could not detect provers automatically"
    fi
}

# Main command execution
case "$1" in
    "check")
        if check_why3; then
            echo "✅ Why3 found"
            setup_provers
        else
            echo "❌ Why3 not found. Run 'make install-why3' to install it."
            exit 1
        fi
        ;;
    "setup")
        setup_provers
        ;;
    "prove")
        shift
        run_why3 why3 prove "$@"
        ;;
    "session")
        shift
        run_why3 why3 session "$@"
        ;;
    "ide")
        shift
        run_why3 why3 ide "$@"
        ;;
    *)
        echo "Usage: $0 {check|setup|prove|session|ide} [args...]"
        exit 1
        ;;
esac
