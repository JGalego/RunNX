#!/bin/bash
# RunNX Code Coverage Script
# ==========================
# This script runs cargo llvm-cov with exclusions defined in .llvm-cov-exclude
#
# Usage: ./scripts/coverage.sh [additional llvm-cov arguments]
# Examples:
#   ./scripts/coverage.sh --summary-only
#   ./scripts/coverage.sh --html --output-dir target/coverage
#   ./scripts/coverage.sh --summary-only --quiet

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
EXCLUDE_FILE=".llvm-cov-exclude"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Help function
show_help() {
    echo -e "${BLUE}RunNX Code Coverage Script${NC}"
    echo "========================="
    
    # Use the effective exclude file (including override)
    local effective_exclude_file="${COVERAGE_EXCLUDE_FILE:-$EXCLUDE_FILE}"
    echo "This script runs cargo llvm-cov with exclusions defined in $effective_exclude_file"
    echo ""
    echo -e "${YELLOW}Usage:${NC} $0 [additional llvm-cov arguments]"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  $0 --summary-only                    # Show coverage summary"
    echo "  $0 --html --output-dir target/cov    # Generate HTML report"
    echo "  $0 --summary-only --quiet            # Quiet summary"
    echo "  $0 --json --output-path coverage.json # JSON output"
    echo ""
    echo -e "${YELLOW}Environment Variables:${NC}"
    echo "  COVERAGE_EXCLUDE_FILE  # Override exclusion file (default: $EXCLUDE_FILE)"
    echo "  COVERAGE_VERBOSE       # Show verbose output (set to 1)"
    echo ""
    if [[ -f "$effective_exclude_file" ]]; then
        echo -e "${YELLOW}Current exclusions from $effective_exclude_file:${NC}"
        # Only show non-comment, non-empty lines
        while IFS= read -r line; do
            # Skip empty lines and comments
            if [[ -n "$line" && ! "$line" =~ ^[[:space:]]*# ]]; then
                # Remove leading/trailing whitespace
                line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
                if [[ -n "$line" ]]; then
                    echo "  - $line"
                fi
            fi
        done < "$effective_exclude_file"
    else
        echo -e "${RED}Warning: $effective_exclude_file not found${NC}"
    fi
}

# Check for help
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    show_help
    exit 0
fi

# Use environment variable override or default
EXCLUDE_FILE="${COVERAGE_EXCLUDE_FILE:-$EXCLUDE_FILE}"

# Check if exclude file exists
if [[ ! -f "$EXCLUDE_FILE" ]]; then
    echo -e "${RED}Error: Exclusion file '$EXCLUDE_FILE' not found${NC}" >&2
    echo "Create the file or set COVERAGE_EXCLUDE_FILE environment variable" >&2
    exit 1
fi

# Parse exclusion file and build regex pattern
build_exclusion_pattern() {
    local patterns=()
    while IFS= read -r line; do
        # Skip empty lines and comments
        if [[ -n "$line" && ! "$line" =~ ^[[:space:]]*# ]]; then
            # Remove leading/trailing whitespace
            line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            if [[ -n "$line" ]]; then
                patterns+=("$line")
            fi
        fi
    done < "$EXCLUDE_FILE"
    
    if [[ ${#patterns[@]} -eq 0 ]]; then
        echo ""
        return
    fi
    
    # Join patterns with |
    local regex_pattern=""
    for pattern in "${patterns[@]}"; do
        if [[ -n "$regex_pattern" ]]; then
            regex_pattern="$regex_pattern|$pattern"
        else
            regex_pattern="$pattern"
        fi
    done
    
    echo "$regex_pattern"
}

# Build exclusion pattern
EXCLUSION_PATTERN=$(build_exclusion_pattern)

# Show verbose information
if [[ "${COVERAGE_VERBOSE:-0}" == "1" ]]; then
    echo -e "${BLUE}RunNX Coverage Configuration:${NC}"
    echo "  Project root: $PROJECT_ROOT"
    echo "  Exclusion file: $EXCLUDE_FILE"
    if [[ -n "$EXCLUSION_PATTERN" ]]; then
        echo "  Exclusion pattern: $EXCLUSION_PATTERN"
    else
        echo "  No exclusions defined"
    fi
    echo ""
fi

# Build cargo llvm-cov command with deduplication
build_cargo_args() {
    local args=("llvm-cov" "--all-features")
    local user_args=("$@")
    
    # Default arguments to add if not provided by user
    local default_args=("--workspace")
    
    # Add exclusion pattern if we have one
    if [[ -n "$EXCLUSION_PATTERN" ]]; then
        default_args+=("--ignore-filename-regex=$EXCLUSION_PATTERN")
    fi
    
    # Check which default args are already in user args
    for default_arg in "${default_args[@]}"; do
        local found=false
        
        # Extract the argument name (before = if present)
        local default_arg_name="${default_arg%%=*}"
        
        for user_arg in "${user_args[@]}"; do
            # Extract the argument name from user arg
            local user_arg_name="${user_arg%%=*}"
            
            # Check if this argument name is already provided by user
            if [[ "$user_arg_name" == "$default_arg_name" ]]; then
                found=true
                break
            fi
        done
        
        # Add default arg only if not found in user args
        if [[ "$found" == "false" ]]; then
            args+=("$default_arg")
        fi
    done
    
    # Add all user arguments
    args+=("${user_args[@]}")
    
    # Return the args array
    printf '%s\n' "${args[@]}"
}

# Build the command arguments
readarray -t CARGO_ARGS < <(build_cargo_args "$@")

# Run coverage
if [[ "${COVERAGE_VERBOSE:-0}" == "1" ]]; then
    echo -e "${GREEN}Running:${NC} cargo ${CARGO_ARGS[*]}"
    echo ""
fi

exec cargo "${CARGO_ARGS[@]}"
